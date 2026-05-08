"""Data-collection policy for the Gazebo training pivot (v26).

Drops in as a Policy implementation but does not attempt insertion. For each
trial it captures wrist-camera frames + the ground-truth port world xyz and
dumps an npz into /output/episode_{scene_id}.npz.

Ground truth is supplied from outside (the driver knows what it told aic_engine
to spawn). The driver writes a JSON list of per-trial metadata (one entry
per yaml trial in order) and the policy steps through it on each insert_cable.

Env vars:
  AIC_TRIAL_META  — path to trial_meta.json (list of {trial_index, scene_id,
                    gt_xyz, board_xyyaw, target_module_name, port_name, ...})
  AIC_OUTPUT_DIR  — defaults to /output

The arm parks at SAFE_POSE then takes 5 viewpoint snapshots: at SAFE_POSE plus
4 tiny wrist-only perturbations to vary camera angle. Total ~25s per trial.
"""

import json
import os
import socket
import time
import urllib.request
from pathlib import Path

import numpy as np
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


_DIAG_URL = os.environ.get("AIC_DIAG_URL", "")
_HOST = socket.gethostname()


def _diag(event: str, **extra) -> None:
    if not _DIAG_URL:
        return
    payload = {"event": event, "host": _HOST, "ts": time.time(), **extra}
    try:
        req = urllib.request.Request(
            _DIAG_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as e:
        print(f"[diag] {event}: {e!r}", flush=True)


_diag("module_imported", policy="GazeboCollector")


CONTROL_HZ = 20
SAFE_POSE = np.array([-0.16, -1.35, -1.66, -1.69, 1.57, 1.41])
TRANSITION_STIFFNESS = [200.0, 200.0, 200.0, 100.0, 100.0, 100.0]
TRANSITION_DAMPING = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]

# Wrist-only perturbations to vary camera viewpoint without moving the base
# of the arm into different joint configurations. Keeps the cluster's
# rate-limited motion cheap (small deltas).
VIEWPOINT_PERTURBATIONS = np.array([
    [0.0, 0.0, 0.0,  0.00,  0.00,  0.00],   # nominal SAFE_POSE
    [0.0, 0.0, 0.0,  0.05,  0.00,  0.00],   # wrist_1 + 0.05
    [0.0, 0.0, 0.0, -0.05,  0.00,  0.00],   # wrist_1 - 0.05
    [0.0, 0.0, 0.0,  0.00,  0.05,  0.00],   # wrist_2 + 0.05
    [0.0, 0.0, 0.0,  0.00, -0.05,  0.00],   # wrist_2 - 0.05
])

OUTPUT_DIR = Path(os.environ.get("AIC_OUTPUT_DIR", "/output"))
SETTLE_TIME = 2.0      # s between viewpoint move and capture
PARK_HOLD = 4.0        # s to settle at SAFE_POSE
SAFE_POSE_SKIP_TOL = 0.10


def _densify(traj: np.ndarray, max_per_joint_step: float) -> np.ndarray:
    out = [traj[0]]
    for i in range(1, len(traj)):
        prev = traj[i - 1]
        delta = traj[i] - prev
        n = max(1, int(np.ceil(np.abs(delta).max() / max_per_joint_step)))
        for k in range(1, n + 1):
            out.append(prev + (k / n) * delta)
    return np.array(out)


IMAGE_DOWNSAMPLE = int(os.environ.get("AIC_IMG_STRIDE", "4"))


def _ros_image_to_numpy(img_msg) -> np.ndarray | None:
    if img_msg.height == 0 or img_msg.width == 0:
        return None
    try:
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, -1
        )
    except Exception:
        return None
    img = img[..., :3] if img.shape[2] >= 3 else img
    if IMAGE_DOWNSAMPLE > 1:
        img = img[::IMAGE_DOWNSAMPLE, ::IMAGE_DOWNSAMPLE].copy()
    return img


class GazeboCollector(Policy):
    """Capture wrist-camera frames + ground-truth port pose for localizer training."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = os.environ.get("AIC_TRIAL_META", "/trial_meta.json")
        self._trial_meta: list[dict] = []
        try:
            with open(meta_path) as f:
                self._trial_meta = json.load(f)
        except Exception as e:
            self.get_logger().error(
                f"Failed to read trial meta from {meta_path}: {e!r}"
            )
        self._trial_idx = 0
        self.get_logger().info(
            f"GazeboCollector ready: {len(self._trial_meta)} trials staged, "
            f"out={OUTPUT_DIR}"
        )
        _diag(
            "collector_init",
            n_trials=len(self._trial_meta),
            scene_ids=[m.get("scene_id") for m in self._trial_meta][:5],
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if self._trial_idx < len(self._trial_meta):
            meta = self._trial_meta[self._trial_idx]
        else:
            meta = {"scene_id": f"overflow_{self._trial_idx}_{int(time.time())}",
                    "gt_xyz": None, "board_xyyaw": None}
        scene_id = meta.get("scene_id", f"unknown_{self._trial_idx}")
        gt_port = (
            np.array(meta["gt_xyz"], dtype=np.float64)
            if meta.get("gt_xyz") is not None else None
        )
        gt_board = (
            np.array(meta["board_xyyaw"], dtype=np.float64)
            if meta.get("board_xyyaw") is not None else None
        )
        # Sanity: log mismatch between meta-supplied module/port and the actual
        # task params if both available — useful for debugging trial-order skew.
        meta_mod = meta.get("target_module_name")
        actual_mod = getattr(task, "target_module_name", None)
        if meta_mod and actual_mod and meta_mod != actual_mod:
            self.get_logger().warn(
                f"trial_idx={self._trial_idx}: meta target_module_name={meta_mod!r} "
                f"but task says {actual_mod!r} — order mismatch?"
            )
        self._trial_idx += 1

        send_feedback(f"collecting {scene_id}")
        _diag(
            "collector_insert_enter",
            trial_idx=self._trial_idx - 1,
            scene_id=scene_id,
            target_module_name=actual_mod,
            port_name=getattr(task, "port_name", None),
            cable_type=task.cable_type,
            port_type=task.port_type,
        )

        trans_msg = JointMotionUpdate(
            target_stiffness=TRANSITION_STIFFNESS,
            target_damping=TRANSITION_DAMPING,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )

        def _swing(start: np.ndarray, end: np.ndarray) -> None:
            path = _densify(np.stack([start, end]), max_per_joint_step=0.0015)
            for q_step in path:
                trans_msg.target_state.positions = q_step.tolist()
                move_robot(joint_motion_update=trans_msg)
                self.sleep_for(1.0 / CONTROL_HZ)

        def _hold(target: np.ndarray, hold_s: float) -> None:
            n = int(hold_s * CONTROL_HZ)
            for _ in range(n):
                trans_msg.target_state.positions = target.tolist()
                move_robot(joint_motion_update=trans_msg)
                self.sleep_for(1.0 / CONTROL_HZ)

        # Read start pose
        start_pose = None
        for _ in range(20):
            obs = get_observation()
            if obs is not None and len(obs.joint_states.position) >= 6:
                start_pose = np.array(obs.joint_states.position[:6])
                break
            self.sleep_for(1.0 / CONTROL_HZ)
        if start_pose is None:
            self.get_logger().warn("no initial obs; using SAFE_POSE")
            start_pose = SAFE_POSE.copy()

        # Park at SAFE_POSE if not already there
        if np.abs(start_pose - SAFE_POSE).max() > SAFE_POSE_SKIP_TOL:
            _swing(start_pose, SAFE_POSE)
            _hold(SAFE_POSE, PARK_HOLD)

        captures = []
        for vp_idx, dq in enumerate(VIEWPOINT_PERTURBATIONS):
            target = SAFE_POSE + dq
            if vp_idx > 0:
                _swing(SAFE_POSE + VIEWPOINT_PERTURBATIONS[vp_idx - 1], target)
            _hold(target, SETTLE_TIME)
            obs = get_observation()
            if obs is None:
                self.get_logger().warn(f"viewpoint {vp_idx}: no observation")
                continue
            imgs = {}
            for name in ("left_image", "center_image", "right_image"):
                msg = getattr(obs, name, None)
                if msg is None:
                    self.get_logger().warn(f"viewpoint {vp_idx}: missing {name}")
                    imgs = None
                    break
                arr = _ros_image_to_numpy(msg)
                if arr is None:
                    self.get_logger().warn(f"viewpoint {vp_idx}: bad image {name}")
                    imgs = None
                    break
                imgs[name] = arr
            if imgs is None:
                continue
            joint_pos = (
                np.array(obs.joint_states.position[:6])
                if len(obs.joint_states.position) >= 6
                else target.copy()
            )
            captures.append(
                {
                    "viewpoint": vp_idx,
                    "joint_pos": joint_pos,
                    "left": imgs["left_image"],
                    "center": imgs["center_image"],
                    "right": imgs["right_image"],
                }
            )

        # Return arm to SAFE_POSE before exit (idempotent for next trial in
        # the same session, in case eval reuses this policy instance).
        _swing(SAFE_POSE + VIEWPOINT_PERTURBATIONS[-1], SAFE_POSE)

        if not captures:
            self.get_logger().error("no successful captures; not writing output")
            _diag("collector_no_captures", scene_id=scene_id)
            return True

        out_path = OUTPUT_DIR / f"episode_{scene_id}.npz"
        save = {
            "scene_id": np.array(scene_id),
            "viewpoints": np.array([c["viewpoint"] for c in captures]),
            "joint_pos": np.stack([c["joint_pos"] for c in captures]),
            "images_left_camera": np.stack([c["left"] for c in captures]),
            "images_center_camera": np.stack([c["center"] for c in captures]),
            "images_right_camera": np.stack([c["right"] for c in captures]),
            "target_module_name": np.array(
                getattr(task, "target_module_name", "") or ""
            ),
            "port_name": np.array(getattr(task, "port_name", "") or ""),
            "cable_type": np.array(task.cable_type or ""),
            "port_type": np.array(task.port_type or ""),
        }
        if gt_port is not None:
            save["port_xy_world"] = gt_port
        if gt_board is not None:
            save["board"] = gt_board
        np.savez_compressed(out_path, **save)
        self.get_logger().info(f"wrote {out_path} ({len(captures)} viewpoints)")
        _diag(
            "collector_wrote",
            scene_id=scene_id,
            n_viewpoints=len(captures),
            path=str(out_path),
        )
        return True
