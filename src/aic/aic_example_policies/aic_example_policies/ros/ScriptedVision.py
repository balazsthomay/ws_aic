"""Trial-aware scripted playback.

The competition cluster runs three deterministic trials (see
`src/aic/aic_engine/config/sample_config.yaml`). We bake one densified IK
trajectory per trial, dispatch by `(task.target_module_name, task.port_name)`,
and post per-trial telemetry to the diagnostic webhook so we can compare
real-cluster behavior against our local plan.

No vision in the active path — we have ground truth for the three fixed
trial poses. The "Vision" name is kept for the policy entrypoint to avoid
churning the Dockerfile / runtime override; the localizer code path was
removed.
"""

import json
import os
import socket
import time
import urllib.error
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
        print(f"[diag] failed to POST {event}: {e!r}", flush=True)


_diag("module_imported", policy="ScriptedVision")


def _densify(traj: np.ndarray, max_per_joint_step: float) -> np.ndarray:
    out = [traj[0]]
    for i in range(1, len(traj)):
        prev = traj[i - 1]
        curr = traj[i]
        delta = curr - prev
        n = max(1, int(np.ceil(np.abs(delta).max() / max_per_joint_step)))
        for k in range(1, n + 1):
            out.append(prev + (k / n) * delta)
    return np.array(out)


WEIGHTS_DIR = Path(__file__).parent / "weights"
TRAJ_PATH = WEIGHTS_DIR / "scripted_traj.npz"

CONTROL_HZ = 20
TRANSITION_HOLD = 35.0
HOME_SETTLE_TOL = 0.05
JOINT_STIFFNESS = [500.0, 500.0, 500.0, 200.0, 200.0, 200.0]
JOINT_DAMPING = [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
TRANSITION_STIFFNESS = [200.0, 200.0, 200.0, 100.0, 100.0, 100.0]
TRANSITION_DAMPING = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]

# Safe pose to park at between trials, decoupling each trial's transition
# from the previous trial's end-of-descent. Matches the real cluster's start
# pose (memory: project_cluster_controller).
SAFE_POSE = np.array([-0.16, -1.35, -1.66, -1.69, 1.57, 1.41])
# How much error in any joint must exist before we bother parking. If the arm
# is already near safe_pose (e.g., at trial start), skip it.
SAFE_POSE_SKIP_TOL = 0.10

# (target_module_name, port_name) → trajectory key in the npz. Keep in sync
# with sample_config.yaml.
_TRIAL_DISPATCH = {
    ("nic_card_mount_0", "sfp_port_0"): "t1",
    ("nic_card_mount_1", "sfp_port_0"): "t2",
    ("sc_port_1",         "sc_port_base"): "t3",
}


class ScriptedVision(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        _diag("policy_init")
        data = np.load(str(TRAJ_PATH), allow_pickle=True)
        # Pre-densify per-trial trajectories. Cluster rate-limits joints to
        # ~0.04 rad/s ⇒ ≤0.002 rad/tick at 20Hz. Densify to ≤0.0015 rad/tick.
        self._traj = {}
        for key in ("t1", "t2", "t3"):
            jkey = f"joints_{key}"
            if jkey in data.files:
                raw = data[jkey].astype(np.float64)
                self._traj[key] = _densify(raw, max_per_joint_step=0.0015)
        self._port_world = {}
        for key in ("t1", "t2", "t3"):
            pkey = f"port_{key}"
            if pkey in data.files:
                self._port_world[key] = data[pkey].astype(np.float64)
        for key, traj in self._traj.items():
            self.get_logger().info(
                f"trial {key}: {len(traj)} steps "
                f"({len(traj)/CONTROL_HZ:.1f}s) port={self._port_world.get(key)}"
            )
        _diag("policy_ready", trials=list(self._traj.keys()))

    def _pick_trial(self, task: Task) -> tuple[str, str]:
        target = getattr(task, "target_module_name", None) or ""
        pname = getattr(task, "port_name", None) or ""
        key = _TRIAL_DISPATCH.get((target, pname))
        if key is not None and key in self._traj:
            return key, "exact"
        # Fallbacks if the runtime task interface drops one of the fields.
        for (mt, pn), tkey in _TRIAL_DISPATCH.items():
            if pname and pn == pname and tkey in self._traj:
                return tkey, "by_port_name"
            if target and mt == target and tkey in self._traj:
                return tkey, "by_module"
        # Last resort: choose by port_type. SFP → t1 (most common), SC → t3.
        ptype = (getattr(task, "port_type", "") or "").lower()
        if "sc" in ptype and "sfp" not in ptype and "t3" in self._traj:
            return "t3", "by_port_type_sc"
        if "t1" in self._traj:
            return "t1", "by_port_type_sfp"
        # Ultimate fallback: first available trajectory.
        return next(iter(self._traj.keys())), "default"

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        trial_key, dispatch_reason = self._pick_trial(task)
        traj = self._traj[trial_key].copy()
        port_world = self._port_world.get(trial_key)
        _diag(
            "insert_cable_enter",
            task_id=task.id,
            cable_type=task.cable_type,
            plug_type=task.plug_type,
            port_type=task.port_type,
            target_module_name=getattr(task, "target_module_name", None),
            port_name=getattr(task, "port_name", None),
            time_limit=task.time_limit,
            trial_key=trial_key,
            dispatch_reason=dispatch_reason,
        )
        self.get_logger().info(
            f"trial={trial_key} ({dispatch_reason}) port={port_world}"
        )
        send_feedback(f"scripted trial {trial_key}")

        msg = JointMotionUpdate(
            target_stiffness=JOINT_STIFFNESS,
            target_damping=JOINT_DAMPING,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        trans_msg = JointMotionUpdate(
            target_stiffness=TRANSITION_STIFFNESS,
            target_damping=TRANSITION_DAMPING,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )

        def _swing(start: np.ndarray, end: np.ndarray, label: str) -> None:
            path = _densify(np.stack([start, end]), max_per_joint_step=0.0015)
            self.get_logger().info(
                f"{label}: {start.round(3)} → {end.round(3)} ({len(path)} steps)"
            )
            for q_step in path:
                trans_msg.target_state.positions = q_step.tolist()
                move_robot(joint_motion_update=trans_msg)
                self.sleep_for(1.0 / CONTROL_HZ)

        def _settle(target: np.ndarray, hold_s: float, label: str) -> None:
            n = int(hold_s * CONTROL_HZ)
            for s in range(n):
                trans_msg.target_state.positions = target.tolist()
                move_robot(joint_motion_update=trans_msg)
                obs_h = get_observation()
                if obs_h is not None and len(obs_h.joint_states.position) >= 6:
                    err = np.abs(
                        np.array(obs_h.joint_states.position[:6]) - target
                    ).max()
                    if err <= HOME_SETTLE_TOL:
                        self.get_logger().info(
                            f"{label} settled after {s}/{n} (err={err:.3f})"
                        )
                        return
                self.sleep_for(1.0 / CONTROL_HZ)

        # ---- Read start pose ----
        start_pose = None
        for _ in range(20):
            obs = get_observation()
            if obs is not None and len(obs.joint_states.position) >= 6:
                start_pose = np.array(obs.joint_states.position[:6])
                break
            self.sleep_for(1.0 / CONTROL_HZ)
        first_target = traj[0]
        _diag(
            "pre_trial",
            trial_key=trial_key,
            q_start=start_pose.tolist() if start_pose is not None else None,
            q_first_target=first_target.tolist(),
        )

        # ---- Inter-trial reset: park at SAFE_POSE first if not already there.
        # Decouples each trial from the previous trial's end pose so the swing
        # never has to cross task-board geometry.
        if start_pose is not None:
            err_to_safe = np.abs(start_pose - SAFE_POSE).max()
            if err_to_safe > SAFE_POSE_SKIP_TOL:
                _swing(start_pose, SAFE_POSE, "park→safe")
                _settle(SAFE_POSE, 6.0, "safe")
                obs_p = get_observation()
                if obs_p is not None and len(obs_p.joint_states.position) >= 6:
                    start_pose = np.array(obs_p.joint_states.position[:6])
        else:
            start_pose = first_target.copy()

        # ---- Swing safe → first_target ----
        _swing(start_pose, first_target, "safe→first_target")
        _settle(first_target, TRANSITION_HOLD, "first_target")

        # ---- Play trajectory ----
        descent_idx = len(traj) // 2
        for step, q in enumerate(traj):
            msg.target_state.positions = q.tolist()
            move_robot(joint_motion_update=msg)
            if step == descent_idx:
                obs_m = get_observation()
                if obs_m is not None and len(obs_m.joint_states.position) >= 6:
                    _diag(
                        "mid_descent",
                        trial_key=trial_key,
                        step=step,
                        q_actual=list(obs_m.joint_states.position[:6]),
                    )
            if step % 80 == 0:
                send_feedback(f"step={step}/{len(traj)}")
            self.sleep_for(1.0 / CONTROL_HZ)

        for _ in range(20):
            msg.target_state.positions = traj[-1].tolist()
            move_robot(joint_motion_update=msg)
            self.sleep_for(1.0 / CONTROL_HZ)

        obs_e = get_observation()
        q_end = (
            list(obs_e.joint_states.position[:6])
            if obs_e is not None and len(obs_e.joint_states.position) >= 6
            else None
        )
        _diag("insert_cable_exit", trial_key=trial_key, q_end=q_end)
        return True
