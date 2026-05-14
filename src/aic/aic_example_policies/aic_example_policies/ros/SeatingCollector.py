"""SeatingCollector — GT-driven seating + per-tick trajectory recording.

Drops in as a Policy. For each trial:
  1. Look up the TRUE port pose via TF (`ground_truth:=true` is required).
  2. Sample a per-episode lateral perturbation (±INITIAL_LATERAL_NOISE_M)
     so the approach pose is OFF from the true port. The recorded "correction"
     segment that follows demonstrates how to slide back onto the port axis
     using vision; that's the corrective behavior ACT learns.
  3. Plan a densified IK descent trajectory in MuJoCo using the GT port XYZ
     and the per-cable plug-in-gripper offset.
  4. Play the trajectory, recording every 20 Hz tick:
       - 3 wrist images (downsampled IMG_STRIDE=4)
       - 32-D state vector matching DINOv2ACT._build_state() exactly
         (joint pos/vel + TCP pose + tip + port_slot + progress + 6-D wrench)
       - 6-D commanded joint action (the IK output)
       - 6-D wrist wrench (Fx, Fy, Fz, Tx, Ty, Tz)
  5. Dump per-episode .npz to /output/episode_{scene_id}.npz, in a format
     `aggregate_seating_demos.py` can concatenate for training. The npz
     also stores per-episode seat-quality metrics (push_contact_step,
     tail_fz, delta_tail_fz) so the aggregator can filter to successful
     seats only.

Env vars:
  AIC_TRIAL_META  — JSON list of per-trial metadata (same schema as
                    GazeboCollector). Optional but recommended for scene_id +
                    fallback GT.
  AIC_OUTPUT_DIR  — defaults to /output

Frame notes:
  TF gives port + plug poses in `base_link`. The MuJoCo IK works in MuJoCo
  world coordinates. We compute the base_link <-> MuJoCo-world transform once
  at init from the MuJoCo model (both scenes use the same UR5e + task_board
  xacro), then convert any base_link TF lookup into MuJoCo world.
"""

import hashlib
import json
import math
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
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException


# --- Diagnostic egress ---

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


_diag("module_imported", policy="SeatingCollector")


# --- Constants (mostly copied from OnlineIK to keep IK behavior identical) ---

WEIGHTS_DIR = Path(__file__).parent / "weights"
SCENE_PATH = WEIGHTS_DIR / "scene" / "scene.xml"

CONTROL_HZ = 20

JOINT_STIFFNESS = [500.0, 500.0, 500.0, 200.0, 200.0, 200.0]
JOINT_DAMPING = [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
TRANSITION_STIFFNESS = [200.0, 200.0, 200.0, 100.0, 100.0, 100.0]
TRANSITION_DAMPING = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]
# Ultra-soft for the force-feedback push: low stiffness lets misaligned
# cable deflect rather than smash. Wrist joints especially soft so the
# gripper can compliantly find the socket.
PUSH_STIFFNESS = [80.0, 80.0, 80.0, 40.0, 40.0, 40.0]
PUSH_DAMPING = [6.0, 6.0, 6.0, 3.0, 3.0, 3.0]

SAFE_POSE = np.array([-0.16, -1.35, -1.66, -1.69, 1.57, 1.41])
SAFE_POSE_SKIP_TOL = 0.10
HOME_SETTLE_TOL = 0.05
MJ_HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])

# Plug-tip-in-gripper offsets per cable (from OnlineIK).
CABLE_OFFSETS = {
    "sfp_sc_cable":           np.array([0.000, -0.0207, 0.0542]),
    "sfp_sc_cable_reversed":  np.array([0.0,    0.015385, 0.04045]),
}
DEFAULT_CABLE_OFFSET = np.array([0.0, 0.015385, 0.04245])

ABOVE_PORT = 0.060
INSERT_DEPTH = 0.015
APPROACH_STEPS = 100   # 5s
CORRECTION_STEPS = 60  # 3s — recorded "slide perturbed XY back to true XY"
DESCENT_STEPS = 160    # 8s
TRANSITION_HOLD = 25.0

# Lateral perturbation applied to the approach pose. The recorded correction
# segment that follows shows the model "off → centered" motion conditioned on
# wrist-camera images. Seeded per-episode from scene_id so collection is
# reproducible. Magnitude matches the OnlineIK localizer's typical 1-2 cm
# error so the model trains on the in-distribution correction range.
INITIAL_LATERAL_NOISE_M = 0.025

# Force-feedback push phase (after planned descent). With ultra-soft stiffness
# the controller is compliant — misaligned cable deflects, aligned cable seats.
PUSH_EXTRA_M = 0.025       # max additional 2.5 cm below planned final
PUSH_STEPS = 100           # 5 s @ 20 Hz
PUSH_FZ_THRESHOLD_N = 8.0  # POSITIVE delta-Fz above air baseline = bottomed-out
PUSH_DESCENT_SKIP_N = 30.0 # Skip push entirely if descent already built >30N
FINAL_HOLD_STEPS = 30      # 1.5 s of post-push settling (recorded)

# State vector layout (32-D) — must match DINOv2ACT._build_state() exactly.
# port_pos is a PLACEHOLDER at inference (DINOv2ACT.NOMINAL_PORT_POS); the
# model is meant to learn spatial info from VISION, not from this slot.
# Collection uses the same placeholder so training matches inference exactly.
# The trailing 6-D wrench (force + torque) is the SOTA-supported addition for
# v39 — every 2024-2026 contact-rich-insertion paper puts force in the
# observation; v38 was force-blind.
STATE_DIM = 32
TIP_OFFSET_TCP_FRAME = np.array([-0.0018, -0.0189, 0.0547])
NOMINAL_PORT_POS = np.array([0.220, -0.013, 1.273])
WRENCH_DIM = 6

# Image downsample stride
IMG_STRIDE = int(os.environ.get("AIC_IMG_STRIDE", "4"))

OUTPUT_DIR = Path(os.environ.get("AIC_OUTPUT_DIR", "/output"))


def _densify(traj: np.ndarray, max_per_joint_step: float) -> np.ndarray:
    out = [traj[0]]
    for i in range(1, len(traj)):
        prev = traj[i - 1]
        delta = traj[i] - prev
        n = max(1, int(np.ceil(np.abs(delta).max() / max_per_joint_step)))
        for k in range(1, n + 1):
            out.append(prev + (k / n) * delta)
    return np.array(out)


def _ros_image_to_numpy(img_msg) -> np.ndarray | None:
    if img_msg is None or img_msg.height == 0 or img_msg.width == 0:
        return None
    try:
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, -1
        )
    except Exception:
        return None
    img = img[..., :3] if img.shape[2] >= 3 else img
    if IMG_STRIDE > 1:
        img = img[::IMG_STRIDE, ::IMG_STRIDE].copy()
    return img


def _read_wrench(obs) -> np.ndarray:
    """Pull the 6-D wrist wrench (Fx,Fy,Fz,Tx,Ty,Tz) from an observation.

    Returns zeros if the field is missing or malformed (e.g., simulator
    publishing a zero-rate wrench at startup). The same access pattern is
    mirrored in DINOv2ACT._build_state — they MUST match exactly.
    """
    try:
        w = obs.wrist_wrench.wrench
        return np.array([
            w.force.x, w.force.y, w.force.z,
            w.torque.x, w.torque.y, w.torque.z,
        ], dtype=np.float64)
    except Exception:
        return np.zeros(WRENCH_DIM, dtype=np.float64)


def _quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


class SeatingCollector(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _diag("policy_init")

        # ---- Trial meta (for scene_id + fallback GT) ----
        meta_path = os.environ.get("AIC_TRIAL_META", "/trial_meta.json")
        self._trial_meta: list[dict] = []
        try:
            with open(meta_path) as f:
                self._trial_meta = json.load(f)
        except Exception as e:
            self.get_logger().warn(
                f"No trial meta at {meta_path} ({e!r}); using anonymous scene_ids"
            )
        self._trial_idx = 0

        # ---- MuJoCo (for IK + base_link <-> world transform) ----
        import mujoco
        self._mj = mujoco
        self._m = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self._d = mujoco.MjData(self._m)
        mujoco.mj_resetDataKeyframe(self._m, self._d, 0)
        mujoco.mj_forward(self._m, self._d)

        joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ]
        self._tcp_site = mujoco.mj_name2id(
            self._m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp"
        )
        self._qids = np.array([
            self._m.jnt_qposadr[mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in joint_names
        ])
        self._dids = np.array([
            self._m.jnt_dofadr[mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in joint_names
        ])
        self._tcp_quat = np.zeros(4)
        mujoco.mju_mat2Quat(
            self._tcp_quat, self._d.site_xmat[self._tcp_site].flatten()
        )
        self._R_tcp = self._d.site_xmat[self._tcp_site].reshape(3, 3).copy()

        # ---- base_link <-> MuJoCo world transform ----
        # Both Gazebo and MuJoCo place the UR5e base_link at the same world
        # position (same xacro). Read MuJoCo's base_link body to get the
        # transform once.
        try:
            base_id = mujoco.mj_name2id(
                self._m, mujoco.mjtObj.mjOBJ_BODY, "base_link"
            )
            self._base_world_pos = self._d.xpos[base_id].copy()
            self._R_base_world = self._d.xmat[base_id].reshape(3, 3).copy()
        except Exception:
            # Fallback per memory note: world rotated 180° around Z, base_link
            # origin at (0.20, -0.20, 1.14) in world.
            self._base_world_pos = np.array([0.20, -0.20, 1.14])
            self._R_base_world = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                                          dtype=float)
        self.get_logger().info(
            f"base_link in world: pos={self._base_world_pos.round(3)} "
            f"R={self._R_base_world.round(2).tolist()}"
        )

        _diag("policy_ready",
              base_world_pos=self._base_world_pos.tolist(),
              n_trials=len(self._trial_meta))

    # --- TF helpers ---

    def _wait_for_tf(self, target: str, source: str, timeout_s: float = 10.0) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_s)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(target, source, Time())
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source}' -> '{target}'... "
                        f"(needs ground_truth:=true)"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source}' -> '{target}' unavailable after {timeout_s}s"
        )
        return False

    def _baselink_to_world(self, p_baselink: np.ndarray) -> np.ndarray:
        """Convert a point in base_link frame to MuJoCo world frame."""
        return self._R_base_world @ p_baselink + self._base_world_pos

    def _lookup_port_world(self, task: Task) -> np.ndarray | None:
        """TF lookup for true port pose, returned in MuJoCo world coords."""
        port_frame = (
            f"task_board/{task.target_module_name}/{task.port_name}_link"
        )
        if not self._wait_for_tf("base_link", port_frame, timeout_s=15.0):
            return None
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time(),
            )
        except TransformException as ex:
            self.get_logger().error(f"Could not look up {port_frame}: {ex}")
            return None
        p_baselink = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ])
        p_world = self._baselink_to_world(p_baselink)
        self.get_logger().info(
            f"port {port_frame}: base_link={p_baselink.round(4)} → world={p_world.round(4)}"
        )
        return p_world

    # --- IK + trajectory (lifted from OnlineIK) ---

    def _solve_ik(self, target_pos: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        m, d = self._m, self._d
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        d_ik = self._mj.MjData(m)
        d_ik.qpos[:] = d.qpos[:]
        d_ik.qpos[self._qids] = q_init.copy()
        for _ in range(150):
            self._mj.mj_forward(m, d_ik)
            pe = target_pos - d_ik.site_xpos[self._tcp_site]
            sm = d_ik.site_xmat[self._tcp_site].reshape(3, 3)
            tm = np.zeros(9)
            self._mj.mju_quat2Mat(tm, self._tcp_quat)
            tm = tm.reshape(3, 3)
            Re = tm @ sm.T
            eq = np.zeros(4)
            self._mj.mju_mat2Quat(eq, Re.flatten())
            if eq[0] < 0:
                eq = -eq
            re = 2.0 * eq[1:4]
            if np.linalg.norm(pe) < 3e-4 and np.linalg.norm(re) < 0.01:
                break
            err = np.concatenate([pe, 0.5 * re])
            self._mj.mj_jacSite(m, d_ik, jacp, jacr, self._tcp_site)
            J = np.vstack([jacp[:, self._dids], jacr[:, self._dids]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            d_ik.qpos[self._qids] += 0.15 * dq
        return d_ik.qpos[self._qids].copy()

    def _build_trajectory(self, port_world_xyz: np.ndarray,
                          gripper_offset: np.ndarray,
                          rng: np.random.Generator,
                          ) -> tuple[np.ndarray, np.ndarray]:
        """Plan approach → correction → descent.

        Approach moves to a perturbed `tcp_above + lateral_noise`.
        Correction smoothly slides XY back to the true `tcp_above` over
        CORRECTION_STEPS while holding Z. Descent then drops Z linearly to
        port_z - INSERT_DEPTH.

        The correction segment is what teaches ACT to recover from lateral
        misalignment using wrist-camera images. Descent is unchanged.
        """
        tcp_home = self._d.site_xpos[self._tcp_site].copy()
        tip_offset = self._R_tcp @ gripper_offset
        tcp_above = port_world_xyz + np.array([0, 0, ABOVE_PORT]) - tip_offset

        # Per-episode lateral perturbation for the approach pose.
        noise_xy = rng.uniform(
            -INITIAL_LATERAL_NOISE_M, INITIAL_LATERAL_NOISE_M, size=2,
        )
        tcp_above_perturbed = tcp_above.copy()
        tcp_above_perturbed[0] += noise_xy[0]
        tcp_above_perturbed[1] += noise_xy[1]

        approach = np.zeros((APPROACH_STEPS, 6))
        q_prev = MJ_HOME.copy()
        for i in range(APPROACH_STEPS):
            alpha = 0.5 * (1 - math.cos(math.pi * i / max(APPROACH_STEPS - 1, 1)))
            tcp_i = (1 - alpha) * tcp_home + alpha * tcp_above_perturbed
            approach[i] = self._solve_ik(tcp_i, q_prev)
            q_prev = approach[i]

        # Correction: smooth interpolate XY from perturbed back to true,
        # holding Z. Cosine ramp matches the approach for smooth blending.
        correction = np.zeros((CORRECTION_STEPS, 6))
        for i in range(CORRECTION_STEPS):
            alpha = 0.5 * (
                1 - math.cos(math.pi * (i + 1) / max(CORRECTION_STEPS, 1))
            )
            tcp_i = (1 - alpha) * tcp_above_perturbed + alpha * tcp_above
            correction[i] = self._solve_ik(tcp_i, q_prev)
            q_prev = correction[i]

        descent = np.zeros((DESCENT_STEPS, 6))
        for i in range(DESCENT_STEPS):
            z_off = ABOVE_PORT - (i / max(DESCENT_STEPS - 1, 1)) * (
                ABOVE_PORT + INSERT_DEPTH
            )
            tcp_i = tcp_above.copy()
            tcp_i[2] = port_world_xyz[2] + z_off - tip_offset[2]
            descent[i] = self._solve_ik(tcp_i, q_prev)
            q_prev = descent[i]
        return np.concatenate([approach, correction, descent], axis=0), noise_xy

    # --- Per-tick state vector (must match DINOv2ACT._build_state exactly) ---

    def _build_state(self, obs, elapsed: float,
                     traj_total_s: float) -> np.ndarray | None:
        js = obs.joint_states
        cs = obs.controller_state
        if len(js.position) < 6 or len(js.velocity) < 6:
            return None
        joint_pos = np.array(js.position[:6])
        joint_vel = np.array(js.velocity[:6])
        tcp_pos = np.array([
            cs.tcp_pose.position.x,
            cs.tcp_pose.position.y,
            cs.tcp_pose.position.z,
        ])
        q = cs.tcp_pose.orientation
        tcp_quat = np.array([q.w, q.x, q.y, q.z])
        tip_pos = tcp_pos + _quat_to_rot(tcp_quat) @ TIP_OFFSET_TCP_FRAME
        progress = min(elapsed / max(traj_total_s, 1e-3), 1.0)
        wrench = _read_wrench(obs)
        # port_pos slot uses the SAME placeholder as DINOv2ACT inference, so
        # the model can't learn to rely on it; vision must carry spatial info.
        return np.concatenate([
            joint_pos, joint_vel, tcp_pos, tcp_quat,
            tip_pos, NOMINAL_PORT_POS, [progress], wrench,
        ])

    # --- Main entry point ---

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        # ---- Per-trial meta lookup (for scene_id) ----
        if self._trial_idx < len(self._trial_meta):
            meta = self._trial_meta[self._trial_idx]
        else:
            meta = {"scene_id": f"trial_{self._trial_idx}_{int(time.time())}"}
        scene_id = meta.get("scene_id", f"trial_{self._trial_idx}")
        self._trial_idx += 1

        cable_type = (task.cable_type or "").lower()
        port_type = (task.port_type or "").lower()
        plug_type = (task.plug_type or "").lower()
        # task.cable_type is the SHORT form ("sfp_sc") which is the same for
        # both SFP and SC trials of this cable family. CABLE_OFFSETS is keyed
        # on the LONG form (sfp_sc_cable for SFP plug end, sfp_sc_cable_reversed
        # for SC plug end). Disambiguate via plug_type / port_type.
        if "sc" in port_type and "sfp" not in port_type:
            gripper_offset = CABLE_OFFSETS["sfp_sc_cable_reversed"]
        else:
            gripper_offset = CABLE_OFFSETS.get(
                "sfp_sc_cable", DEFAULT_CABLE_OFFSET,
            )

        _diag(
            "collector_insert_enter",
            scene_id=scene_id,
            trial_idx=self._trial_idx - 1,
            cable_type=task.cable_type,
            port_type=task.port_type,
            target_module_name=getattr(task, "target_module_name", None),
            port_name=getattr(task, "port_name", None),
            gripper_offset=gripper_offset.tolist(),
        )

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
        if start_pose is None:
            self.get_logger().warn("No initial obs; using SAFE_POSE")
            start_pose = SAFE_POSE.copy()

        # ---- Park at SAFE_POSE ----
        if np.abs(start_pose - SAFE_POSE).max() > SAFE_POSE_SKIP_TOL:
            _swing(start_pose, SAFE_POSE, "park→safe")
            _settle(SAFE_POSE, 6.0, "safe")

        # ---- GT port world XYZ ----
        # Prefer meta-supplied gt_xyz (computed locally via MuJoCo, already in
        # the IK frame). TF lookup is only a fallback — TF gives base_link
        # coords and our base_link→world transform is unreliable.
        gt = meta.get("gt_xyz")
        if gt is not None:
            port_world = np.array(gt, dtype=np.float64)
            self.get_logger().info(f"port_world (from meta): {port_world.round(4)}")
        else:
            port_world = self._lookup_port_world(task)
            if port_world is None:
                self.get_logger().error("No GT port pose; aborting trial")
                _diag("collector_no_gt", scene_id=scene_id)
                return False

        _diag("collector_port_gt", scene_id=scene_id, port_world=port_world.tolist())

        # ---- Plan IK trajectory (with per-episode lateral perturbation) ----
        # Seed RNG from scene_id so collection is reproducible. The
        # perturbation is small (±2.5 cm) and the recorded "correction"
        # segment that follows is what teaches ACT to slide back onto the
        # port axis using wrist-camera images.
        seed = int(hashlib.md5(scene_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        traj, noise_xy_logged = self._build_trajectory(
            port_world, gripper_offset, rng,
        )
        self.get_logger().info(
            f"perturbation noise_xy = {noise_xy_logged.round(4).tolist()} m"
        )
        traj_dense = _densify(traj, max_per_joint_step=0.0015)
        traj_total_s = len(traj_dense) / CONTROL_HZ
        self.get_logger().info(
            f"trajectory: {len(traj_dense)} steps ({traj_total_s:.1f}s)"
        )

        # ---- Swing safe → first_target ----
        first_target = traj_dense[0]
        _swing(start_pose, first_target, "safe→first_target")
        _settle(first_target, TRANSITION_HOLD, "first_target")

        # ---- Play trajectory + record per-tick ----
        rec_states: list[np.ndarray] = []
        rec_actions: list[np.ndarray] = []
        rec_wrench: list[np.ndarray] = []  # full 6-D wrench per tick
        rec_imgs_left: list[np.ndarray] = []
        rec_imgs_center: list[np.ndarray] = []
        rec_imgs_right: list[np.ndarray] = []
        recs = (rec_states, rec_actions, rec_wrench,
                rec_imgs_left, rec_imgs_center, rec_imgs_right)

        play_start = self.time_now()

        def _step_and_record(q_step: np.ndarray, msg_obj) -> float | None:
            """Send one joint command and record the resulting obs. Returns Fz
            (or None if the tick was skipped). Sleeps one control period at end.
            """
            elapsed = (self.time_now() - play_start).nanoseconds / 1e9
            msg_obj.target_state.positions = q_step.tolist()
            move_robot(joint_motion_update=msg_obj)
            obs_t = get_observation()
            if obs_t is None:
                self.sleep_for(1.0 / CONTROL_HZ)
                return None
            state = self._build_state(obs_t, elapsed, traj_total_s)
            if state is None:
                self.sleep_for(1.0 / CONTROL_HZ)
                return None
            img_l = _ros_image_to_numpy(obs_t.left_image)
            img_c = _ros_image_to_numpy(obs_t.center_image)
            img_r = _ros_image_to_numpy(obs_t.right_image)
            if img_l is None or img_c is None or img_r is None:
                self.sleep_for(1.0 / CONTROL_HZ)
                return None
            wrench = _read_wrench(obs_t)
            recs[0].append(state)
            recs[1].append(q_step.astype(np.float32))
            recs[2].append(wrench.astype(np.float32))
            recs[3].append(img_l)
            recs[4].append(img_c)
            recs[5].append(img_r)
            self.sleep_for(1.0 / CONTROL_HZ)
            return float(wrench[2])

        for step, q in enumerate(traj_dense):
            fz = _step_and_record(q, msg)
            if step % 80 == 0:
                fz_str = f"{fz:.2f}" if fz is not None else "skip"
                elapsed = (self.time_now() - play_start).nanoseconds / 1e9
                send_feedback(
                    f"t={elapsed:.1f}s step={step}/{len(traj_dense)} fz={fz_str}"
                )

        # ---- Force-feedback final push ----
        # Even with GT pose + measured cable offsets, residual error (~5-10 mm)
        # can leave the plug above the port. Switch to softer stiffness and
        # push extra 4 cm in 6 s. Recording stays on so the wrench tail
        # captures the seat signature for training.
        # Baseline = mean of FIRST 20 ticks (settled at first_target, in air,
        # no port contact). Last-20 baseline is contaminated by descent contact.
        baseline_n = min(20, len(rec_wrench))
        fz_baseline = (
            float(np.mean([w[2] for w in rec_wrench[:baseline_n]]))
            if baseline_n else 0.0
        )

        # Anchor the push at the FK position of the last commanded q. Use a
        # scratch MjData so we don't mutate self._d (which would corrupt
        # trial 2's tcp_home in _build_trajectory).
        q_anchor = traj_dense[-1].copy()
        d_anchor = self._mj.MjData(self._m)
        d_anchor.qpos[:] = self._d.qpos[:]
        d_anchor.qpos[self._qids] = q_anchor
        self._mj.mj_forward(self._m, d_anchor)
        tcp_anchor = d_anchor.site_xpos[self._tcp_site].copy()

        # Skip push entirely if descent already built up high force —
        # pushing further would just smash. (Trial 5 earlier built 82N then
        # push amplified to 643N — bad for both training data and safety.)
        descent_end_fz = (
            float(np.mean([w[2] for w in rec_wrench[-10:]]))
            if len(rec_wrench) >= 10 else 0.0
        )
        descent_delta = descent_end_fz - fz_baseline
        skip_push = abs(descent_delta) > PUSH_DESCENT_SKIP_N

        # Ultra-soft stiffness for the push — controller backs off on
        # misalignment, allowing compliant alignment + seating.
        push_msg = JointMotionUpdate(
            target_stiffness=PUSH_STIFFNESS,
            target_damping=PUSH_DAMPING,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )

        push_contact_step = -1
        q_prev = q_anchor.copy()
        last_q_push = q_anchor.copy()

        if skip_push:
            self.get_logger().info(
                f"push: SKIPPED — descent already loaded "
                f"(end_fz={descent_end_fz:.2f}N, baseline={fz_baseline:.2f}N, "
                f"delta={descent_delta:+.2f}N > ±{PUSH_DESCENT_SKIP_N:.0f}N)"
            )
            push_contact_step = -2  # special marker for "skipped"
        else:
            for i in range(PUSH_STEPS):
                z_extra = (i + 1) / PUSH_STEPS * PUSH_EXTRA_M
                tcp_push = tcp_anchor.copy()
                tcp_push[2] -= z_extra
                q_push = self._solve_ik(tcp_push, q_prev)
                q_prev = q_push
                last_q_push = q_push

                fz = _step_and_record(q_push, push_msg)
                if fz is None:
                    continue
                # Only abort on POSITIVE delta — the gripper bottoming out and
                # generating UPWARD reaction force at the wrist. Fz drops mean
                # the cable slipped into the socket (good) so keep pushing.
                if (fz - fz_baseline) > PUSH_FZ_THRESHOLD_N:
                    push_contact_step = i
                    self.get_logger().info(
                        f"push: bottomed-out at step {i}/{PUSH_STEPS} "
                        f"(fz={fz:.2f}N, baseline={fz_baseline:.2f}N, "
                        f"delta=+{fz-fz_baseline:.2f}N)"
                    )
                    break

            self.get_logger().info(
                f"push: ended at step {push_contact_step if push_contact_step>=0 else PUSH_STEPS}"
                f"/{PUSH_STEPS} (z_extra={(push_contact_step+1 if push_contact_step>=0 else PUSH_STEPS)/PUSH_STEPS*PUSH_EXTRA_M*1000:.1f}mm)"
            )

        # ---- Final hold so seat signature lands in the recorded wrench tail ----
        for _ in range(FINAL_HOLD_STEPS):
            _step_and_record(last_q_push, push_msg)

        # ---- Save episode ----
        if not rec_states:
            self.get_logger().error("No recorded ticks; not writing output")
            _diag("collector_empty", scene_id=scene_id)
            return False

        # Write atomically: tmp file then rename. Otherwise the polling worker
        # may detect the file mid-write and tear down compose before
        # savez_compressed finishes flushing the zip.
        # NOTE: tmp name MUST end in .npz; savez_compressed auto-appends .npz
        # if the filename doesn't already end with it (turning foo.npz.tmp
        # into foo.npz.tmp.npz, breaking the rename).
        out_path = OUTPUT_DIR / f"episode_{scene_id}.npz"
        tmp_path = OUTPUT_DIR / f".tmp_{scene_id}.npz"
        wrench_arr = (
            np.stack(rec_wrench).astype(np.float32)
            if rec_wrench
            else np.zeros((0, WRENCH_DIM), dtype=np.float32)
        )
        peak_fz = (
            float(np.max(np.abs(wrench_arr[:, 2]))) if wrench_arr.size else 0.0
        )
        tail_n = min(FINAL_HOLD_STEPS, wrench_arr.shape[0])
        tail_fz = (
            float(np.mean(wrench_arr[-tail_n:, 2])) if tail_n else 0.0
        )
        delta_tail = tail_fz - fz_baseline
        save = {
            "scene_id": np.array(scene_id),
            "states": np.stack(rec_states).astype(np.float32),
            "actions": np.stack(rec_actions).astype(np.float32),
            "wrench": wrench_arr,                     # (T, 6) full wrench
            "wrench_z": wrench_arr[:, 2].copy(),      # (T,) Fz for back-compat
            "images_left_camera": np.stack(rec_imgs_left),
            "images_center_camera": np.stack(rec_imgs_center),
            "images_right_camera": np.stack(rec_imgs_right),
            "port_world": port_world.astype(np.float32),
            "cable_type": np.array(task.cable_type or ""),
            "port_type": np.array(task.port_type or ""),
            "target_module_name": np.array(
                getattr(task, "target_module_name", "") or ""
            ),
            "port_name": np.array(getattr(task, "port_name", "") or ""),
            "gripper_offset": gripper_offset.astype(np.float32),
            # v39 seat-quality metrics — aggregator filters on these
            "push_contact_step": np.array(push_contact_step, dtype=np.int32),
            "tail_fz": np.array(tail_fz, dtype=np.float32),
            "delta_tail_fz": np.array(delta_tail, dtype=np.float32),
            "fz_baseline": np.array(fz_baseline, dtype=np.float32),
            "lateral_noise_xy": np.array(noise_xy_logged, dtype=np.float32),
        }
        np.savez_compressed(tmp_path, **save)
        os.replace(tmp_path, out_path)  # atomic rename
        self.get_logger().info(
            f"wrote {out_path} ({len(rec_states)} ticks, peak |Fz|={peak_fz:.2f}N, "
            f"tail Fz={tail_fz:+.2f}N, deltaFz_tail={delta_tail:+.2f}N, "
            f"push_contact_step={push_contact_step})"
        )
        _diag(
            "collector_wrote",
            scene_id=scene_id,
            n_ticks=len(rec_states),
            peak_fz=peak_fz,
            tail_fz=tail_fz,
            delta_tail_fz=delta_tail,
            push_contact_step=push_contact_step,
            path=str(out_path),
        )
        return True
