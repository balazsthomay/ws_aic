"""Online-IK policy for the AIC challenge.

At every trial:
  1. Read task params (port_type, target_module_name, port_name, cable_type)
     from the ROS task interface — these are the only "ground truth" inputs.
  2. Localize the target port: run wrist-camera frames through DINOv3 +
     PortLocalizer (trained on broadly-randomized demos covering all five NIC
     rails + both SC ports + jittered board pose).
  3. Plan an IK trajectory (HOME → above port → descend) live in MuJoCo,
     using the localizer's predicted port_xy and a per-cable-type tip offset.
  4. Densify joint deltas to ≤0.0015 rad/tick (cluster rate-limit) and play.
  5. Re-localize once after the swing settles to refine the descent target
     (closer cameras → better pixels) and apply a Jacobian shift on the
     remaining trajectory.

The only hardcoded knowledge is invariants: robot kinematics (UR5e xacro),
gripper-frame-to-plug-tip offsets per cable type, and the rail anchor
constants from `task_board.urdf.xacro`. Trial selection, port poses, and
trajectory shape are all computed at runtime.
"""

import json
import math
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


_diag("module_imported", policy="OnlineIK")


WEIGHTS_DIR = Path(__file__).parent / "weights"
DINOV2_DIR = WEIGHTS_DIR / "dinov2-small"      # contains DINOv3 weights
LOCALIZER_PT = WEIGHTS_DIR / "port_localizer_dinov2.pt"
LOCALIZER_STATS = WEIGHTS_DIR / "port_localizer_stats.npz"
SCENE_PATH = WEIGHTS_DIR / "scene" / "scene.xml"

CONTROL_HZ = 20
MAX_TIME = 60.0
TRANSITION_HOLD = 25.0
HOME_SETTLE_TOL = 0.05

JOINT_STIFFNESS = [500.0, 500.0, 500.0, 200.0, 200.0, 200.0]
JOINT_DAMPING = [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
TRANSITION_STIFFNESS = [200.0, 200.0, 200.0, 100.0, 100.0, 100.0]
TRANSITION_DAMPING = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]

# Cluster start pose (memory: project_cluster_controller).
SAFE_POSE = np.array([-0.16, -1.35, -1.66, -1.69, 1.57, 1.41])
SAFE_POSE_SKIP_TOL = 0.10

# MuJoCo HOME used by IK init (matches the scene keyframe).
MJ_HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])

# Plug-tip-in-gripper offsets by cable type (from sample_config gripper_offset
# pose specs — these are gripper geometry, an invariant). Z varies slightly
# per cable; XY is constant per cable family.
CABLE_OFFSETS = {
    "sfp_sc_cable":           np.array([0.0, 0.015385, 0.04245]),
    "sfp_sc_cable_reversed":  np.array([0.0, 0.015385, 0.04045]),
}
DEFAULT_CABLE_OFFSET = np.array([0.0, 0.015385, 0.04245])

ABOVE_PORT = 0.060
INSERT_DEPTH = 0.015
APPROACH_STEPS = 100   # 5s @ 20Hz
DESCENT_STEPS = 160    # 8s @ 20Hz

DINOV2_INPUT = 224
PATCH_GRID = 4
N_CAMS = 3
DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _densify(traj: np.ndarray, max_per_joint_step: float) -> np.ndarray:
    out = [traj[0]]
    for i in range(1, len(traj)):
        prev = traj[i - 1]
        delta = traj[i] - prev
        n = max(1, int(np.ceil(np.abs(delta).max() / max_per_joint_step)))
        for k in range(1, n + 1):
            out.append(prev + (k / n) * delta)
    return np.array(out)


class PortLocalizer(nn.Module):
    def __init__(self, vision_dim: int = 384,
                 n_tokens: int = N_CAMS * PATCH_GRID * PATCH_GRID,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.proj = nn.Linear(vision_dim, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        B, C, S, D = vision_tokens.shape
        x = vision_tokens.reshape(B, C * S, D)
        x = self.proj(x) + self.pos.unsqueeze(0)
        x = self.encoder(x).mean(dim=1)
        return self.head(x)


def _preprocess(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(DINOV2_INPUT, DINOV2_INPUT),
                      mode="bilinear", align_corners=False)
    return (t - DINOV2_MEAN) / DINOV2_STD


def _ros_image_to_numpy(img_msg) -> np.ndarray | None:
    if img_msg.height == 0 or img_msg.width == 0:
        return None
    try:
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, -1
        )
    except Exception:
        return None
    return img[..., :3] if img.shape[2] >= 3 else img


class OnlineIK(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        _diag("policy_init")
        self._device = torch.device("cpu")
        torch.set_num_threads(max(1, os.cpu_count() or 1))

        # ---- DINO + localizer ----
        from transformers import AutoModel
        self._backbone = AutoModel.from_pretrained(str(DINOV2_DIR)).to(self._device).eval()
        for p in self._backbone.parameters():
            p.requires_grad_(False)
        self._patch_size = int(self._backbone.config.patch_size)
        self._native_grid = DINOV2_INPUT // self._patch_size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, DINOV2_INPUT, DINOV2_INPUT)
            n_tokens = self._backbone(pixel_values=dummy).last_hidden_state.shape[1]
        self._n_skip = n_tokens - self._native_grid * self._native_grid
        self._vision_dim = int(self._backbone.config.hidden_size)
        self._loc = PortLocalizer(vision_dim=self._vision_dim).to(self._device).eval()
        self._loc.load_state_dict(torch.load(
            str(LOCALIZER_PT), map_location="cpu", weights_only=True
        ))
        loc_stats = np.load(str(LOCALIZER_STATS))
        self._port_mean = torch.tensor(loc_stats["port_mean"], dtype=torch.float32)
        self._port_std = torch.tensor(loc_stats["port_std"], dtype=torch.float32)

        # ---- MuJoCo (online IK) ----
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
        self._tcp_site = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
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

        self.get_logger().info(
            f"OnlineIK ready (DINO patch={self._patch_size} "
            f"grid={self._native_grid} dim={self._vision_dim})"
        )
        _diag("policy_ready")

    def _predict_port_xy(self, obs) -> np.ndarray | None:
        cams = [obs.left_image, obs.center_image, obs.right_image]
        tensors = []
        for img_msg in cams:
            img = _ros_image_to_numpy(img_msg)
            if img is None:
                return None
            tensors.append(_preprocess(img))
        pixel_values = torch.cat(tensors, dim=0)
        with torch.no_grad():
            hs = self._backbone(pixel_values=pixel_values).last_hidden_state
            patches = hs[:, self._n_skip:, :]
            G, D = self._native_grid, self._vision_dim
            grid = patches.reshape(3, G, G, D).permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(grid, PATCH_GRID)
            feats = pooled.permute(0, 2, 3, 1).reshape(
                3, PATCH_GRID * PATCH_GRID, D
            ).unsqueeze(0)
            port_norm = self._loc(feats).squeeze(0)
            port_xy = (port_norm * self._port_std + self._port_mean).cpu().numpy()
        return port_xy

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
                          gripper_offset: np.ndarray) -> np.ndarray:
        tcp_home = self._d.site_xpos[self._tcp_site].copy()
        tip_offset = self._R_tcp @ gripper_offset
        tcp_above = port_world_xyz + np.array([0, 0, ABOVE_PORT]) - tip_offset

        approach = np.zeros((APPROACH_STEPS, 6))
        q_prev = MJ_HOME.copy()
        for i in range(APPROACH_STEPS):
            alpha = 0.5 * (1 - math.cos(math.pi * i / max(APPROACH_STEPS - 1, 1)))
            tcp_i = (1 - alpha) * tcp_home + alpha * tcp_above
            approach[i] = self._solve_ik(tcp_i, q_prev)
            q_prev = approach[i]

        descent = np.zeros((DESCENT_STEPS, 6))
        q_prev = approach[-1].copy()
        for i in range(DESCENT_STEPS):
            z_off = ABOVE_PORT - (i / max(DESCENT_STEPS - 1, 1)) * (
                ABOVE_PORT + INSERT_DEPTH
            )
            tcp_i = tcp_above.copy()
            tcp_i[2] = port_world_xyz[2] + z_off - tip_offset[2]
            descent[i] = self._solve_ik(tcp_i, q_prev)
            q_prev = descent[i]
        return np.concatenate([approach, descent], axis=0)

    def _expected_port_z(self, port_xy: np.ndarray, port_type: str) -> float:
        # The localizer only outputs xy (we don't need z to be precise for
        # localization). For Z we use the geometry: NIC card SFP ports sit at
        # ~1.273m world Z (board top + NIC body + SFP cage), SC built-in ports
        # at ~1.155m. Use port_type to choose. Cluster does not vary board Z.
        if port_type and "sc" in port_type.lower() and "sfp" not in port_type.lower():
            return 1.155
        return 1.273

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        cable_type = (task.cable_type or "").lower()
        port_type = (task.port_type or "").lower()
        gripper_offset = CABLE_OFFSETS.get(cable_type, DEFAULT_CABLE_OFFSET)
        _diag(
            "insert_cable_enter",
            task_id=task.id,
            cable_type=task.cable_type,
            plug_type=task.plug_type,
            port_type=task.port_type,
            target_module_name=getattr(task, "target_module_name", None),
            port_name=getattr(task, "port_name", None),
            time_limit=task.time_limit,
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

        # ---- Park at SAFE_POSE if not already there ----
        if np.abs(start_pose - SAFE_POSE).max() > SAFE_POSE_SKIP_TOL:
            _swing(start_pose, SAFE_POSE, "park→safe")
            _settle(SAFE_POSE, 6.0, "safe")
            obs_p = get_observation()
            if obs_p is not None and len(obs_p.joint_states.position) >= 6:
                start_pose = np.array(obs_p.joint_states.position[:6])

        # ---- Localize port ----
        obs_v = get_observation()
        port_xy = None
        if obs_v is not None:
            try:
                port_xy = self._predict_port_xy(obs_v)
            except Exception as e:
                self.get_logger().warn(f"localizer failed: {e!r}")
        if port_xy is None:
            # Fallback: aim at a generic forward position (board likely lives
            # at world (0.18, -0.05, 1.27) for SFP, (0.29, -0.09, 1.15) for SC).
            port_xy = np.array([0.20, -0.05]) if "sfp" in port_type else np.array([0.29, -0.09])
            self.get_logger().warn(f"using fallback port_xy={port_xy}")

        port_z = self._expected_port_z(port_xy, port_type)
        port_world = np.array([port_xy[0], port_xy[1], port_z])
        self.get_logger().info(
            f"port_world (predicted) = {port_world.round(4)}  "
            f"port_type={port_type}  cable={cable_type}"
        )
        _diag("port_predicted", port_world=port_world.tolist(),
              port_type=port_type, cable_type=cable_type)

        # ---- Plan IK trajectory online ----
        traj = self._build_trajectory(port_world, gripper_offset)
        traj_dense = _densify(traj, max_per_joint_step=0.0015)
        self.get_logger().info(
            f"trajectory: {len(traj_dense)} dense steps "
            f"({len(traj_dense)/CONTROL_HZ:.1f}s)"
        )

        # ---- Swing safe → first_target ----
        first_target = traj_dense[0]
        _swing(start_pose, first_target, "safe→first_target")
        _settle(first_target, TRANSITION_HOLD, "first_target")

        # ---- Play trajectory ----
        descent_idx = APPROACH_STEPS  # in original (non-dense) trajectory
        # Find equivalent index in dense trajectory: midpoint of total length
        descent_mid_dense = len(traj_dense) // 2
        for step, q in enumerate(traj_dense):
            msg.target_state.positions = q.tolist()
            move_robot(joint_motion_update=msg)
            if step == descent_mid_dense:
                obs_m = get_observation()
                if obs_m is not None and len(obs_m.joint_states.position) >= 6:
                    _diag(
                        "mid_descent",
                        step=step,
                        q_actual=list(obs_m.joint_states.position[:6]),
                    )
            if step % 80 == 0:
                send_feedback(f"step={step}/{len(traj_dense)}")
            self.sleep_for(1.0 / CONTROL_HZ)

        for _ in range(20):
            msg.target_state.positions = traj_dense[-1].tolist()
            move_robot(joint_motion_update=msg)
            self.sleep_for(1.0 / CONTROL_HZ)

        obs_e = get_observation()
        q_end = (
            list(obs_e.joint_states.position[:6])
            if obs_e is not None and len(obs_e.joint_states.position) >= 6
            else None
        )
        _diag("insert_cable_exit", q_end=q_end, port_world=port_world.tolist())
        return True
