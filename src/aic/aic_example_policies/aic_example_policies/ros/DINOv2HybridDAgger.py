"""Hybrid policy: DINOv2 → port localizer → DAgger MLP.

Cleaner separation than DINOv2ACT:
  1. Frozen DINOv2 encodes 3 wrist images → patch tokens
  2. PortLocalizer head predicts (port_x, port_y) in world coords
  3. Predicted port is substituted into the 26D state vector
  4. The existing DAgger MLP (which gets ~90% with GT port in MuJoCo) acts on it

Why this might beat DINOv2ACT:
  - The MLP head is small and well-trained on a clean dataset
  - The localizer can be evaluated independently
  - Easier to iterate on either component separately
"""

import json
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
from aic_model_interfaces.msg import Observation
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


_diag("module_imported", policy="DINOv2HybridDAgger")


WEIGHTS_DIR = Path(__file__).parent / "weights"
DINOV2_DIR = WEIGHTS_DIR / "dinov2-small"
LOCALIZER_PT = WEIGHTS_DIR / "port_localizer_dinov2.pt"
LOCALIZER_STATS = WEIGHTS_DIR / "port_localizer_stats.npz"
MLP_PT = WEIGHTS_DIR / "mlp_policy_best.pt"
MLP_STATS = WEIGHTS_DIR / "norm_stats.npz"

CONTROL_HZ = 20
MAX_TIME = 30.0                # slow the trajectory to match cluster's ~0.04 rad/s
                               # joint rate limit; trained trajectory was 14s
VISION_INTERVAL = 4
TRANSITION_HOLD = 35.0         # cluster rate-limits ~0.04 rad/s; wrist swing needs ~20s
HOME_SETTLE_TOL = 0.05         # rad — exit hold early if max-joint-error <= this
# Low damping during transition so the wrist (q[5]) actually swings.
# v12 saw err=0.81 stuck because damping=15 ate the small wrist torque
# budget. Matches SpeedDemon's gains, which DO swing the arm fast.
TRANSITION_STIFFNESS = [500.0, 500.0, 500.0, 200.0, 200.0, 200.0]
TRANSITION_DAMPING   = [5.0,   5.0,   5.0,   2.0,   2.0,   2.0]
N_CAMS = 3
PATCH_GRID = 4
DINOV2_INPUT = 224
DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

TIP_OFFSET_TCP_FRAME = np.array([-0.0018, -0.0189, 0.0547])
NOMINAL_PORT_POS = np.array([0.220, -0.013, 1.273])
PORT_Z = 1.2735

JOINT_STIFFNESS = [500.0, 500.0, 500.0, 200.0, 200.0, 200.0]
JOINT_DAMPING = [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
# MuJoCo HOME — every demo trajectory starts here. Policy is OOD elsewhere.
MUJOCO_HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])
JOINT_LOWER = np.array([-2 * np.pi] * 6)
JOINT_UPPER = np.array([2 * np.pi] * 6)
MAX_JOINT_STEP = 0.05


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
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)


class MLPPolicy(nn.Module):
    """Mirror of DAgger MLP architecture (same as DAggerInsert)."""
    def __init__(self, state_dim: int = 26, action_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DINOv2HybridDAgger(Policy):
    """DINOv2 → port localizer → DAgger MLP."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        _diag("policy_init")
        self._device = torch.device("cpu")
        torch.set_num_threads(max(1, os.cpu_count() or 1))

        from transformers import AutoModel
        self.get_logger().info(f"Loading DINOv2 from {DINOV2_DIR}")
        self._dinov2 = AutoModel.from_pretrained(str(DINOV2_DIR)).to(self._device).eval()
        for p in self._dinov2.parameters():
            p.requires_grad_(False)

        self._loc = PortLocalizer().to(self._device).eval()
        self._loc.load_state_dict(torch.load(
            str(LOCALIZER_PT), map_location="cpu", weights_only=True
        ))
        loc_stats = np.load(str(LOCALIZER_STATS))
        self._port_mean = torch.tensor(loc_stats["port_mean"], dtype=torch.float32)
        self._port_std = torch.tensor(loc_stats["port_std"], dtype=torch.float32)

        self._mlp = MLPPolicy().to(self._device).eval()
        self._mlp.load_state_dict(torch.load(
            str(MLP_PT), map_location="cpu", weights_only=True
        ))
        mlp_stats = np.load(str(MLP_STATS))
        self._s_mean = torch.tensor(mlp_stats["state_mean"], dtype=torch.float32)
        self._s_std = torch.tensor(mlp_stats["state_std"], dtype=torch.float32)
        self._a_mean = torch.tensor(mlp_stats["action_mean"], dtype=torch.float32)
        self._a_std = torch.tensor(mlp_stats["action_std"], dtype=torch.float32)
        self.get_logger().info("DINOv2 + Localizer + MLP loaded")
        _diag("policy_ready")

        self._cached_port: np.ndarray | None = None
        self._port_history: list[np.ndarray] = []
        self._last_joint_target: np.ndarray | None = None

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        _diag("insert_cable_enter", task_id=task.id)
        send_feedback("dinov2 hybrid policy")
        self._cached_port = None
        self._port_history = []
        self._last_joint_target = None

        # ---- Transition + settle: actually reach MuJoCo HOME before policy ----
        # The cluster spawns the arm well away from MuJoCo HOME (~0.7 rad on
        # shoulder pan). With stiffness 500 the impedance controller takes
        # >4 s to swing that far, so a fixed 4 s ramp leaves the arm short and
        # the policy starts on out-of-distribution joint state.
        start_pose = None
        for _ in range(20):
            obs0 = get_observation()
            if obs0 is not None and len(obs0.joint_states.position) >= 6:
                start_pose = np.array(obs0.joint_states.position[:6])
                break
            self.sleep_for(1.0 / CONTROL_HZ)
        if start_pose is None:
            start_pose = MUJOCO_HOME.copy()
        else:
            self.get_logger().info(
                f"Start: {start_pose.round(3)} → HOME {MUJOCO_HOME.round(3)}"
            )
        # NO ramp — send HOME target immediately with SpeedDemon-style low damping
        # so the spring-mass system actually swings the arm. A cosine ramp keeps
        # (target − current) tiny, which means tiny force → tiny motion. v13
        # learned that the hard way: 20s ramp barely moved the wrist.
        # This is essentially how SpeedDemon (v7) drives the arm — accept the
        # force-penalty risk to actually reach HOME.
        # Hold + verify the joints actually settled at HOME
        n_hold = int(TRANSITION_HOLD * CONTROL_HZ)
        settled = False
        for s in range(n_hold):
            self._send_joints(move_robot, MUJOCO_HOME,
                              stiffness=TRANSITION_STIFFNESS,
                              damping=TRANSITION_DAMPING)
            obs_h = get_observation()
            if obs_h is not None and len(obs_h.joint_states.position) >= 6:
                err = np.abs(np.array(obs_h.joint_states.position[:6]) - MUJOCO_HOME).max()
                if err <= HOME_SETTLE_TOL:
                    self.get_logger().info(f"Settled at HOME after {s}/{n_hold} ticks (err={err:.3f})")
                    settled = True
                    break
            self.sleep_for(1.0 / CONTROL_HZ)
        if not settled:
            obs_h = get_observation()
            if obs_h is not None and len(obs_h.joint_states.position) >= 6:
                err = np.abs(np.array(obs_h.joint_states.position[:6]) - MUJOCO_HOME).max()
                self.get_logger().warn(f"Did NOT settle (err={err:.3f}); proceeding anyway")
        self._last_joint_target = MUJOCO_HOME.copy()

        start_time = self.time_now()
        step = 0

        while True:
            elapsed = (self.time_now() - start_time).nanoseconds / 1e9
            if elapsed >= MAX_TIME:
                break

            obs = get_observation()
            if obs is None:
                self.sleep_for(1.0 / CONTROL_HZ)
                continue

            # Refresh port estimate every VISION_INTERVAL ticks
            if step % VISION_INTERVAL == 0 or self._cached_port is None:
                port_xy = self._predict_port(obs)
                self._port_history.append(port_xy)
                # Running median for stability
                hist = np.array(self._port_history[-20:])
                self._cached_port = np.median(hist, axis=0)

            port_pos = np.array([self._cached_port[0], self._cached_port[1], PORT_Z])

            state = self._build_state(obs, elapsed, port_pos)
            if state is None:
                self.sleep_for(1.0 / CONTROL_HZ)
                continue

            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state_norm = (state_t - self._s_mean) / self._s_std
            with torch.no_grad():
                action_norm = self._mlp(state_norm).squeeze(0)
                action = (action_norm * self._a_std + self._a_mean).numpy()

            action = np.clip(action, JOINT_LOWER[:6], JOINT_UPPER[:6])
            if self._last_joint_target is not None:
                delta = action - self._last_joint_target
                action = self._last_joint_target + np.clip(
                    delta, -MAX_JOINT_STEP, MAX_JOINT_STEP
                )
            self._last_joint_target = action.copy()

            self._send_joints(move_robot, action)

            if step % 40 == 0:
                send_feedback(f"t={elapsed:.1f}s port=({port_pos[0]:.3f},{port_pos[1]:.3f})")
                self.get_logger().info(
                    f"t={elapsed:.1f}s port={port_pos[:2].round(4)} q={action[:3].round(3)}"
                )

            step += 1
            self.sleep_for(1.0 / CONTROL_HZ)

        for _ in range(20):
            if self._last_joint_target is not None:
                self._send_joints(move_robot, self._last_joint_target)
            self.sleep_for(0.05)

        _diag("insert_cable_exit", steps=step)
        return True

    def _predict_port(self, obs: Observation) -> np.ndarray:
        camera_images = [obs.left_image, obs.center_image, obs.right_image]
        tensors = []
        for img_msg in camera_images:
            img = _ros_image_to_numpy(img_msg)
            if img is None:
                img = np.zeros((DINOV2_INPUT, DINOV2_INPUT, 3), dtype=np.uint8)
            tensors.append(_preprocess_image(img))
        pixel_values = torch.cat(tensors, dim=0)

        with torch.no_grad():
            hs = self._dinov2(pixel_values=pixel_values).last_hidden_state
            patches = hs[:, 1:, :]                # (3, 256, 384)
            grid = patches.reshape(3, 16, 16, 384).permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(grid, PATCH_GRID)
            feats = pooled.permute(0, 2, 3, 1).reshape(
                3, PATCH_GRID * PATCH_GRID, 384
            ).unsqueeze(0)                         # (1, 3, 16, 384)

            port_norm = self._loc(feats).squeeze(0)
            port_xy = (port_norm * self._port_std + self._port_mean).cpu().numpy()
        return port_xy

    def _build_state(self, obs: Observation, elapsed: float,
                     port_pos: np.ndarray) -> np.ndarray | None:
        js = obs.joint_states
        cs = obs.controller_state
        if len(js.position) < 6 or len(js.velocity) < 6:
            return None

        joint_pos = np.array(js.position[:6])
        joint_vel = np.array(js.velocity[:6])
        tcp_pos = np.array([
            cs.tcp_pose.position.x, cs.tcp_pose.position.y, cs.tcp_pose.position.z,
        ])
        q = cs.tcp_pose.orientation
        tcp_quat = np.array([q.w, q.x, q.y, q.z])
        tip_pos = tcp_pos + _quat_to_rot(tcp_quat) @ TIP_OFFSET_TCP_FRAME
        progress = min(elapsed / MAX_TIME, 1.0)
        return np.concatenate([
            joint_pos, joint_vel, tcp_pos, tcp_quat,
            tip_pos, port_pos, [progress],
        ])

    def _send_joints(self, move_robot: MoveRobotCallback, q_target: np.ndarray,
                     stiffness: list | None = None,
                     damping: list | None = None) -> None:
        msg = JointMotionUpdate(
            target_stiffness=stiffness if stiffness is not None else JOINT_STIFFNESS,
            target_damping=damping if damping is not None else JOINT_DAMPING,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        msg.target_state.positions = q_target.tolist()
        move_robot(joint_motion_update=msg)


def _preprocess_image(img: np.ndarray) -> torch.Tensor:
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
    except ValueError:
        return None
    if img_msg.encoding == "bgr8":
        img = img[:, :, ::-1].copy()
    elif img_msg.encoding == "rgb8":
        pass
    elif img_msg.encoding == "rgba8":
        img = img[:, :, :3]
    elif img_msg.encoding == "bgra8":
        img = img[:, :, 2::-1].copy()
    else:
        img = img[:, :, :3]
    return img


def _quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])
