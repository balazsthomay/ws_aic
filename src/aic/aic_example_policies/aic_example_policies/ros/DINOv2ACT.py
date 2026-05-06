"""DINOv2 + ACT (Action Chunking Transformer) policy for SFP cable insertion.

Architecture:
  Vision: frozen DINOv2-S/14 → 384-d CLS token per camera (3 cameras).
  Policy: ACT head (transformer encoder + decoder) over (3 vision tokens,
          1 state token) → K=16 step action chunk.
  Control: execute one chunked action per 50 ms tick, re-encode vision every
           VISION_INTERVAL ticks. Same joint-impedance controller as
           DAggerInsert (high stiffness for position tracking).

Why this design:
  - Frozen DINOv2 features are renderer-agnostic, so they survive the
    MuJoCo→Gazebo render gap that breaks the green-pixel detector in
    DAggerInsert.
  - ACT chunking smooths over vision noise across multiple steps.
  - Loads ~22M-param DINOv2 + small ACT head; runs on CPU at >5 Hz.
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


# --- Diagnostic egress (same pattern as InstrumentedGentleGiant) ---

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


_diag("module_imported", policy="DINOv2ACT")


# --- Constants ---

WEIGHTS_DIR = Path(__file__).parent / "weights"
DINOV2_DIR = WEIGHTS_DIR / "dinov2-small"  # baked into Docker image
ACT_WEIGHTS = WEIGHTS_DIR / "act_policy.pt"
ACT_STATS = WEIGHTS_DIR / "act_norm_stats.npz"

CONTROL_HZ = 20
MAX_TIME = 30.0  # slow trajectory to match cluster's ~0.04 rad/s rate limit
TRANSITION_HOLD = 35.0  # cluster rate-limits ~0.04 rad/s; wrist swing needs ~20s
HOME_SETTLE_TOL = 0.05  # rad — exit hold early if max-joint-error <= this
TRANSITION_STIFFNESS = [500.0, 500.0, 500.0, 200.0, 200.0, 200.0]
TRANSITION_DAMPING   = [5.0,   5.0,   5.0,   2.0,   2.0,   2.0]   # low → swing fast
VISION_INTERVAL = 4     # re-encode vision every N control ticks (= 5 Hz)
CHUNK_SIZE = 16
N_CAMS = 3
PATCH_GRID = 4   # for "patches" mode: avg-pool 16×16 → 4×4 → 16 spatial tokens

# MuJoCo HOME — the joint pose every demo trajectory starts from. The cluster's
# default start pose is different (q[0]≈-0.13 vs MuJoCo's 0.6), so without a
# transition phase the policy never sees an in-distribution state at t=0.
MUJOCO_HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])

# DINOv2 input
DINOV2_INPUT = 224
DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# TCP-to-tip offset (matches DAggerInsert)
TIP_OFFSET_TCP_FRAME = np.array([-0.0018, -0.0189, 0.0547])

# Joint impedance (high stiffness for position tracking, low force-penalty risk
# requires precise targets — we trust the IK-cloned trajectory)
JOINT_STIFFNESS = [500.0, 500.0, 500.0, 200.0, 200.0, 200.0]
JOINT_DAMPING = [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
JOINT_LOWER = np.array([-2 * np.pi] * 6)
JOINT_UPPER = np.array([2 * np.pi] * 6)
MAX_JOINT_STEP = 0.05

# Port placeholder (port_pos slot in 26D state — not directly used by ACT, but
# matches training format)
NOMINAL_PORT_POS = np.array([0.220, -0.013, 1.273])


# --- ACT head (must match training architecture in scripts/train_dinov2_act.py) ---

class ACTHead(nn.Module):
    """Mirror of training-time ACTHead. Accepts CLS (n_spatial=1) or patches
    (n_spatial=PATCH_GRID*PATCH_GRID) inputs."""

    def __init__(
        self,
        vision_dim: int = 384,
        state_dim: int = 26,
        action_dim: int = 6,
        chunk_size: int = CHUNK_SIZE,
        n_cams: int = N_CAMS,
        n_spatial: int = 1,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_cams = n_cams
        self.n_spatial = n_spatial
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.input_pos = nn.Parameter(torch.randn(n_cams * n_spatial + 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.action_queries = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, vision_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # vision_tokens: (B, n_cams, D) or (B, n_cams, n_spatial, D)
        B = vision_tokens.shape[0]
        if vision_tokens.dim() == 3:
            vt = vision_tokens.unsqueeze(2)
        else:
            vt = vision_tokens
        vt = vt.reshape(B, -1, vt.shape[-1])
        v = self.vision_proj(vt)
        s = self.state_proj(state).unsqueeze(1)
        x = torch.cat([v, s], dim=1) + self.input_pos.unsqueeze(0)
        memory = self.encoder(x)
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)
        return self.action_head(out)


# --- Policy ---

class DINOv2ACT(Policy):
    """Vision-based insertion via frozen DINOv2 + ACT chunked actions."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        _diag("policy_init")
        self._device = torch.device("cpu")
        torch.set_num_threads(max(1, os.cpu_count() or 1))

        # Load frozen DINOv2 from local path (baked into image)
        from transformers import AutoModel

        self.get_logger().info(f"Loading DINOv2 from {DINOV2_DIR}")
        self._dinov2 = AutoModel.from_pretrained(str(DINOV2_DIR)).to(self._device).eval()
        for p in self._dinov2.parameters():
            p.requires_grad_(False)
        n_params = sum(p.numel() for p in self._dinov2.parameters()) / 1e6
        self.get_logger().info(f"DINOv2 loaded: {n_params:.1f}M params (frozen)")

        # Load ACT head — auto-detect mode (cls vs patches) from checkpoint
        payload = torch.load(str(ACT_WEIGHTS), map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            cfg = payload["config"]
            sd = payload["state_dict"]
        else:
            n_pos = payload["input_pos"].shape[0]
            n_spatial = max(1, (n_pos - 1) // N_CAMS)
            cfg = {"mode": "cls" if n_spatial == 1 else "patches",
                   "n_spatial": n_spatial}
            sd = payload
        self._mode = cfg.get("mode", "cls")
        self._n_spatial = cfg["n_spatial"]
        self._act = ACTHead(n_spatial=self._n_spatial).to(self._device).eval()
        self._act.load_state_dict(sd)
        self.get_logger().info(
            f"ACT loaded: mode={self._mode}, n_spatial={self._n_spatial}"
        )

        # Norm stats (action chunks were normalized at training time)
        stats = np.load(str(ACT_STATS))
        self._s_mean = torch.tensor(stats["state_mean"], dtype=torch.float32)
        self._s_std = torch.tensor(stats["state_std"], dtype=torch.float32)
        self._a_mean = torch.tensor(stats["action_mean"], dtype=torch.float32)
        self._a_std = torch.tensor(stats["action_std"], dtype=torch.float32)

        # Action-chunk buffer
        self._chunk: torch.Tensor | None = None
        self._chunk_idx = 0
        self._last_joint_target: np.ndarray | None = None
        _diag("policy_ready", n_params=n_params)

    # --- Main entry point ---

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        _diag(
            "insert_cable_enter",
            task_id=task.id,
            cable_type=task.cable_type,
            plug_type=task.plug_type,
            port_type=task.port_type,
            time_limit=task.time_limit,
        )
        self.get_logger().info(f"DINOv2ACT.insert_cable() task: {task}")
        send_feedback("dinov2-act policy")
        self._chunk = None
        self._chunk_idx = 0
        self._last_joint_target = None

        # ---- Transition phase: ease from current pose to MuJoCo HOME ----
        # The training data starts every trajectory at MuJoCo HOME; without
        # this the policy gets out-of-distribution states and freezes.
        start_pose = None
        for _ in range(20):
            obs0 = get_observation()
            if obs0 is not None and len(obs0.joint_states.position) >= 6:
                start_pose = np.array(obs0.joint_states.position[:6])
                break
            self.sleep_for(1.0 / CONTROL_HZ)
        if start_pose is None:
            self.get_logger().warn("No initial obs; assuming HOME")
            start_pose = MUJOCO_HOME.copy()
        else:
            self.get_logger().info(f"Start pose: {start_pose.round(3)} → HOME {MUJOCO_HOME.round(3)}")
        # No ramp — send HOME target immediately. Cosine interpolation makes
        # (target − current) tiny, so the spring-mass system applies tiny force
        # and the arm barely moves (v13 confirmed this). Just send HOME and let
        # it swing.
        # Hold at HOME until joints settle (or hold time expires)
        n_hold = int(TRANSITION_HOLD * CONTROL_HZ)
        for s in range(n_hold):
            self._send_joints(move_robot, MUJOCO_HOME,
                              stiffness=TRANSITION_STIFFNESS,
                              damping=TRANSITION_DAMPING)
            obs_h = get_observation()
            if obs_h is not None and len(obs_h.joint_states.position) >= 6:
                err = np.abs(np.array(obs_h.joint_states.position[:6]) - MUJOCO_HOME).max()
                if err <= HOME_SETTLE_TOL:
                    self.get_logger().info(f"Settled at HOME after {s}/{n_hold} (err={err:.3f})")
                    break
            self.sleep_for(1.0 / CONTROL_HZ)
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

            # Refresh action chunk every VISION_INTERVAL ticks (or when exhausted)
            if (
                self._chunk is None
                or self._chunk_idx >= self._chunk.shape[0]
                or step % VISION_INTERVAL == 0
            ):
                state = self._build_state(obs, elapsed)
                if state is None:
                    self.sleep_for(1.0 / CONTROL_HZ)
                    continue
                self._chunk = self._predict_chunk(obs, state)
                self._chunk_idx = 0

            action = self._chunk[self._chunk_idx].numpy()
            self._chunk_idx += 1

            # Safety: clamp + rate-limit
            action = np.clip(action, JOINT_LOWER[:6], JOINT_UPPER[:6])
            if self._last_joint_target is not None:
                delta = action - self._last_joint_target
                action = self._last_joint_target + np.clip(
                    delta, -MAX_JOINT_STEP, MAX_JOINT_STEP
                )
            self._last_joint_target = action.copy()

            self._send_joints(move_robot, action)

            if step % 40 == 0:
                send_feedback(f"t={elapsed:.1f}s step={step}")
                self.get_logger().info(f"t={elapsed:.1f}s q={action[:3]}")

            step += 1
            self.sleep_for(1.0 / CONTROL_HZ)

        # Hold final position
        self.get_logger().info("Holding final position...")
        for _ in range(20):
            if self._last_joint_target is not None:
                self._send_joints(move_robot, self._last_joint_target)
            self.sleep_for(0.05)

        _diag("insert_cable_exit", steps=step)
        return True

    # --- Chunk prediction ---

    def _predict_chunk(self, obs: Observation, state: np.ndarray) -> torch.Tensor:
        """Encode 3 wrist cameras + state → un-normalized K-step action chunk."""
        camera_images = [obs.left_image, obs.center_image, obs.right_image]
        tensors = []
        for img_msg in camera_images:
            img = _ros_image_to_numpy(img_msg)
            if img is None:
                # No image available → use zeros (DINOv2 will produce some token)
                img = np.zeros((DINOV2_INPUT, DINOV2_INPUT, 3), dtype=np.uint8)
            tensors.append(_preprocess_image(img))
        pixel_values = torch.cat(tensors, dim=0)  # (3, 3, 224, 224)

        with torch.no_grad():
            hs = self._dinov2(pixel_values=pixel_values).last_hidden_state
            if self._mode == "cls":
                feats = hs[:, 0, :]                      # (3, 384)
                vision_tokens = feats.unsqueeze(0)       # (1, 3, 384)
            else:
                # Patches: drop CLS, pool 16x16 → PATCH_GRID
                patches = hs[:, 1:, :]                   # (3, 256, 384)
                Bv, _, D = patches.shape
                grid = patches.reshape(Bv, 16, 16, D).permute(0, 3, 1, 2)
                pooled = F.adaptive_avg_pool2d(grid, PATCH_GRID)
                feats = pooled.permute(0, 2, 3, 1).reshape(
                    Bv, PATCH_GRID * PATCH_GRID, D
                )                                         # (3, 16, 384)
                vision_tokens = feats.unsqueeze(0)       # (1, 3, 16, 384)

            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state_norm = (state_t - self._s_mean) / self._s_std

            action_norm = self._act(vision_tokens, state_norm).squeeze(0)
            action = action_norm * self._a_std + self._a_mean

        return action.cpu()

    # --- State construction (matches DAggerInsert / collect_demos.py) ---

    def _build_state(self, obs: Observation, elapsed: float) -> np.ndarray | None:
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
        tcp_quat = np.array([q.w, q.x, q.y, q.z])  # ROS xyzw → MuJoCo wxyz

        tip_pos = tcp_pos + _quat_to_rot(tcp_quat) @ TIP_OFFSET_TCP_FRAME

        # Port slot in state vector — model learns to ignore via vision
        port_pos = NOMINAL_PORT_POS

        progress = min(elapsed / MAX_TIME, 1.0)

        return np.concatenate([
            joint_pos, joint_vel, tcp_pos, tcp_quat,
            tip_pos, port_pos, [progress],
        ])

    # --- Joint command (same as DAggerInsert) ---

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


# --- Image preprocessing ---

def _preprocess_image(img: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H, W, 3) → DINOv2 input (1, 3, 224, 224) float."""
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
