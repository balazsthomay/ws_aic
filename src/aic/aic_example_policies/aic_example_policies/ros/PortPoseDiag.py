"""Diagnostic policy: print Gazebo's actual port pose (from TF) alongside the
v26 localizer's predicted port_xy, plus the MuJoCo-replayed value if the
container has the scene.

Use only with `ground_truth:=true` so /scoring/tf publishes port frames.

Output: one log line + one diag webhook POST per trial:
  port_actual = (x, y, z)        # Gazebo via TF
  port_predicted = (x, y)        # v26 localizer
  delta_xy_pred = (dx, dy)       # localizer error in xy
"""
import json
import os
import socket
import time
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
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException


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


WEIGHTS_DIR = Path(__file__).parent / "weights"
DINOV2_DIR = WEIGHTS_DIR / "dinov2-small"
LOCALIZER_PT = WEIGHTS_DIR / "port_localizer_dinov2.pt"
LOCALIZER_STATS = WEIGHTS_DIR / "port_localizer_stats.npz"

CONTROL_HZ = 20
SAFE_POSE = np.array([-0.16, -1.35, -1.66, -1.69, 1.57, 1.41])
SAFE_POSE_SKIP_TOL = 0.10
TRANSITION_STIFFNESS = [200.0, 200.0, 200.0, 100.0, 100.0, 100.0]
TRANSITION_DAMPING = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]

DINOV2_INPUT = 224
PATCH_GRID = 4
N_CAMS = 3
DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _PortLocalizer(nn.Module):
    def __init__(self, vision_dim=384, n_tokens=N_CAMS * PATCH_GRID * PATCH_GRID,
                 d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.proj = nn.Linear(vision_dim, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        B, C, S, D = x.shape
        x = x.reshape(B, C * S, D)
        x = self.proj(x) + self.pos.unsqueeze(0)
        x = self.encoder(x).mean(dim=1)
        return self.head(x)


def _preprocess(img):
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(DINOV2_INPUT, DINOV2_INPUT), mode="bilinear", align_corners=False)
    return (t - DINOV2_MEAN) / DINOV2_STD


def _ros_image_to_numpy(img_msg):
    if img_msg.height == 0 or img_msg.width == 0:
        return None
    try:
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, -1
        )
    except Exception:
        return None
    return img[..., :3] if img.shape[2] >= 3 else img


class PortPoseDiag(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        from transformers import AutoModel
        self._device = torch.device("cpu")
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
        self._loc = _PortLocalizer(vision_dim=self._vision_dim).to(self._device).eval()
        self._loc.load_state_dict(torch.load(
            str(LOCALIZER_PT), map_location="cpu", weights_only=True
        ))
        stats = np.load(str(LOCALIZER_STATS))
        self._port_mean = torch.tensor(stats["port_mean"], dtype=torch.float32)
        self._port_std = torch.tensor(stats["port_std"], dtype=torch.float32)
        self.get_logger().info("PortPoseDiag ready")
        _diag("portposediag_init")

    def _wait_for_tf(self, target_frame, source_frame, timeout_sec=15.0):
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame, source_frame, Time()
                )
                return True
            except TransformException:
                self.sleep_for(0.1)
        return False

    def _predict_port_xy(self, obs):
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

    def _swing_to_safe(self, get_observation, move_robot):
        trans_msg = JointMotionUpdate(
            target_stiffness=TRANSITION_STIFFNESS,
            target_damping=TRANSITION_DAMPING,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        # Read start
        start_pose = None
        for _ in range(20):
            obs = get_observation()
            if obs is not None and len(obs.joint_states.position) >= 6:
                start_pose = np.array(obs.joint_states.position[:6])
                break
            self.sleep_for(1.0 / CONTROL_HZ)
        if start_pose is None:
            start_pose = SAFE_POSE.copy()
        if np.abs(start_pose - SAFE_POSE).max() > SAFE_POSE_SKIP_TOL:
            # Densified swing
            n = max(1, int(np.ceil(np.abs(SAFE_POSE - start_pose).max() / 0.0015)))
            for k in range(1, n + 1):
                q = start_pose + (k / n) * (SAFE_POSE - start_pose)
                trans_msg.target_state.positions = q.tolist()
                move_robot(joint_motion_update=trans_msg)
                self.sleep_for(1.0 / CONTROL_HZ)
        # Settle
        for _ in range(int(4.0 * CONTROL_HZ)):
            trans_msg.target_state.positions = SAFE_POSE.tolist()
            move_robot(joint_motion_update=trans_msg)
            self.sleep_for(1.0 / CONTROL_HZ)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        info = {
            "task_id": task.id,
            "cable_type": task.cable_type,
            "port_type": task.port_type,
            "target_module_name": task.target_module_name,
            "port_name": task.port_name,
            "port_frame": port_frame,
        }
        self.get_logger().info(f"[diag] task: {info}")

        # Park at SAFE_POSE so the localizer sees the same view as v26.
        self._swing_to_safe(get_observation, move_robot)

        # Dump the TF tree so we can map the topology.
        try:
            tree_yaml = self._parent_node._tf_buffer.all_frames_as_yaml()
            self.get_logger().info(f"[diag] TF tree:\n{tree_yaml}")
        except Exception as e:
            self.get_logger().warn(f"[diag] failed to dump TF tree: {e!r}")

        plug_frame = f"{task.cable_name}/{task.plug_name}_link"
        # Try multiple lookup targets to figure out which frame the port is in,
        # and where the plug actually sits relative to the gripper TCP.
        candidates = [
            ("base_link", port_frame),
            ("world", port_frame),
            ("base_link", plug_frame),
            ("world", plug_frame),
            ("gripper/tcp", plug_frame),         # ← the smoking gun: plug-in-gripper offset
            ("base_link", "gripper/tcp"),
            ("world", "gripper/tcp"),
        ]
        lookups = {}
        for target, source in candidates:
            try:
                tf = self._parent_node._tf_buffer.lookup_transform(target, source, Time())
                t = tf.transform.translation
                lookups[f"{target}_to_{source}"] = [t.x, t.y, t.z]
                self.get_logger().info(
                    f"[diag] {target} → {source} = ({t.x:.4f}, {t.y:.4f}, {t.z:.4f})"
                )
            except TransformException as e:
                lookups[f"{target}_to_{source}"] = f"FAIL: {e!r}"
                self.get_logger().info(f"[diag] {target} → {source} = FAIL: {e!r}")

        # Use base_link → port as the port_actual reference (might be wrong).
        port_actual_key = f"base_link_to_{port_frame}"
        port_actual = lookups.get(port_actual_key)
        if isinstance(port_actual, str) or port_actual is None:
            port_actual = np.array([np.nan, np.nan, np.nan])
        else:
            port_actual = np.array(port_actual)

        # Run localizer.
        obs = get_observation()
        port_pred = None
        if obs is not None:
            try:
                port_pred = self._predict_port_xy(obs)
            except Exception as e:
                self.get_logger().error(f"[diag] localizer failed: {e!r}")

        delta = None
        if port_pred is not None:
            delta = port_actual[:2] - np.array(port_pred[:2])
        self.get_logger().info(
            f"[diag] port_actual_xyz = {port_actual.round(4).tolist()}  "
            f"port_pred_xy = {None if port_pred is None else np.array(port_pred).round(4).tolist()}  "
            f"delta_xy = {None if delta is None else delta.round(4).tolist()}"
        )
        _diag(
            "port_pose_compare",
            port_actual=port_actual.tolist(),
            port_pred=None if port_pred is None else np.array(port_pred).tolist(),
            delta_xy=None if delta is None else delta.tolist(),
            tf_lookups=lookups,
            **info,
        )
        return True
