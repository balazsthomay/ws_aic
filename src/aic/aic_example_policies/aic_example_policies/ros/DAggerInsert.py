"""Vision-based DAgger MLP policy for SFP cable insertion.

Two-stage pipeline:
  1. Perception: Green NIC card detection → geometric back-projection → port 3D position
  2. Control: DAgger-trained MLP (26D state → 6D joint targets)

Usage:
  ros2 run aic_model aic_model --ros-args -p policy:=aic_example_policies.ros.DAggerInsert
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from tf2_ros import TransformException


# --- Neural network (must match training architecture) ---

class MLPPolicy(nn.Module):
    def __init__(self, state_dim: int = 26, action_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# --- Constants ---

WEIGHTS_DIR = Path(__file__).parent / "weights"
CONTROL_HZ = 20
MAX_TIME = 14.0

# TCP-to-tip offset in TCP frame (measured from MuJoCo at home keyframe)
TIP_OFFSET_TCP_FRAME = np.array([-0.0018, -0.0189, 0.0547])

# Nominal port position (fallback when vision fails)
NOMINAL_PORT_POS = np.array([0.220, -0.013, 1.273])
PORT_Z = 1.2735

# NIC card centroid → port offset (calibrated from policy rollouts)
NIC_TO_PORT_OFFSET = np.array([-0.0095, 0.0065])

# Camera names matching the Observation message fields
CAMERA_FIELDS = ["left", "center", "right"]
CAMERA_TF_FRAMES = [
    "left_camera/optical_frame",
    "center_camera/optical_frame",
    "right_camera/optical_frame",
]

# Joint-space impedance gains (high stiffness for position tracking)
JOINT_STIFFNESS = [500.0, 500.0, 500.0, 200.0, 200.0, 200.0]
JOINT_DAMPING = [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]

# Joint limits (UR5e, radians)
JOINT_LOWER = np.array([-2 * np.pi] * 6)
JOINT_UPPER = np.array([2 * np.pi] * 6)
MAX_JOINT_STEP = 0.05  # rad per control step (1 rad/s at 20Hz)


# --- Policy ---

class DAggerInsert(Policy):
    """Vision-based cable insertion using DAgger-trained MLP."""

    def __init__(self, parent_node):
        super().__init__(parent_node)

        # Load MLP policy
        self._device = torch.device("cpu")
        self._model = MLPPolicy()
        self._model.load_state_dict(torch.load(
            WEIGHTS_DIR / "mlp_policy_best.pt",
            map_location="cpu", weights_only=True,
        ))
        self._model.eval()
        self.get_logger().info(f"Loaded MLP policy from {WEIGHTS_DIR}")

        # Load normalization stats
        stats = np.load(str(WEIGHTS_DIR / "norm_stats.npz"))
        self._s_mean = torch.tensor(stats["state_mean"], dtype=torch.float32)
        self._s_std = torch.tensor(stats["state_std"], dtype=torch.float32)
        self._a_mean = torch.tensor(stats["action_mean"], dtype=torch.float32)
        self._a_std = torch.tensor(stats["action_std"], dtype=torch.float32)
        self.get_logger().info(f"State dim: {self._s_mean.shape[0]}, Action dim: {self._a_mean.shape[0]}")

        # Port prediction state
        self._port_history: list[np.ndarray] = []
        self._last_joint_target: np.ndarray | None = None

    # --- Main entry point ---

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"DAggerInsert.insert_cable() task: {task}")
        self._port_history = []
        self._last_joint_target = None

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

            # Build state vector
            state = self._build_state(obs, elapsed)
            if state is None:
                self.sleep_for(1.0 / CONTROL_HZ)
                continue

            # MLP inference
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state_norm = (state_t - self._s_mean) / self._s_std

            with torch.no_grad():
                action_norm = self._model(state_norm).squeeze(0)
                action = (action_norm * self._a_std + self._a_mean).numpy()

            # Safety: clamp and rate-limit joint targets
            action = np.clip(action, JOINT_LOWER[:6], JOINT_UPPER[:6])
            if self._last_joint_target is not None:
                delta = action - self._last_joint_target
                action = self._last_joint_target + np.clip(delta, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            self._last_joint_target = action.copy()

            # Send joint command
            self._send_joints(move_robot, action)

            if step % 40 == 0:
                send_feedback(f"t={elapsed:.1f}s step={step}")
                self.get_logger().info(f"t={elapsed:.1f}s q={action[:3]}")

            step += 1
            self.sleep_for(1.0 / CONTROL_HZ)

        # Hold final position briefly
        self.get_logger().info("Holding final position...")
        for _ in range(20):
            if self._last_joint_target is not None:
                self._send_joints(move_robot, self._last_joint_target)
            self.sleep_for(0.05)

        self.get_logger().info("DAggerInsert.insert_cable() exiting")
        return True

    # --- State construction ---

    def _build_state(self, obs: Observation, elapsed: float) -> np.ndarray | None:
        """Convert ROS Observation → 26D state vector matching MuJoCo training format."""
        js = obs.joint_states
        cs = obs.controller_state

        if len(js.position) < 6 or len(js.velocity) < 6:
            return None

        # Joint state
        joint_pos = np.array(js.position[:6])
        joint_vel = np.array(js.velocity[:6])

        # TCP pose (reorder quaternion: ROS xyzw → MuJoCo wxyz)
        tcp_pos = np.array([
            cs.tcp_pose.position.x,
            cs.tcp_pose.position.y,
            cs.tcp_pose.position.z,
        ])
        q = cs.tcp_pose.orientation
        tcp_quat = np.array([q.w, q.x, q.y, q.z])  # wxyz

        # Tip position: TCP + rotated offset
        tip_pos = self._compute_tip_pos(tcp_pos, tcp_quat)

        # Port position: vision or fallback
        port_pos = self._predict_port(obs)

        # Progress
        progress = min(elapsed / MAX_TIME, 1.0)

        return np.concatenate([
            joint_pos, joint_vel, tcp_pos, tcp_quat,
            tip_pos, port_pos, [progress],
        ])

    def _compute_tip_pos(self, tcp_pos: np.ndarray, tcp_quat_wxyz: np.ndarray) -> np.ndarray:
        """Compute plug tip position from TCP pose + known offset."""
        R = _quat_to_rot(tcp_quat_wxyz)
        return tcp_pos + R @ TIP_OFFSET_TCP_FRAME

    # --- Vision-based port localization ---

    def _predict_port(self, obs: Observation) -> np.ndarray:
        """Detect green NIC card in cameras, back-project to 3D port position."""
        points = []
        weights = []

        camera_images = [obs.left_image, obs.center_image, obs.right_image]
        camera_infos = [obs.left_camera_info, obs.center_camera_info, obs.right_camera_info]

        for img_msg, info_msg, tf_frame in zip(camera_images, camera_infos, CAMERA_TF_FRAMES):
            # Convert ROS Image to numpy
            img = _ros_image_to_numpy(img_msg)
            if img is None:
                continue

            # Detect green pixels
            mask = (img[:, :, 1] > 100) & (img[:, :, 0] < 100) & (img[:, :, 2] < 100)
            n_pixels = mask.sum()
            if n_pixels < 20:
                continue

            ys, xs = np.where(mask)
            cu, cv = xs.mean(), ys.mean()

            # Camera intrinsics from CameraInfo K matrix
            K = np.array(info_msg.k).reshape(3, 3)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            if fx < 1 or fy < 1:
                continue

            # Camera extrinsics from TF
            try:
                tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", tf_frame, Time(),
                )
            except TransformException:
                continue

            cam_pos, cam_rot = _tf_to_pos_rot(tf_stamped.transform)

            # Ray in camera optical frame (Z forward, X right, Y down)
            ray_cam = np.array([
                (cu - cx) / fx,
                (cv - cy) / fy,
                1.0,
            ])

            # Transform to world frame
            ray_world = cam_rot @ ray_cam
            ray_world /= np.linalg.norm(ray_world)

            # Intersect with z = PORT_Z plane
            if abs(ray_world[2]) < 1e-6:
                continue
            t = (PORT_Z - cam_pos[2]) / ray_world[2]
            if t < 0:
                continue

            pt = cam_pos + t * ray_world
            points.append(pt[:2])
            weights.append(min(n_pixels / 500.0, 1.0))

        if not points:
            if self._port_history:
                med = np.median(self._port_history, axis=0)
                return np.array([med[0], med[1], PORT_Z])
            return NOMINAL_PORT_POS.copy()

        # Weighted average + offset correction
        pts = np.array(points)
        wts = np.array(weights)
        wts /= wts.sum()
        avg_xy = (pts * wts[:, None]).sum(axis=0) + NIC_TO_PORT_OFFSET
        self._port_history.append(avg_xy.copy())

        # Running median
        med = np.median(self._port_history, axis=0)
        return np.array([med[0], med[1], PORT_Z])

    # --- Joint command ---

    def _send_joints(self, move_robot: MoveRobotCallback, q_target: np.ndarray) -> None:
        """Send joint position target via impedance controller."""
        msg = JointMotionUpdate(
            target_stiffness=JOINT_STIFFNESS,
            target_damping=JOINT_DAMPING,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        msg.target_state.positions = q_target.tolist()
        move_robot(joint_motion_update=msg)


# --- Utility functions ---

def _quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def _ros_image_to_numpy(img_msg) -> np.ndarray | None:
    """Convert sensor_msgs/Image to (H, W, 3) uint8 numpy array."""
    if img_msg.height == 0 or img_msg.width == 0:
        return None

    try:
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, -1
        )
    except ValueError:
        return None

    # Handle BGR (common in ROS) vs RGB
    if img_msg.encoding == "bgr8":
        img = img[:, :, ::-1].copy()
    elif img_msg.encoding == "rgb8":
        pass
    elif img_msg.encoding == "rgba8":
        img = img[:, :, :3]
    elif img_msg.encoding == "bgra8":
        img = img[:, :, 2::-1].copy()
    else:
        # Assume RGB-like
        img = img[:, :, :3]

    return img


def _tf_to_pos_rot(transform) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and rotation matrix from geometry_msgs/Transform."""
    t = transform.translation
    pos = np.array([t.x, t.y, t.z])

    q = transform.rotation
    rot = _quat_to_rot(np.array([q.w, q.x, q.y, q.z]))

    return pos, rot
