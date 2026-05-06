"""Deterministic blind playback of an IK trajectory at the NOMINAL port.

No vision, no learning. Useful as a baseline / safety net: even with the
board randomized ±15 mm + ±8.6° yaw, a trajectory aimed at the nominal port
should land in the proximity-credit window (max ≈ 0.5 × initial_dist).
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


_diag("module_imported", policy="ScriptedPlay")


WEIGHTS_DIR = Path(__file__).parent / "weights"
TRAJ_PATH = WEIGHTS_DIR / "scripted_traj.npz"

CONTROL_HZ = 20
TRANSITION_HOLD = 35.0   # cluster rate-limits ~0.04 rad/s; wrist swing needs ~20s
HOME_SETTLE_TOL = 0.05
JOINT_STIFFNESS = [500.0, 500.0, 500.0, 200.0, 200.0, 200.0]
JOINT_DAMPING = [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
TRANSITION_DAMPING = [5.0, 5.0, 5.0, 2.0, 2.0, 2.0]   # low → swing fast


class ScriptedPlay(Policy):
    """Replay pre-computed joint trajectory targeting the nominal port."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        _diag("policy_init")
        data = np.load(str(TRAJ_PATH))
        self._traj = data["joints"].astype(np.float64)
        self.get_logger().info(
            f"ScriptedPlay loaded {len(self._traj)} steps from {TRAJ_PATH}"
        )

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
        send_feedback("scripted blind playback")

        msg = JointMotionUpdate(
            target_stiffness=JOINT_STIFFNESS,
            target_damping=JOINT_DAMPING,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        # Separate msg for transition phase with low damping → wrist swings fast
        trans_msg = JointMotionUpdate(
            target_stiffness=JOINT_STIFFNESS,
            target_damping=TRANSITION_DAMPING,
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )

        # Transition: smoothly ease from current pose to the trajectory's first
        # joint configuration (= MuJoCo HOME). Without this, the controller
        # sees a large position step on tick 0 and triggers force penalties.
        start_pose = None
        for _ in range(20):
            obs = get_observation()
            if obs is not None and len(obs.joint_states.position) >= 6:
                start_pose = np.array(obs.joint_states.position[:6])
                break
            self.sleep_for(1.0 / CONTROL_HZ)
        first_target = self._traj[0]
        if start_pose is None:
            self.get_logger().warn("No initial obs; skipping transition")
            start_pose = first_target.copy()
        else:
            self.get_logger().info(f"Transition {start_pose.round(3)} → {first_target.round(3)}")
        # No ramp — send first_target immediately, let low-damping spring swing
        # the arm there. Cosine ramp made the spring barely deflect (v13).
        # Hold at first_target until joints settle (or timeout)
        n_hold = int(TRANSITION_HOLD * CONTROL_HZ)
        for s in range(n_hold):
            trans_msg.target_state.positions = first_target.tolist()
            move_robot(joint_motion_update=trans_msg)
            obs_h = get_observation()
            if obs_h is not None and len(obs_h.joint_states.position) >= 6:
                err = np.abs(np.array(obs_h.joint_states.position[:6]) - first_target).max()
                if err <= HOME_SETTLE_TOL:
                    self.get_logger().info(f"Settled after {s}/{n_hold} (err={err:.3f})")
                    break
            self.sleep_for(1.0 / CONTROL_HZ)

        for step, q in enumerate(self._traj):
            msg.target_state.positions = q.tolist()
            move_robot(joint_motion_update=msg)
            if step % 40 == 0:
                send_feedback(f"step={step}")
            self.sleep_for(1.0 / CONTROL_HZ)

        # Hold final pose briefly
        for _ in range(20):
            msg.target_state.positions = self._traj[-1].tolist()
            move_robot(joint_motion_update=msg)
            self.sleep_for(1.0 / CONTROL_HZ)

        _diag("insert_cable_exit")
        return True
