"""SpeedDemon clone instrumented with HTTP egress to AIC_DIAG_URL.

POSTs at every lifecycle/policy stage so we know exactly where the
container's flow stops on the cluster (vs locally).
"""

import json
import os
import socket
import time
import urllib.error
import urllib.request

from rclpy.duration import Duration

from aic_control_interfaces.msg import JointMotionUpdate
from aic_control_interfaces.msg import TrajectoryGenerationMode
from aic_model.policy import (
    Policy,
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


_DIAG_URL = os.environ.get("AIC_DIAG_URL", "")
_HOST = socket.gethostname()


def _diag(event: str, **extra) -> None:
    """Best-effort POST. Never raise."""
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


_diag("module_imported")


class InstrumentedSpeedDemon(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        _diag("policy_init")
        self.get_logger().info("InstrumentedSpeedDemon.__init__()")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        _diag(
            "insert_cable_enter",
            task_id=task.id,
            cable_type=task.cable_type,
            plug_type=task.plug_type,
            port_type=task.port_type,
            time_limit=task.time_limit,
        )
        self.get_logger().info(f"InstrumentedSpeedDemon.insert_cable() task: {task}")
        send_feedback("instrumented diagnostic policy")

        joint_motion_update = JointMotionUpdate(
            target_stiffness=[500.0, 500.0, 500.0, 200.0, 200.0, 200.0],
            target_damping=[5.0, 5.0, 5.0, 2.0, 2.0, 2.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )

        home = [-0.16, -1.35, -1.66, -1.69, 1.57, 1.41]
        target = [0.6, -1.3, -1.9, -1.57, 1.57, 0.6]

        for cycle in range(3):
            _diag("insert_cable_cycle", cycle=cycle + 1, phase="to_target")
            for _ in range(50):
                joint_motion_update.target_state.positions = target
                move_robot(joint_motion_update=joint_motion_update)
                self.sleep_for(0.1)
            _diag("insert_cable_cycle", cycle=cycle + 1, phase="to_home")
            for _ in range(50):
                joint_motion_update.target_state.positions = home
                move_robot(joint_motion_update=joint_motion_update)
                self.sleep_for(0.1)

        joint_motion_update.target_stiffness = [200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
        joint_motion_update.target_damping = [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        for _ in range(30):
            joint_motion_update.target_state.positions = home
            move_robot(joint_motion_update=joint_motion_update)
            self.sleep_for(0.1)

        _diag("insert_cable_return", task_id=task.id, returning=True)
        return True
