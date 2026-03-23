#!/usr/bin/env python3
"""Port of WaveArm policy to standalone MuJoCo.

Sweeps the end-effector sinusoidally in Y around the home position,
maintaining home orientation. Trajectory is pre-computed via IK
with nullspace bias to stay near home config.

Control output is rate-limited so resets never cause violent snaps.

Usage:
    .venv/bin/python3 scripts/wave.py
"""

import math
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

SCENE = Path(__file__).resolve().parent.parent / "src/aic/aic_utils/aic_mujoco/mjcf/scene.xml"

HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

LOOP_SECONDS = 6.0
AMPLITUDE = 0.15  # meters in Y
NPTS = 200         # trajectory waypoints per cycle
MAX_CTRL_RATE = 0.5  # rad/s max ctrl change rate — prevents violent snaps

# --- Load and initialize ---
m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

ee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ati/tool_link")
home_ee_pos = d.xpos[ee_id].copy()
home_ee_quat = d.xquat[ee_id].copy()

aids, qids, dids = [], [], []
for name in JOINT_NAMES:
    aids.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_motor"))
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    qids.append(m.jnt_qposadr[jid])
    dids.append(m.jnt_dofadr[jid])
aids, qids, dids = np.array(aids), np.array(qids), np.array(dids)


def _solve_ik(d_ik, target_pos, q_init):
    d_ik.qpos[qids] = q_init.copy()
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for _ in range(100):
        mujoco.mj_forward(m, d_ik)
        pe = target_pos - d_ik.xpos[ee_id]
        qc = d_ik.xquat[ee_id].copy()
        qc[1:] *= -1
        qe = np.zeros(4)
        mujoco.mju_mulQuat(qe, home_ee_quat, qc)
        if qe[0] < 0:
            qe = -qe
        re = 2.0 * qe[1:4]
        if np.linalg.norm(pe) < 1e-4 and np.linalg.norm(re) < 0.01:
            break
        err = np.concatenate([pe, 0.3 * re])
        mujoco.mj_jac(m, d_ik, jacp, jacr, d_ik.xpos[ee_id], ee_id)
        J = np.vstack([jacp[:, dids], jacr[:, dids]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        ns = (np.eye(6) - np.linalg.pinv(J) @ J) @ (HOME - d_ik.qpos[qids])
        d_ik.qpos[qids] += 0.2 * (dq + 0.01 * ns)
    return d_ik.qpos[qids].copy()


# --- Pre-compute trajectory ---
print(f"Pre-computing {NPTS} IK waypoints...")
d_ik = mujoco.MjData(m)
d_ik.qpos[:] = d.qpos[:]
_trajectory = np.zeros((NPTS, 6))
q_prev = HOME.copy()
for i in range(NPTS):
    t = (i / NPTS) * LOOP_SECONDS
    y_offset = AMPLITUDE * math.sin(2 * math.pi * t / LOOP_SECONDS)
    target = home_ee_pos + np.array([0, y_offset, 0])
    q_prev = _solve_ik(d_ik, target, q_prev)
    _trajectory[i] = q_prev

print("Trajectory ready. Launching viewer...")

_current_ctrl = None
_prev_time = 0.0


def controller(model, data):
    global _current_ctrl, _prev_time

    dt = model.opt.timestep
    max_step = MAX_CTRL_RATE * dt

    t = data.time

    # Detect reset (time jumped back) or first call: seed ctrl from actual qpos
    if _current_ctrl is None or t < _prev_time - 0.5:
        _current_ctrl = data.qpos[qids].copy()
    _prev_time = t

    # Wave trajectory lookup
    frac = (t % LOOP_SECONDS) / LOOP_SECONDS * NPTS
    idx = int(frac) % NPTS
    nxt = (idx + 1) % NPTS
    a = frac - int(frac)
    q_desired = (1 - a) * _trajectory[idx] + a * _trajectory[nxt]

    # Rate-limit: ctrl moves toward desired, max MAX_CTRL_RATE rad/s
    diff = q_desired - _current_ctrl
    _current_ctrl = _current_ctrl + np.clip(diff, -max_step, max_step)

    for i, aid in enumerate(aids):
        data.ctrl[aid] = _current_ctrl[i]
    data.ctrl[6] = 0.0


mujoco.set_mjcb_control(controller)
mujoco.viewer.launch(m, d)
