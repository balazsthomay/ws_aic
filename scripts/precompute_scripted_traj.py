#!/usr/bin/env python3
"""Pre-compute a scripted joint trajectory: HOME → above NOMINAL port → descent.

The output file (data/outputs/scripted_traj.npz) is baked into a ROS2 policy
as a deterministic baseline. No vision; just plays back the trajectory.

Useful as a sanity check / floor: the cluster's randomization is small enough
that even a blind trajectory at the nominal port may earn proximity credit.
"""

import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "src/aic/aic_utils/aic_mujoco/mjcf/scene.xml"
OUT = ROOT / "data/outputs/scripted_traj.npz"

HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

CONTROL_HZ = 20
APPROACH_TIME = 5.0
DESCENT_TIME = 8.0
HOLD_TIME = 1.0
ABOVE_PORT = 0.060
INSERT_DEPTH = 0.015


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_tip_link")
    port_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_port_0_link")
    qids = np.array([
        m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in JOINT_NAMES
    ])
    dids = np.array([
        m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in JOINT_NAMES
    ])

    port_pos = d.xpos[port_id].copy()
    tcp_home = d.site_xpos[tcp_site].copy()
    tcp_quat = np.zeros(4)
    mujoco.mju_mat2Quat(tcp_quat, d.site_xmat[tcp_site].flatten())
    tip_offset = d.xpos[tip_id] - d.site_xpos[tcp_site]

    print(f"port_pos: {port_pos.round(4)}")
    print(f"tcp_home: {tcp_home.round(4)}")

    d_ik = mujoco.MjData(m)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))

    def solve_ik(target, q_init):
        d_ik.qpos[:] = d.qpos[:]
        d_ik.qpos[qids] = q_init.copy()
        for _ in range(100):
            mujoco.mj_forward(m, d_ik)
            pe = target - d_ik.site_xpos[tcp_site]
            sm = d_ik.site_xmat[tcp_site].reshape(3, 3)
            tm = np.zeros(9)
            mujoco.mju_quat2Mat(tm, tcp_quat)
            tm = tm.reshape(3, 3)
            Re = tm @ sm.T
            eq = np.zeros(4)
            mujoco.mju_mat2Quat(eq, Re.flatten())
            if eq[0] < 0:
                eq = -eq
            re = 2.0 * eq[1:4]
            if np.linalg.norm(pe) < 3e-4 and np.linalg.norm(re) < 0.01:
                break
            err = np.concatenate([pe, 0.5 * re])
            mujoco.mj_jacSite(m, d_ik, jacp, jacr, tcp_site)
            J = np.vstack([jacp[:, dids], jacr[:, dids]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            d_ik.qpos[qids] += 0.15 * dq
        return d_ik.qpos[qids].copy()

    tcp_above = port_pos + np.array([0, 0, ABOVE_PORT]) - tip_offset

    # Phase 1: approach (cosine-interpolated)
    n_app = int(APPROACH_TIME * CONTROL_HZ)
    approach = np.zeros((n_app, 6))
    q_prev = HOME.copy()
    for i in range(n_app):
        alpha = 0.5 * (1 - math.cos(math.pi * i / (n_app - 1)))
        tcp_i = (1 - alpha) * tcp_home + alpha * tcp_above
        approach[i] = solve_ik(tcp_i, q_prev)
        q_prev = approach[i]

    # Phase 2: descent (linear in Z)
    n_desc = int(DESCENT_TIME * CONTROL_HZ)
    descent = np.zeros((n_desc, 6))
    q_prev = approach[-1].copy()
    for i in range(n_desc):
        z_off = ABOVE_PORT - (i / (n_desc - 1)) * (ABOVE_PORT + INSERT_DEPTH)
        tcp_i = tcp_above.copy()
        tcp_i[2] = port_pos[2] + z_off - tip_offset[2]
        descent[i] = solve_ik(tcp_i, q_prev)
        q_prev = descent[i]

    # Phase 3: hold (repeat final descent pose)
    n_hold = int(HOLD_TIME * CONTROL_HZ)
    hold = np.tile(descent[-1], (n_hold, 1))

    full = np.concatenate([approach, descent, hold], axis=0)
    print(f"Trajectory: {full.shape} ({full.shape[0]/CONTROL_HZ:.1f}s)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, joints=full, control_hz=CONTROL_HZ)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
