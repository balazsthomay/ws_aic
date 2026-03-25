#!/usr/bin/env python3
"""SFP cable insertion policy for MuJoCo standalone.

Ports the CheatCode strategy: approach above port, descend straight down.
Uses ground truth positions (sfp_port_0_link body) for targeting.
Trajectory is pre-computed via IK, then played back with set_mjcb_control.

Usage:
    .venv/bin/python3 scripts/insert.py          # viewer mode
    .venv/bin/python3 scripts/insert.py --headless  # headless test
"""

import argparse
import math
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / "src/aic/aic_utils/aic_mujoco/mjcf/scene.xml"

HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Timing
APPROACH_TIME = 5.0  # seconds to move from home to above port
DESCENT_TIME = 8.0   # seconds to descend through port
HOLD_TIME = 2.0      # seconds to hold inserted position

# Geometry
ABOVE_PORT = 0.060   # start 60mm above port (clear of receptacle)
INSERT_DEPTH = 0.015  # insert 15mm past port reference


def build_trajectory(m, d):
    """Pre-compute approach + descent IK trajectory."""
    tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_tip_link")
    port_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_port_0_link")

    qids, dids, aids = [], [], []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qids.append(m.jnt_qposadr[jid])
        dids.append(m.jnt_dofadr[jid])
        aids.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_motor"))
    qids, dids, aids = np.array(qids), np.array(dids), np.array(aids)

    port_pos = d.xpos[port_id].copy()
    tcp_home = d.site_xpos[tcp_site].copy()
    tcp_quat_home = np.zeros(4)
    mujoco.mju_mat2Quat(tcp_quat_home, d.site_xmat[tcp_site].flatten())
    tip_offset = d.xpos[tip_id] - d.site_xpos[tcp_site]

    d_ik = mujoco.MjData(m)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))

    def solve_ik(target_pos, q_init, max_iter=100):
        d_ik.qpos[:] = d.qpos[:]
        d_ik.qpos[qids] = q_init.copy()
        for _ in range(max_iter):
            mujoco.mj_forward(m, d_ik)
            pe = target_pos - d_ik.site_xpos[tcp_site]
            sm = d_ik.site_xmat[tcp_site].reshape(3, 3)
            tm = np.zeros(9)
            mujoco.mju_quat2Mat(tm, tcp_quat_home)
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

    # TCP target when tip is above port
    tcp_above = port_pos + np.array([0, 0, ABOVE_PORT]) - tip_offset

    # Phase 1: Approach (cosine-interpolated home → above port)
    print("Computing approach...")
    N_APPROACH = 80
    approach = np.zeros((N_APPROACH, 6))
    q_prev = HOME.copy()
    for i in range(N_APPROACH):
        alpha = 0.5 * (1 - math.cos(math.pi * i / (N_APPROACH - 1)))
        tcp_i = (1 - alpha) * tcp_home + alpha * tcp_above
        approach[i] = solve_ik(tcp_i, q_prev)
        q_prev = approach[i]

    # Phase 2: Descent (lower Z from +60mm to -15mm relative to port)
    print("Computing descent...")
    N_DESCENT = 150
    descent = np.zeros((N_DESCENT, 6))
    q_prev = approach[-1].copy()
    for i in range(N_DESCENT):
        z_off = ABOVE_PORT - (i / (N_DESCENT - 1)) * (ABOVE_PORT + INSERT_DEPTH)
        tcp_i = tcp_above.copy()
        tcp_i[2] = port_pos[2] + z_off - tip_offset[2]
        descent[i] = solve_ik(tcp_i, q_prev)
        q_prev = descent[i]

    print("Trajectory ready.")
    return approach, descent, aids, qids, tip_id, port_id, dids


def get_ctrl(t, approach, descent):
    """Look up ctrl target at simulation time t."""
    total = APPROACH_TIME + DESCENT_TIME
    if t < APPROACH_TIME:
        n = len(approach)
        frac = t / APPROACH_TIME * (n - 1)
        i0 = min(int(frac), n - 2)
        a = frac - i0
        return (1 - a) * approach[i0] + a * approach[i0 + 1]
    elif t < total:
        n = len(descent)
        frac = (t - APPROACH_TIME) / DESCENT_TIME * (n - 1)
        i0 = min(int(frac), n - 2)
        a = frac - i0
        return (1 - a) * descent[i0] + a * descent[i0 + 1]
    else:
        return descent[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    approach, descent, aids, qids, tip_id, port_id, dids = build_trajectory(m, d)
    port_pos = d.xpos[port_id].copy()

    # Reset for simulation
    mujoco.mj_resetDataKeyframe(m, d, 0)

    if args.headless:
        # Headless test
        dt = m.opt.timestep
        total = APPROACH_TIME + DESCENT_TIME + HOLD_TIME
        for s in range(int(total / dt)):
            t = s * dt
            ctrl = get_ctrl(t, approach, descent)
            for i, ai in enumerate(aids):
                d.ctrl[ai] = ctrl[i]
            d.ctrl[6] = 0.0
            mujoco.mj_step(m, d)

            if s % 1000 == 999:
                tip = d.xpos[tip_id]
                xy_e = np.linalg.norm(tip[:2] - port_pos[:2])
                z_rel = (tip[2] - port_pos[2]) * 1000
                print(f"t={t:.1f}s  xy={xy_e*1000:.1f}mm  z={z_rel:.1f}mm")

        tip = d.xpos[tip_id]
        xy_e = np.linalg.norm(tip[:2] - port_pos[:2])
        z_rel = (tip[2] - port_pos[2]) * 1000
        print(f"\nXY: {xy_e*1000:.1f}mm  Z: {z_rel:.1f}mm")
        print("SUCCESS" if xy_e < 5 / 1000 and z_rel < -5 else "FAIL")
    else:
        # Viewer mode with rate-limited ctrl (safe on reset)
        state = {"ctrl": None, "prev_time": 0.0}
        max_ctrl_rate = 0.5  # rad/s

        def controller(model, data):
            dt = model.opt.timestep
            max_step = max_ctrl_rate * dt
            t = data.time

            # Detect reset or first call: seed from actual qpos
            if state["ctrl"] is None or t < state["prev_time"] - 0.5:
                state["ctrl"] = data.qpos[qids].copy()
            state["prev_time"] = t

            q_desired = get_ctrl(t, approach, descent)
            diff = q_desired - state["ctrl"]
            state["ctrl"] = state["ctrl"] + np.clip(diff, -max_step, max_step)

            for i, aid in enumerate(aids):
                data.ctrl[aid] = state["ctrl"][i]
            data.ctrl[6] = 0.0

        mujoco.set_mjcb_control(controller)
        mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    main()
