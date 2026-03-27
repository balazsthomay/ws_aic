#!/usr/bin/env python3
"""Collect insertion demonstrations with randomized task board positions.

Runs the ground-truth insertion policy N times, each with a randomized
task board yaw and slight XY offset. Saves (state, action) trajectories
for training a learned policy.

Usage:
    .venv/bin/python3 scripts/collect_demos.py --episodes 100 --out data/demos.npz
"""

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src/aic/aic_utils/aic_mujoco"))
from mujoco_obs import MuJoCoObserver

SCENE = Path(__file__).resolve().parent.parent / "src/aic/aic_utils/aic_mujoco/mjcf/scene.xml"

HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Timing
APPROACH_TIME = 5.0
DESCENT_TIME = 8.0
HOLD_TIME = 1.0
RECORD_HZ = 20  # save observations at 20Hz

# Randomization ranges
BOARD_YAW_RANGE = 0.15      # ±8.6 degrees
BOARD_XY_RANGE = 0.015      # ±15mm
NIC_RAIL_RANGE = 0.02       # ±20mm along rail

# Approach geometry
ABOVE_PORT = 0.060
INSERT_DEPTH = 0.015


def setup_indices(m):
    """Pre-compute joint/actuator/body indices."""
    tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_tip_link")
    port_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_port_0_link")
    board_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "task_board_base_link")

    qids, dids, aids = [], [], []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qids.append(m.jnt_qposadr[jid])
        dids.append(m.jnt_dofadr[jid])
        aids.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_motor"))

    return {
        "tcp_site": tcp_site,
        "tip_id": tip_id,
        "port_id": port_id,
        "board_id": board_id,
        "qids": np.array(qids),
        "dids": np.array(dids),
        "aids": np.array(aids),
    }


def solve_ik(m, d, idx, target_tcp_pos, tcp_quat, q_init, d_ik):
    """Position + orientation IK."""
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    d_ik.qpos[:] = d.qpos[:]
    d_ik.qpos[idx["qids"]] = q_init.copy()

    for _ in range(100):
        mujoco.mj_forward(m, d_ik)
        pe = target_tcp_pos - d_ik.site_xpos[idx["tcp_site"]]
        sm = d_ik.site_xmat[idx["tcp_site"]].reshape(3, 3)
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
        mujoco.mj_jacSite(m, d_ik, jacp, jacr, idx["tcp_site"])
        J = np.vstack([jacp[:, idx["dids"]], jacr[:, idx["dids"]]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        d_ik.qpos[idx["qids"]] += 0.15 * dq

    return d_ik.qpos[idx["qids"]].copy()


def build_trajectory(m, d, idx, d_ik):
    """Build approach + descent trajectory for current port position."""
    port_pos = d.xpos[idx["port_id"]].copy()
    tcp_home = d.site_xpos[idx["tcp_site"]].copy()
    tcp_quat = np.zeros(4)
    mujoco.mju_mat2Quat(tcp_quat, d.site_xmat[idx["tcp_site"]].flatten())
    tip_offset = d.xpos[idx["tip_id"]] - d.site_xpos[idx["tcp_site"]]

    tcp_above = port_pos + np.array([0, 0, ABOVE_PORT]) - tip_offset

    # Approach
    N_APP = 80
    approach = np.zeros((N_APP, 6))
    q_prev = HOME.copy()
    for i in range(N_APP):
        alpha = 0.5 * (1 - math.cos(math.pi * i / (N_APP - 1)))
        tcp_i = (1 - alpha) * tcp_home + alpha * tcp_above
        approach[i] = solve_ik(m, d, idx, tcp_i, tcp_quat, q_prev, d_ik)
        q_prev = approach[i]

    # Descent
    N_DESC = 150
    descent = np.zeros((N_DESC, 6))
    q_prev = approach[-1].copy()
    for i in range(N_DESC):
        z_off = ABOVE_PORT - (i / (N_DESC - 1)) * (ABOVE_PORT + INSERT_DEPTH)
        tcp_i = tcp_above.copy()
        tcp_i[2] = port_pos[2] + z_off - tip_offset[2]
        descent[i] = solve_ik(m, d, idx, tcp_i, tcp_quat, q_prev, d_ik)
        q_prev = descent[i]

    return approach, descent


def get_ctrl(t, approach, descent):
    """Look up trajectory ctrl at time t."""
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


def randomize_board(m, idx, rng, nominal_pos, nominal_quat):
    """Randomize task board position and yaw."""
    dx = rng.uniform(-BOARD_XY_RANGE, BOARD_XY_RANGE)
    dy = rng.uniform(-BOARD_XY_RANGE, BOARD_XY_RANGE)
    dyaw = rng.uniform(-BOARD_YAW_RANGE, BOARD_YAW_RANGE)

    m.body_pos[idx["board_id"]] = nominal_pos + np.array([dx, dy, 0])

    # Apply yaw rotation to nominal quaternion
    cyaw = math.cos(dyaw / 2)
    syaw = math.sin(dyaw / 2)
    dq = np.array([cyaw, 0, 0, syaw])  # rotation around Z
    new_quat = np.zeros(4)
    mujoco.mju_mulQuat(new_quat, dq, nominal_quat)
    m.body_quat[idx["board_id"]] = new_quat

    return {"dx": dx, "dy": dy, "dyaw": dyaw}


def run_episode(m, d, idx, d_ik, approach, descent, observer=None):
    """Simulate one insertion episode, return trajectory data."""
    mujoco.mj_resetDataKeyframe(m, d, 0)
    dt = m.opt.timestep
    total_time = APPROACH_TIME + DESCENT_TIME + HOLD_TIME
    record_interval = int(1.0 / (RECORD_HZ * dt))

    states = []
    actions = []
    images = {name: [] for name in ["left_camera", "center_camera", "right_camera"]}

    for s in range(int(total_time / dt)):
        t = s * dt
        ctrl = get_ctrl(t, approach, descent)

        for i, ai in enumerate(idx["aids"]):
            d.ctrl[ai] = ctrl[i]
        d.ctrl[6] = 0.0
        mujoco.mj_step(m, d)

        # Record at RECORD_HZ
        if s % record_interval == 0:
            tcp_pos = d.site_xpos[idx["tcp_site"]].copy()
            tcp_mat = d.site_xmat[idx["tcp_site"]].reshape(3, 3)
            tcp_quat = np.zeros(4)
            mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())

            state = np.concatenate([
                d.qpos[idx["qids"]].copy(),      # joint positions (6)
                d.qvel[idx["dids"]].copy(),       # joint velocities (6)
                tcp_pos,                           # TCP position (3)
                tcp_quat,                          # TCP quaternion (4)
                d.xpos[idx["tip_id"]].copy(),     # tip position (3)
                d.xpos[idx["port_id"]].copy(),    # port position (3)
            ])
            states.append(state)
            actions.append(ctrl.copy())

            if observer is not None:
                obs = observer.get_observation()
                for name in images:
                    images[name].append(obs.images[name])

    # Final check
    tip = d.xpos[idx["tip_id"]]
    port = d.xpos[idx["port_id"]]
    xy_err = np.linalg.norm(tip[:2] - port[:2])
    z_rel = (tip[2] - port[2]) * 1000
    success = xy_err < 0.005 and z_rel < -5

    # Stack images
    images_np = {}
    if observer is not None:
        for name in images:
            images_np[name] = np.array(images[name], dtype=np.uint8)

    return np.array(states), np.array(actions), images_np, success, xy_err, z_rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="data/demos.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images", action="store_true", help="Save camera images (slower, larger files)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    d_ik = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    idx = setup_indices(m)
    rng = np.random.default_rng(args.seed)

    observer = None
    if args.images:
        observer = MuJoCoObserver(m, d, image_scale=0.25)
        print("Image recording enabled (0.25x scale)")

    # Save nominal board pose
    nominal_pos = m.body_pos[idx["board_id"]].copy()
    nominal_quat = m.body_quat[idx["board_id"]].copy()

    all_states = []
    all_actions = []
    all_images = {name: [] for name in ["left_camera", "center_camera", "right_camera"]}
    episode_lengths = []
    episode_success = []
    episode_params = []

    print(f"Collecting {args.episodes} demonstrations...")

    for ep in range(args.episodes):
        # Randomize board
        params = randomize_board(m, idx, rng, nominal_pos, nominal_quat)

        # Reset and forward to update body positions
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)

        # Build trajectory for this board configuration
        approach, descent = build_trajectory(m, d, idx, d_ik)

        # Run episode
        states, actions, images, success, xy_err, z_rel = run_episode(
            m, d, idx, d_ik, approach, descent, observer
        )

        all_states.append(states)
        all_actions.append(actions)
        if args.images:
            for name in all_images:
                all_images[name].append(images[name])
        episode_lengths.append(len(states))
        episode_success.append(success)
        episode_params.append([params["dx"], params["dy"], params["dyaw"]])

        status = "OK" if success else "FAIL"
        if (ep + 1) % 10 == 0 or not success:
            print(
                f"  [{ep+1}/{args.episodes}] {status} "
                f"xy={xy_err*1000:.1f}mm z={z_rel:.1f}mm "
                f"dx={params['dx']*1000:.1f}mm dy={params['dy']*1000:.1f}mm "
                f"dyaw={math.degrees(params['dyaw']):.1f}°"
            )

    # Restore nominal board pose
    m.body_pos[idx["board_id"]] = nominal_pos
    m.body_quat[idx["board_id"]] = nominal_quat

    # Save
    success_rate = sum(episode_success) / len(episode_success)
    print(f"\nSuccess rate: {sum(episode_success)}/{len(episode_success)} ({success_rate:.0%})")

    # Pad to uniform length and save
    max_len = max(episode_lengths)
    n_eps = len(all_states)
    state_dim = all_states[0].shape[1]
    action_dim = all_actions[0].shape[1]

    states_padded = np.zeros((n_eps, max_len, state_dim))
    actions_padded = np.zeros((n_eps, max_len, action_dim))
    for i in range(n_eps):
        L = episode_lengths[i]
        states_padded[i, :L] = all_states[i]
        actions_padded[i, :L] = all_actions[i]

    save_dict = dict(
        states=states_padded,
        actions=actions_padded,
        episode_lengths=np.array(episode_lengths),
        episode_success=np.array(episode_success),
        episode_params=np.array(episode_params),
        state_labels=[
            "joint_pos_0", "joint_pos_1", "joint_pos_2",
            "joint_pos_3", "joint_pos_4", "joint_pos_5",
            "joint_vel_0", "joint_vel_1", "joint_vel_2",
            "joint_vel_3", "joint_vel_4", "joint_vel_5",
            "tcp_x", "tcp_y", "tcp_z",
            "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz",
            "tip_x", "tip_y", "tip_z",
            "port_x", "port_y", "port_z",
        ],
        action_labels=[
            "ctrl_0", "ctrl_1", "ctrl_2", "ctrl_3", "ctrl_4", "ctrl_5",
        ],
    )

    if args.images:
        for name in all_images:
            # Stack: (n_eps, max_len, H, W, 3)
            imgs = all_images[name]
            H, W = imgs[0].shape[1], imgs[0].shape[2]
            padded = np.zeros((n_eps, max_len, H, W, 3), dtype=np.uint8)
            for i in range(n_eps):
                L = episode_lengths[i]
                padded[i, :L] = imgs[i]
            save_dict[f"images_{name}"] = padded

    np.savez_compressed(out_path, **save_dict)

    if observer is not None:
        observer.close()

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {n_eps} episodes ({max_len} steps each) to {out_path} ({size_mb:.1f}MB)")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    if args.images:
        print(f"Images: 3 cameras at {H}x{W}")


if __name__ == "__main__":
    main()
