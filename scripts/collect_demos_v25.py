#!/usr/bin/env python3
"""Broad-distribution demo collection for v25.

Each episode samples uniformly from {SFP on nic_card_mount_0..4} ∪
{SC on sc_port_0, sc_port_1}, with rail-translation jitter and board-pose
jitter. The MJCF only ships nic_card_mount_0 + sc_port_0 — for the other
configurations we mutate body_pos before mj_forward (rail anchor +
translation) and read the relevant child body's world xpos as the target.

Output is a single npz with `port_xy_world` per episode (the label the
localizer learns) plus images, joint trajectories, and a `target_id` string
identifying the configuration.
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

APPROACH_TIME = 5.0
DESCENT_TIME = 8.0
HOLD_TIME = 1.0
RECORD_HZ = 20

ABOVE_PORT = 0.060
INSERT_DEPTH = 0.015

# NIC rail anchors in board frame (from task_board.urdf.xacro).
NIC_RAIL_Y = {0: -0.1745, 1: -0.1345, 2: -0.0945, 3: -0.0545, 4: -0.0145}
NIC_TRANSLATION_BOUNDS = (-0.0215, 0.0234)  # from sample_config task_board_limits

# SC port anchors in board frame.
SC_PORT_BASE = {
    0: np.array([-0.075, 0.0295, 0.0165]),  # sc_port_0
    1: np.array([-0.075, 0.0705, 0.0165]),  # sc_port_1
}
SC_TRANSLATION_BOUNDS = (-0.06, 0.055)

# Board pose jitter — broad enough to cover trial 1/2 (yaw=π) and trial 3
# (yaw=3.0) plus margin. Real cluster's range is unknown.
BOARD_DX_RANGE = 0.05      # ±50mm
BOARD_DY_RANGE = 0.15      # ±150mm covers (-0.2 vs 0.0)
BOARD_DYAW_RANGE = 0.3     # ±0.3 rad (~17°)


def setup_indices(m):
    return {
        "tcp_site": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp"),
        "sfp_tip_id": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_tip_link"),
        "board_id": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "task_board_base_link"),
        "nic_mount_id": mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "nic_card_mount_0::nic_card_mount_link"
        ),
        "sc_port_id": mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "sc_port_0::sc_port_link"
        ),
        "sfp_port_0_id": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_port_0_link"),
        "sc_port_base_id": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sc_port_base_link"),
        "qids": np.array([
            m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES
        ]),
        "dids": np.array([
            m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES
        ]),
        "aids": np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_motor")
            for n in JOINT_NAMES
        ]),
    }


def yaw_to_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def configure_scene(m, idx, board_pos, board_yaw, target):
    """Mutate MJCF body_pos / body_quat to encode this episode's scene.

    target: dict with port_kind ("sfp" or "sc") and the placement parameters
    needed to position the relevant body. Returns the body id of the target
    port whose world xpos becomes the IK target.
    """
    m.body_pos[idx["board_id"]] = board_pos
    m.body_quat[idx["board_id"]] = yaw_to_quat(board_yaw)

    if target["port_kind"] == "sfp":
        # Move nic_card_mount_0 to the chosen rail's anchor + translation.
        m.body_pos[idx["nic_mount_id"]] = np.array([
            -0.081418 + target["nic_translation"],
            NIC_RAIL_Y[target["rail"]],
            0.012,
        ])
        return idx["sfp_port_0_id"]
    elif target["port_kind"] == "sc":
        sc_anchor = SC_PORT_BASE[target["sc_port_index"]].copy()
        sc_anchor[0] += target["sc_translation"]
        m.body_pos[idx["sc_port_id"]] = sc_anchor
        return idx["sc_port_base_id"]
    else:
        raise ValueError(target["port_kind"])


def sample_target(rng):
    # 5 SFP configs + 2 SC configs → 7 buckets, sample uniformly.
    bucket = rng.integers(0, 7)
    if bucket < 5:
        return {
            "port_kind": "sfp",
            "rail": int(bucket),
            "nic_translation": float(rng.uniform(*NIC_TRANSLATION_BOUNDS)),
            "id": f"sfp/nic_card_mount_{bucket}/sfp_port_0",
        }
    elif bucket == 5:
        return {
            "port_kind": "sc",
            "sc_port_index": 0,
            "sc_translation": float(rng.uniform(*SC_TRANSLATION_BOUNDS)),
            "id": "sc/sc_port_0/sc_port_base",
        }
    else:
        return {
            "port_kind": "sc",
            "sc_port_index": 1,
            "sc_translation": float(rng.uniform(*SC_TRANSLATION_BOUNDS)),
            "id": "sc/sc_port_1/sc_port_base",
        }


def solve_ik(m, idx, target_pos, tcp_quat, q_init, d_ik):
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    d_ik.qpos[idx["qids"]] = q_init.copy()
    for _ in range(150):
        mujoco.mj_forward(m, d_ik)
        pe = target_pos - d_ik.site_xpos[idx["tcp_site"]]
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


def build_trajectory(m, d, idx, port_pos, d_ik):
    tcp_home = d.site_xpos[idx["tcp_site"]].copy()
    tcp_quat = np.zeros(4)
    mujoco.mju_mat2Quat(tcp_quat, d.site_xmat[idx["tcp_site"]].flatten())
    R_tcp = d.site_xmat[idx["tcp_site"]].reshape(3, 3).copy()
    # Tip offset proxy: use sfp_tip's world position relative to TCP. The
    # actual cable-plug-to-TCP offset per cable type is small (~3mm) — close
    # enough for demo collection.
    tip_offset = d.xpos[idx["sfp_tip_id"]] - d.site_xpos[idx["tcp_site"]]

    tcp_above = port_pos + np.array([0, 0, ABOVE_PORT]) - tip_offset

    N_APP = 80
    approach = np.zeros((N_APP, 6))
    q_prev = HOME.copy()
    for i in range(N_APP):
        alpha = 0.5 * (1 - math.cos(math.pi * i / (N_APP - 1)))
        tcp_i = (1 - alpha) * tcp_home + alpha * tcp_above
        approach[i] = solve_ik(m, idx, tcp_i, tcp_quat, q_prev, d_ik)
        q_prev = approach[i]

    N_DESC = 150
    descent = np.zeros((N_DESC, 6))
    q_prev = approach[-1].copy()
    for i in range(N_DESC):
        z_off = ABOVE_PORT - (i / (N_DESC - 1)) * (ABOVE_PORT + INSERT_DEPTH)
        tcp_i = tcp_above.copy()
        tcp_i[2] = port_pos[2] + z_off - tip_offset[2]
        descent[i] = solve_ik(m, idx, tcp_i, tcp_quat, q_prev, d_ik)
        q_prev = descent[i]

    return approach, descent


def get_ctrl(t, approach, descent):
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
    return descent[-1]


def run_episode(m, d, idx, approach, descent, target_port_body, observer=None):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    dt = m.opt.timestep
    total_time = APPROACH_TIME + DESCENT_TIME + HOLD_TIME
    record_interval = int(1.0 / (RECORD_HZ * dt))

    states, actions = [], []
    images = {n: [] for n in ["left_camera", "center_camera", "right_camera"]}

    for s in range(int(total_time / dt)):
        t = s * dt
        ctrl = get_ctrl(t, approach, descent)
        for i, ai in enumerate(idx["aids"]):
            d.ctrl[ai] = ctrl[i]
        d.ctrl[6] = 0.0
        mujoco.mj_step(m, d)

        if s % record_interval == 0:
            tcp_pos = d.site_xpos[idx["tcp_site"]].copy()
            tcp_mat = d.site_xmat[idx["tcp_site"]].reshape(3, 3)
            tcp_quat = np.zeros(4)
            mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())
            progress = t / total_time
            state = np.concatenate([
                d.qpos[idx["qids"]].copy(),
                d.qvel[idx["dids"]].copy(),
                tcp_pos,
                tcp_quat,
                d.xpos[idx["sfp_tip_id"]].copy(),     # tip proxy
                d.xpos[target_port_body].copy(),       # target port world pos
                [progress],
            ])
            states.append(state)
            actions.append(ctrl.copy())
            if observer is not None:
                obs = observer.get_observation()
                for name in images:
                    images[name].append(obs.images[name])

    images_np = {n: np.array(images[n], dtype=np.uint8) for n in images} if observer else {}
    return np.array(states), np.array(actions), images_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--out", type=str, default="data/demos_v25.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-images", action="store_true")
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

    observer = MuJoCoObserver(m, d, image_scale=0.25) if not args.no_images else None
    if observer:
        print("Image recording enabled (0.25x)")

    nominal_pos = m.body_pos[idx["board_id"]].copy()
    nominal_quat = m.body_quat[idx["board_id"]].copy()
    nominal_yaw = math.atan2(2 * nominal_quat[0] * nominal_quat[3],
                             1 - 2 * nominal_quat[3] ** 2)
    nominal_nic = m.body_pos[idx["nic_mount_id"]].copy()
    nominal_sc = m.body_pos[idx["sc_port_id"]].copy()

    all_states, all_actions = [], []
    all_images = {n: [] for n in ["left_camera", "center_camera", "right_camera"]}
    ep_lengths, ep_targets = [], []
    ep_port_xy_world, ep_board = [], []

    print(f"Collecting {args.episodes} demos…")
    for ep in range(args.episodes):
        target = sample_target(rng)
        board_pos = nominal_pos + np.array([
            rng.uniform(-BOARD_DX_RANGE, BOARD_DX_RANGE),
            rng.uniform(-BOARD_DY_RANGE, BOARD_DY_RANGE),
            0.0,
        ])
        board_yaw = nominal_yaw + rng.uniform(-BOARD_DYAW_RANGE, BOARD_DYAW_RANGE)
        target_port_body = configure_scene(m, idx, board_pos, board_yaw, target)

        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        port_pos = d.xpos[target_port_body].copy()

        approach, descent = build_trajectory(m, d, idx, port_pos, d_ik)
        states, actions, images = run_episode(
            m, d, idx, approach, descent, target_port_body, observer
        )
        all_states.append(states)
        all_actions.append(actions)
        if observer:
            for name in all_images:
                all_images[name].append(images[name])
        ep_lengths.append(len(states))
        ep_targets.append(target["id"])
        ep_port_xy_world.append(port_pos)
        ep_board.append([board_pos[0], board_pos[1], board_yaw])

        if (ep + 1) % 10 == 0:
            print(f"  [{ep+1}/{args.episodes}] target={target['id']}")

        # Restore for next episode (configure_scene will re-set as needed).
        m.body_pos[idx["nic_mount_id"]] = nominal_nic
        m.body_pos[idx["sc_port_id"]] = nominal_sc

    m.body_pos[idx["board_id"]] = nominal_pos
    m.body_quat[idx["board_id"]] = nominal_quat

    max_len = max(ep_lengths)
    n = len(all_states)
    state_dim = all_states[0].shape[1]
    action_dim = all_actions[0].shape[1]
    states_padded = np.zeros((n, max_len, state_dim))
    actions_padded = np.zeros((n, max_len, action_dim))
    for i in range(n):
        L = ep_lengths[i]
        states_padded[i, :L] = all_states[i]
        actions_padded[i, :L] = all_actions[i]

    save = dict(
        states=states_padded,
        actions=actions_padded,
        episode_lengths=np.array(ep_lengths),
        targets=np.array(ep_targets, dtype=object),
        port_xy_world=np.array(ep_port_xy_world),  # 3D xyz (use [:2] for xy)
        board=np.array(ep_board),                  # bx, by, byaw
    )
    if observer:
        for name in all_images:
            save[f"images_{name}"] = np.stack(all_images[name])

    np.savez(out_path, **save)
    print(f"Saved → {out_path}")
    print(f"  episodes: {n}, by target:")
    from collections import Counter
    for t, c in Counter(ep_targets).items():
        print(f"    {t}: {c}")


if __name__ == "__main__":
    main()
