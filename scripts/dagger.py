#!/usr/bin/env python3
"""DAgger: Dataset Aggregation for behavior cloning.

Iteratively:
1. Roll out the learned policy in simulation
2. At each visited state, query the expert (IK) for the correct action
3. Add these (state, expert_action) pairs to the training set
4. Retrain the policy on the aggregated dataset

Usage:
    # Run locally (CPU training, fast iteration)
    .venv/bin/python3 scripts/dagger.py --iterations 5 --episodes-per-iter 20

    # With more episodes for better coverage
    .venv/bin/python3 scripts/dagger.py --iterations 10 --episodes-per-iter 50
"""

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn

SCENE = Path(__file__).resolve().parent.parent / "src/aic/aic_utils/aic_mujoco/mjcf/scene.xml"

HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

APPROACH_TIME = 5.0
DESCENT_TIME = 8.0
HOLD_TIME = 1.0
TOTAL_TIME = APPROACH_TIME + DESCENT_TIME + HOLD_TIME
CONTROL_HZ = 20
ABOVE_PORT = 0.060
INSERT_DEPTH = 0.015
BOARD_YAW_RANGE = 0.15
BOARD_XY_RANGE = 0.015


class MLPPolicy(nn.Module):
    def __init__(self, state_dim=26, action_dim=6, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state):
        return self.net(state)


def setup_indices(m):
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
        "tcp_site": tcp_site, "tip_id": tip_id, "port_id": port_id,
        "board_id": board_id,
        "qids": np.array(qids), "dids": np.array(dids), "aids": np.array(aids),
    }


def get_state(d, idx, progress):
    tcp_pos = d.site_xpos[idx["tcp_site"]].copy()
    tcp_mat = d.site_xmat[idx["tcp_site"]].reshape(3, 3)
    tcp_quat = np.zeros(4)
    mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())
    return np.concatenate([
        d.qpos[idx["qids"]].copy(), d.qvel[idx["dids"]].copy(),
        tcp_pos, tcp_quat,
        d.xpos[idx["tip_id"]].copy(), d.xpos[idx["port_id"]].copy(),
        [progress],
    ])


def solve_ik(m, d, idx, target_tcp_pos, tcp_quat, q_init, d_ik):
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


def build_expert_trajectory(m, d, idx, d_ik):
    """Build approach + descent IK trajectory (the expert)."""
    port_pos = d.xpos[idx["port_id"]].copy()
    tcp_home = d.site_xpos[idx["tcp_site"]].copy()
    tcp_quat = np.zeros(4)
    mujoco.mju_mat2Quat(tcp_quat, d.site_xmat[idx["tcp_site"]].flatten())
    tip_offset = d.xpos[idx["tip_id"]] - d.site_xpos[idx["tcp_site"]]
    tcp_above = port_pos + np.array([0, 0, ABOVE_PORT]) - tip_offset

    N_APP = 80
    approach = np.zeros((N_APP, 6))
    q_prev = HOME.copy()
    for i in range(N_APP):
        alpha = 0.5 * (1 - math.cos(math.pi * i / (N_APP - 1)))
        tcp_i = (1 - alpha) * tcp_home + alpha * tcp_above
        approach[i] = solve_ik(m, d, idx, tcp_i, tcp_quat, q_prev, d_ik)
        q_prev = approach[i]

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


def get_expert_ctrl(t, approach, descent):
    """Look up expert ctrl at time t."""
    if t < APPROACH_TIME:
        n = len(approach)
        frac = t / APPROACH_TIME * (n - 1)
        i0 = min(int(frac), n - 2)
        a = frac - i0
        return (1 - a) * approach[i0] + a * approach[i0 + 1]
    elif t < APPROACH_TIME + DESCENT_TIME:
        n = len(descent)
        frac = (t - APPROACH_TIME) / DESCENT_TIME * (n - 1)
        i0 = min(int(frac), n - 2)
        a = frac - i0
        return (1 - a) * descent[i0] + a * descent[i0 + 1]
    else:
        return descent[-1]


def randomize_board(m, idx, rng, nominal_pos, nominal_quat):
    dx = rng.uniform(-BOARD_XY_RANGE, BOARD_XY_RANGE)
    dy = rng.uniform(-BOARD_XY_RANGE, BOARD_XY_RANGE)
    dyaw = rng.uniform(-BOARD_YAW_RANGE, BOARD_YAW_RANGE)
    m.body_pos[idx["board_id"]] = nominal_pos + np.array([dx, dy, 0])
    cyaw, syaw = math.cos(dyaw / 2), math.sin(dyaw / 2)
    dq = np.array([cyaw, 0, 0, syaw])
    new_quat = np.zeros(4)
    mujoco.mju_mulQuat(new_quat, dq, nominal_quat)
    m.body_quat[idx["board_id"]] = new_quat


def rollout_with_expert_labels(m, d, idx, d_ik, model, s_mean, s_std, a_mean, a_std,
                                approach, descent, beta=0.5):
    """Roll out a mix of policy + expert, label all states with expert actions.

    beta: probability of using expert action (1.0 = pure expert, 0.0 = pure policy)
    Returns: states, expert_actions, success, xy_err, z_rel
    """
    mujoco.mj_resetDataKeyframe(m, d, 0)
    dt = m.opt.timestep
    control_interval = int(1.0 / (CONTROL_HZ * dt))
    max_steps = int(TOTAL_TIME / dt)

    states = []
    expert_actions = []
    ctrl = HOME.copy()
    rng = np.random.default_rng()

    for s in range(max_steps):
        t = s * dt
        if s % control_interval == 0:
            progress = t / TOTAL_TIME
            state = get_state(d, idx, progress)
            expert_ctrl = get_expert_ctrl(t, approach, descent)

            # Policy prediction
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state_norm = (state_t - s_mean) / s_std
            with torch.no_grad():
                action_norm = model(state_norm).squeeze(0)
                policy_ctrl = (action_norm * a_std + a_mean).numpy()

            # Mix: use expert with probability beta
            if rng.random() < beta:
                ctrl = expert_ctrl
            else:
                ctrl = policy_ctrl

            # Always record state with EXPERT label
            states.append(state)
            expert_actions.append(expert_ctrl)

        for i, ai in enumerate(idx["aids"]):
            d.ctrl[ai] = ctrl[i]
        d.ctrl[6] = 0.0
        mujoco.mj_step(m, d)

    tip = d.xpos[idx["tip_id"]]
    port = d.xpos[idx["port_id"]]
    xy_err = np.linalg.norm(tip[:2] - port[:2])
    z_rel = (tip[2] - port[2]) * 1000
    success = xy_err < 0.005 and z_rel < -5.0

    return np.array(states), np.array(expert_actions), success, xy_err, z_rel


def train_on_dataset(all_states, all_actions, model, epochs=500, lr=3e-4,
                     batch_size=256, noise_scale=0.2):
    """Train MLP on aggregated dataset."""
    device = "cpu"
    s_mean_np = all_states.mean(axis=0)
    s_std_np = all_states.std(axis=0) + 1e-6
    a_mean_np = all_actions.mean(axis=0)
    a_std_np = all_actions.std(axis=0) + 1e-6

    s_mean = torch.tensor(s_mean_np, dtype=torch.float32, device=device)
    s_std = torch.tensor(s_std_np, dtype=torch.float32, device=device)
    a_mean = torch.tensor(a_mean_np, dtype=torch.float32, device=device)
    a_std = torch.tensor(a_std_np, dtype=torch.float32, device=device)

    S = torch.tensor(all_states, dtype=torch.float32, device=device)
    A = torch.tensor(all_actions, dtype=torch.float32, device=device)
    S_norm = (S - s_mean) / s_std
    A_norm = (A - a_mean) / a_std

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n = len(S)
    for epoch in range(epochs):
        idx = torch.randperm(n)[:batch_size]
        S_batch = S_norm[idx] + noise_scale * torch.randn(batch_size, S.shape[1])
        pred = model(S_batch)
        loss = nn.functional.mse_loss(pred, A_norm[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        full_loss = nn.functional.mse_loss(model(S_norm), A_norm).item()

    return model, s_mean_np, s_std_np, a_mean_np, a_std_np, full_loss


def evaluate(m, d, idx, model, s_mean, s_std, a_mean, a_std, n_trials=5):
    """Evaluate policy without expert mixing."""
    nominal_pos = m.body_pos[idx["board_id"]].copy()
    nominal_quat = m.body_quat[idx["board_id"]].copy()
    rng = np.random.default_rng(999)

    successes = 0
    xy_errs = []
    for trial in range(n_trials):
        randomize_board(m, idx, rng, nominal_pos, nominal_quat)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)

        dt = m.opt.timestep
        ctrl = HOME.copy()
        control_interval = int(1.0 / (CONTROL_HZ * dt))

        for s in range(int(TOTAL_TIME / dt)):
            t = s * dt
            if s % control_interval == 0:
                progress = t / TOTAL_TIME
                state = get_state(d, idx, progress)
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                state_norm = (state_t - s_mean) / s_std
                with torch.no_grad():
                    action_norm = model(state_norm).squeeze(0)
                    ctrl = (action_norm * a_std + a_mean).numpy()
            for i, ai in enumerate(idx["aids"]):
                d.ctrl[ai] = ctrl[i]
            d.ctrl[6] = 0.0
            mujoco.mj_step(m, d)

        tip = d.xpos[idx["tip_id"]]
        port = d.xpos[idx["port_id"]]
        xy_err = np.linalg.norm(tip[:2] - port[:2])
        z_rel = (tip[2] - port[2]) * 1000
        success = xy_err < 0.005 and z_rel < -5.0
        successes += success
        xy_errs.append(xy_err * 1000)

    m.body_pos[idx["board_id"]] = nominal_pos
    m.body_quat[idx["board_id"]] = nominal_quat

    return successes, n_trials, np.mean(xy_errs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--episodes-per-iter", type=int, default=20)
    parser.add_argument("--epochs-per-iter", type=int, default=500)
    parser.add_argument("--eval-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/outputs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    d_ik = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    idx = setup_indices(m)
    rng = np.random.default_rng(args.seed)

    nominal_pos = m.body_pos[idx["board_id"]].copy()
    nominal_quat = m.body_quat[idx["board_id"]].copy()

    # Load initial dataset
    initial = np.load("data/demos.npz")
    all_states = initial["states"].reshape(-1, 26)  # flatten episodes
    all_actions = initial["actions"].reshape(-1, 6)
    # Remove zero-padded rows
    mask = np.any(all_states != 0, axis=1)
    all_states = all_states[mask]
    all_actions = all_actions[mask]
    print(f"Initial dataset: {len(all_states)} samples")

    # Train initial model
    model = MLPPolicy()
    model, s_mean, s_std, a_mean, a_std, loss = train_on_dataset(
        all_states, all_actions, model, epochs=args.epochs_per_iter
    )
    s_mean_t = torch.tensor(s_mean, dtype=torch.float32)
    s_std_t = torch.tensor(s_std, dtype=torch.float32)
    a_mean_t = torch.tensor(a_mean, dtype=torch.float32)
    a_std_t = torch.tensor(a_std, dtype=torch.float32)

    print(f"Initial train loss: {loss:.6f}")
    succ, total, avg_xy = evaluate(m, d, idx, model, s_mean_t, s_std_t, a_mean_t, a_std_t, args.eval_trials)
    print(f"Initial eval: {succ}/{total} success, avg_xy={avg_xy:.1f}mm\n")

    for iteration in range(args.iterations):
        # Anneal beta: start with more expert, shift to more policy
        beta = max(0.1, 1.0 - iteration * 0.2)
        print(f"=== DAgger iteration {iteration+1}/{args.iterations} (beta={beta:.1f}) ===")

        new_states = []
        new_actions = []

        for ep in range(args.episodes_per_iter):
            randomize_board(m, idx, rng, nominal_pos, nominal_quat)
            mujoco.mj_resetDataKeyframe(m, d, 0)
            mujoco.mj_forward(m, d)

            approach, descent = build_expert_trajectory(m, d, idx, d_ik)

            states, expert_actions, success, xy_err, z_rel = rollout_with_expert_labels(
                m, d, idx, d_ik, model, s_mean_t, s_std_t, a_mean_t, a_std_t,
                approach, descent, beta=beta,
            )
            new_states.append(states)
            new_actions.append(expert_actions)

        # Aggregate
        new_s = np.concatenate(new_states, axis=0)
        new_a = np.concatenate(new_actions, axis=0)
        all_states = np.concatenate([all_states, new_s], axis=0)
        all_actions = np.concatenate([all_actions, new_a], axis=0)
        print(f"  Dataset size: {len(all_states)} samples (+{len(new_s)})")

        # Retrain from scratch on full dataset
        model = MLPPolicy()
        model, s_mean, s_std, a_mean, a_std, loss = train_on_dataset(
            all_states, all_actions, model, epochs=args.epochs_per_iter
        )
        s_mean_t = torch.tensor(s_mean, dtype=torch.float32)
        s_std_t = torch.tensor(s_std, dtype=torch.float32)
        a_mean_t = torch.tensor(a_mean, dtype=torch.float32)
        a_std_t = torch.tensor(a_std, dtype=torch.float32)
        print(f"  Train loss: {loss:.6f}")

        # Evaluate
        succ, total, avg_xy = evaluate(m, d, idx, model, s_mean_t, s_std_t, a_mean_t, a_std_t, args.eval_trials)
        print(f"  Eval: {succ}/{total} success, avg_xy={avg_xy:.1f}mm\n")

        # Save checkpoint
        torch.save(model.state_dict(), out_dir / f"dagger_iter{iteration+1}.pt")
        np.savez(out_dir / "norm_stats.npz",
                 state_mean=s_mean, state_std=s_std,
                 action_mean=a_mean, action_std=a_std)

    # Save final
    torch.save(model.state_dict(), out_dir / "mlp_policy.pt")
    m.body_pos[idx["board_id"]] = nominal_pos
    m.body_quat[idx["board_id"]] = nominal_quat
    print("Done. Final model saved to", out_dir / "mlp_policy.pt")


if __name__ == "__main__":
    main()
