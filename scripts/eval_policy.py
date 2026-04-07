#!/usr/bin/env python3
"""Evaluate a trained policy in MuJoCo simulation.

Usage:
    # MLP policy (headless)
    .venv/bin/python3 scripts/eval_policy.py --arch mlp --weights data/outputs/mlp_policy.pt

    # Diffusion policy (headless)
    .venv/bin/python3 scripts/eval_policy.py --arch diffusion --weights data/outputs/diffusion_policy.pt

    # With viewer
    .venv/bin/python3 scripts/eval_policy.py --arch mlp --weights data/outputs/mlp_policy.pt --viewer

    # Multiple trials with randomized board
    .venv/bin/python3 scripts/eval_policy.py --arch mlp --weights data/outputs/mlp_policy.pt --trials 10
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

SCENE = Path(__file__).resolve().parent.parent / "src/aic/aic_utils/aic_mujoco/mjcf/scene.xml"

HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

BOARD_YAW_RANGE = 0.15
BOARD_XY_RANGE = 0.015
CONTROL_HZ = 20
MAX_TIME = 14.0  # must match demo collection: approach(5) + descent(8) + hold(1)


# --- Model definitions (must match train_modal.py) ---

class MLPPolicy(nn.Module):
    def __init__(self, state_dim=26, action_dim=6, hidden=256):
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

    def forward(self, state):
        return self.net(state)


class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim=26, action_dim=6, chunk_size=16,
                 hidden=512, n_diffusion_steps=50):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_steps = n_diffusion_steps

        action_input = action_dim * chunk_size
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.time_embed = nn.Embedding(n_diffusion_steps, hidden)
        self.denoiser = nn.Sequential(
            nn.Linear(action_input + hidden + hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_input),
        )

        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward_denoise(self, noisy_actions, state, timestep):
        B = noisy_actions.shape[0]
        flat_actions = noisy_actions.reshape(B, -1)
        state_feat = self.state_encoder(state)
        time_feat = self.time_embed(timestep)
        x = torch.cat([flat_actions, state_feat, time_feat], dim=-1)
        pred = self.denoiser(x)
        return pred.reshape(B, self.chunk_size, self.action_dim)

    @torch.no_grad()
    def sample(self, state):
        B = state.shape[0]
        device = state.device
        x = torch.randn(B, self.chunk_size, self.action_dim, device=device)
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((B,), t, dtype=torch.long, device=device)
            pred_noise = self.forward_denoise(x, state, t_batch)
            alpha = self.alphas[t]
            alpha_cum = self.alphas_cumprod[t]
            beta = self.betas[t]
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cum)) * pred_noise
            )
            if t > 0:
                x += torch.sqrt(beta) * torch.randn_like(x)
        return x


def load_policy(arch, weights_path, stats_path):
    """Load trained policy and normalization stats."""
    stats = np.load(str(stats_path))
    s_mean = torch.tensor(stats["state_mean"], dtype=torch.float32)
    s_std = torch.tensor(stats["state_std"], dtype=torch.float32)
    a_mean = torch.tensor(stats["action_mean"], dtype=torch.float32)
    a_std = torch.tensor(stats["action_std"], dtype=torch.float32)

    if arch == "mlp":
        model = MLPPolicy()
    elif arch == "diffusion":
        model = DiffusionPolicy()
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.load_state_dict(torch.load(str(weights_path), map_location="cpu", weights_only=True))
    model.eval()

    return model, s_mean, s_std, a_mean, a_std


def get_state(d, idx, progress=0.0):
    """Extract 26D state vector from simulation data."""
    tcp_pos = d.site_xpos[idx["tcp_site"]].copy()
    tcp_mat = d.site_xmat[idx["tcp_site"]].reshape(3, 3)
    tcp_quat = np.zeros(4)
    mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())

    return np.concatenate([
        d.qpos[idx["qids"]].copy(),
        d.qvel[idx["dids"]].copy(),
        tcp_pos,
        tcp_quat,
        d.xpos[idx["tip_id"]].copy(),
        d.xpos[idx["port_id"]].copy(),
        [progress],
    ])


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
    return dx, dy, dyaw


def run_trial(m, d, idx, model, s_mean, s_std, a_mean, a_std, arch):
    """Run one evaluation trial with the learned policy."""
    mujoco.mj_resetDataKeyframe(m, d, 0)
    dt = m.opt.timestep
    control_interval = int(1.0 / (CONTROL_HZ * dt))
    max_steps = int(MAX_TIME / dt)

    ctrl = HOME.copy()

    # For diffusion: action chunk buffer
    action_chunk = None
    chunk_idx = 0

    for s in range(max_steps):
        if s % control_interval == 0:
            progress = (s * dt) / MAX_TIME
            state = get_state(d, idx, progress)
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state_norm = (state_t - s_mean) / s_std

            with torch.no_grad():
                if arch == "mlp":
                    action_norm = model(state_norm).squeeze(0)
                    action = (action_norm * a_std + a_mean).numpy()
                elif arch == "diffusion":
                    if action_chunk is None or chunk_idx >= model.chunk_size:
                        action_chunk = model.sample(state_norm).squeeze(0)
                        chunk_idx = 0
                    action_norm = action_chunk[chunk_idx]
                    action = (action_norm * a_std + a_mean).numpy()
                    chunk_idx += 1

            ctrl = action

        for i, ai in enumerate(idx["aids"]):
            d.ctrl[ai] = ctrl[i]
        d.ctrl[6] = 0.0
        mujoco.mj_step(m, d)

    tip = d.xpos[idx["tip_id"]]
    port = d.xpos[idx["port_id"]]
    xy_err = np.linalg.norm(tip[:2] - port[:2])
    z_rel = (tip[2] - port[2]) * 1000
    success = xy_err < 0.005 and z_rel < -5.0
    return success, xy_err, z_rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["mlp", "diffusion"])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--stats", default="data/outputs/norm_stats.npz")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    idx = setup_indices(m)
    model, s_mean, s_std, a_mean, a_std = load_policy(args.arch, args.weights, args.stats)

    nominal_pos = m.body_pos[idx["board_id"]].copy()
    nominal_quat = m.body_quat[idx["board_id"]].copy()
    rng = np.random.default_rng(args.seed)

    if args.viewer:
        # Single trial in viewer
        if args.trials > 1:
            randomize_board(m, idx, rng, nominal_pos, nominal_quat)

        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)

        ctrl = HOME.copy()
        max_step = 1.5 * m.opt.timestep
        control_interval = int(1.0 / (CONTROL_HZ * m.opt.timestep))
        step_count = 0
        action_chunk = None
        chunk_idx = 0

        def controller(model_mj, data):
            nonlocal ctrl, step_count, action_chunk, chunk_idx
            ctrl_target = ctrl

            if step_count % control_interval == 0 and data.time < MAX_TIME:
                progress = data.time / MAX_TIME
                state = get_state(data, idx, progress)
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                state_norm = (state_t - s_mean) / s_std
                with torch.no_grad():
                    if args.arch == "mlp":
                        action_norm = model(state_norm).squeeze(0)
                        action = (action_norm * a_std + a_mean).numpy()
                    elif args.arch == "diffusion":
                        if action_chunk is None or chunk_idx >= model.chunk_size:
                            action_chunk = model.sample(state_norm).squeeze(0)
                            chunk_idx = 0
                        action_norm = action_chunk[chunk_idx]
                        action = (action_norm * a_std + a_mean).numpy()
                        chunk_idx += 1
                ctrl_target = action

            diff = ctrl_target - ctrl
            ctrl = ctrl + np.clip(diff, -max_step, max_step)

            for i, ai in enumerate(idx["aids"]):
                data.ctrl[ai] = ctrl[i]
            data.ctrl[6] = 0.0
            step_count += 1

        mujoco.set_mjcb_control(controller)
        mujoco.viewer.launch(m, d)
    else:
        # Headless trials
        successes = 0
        for trial in range(args.trials):
            if args.trials > 1:
                dx, dy, dyaw = randomize_board(m, idx, rng, nominal_pos, nominal_quat)
            else:
                dx = dy = dyaw = 0.0

            mujoco.mj_resetDataKeyframe(m, d, 0)
            mujoco.mj_forward(m, d)

            success, xy_err, z_rel = run_trial(
                m, d, idx, model, s_mean, s_std, a_mean, a_std, args.arch
            )
            successes += success
            status = "OK" if success else "FAIL"
            print(
                f"  [{trial+1}/{args.trials}] {status} "
                f"xy={xy_err*1000:.1f}mm z={z_rel:.1f}mm"
            )

        # Restore
        m.body_pos[idx["board_id"]] = nominal_pos
        m.body_quat[idx["board_id"]] = nominal_quat

        rate = successes / args.trials
        print(f"\nSuccess: {successes}/{args.trials} ({rate:.0%})")


if __name__ == "__main__":
    main()
