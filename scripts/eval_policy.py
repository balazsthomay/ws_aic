#!/usr/bin/env python3
"""Evaluate a trained policy in MuJoCo simulation.

Usage:
    # MLP policy (headless)
    .venv/bin/python3 scripts/eval_policy.py --arch mlp --weights data/outputs/mlp_policy.pt

    # With vision (no ground truth port position)
    .venv/bin/python3 scripts/eval_policy.py --arch mlp --weights data/outputs/mlp_policy_best.pt --vision

    # With viewer
    .venv/bin/python3 scripts/eval_policy.py --arch mlp --weights data/outputs/mlp_policy.pt --viewer

    # Multiple trials with randomized board
    .venv/bin/python3 scripts/eval_policy.py --arch mlp --weights data/outputs/mlp_policy.pt --trials 10
"""

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src/aic/aic_utils/aic_mujoco"))

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


class FeatureLocalizer(nn.Module):
    """Small MLP: green pixel features + proprioception → port 3D position."""

    def __init__(self, visual_dim: int = 15, proprio_dim: int = 13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_dim + proprio_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, visual: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([visual, proprio], dim=-1))


CAMERA_NAMES = ["left_camera", "center_camera", "right_camera"]


def extract_green_features(img: np.ndarray) -> np.ndarray:
    """Extract green (NIC card) pixel statistics from a camera image."""
    H, W = img.shape[:2]
    mask = (img[:, :, 1] > 100) & (img[:, :, 0] < 100) & (img[:, :, 2] < 100)
    n_pixels = mask.sum()
    if n_pixels < 10:
        return np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    ys, xs = np.where(mask)
    return np.array([
        xs.mean() / W, ys.mean() / H,
        np.log1p(n_pixels) / np.log1p(H * W),
        xs.std() / W if n_pixels > 1 else 0.0,
        ys.std() / H if n_pixels > 1 else 0.0,
    ], dtype=np.float32)


def load_localizer(weights_path, stats_path):
    """Load feature-based port localizer and normalization stats."""
    stats = np.load(str(stats_path))
    loc_model = FeatureLocalizer()
    loc_model.load_state_dict(
        torch.load(str(weights_path), map_location="cpu", weights_only=True)
    )
    loc_model.eval()
    return (
        loc_model,
        torch.tensor(stats["proprio_mean"], dtype=torch.float32),
        torch.tensor(stats["proprio_std"], dtype=torch.float32),
        torch.tensor(stats["target_mean"], dtype=torch.float32),
        torch.tensor(stats["target_std"], dtype=torch.float32),
    )


class PortPredictor:
    """Predict port position using geometric back-projection from camera images.

    Uses green (NIC card) pixel detection, ray-plane intersection, and
    temporal smoothing (EMA) for stable 3D port localization.
    """

    # Empirical offset: NIC card centroid → SFP port (meters)
    # Calibrated from policy rollouts at t=11s (NOT expert demos)
    NIC_TO_PORT_OFFSET = np.array([-0.0095, 0.0065])

    def __init__(self, observer, model, nominal_port_pos):
        self.observer = observer
        self.model = model
        self.nominal_port_pos = nominal_port_pos.copy()
        self.H = observer.img_h
        self.W = observer.img_w
        self.history = []  # list of XY predictions

        # Pre-compute camera intrinsics
        self.cam_fy = {}
        for name in CAMERA_NAMES:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            fovy_rad = np.radians(model.cam_fovy[cid])
            self.cam_fy[name] = 0.5 * self.H / np.tan(fovy_rad / 2)

    def reset(self):
        self.history = []

    def predict(self, d, idx):
        """Render cameras, back-project green centroids, return smoothed port pos."""
        obs = self.observer.get_observation()
        z_target = self.nominal_port_pos[2]

        points = []
        weights = []

        for name in CAMERA_NAMES:
            img = obs.images[name]
            mask = (img[:, :, 1] > 100) & (img[:, :, 0] < 100) & (img[:, :, 2] < 100)
            n_pixels = mask.sum()
            if n_pixels < 20:
                continue

            ys, xs = np.where(mask)
            cu, cv = xs.mean(), ys.mean()

            # Ray in camera frame
            fy = self.cam_fy[name]
            ray_cam = np.array([(cu - self.W / 2) / fy, -(cv - self.H / 2) / fy, -1.0])

            # Transform to world frame
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            R = d.cam_xmat[cid].reshape(3, 3)
            cam_pos = d.cam_xpos[cid]
            ray_world = R @ ray_cam
            ray_world /= np.linalg.norm(ray_world)

            # Ray-plane intersection at z = z_target
            if abs(ray_world[2]) < 1e-6:
                continue
            t = (z_target - cam_pos[2]) / ray_world[2]
            if t < 0:
                continue

            pt = cam_pos + t * ray_world
            points.append(pt[:2])
            weights.append(min(n_pixels / 500.0, 1.0))

        if not points:
            # No green detected — use running median or nominal
            if self.history:
                h = np.array(self.history)
                med_xy = np.median(h, axis=0)
                return np.array([med_xy[0], med_xy[1], z_target])
            return self.nominal_port_pos.copy()

        # Weighted average of back-projected points + offset correction
        points = np.array(points)
        weights = np.array(weights)
        weights /= weights.sum()
        avg_xy = (points * weights[:, None]).sum(axis=0)
        avg_xy += self.NIC_TO_PORT_OFFSET

        self.history.append(avg_xy.copy())

        # Use running median (robust to outliers)
        h = np.array(self.history)
        med_xy = np.median(h, axis=0)

        return np.array([med_xy[0], med_xy[1], z_target])


class LearnedPortPredictor:
    """Predict port position using learned feature localizer with EMA smoothing.

    Includes prediction clamping (port can't move more than MAX_OFFSET from
    the target mean in training data) to prevent catastrophic extrapolation.
    """

    MAX_OFFSET = 0.025  # 25mm max deviation from training mean

    def __init__(self, observer, loc_model, p_mean, p_std, t_mean, t_std, alpha=0.3):
        self.observer = observer
        self.loc_model = loc_model
        self.p_mean = p_mean
        self.p_std = p_std
        self.t_mean = t_mean
        self.t_std = t_std
        self.alpha = alpha
        self.ema = None
        # Nominal port position (training mean) for clamping
        self.nominal = t_mean.numpy().copy()

    def reset(self):
        self.ema = None

    def predict(self, d, idx):
        obs = self.observer.get_observation()

        feats = [extract_green_features(obs.images[n]) for n in CAMERA_NAMES]
        visual = np.concatenate(feats)
        visual_t = torch.tensor(visual, dtype=torch.float32).unsqueeze(0)

        # Check if any camera actually sees green pixels (area feature > 0)
        has_green = any(f[2] > 0.01 for f in feats)

        tcp_pos = d.site_xpos[idx["tcp_site"]].copy()
        tcp_mat = d.site_xmat[idx["tcp_site"]].reshape(3, 3)
        tcp_quat = np.zeros(4)
        mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())
        proprio_raw = np.concatenate([d.qpos[idx["qids"]].copy(), tcp_pos, tcp_quat])
        proprio = (torch.tensor(proprio_raw, dtype=torch.float32) - self.p_mean) / self.p_std
        proprio = proprio.unsqueeze(0)

        with torch.no_grad():
            pred_norm = self.loc_model(visual_t, proprio).squeeze(0)
            pred = (pred_norm * self.t_std + self.t_mean).numpy()

        # Clamp prediction to reasonable range around nominal
        pred[:2] = np.clip(pred[:2],
                           self.nominal[:2] - self.MAX_OFFSET,
                           self.nominal[:2] + self.MAX_OFFSET)
        pred[2] = self.nominal[2]  # Z is constant

        # Only update EMA when green pixels are visible (confident detection)
        if has_green:
            if self.ema is None:
                self.ema = pred.copy()
            else:
                self.ema = self.alpha * pred + (1 - self.alpha) * self.ema

        # Return EMA if available, otherwise nominal
        if self.ema is not None:
            return self.ema.copy()
        return self.nominal.copy()


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


def get_state(d, idx, progress=0.0, port_pos_override=None):
    """Extract 26D state vector from simulation data.

    If port_pos_override is given, uses it instead of ground truth port position.
    """
    tcp_pos = d.site_xpos[idx["tcp_site"]].copy()
    tcp_mat = d.site_xmat[idx["tcp_site"]].reshape(3, 3)
    tcp_quat = np.zeros(4)
    mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())

    port_pos = port_pos_override if port_pos_override is not None else d.xpos[idx["port_id"]].copy()

    return np.concatenate([
        d.qpos[idx["qids"]].copy(),
        d.qvel[idx["dids"]].copy(),
        tcp_pos,
        tcp_quat,
        d.xpos[idx["tip_id"]].copy(),
        port_pos,
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


def run_trial(m, d, idx, model, s_mean, s_std, a_mean, a_std, arch,
              port_predictor=None):
    """Run one evaluation trial with the learned policy.

    If port_predictor is provided, uses vision-predicted port position
    instead of ground truth.
    """
    mujoco.mj_resetDataKeyframe(m, d, 0)
    dt = m.opt.timestep
    control_interval = int(1.0 / (CONTROL_HZ * dt))
    max_steps = int(MAX_TIME / dt)

    ctrl = HOME.copy()

    if port_predictor is not None:
        port_predictor.reset()

    # For diffusion: action chunk buffer
    action_chunk = None
    chunk_idx = 0

    for s in range(max_steps):
        if s % control_interval == 0:
            progress = (s * dt) / MAX_TIME

            # Predict port position from cameras or use ground truth
            port_pos_override = None
            if port_predictor is not None:
                mujoco.mj_forward(m, d)
                port_pos_override = port_predictor.predict(d, idx)

            state = get_state(d, idx, progress, port_pos_override)
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
    parser.add_argument("--vision", action="store_true",
                        help="Use vision instead of ground truth port position")
    parser.add_argument("--vision-mode", default="learned", choices=["learned", "geometric"],
                        help="Vision method: 'learned' (feature MLP) or 'geometric' (back-projection)")
    parser.add_argument("--localizer", default="data/outputs/port_localizer.pt",
                        help="Port localizer weights (used with --vision --vision-mode learned)")
    parser.add_argument("--localizer-stats", default="data/outputs/port_localizer_stats.npz",
                        help="Port localizer normalization stats")
    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    idx = setup_indices(m)
    model, s_mean, s_std, a_mean, a_std = load_policy(args.arch, args.weights, args.stats)

    # Vision setup
    port_predictor = None
    if args.vision:
        from mujoco_obs import MuJoCoObserver
        observer = MuJoCoObserver(m, d, image_scale=0.25)
        if args.vision_mode == "geometric":
            nominal_port_pos = d.xpos[idx["port_id"]].copy()
            port_predictor = PortPredictor(observer, m, nominal_port_pos)
            print("Vision mode: geometric back-projection with EMA smoothing")
        else:
            loc_model, p_mean, p_std, t_mean, t_std = load_localizer(
                args.localizer, args.localizer_stats
            )
            port_predictor = LearnedPortPredictor(
                observer, loc_model, p_mean, p_std, t_mean, t_std
            )
            print("Vision mode: learned feature localizer with EMA smoothing")

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
        control_interval = int(1.0 / (CONTROL_HZ * m.opt.timestep))
        step_count = 0
        action_chunk = None
        chunk_idx = 0

        def controller(model_mj, data):
            nonlocal ctrl, step_count, action_chunk, chunk_idx

            if step_count % control_interval == 0 and data.time < MAX_TIME:
                progress = data.time / MAX_TIME

                port_pos_override = None
                if port_predictor is not None:
                    mujoco.mj_forward(m, data)
                    port_pos_override = port_predictor.predict(data, idx)

                state = get_state(data, idx, progress, port_pos_override)
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                state_norm = (state_t - s_mean) / s_std
                with torch.no_grad():
                    if args.arch == "mlp":
                        action_norm = model(state_norm).squeeze(0)
                        ctrl = (action_norm * a_std + a_mean).numpy()
                    elif args.arch == "diffusion":
                        if action_chunk is None or chunk_idx >= model.chunk_size:
                            action_chunk = model.sample(state_norm).squeeze(0)
                            chunk_idx = 0
                        action_norm = action_chunk[chunk_idx]
                        ctrl = (action_norm * a_std + a_mean).numpy()
                        chunk_idx += 1

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
                m, d, idx, model, s_mean, s_std, a_mean, a_std, args.arch,
                port_predictor,
            )
            successes += success
            status = "OK" if success else "FAIL"
            tag = " [vision]" if args.vision else ""
            print(
                f"  [{trial+1}/{args.trials}] {status}{tag} "
                f"xy={xy_err*1000:.1f}mm z={z_rel:.1f}mm"
            )

        # Restore
        m.body_pos[idx["board_id"]] = nominal_pos
        m.body_quat[idx["board_id"]] = nominal_quat

        rate = successes / args.trials
        mode = "vision" if args.vision else "ground truth"
        print(f"\nSuccess ({mode}): {successes}/{args.trials} ({rate:.0%})")

    if port_predictor is not None:
        port_predictor.observer.close()


if __name__ == "__main__":
    main()
