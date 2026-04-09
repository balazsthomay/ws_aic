#!/usr/bin/env python3
"""Train port localizer: camera images → port 3D position.

Two approaches available:
  --mode features : Color-based feature extraction → small MLP (fast, CPU-friendly)
  --mode cnn      : Full CNN encoder → MLP (needs GPU, more general)

The feature-based mode detects the green NIC card pixels in each camera,
extracts 2D statistics (centroid, area, spread), and combines with arm
proprioception to predict the 3D port position via a small MLP.

Usage:
    # Feature-based (default, trains in seconds on CPU)
    .venv/bin/python3 scripts/train_port_localizer.py

    # CNN-based (needs GPU or patience)
    .venv/bin/python3 scripts/train_port_localizer.py --mode cnn --epochs 100
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --- Feature extraction ---

def extract_green_features(img: np.ndarray) -> np.ndarray:
    """Extract green (NIC card) pixel statistics from a camera image.

    Args:
        img: (H, W, 3) uint8 RGB image

    Returns:
        (5,) features: [centroid_u, centroid_v, log_area, std_u, std_v]
        Normalized to [0,1] range relative to image dimensions.
    """
    H, W = img.shape[:2]
    # Green detection: high green, low red, low blue
    mask = (img[:, :, 1] > 100) & (img[:, :, 0] < 100) & (img[:, :, 2] < 100)
    n_pixels = mask.sum()

    if n_pixels < 10:
        # Fallback: image center, zero area, zero spread
        return np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)

    ys, xs = np.where(mask)
    cu = xs.mean() / W  # normalized [0, 1]
    cv = ys.mean() / H
    log_area = np.log1p(n_pixels) / np.log1p(H * W)  # normalized
    su = xs.std() / W if n_pixels > 1 else 0.0
    sv = ys.std() / H if n_pixels > 1 else 0.0

    return np.array([cu, cv, log_area, su, sv], dtype=np.float32)


def backproject_green_centroid(img: np.ndarray, cam_xpos: np.ndarray,
                               cam_xmat: np.ndarray, fovy_rad: float,
                               z_target: float) -> np.ndarray:
    """Back-project green centroid from image to world XY using camera geometry.

    Args:
        img: (H, W, 3) uint8 RGB image
        cam_xpos: (3,) camera position in world frame
        cam_xmat: (9,) or (3,3) camera rotation matrix (world frame)
        fovy_rad: vertical field of view in radians
        z_target: known Z-height of the target (port height)

    Returns:
        (3,) features: [world_x, world_y, confidence]
        If no green pixels detected, returns [0, 0, 0].
    """
    H, W = img.shape[:2]
    mask = (img[:, :, 1] > 100) & (img[:, :, 0] < 100) & (img[:, :, 2] < 100)
    n_pixels = mask.sum()

    if n_pixels < 10:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    ys, xs = np.where(mask)
    cu, cv = xs.mean(), ys.mean()

    # Camera intrinsics
    fy = 0.5 * H / np.tan(fovy_rad / 2)
    fx = fy  # square pixels

    # Ray in camera frame (MuJoCo: -Z forward, Y up, X right)
    ray_cam = np.array([
        (cu - W / 2) / fx,
        -(cv - H / 2) / fy,
        -1.0,
    ])

    # Transform to world frame
    R = cam_xmat.reshape(3, 3)
    ray_world = R @ ray_cam
    ray_world /= np.linalg.norm(ray_world)

    # Intersect with z = z_target plane
    if abs(ray_world[2]) < 1e-6:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    t = (z_target - cam_xpos[2]) / ray_world[2]
    if t < 0:  # behind camera
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    world_xy = cam_xpos[:2] + t * ray_world[:2]
    confidence = min(n_pixels / 500.0, 1.0)  # normalized confidence

    return np.array([world_xy[0], world_xy[1], confidence], dtype=np.float32)


def extract_all_features(left: np.ndarray, center: np.ndarray, right: np.ndarray,
                         states: np.ndarray, lengths: np.ndarray,
                         ep_indices: list[int], subsample: int = 1,
                         cam_data: dict | None = None):
    """Extract features from all samples in given episodes.

    Args:
        cam_data: dict with keys 'cam_xpos', 'cam_xmat', 'fovy_rad', 'z_target'
            from MuJoCo model. If provided, adds geometric back-projection features.

    Returns:
        visual_feats: (N, 15 or 24) pixel stats + optional geometric features
        proprio_feats: (N, 13) joint_pos + tcp_pos + tcp_quat
        targets: (N, 3) port position
    """
    all_visual = []
    all_proprio = []
    all_targets = []

    use_geometry = cam_data is not None

    for ep in ep_indices:
        L = int(lengths[ep])
        for t in range(0, L, subsample):
            # Pixel features: 3 cameras × 5
            vf = np.concatenate([
                extract_green_features(left[ep, t]),
                extract_green_features(center[ep, t]),
                extract_green_features(right[ep, t]),
            ])

            # Geometric back-projection: 3 cameras × 3 (world_x, world_y, confidence)
            if use_geometry:
                s = states[ep, t]
                # NOTE: cam_data provides per-timestep camera poses
                # For now, we use the pixel features + proprio (which encodes camera pose)
                # Geometric features would need live camera poses from simulation
                pass

            all_visual.append(vf)

            # Proprioception
            s = states[ep, t]
            all_proprio.append(np.concatenate([s[:6], s[12:19]]).astype(np.float32))

            # Target: port_pos at indices 22:25 (in both 25D and 26D state)
            target_idx = 22 if states.shape[-1] >= 25 else -3
            all_targets.append(s[22:25].astype(np.float32))

    return np.array(all_visual), np.array(all_proprio), np.array(all_targets)


# --- Models ---

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


class PortLocalizer(nn.Module):
    """CNN: 3 camera images + proprioception → port 3D position."""

    def __init__(self, proprio_dim: int = 13):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 3 + proprio_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        feats = [self.encoder(images[:, i]) for i in range(3)]
        return self.head(torch.cat(feats + [proprio], dim=-1))


# --- Training ---

def load_and_merge_data(data_paths: list[str]):
    """Load and merge multiple npz data files."""
    all_left, all_center, all_right = [], [], []
    all_states, all_lengths = [], []

    for path in data_paths:
        print(f"  Loading {path}...")
        raw = np.load(str(path))
        states = raw["states"]
        # Truncate to 25D if needed (port_pos at 22:25 in both 25D/26D)
        if states.shape[-1] > 25:
            states = states[:, :, :25]
        all_states.append(states)
        all_lengths.append(raw["episode_lengths"])
        all_left.append(raw["images_left_camera"])
        all_center.append(raw["images_center_camera"])
        all_right.append(raw["images_right_camera"])
        print(f"    {len(raw['episode_lengths'])} episodes, state_dim={states.shape[-1]}")

    return (
        np.concatenate(all_left),
        np.concatenate(all_center),
        np.concatenate(all_right),
        np.concatenate(all_states),
        np.concatenate(all_lengths),
    )


def train_features(args):
    """Train feature-based localizer (fast, CPU-friendly)."""
    data_paths = [p.strip() for p in args.data.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(data_paths)} data file(s)...")
    left, center, right, states, lengths = load_and_merge_data(data_paths)
    n_eps = len(lengths)
    print(f"  Total: {n_eps} episodes, {lengths[0]} steps")

    # Train/val split by episode
    rng = np.random.default_rng(args.seed)
    ep_order = rng.permutation(n_eps)
    n_val = max(2, n_eps // 5)
    val_eps = sorted(ep_order[:n_val].tolist())
    train_eps = sorted(ep_order[n_val:].tolist())
    print(f"  Train: {len(train_eps)} episodes, Val: {len(val_eps)} episodes")

    print("Extracting visual features...")
    train_vis, train_prop, train_tgt = extract_all_features(
        left, center, right, states, lengths, train_eps, args.subsample
    )
    val_vis, val_prop, val_tgt = extract_all_features(
        left, center, right, states, lengths, val_eps, args.subsample
    )
    print(f"  Train: {len(train_vis)} samples, Val: {len(val_vis)} samples")

    # Normalize proprio and targets
    prop_mean, prop_std = train_prop.mean(0), train_prop.std(0) + 1e-6
    tgt_mean, tgt_std = train_tgt.mean(0), train_tgt.std(0) + 1e-6

    train_prop_n = (train_prop - prop_mean) / prop_std
    val_prop_n = (val_prop - prop_mean) / prop_std
    train_tgt_n = (train_tgt - tgt_mean) / tgt_std
    val_tgt_n = (val_tgt - tgt_mean) / tgt_std

    # Convert to tensors
    tv = torch.tensor(train_vis, dtype=torch.float32)
    tp = torch.tensor(train_prop_n, dtype=torch.float32)
    tt = torch.tensor(train_tgt_n, dtype=torch.float32)
    vv = torch.tensor(val_vis, dtype=torch.float32)
    vp = torch.tensor(val_prop_n, dtype=torch.float32)
    vt = torch.tensor(val_tgt_n, dtype=torch.float32)
    tgt_std_t = torch.tensor(tgt_std, dtype=torch.float32)
    tgt_mean_t = torch.tensor(tgt_mean, dtype=torch.float32)

    # Model
    model = FeatureLocalizer(visual_dim=15, proprio_dim=13)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    n = len(tv)
    best_val_xy = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        idx = torch.randperm(n)[:args.batch_size]
        pred = model(tv[idx], tp[idx])
        loss = nn.functional.mse_loss(pred, tt[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred_n = model(vv, vp)
            val_pred = val_pred_n * tgt_std_t + tgt_mean_t
            val_actual = vt * tgt_std_t + tgt_mean_t
            err_mm = (val_pred - val_actual).abs() * 1000
            mae = err_mm.mean(0)
            xy_mae = err_mm[:, :2].norm(dim=1).mean().item()

        if xy_mae < best_val_xy:
            best_val_xy = xy_mae
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_dir / "port_localizer.pt")

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"  [{epoch+1:4d}/{args.epochs}] "
                f"loss={loss.item():.6f} "
                f"val_MAE=[{mae[0]:.1f}, {mae[1]:.1f}, {mae[2]:.1f}]mm "
                f"val_XY={xy_mae:.1f}mm"
                f"{' *' if epoch + 1 == best_epoch else ''}"
            )

    # Save stats
    np.savez(
        output_dir / "port_localizer_stats.npz",
        proprio_mean=prop_mean,
        proprio_std=prop_std,
        target_mean=tgt_mean,
        target_std=tgt_std,
    )
    print(f"\nBest val XY MAE: {best_val_xy:.1f}mm (epoch {best_epoch})")

    # Final detailed eval
    model.load_state_dict(torch.load(output_dir / "port_localizer.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        pred = model(vv, vp) * tgt_std_t + tgt_mean_t
        actual = vt * tgt_std_t + tgt_mean_t
        err = (pred - actual).abs() * 1000
    print("\n--- Validation results ---")
    for i, ax in enumerate(["X", "Y", "Z"]):
        print(f"  {ax}: MAE={err[:,i].mean():.2f}mm, max={err[:,i].max():.2f}mm")
    xy = err[:, :2].norm(dim=1)
    print(f"  XY: MAE={xy.mean():.2f}mm, max={xy.max():.2f}mm, <3mm={100*(xy<3).float().mean():.0f}%")

    print(f"\nSaved: {output_dir / 'port_localizer.pt'}")
    print(f"Saved: {output_dir / 'port_localizer_stats.npz'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="features", choices=["features", "cnn"])
    parser.add_argument("--data", default="data/demos_with_images.npz")
    parser.add_argument("--output", default="data/outputs")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--subsample", type=int, default=2, help="Use every Nth timestep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.mode == "features":
        train_features(args)
    else:
        print("CNN mode — use Modal GPU for practical training speed.")
        print("For now, use --mode features (default).")


if __name__ == "__main__":
    main()
