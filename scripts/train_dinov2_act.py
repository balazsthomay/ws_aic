#!/usr/bin/env python3
"""Train a DINOv2 + ACT (Action Chunking Transformer) policy for SFP insertion.

Pipeline:
  1. Cache: encode every wrist-camera image through frozen DINOv2-S/14 (CLS token)
     into data/outputs/dinov2_feats_30.npz (~40 MB).
  2. Train: ACT head over (3 vision tokens, 1 state token) → K=16 action chunk.
     L1 loss on normalized actions. Trains in ~5 min on a single GPU.
  3. Save: act_policy.pt + norm_stats.npz to data/outputs/.

Why DINOv2-frozen + ACT:
  - Renderer-agnostic features (DINOv2 trained on 142M internet images,
    no MuJoCo-vs-Gazebo bias) — replaces the brittle green-pixel detector.
  - ACT's chunked decoder smooths out vision noise across multiple steps.
  - Small enough for CPU inference inside the ROS2 model container.

Usage:
    # Step 1: cache features (one-time, MPS or CPU)
    .venv/bin/python3 scripts/train_dinov2_act.py cache

    # Step 2: train ACT head on cached features
    .venv/bin/python3 scripts/train_dinov2_act.py train --epochs 5000

    # Or do both in one shot:
    .venv/bin/python3 scripts/train_dinov2_act.py all --epochs 5000
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import os as _os_top
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(_os_top.environ.get(
    "DATA_PATH", str(ROOT / "data" / "demos_with_images_30.npz")
))
FEATS_PATH = Path(_os_top.environ.get(
    "FEATS_PATH", str(ROOT / "data" / "outputs" / "dinov2_feats_30.npz")
))
PATCHES_PATH = Path(_os_top.environ.get(
    "PATCHES_PATH", str(ROOT / "data" / "outputs" / "dinov2_patches_30.npz")
))
WEIGHTS_PATH = ROOT / "data" / "outputs" / "act_policy.pt"
STATS_PATH = ROOT / "data" / "outputs" / "act_norm_stats.npz"

# Patch-token grid: DINOv2-S/14 at 224 input → 16×16 patches.
# Average-pool down to PATCH_GRID×PATCH_GRID for a tractable input to ACT.
PATCH_GRID = 4   # 4x4 = 16 spatial tokens per camera (× 3 cameras = 48 + 1 state)

# Port slot in 26D state (indices 22, 23, 24). Replaced with nominal during
# training so the model can't cheat by reading ground-truth port from state —
# it must extract board pose from the DINOv2 vision tokens instead.
PORT_SLOT = slice(22, 25)
NOMINAL_PORT_POS = np.array([0.220, -0.013, 1.273], dtype=np.float32)

DINOV2_MODEL = "facebook/dinov2-small"   # ViT-S/14, 22M params, 384-d CLS
DINOV2_INPUT = 224                        # multiple of 14
CHUNK_SIZE = 16
CAMERAS = ("left", "center", "right")


# --- Vision encoder (frozen DINOv2) ---

def load_dinov2(device: str = "cpu"):
    """Load frozen DINOv2-S/14 from HuggingFace."""
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
    # DINOv2 expects 224x224 input
    processor.size = {"height": DINOV2_INPUT, "width": DINOV2_INPUT}
    model = AutoModel.from_pretrained(DINOV2_MODEL).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


def encode_batch(processor, model, imgs: np.ndarray, device: str,
                 mode: str = "cls") -> np.ndarray:
    """Encode (B, H, W, 3) uint8 images → DINOv2 features.

    mode="cls": returns (B, 384) CLS-token features.
    mode="patches": returns (B, PATCH_GRID*PATCH_GRID, 384) avg-pooled patch tokens.
    """
    inputs = processor(images=list(imgs), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    if mode == "cls":
        return out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    elif mode == "patches":
        # last_hidden_state: (B, 1+256, 384). Drop CLS, reshape to 16x16 grid.
        patches = out.last_hidden_state[:, 1:, :]                 # (B, 256, 384)
        B, _, D = patches.shape
        grid = patches.reshape(B, 16, 16, D).permute(0, 3, 1, 2)  # (B, D, 16, 16)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid, PATCH_GRID)
        return pooled.permute(0, 2, 3, 1).reshape(B, PATCH_GRID*PATCH_GRID, D)\
            .cpu().numpy().astype(np.float32)
    raise ValueError(f"Unknown mode: {mode}")


# --- Cache step ---

def cache_features(batch_size: int = 32, mode: str = "cls") -> None:
    """Encode all demo images through frozen DINOv2 → save .npz cache.

    mode="cls":     feats[cam] is (n_eps, T, 384)            → FEATS_PATH
    mode="patches": feats[cam] is (n_eps, T, P*P, 384)        → PATCHES_PATH
                    where P = PATCH_GRID
    """
    target_path = FEATS_PATH if mode == "cls" else PATCHES_PATH
    if target_path.exists():
        print(f"Cache already exists: {target_path}")
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Caching DINOv2 features on {device}...")

    processor, model = load_dinov2(device)

    raw = np.load(DATA_PATH)
    n_eps, T = raw["states"].shape[:2]
    print(f"Demos: {n_eps} eps × {T} frames × {len(CAMERAS)} cameras")

    out = {
        "states": raw["states"].astype(np.float32),
        "actions": raw["actions"].astype(np.float32),
        "lengths": raw["episode_lengths"].astype(np.int64),
    }

    if mode == "cls":
        feat_shape = (384,)
    elif mode == "patches":
        feat_shape = (PATCH_GRID * PATCH_GRID, 384)
    else:
        raise ValueError(mode)

    for cam in CAMERAS:
        key = f"images_{cam}_camera"
        all_imgs = raw[key]                       # (n_eps, T, H, W, 3) uint8
        flat = all_imgs.reshape(-1, *all_imgs.shape[2:])
        n = flat.shape[0]
        feats = np.zeros((n, *feat_shape), dtype=np.float32)
        t0 = time.time()
        for i in range(0, n, batch_size):
            feats[i : i + batch_size] = encode_batch(
                processor, model, flat[i : i + batch_size], device, mode=mode,
            )
            if (i // batch_size) % 20 == 0:
                pct = (i + batch_size) / n
                eta = (time.time() - t0) / max(pct, 1e-3) * (1 - pct)
                print(f"  {cam:>6} {i:>5}/{n}  {pct*100:5.1f}%  eta {eta/60:4.1f}min", flush=True)
        out[f"feats_{cam}"] = feats.reshape(n_eps, T, *feat_shape)
        print(f"  {cam} done in {(time.time()-t0)/60:.1f} min", flush=True)

    np.savez_compressed(target_path, **out)
    size_mb = target_path.stat().st_size / 1e6
    print(f"Saved {target_path} ({size_mb:.1f} MB)")


# --- ACT head ---

class ACTHead(nn.Module):
    """Action Chunking Transformer over (vision tokens, state token).

    Accepts either (B, n_cams, D) for CLS mode or (B, n_cams, P*P, D) for
    patch-token mode. Both flatten to (B, n_cams * n_spatial, D) before the
    encoder. No CVAE conditioning — pure encoder–decoder L1 regression.
    """

    def __init__(
        self,
        vision_dim: int = 384,
        state_dim: int = 26,
        action_dim: int = 6,
        chunk_size: int = CHUNK_SIZE,
        n_cams: int = 3,
        n_spatial: int = 1,         # 1 = CLS token; PATCH_GRID*PATCH_GRID for patches
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.n_cams = n_cams
        self.n_spatial = n_spatial

        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.input_pos = nn.Parameter(torch.randn(n_cams * n_spatial + 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.action_queries = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, vision_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # vision_tokens: (B, n_cams, D) or (B, n_cams, n_spatial, D)
        # state: (B, state_dim)
        B = vision_tokens.shape[0]
        if vision_tokens.dim() == 3:
            vt = vision_tokens.unsqueeze(2)        # (B, n_cams, 1, D)
        else:
            vt = vision_tokens                      # (B, n_cams, n_spatial, D)
        vt = vt.reshape(B, -1, vt.shape[-1])       # (B, n_cams*n_spatial, D)
        v = self.vision_proj(vt)
        s = self.state_proj(state).unsqueeze(1)
        x = torch.cat([v, s], dim=1) + self.input_pos.unsqueeze(0)
        memory = self.encoder(x)
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)
        return self.action_head(out)  # (B, K, action_dim)


# --- Training step ---

def build_chunks(feats_path: Path, chunk_size: int = CHUNK_SIZE):
    """Load cached features and slice into (vision, state, action_chunk) tuples.

    The port_pos slot in the state vector is replaced with NOMINAL_PORT_POS so
    the model cannot read ground-truth board pose from state — it must learn to
    extract it from the vision tokens.

    Returns vision shape (n, n_cams, [n_spatial,] D) depending on cache type.
    """
    raw = np.load(feats_path)
    states = raw["states"].copy()                  # (n_eps, T, 26)
    states[..., PORT_SLOT] = NOMINAL_PORT_POS      # mask out the leakage
    actions = raw["actions"]                       # (n_eps, T, 6)
    lengths = raw["lengths"]
    feats = np.stack(
        [raw[f"feats_{c}"] for c in CAMERAS], axis=2
    )                                              # (n_eps, T, n_cams, [P*P,] D)

    vision_list, state_list, chunk_list = [], [], []
    for ep in range(states.shape[0]):
        L = int(lengths[ep])
        for t in range(L - chunk_size):
            vision_list.append(feats[ep, t])       # (n_cams, [n_spatial,] D)
            state_list.append(states[ep, t])       # (26,)
            chunk_list.append(actions[ep, t : t + chunk_size])  # (K, 6)

    V = np.stack(vision_list).astype(np.float32)
    S = np.stack(state_list).astype(np.float32)
    A = np.stack(chunk_list).astype(np.float32)

    s_mean, s_std = S.mean(0), S.std(0) + 1e-6
    a_flat = A.reshape(-1, A.shape[-1])
    a_mean, a_std = a_flat.mean(0), a_flat.std(0) + 1e-6
    return V, S, A, dict(state_mean=s_mean, state_std=s_std,
                         action_mean=a_mean, action_std=a_std)


def train(epochs: int = 5000, batch_size: int = 256, lr: float = 1e-4,
          val_frac: float = 0.1, mode: str = "cls") -> None:
    feats_path = FEATS_PATH if mode == "cls" else PATCHES_PATH
    if not feats_path.exists():
        raise FileNotFoundError(
            f"Run `cache --mode {mode}` first: {feats_path} missing"
        )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training ACT on {device}...")

    V, S, A, stats = build_chunks(feats_path)
    n_spatial = 1 if mode == "cls" else PATCH_GRID * PATCH_GRID
    print(f"Chunks: {len(V)} (vision[mode={mode}, n_spatial={n_spatial}] × state "
          f"→ {CHUNK_SIZE}-step action)")

    # Normalize
    s_mean = torch.tensor(stats["state_mean"], dtype=torch.float32, device=device)
    s_std = torch.tensor(stats["state_std"], dtype=torch.float32, device=device)
    a_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device=device)
    a_std = torch.tensor(stats["action_std"], dtype=torch.float32, device=device)

    Vt = torch.tensor(V, device=device)
    St = (torch.tensor(S, device=device) - s_mean) / s_std
    At = (torch.tensor(A, device=device) - a_mean) / a_std

    n = Vt.shape[0]
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=rng)
    n_val = int(n * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    model = ACTHead(n_spatial=n_spatial).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    # Augmentation: small Gaussian noise on state and vision features.
    # Configurable via env vars so we can sweep without editing the file.
    import os as _os
    state_noise = float(_os.environ.get("STATE_NOISE", "0.05"))
    vision_noise = float(_os.environ.get("VISION_NOISE", "0.02"))

    best_val = float("inf")
    t0 = time.time()
    for ep in range(epochs):
        idx = train_idx[torch.randint(len(train_idx), (batch_size,))]
        v_aug = Vt[idx] + vision_noise * torch.randn_like(Vt[idx])
        s_aug = St[idx] + state_noise * torch.randn_like(St[idx])
        pred = model(v_aug, s_aug)
        loss = F.l1_loss(pred, At[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (ep + 1) % 250 == 0:
            with torch.no_grad():
                model.eval()
                # Validation in mini-batches to avoid OOM
                val_l = 0.0
                B = 1024
                for j in range(0, len(val_idx), B):
                    vi = val_idx[j : j + B]
                    val_l += F.l1_loss(model(Vt[vi], St[vi]), At[vi]).item() * len(vi)
                val_l /= len(val_idx)
                model.train()
            tag = ""
            if val_l < best_val:
                best_val = val_l
                tag = "  *best*"
                STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": {
                        "mode": mode,
                        "n_spatial": n_spatial,
                        "vision_dim": 384,
                        "state_dim": 26,
                        "action_dim": 6,
                        "chunk_size": CHUNK_SIZE,
                        "n_cams": 3,
                        "d_model": 256,
                        "nhead": 4,
                        "num_layers": 4,
                    },
                }, WEIGHTS_PATH)
                np.savez(STATS_PATH, **stats)
            print(f"  [{ep+1:>5}/{epochs}] train={loss.item():.4f}  val={val_l:.4f}{tag}")

    print(f"Done in {(time.time()-t0)/60:.1f} min  best_val={best_val:.4f}")
    print(f"Saved → {WEIGHTS_PATH}")
    print(f"       {STATS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["cache", "train", "all"])
    parser.add_argument("--mode", choices=["cls", "patches"], default="cls",
                        help="DINOv2 token mode: cls (1 token/cam) or patches (P*P/cam)")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cache-batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.action in ("cache", "all"):
        cache_features(batch_size=args.cache_batch_size, mode=args.mode)
    if args.action in ("train", "all"):
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              mode=args.mode)


if __name__ == "__main__":
    main()
