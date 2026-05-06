#!/usr/bin/env python3
"""Train a small DINOv2-patches → (port_x, port_y) regression head.

Idea: instead of an end-to-end vision-to-action policy, train an explicit
port-localization head. At deployment time:
  1. Encode wrist images through frozen DINOv2 → patch tokens
  2. Localizer predicts (port_x, port_y) in world coords
  3. Feed predicted port_pos into the existing DAgger-MLP's state slot
     (which already achieves ~90% with ground-truth port).

Cleaner separation of concerns; the localizer can be evaluated on its own.

Usage (assumes patches cache already exists):
    .venv/bin/python3 scripts/train_dinov2_port_localizer.py \\
        --epochs 5000 --batch-size 256
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATCHES = ROOT / "data" / "outputs" / "dinov2_patches_30.npz"
WEIGHTS_PATH = ROOT / "data" / "outputs" / "port_localizer_dinov2.pt"
STATS_PATH = ROOT / "data" / "outputs" / "port_localizer_stats.npz"
PORT_SLOT = slice(22, 25)
N_CAMS = 3
PATCH_GRID = 4


class PortLocalizer(nn.Module):
    """Small transformer over (n_cams * P*P) DINOv2 patch tokens → (x, y)."""

    def __init__(self, vision_dim: int = 384, n_tokens: int = 48,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.proj = nn.Linear(vision_dim, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)   # (x, y)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        # vision_tokens: (B, n_cams, P*P, D) → flatten to (B, n_tokens, D)
        B, C, S, D = vision_tokens.shape
        x = vision_tokens.reshape(B, C * S, D)
        x = self.proj(x) + self.pos.unsqueeze(0)
        x = self.encoder(x)
        x = x.mean(dim=1)                    # (B, d_model) — pool tokens
        return self.head(x)                   # (B, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training port localizer on {device}", flush=True)

    raw = np.load(args.patches)
    states = raw["states"]                              # (n_eps, T, 26)
    feats = np.stack([raw[f"feats_{c}"] for c in ("left", "center", "right")],
                     axis=2)                            # (n_eps, T, n_cams, P*P, 384)
    lengths = raw["lengths"]

    V_list, P_list = [], []
    for ep in range(states.shape[0]):
        L = int(lengths[ep])
        V_list.append(feats[ep, :L])
        # Use only the first 2 components (x, y); z is fixed.
        P_list.append(states[ep, :L, 22:24])
    V = np.concatenate(V_list, axis=0).astype(np.float32)
    P = np.concatenate(P_list, axis=0).astype(np.float32)
    print(f"Samples: {len(V)}  vision_shape={V.shape[1:]}  port_shape={P.shape[1:]}",
          flush=True)

    # Normalize port targets
    p_mean = P.mean(axis=0)
    p_std = P.std(axis=0) + 1e-6
    print(f"port mean: {p_mean}, std: {p_std}", flush=True)

    Vt = torch.tensor(V, device=device)
    Pt = (torch.tensor(P, device=device) - torch.tensor(p_mean, device=device)) \
         / torch.tensor(p_std, device=device)

    n = len(Vt)
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=rng)
    n_val = int(n * args.val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    model = PortLocalizer(n_tokens=N_CAMS * PATCH_GRID * PATCH_GRID).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    state_noise_std = 0.02   # mild vision augmentation

    best_val = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        idx = train_idx[torch.randint(len(train_idx), (args.batch_size,))]
        v_aug = Vt[idx] + state_noise_std * torch.randn_like(Vt[idx])
        pred = model(v_aug)
        loss = F.l1_loss(pred, Pt[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (ep + 1) % 200 == 0:
            with torch.no_grad():
                model.eval()
                vl = 0.0
                B = 1024
                for j in range(0, len(val_idx), B):
                    vi = val_idx[j : j + B]
                    vl += F.l1_loss(model(Vt[vi]), Pt[vi]).item() * len(vi)
                vl /= len(val_idx)
                model.train()
            tag = ""
            if vl < best_val:
                best_val = vl
                tag = "  *best*"
                torch.save(model.state_dict(), WEIGHTS_PATH)
                np.savez(STATS_PATH, port_mean=p_mean, port_std=p_std)
            # Also report in millimeters (denormalized)
            mm = vl * p_std.mean() * 1000
            print(f"  [{ep+1:>5}/{args.epochs}] train={loss.item():.4f}  val={vl:.4f} "
                  f"(~{mm:.1f}mm){tag}", flush=True)

    print(f"Done in {(time.time()-t0)/60:.1f} min  best_val={best_val:.4f}",
          flush=True)
    print(f"Saved → {WEIGHTS_PATH}\n       {STATS_PATH}")


if __name__ == "__main__":
    main()
