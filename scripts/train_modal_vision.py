#!/usr/bin/env python3
"""Train DINOv2/v3 + ACT + Port Localizer on Modal T4 GPU.

Replaces local-Mac training entirely. The user's Mac doesn't have memory
headroom for the simultaneous DINOv2 forward + transformer training.

Workflow:
    # 1. Upload demos to Modal volume (one-time per dataset)
    modal volume put aic-data data/demos_with_images_80.npz /data/

    # 2. Run full pipeline (cache → ACT → localizer) on Modal T4
    modal run scripts/train_modal_vision.py::pipeline \\
        --demos-name demos_with_images_80.npz \\
        --backbone facebook/dinov2-small \\
        --act-epochs 8000 \\
        --loc-epochs 5000

    # 3. Download outputs
    modal volume get aic-data /outputs/ data/outputs/

The Modal volume layout after a run:
    /vol/data/<demos_name>.npz
    /vol/outputs/dinov2_patches_<tag>.npz       (cached features)
    /vol/outputs/act_policy_<tag>.pt
    /vol/outputs/act_norm_stats_<tag>.npz
    /vol/outputs/port_localizer_<tag>.pt
    /vol/outputs/port_localizer_stats_<tag>.npz
"""

from pathlib import Path

import modal

vol = modal.Volume.from_name("aic-data", create_if_missing=True)
VOL = Path("/vol")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch~=2.7.0",
        "torchvision~=0.22.0",
        "transformers>=4.55.0",   # DINOv3 needs >=4.49 for the model_type
        "safetensors",
        "hf-xet",
        "einops",
        "numpy",
    )
)

app = modal.App("aic-train-vision", image=image)


# --- The training code is embedded as a string so Modal pickles it cleanly ---
TRAIN_CODE = '''
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PORT_SLOT = slice(22, 25)
NOMINAL_PORT_POS = np.array([0.220, -0.013, 1.273], dtype=np.float32)
CAMERAS = ("left", "center", "right")


def load_backbone(name: str, device: str):
    from transformers import AutoImageProcessor, AutoModel
    print(f"Loading {name}...", flush=True)
    proc = AutoImageProcessor.from_pretrained(name)
    proc.size = {"height": 224, "width": 224}
    model = AutoModel.from_pretrained(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return proc, model


def encode_batch_patches(proc, model, imgs, device, patch_grid: int,
                         n_register: int, dim: int, native_grid: int):
    inputs = proc(images=list(imgs), return_tensors="pt")
    with torch.no_grad():
        out = model(pixel_values=inputs["pixel_values"].to(device))
    # Drop CLS + register tokens, keep N×N patches
    skip = 1 + n_register
    patches = out.last_hidden_state[:, skip:, :]
    B, T, D = patches.shape
    grid = patches.reshape(B, native_grid, native_grid, D).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, patch_grid)
    return pooled.permute(0, 2, 3, 1).reshape(
        B, patch_grid * patch_grid, D
    ).cpu().numpy().astype(np.float32)


def cache_features(demos_path: Path, out_path: Path, backbone: str,
                   patch_grid: int = 4, batch_size: int = 32) -> None:
    if out_path.exists():
        print(f"Cache exists: {out_path}", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load_backbone(backbone, device)

    # Probe model dims
    raw = np.load(demos_path)
    sample = raw["images_left_camera"][0, 0]
    inp = proc(images=[sample], return_tensors="pt")
    with torch.no_grad():
        out = model(pixel_values=inp["pixel_values"].to(device))
    n_total = out.last_hidden_state.shape[1]
    dim = out.last_hidden_state.shape[2]
    # DINOv3 has 4 register tokens, DINOv2 has 0
    n_register = 4 if "dinov3" in backbone.lower() else 0
    n_patches = n_total - 1 - n_register
    native_grid = int(n_patches ** 0.5)
    print(f"Backbone: {n_total} tokens (1 CLS + {n_register} reg + "
          f"{n_patches} patches, {native_grid}x{native_grid}, {dim}-d)", flush=True)

    n_eps, T = raw["states"].shape[:2]
    print(f"Demos: {n_eps} eps × {T} frames × {len(CAMERAS)} cameras", flush=True)

    out_dict = {
        "states": raw["states"].astype(np.float32),
        "actions": raw["actions"].astype(np.float32),
        "lengths": raw["episode_lengths"].astype(np.int64),
    }
    for cam in CAMERAS:
        all_imgs = raw[f"images_{cam}_camera"]
        flat = all_imgs.reshape(-1, *all_imgs.shape[2:])
        n = flat.shape[0]
        feats = np.zeros((n, patch_grid * patch_grid, dim), dtype=np.float32)
        t0 = time.time()
        for i in range(0, n, batch_size):
            feats[i : i + batch_size] = encode_batch_patches(
                proc, model, flat[i : i + batch_size], device,
                patch_grid, n_register, dim, native_grid,
            )
            if (i // batch_size) % 30 == 0:
                pct = (i + batch_size) / n
                eta = (time.time() - t0) / max(pct, 1e-3) * (1 - pct)
                print(f"  {cam:>6} {i:>5}/{n}  {pct*100:5.1f}%  eta {eta/60:4.1f}min",
                      flush=True)
        out_dict[f"feats_{cam}"] = feats.reshape(n_eps, T, patch_grid * patch_grid, dim)
        print(f"  {cam} done in {(time.time()-t0)/60:.1f} min", flush=True)

    np.savez_compressed(out_path, **out_dict)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {out_path} ({size_mb:.1f} MB)", flush=True)


# --- ACT head ---

class ACTHead(nn.Module):
    def __init__(self, vision_dim=384, state_dim=26, action_dim=6,
                 chunk_size=16, n_cams=3, n_spatial=16,
                 d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.input_pos = nn.Parameter(
            torch.randn(n_cams * n_spatial + 1, d_model) * 0.02
        )
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.action_queries = nn.Parameter(torch.randn(chunk_size, d_model) * 0.02)
        dec = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, vision_tokens, state):
        B = vision_tokens.shape[0]
        if vision_tokens.dim() == 3:
            vt = vision_tokens.unsqueeze(2)
        else:
            vt = vision_tokens
        vt = vt.reshape(B, -1, vt.shape[-1])
        v = self.vision_proj(vt)
        s = self.state_proj(state).unsqueeze(1)
        x = torch.cat([v, s], dim=1) + self.input_pos.unsqueeze(0)
        memory = self.encoder(x)
        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        return self.action_head(self.decoder(queries, memory))


def build_chunks(feats_path, chunk_size=16):
    raw = np.load(feats_path)
    states = raw["states"].copy()
    states[..., PORT_SLOT] = NOMINAL_PORT_POS
    actions = raw["actions"]
    lengths = raw["lengths"]
    feats = np.stack([raw[f"feats_{c}"] for c in CAMERAS], axis=2)

    V_l, S_l, A_l = [], [], []
    for ep in range(states.shape[0]):
        L = int(lengths[ep])
        for t in range(L - chunk_size):
            V_l.append(feats[ep, t]); S_l.append(states[ep, t])
            A_l.append(actions[ep, t : t + chunk_size])
    V = np.stack(V_l).astype(np.float32)
    S = np.stack(S_l).astype(np.float32)
    A = np.stack(A_l).astype(np.float32)
    s_mean, s_std = S.mean(0), S.std(0) + 1e-6
    a_flat = A.reshape(-1, A.shape[-1])
    a_mean, a_std = a_flat.mean(0), a_flat.std(0) + 1e-6
    return V, S, A, dict(state_mean=s_mean, state_std=s_std,
                         action_mean=a_mean, action_std=a_std)


def train_act(feats_path: Path, weights_path: Path, stats_path: Path,
              epochs=8000, batch_size=128, lr=1e-4, val_frac=0.1,
              state_noise=0.05, vision_noise=0.02):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Train ACT on {device}", flush=True)
    V, S, A, stats = build_chunks(feats_path)
    n_spatial = V.shape[2]
    vision_dim = V.shape[-1]
    print(f"Chunks: {len(V)}, n_spatial={n_spatial}, dim={vision_dim}", flush=True)

    s_mean = torch.tensor(stats["state_mean"], dtype=torch.float32, device=device)
    s_std = torch.tensor(stats["state_std"], dtype=torch.float32, device=device)
    a_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device=device)
    a_std = torch.tensor(stats["action_std"], dtype=torch.float32, device=device)
    Vt = torch.tensor(V, device=device)
    St = (torch.tensor(S, device=device) - s_mean) / s_std
    At = (torch.tensor(A, device=device) - a_mean) / a_std

    n = len(Vt)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = ACTHead(vision_dim=vision_dim, n_spatial=n_spatial).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_val = float("inf")
    t0 = time.time()
    for ep in range(epochs):
        idx = train_idx[torch.randint(len(train_idx), (batch_size,))]
        v_aug = Vt[idx] + vision_noise * torch.randn_like(Vt[idx])
        s_aug = St[idx] + state_noise * torch.randn_like(St[idx])
        loss = F.l1_loss(model(v_aug, s_aug), At[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (ep + 1) % 250 == 0:
            with torch.no_grad():
                model.eval()
                vl = 0.0; B = 1024
                for j in range(0, len(val_idx), B):
                    vi = val_idx[j : j + B]
                    vl += F.l1_loss(model(Vt[vi], St[vi]), At[vi]).item() * len(vi)
                vl /= len(val_idx); model.train()
            tag = ""
            if vl < best_val:
                best_val = vl; tag = "  *best*"
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": dict(mode="patches", n_spatial=n_spatial,
                                   vision_dim=vision_dim, state_dim=26, action_dim=6,
                                   chunk_size=16, n_cams=3, d_model=256,
                                   nhead=4, num_layers=4),
                }, weights_path)
                np.savez(stats_path, **stats)
            print(f"  [{ep+1:>5}/{epochs}] train={loss.item():.4f}  val={vl:.4f}{tag}",
                  flush=True)
    print(f"ACT done in {(time.time()-t0)/60:.1f} min  best_val={best_val:.4f}",
          flush=True)


# --- Port Localizer ---

class PortLocalizer(nn.Module):
    def __init__(self, vision_dim=384, n_tokens=48, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.proj = nn.Linear(vision_dim, d_model)
        self.pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, vision_tokens):
        B, C, S, D = vision_tokens.shape
        x = vision_tokens.reshape(B, C * S, D)
        x = self.proj(x) + self.pos.unsqueeze(0)
        x = self.encoder(x).mean(dim=1)
        return self.head(x)


def train_localizer(feats_path: Path, weights_path: Path, stats_path: Path,
                    epochs=5000, batch_size=256, lr=2e-4, val_frac=0.1,
                    vision_noise=0.02):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Train localizer on {device}", flush=True)
    raw = np.load(feats_path)
    states = raw["states"]
    feats = np.stack([raw[f"feats_{c}"] for c in CAMERAS], axis=2)
    lengths = raw["lengths"]
    n_spatial = feats.shape[3]
    vision_dim = feats.shape[-1]

    V_l, P_l = [], []
    for ep in range(states.shape[0]):
        L = int(lengths[ep])
        V_l.append(feats[ep, :L])
        P_l.append(states[ep, :L, 22:24])
    V = np.concatenate(V_l, axis=0).astype(np.float32)
    P = np.concatenate(P_l, axis=0).astype(np.float32)
    p_mean = P.mean(0); p_std = P.std(0) + 1e-6
    print(f"Samples: {len(V)}  port std: {p_std}", flush=True)

    Vt = torch.tensor(V, device=device)
    Pt = (torch.tensor(P, device=device) - torch.tensor(p_mean, device=device)) \\
         / torch.tensor(p_std, device=device)

    n = len(Vt)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_val = int(n * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = PortLocalizer(vision_dim=vision_dim, n_tokens=3 * n_spatial).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_val = float("inf")
    t0 = time.time()
    for ep in range(epochs):
        idx = train_idx[torch.randint(len(train_idx), (batch_size,))]
        v_aug = Vt[idx] + vision_noise * torch.randn_like(Vt[idx])
        loss = F.l1_loss(model(v_aug), Pt[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (ep + 1) % 200 == 0:
            with torch.no_grad():
                model.eval()
                vl = 0.0; B = 1024
                for j in range(0, len(val_idx), B):
                    vi = val_idx[j : j + B]
                    vl += F.l1_loss(model(Vt[vi]), Pt[vi]).item() * len(vi)
                vl /= len(val_idx); model.train()
            tag = ""
            if vl < best_val:
                best_val = vl; tag = "  *best*"
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), weights_path)
                np.savez(stats_path, port_mean=p_mean, port_std=p_std)
            mm = vl * p_std.mean() * 1000
            print(f"  [{ep+1:>4}/{epochs}] train={loss.item():.4f}  val={vl:.4f} "
                  f"(~{mm:.1f}mm){tag}", flush=True)
    print(f"Localizer done in {(time.time()-t0)/60:.1f} min  best_val={best_val:.4f}",
          flush=True)
'''


@app.function(
    gpu="T4",
    volumes={VOL: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,
)
def pipeline(
    demos_name: str = "demos_with_images_80.npz",
    backbone: str = "facebook/dinov2-small",
    tag: str | None = None,
    act_epochs: int = 8000,
    loc_epochs: int = 5000,
    cache_batch_size: int = 32,
    skip_cache: bool = False,
    skip_act: bool = False,
    skip_loc: bool = False,
):
    """Full pipeline: cache features → train ACT → train localizer."""
    exec(TRAIN_CODE, globals())

    backbone_tag = backbone.split("/")[-1].replace("-", "_")
    demos_tag = Path(demos_name).stem
    if tag is None:
        tag = f"{backbone_tag}_{demos_tag}"
    print(f"=== pipeline (tag={tag}) ===", flush=True)

    demos_path = VOL / "data" / demos_name
    feats_path = VOL / "outputs" / f"feats_{tag}.npz"
    act_w = VOL / "outputs" / f"act_{tag}.pt"
    act_s = VOL / "outputs" / f"act_stats_{tag}.npz"
    loc_w = VOL / "outputs" / f"loc_{tag}.pt"
    loc_s = VOL / "outputs" / f"loc_stats_{tag}.npz"

    if not skip_cache:
        cache_features(demos_path, feats_path, backbone, batch_size=cache_batch_size)
        vol.commit()
    if not skip_act:
        train_act(feats_path, act_w, act_s, epochs=act_epochs)
        vol.commit()
    if not skip_loc:
        train_localizer(feats_path, loc_w, loc_s, epochs=loc_epochs)
        vol.commit()

    return {"tag": tag, "feats": str(feats_path),
            "act": str(act_w), "loc": str(loc_w)}


@app.local_entrypoint()
def main(
    demos_name: str = "demos_with_images_80.npz",
    backbone: str = "facebook/dinov2-small",
    tag: str = "",
    act_epochs: int = 8000,
    loc_epochs: int = 5000,
):
    print(f"Submitting Modal job: backbone={backbone}, demos={demos_name}")
    out = pipeline.remote(
        demos_name=demos_name,
        backbone=backbone,
        tag=tag if tag else None,
        act_epochs=act_epochs,
        loc_epochs=loc_epochs,
    )
    print(f"Done. Outputs in /vol/outputs/ (tag={out['tag']})")
    print(f"Download: modal volume get aic-data /outputs/ data/outputs/")
