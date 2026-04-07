#!/usr/bin/env python3
"""Train insertion policy on Modal T4 GPU.

Supports two architectures:
  - mlp: Simple behavior cloning MLP (fast, state-only baseline)
  - diffusion: Diffusion policy with action chunking (slower, better)

Usage:
    # First: upload data to Modal volume
    modal volume create aic-data
    modal volume put aic-data data/demos.npz /data/

    # Train MLP baseline (~2 min on T4)
    modal run scripts/train_modal.py --arch mlp --epochs 200

    # Train diffusion policy (~30 min on T4)
    modal run scripts/train_modal.py --arch diffusion --epochs 500

    # Download trained weights
    modal volume get aic-data /outputs/ data/outputs/
"""

from pathlib import Path

import modal

# --- Modal infrastructure ---
vol = modal.Volume.from_name("aic-data", create_if_missing=True)
VOL_PATH = Path("/vol")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch~=2.4.0",
    "numpy",
)

app = modal.App("aic-train", image=image)

# --- Model definitions (inside the training function) ---

TRAIN_CODE = '''
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class MLPPolicy(nn.Module):
    """Simple MLP: state -> action."""

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
    """Diffusion policy with action chunking.

    Predicts a chunk of T future actions conditioned on current state.
    Uses DDPM-style denoising with a simple MLP denoiser.
    """

    def __init__(self, state_dim=26, action_dim=6, chunk_size=16,
                 hidden=512, n_diffusion_steps=50):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_steps = n_diffusion_steps

        # Denoiser: (noisy_action_chunk, state, timestep) -> denoised_action_chunk
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

        # DDPM schedule
        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward_denoise(self, noisy_actions, state, timestep):
        """Predict noise given noisy actions, state, and diffusion timestep."""
        B = noisy_actions.shape[0]
        flat_actions = noisy_actions.reshape(B, -1)
        state_feat = self.state_encoder(state)
        time_feat = self.time_embed(timestep)
        x = torch.cat([flat_actions, state_feat, time_feat], dim=-1)
        pred = self.denoiser(x)
        return pred.reshape(B, self.chunk_size, self.action_dim)

    def compute_loss(self, states, action_chunks):
        """DDPM training loss: predict the noise added to action chunks."""
        B = states.shape[0]
        device = states.device

        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(action_chunks)

        # Forward diffusion: add noise
        sqrt_a = self.sqrt_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)
        noisy = sqrt_a * action_chunks + sqrt_1ma * noise

        # Predict noise
        pred_noise = self.forward_denoise(noisy, states, t)

        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, state, device="cuda"):
        """DDPM sampling: denoise from pure noise to action chunk."""
        B = state.shape[0]
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


def load_data(data_path, chunk_size=16):
    """Load demos and create (state, action_chunk) pairs."""
    raw = np.load(str(data_path))
    states_all = raw["states"]      # (n_eps, max_len, 25)
    actions_all = raw["actions"]    # (n_eps, max_len, 6)
    lengths = raw["episode_lengths"]

    all_states = []
    all_actions = []
    all_chunks = []

    for ep in range(len(lengths)):
        L = lengths[ep]
        s = states_all[ep, :L]
        a = actions_all[ep, :L]
        all_states.append(s)
        all_actions.append(a)

        # Create action chunks (for diffusion policy)
        for t in range(L - chunk_size):
            all_chunks.append((s[t], a[t:t + chunk_size]))

    states_flat = np.concatenate(all_states, axis=0)
    actions_flat = np.concatenate(all_actions, axis=0)

    # Compute normalization stats
    state_mean = states_flat.mean(axis=0)
    state_std = states_flat.std(axis=0) + 1e-6
    action_mean = actions_flat.mean(axis=0)
    action_std = actions_flat.std(axis=0) + 1e-6

    stats = {
        "state_mean": state_mean, "state_std": state_std,
        "action_mean": action_mean, "action_std": action_std,
    }

    return states_flat, actions_flat, all_chunks, stats


def train_mlp(data_path, output_dir, epochs=200, lr=1e-3, batch_size=256):
    """Train MLP behavior cloning policy."""
    states, actions, _, stats = load_data(data_path)
    print(f"MLP training: {len(states)} samples, {epochs} epochs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Normalize
    s_mean = torch.tensor(stats["state_mean"], dtype=torch.float32, device=device)
    s_std = torch.tensor(stats["state_std"], dtype=torch.float32, device=device)
    a_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device=device)
    a_std = torch.tensor(stats["action_std"], dtype=torch.float32, device=device)

    S = torch.tensor(states, dtype=torch.float32, device=device)
    A = torch.tensor(actions, dtype=torch.float32, device=device)
    S = (S - s_mean) / s_std
    A = (A - a_mean) / a_std

    model = MLPPolicy(state_dim=26, action_dim=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n = len(S)
    noise_scale = 0.3  # aggressive state noise for rollout robustness
    for epoch in range(epochs):
        idx = torch.randperm(n, device=device)[:batch_size]
        # Noise injection: perturb state, keep action label
        S_noisy = S[idx] + noise_scale * torch.randn_like(S[idx])
        pred = model(S_noisy)
        loss = nn.functional.mse_loss(pred, A[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                full_loss = nn.functional.mse_loss(model(S), A)
            print(f"  [{epoch+1}/{epochs}] batch_loss={loss.item():.6f} full_loss={full_loss.item():.6f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "mlp_policy.pt")
    np.savez(output_dir / "norm_stats.npz", **{k: v for k, v in stats.items()})
    with open(output_dir / "config.json", "w") as f:
        json.dump({"arch": "mlp", "state_dim": 25, "action_dim": 6, "hidden": 256}, f)

    with torch.no_grad():
        final_loss = nn.functional.mse_loss(model(S), A).item()
    print(f"Final loss: {final_loss:.6f}")
    return final_loss


def train_diffusion(data_path, output_dir, epochs=500, lr=1e-4,
                    batch_size=128, chunk_size=16):
    """Train diffusion policy with action chunking."""
    _, _, chunks, stats = load_data(data_path, chunk_size=chunk_size)
    print(f"Diffusion training: {len(chunks)} chunks, {epochs} epochs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    s_mean = torch.tensor(stats["state_mean"], dtype=torch.float32, device=device)
    s_std = torch.tensor(stats["state_std"], dtype=torch.float32, device=device)
    a_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device=device)
    a_std = torch.tensor(stats["action_std"], dtype=torch.float32, device=device)

    # Build tensors
    chunk_states = torch.tensor(
        np.array([c[0] for c in chunks]), dtype=torch.float32, device=device
    )
    chunk_actions = torch.tensor(
        np.array([c[1] for c in chunks]), dtype=torch.float32, device=device
    )
    chunk_states = (chunk_states - s_mean) / s_std
    chunk_actions = (chunk_actions - a_mean.unsqueeze(0)) / a_std.unsqueeze(0)

    model = DiffusionPolicy(
        state_dim=26, action_dim=6, chunk_size=chunk_size
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n = len(chunk_states)
    for epoch in range(epochs):
        idx = torch.randperm(n, device=device)[:batch_size]
        loss = model.compute_loss(chunk_states[idx], chunk_actions[idx])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"  [{epoch+1}/{epochs}] loss={loss.item():.6f}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "diffusion_policy.pt")
    np.savez(output_dir / "norm_stats.npz", **{k: v for k, v in stats.items()})
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "arch": "diffusion", "state_dim": 25, "action_dim": 6,
            "chunk_size": chunk_size, "hidden": 512, "n_diffusion_steps": 50,
        }, f)

    print(f"Final loss: {loss.item():.6f}")
    return loss.item()
'''


@app.function(
    gpu="T4",
    volumes={VOL_PATH: vol},
    timeout=3600,
)
def train_on_gpu(arch: str = "mlp", epochs: int = 200, lr: float = 1e-3):
    exec(TRAIN_CODE, globals())
    data_path = VOL_PATH / "data" / "demos.npz"
    output_dir = VOL_PATH / "outputs"

    if arch == "mlp":
        loss = train_mlp(data_path, output_dir, epochs=epochs, lr=lr)
    elif arch == "diffusion":
        loss = train_diffusion(data_path, output_dir, epochs=epochs, lr=lr)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    vol.commit()
    return {"arch": arch, "loss": loss}


@app.local_entrypoint()
def main(arch: str = "mlp", epochs: int = 200, lr: float = 1e-3):
    print(f"Training {arch} policy for {epochs} epochs on T4...")
    result = train_on_gpu.remote(arch=arch, epochs=epochs, lr=lr)
    print(f"Done! Final loss: {result['loss']:.6f}")
    print("Download weights: modal volume get aic-data /outputs/ data/outputs/")
