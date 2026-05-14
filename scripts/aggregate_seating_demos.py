#!/usr/bin/env python3
"""Aggregate per-episode seating .npz files into a single training-format file.

Reads `data/seating_episodes/episode_*.npz` (downloaded from the Modal
`aic-data` volume) and writes `data/outputs/demos_seat_vN.npz` matching the
schema `scripts/train_modal_vision.py` consumes.

For v39+ episodes the state is 32-D (26-D legacy layout + 6-D wrench
appended). For legacy v38 episodes it is 26-D — the aggregator will reject
them when running v39 training.

Episode lengths vary; we pad to the max with the last frame's data (a no-op
for image and a hold-position for actions). Padding is then masked-out in
training via `episode_lengths`.

Also writes per-episode metadata:
  scene_ids              (n_eps,)       object
  port_world             (n_eps, 3)     float32
  cable_types, port_types (n_eps,)      object
  peak_wrench_z          (n_eps,)       float32
  push_contact_step      (n_eps,)       int32  (v39+)
  tail_fz, delta_tail_fz (n_eps,)       float32 (v39+)

Usage:
  uv run scripts/aggregate_seating_demos.py
  uv run scripts/aggregate_seating_demos.py --download-from-modal
  uv run scripts/aggregate_seating_demos.py --filter-seated  # drop bad seats
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def _download_from_modal(local_dir: Path) -> None:
    """Pull seating_episodes/ from the Modal aic-data volume."""
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading seating_episodes from Modal volume → {local_dir}")
    subprocess.check_call([
        "modal", "volume", "get", "aic-data", "/seating_episodes/",
        str(local_dir.parent), "--force",
    ])
    print("Download complete")


def _load_one(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)
    out = {
        "scene_id": str(z["scene_id"]),
        "states": z["states"].astype(np.float32),
        "actions": z["actions"].astype(np.float32),
        "wrench_z": z["wrench_z"].astype(np.float32),
        "images_left_camera": z["images_left_camera"],
        "images_center_camera": z["images_center_camera"],
        "images_right_camera": z["images_right_camera"],
        "port_world": z["port_world"].astype(np.float32),
        "cable_type": str(z["cable_type"]),
        "port_type": str(z["port_type"]),
    }
    # v39 seat-quality metrics (back-compat: missing in v38 episodes).
    out["push_contact_step"] = (
        int(z["push_contact_step"]) if "push_contact_step" in keys else None
    )
    out["tail_fz"] = (
        float(z["tail_fz"]) if "tail_fz" in keys else None
    )
    out["delta_tail_fz"] = (
        float(z["delta_tail_fz"]) if "delta_tail_fz" in keys else None
    )
    return out


# Filter rule: a "seated" episode is one where the push didn't bottom out
# on the faceplate AND the post-push tail wrench shows the cable settled
# (NOT stuck in compression on the faceplate).
#
# Wrench convention from observed data: in-air baseline Fz ≈ +20-25 N
# (gripper + cable mass supported by the wrist sensor). Two failure modes:
#  - push_contact_step >= 0: push hit faceplate; aborted on +8 N spike.
#  - tail Fz STAYS HIGH-POSITIVE: gripper still pressing on faceplate
#    after push, no insertion. delta_tail_fz > +5 captures this.
# A LARGE NEGATIVE delta_tail_fz means the cable inserted and the gripper
# continued descending into the port — that is a SUCCESSFUL seat, not a
# failure. The original |delta| < 5 rule wrongly rejected these; the rule
# below uses signed delta so insertion patterns pass.
SEATED_DELTA_TAIL_FZ_MAX = 5.0


def _is_seated(ep: dict) -> bool:
    """Return True if this episode looks like a successful insert.

    Falls back to True for legacy episodes missing the metrics — those
    won't load anyway when state_dim mismatches, so this is safe.
    """
    pcs = ep.get("push_contact_step")
    dtf = ep.get("delta_tail_fz")
    if pcs is None or dtf is None:
        return True
    if pcs != -1:
        # -2 = push skipped (descent already loaded too high)
        # >=0 = push bottomed out on faceplate
        return False
    # Signed delta: positive means faceplate compression (failure). Large
    # negative means insertion + compression past the seat (success).
    return dtf < SEATED_DELTA_TAIL_FZ_MAX


def _pad_to(arr: np.ndarray, target_T: int) -> np.ndarray:
    """Pad along axis 0 (time) to length target_T by repeating the last frame."""
    cur_T = arr.shape[0]
    if cur_T >= target_T:
        return arr[:target_T]
    last = arr[-1:]
    pad = np.repeat(last, target_T - cur_T, axis=0)
    return np.concatenate([arr, pad], axis=0)


def _truncate_from_end(arr: np.ndarray, max_T: int) -> np.ndarray:
    """If arr is longer than max_T along axis 0, keep the LAST max_T entries
    (preserves the descent + push that comes near the end of trajectories)."""
    return arr[-max_T:] if arr.shape[0] > max_T else arr


def _downsample_2x(img: np.ndarray) -> np.ndarray:
    """2x stride downsample images (T, H, W, 3) → (T, H/2, W/2, 3)."""
    return img[:, ::2, ::2].copy()


def aggregate(in_dir: Path, out_path: Path,
              expected_state_dim: int = 32,
              expected_action_dim: int = 6,
              max_t_cap: int | None = None,
              downsample: int = 1,
              filter_seated: bool = False) -> None:
    paths = sorted(in_dir.glob("episode_*.npz"))
    if not paths:
        sys.exit(f"No episode_*.npz files in {in_dir}")
    print(f"Found {len(paths)} episodes in {in_dir}")

    eps: list[dict] = []
    rejected = 0
    for p in paths:
        try:
            ep = _load_one(p)
        except Exception as e:
            print(f"  REJECT {p.name}: load failed ({e!r})")
            rejected += 1
            continue
        T = ep["states"].shape[0]
        if T < 10:
            print(f"  REJECT {p.name}: too short ({T} ticks)")
            rejected += 1
            continue
        if ep["states"].shape[1] != expected_state_dim:
            print(f"  REJECT {p.name}: state_dim {ep['states'].shape[1]} != "
                  f"{expected_state_dim}")
            rejected += 1
            continue
        if ep["actions"].shape[1] != expected_action_dim:
            print(f"  REJECT {p.name}: action_dim {ep['actions'].shape[1]} != "
                  f"{expected_action_dim}")
            rejected += 1
            continue
        if filter_seated and not _is_seated(ep):
            print(f"  REJECT {p.name}: not seated "
                  f"(push_contact_step={ep.get('push_contact_step')}, "
                  f"delta_tail_fz={ep.get('delta_tail_fz')})")
            rejected += 1
            continue
        eps.append(ep)

    if not eps:
        sys.exit(f"All {len(paths)} episodes rejected")
    print(f"Loaded {len(eps)}/{len(paths)} episodes ({rejected} rejected)")

    # Optional: truncate from end to keep recent frames (descent + push)
    if max_t_cap is not None:
        for ep in eps:
            for k in ("states", "actions", "wrench_z",
                      "images_left_camera", "images_center_camera",
                      "images_right_camera"):
                if k in ep and ep[k].ndim >= 1:
                    ep[k] = _truncate_from_end(ep[k], max_t_cap)
        print(f"Truncated episodes to last {max_t_cap} ticks each (max).")

    # Optional: downsample images N×
    if downsample > 1:
        if downsample != 2:
            sys.exit(f"Only --downsample 2 supported (got {downsample})")
        for ep in eps:
            for cam in ("images_left_camera", "images_center_camera",
                        "images_right_camera"):
                ep[cam] = _downsample_2x(ep[cam])
        print(f"Downsampled images {downsample}× per axis.")

    # Image dims must be consistent
    img_shape = eps[0]["images_left_camera"].shape[1:]
    for i, ep in enumerate(eps):
        for cam in ("images_left_camera", "images_center_camera",
                    "images_right_camera"):
            shp = ep[cam].shape[1:]
            if shp != img_shape:
                sys.exit(f"Episode {i} {cam} shape {shp} != {img_shape}")

    # Pad to max T
    max_T = max(ep["states"].shape[0] for ep in eps)
    n = len(eps)
    print(f"Episode length: min={min(ep['states'].shape[0] for ep in eps)}, "
          f"max={max_T}, mean={np.mean([ep['states'].shape[0] for ep in eps]):.1f}")
    # RAM estimate
    bytes_per_stream = n * max_T
    for d in img_shape:
        bytes_per_stream *= d
    bytes_total = 3 * bytes_per_stream + n * max_T * (expected_state_dim + expected_action_dim) * 4
    print(f"Estimated peak RAM: {bytes_total/1e9:.1f} GB "
          f"({n} eps × {max_T} T × {img_shape}, 3 streams)")

    states = np.zeros((n, max_T, expected_state_dim), dtype=np.float32)
    actions = np.zeros((n, max_T, expected_action_dim), dtype=np.float32)
    images_l = np.zeros((n, max_T) + img_shape, dtype=np.uint8)
    images_c = np.zeros((n, max_T) + img_shape, dtype=np.uint8)
    images_r = np.zeros((n, max_T) + img_shape, dtype=np.uint8)
    episode_lengths = np.zeros((n,), dtype=np.int64)

    scene_ids = np.empty((n,), dtype=object)
    port_world = np.zeros((n, 3), dtype=np.float32)
    cable_types = np.empty((n,), dtype=object)
    port_types = np.empty((n,), dtype=object)
    peak_wz = np.zeros((n,), dtype=np.float32)
    push_contact_step = np.full((n,), np.iinfo(np.int32).min, dtype=np.int32)
    tail_fz = np.full((n,), np.nan, dtype=np.float32)
    delta_tail_fz = np.full((n,), np.nan, dtype=np.float32)

    for i, ep in enumerate(eps):
        T = ep["states"].shape[0]
        states[i] = _pad_to(ep["states"], max_T)
        actions[i] = _pad_to(ep["actions"], max_T)
        images_l[i] = _pad_to(ep["images_left_camera"], max_T)
        images_c[i] = _pad_to(ep["images_center_camera"], max_T)
        images_r[i] = _pad_to(ep["images_right_camera"], max_T)
        episode_lengths[i] = T
        scene_ids[i] = ep["scene_id"]
        port_world[i] = ep["port_world"]
        cable_types[i] = ep["cable_type"]
        port_types[i] = ep["port_type"]
        peak_wz[i] = float(np.max(np.abs(ep["wrench_z"]))) if ep["wrench_z"].size else 0.0
        if ep.get("push_contact_step") is not None:
            push_contact_step[i] = ep["push_contact_step"]
        if ep.get("tail_fz") is not None:
            tail_fz[i] = ep["tail_fz"]
        if ep.get("delta_tail_fz") is not None:
            delta_tail_fz[i] = ep["delta_tail_fz"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=states,
        actions=actions,
        episode_lengths=episode_lengths,
        images_left_camera=images_l,
        images_center_camera=images_c,
        images_right_camera=images_r,
        scene_ids=scene_ids,
        port_world=port_world,
        cable_types=cable_types,
        port_types=port_types,
        peak_wrench_z=peak_wz,
        push_contact_step=push_contact_step,
        tail_fz=tail_fz,
        delta_tail_fz=delta_tail_fz,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.1f} MB, {n} episodes, max_T={max_T}, "
          f"img_shape={img_shape})")
    # Distribution
    by_port = {}
    for pt in port_types:
        by_port[pt] = by_port.get(pt, 0) + 1
    print(f"Port-type distribution: {by_port}")
    print(f"Peak |Fz| stats: mean={peak_wz.mean():.2f} N, max={peak_wz.max():.2f} N "
          f"({np.sum(peak_wz > 4.0)}/{n} eps with peak>4N — likely seated)")
    valid = ~np.isnan(delta_tail_fz)
    if valid.any():
        seated = (push_contact_step[valid] == -1) & (
            np.abs(delta_tail_fz[valid]) < SEATED_DELTA_TAIL_FZ_MAX
        )
        print(f"Seat metrics ({valid.sum()}/{n} eps with v39 metrics): "
              f"{int(seated.sum())} seated, "
              f"{int(((push_contact_step[valid] >= 0)).sum())} bottomed-out, "
              f"{int(((push_contact_step[valid] == -2)).sum())} skipped")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="data/seating_episodes",
                    help="Directory of episode_*.npz")
    ap.add_argument("--out", dest="out_path",
                    default="data/outputs/demos_seat_v1.npz",
                    help="Output aggregated .npz")
    ap.add_argument("--download-from-modal", action="store_true",
                    help="First pull seating_episodes/ from Modal aic-data volume")
    ap.add_argument("--max-t-cap", type=int, default=None,
                    help="Truncate each episode to last N ticks (default: no cap)")
    ap.add_argument("--downsample", type=int, default=1,
                    help="Spatial downsample factor for images (1 or 2)")
    ap.add_argument("--filter-seated", action="store_true",
                    help="Drop episodes where the push bottomed out or the "
                         "tail Fz indicates the cable did not seat")
    ap.add_argument("--state-dim", type=int, default=32,
                    help="Expected state vector dimension "
                         "(32 for v39+ with wrench, 26 for v38 legacy)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out_path)

    if args.download_from_modal:
        _download_from_modal(in_dir)

    aggregate(in_dir, out_path,
              expected_state_dim=args.state_dim,
              max_t_cap=args.max_t_cap,
              downsample=args.downsample,
              filter_seated=args.filter_seated)


if __name__ == "__main__":
    main()
