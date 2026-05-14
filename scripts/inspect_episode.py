#!/usr/bin/env python3
"""Inspect a per-episode .npz from SeatingCollector. Verifies shape, dtypes,
and that the wrench shows a contact signature (proves the policy actually seated).

Usage:
  uv run scripts/inspect_episode.py data/seating_episodes/episode_smoke.npz
  uv run scripts/inspect_episode.py --download-from-modal episode_smoke.npz
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def _download(name: str, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading /seating_episodes/{name} from Modal aic-data → {dst_dir}")
    subprocess.check_call([
        "modal", "volume", "get", "aic-data",
        f"/seating_episodes/{name}",
        str(dst_dir / name),
        "--force",
    ])
    return dst_dir / name


def inspect(path: Path) -> None:
    z = np.load(path, allow_pickle=True)
    print(f"\n=== {path.name} ===")
    print(f"keys: {list(z.keys())}")
    print(f"scene_id: {z['scene_id']}")
    print(f"cable_type: {z['cable_type']}, port_type: {z['port_type']}")
    print(f"target: {z['target_module_name']}/{z['port_name']}")
    print(f"port_world: {z['port_world']}")
    print(f"gripper_offset: {z['gripper_offset']}")

    states = z["states"]
    actions = z["actions"]
    wrench = z["wrench_z"]
    print(f"\nstates  shape={states.shape}  dtype={states.dtype}")
    print(f"actions shape={actions.shape}  dtype={actions.dtype}")
    print(f"wrench  shape={wrench.shape}   dtype={wrench.dtype}")

    for cam in ("images_left_camera", "images_center_camera", "images_right_camera"):
        a = z[cam]
        print(f"{cam}: shape={a.shape}  dtype={a.dtype}  size={a.nbytes/1024:.0f} KB")

    # Sanity: state vector must be 26-D (matches DINOv2ACT)
    if states.shape[1] != 26:
        print(f"\n!! WARN: state_dim {states.shape[1]} != 26 (DINOv2ACT expects 26)")
    if actions.shape[1] != 6:
        print(f"\n!! WARN: action_dim {actions.shape[1]} != 6")

    # Wrench: did the cable contact the port?
    print(f"\nwrench_z stats:")
    print(f"  baseline (first 20 ticks): {wrench[:20].mean():.2f} ± {wrench[:20].std():.2f} N")
    print(f"  abs max:                   {np.abs(wrench).max():.2f} N at tick {np.argmax(np.abs(wrench))}/{len(wrench)}")
    print(f"  final 10 ticks mean:       {wrench[-10:].mean():.2f} N")

    contact_threshold = 4.0
    above = np.abs(wrench) > contact_threshold
    n_contact = int(above.sum())
    print(f"  ticks with |Fz|>{contact_threshold}N: {n_contact}/{len(wrench)} "
          f"({100*n_contact/len(wrench):.1f}%)")

    if n_contact >= 5:
        print(f"\n>>> SEATING SIGNATURE PRESENT (peak |Fz| = {np.abs(wrench).max():.1f} N) <<<")
    else:
        print(f"\n!! NO CLEAR CONTACT — peak |Fz| = {np.abs(wrench).max():.2f} N "
              f"(threshold {contact_threshold} N). Cable may not be seating.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to episode .npz, or filename if "
                                 "--download-from-modal")
    ap.add_argument("--download-from-modal", action="store_true",
                    help="Pull from Modal aic-data volume first")
    args = ap.parse_args()

    if args.download_from_modal:
        path = _download(args.path, Path("data/seating_episodes"))
    else:
        path = Path(args.path)
        if not path.exists():
            sys.exit(f"Not found: {path}")

    inspect(path)


if __name__ == "__main__":
    main()
