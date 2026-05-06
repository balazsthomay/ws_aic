#!/usr/bin/env python3
"""Concatenate two demo .npz files (with images) into one larger dataset.

Usage:
    .venv/bin/python3 scripts/merge_demos.py \\
        data/demos_with_images_30.npz \\
        data/demos_with_images_50.npz \\
        data/demos_with_images_80.npz
"""

import sys
from pathlib import Path

import numpy as np

if len(sys.argv) < 4:
    sys.exit("usage: merge_demos.py IN1.npz IN2.npz OUT.npz")

p1, p2, out = map(Path, sys.argv[1:4])
print(f"Loading {p1}…")
a = np.load(p1)
print(f"Loading {p2}…")
b = np.load(p2)

# Pad shorter to longer length per array (states/actions/images)
def cat(k, *arrs):
    if k in ("episode_lengths", "episode_success", "episode_params"):
        return np.concatenate(arrs, axis=0)
    if k.startswith("images_") or k in ("states", "actions"):
        # axis 0 = episodes; axis 1 = T (may differ)
        n_pad = max(x.shape[1] for x in arrs)
        padded = []
        for x in arrs:
            if x.shape[1] < n_pad:
                pad = np.zeros((x.shape[0], n_pad - x.shape[1], *x.shape[2:]),
                               dtype=x.dtype)
                x = np.concatenate([x, pad], axis=1)
            padded.append(x)
        return np.concatenate(padded, axis=0)
    return arrs[0]   # state_labels / action_labels: just keep first

merged = {}
for key in a.keys():
    if key in b:
        merged[key] = cat(key, a[key], b[key])
        if hasattr(merged[key], "shape"):
            print(f"  {key}: {merged[key].shape}")
    else:
        merged[key] = a[key]

print(f"Saving {out}…")
np.savez_compressed(out, **merged)
print(f"Saved ({out.stat().st_size/1e6:.1f} MB)")
