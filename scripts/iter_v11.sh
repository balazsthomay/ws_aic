#!/usr/bin/env bash
# After collect_50 finishes, merge with existing 30 → 80 demos, cache patch
# tokens, train ACT, build + push as v11, kick off GCP eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEMOS_30="data/demos_with_images_30.npz"
DEMOS_50="data/demos_with_images_50.npz"
DEMOS_80="data/demos_with_images_80.npz"
FEATS_PATH="data/outputs/dinov2_patches_30.npz"
FEATS_BACKUP="data/outputs/dinov2_patches_30_orig.npz"

if [[ ! -f "$DEMOS_50" ]]; then
  echo "Missing $DEMOS_50 — wait for collect_50 to finish" >&2
  exit 1
fi

echo "==> Merge demos"
.venv/bin/python3 scripts/merge_demos.py "$DEMOS_30" "$DEMOS_50" "$DEMOS_80"

# Re-cache patches from 80-episode demo set. The cache step skips if the file
# exists; rotate the file path or remove old cache first.
if [[ -f "$FEATS_PATH" ]]; then
  echo "==> Backing up existing 30-demo cache"
  mv "$FEATS_PATH" "$FEATS_BACKUP"
fi

echo "==> Re-cache patches for 80 episodes"
# Temporarily point train script at the 80-demo file
DATA_PATH="$DEMOS_80" \
  .venv/bin/python3 -u scripts/train_dinov2_act.py cache --mode patches --cache-batch-size 16

echo "==> Train ACT (patches+aug, 8000 epochs)"
STATE_NOISE=0.05 VISION_NOISE=0.02 \
  .venv/bin/python3 -u scripts/train_dinov2_act.py train --mode patches --epochs 8000 --batch-size 128 --lr 1e-4

echo "==> Quick local eval (5 trials)"
.venv/bin/python3 scripts/eval_dinov2_act.py --trials 5 --seed 42

echo "==> Build + push as v11"
TAG=v11 ./scripts/build_v9.sh

echo "==> GCP eval v11"
./scripts/gcp_eval.sh v11
