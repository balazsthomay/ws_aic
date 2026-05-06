#!/usr/bin/env bash
# Generic train → build → push pipeline for a given demo set.
#
# Usage:
#   DATA_FILE=data/demos_with_images_80.npz \
#   FEATS_FILE=data/outputs/dinov2_patches_80.npz \
#   TAG=v12 \
#   ./scripts/iter_pipeline.sh
#
# Steps: cache patches → train ACT → train localizer → build + push image.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_FILE="${DATA_FILE:?missing DATA_FILE env}"
FEATS_FILE="${FEATS_FILE:?missing FEATS_FILE env}"
TAG="${TAG:?missing TAG env}"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "Missing $DATA_FILE" >&2; exit 1
fi

echo "==> Cache patches from $DATA_FILE → $FEATS_FILE"
DATA_PATH="$DATA_FILE" PATCHES_PATH="$FEATS_FILE" \
  .venv/bin/python3 -u scripts/train_dinov2_act.py cache --mode patches --cache-batch-size 16

echo "==> Train ACT (patches+aug) on $FEATS_FILE"
PATCHES_PATH="$FEATS_FILE" \
  .venv/bin/python3 -u scripts/train_dinov2_act.py train --mode patches \
    --epochs 8000 --batch-size 128 --lr 1e-4

echo "==> Train port localizer on $FEATS_FILE"
.venv/bin/python3 -u scripts/train_dinov2_port_localizer.py \
  --patches "$FEATS_FILE" --epochs 5000 --batch-size 256

echo "==> Quick local eval"
.venv/bin/python3 scripts/eval_dinov2_act.py --trials 5 --seed 42 || true

echo "==> Build + push as $TAG"
TAG="$TAG" ./scripts/build_v9.sh

echo "Done. Try: POLICY_OVERRIDE=DINOv2HybridDAgger ./scripts/gcp_eval.sh $TAG"
