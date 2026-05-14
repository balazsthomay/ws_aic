#!/usr/bin/env bash
# v39 post-bulk: download → aggregate (filter seated) → upload to Modal →
# train ACT (32-D state with wrench) → download weights → build v39 image.
#
# Usage: ./scripts/post_bulk_train_v39.sh
#
# Prereqs:
#   - Episodes in Modal volume aic-data:/seating_episodes (collected via
#     `modal run scripts/modal_collect_demos.py::collect --n 200`)
#   - HF_TOKEN (for DINOv3) in env or ~/.cache/huggingface/token
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WEIGHTS_DIR="$ROOT/src/aic/aic_example_policies/aic_example_policies/ros/weights"
DEMOS_NAME="demos_seat_v2.npz"
TAG="v39_seat"

echo "=== [1/5] Aggregate per-episode .npz from Modal → ${DEMOS_NAME} ==="
# 200 eps × max_T 1100 × full image dims would OOM. Cap at last 600 ticks
# (covers correction + descent + push + hold, drops most of the approach)
# and 2× downsample images (256x288 → 128x144). --filter-seated drops
# episodes that didn't seat (push bottomed out OR delta_tail_fz > 5N).
uv run python scripts/aggregate_seating_demos.py \
    --download-from-modal \
    --in data/seating_episodes \
    --out "data/outputs/${DEMOS_NAME}" \
    --max-t-cap 600 \
    --downsample 2 \
    --filter-seated \
    --state-dim 32
ls -lh "data/outputs/${DEMOS_NAME}"

echo
echo "=== [2/5] Upload demos to Modal volume aic-data:/data/${DEMOS_NAME} ==="
modal volume put aic-data "data/outputs/${DEMOS_NAME}" "/data/${DEMOS_NAME}" --force

echo
echo "=== [3/5] Train ACT on Modal (T4, ~25-30 min) ==="
# --detach so the local bash-timeout doesn't kill the training mid-run.
# train_modal_vision.py auto-detects state_dim from the demos shape (32-D
# for v39).
uv run modal run --detach scripts/train_modal_vision.py \
    --demos-name "${DEMOS_NAME}" \
    --backbone facebook/dinov3-vits16-pretrain-lvd1689m \
    --tag "${TAG}"
echo "  [poll] Waiting for act_${TAG}.pt to land in aic-data:/outputs..."
for i in $(seq 1 60); do
    if modal volume ls aic-data outputs 2>/dev/null | grep -q "act_${TAG}.pt"; then
        echo "  [poll] Found act_${TAG}.pt after ${i}min"
        break
    fi
    sleep 60
done

echo
echo "=== [4/5] Download trained ACT weights ==="
modal volume get aic-data "outputs/act_${TAG}.pt" "${WEIGHTS_DIR}/act_policy.pt" --force
modal volume get aic-data "outputs/act_stats_${TAG}.npz" "${WEIGHTS_DIR}/act_norm_stats.npz" --force
ls -lh "${WEIGHTS_DIR}/act_policy.pt" "${WEIGHTS_DIR}/act_norm_stats.npz"

echo
echo "=== [5/5] Build v39 image and push to ECR ==="
TAG=v39 ./scripts/build_v39.sh

echo
echo "=== DONE ==="
echo "v39 image pushed. Submit to cluster (or eval locally with ./scripts/gcp_eval.sh v39)"
