#!/usr/bin/env bash
# Post-bulk: download → aggregate → upload to Modal → train ACT → download weights
# Run after bulk collection completes.
#
# Usage: ./scripts/post_bulk_train_v38.sh
#
# Prereqs:
#   - Episodes in Modal volume aic-data:/seating_episodes
#   - HF_TOKEN (for DINOv3) in env or ~/.cache/huggingface/token
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WEIGHTS_DIR="$ROOT/src/aic/aic_example_policies/aic_example_policies/ros/weights"
DEMOS_NAME="demos_seat_v1.npz"
TAG="v38_seat"

echo "=== [1/5] Aggregate per-episode .npz from Modal volume → demos_seat_v1.npz ==="
# 200 eps × max_T 1024 × full image dims would OOM. Cap at 600 ticks
# (last 30s — keeps the descent + push) and 2× downsample images
# (256x288 → 128x144). Reduces peak RAM ~10× to ~20 GB.
uv run python scripts/aggregate_seating_demos.py \
    --download-from-modal \
    --in data/seating_episodes \
    --out "data/outputs/${DEMOS_NAME}" \
    --max-t-cap 600 \
    --downsample 2
ls -lh "data/outputs/${DEMOS_NAME}"

echo
echo "=== [2/5] Upload demos to Modal volume aic-data:/data/${DEMOS_NAME} ==="
modal volume put aic-data "data/outputs/${DEMOS_NAME}" "/data/${DEMOS_NAME}" --force

echo
echo "=== [3/5] Train ACT on Modal (T4 GPU, ~25-30 min) ==="
# --detach so this script's bash-timeout doesn't kill the training mid-run.
# Then poll the volume for the trained weight files.
uv run modal run --detach scripts/train_modal_vision.py \
    --demos-name "${DEMOS_NAME}" \
    --backbone facebook/dinov3-vits16-pretrain-lvd1689m \
    --tag "${TAG}"
echo "  [poll] Waiting for trained ACT weights to land in volume..."
WEIGHT_PATH="outputs/act_${TAG}.pt"
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
echo "=== [5/5] Build v38 image and push to ECR ==="
TAG=v38 ./scripts/build_v38.sh

echo
echo "=== DONE ==="
echo "v38 image pushed. User can now submit to cluster."
