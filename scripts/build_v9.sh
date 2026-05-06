#!/usr/bin/env bash
# Build, tag, and push the v9 (DINOv2 + ACT) submission image to ECR.
#
# Usage:
#   ./scripts/build_v9.sh
#
# Prereqs:
#   - my-solution:v1 already built locally (the Dockerfile FROM target)
#   - data/outputs/act_policy.pt + act_norm_stats.npz from train_dinov2_act.py
#   - aws --profile aic configured
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/aic"
WEIGHTS_DST="$SRC/aic_example_policies/aic_example_policies/ros/weights"
ACT_PT="$ROOT/data/outputs/act_policy.pt"
ACT_STATS="$ROOT/data/outputs/act_norm_stats.npz"
ECR="973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs"
TAG="${TAG:-v9}"
# Use temp docker config to bypass macOS keychain credential helper hang
export DOCKER_CONFIG="${DOCKER_CONFIG:-/tmp/dockercfg}"

if [[ ! -f "$ACT_PT" || ! -f "$ACT_STATS" ]]; then
  echo "Missing trained weights. Run: .venv/bin/python3 scripts/train_dinov2_act.py all" >&2
  exit 1
fi

echo "==> Staging trained weights into build context"
mkdir -p "$WEIGHTS_DST"
cp "$ACT_PT" "$WEIGHTS_DST/act_policy.pt"
cp "$ACT_STATS" "$WEIGHTS_DST/act_norm_stats.npz"
# Also stage the deterministic IK trajectory used by ScriptedPlay fallback
if [[ -f "$ROOT/data/outputs/scripted_traj.npz" ]]; then
  cp "$ROOT/data/outputs/scripted_traj.npz" "$WEIGHTS_DST/scripted_traj.npz"
fi
# Port localizer + DAgger MLP weights for the DINOv2HybridDAgger policy
if [[ -f "$ROOT/data/outputs/port_localizer_dinov2.pt" ]]; then
  cp "$ROOT/data/outputs/port_localizer_dinov2.pt" "$WEIGHTS_DST/port_localizer_dinov2.pt"
  cp "$ROOT/data/outputs/port_localizer_stats.npz" "$WEIGHTS_DST/port_localizer_stats.npz"
fi
# DAgger MLP (already in repo at this path normally; mirror just in case)
if [[ -f "$ROOT/data/outputs/mlp_policy_best.pt" ]]; then
  cp "$ROOT/data/outputs/mlp_policy_best.pt" "$WEIGHTS_DST/mlp_policy_best.pt"
  cp "$ROOT/data/outputs/norm_stats.npz" "$WEIGHTS_DST/norm_stats.npz"
fi

DOCKERFILE="${DOCKERFILE:-$SRC/docker/aic_model/Dockerfile.v9}"
echo "==> Building my-solution:$TAG (from $DOCKERFILE)"
# Use legacy builder (no buildx) so this works under temp DOCKER_CONFIG.
# Dockerfile.v9 was updated to use a separate RUN chmod step instead of
# COPY --chmod, which keeps it compatible with the legacy builder.
export DOCKER_BUILDKIT=0
docker build --platform linux/amd64 \
  -f "$DOCKERFILE" \
  -t "my-solution:$TAG" \
  "$SRC"

echo "==> ECR login"
mkdir -p "$DOCKER_CONFIG"
aws --profile aic ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
      973918476471.dkr.ecr.us-east-1.amazonaws.com

echo "==> Tag + push"
docker tag "my-solution:$TAG" "$ECR:$TAG"
docker push "$ECR:$TAG"

echo "==> Cleanup: drop local image tags (keep my-solution:v1 as base)"
docker rmi "$ECR:$TAG" >/dev/null 2>&1 || true
if [[ "$TAG" != "v1" ]]; then
  docker rmi "my-solution:$TAG" >/dev/null 2>&1 || true
fi

echo
echo "Done. Eval with: ./scripts/gcp_eval.sh $TAG"
