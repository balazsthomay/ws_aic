#!/usr/bin/env bash
# v25: OnlineIK — runtime IK + DINO localizer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/aic"
WEIGHTS_DST="$SRC/aic_example_policies/aic_example_policies/ros/weights"
ECR="973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs"
TAG="${TAG:-v25}"
DOCKERFILE="${DOCKERFILE:-$SRC/docker/aic_model/Dockerfile.v25}"

export DOCKER_CONFIG="${DOCKER_CONFIG:-/tmp/dockercfg}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f ~/.cache/huggingface/token ]]; then
    HF_TOKEN="$(cat ~/.cache/huggingface/token)"
  else
    echo "Set HF_TOKEN (DINOv3 weights are auth-required)" >&2
    exit 1
  fi
fi

# Required: v25 localizer (downloaded from Modal under loc_v25 alias).
if [[ ! -f "$WEIGHTS_DST/port_localizer_dinov2.pt" || ! -f "$WEIGHTS_DST/port_localizer_stats.npz" ]]; then
  echo "Missing localizer weights at $WEIGHTS_DST" >&2
  echo "Stage with: cp data/outputs/loc_v25.pt $WEIGHTS_DST/port_localizer_dinov2.pt" >&2
  exit 1
fi

# Ensure scripted_traj.npz exists (carried forward as fallback for ScriptedVision).
if [[ ! -f "$WEIGHTS_DST/scripted_traj.npz" ]]; then
  if [[ -f "$ROOT/data/outputs/scripted_traj.npz" ]]; then
    cp "$ROOT/data/outputs/scripted_traj.npz" "$WEIGHTS_DST/scripted_traj.npz"
  fi
fi

echo "==> Building my-solution:$TAG ($DOCKERFILE)"
export DOCKER_BUILDKIT=0
docker build --platform linux/amd64 \
  --build-arg HF_TOKEN="$HF_TOKEN" \
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

echo "==> Cleanup"
docker rmi "$ECR:$TAG" >/dev/null 2>&1 || true
[[ "$TAG" != "v1" ]] && docker rmi "my-solution:$TAG" >/dev/null 2>&1 || true

echo "Done. Eval with: ./scripts/gcp_eval.sh $TAG"
