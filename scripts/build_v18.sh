#!/usr/bin/env bash
# v18: DINOv3-vits16 + ACT/HybridDAgger.
#
# Differs from build_v9.sh only in:
#   - DOCKERFILE points at Dockerfile.v18
#   - TAG defaults to v18
#   - HF_TOKEN is forwarded as a build-arg (DINOv3 is auth-required)
#   - ACT/loc weights default to dinov3 outputs from Modal
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/aic"
WEIGHTS_DST="$SRC/aic_example_policies/aic_example_policies/ros/weights"
ECR="973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs"
TAG="${TAG:-v18}"
DOCKERFILE="${DOCKERFILE:-$SRC/docker/aic_model/Dockerfile.v18}"

# DINOv3 weights (downloaded from Modal under the dinov3 tag).
ACT_PT="${ACT_PT:-$ROOT/data/outputs/act_dinov3_vits16_pretrain_lvd1689m_demos_with_images_30.pt}"
ACT_STATS="${ACT_STATS:-$ROOT/data/outputs/act_stats_dinov3_vits16_pretrain_lvd1689m_demos_with_images_30.npz}"
LOC_PT="${LOC_PT:-$ROOT/data/outputs/loc_dinov3_vits16_pretrain_lvd1689m_demos_with_images_30.pt}"
LOC_STATS="${LOC_STATS:-$ROOT/data/outputs/loc_stats_dinov3_vits16_pretrain_lvd1689m_demos_with_images_30.npz}"

export DOCKER_CONFIG="${DOCKER_CONFIG:-/tmp/dockercfg}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f ~/.cache/huggingface/token ]]; then
    HF_TOKEN="$(cat ~/.cache/huggingface/token)"
  else
    echo "Set HF_TOKEN (DINOv3 weights are auth-required)" >&2
    exit 1
  fi
fi

for f in "$ACT_PT" "$ACT_STATS" "$LOC_PT" "$LOC_STATS"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing: $f" >&2
    echo "Run: modal run scripts/train_modal_vision.py --backbone facebook/dinov3-vits16-pretrain-lvd1689m" >&2
    echo "Then: modal volume get aic-data /outputs/ data/outputs/" >&2
    exit 1
  fi
done

echo "==> Staging DINOv3 weights into build context"
mkdir -p "$WEIGHTS_DST"
cp "$ACT_PT" "$WEIGHTS_DST/act_policy.pt"
cp "$ACT_STATS" "$WEIGHTS_DST/act_norm_stats.npz"
cp "$LOC_PT" "$WEIGHTS_DST/port_localizer_dinov2.pt"
cp "$LOC_STATS" "$WEIGHTS_DST/port_localizer_stats.npz"
[[ -f "$ROOT/data/outputs/scripted_traj.npz" ]] && \
  cp "$ROOT/data/outputs/scripted_traj.npz" "$WEIGHTS_DST/scripted_traj.npz"
[[ -f "$ROOT/data/outputs/mlp_policy_best.pt" ]] && \
  cp "$ROOT/data/outputs/mlp_policy_best.pt" "$WEIGHTS_DST/mlp_policy_best.pt" && \
  cp "$ROOT/data/outputs/norm_stats.npz" "$WEIGHTS_DST/norm_stats.npz"

echo "==> Building my-solution:$TAG (DINOv3, $DOCKERFILE)"
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

echo "Done. Eval with: POLICY_OVERRIDE=DINOv2HybridDAgger ./scripts/gcp_eval.sh $TAG"
