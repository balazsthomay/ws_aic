#!/usr/bin/env bash
# v39: ACT trained on perturbed-and-corrected seating demos with FORCE in state.
# Inference policy: DINOv2ACT (DINOv3 + ACT, 32-D state).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/aic"
WEIGHTS_DST="$SRC/aic_example_policies/aic_example_policies/ros/weights"
ECR="973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs"
TAG="${TAG:-v39}"
DOCKERFILE="${DOCKERFILE:-$SRC/docker/aic_model/Dockerfile.v39}"

export DOCKER_CONFIG="${DOCKER_CONFIG:-/tmp/dockercfg}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f ~/.cache/huggingface/token ]]; then
    HF_TOKEN="$(cat ~/.cache/huggingface/token)"
  else
    echo "Set HF_TOKEN (DINOv3 weights are auth-required)" >&2
    exit 1
  fi
fi

# Localizer weights (carry-forward from v26 for OnlineIK fallback)
for f in port_localizer_dinov2.pt port_localizer_stats.npz; do
  if [[ ! -f "$WEIGHTS_DST/$f" ]]; then
    echo "Missing $WEIGHTS_DST/$f (re-stage from v26 build)" >&2
    exit 1
  fi
done

# ACT weights (NEW for v39 — must be trained first via train_modal_vision.py)
for f in act_policy.pt act_norm_stats.npz; do
  if [[ ! -f "$WEIGHTS_DST/$f" ]]; then
    echo "Missing $WEIGHTS_DST/$f" >&2
    echo "  Train first: modal run --detach scripts/train_modal_vision.py --demos-name demos_seat_v2.npz --tag v39_seat" >&2
    echo "  Then download: modal volume get aic-data outputs/act_v39_seat.pt $WEIGHTS_DST/act_policy.pt --force" >&2
    echo "                 modal volume get aic-data outputs/act_stats_v39_seat.npz $WEIGHTS_DST/act_norm_stats.npz --force" >&2
    exit 1
  fi
done

# Pull base if not present
if ! docker image inspect my-solution:v1 >/dev/null 2>&1; then
  echo "==> Pulling my-solution:v1 from ECR (base layer)"
  aws --profile aic ecr get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin \
        973918476471.dkr.ecr.us-east-1.amazonaws.com
  docker pull --platform linux/amd64 "$ECR:v1"
  docker tag "$ECR:v1" my-solution:v1
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

echo "Done. Submit to cluster (or eval locally with ./scripts/gcp_eval.sh $TAG)"
