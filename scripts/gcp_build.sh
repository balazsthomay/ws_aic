#!/usr/bin/env bash
# Build the my-solution:vN image on a GCP spot VM (no local CPU/QEMU).
# Mirrors gcp_eval.sh's lifecycle: spin up → SCP context → build+push → terminate.
#
# Usage:
#   ./scripts/gcp_build.sh v44
#
# Cost: ~$0.02/build (c3-standard-8 spot, ~8 min total).
# Requires: aic-eval-base GCE image (docker + aws-cli pre-installed),
#           local AWS profile 'aic' with ECR push perms,
#           HF_TOKEN env var or ~/.cache/huggingface/token.
set -euo pipefail

GCLOUD=${GCLOUD:-/opt/homebrew/share/google-cloud-sdk/bin/gcloud}
TAG="${1:?usage: $0 vN}"
ZONE="${ZONE:-us-east1-b}"
MACHINE="${MACHINE:-c3-standard-8}"
IMAGE="${IMAGE:-aic-eval-base}"
INSTANCE="aic-build-$(date +%s)-$RANDOM"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/aic"
DOCKERFILE_LOCAL="$SRC/docker/aic_model/Dockerfile.$TAG"
[[ -f "$DOCKERFILE_LOCAL" ]] || { echo "missing $DOCKERFILE_LOCAL" >&2; exit 1; }

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f ~/.cache/huggingface/token ]]; then
    HF_TOKEN="$(cat ~/.cache/huggingface/token)"
  else
    echo "Set HF_TOKEN (DINOv3 weights are auth-required)" >&2
    exit 1
  fi
fi

ECR="973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs"

# Tarball the build context
TARBALL="/tmp/aic_build_${TAG}.tgz"
echo "==> Tarring build context from $SRC"
tar czf "$TARBALL" -C "$SRC" .
echo "    $(du -h "$TARBALL" | awk '{print $1}')"

# AWS creds for ECR — same approach as gcp_eval.sh
CREDS=$(aws --profile aic configure export-credentials)
AWS_AKID=$(echo "$CREDS" | python3 -c "import sys,json;print(json.load(sys.stdin)['AccessKeyId'])")
AWS_SAK=$(echo "$CREDS" | python3 -c "import sys,json;print(json.load(sys.stdin)['SecretAccessKey'])")

# Try multiple zones for spot capacity
ZONES=(us-east1-b us-east1-c us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b)
ZONE_PICKED=""
for Z in "${ZONES[@]}"; do
  echo "==> Trying $INSTANCE in $Z ($MACHINE)"
  if "$GCLOUD" compute instances create "$INSTANCE" \
       --zone="$Z" \
       --machine-type="$MACHINE" \
       --image="$IMAGE" \
       --boot-disk-size=80GB \
       --boot-disk-type=pd-balanced \
       --provisioning-model=SPOT \
       --instance-termination-action=DELETE 2>/tmp/gcp_build_create.err; then
    ZONE_PICKED="$Z"
    echo "==> Created in $Z"
    break
  fi
  if grep -q "ZONE_RESOURCE_POOL_EXHAUSTED" /tmp/gcp_build_create.err; then
    echo "  $Z exhausted, trying next..."
    continue
  fi
  echo "  fatal error in $Z:"
  cat /tmp/gcp_build_create.err
  exit 1
done
[[ -n "$ZONE_PICKED" ]] || { echo "==> all zones exhausted" >&2; exit 1; }
ZONE="$ZONE_PICKED"

cleanup() {
  echo "==> Deleting VM $INSTANCE"
  "$GCLOUD" compute instances delete "$INSTANCE" --zone="$ZONE" --quiet >/dev/null 2>&1 || true
  rm -f "$TARBALL"
}
trap cleanup EXIT

# Wait for SSH to come up
echo "==> Waiting for SSH..."
for i in {1..30}; do
  if "$GCLOUD" compute ssh "$INSTANCE" --zone="$ZONE" --command="true" 2>/dev/null; then
    echo "    SSH ready after ${i} attempts"
    break
  fi
  sleep 5
done

# SCP build context
echo "==> SCP build context"
"$GCLOUD" compute scp --zone="$ZONE" "$TARBALL" "$INSTANCE:/tmp/build.tgz" 2>&1 | tail -3

# Run build + push remotely. Use sudo for all docker commands — the SSH
# user on aic-eval-base isn't in the docker group (the gcp_eval.sh startup
# script runs as root, that's why it doesn't need sudo).
echo "==> Building $ECR:$TAG remotely (native linux/amd64)"
"$GCLOUD" compute ssh "$INSTANCE" --zone="$ZONE" --command="bash -s" <<EOF
set -ex
mkdir -p /tmp/build
tar xzf /tmp/build.tgz -C /tmp/build

export AWS_ACCESS_KEY_ID=$AWS_AKID
export AWS_SECRET_ACCESS_KEY=$AWS_SAK
export AWS_DEFAULT_REGION=us-east-1

aws ecr get-login-password --region us-east-1 \
  | sudo docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com

sudo docker pull $ECR:v1
sudo docker tag $ECR:v1 my-solution:v1

sudo docker build \
  --build-arg HF_TOKEN=$HF_TOKEN \
  -f /tmp/build/docker/aic_model/Dockerfile.$TAG \
  -t my-solution:$TAG \
  /tmp/build

sudo docker tag my-solution:$TAG $ECR:$TAG
sudo docker push $ECR:$TAG

echo BUILD_DONE
EOF

echo "==> $ECR:$TAG pushed"
