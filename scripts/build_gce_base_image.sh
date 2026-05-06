#!/usr/bin/env bash
# Build a custom GCE image with Docker + AWS CLI + aic_eval:latest pre-pulled.
# Saves ~5 min per gcp_eval run (skips apt install + 9GB ghcr pull).
#
# Usage:
#   ./scripts/build_gce_base_image.sh
#
# Output: GCE image `aic-eval-base` in your project. Then update
# gcp_eval.sh to use --image=aic-eval-base instead of the Ubuntu family.
set -euo pipefail

GCLOUD=${GCLOUD:-/opt/homebrew/share/google-cloud-sdk/bin/gcloud}
ZONE="${ZONE:-us-central1-a}"
INSTANCE="aic-base-builder-$(date +%s)"
IMAGE_NAME="aic-eval-base"

# Setup script that runs on the VM
SETUP=$(cat <<'EOF'
#!/bin/bash
set -ex
exec > >(tee /var/log/aic_base_setup.log) 2>&1

apt-get update -y
apt-get install -y ca-certificates curl gnupg unzip

# AWS CLI v2 (not in Ubuntu 24.04 apt)
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install --update
rm -rf /tmp/awscli*

# Docker
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu noble stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable --now docker

# Pre-pull aic_eval:latest (public ghcr) — saves ~5 min per eval
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest

# Trim apt caches and unused stuff to keep the image small
apt-get clean
rm -rf /var/lib/apt/lists/*

touch /var/log/aic_base_setup.log
echo "AIC_BASE_SETUP_DONE"
EOF
)

echo "==> Provisioning builder VM $INSTANCE in $ZONE"
"$GCLOUD" compute instances create "$INSTANCE" \
  --zone="$ZONE" \
  --machine-type=c3-standard-8 \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=60GB \
  --boot-disk-type=pd-balanced \
  --metadata-from-file=startup-script=<(echo "$SETUP")

cleanup() {
  echo "==> Cleanup builder VM"
  "$GCLOUD" compute instances delete "$INSTANCE" --zone="$ZONE" --quiet >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> Waiting for setup script (poll serial console)"
for i in $(seq 1 40); do
  sleep 30
  if "$GCLOUD" compute instances get-serial-port-output "$INSTANCE" --zone="$ZONE" 2>/dev/null \
        | grep -q "AIC_BASE_SETUP_DONE"; then
    echo "==> Setup done"
    break
  fi
  echo "  ($i) still running ..."
done

echo "==> Stopping VM (required to snapshot disk)"
"$GCLOUD" compute instances stop "$INSTANCE" --zone="$ZONE"

echo "==> Deleting any prior version of $IMAGE_NAME"
"$GCLOUD" compute images delete "$IMAGE_NAME" --quiet >/dev/null 2>&1 || true

echo "==> Creating image $IMAGE_NAME from disk $INSTANCE"
"$GCLOUD" compute images create "$IMAGE_NAME" \
  --source-disk="$INSTANCE" \
  --source-disk-zone="$ZONE" \
  --family=aic-eval

echo "==> Done. Update gcp_eval.sh: --image=$IMAGE_NAME (no --image-family/project)."
