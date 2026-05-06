#!/usr/bin/env bash
# Run the AIC eval on a fresh GCP Compute Engine VM and return scores.
#
# Prereqs (one-time):
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT
#   gcloud services enable compute.googleapis.com
#
# Usage:
#   ./scripts/gcp_eval.sh v8

set -euo pipefail

GCLOUD=${GCLOUD:-/opt/homebrew/share/google-cloud-sdk/bin/gcloud}
MODEL_TAG="${1:-v8}"
ZONE="${ZONE:-us-east1-b}"
MACHINE="${MACHINE:-c3-standard-8}"
IMAGE="${IMAGE:-aic-eval-base}"   # custom image: docker + aws-cli + aic_eval pre-pulled
INSTANCE="aic-eval-$(date +%s)-$RANDOM"

# AWS creds for ECR (read once from local 'aic' profile, passed via instance metadata)
CREDS=$(aws --profile aic configure export-credentials)
AWS_AKID=$(echo "$CREDS" | python3 -c "import sys,json;print(json.load(sys.stdin)['AccessKeyId'])")
AWS_SAK=$(echo "$CREDS" | python3 -c "import sys,json;print(json.load(sys.stdin)['SecretAccessKey'])")

# Startup script that runs on the VM. Installs docker, runs compose, exits.
STARTUP=$(cat <<EOF
#!/bin/bash
set -ex
exec > >(tee /var/log/aic_eval.log) 2>&1

# Image already has docker + aws-cli + aic_eval:latest baked in via aic-eval-base.

# ECR login
export AWS_ACCESS_KEY_ID=$AWS_AKID
export AWS_SECRET_ACCESS_KEY=$AWS_SAK
export AWS_DEFAULT_REGION=us-east-1
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com

# aic_eval is already in the image. Only need to pull the model image.
echo "[gcp_eval] pulling model image..."
docker pull 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs:$MODEL_TAG
echo "[gcp_eval] pulls done"

# Compose file (write line-by-line; nested heredocs in metadata-script can break)
{
  echo 'name: aic'
  echo 'services:'
  echo '  eval:'
  echo '    image: ghcr.io/intrinsic-dev/aic/aic_eval'
  echo '    command: gazebo_gui:=false launch_rviz:=false ground_truth:=false start_aic_engine:=true shutdown_on_aic_engine_exit:=true'
  echo '    networks: [default]'
  echo '    environment:'
  echo '      AIC_EVAL_PASSWD: CHANGE_IN_PROD'
  echo '      AIC_MODEL_PASSWD: CHANGE_IN_PROD'
  echo '      AIC_ENABLE_ACL: "true"'
  echo '  model:'
  echo "    image: 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs:$MODEL_TAG"
  : "${POLICY_OVERRIDE:=}"
  if [[ -n "$POLICY_OVERRIDE" ]]; then
    echo "    command: ['--ros-args', '-p', 'policy:=aic_example_policies.ros.${POLICY_OVERRIDE}']"
  fi
  echo '    networks: [default]'
  echo '    environment:'
  echo '      RMW_IMPLEMENTATION: rmw_zenoh_cpp'
  echo '      ZENOH_ROUTER_CHECK_ATTEMPTS: "-1"'
  echo '      AIC_ROUTER_ADDR: eval:7447'
  echo '      AIC_MODEL_PASSWD: CHANGE_IN_PROD'
  echo '      AIC_ENABLE_ACL: "true"'
  echo 'networks:'
  echo '  default:'
  echo '    internal: true'
} > /tmp/compose.yaml

echo "[gcp_eval] starting compose"
docker compose -f /tmp/compose.yaml up --abort-on-container-exit || true
docker compose -f /tmp/compose.yaml down
echo "AIC_EVAL_DONE"
EOF
)

ZONES=(us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-b us-west1-a us-west1-b)
ZONE_PICKED=""
for Z in "${ZONES[@]}"; do
  echo "==> Trying VM $INSTANCE in $Z ($MACHINE)"
  if "$GCLOUD" compute instances create "$INSTANCE" \
       --zone="$Z" \
       --machine-type="$MACHINE" \
       --image="$IMAGE" \
       --boot-disk-size=60GB \
       --boot-disk-type=pd-balanced \
       --provisioning-model=SPOT \
       --instance-termination-action=DELETE \
       --metadata-from-file=startup-script=<(echo "$STARTUP") 2>/tmp/gcp_create.err; then
    ZONE_PICKED="$Z"
    echo "==> Created in $Z"
    break
  fi
  if grep -q "ZONE_RESOURCE_POOL_EXHAUSTED" /tmp/gcp_create.err; then
    echo "  $Z exhausted, trying next..."
    continue
  fi
  echo "  fatal error in $Z:"; cat /tmp/gcp_create.err; exit 1
done
if [[ -z "$ZONE_PICKED" ]]; then
  echo "==> all zones exhausted"; exit 1
fi
ZONE="$ZONE_PICKED"

cleanup() {
  echo "==> Deleting VM $INSTANCE"
  "$GCLOUD" compute instances delete "$INSTANCE" --zone="$ZONE" --quiet >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> Waiting for startup script (poll serial console)"
for i in $(seq 1 80); do
  sleep 30
  if "$GCLOUD" compute instances get-serial-port-output "$INSTANCE" --zone="$ZONE" 2>/dev/null | grep -q "AIC_EVAL_DONE"; then
    echo "==> Eval finished"
    break
  fi
  echo "  ($i) still running ..."
done

LOCAL_LOG="/tmp/aic_eval_${INSTANCE}.log"
echo "==> Copying full eval log to $LOCAL_LOG"
"$GCLOUD" compute scp --zone="$ZONE" "$INSTANCE:/var/log/aic_eval.log" "$LOCAL_LOG" 2>&1 | tail -3

echo
echo "===== TRIAL SCORE BREAKDOWN ====="
# Strip ANSI then pull each trial's scoring block
python3 - "$LOCAL_LOG" <<'PY'
import re, sys
ANSI = re.compile(r'\x1b\[[0-9;]*m')
text = open(sys.argv[1]).read()
text = ANSI.sub('', text)
m = re.search(r'(trial_1:.*?)(?=AIC_EVAL_DONE|\Z)', text, re.DOTALL)
print(m.group(1) if m else "(no trial breakdown found in log)")
PY

echo
echo "===== one-line summary ====="
grep -E "Trial '.*completed.*Score:" "$LOCAL_LOG" | sed -E 's/\x1b\[[0-9;]*m//g'
