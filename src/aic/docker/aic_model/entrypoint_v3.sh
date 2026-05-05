#!/bin/bash
# v3 entrypoint: mirror the canonical zenoh_config_model_session.sh pattern.
# Use the canonical aic_zenoh_config.json5 as the session config; only
# tweak with ZENOH_CONFIG_OVERRIDE for things that work as scalar overrides
# (auth credentials).
#
# If AIC_ROUTER_ADDR is set AND points to something other than the canonical
# default (localhost:7447), sed-replace the endpoint in a copy of the config.
# Otherwise, use the canonical config unchanged — same as cluster expects.

# DO NOT use `set -e`: a single bash error must not crash the container
# silently. Log everything we can.

# Sanity: we need the canonical config baked into the image.
CANONICAL_CONFIG=/aic_zenoh_config.json5
if [[ ! -f "$CANONICAL_CONFIG" ]]; then
  echo "FATAL: canonical zenoh config $CANONICAL_CONFIG not found in image"
  ls -la / 2>&1
  exit 2
fi

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# Build the session config: copy canonical, optionally retarget the endpoint.
SESSION_CONFIG=/tmp/zenoh_session_config.json5
cp "$CANONICAL_CONFIG" "$SESSION_CONFIG"

if [[ -n "$AIC_ROUTER_ADDR" && "$AIC_ROUTER_ADDR" != "localhost:7447" ]]; then
  echo "Retargeting Zenoh router endpoint to $AIC_ROUTER_ADDR"
  sed -i "s|tcp/localhost:7447|tcp/$AIC_ROUTER_ADDR|g" "$SESSION_CONFIG"
fi

export ZENOH_SESSION_CONFIG_URI="$SESSION_CONFIG"
echo "ZENOH_SESSION_CONFIG_URI=$ZENOH_SESSION_CONFIG_URI"

# Auth credentials via override (this kind of override works; only array
# overrides like connect/endpoints are silently broken).
should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
  if [[ -z "$AIC_MODEL_PASSWD" ]]; then
    echo "WARNING: AIC_ENABLE_ACL set but AIC_MODEL_PASSWD missing; using placeholder"
    AIC_MODEL_PASSWD="CHANGE_IN_PROD"
  fi
  ZCO='transport/auth/usrpwd/user="model"'
  ZCO+=";transport/auth/usrpwd/password=\"$AIC_MODEL_PASSWD\""
  ZCO+=';transport/shared_memory/enabled=false'
  export ZENOH_CONFIG_OVERRIDE="$ZCO"
else
  export ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=false'
fi
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"

exec pixi run --as-is ros2 run aic_model aic_model "$@"
