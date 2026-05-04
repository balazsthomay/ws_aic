#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "$AIC_ROUTER_ADDR" ]]; then
  echo "AIC_ROUTER_ADDR must be provided"
  exit 1
fi

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

ACL_BLOCK=""
if should_enable_acl; then
  ACL_BLOCK=$(cat <<ACLEOF
,
  transport: {
    auth: {
      usrpwd: {
        user: "model",
        password: "$AIC_MODEL_PASSWD",
        dictionary_file: "/credentials.txt",
      },
    },
    shared_memory: { enabled: false },
  }
ACLEOF
)
else
  ACL_BLOCK=$(cat <<ACLEOF
,
  transport: { shared_memory: { enabled: false } }
ACLEOF
)
fi

cat > /tmp/zenoh_session_config.json5 <<ZEOF
{
  mode: "peer",
  connect: {
    endpoints: ["tcp/$AIC_ROUTER_ADDR"],
    timeout_ms: 10000,
  },
  listen: {
    endpoints: ["tcp/0.0.0.0:0"],
  },
  scouting: {
    multicast: { enabled: false },
    gossip: { enabled: true },
  }$ACL_BLOCK
}
ZEOF
export ZENOH_SESSION_CONFIG_URI=/tmp/zenoh_session_config.json5
echo "ZENOH_SESSION_CONFIG_URI=$ZENOH_SESSION_CONFIG_URI"
echo "--- session config ---"
cat $ZENOH_SESSION_CONFIG_URI
echo "--- end session config ---"
exec pixi run --as-is ros2 run aic_model aic_model "$@"
