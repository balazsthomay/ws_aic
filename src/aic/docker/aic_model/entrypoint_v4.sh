#!/bin/bash
# v4: same approach as v2 (which worked locally with and without ACL),
# but defaults AIC_ROUTER_ADDR to localhost:7447 instead of erroring.
# Cluster likely doesn't set AIC_ROUTER_ADDR (router and model share
# network namespace), so v2's hard-fail on missing AIC_ROUTER_ADDR is
# the most likely cause of the 133s "Failed" status on the cluster.

# DO NOT use `set -e` so a small bash issue cannot silently kill the
# container; we want any error to be visible at runtime instead.

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# Cluster-friendly default: assume router on localhost (canonical pattern).
: "${AIC_ROUTER_ADDR:=localhost:7447}"
echo "AIC_ROUTER_ADDR=$AIC_ROUTER_ADDR"

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
  : "${AIC_MODEL_PASSWD:=CHANGE_IN_PROD}"
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
    timeout_ms: -1,
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

exec pixi run --as-is ros2 run aic_model aic_model "$@"
