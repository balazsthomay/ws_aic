#!/bin/bash
# v7: simplified diagnostic entrypoint.
#
# - Drops multicast scouting (may have caused v6 early exit on cluster)
# - Uses v2/v4-style custom config that's PROVEN to work locally
# - Tries 3 candidate connect endpoints (was 8 in v6)
# - POSTs hostname/env/network to AIC_DIAG_URL on startup
# - Does NOT sleep on aic_model exit (let it die naturally so we can
#   see the actual cluster failure mode in the cluster's portal)
# - Pairs with InstrumentedSpeedDemon which POSTs at every lifecycle stage

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

post_diag() {
  local event="$1"; shift
  if [[ -z "$AIC_DIAG_URL" ]]; then return 0; fi
  local payload="{\"event\":\"$event\",\"host\":\"$(hostname)\",\"ts\":$(date +%s)$*}"
  curl -s -m 5 -X POST -H "Content-Type: application/json" \
    -d "$payload" "$AIC_DIAG_URL" >/dev/null 2>&1
}

echo "==== v7 entrypoint start ===="
date -u
hostname
env | grep -iE "(AIC_|ZENOH_|RMW_|ROS_|HOSTNAME)" | sort

# POST a structured startup diagnostic with key env values.
ENV_DUMP=$(env | grep -iE "(AIC_|ZENOH_|RMW_|HOSTNAME|HOME|PATH)" \
  | sed 's/"/\\"/g' | tr '\n' '|' | sed 's/|$//')
HOSTS=$(cat /etc/hosts 2>/dev/null | tr '\n' '|' | sed 's/"/\\"/g')
post_diag "startup" \
  ",\"router_addr\":\"${AIC_ROUTER_ADDR:-unset}\"" \
  ",\"acl\":\"${AIC_ENABLE_ACL:-unset}\"" \
  ",\"env\":\"$ENV_DUMP\"" \
  ",\"etc_hosts\":\"$HOSTS\""

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
  : "${AIC_MODEL_PASSWD:=CHANGE_IN_PROD}"
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

# Multiple candidate endpoints. localhost first (canonical assumption),
# then AIC_ROUTER_ADDR if set, then docker-compose service name.
ENDPOINT_LIST='"tcp/localhost:7447"'
if [[ -n "$AIC_ROUTER_ADDR" && "$AIC_ROUTER_ADDR" != "localhost:7447" ]]; then
  ENDPOINT_LIST="$ENDPOINT_LIST, \"tcp/$AIC_ROUTER_ADDR\""
fi
ENDPOINT_LIST="$ENDPOINT_LIST, \"tcp/eval:7447\""

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
    endpoints: [$ENDPOINT_LIST],
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
echo "==== zenoh session config ===="
cat $ZENOH_SESSION_CONFIG_URI

post_diag "before_aic_model" ",\"endpoints\":\"$ENDPOINT_LIST\""

echo "==== launching aic_model ===="
pixi run --as-is ros2 run aic_model aic_model "$@"
RC=$?
echo "==== aic_model exited with code $RC ===="
post_diag "aic_model_exit" ",\"code\":$RC"
exit $RC
