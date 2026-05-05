#!/bin/bash
# v6: maximally permissive entrypoint.
#
# Cluster failure analysis: both upstream-canonical (v1, v5) AND our
# custom-config approach (v2, v4) failed. v2/v4 worked locally with
# both ACL on/off, v1/v5 failed locally. So cluster topology differs
# from both our docker-compose-with-DNS and shared-network-namespace
# assumptions.
#
# Strategy: provide MANY candidate connect endpoints in a Zenoh list.
# Zenoh peer mode tries them all. Whichever the cluster's router is
# reachable at will succeed. Also enable scouting/gossip for any
# additional discovery mechanism the cluster might support.
#
# Plus: dump diagnostics to stdout, never exit non-zero from the
# script itself, and wrap aic_model in a restart-then-sleep loop so
# the container stays alive long enough for the cluster's "Failed"
# log to capture meaningful info.

# DO NOT set -e: keep going on bash errors.

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

echo "==== v6 startup diagnostic ===="
date -u
hostname
echo "--- env (filtered) ---"
env | grep -iE "(AIC_|ZENOH_|RMW_|ROS_|HOSTNAME|DISPLAY|HOME|PATH)" | sort
echo "--- network ---"
ip a 2>&1 | head -40 || ifconfig 2>&1 | head -40
cat /etc/resolv.conf 2>&1 | head -10
echo "--- /etc/hosts ---"
cat /etc/hosts 2>&1
echo "==== end diagnostic ===="

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

if should_enable_acl; then
  : "${AIC_MODEL_PASSWD:=CHANGE_IN_PROD}"
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

# Build a list of every candidate router endpoint we can think of.
# Zenoh peer mode will try them all; whichever resolves wins.
ENDPOINT_LIST='"tcp/localhost:7447", "tcp/127.0.0.1:7447"'
if [[ -n "$AIC_ROUTER_ADDR" ]]; then
  ENDPOINT_LIST="\"tcp/$AIC_ROUTER_ADDR\", $ENDPOINT_LIST"
fi
# Common service-name candidates across docker-compose / k8s / etc.
for h in eval aic-eval aic_eval aic-eval-1 router zenoh-router zenoh; do
  ENDPOINT_LIST="$ENDPOINT_LIST, \"tcp/$h:7447\""
done

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
    multicast: { enabled: true, autoconnect: { router: "router|peer", peer: "router|peer" } },
    gossip: { enabled: true, multihop: true },
  }$ACL_BLOCK
}
ZEOF
export ZENOH_SESSION_CONFIG_URI=/tmp/zenoh_session_config.json5
echo "==== zenoh session config ===="
cat $ZENOH_SESSION_CONFIG_URI
echo "==== end zenoh session config ===="

# If a diagnostic webhook URL is provided via env, POST a heartbeat.
# Cluster may block egress; ignore failures.
if [[ -n "$AIC_DIAG_URL" ]]; then
  (
    curl -s -m 10 -X POST -H "Content-Type: application/json" "$AIC_DIAG_URL" \
      --data "{\"event\":\"startup\",\"hostname\":\"$(hostname)\",\"router_addr\":\"${AIC_ROUTER_ADDR:-unset}\",\"acl\":\"${AIC_ENABLE_ACL:-unset}\",\"endpoints\":\"$ENDPOINT_LIST\"}" \
      2>&1 | head -3
  ) &
fi

# Run the model. If it crashes, log loudly and sleep until cluster
# timeout — better than instant "Failed" with no info.
echo "==== launching aic_model ===="
pixi run --as-is ros2 run aic_model aic_model "$@"
RC=$?
echo "==== aic_model exited with code $RC ===="

if [[ -n "$AIC_DIAG_URL" ]]; then
  curl -s -m 10 -X POST -H "Content-Type: application/json" "$AIC_DIAG_URL" \
    --data "{\"event\":\"exit\",\"code\":$RC,\"hostname\":\"$(hostname)\"}" 2>&1 | head -3
fi

echo "==== sleeping to keep container alive for cluster diagnosis ===="
sleep 600
exit $RC
