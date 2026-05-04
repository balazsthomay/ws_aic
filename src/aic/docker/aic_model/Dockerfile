FROM docker.io/library/ros:kilted-ros-core AS build

# Install pixi
RUN apt update && apt install -y git && curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# Add base dependencies
COPY aic_example_policies /ws_aic/src/aic/aic_example_policies
COPY aic_model /ws_aic/src/aic/aic_model
COPY aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY aic_utils /ws_aic/src/aic/aic_utils
COPY pixi.toml pixi.lock /ws_aic/src/aic/
COPY pixi_env_setup.sh /ws_aic/src/aic/

# Add other dependencies

SHELL ["/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/rattler/cache --mount=type=cache,target=/ws_aic/src/aic/.pixi/build \
    cd /ws_aic/src/aic && pixi install --frozen

WORKDIR /ws_aic/src/aic
COPY --chmod=755 <<"EOF" /entrypoint.sh
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

# prepare the credentials file
if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

# Build a Zenoh session config FILE rather than relying on ZENOH_CONFIG_OVERRIDE.
# rmw_zenoh_cpp's override env var is silently ignored when overriding connect/endpoints
# from the default localhost:7447 -- causes silent submission failure (no router connect,
# no node discovery, engine never sees aic_model, no score).
ACL_AUTH=""
if should_enable_acl; then
  ACL_AUTH=$(cat <<ACLEOF
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
  ACL_AUTH=$(cat <<ACLEOF
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
  }$ACL_AUTH
}
ZEOF
export ZENOH_SESSION_CONFIG_URI=/tmp/zenoh_session_config.json5
echo "ZENOH_SESSION_CONFIG_URI=$ZENOH_SESSION_CONFIG_URI"
exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=aic_example_policies.ros.DAggerInsert"]
