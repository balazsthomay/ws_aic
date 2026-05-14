"""Modal-based seating-demo collection.

Runs the existing eval+model docker-compose stack inside a Modal CPU sandbox
with Docker-in-Docker. Episodes land in the Modal `aic-data` volume.

Architecture:
- Modal Function with experimental_options enable_docker (DinD alpha).
- dockerd inside runs --iptables=false --bridge=none --ip-forward=false
  (gVisor blocks iptables nft).
- Compose services use network_mode: host. Model talks to localhost:7447.
- One Modal function call per batch. Many batches run in parallel via spawn.

Reuses collect_demos_gazebo.py helpers for scene-config generation.

Entrypoints:
  modal run scripts/modal_collect_demos.py::smoke_test
  modal run scripts/modal_collect_demos.py::collect --n 200 --batch-size 10
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import modal
import numpy as np
import yaml

# Scene-config helpers from collect_demos_gazebo (mujoco-dependent) are
# imported lazily inside local entrypoints — the Modal worker doesn't need them.

ROOT = Path(__file__).resolve().parent.parent

ECR_IMAGE = "973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs:v26"
EVAL_IMAGE = "ghcr.io/intrinsic-dev/aic/aic_eval"

COLLECTOR_SRC_PATH = (
    ROOT
    / "src/aic/aic_example_policies/aic_example_policies/ros/SeatingCollector.py"
)

DOCKER_IMAGE = (
    modal.Image.debian_slim()
    .apt_install("docker.io", "docker-compose", "ca-certificates",
                 "curl", "less", "iproute2", "awscli")
    .pip_install("numpy", "pyyaml")  # imported at module top level
)

app = modal.App("aic-seat-collect")
volume = modal.Volume.from_name("aic-data", create_if_missing=True)


# --- Local helpers ---

def _aws_credentials() -> dict:
    """Pull AWS creds from local `aic` profile."""
    try:
        out = subprocess.check_output(
            ["aws", "--profile", "aic", "configure", "export-credentials"],
            text=True,
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"AWS credentials lookup failed: {e}")
    cred = json.loads(out)
    out = {
        "AWS_ACCESS_KEY_ID": cred["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": cred["SecretAccessKey"],
        "AWS_DEFAULT_REGION": "us-east-1",
    }
    if cred.get("SessionToken"):
        out["AWS_SESSION_TOKEN"] = cred["SessionToken"]
    return out


def _gen_batch(rng: np.random.Generator, batch_size: int, batch_idx: int,
               time_limit: int = 240) -> tuple[str, str, list[dict]]:
    """Generate one batch (yaml + meta json) of `batch_size` trials.

    Lazy-imports collect_demos_gazebo (mujoco-dependent) so the Modal worker
    doesn't need mujoco installed to import this module.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from collect_demos_gazebo import (
        BOARD_DX, BOARD_DY, BOARD_DYAW,
        build_yaml, compute_gt_port_xyz, sample_target,
    )
    nominal_board_z = 1.14
    trials = {}
    meta = []
    last_cfg = None
    for j in range(batch_size):
        target = sample_target(rng)
        board_pos = np.array([
            0.15 + rng.uniform(-BOARD_DX, BOARD_DX),
            0.0  + rng.uniform(-BOARD_DY, BOARD_DY),
            nominal_board_z,
        ])
        board_yaw = math.pi + rng.uniform(-BOARD_DYAW, BOARD_DYAW)
        gt_xyz = compute_gt_port_xyz(target, board_pos, board_yaw)
        cfg, mod_name, port_name = build_yaml(target, board_pos, board_yaw,
                                              time_limit=time_limit)
        last_cfg = cfg
        ep_idx = batch_idx * 1000 + j
        scene_id = f"{ep_idx:05d}_{target['kind']}_rail{target['rail']}"
        trials[f"trial_{j+1}"] = cfg["trials"]["trial_1"]
        meta.append({
            "trial_index": j,
            "scene_id": scene_id,
            "gt_xyz": gt_xyz.tolist(),
            "board_xyyaw": [float(board_pos[0]), float(board_pos[1]),
                            float(board_yaw)],
            "target_module_name": mod_name,
            "port_name": port_name,
            "kind": target["kind"],
            "rail": int(target["rail"]),
        })
    batch_yaml_str = yaml.safe_dump({
        "scoring": last_cfg["scoring"],
        "task_board_limits": last_cfg["task_board_limits"],
        "trials": trials,
        "robot": last_cfg["robot"],
    }, sort_keys=False)
    return batch_yaml_str, json.dumps(meta), meta


# --- Modal function ---

COMPOSE_TEMPLATE = """version: '3'
services:
  eval:
    image: {eval_image}
    network_mode: host
    extra_hosts:
      - "eval:127.0.0.1"
      - "model:127.0.0.1"
    command: gazebo_gui:=false launch_rviz:=false ground_truth:={ground_truth} start_aic_engine:=true shutdown_on_aic_engine_exit:=true aic_engine_config_file:=/aic_config/trial.yaml
    volumes:
      - /work/configs/{batch_id}.yaml:/aic_config/trial.yaml:ro
    environment:
      AIC_EVAL_PASSWD: CHANGE_IN_PROD
      AIC_MODEL_PASSWD: CHANGE_IN_PROD
      AIC_ENABLE_ACL: "true"
  model:
    image: {ecr_image}
    # Share eval's network + PID namespace. With both on the SAME namespace,
    # Zenoh action-server registration becomes intra-process (no cross-namespace
    # routing), which we hope fixes the action-goal-not-delivered issue under
    # Modal's gVisor-restricted DinD.
    network_mode: "service:eval"
    pid: "service:eval"
    command: ['--ros-args', '-p', 'policy:=aic_example_policies.ros.{policy_name}']
    depends_on: [eval]
    volumes:
      - /work/{policy_file}:/ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/{policy_file}:ro
      - /work/configs/{batch_id}.json:/trial_meta.json:ro
      - /work/output:/output
    environment:
      RMW_IMPLEMENTATION: rmw_zenoh_cpp
      ZENOH_ROUTER_CHECK_ATTEMPTS: "-1"
      AIC_ROUTER_ADDR: localhost:7447
      AIC_MODEL_PASSWD: CHANGE_IN_PROD
      AIC_ENABLE_ACL: "true"
      AIC_TRIAL_META: "/trial_meta.json"
      AIC_OUTPUT_DIR: "/output"
"""


@app.function(
    image=DOCKER_IMAGE,
    cpu=8.0, memory=16384,
    timeout=7200,
    volumes={"/data": volume},
    experimental_options={"enable_docker": True},
)
def collect_batch(batch_id: str, batch_yaml: str, batch_meta: str,
                  collector_src: str, n_trials: int,
                  aws_creds: dict,
                  policy_name: str = "SeatingCollector",
                  policy_file: str = "SeatingCollector.py",
                  ground_truth: bool = True,
                  per_trial_budget_s: int = 270) -> dict:
    """Run one batch inside a Modal sandbox with DinD.

    policy_name/policy_file: which policy to launch in the model container.
        Defaults to SeatingCollector. Override to test with a known-good
        policy like GazeboCollector for A/B debugging.
    ground_truth: pass to eval container's launch — needed for SeatingCollector's
        TF lookup; not needed for GazeboCollector (which only takes snapshots).
    """
    import os
    import shutil
    print(f"=== [{batch_id}] starting on Modal worker (policy={policy_name}) ===")

    # Inject AWS creds into env for ECR login
    for k, v in aws_creds.items():
        os.environ[k] = v

    work = Path("/work")
    (work / "configs").mkdir(parents=True, exist_ok=True)
    (work / "output").mkdir(parents=True, exist_ok=True)
    (work / f"configs/{batch_id}.yaml").write_text(batch_yaml)
    (work / f"configs/{batch_id}.json").write_text(batch_meta)
    # Only write our custom policy source if it differs from the baked image's.
    # GazeboCollector is already in the v26 image; no need to mount.
    if policy_file == "SeatingCollector.py":
        (work / policy_file).write_text(collector_src)

    # Start dockerd
    print("--- Starting dockerd ---")
    rc = subprocess.call([
        "bash", "-c",
        "nohup dockerd --iptables=false --bridge=none --ip-forward=false "
        "  > /tmp/dockerd.log 2>&1 & "
        "for i in $(seq 1 60); do "
        "  if docker info >/dev/null 2>&1; then echo dockerd-ready; exit 0; fi; "
        "  sleep 1; "
        "done; "
        "echo dockerd-failed; tail -100 /tmp/dockerd.log; exit 1"
    ])
    if rc != 0:
        return {"batch_id": batch_id, "n_episodes": 0, "error": "dockerd failed"}

    # ECR login
    print("--- ECR login ---")
    rc = subprocess.call([
        "bash", "-c",
        "aws ecr get-login-password --region us-east-1 "
        "| docker login --username AWS --password-stdin "
        "  973918476471.dkr.ecr.us-east-1.amazonaws.com"
    ])
    if rc != 0:
        return {"batch_id": batch_id, "n_episodes": 0, "error": "ecr login failed"}

    # Pull both images in parallel
    print("--- Pulling eval + model images ---")
    p_eval = subprocess.Popen(["docker", "pull", EVAL_IMAGE])
    p_model = subprocess.Popen(["docker", "pull", ECR_IMAGE])
    p_eval.wait()
    p_model.wait()
    if p_eval.returncode or p_model.returncode:
        return {"batch_id": batch_id, "n_episodes": 0,
                "error": f"pull failed eval={p_eval.returncode} "
                         f"model={p_model.returncode}"}

    # Write compose
    compose = COMPOSE_TEMPLATE.format(
        eval_image=EVAL_IMAGE, ecr_image=ECR_IMAGE, batch_id=batch_id,
        policy_name=policy_name, policy_file=policy_file,
        ground_truth="true" if ground_truth else "false",
    )
    (work / "compose.yaml").write_text(compose)

    # Run compose detached so we can poll + dump per-container logs cleanly.
    print(f"--- Running compose ({n_trials} trials) ---")
    # Boot grace shorter for diagnosis — if eval doesn't dispatch in 240s
    # there's a problem worth catching early.
    boot_grace_s = 240 if n_trials == 1 else 360
    timeout_s = boot_grace_s + per_trial_budget_s * n_trials
    print(f"  budget: {timeout_s}s wall ({boot_grace_s}s boot + "
          f"{per_trial_budget_s*n_trials}s trials)")

    subprocess.call([
        "bash", "-c",
        "docker-compose -f /work/compose.yaml up -d 2>&1 | tail -20"
    ])

    # Poll for episode .npz appearing OR compose exit OR timeout.
    import time as _time
    start = _time.time()
    last_status = ""
    while _time.time() - start < timeout_s:
        elapsed = int(_time.time() - start)
        # Did an episode land?
        eps = list((work / "output").glob("episode_*.npz"))
        # Is eval still up?
        ps_out = subprocess.run(
            ["docker-compose", "-f", "/work/compose.yaml", "ps", "--services",
             "--filter", "status=running"],
            capture_output=True, text=True,
        ).stdout
        running = set(s.strip() for s in ps_out.splitlines() if s.strip())
        status = f"running={running} episodes={len(eps)}"
        if status != last_status:
            print(f"  [{elapsed}s] {status}")
            last_status = status
        if eps and n_trials == 1:
            # Single-trial smoke: we can exit early once the (atomically
            # renamed) episode appears. For multi-trial batches, wait for
            # all trials to finish (eval container will exit naturally).
            print(f"  [{elapsed}s] episode landed — exiting wait early")
            break
        if "eval" not in running and elapsed > 60:
            print(f"  [{elapsed}s] eval exited — checking logs")
            break
        _time.sleep(5)

    # Dump per-container logs for diagnosis. Model gets FULL log (small);
    # eval gets first+last 80 lines (huge log filled with sim debug spam).
    print("\n--- FULL LOG: model container ---")
    subprocess.call(["bash", "-c",
                     "docker-compose -f /work/compose.yaml logs --no-color model 2>&1 | grep -v 'execute loop' | head -200 || true"])
    print("\n--- LAST 50 LINES: model container (incl execute loop) ---")
    subprocess.call(["bash", "-c",
                     "docker-compose -f /work/compose.yaml logs --no-color --tail=50 model 2>&1 || true"])
    print("\n--- EVAL container: first 80 + last 80 (filtered) ---")
    subprocess.call(["bash", "-c",
                     "docker-compose -f /work/compose.yaml logs --no-color eval 2>&1 | "
                     "grep -vE '(out of limits|throttled log|effort:|position:|velocity:|desired period|InitializeCanonical)' | "
                     "head -80 || true"])
    subprocess.call(["bash", "-c",
                     "docker-compose -f /work/compose.yaml logs --no-color --tail=80 eval 2>&1 | "
                     "grep -vE '(out of limits|throttled log|effort:|position:|velocity:|desired period|InitializeCanonical)' || true"])

    # Tear down
    subprocess.call([
        "bash", "-c",
        "docker-compose -f /work/compose.yaml down --remove-orphans 2>&1 | tail -10"
    ])
    rc = 0  # diagnostic mode; rc isn't meaningful

    # Copy outputs to volume
    out_dir = Path("/data/seating_episodes")
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = list((work / "output").glob("episode_*.npz"))
    for ep in episodes:
        shutil.copyfile(ep, out_dir / ep.name)
    volume.commit()
    print(f"=== [{batch_id}] wrote {len(episodes)} episodes to "
          f"/data/seating_episodes ===")

    return {"batch_id": batch_id, "n_episodes": len(episodes), "compose_rc": rc}


# --- Local entrypoints ---

@app.local_entrypoint()
def smoke_test(seed: int = 0, n: int = 1, policy: str = "SeatingCollector"):
    """Single Modal sandbox, n trials. Verifies whole pipeline + lets us
    sample multiple scenes per smoke (~$0.10/episode amortized).

    Uses .spawn() so detached runs survive local CLI exit. After spawn,
    follow with `modal app logs <app-id>` and check the aic-data volume
    `seating_episodes/` for the resulting episode_*.npz.

    n: number of trials in this single batch (default 1).
    policy: which policy to test (default SeatingCollector). Pass
        --policy GazeboCollector to A/B against a known-good policy.
    """
    print(f"=== Smoke test (seed={seed}, n={n}, policy={policy}) ===")
    rng = np.random.default_rng(seed)
    batch_yaml, batch_meta, meta = _gen_batch(rng, batch_size=n, batch_idx=0)
    for m in meta:
        print(f"  - {m['scene_id']} (target={m['kind']}_rail{m['rail']})")
    collector_src = COLLECTOR_SRC_PATH.read_text() if policy == "SeatingCollector" else ""
    aws_creds = _aws_credentials()
    handle = collect_batch.spawn(
        batch_id="smoke",
        batch_yaml=batch_yaml,
        batch_meta=batch_meta,
        collector_src=collector_src,
        n_trials=n,
        aws_creds=aws_creds,
        policy_name=policy,
        policy_file=f"{policy}.py",
        # GazeboCollector doesn't need ground_truth (it's snapshot-only)
        ground_truth=(policy != "GazeboCollector"),
    )
    print(f"\n=== Spawned function call: {handle.object_id} ===")
    print("Follow with: modal app logs <app-id> (see View run URL above)")


@app.local_entrypoint()
def collect(n: int = 100, batch_size: int = 10, seed: int = 0,
            wait: bool = False):
    """Bulk collection: generate N scenes batched, parallel collect on Modal.

    By default (wait=False), spawns all batches and exits — so `--detach`
    works correctly without the local bash being killed mid-wait. Inspect
    progress at the View run URL or via `modal volume ls aic-data
    seating_episodes` after the runs complete.

    Pass --wait to block locally for results (only useful in foreground).
    """
    print(f"=== Collect (n={n}, batch_size={batch_size}, seed={seed}, "
          f"wait={wait}) ===")
    rng = np.random.default_rng(seed)
    n_batches = (n + batch_size - 1) // batch_size

    collector_src = COLLECTOR_SRC_PATH.read_text()
    aws_creds = _aws_credentials()

    # Generate all batches
    args = []
    for b in range(n_batches):
        b_size = min(batch_size, n - b * batch_size)
        batch_yaml, batch_meta, _ = _gen_batch(rng, b_size, b)
        args.append((f"batch_{b:03d}", batch_yaml, batch_meta, b_size))
    print(f"Generated {n_batches} batches × ~{batch_size} = {n} trials")

    # Spawn in parallel
    handles = []
    for (bid, byml, bmeta, bn) in args:
        h = collect_batch.spawn(
            batch_id=bid, batch_yaml=byml, batch_meta=bmeta,
            collector_src=collector_src, n_trials=bn,
            aws_creds=aws_creds,
        )
        handles.append((bid, h))
        print(f"  [{bid}] spawned: {h.object_id} ({bn} trials)")
    print(f"Spawned {len(handles)} parallel sandboxes")

    if not wait:
        print("\nNot waiting (use --wait to block). Track progress at app URL.")
        print("Once stopped, check episodes with:")
        print("  modal volume ls aic-data seating_episodes")
        return

    results = []
    for bid, h in handles:
        try:
            r = h.get()
            print(f"[{bid}] {r}")
            results.append(r)
        except Exception as e:
            print(f"[{bid}] FAILED: {e!r}")
            results.append({"batch_id": bid, "n_episodes": 0, "error": str(e)})
    total = sum(r.get("n_episodes", 0) for r in results)
    print(f"\n=== Total: {total}/{n} episodes ===")
