#!/usr/bin/env python3
"""Collect localizer training data from Gazebo on a single GCP spot VM.

Phases:
  generate  — sample N scene configs (v25 distribution), compute GT port_xy
              via local MuJoCo, write per-config yaml + GT to data/gz_collect/
  collect   — provision a GCP VM, ship configs, loop docker-compose per config
              with GazeboCollector policy, scp per-episode npzs back
  aggregate — concatenate per-episode npzs into data/demos_gz_v1.npz

Default: run all three phases end-to-end.

Usage:
  uv run scripts/collect_demos_gazebo.py --episodes 80
  uv run scripts/collect_demos_gazebo.py --phase generate --episodes 5
  uv run scripts/collect_demos_gazebo.py --phase collect
  uv run scripts/collect_demos_gazebo.py --phase aggregate
"""

import argparse
import base64
import io
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCENE_XML = ROOT / "src/aic/aic_utils/aic_mujoco/mjcf/scene.xml"
COLLECT_DIR = ROOT / "data/gz_collect"
EPISODES_DIR = COLLECT_DIR / "episodes"
CONFIGS_DIR = COLLECT_DIR / "configs"
GT_PATH = COLLECT_DIR / "gt.npz"

# Mirrors collect_demos_v25.py.
NIC_RAIL_Y = {0: -0.1745, 1: -0.1345, 2: -0.0945, 3: -0.0545, 4: -0.0145}
NIC_TRANSLATION_BOUNDS = (-0.0215, 0.0234)
SC_PORT_BASE = {
    0: np.array([-0.075, 0.0295, 0.0165]),
    1: np.array([-0.075, 0.0705, 0.0165]),
}
SC_TRANSLATION_BOUNDS = (-0.06, 0.055)
BOARD_DX = 0.05
BOARD_DY = 0.15
BOARD_DYAW = 0.3

# Plausible sample defaults for the rails we're not actively targeting.
DEFAULT_OTHER_RAILS = {
    "lc_mount_rail_0": dict(present=True, name="lc_mount_0", translation=0.02),
    "lc_mount_rail_1": dict(present=True, name="lc_mount_1", translation=-0.01),
    "sfp_mount_rail_0": dict(present=True, name="sfp_mount_0", translation=0.03),
    "sfp_mount_rail_1": dict(present=False),
    "sc_mount_rail_0": dict(present=True, name="sc_mount_0", translation=-0.02),
    "sc_mount_rail_1": dict(present=False),
}

ECR_IMAGE = "973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/balazs:v25"
EVAL_IMAGE = "ghcr.io/intrinsic-dev/aic/aic_eval"


def yaw_to_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def sample_target(rng):
    bucket = rng.integers(0, 7)
    if bucket < 5:
        return {
            "kind": "sfp",
            "rail": int(bucket),
            "translation": float(rng.uniform(*NIC_TRANSLATION_BOUNDS)),
        }
    elif bucket == 5:
        return {
            "kind": "sc",
            "rail": 0,
            "translation": float(rng.uniform(*SC_TRANSLATION_BOUNDS)),
        }
    else:
        return {
            "kind": "sc",
            "rail": 1,
            "translation": float(rng.uniform(*SC_TRANSLATION_BOUNDS)),
        }


def compute_gt_port_xyz(target, board_pos, board_yaw):
    """Replay v25's MuJoCo body_pos mutation to derive port world xyz."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    board_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "task_board_base_link")
    nic_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "nic_card_mount_0::nic_card_mount_link")
    sc_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sc_port_0::sc_port_link")
    sfp_port_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_port_0_link")
    sc_port_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sc_port_base_link")

    m.body_pos[board_id] = np.array(board_pos)
    m.body_quat[board_id] = yaw_to_quat(board_yaw)

    if target["kind"] == "sfp":
        m.body_pos[nic_id] = np.array([
            -0.081418 + target["translation"],
            NIC_RAIL_Y[target["rail"]],
            0.012,
        ])
        target_id = sfp_port_id
    else:
        anchor = SC_PORT_BASE[target["rail"]].copy()
        anchor[0] += target["translation"]
        m.body_pos[sc_id] = anchor
        target_id = sc_port_base_id

    mujoco.mj_forward(m, d)
    return d.xpos[target_id].copy()


def build_yaml(target, board_pos, board_yaw, time_limit=60):
    """Single-trial yaml in the aic_engine schema."""
    scene = {
        "task_board": {
            "pose": {
                "x": float(board_pos[0]),
                "y": float(board_pos[1]),
                "z": float(board_pos[2]),
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": float(board_yaw),
            },
        },
    }
    # Default rails (always present so the visual context matches eval).
    for rail in ("nic_rail_0", "nic_rail_1", "nic_rail_2", "nic_rail_3", "nic_rail_4"):
        scene["task_board"][rail] = {"entity_present": False}
    for rail in ("sc_rail_0", "sc_rail_1"):
        scene["task_board"][rail] = {"entity_present": False}
    for name, spec in DEFAULT_OTHER_RAILS.items():
        if spec["present"]:
            scene["task_board"][name] = {
                "entity_present": True,
                "entity_name": spec["name"],
                "entity_pose": {
                    "translation": float(spec["translation"]),
                    "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                },
            }
        else:
            scene["task_board"][name] = {"entity_present": False}

    if target["kind"] == "sfp":
        rail = f"nic_rail_{target['rail']}"
        scene["task_board"][rail] = {
            "entity_present": True,
            "entity_name": f"nic_card_{target['rail']}",
            "entity_pose": {
                "translation": float(target["translation"]),
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            },
        }
        target_module_name = f"nic_card_mount_{target['rail']}"
        port_name = "sfp_port_0"
        port_type = "sfp"
        cable_name = "cable_0"
        cable_type_short = "sfp_sc"
        cable_type_long = "sfp_sc_cable"
        plug_type = "sfp"
        plug_name = "sfp_tip"
        gripper_z = 0.04245
    else:
        rail = f"sc_rail_{target['rail']}"
        scene["task_board"][rail] = {
            "entity_present": True,
            "entity_name": f"sc_mount_{target['rail']}",
            "entity_pose": {
                "translation": float(target["translation"]),
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            },
        }
        target_module_name = f"sc_port_{target['rail']}"
        port_name = "sc_port_base"
        port_type = "sc"
        cable_name = "cable_1"
        cable_type_short = "sfp_sc"
        cable_type_long = "sfp_sc_cable_reversed"
        plug_type = "sc"
        plug_name = "sc_tip"
        gripper_z = 0.04045

    scene["cables"] = {
        cable_name: {
            "pose": {
                "gripper_offset": {
                    "x": 0.0, "y": 0.015385, "z": float(gripper_z),
                },
                "roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303,
            },
            "attach_cable_to_gripper": True,
            "cable_type": cable_type_long,
        },
    }

    cfg = {
        # Reuse the scoring topic list verbatim from sample_config.yaml.
        "scoring": {
            "topics": [
                {"topic": {"name": "/joint_states", "type": "sensor_msgs/msg/JointState"}},
                {"topic": {"name": "/tf", "type": "tf2_msgs/msg/TFMessage"}},
                {"topic": {"name": "/tf_static", "type": "tf2_msgs/msg/TFMessage", "latched": True}},
                {"topic": {"name": "/scoring/tf", "type": "tf2_msgs/msg/TFMessage"}},
                {"topic": {"name": "/aic/gazebo/contacts/off_limit", "type": "ros_gz_interfaces/msg/Contacts"}},
                {"topic": {"name": "/fts_broadcaster/wrench", "type": "geometry_msgs/msg/WrenchStamped"}},
                {"topic": {"name": "/aic_controller/joint_commands", "type": "aic_control_interfaces/msg/JointMotionUpdate"}},
                {"topic": {"name": "/aic_controller/pose_commands", "type": "aic_control_interfaces/msg/MotionUpdate"}},
                {"topic": {"name": "/scoring/insertion_event", "type": "std_msgs/msg/String"}},
                {"topic": {"name": "/aic_controller/controller_state", "type": "aic_control_interfaces/msg/ControllerState"}},
            ],
        },
        "task_board_limits": {
            "nic_rail":   {"min_translation": -0.0215, "max_translation": 0.0234},
            "sc_rail":    {"min_translation": -0.06,   "max_translation": 0.055},
            "mount_rail": {"min_translation": -0.09425, "max_translation": 0.09425},
        },
        "trials": {
            "trial_1": {
                "scene": scene,
                "tasks": {
                    "task_1": {
                        "cable_type": cable_type_short,
                        "cable_name": cable_name,
                        "plug_type": plug_type,
                        "plug_name": plug_name,
                        "port_type": port_type,
                        "port_name": port_name,
                        "target_module_name": target_module_name,
                        "time_limit": int(time_limit),
                    },
                },
            },
        },
        "robot": {
            "home_joint_positions": {
                "shoulder_pan_joint": -0.1597,
                "shoulder_lift_joint": -1.3542,
                "elbow_joint": -1.6648,
                "wrist_1_joint": -1.6933,
                "wrist_2_joint": 1.5710,
                "wrist_3_joint": 1.4110,
            },
        },
    }
    return cfg, target_module_name, port_name


def phase_generate(args):
    """Generate scenes batched into B yamls, each with M trials.

    Batching amortizes the per-docker-compose boot cost (~5 min) over many
    trials. Each batch uses one yaml with `batch_size` trials. The policy
    indexes through the trials in order and reads GT from a per-batch JSON
    file mounted alongside the yaml.
    """
    rng = np.random.default_rng(args.seed)
    COLLECT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)
    # Wipe stale files
    for p in CONFIGS_DIR.glob("*"):
        p.unlink()

    batches = []
    nominal_board_z = 1.14
    n_batches = (args.episodes + args.batch_size - 1) // args.batch_size

    global_gt = []
    for b in range(n_batches):
        b_size = min(args.batch_size, args.episodes - b * args.batch_size)
        trials_yaml = {}
        meta = []
        for j in range(b_size):
            target = sample_target(rng)
            board_pos = np.array([
                0.15 + rng.uniform(-BOARD_DX, BOARD_DX),
                0.0  + rng.uniform(-BOARD_DY, BOARD_DY),
                nominal_board_z,
            ])
            board_yaw = math.pi + rng.uniform(-BOARD_DYAW, BOARD_DYAW)
            gt_xyz = compute_gt_port_xyz(target, board_pos, board_yaw)
            cfg, mod_name, port_name = build_yaml(
                target, board_pos, board_yaw, time_limit=args.time_limit
            )
            ep_idx = b * args.batch_size + j
            scene_id = f"{ep_idx:04d}_{target['kind']}_rail{target['rail']}"
            trials_yaml[f"trial_{j+1}"] = cfg["trials"]["trial_1"]
            meta.append({
                "trial_index": j,
                "scene_id": scene_id,
                "gt_xyz": gt_xyz.tolist(),
                "board_xyyaw": [float(board_pos[0]), float(board_pos[1]), float(board_yaw)],
                "target_module_name": mod_name,
                "port_name": port_name,
                "kind": target["kind"],
                "rail": int(target["rail"]),
            })
            global_gt.append(meta[-1])

        # Build the full yaml for this batch (reuse non-trials sections from the
        # last build_yaml invocation by reading them off `cfg`).
        batch_cfg = {
            "scoring": cfg["scoring"],
            "task_board_limits": cfg["task_board_limits"],
            "trials": trials_yaml,
            "robot": cfg["robot"],
        }

        batch_id = f"batch_{b:03d}"
        yaml_path = CONFIGS_DIR / f"{batch_id}.yaml"
        meta_path = CONFIGS_DIR / f"{batch_id}.json"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(batch_cfg, f, sort_keys=False)
        import json as _json
        with open(meta_path, "w") as f:
            _json.dump(meta, f)
        batches.append(batch_id)

    np.savez(
        GT_PATH,
        scene_id=np.array([r["scene_id"] for r in global_gt], dtype=object),
        gt_xyz=np.stack([r["gt_xyz"] for r in global_gt]),
        board_xyyaw=np.stack([r["board_xyyaw"] for r in global_gt]),
        target_module_name=np.array([r["target_module_name"] for r in global_gt], dtype=object),
        port_name=np.array([r["port_name"] for r in global_gt], dtype=object),
        kind=np.array([r["kind"] for r in global_gt], dtype=object),
        rail=np.array([r["rail"] for r in global_gt]),
        batches=np.array(batches, dtype=object),
    )
    print(f"[generate] wrote {n_batches} batch yamls × ~{args.batch_size} trials → {CONFIGS_DIR}")
    print(f"[generate] total scenes: {len(global_gt)}")
    print(f"[generate] wrote GT → {GT_PATH}")
    by_kind = {"sfp": 0, "sc": 0}
    for r in global_gt:
        by_kind[r["kind"]] += 1
    print(f"[generate] distribution: {by_kind}")


def make_collection_payload(collector_path: Path) -> str:
    """Tar+gz the per-batch yamls + jsons, GazeboCollector.py.
    Return base64-encoded blob to embed in VM startup script."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(CONFIGS_DIR.iterdir()):
            tf.add(p, arcname=f"configs/{p.name}")
        tf.add(collector_path, arcname="GazeboCollector.py")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def phase_collect(args):
    if not GT_PATH.exists():
        sys.exit("Run --phase generate first (no gt.npz)")
    collector_src = ROOT / "src/aic/aic_example_policies/aic_example_policies/ros/GazeboCollector.py"
    if not collector_src.exists():
        sys.exit(f"Missing GazeboCollector.py at {collector_src}")

    gt = np.load(GT_PATH, allow_pickle=True)
    n = len(gt["scene_id"])
    batches = [str(b) for b in gt["batches"]]
    print(f"[collect] payload: {len(batches)} batches × ~{n // len(batches)} trials = {n} scenes")

    payload_b64 = make_collection_payload(collector_src)
    print(f"[collect] payload size: {len(payload_b64) // 1024}KB")

    creds = subprocess.check_output(
        ["aws", "--profile", "aic", "configure", "export-credentials"]
    ).decode()
    import json as _json
    cred = _json.loads(creds)
    aws_akid = cred["AccessKeyId"]
    aws_sak = cred["SecretAccessKey"]

    batches_str = " ".join(batches)
    # Per-batch wall-clock budget: boot (~5 min) + per-trial (~3 min) + margin.
    per_batch_seconds = 360 + (args.batch_size * 240)

    instance = f"aic-collect-{int(time.time())}-{random.randint(0, 9999)}"

    startup = f"""#!/bin/bash
set -ex
exec > >(tee /var/log/aic_collect.log) 2>&1

mkdir -p /work/configs /work/output
cd /work
echo "{payload_b64}" | base64 -d | tar xzf - -C /work
ls -la /work
ls /work/configs | head -5

# ECR login
export AWS_ACCESS_KEY_ID={aws_akid}
export AWS_SECRET_ACCESS_KEY={aws_sak}
export AWS_DEFAULT_REGION=us-east-1
aws ecr get-login-password --region us-east-1 \\
  | docker login --username AWS --password-stdin \\
      973918476471.dkr.ecr.us-east-1.amazonaws.com

echo "[collect] pulling model image..."
docker pull {ECR_IMAGE}
echo "[collect] pull done"

BATCHES=({batches_str})
echo "[collect] running ${{#BATCHES[@]}} batches"

for BATCH in "${{BATCHES[@]}}"; do
  CONFIG_PATH="/work/configs/${{BATCH}}.yaml"
  META_PATH="/work/configs/${{BATCH}}.json"
  if [[ ! -f "$CONFIG_PATH" || ! -f "$META_PATH" ]]; then
    echo "[collect] missing files for $BATCH, skipping"
    continue
  fi
  echo "[collect] === batch $BATCH ==="

  cat > /tmp/compose.yaml <<EOF
name: aic
services:
  eval:
    image: {EVAL_IMAGE}
    command: gazebo_gui:=false launch_rviz:=false ground_truth:=false start_aic_engine:=true shutdown_on_aic_engine_exit:=true aic_engine_config_file:=/aic_config/trial.yaml
    networks: [default]
    volumes:
      - /work/configs/${{BATCH}}.yaml:/aic_config/trial.yaml:ro
    environment:
      AIC_EVAL_PASSWD: CHANGE_IN_PROD
      AIC_MODEL_PASSWD: CHANGE_IN_PROD
      AIC_ENABLE_ACL: "true"
  model:
    image: {ECR_IMAGE}
    command: ['--ros-args', '-p', 'policy:=aic_example_policies.ros.GazeboCollector']
    networks: [default]
    volumes:
      - /work/GazeboCollector.py:/ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/GazeboCollector.py:ro
      - /work/configs/${{BATCH}}.json:/trial_meta.json:ro
      - /work/output:/output
    environment:
      RMW_IMPLEMENTATION: rmw_zenoh_cpp
      ZENOH_ROUTER_CHECK_ATTEMPTS: "-1"
      AIC_ROUTER_ADDR: eval:7447
      AIC_MODEL_PASSWD: CHANGE_IN_PROD
      AIC_ENABLE_ACL: "true"
      AIC_TRIAL_META: "/trial_meta.json"
      AIC_OUTPUT_DIR: "/output"
networks:
  default:
    internal: true
EOF

  timeout {per_batch_seconds} docker compose -f /tmp/compose.yaml up --abort-on-container-exit || true
  docker compose -f /tmp/compose.yaml down --remove-orphans || true
  echo "[collect] batch $BATCH done; output count:"
  ls /work/output | wc -l
done

echo "[collect] tarring outputs"
tar czf /var/log/aic_collect_episodes.tgz -C /work/output .

echo "AIC_COLLECT_DONE"
"""

    zones = ["us-central1-a", "us-central1-b", "us-east1-b", "us-east1-c",
             "us-east4-a", "us-west1-b"]
    zone_picked = None
    gcloud = os.environ.get("GCLOUD",
                            "/opt/homebrew/share/google-cloud-sdk/bin/gcloud")

    for z in zones:
        print(f"[collect] trying VM {instance} in {z}")
        try:
            with open("/tmp/aic_startup.sh", "w") as f:
                f.write(startup)
            subprocess.check_call([
                gcloud, "compute", "instances", "create", instance,
                f"--zone={z}",
                "--machine-type=c3-standard-8",
                "--image=aic-eval-base",
                "--boot-disk-size=80GB",
                "--boot-disk-type=pd-balanced",
                "--provisioning-model=SPOT",
                "--instance-termination-action=DELETE",
                f"--metadata-from-file=startup-script=/tmp/aic_startup.sh",
            ])
            zone_picked = z
            print(f"[collect] created in {z}")
            break
        except subprocess.CalledProcessError as e:
            print(f"[collect] {z} failed ({e}), trying next")
            continue
    if not zone_picked:
        sys.exit("[collect] all zones exhausted")

    try:
        # Wait for completion: per-batch budget × number of batches + slack.
        budget_min = max(20, 5 + (len(batches) * (per_batch_seconds // 60 + 2)))
        print(f"[collect] waiting up to {budget_min} min for AIC_COLLECT_DONE")
        deadline = time.time() + budget_min * 60
        while time.time() < deadline:
            time.sleep(60)
            try:
                out = subprocess.check_output([
                    gcloud, "compute", "instances", "get-serial-port-output",
                    instance, f"--zone={zone_picked}",
                ], stderr=subprocess.DEVNULL).decode()
                if "AIC_COLLECT_DONE" in out:
                    print("[collect] done signal seen")
                    break
                # Progress hint
                last_scene = None
                for line in out.splitlines()[-200:]:
                    if "[collect] === scene" in line:
                        last_scene = line.strip()
                if last_scene:
                    print(f"  {last_scene}")
            except subprocess.CalledProcessError:
                pass
        else:
            print("[collect] WARNING: deadline reached without done signal")

        # SCP outputs back
        EPISODES_DIR.mkdir(parents=True, exist_ok=True)
        local_tgz = EPISODES_DIR / "remote.tgz"
        print(f"[collect] scp outputs → {local_tgz}")
        subprocess.check_call([
            gcloud, "compute", "scp", f"--zone={zone_picked}",
            f"{instance}:/var/log/aic_collect_episodes.tgz", str(local_tgz),
        ])
        subprocess.check_call(["tar", "xzf", str(local_tgz), "-C", str(EPISODES_DIR)])
        local_tgz.unlink()

        local_log = EPISODES_DIR / "aic_collect.log"
        print(f"[collect] scp log → {local_log}")
        subprocess.call([
            gcloud, "compute", "scp", f"--zone={zone_picked}",
            f"{instance}:/var/log/aic_collect.log", str(local_log),
        ])
    finally:
        print(f"[collect] deleting VM {instance}")
        subprocess.call([
            gcloud, "compute", "instances", "delete", instance,
            f"--zone={zone_picked}", "--quiet",
        ])

    n_files = len(list(EPISODES_DIR.glob("episode_*.npz")))
    print(f"[collect] {n_files}/{n} episodes captured")


def phase_aggregate(args):
    files = sorted(EPISODES_DIR.glob("episode_*.npz"))
    if not files:
        sys.exit(f"No episodes in {EPISODES_DIR}; run --phase collect first")
    print(f"[aggregate] {len(files)} episodes")

    eps = [np.load(p, allow_pickle=True) for p in files]
    # Each episode has variable viewpoint count; pad to max for npz.
    max_v = max(int(e["images_left_camera"].shape[0]) for e in eps)
    H = eps[0]["images_left_camera"].shape[1]
    W = eps[0]["images_left_camera"].shape[2]
    n = len(eps)

    images = {
        "images_left_camera": np.zeros((n, max_v, H, W, 3), dtype=np.uint8),
        "images_center_camera": np.zeros((n, max_v, H, W, 3), dtype=np.uint8),
        "images_right_camera": np.zeros((n, max_v, H, W, 3), dtype=np.uint8),
    }
    states = np.zeros((n, max_v, 26), dtype=np.float32)
    episode_lengths = np.zeros(n, dtype=np.int64)
    port_xy_world = np.zeros((n, 3), dtype=np.float32)
    board = np.zeros((n, 3), dtype=np.float32)
    targets = np.empty(n, dtype=object)

    for i, e in enumerate(eps):
        v = int(e["images_left_camera"].shape[0])
        episode_lengths[i] = v
        images["images_left_camera"][i, :v] = e["images_left_camera"]
        images["images_center_camera"][i, :v] = e["images_center_camera"]
        images["images_right_camera"][i, :v] = e["images_right_camera"]
        # Pack the 26-dim state slot used by train_modal_vision: indices 22:24
        # are port_xy (label). Other slots: the localizer pipeline only reads
        # port_xy_world (top-level field) and pulls images, so the state slot
        # is best-effort here. Fill joints + zero padding.
        joints = e["joint_pos"].astype(np.float32)  # (v, 6)
        states[i, :v, 0:6] = joints
        if "port_xy_world" in e.files:
            states[i, :v, 22:24] = e["port_xy_world"][:2]
        targets[i] = (
            f"{str(e['port_type'])}/"
            f"{str(e['target_module_name'])}/"
            f"{str(e['port_name'])}"
        )
        if "port_xy_world" in e.files:
            port_xy_world[i] = e["port_xy_world"]
        if "board" in e.files:
            board[i] = e["board"]

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save = dict(
        states=states,
        actions=np.zeros((n, max_v, 6), dtype=np.float32),
        episode_lengths=episode_lengths,
        targets=targets,
        port_xy_world=port_xy_world,
        board=board,
        **images,
    )
    np.savez_compressed(out_path, **save)
    print(f"[aggregate] saved {out_path} ({out_path.stat().st_size // (1024*1024)}MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["all", "generate", "collect", "aggregate"],
                    default="all")
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=10,
                    help="Trials per docker-compose lifecycle (amortizes ~5min boot)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-limit", type=int, default=60,
                    help="Per-trial time limit in aic_engine yaml")
    ap.add_argument("--output", type=str, default="data/demos_gz_v1.npz")
    args = ap.parse_args()

    if args.phase in ("all", "generate"):
        phase_generate(args)
    if args.phase in ("all", "collect"):
        phase_collect(args)
    if args.phase in ("all", "aggregate"):
        phase_aggregate(args)


if __name__ == "__main__":
    main()
