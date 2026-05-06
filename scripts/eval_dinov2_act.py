#!/usr/bin/env python3
"""Headless MuJoCo eval of the trained DINOv2 + ACT policy.

Re-encodes the 3 wrist cameras through frozen DINOv2 every VISION_INTERVAL
control ticks, predicts a K-step action chunk with the ACT head, and steps
the simulator. Reports per-trial success / xy-error / z-error.

Usage:
    .venv/bin/python3 scripts/eval_dinov2_act.py --trials 10
"""

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src/aic/aic_utils/aic_mujoco"))
sys.path.insert(0, str(ROOT / "scripts"))
from mujoco_obs import MuJoCoObserver  # noqa: E402

from train_dinov2_act import ACTHead, PATCH_GRID, load_dinov2  # noqa: E402

SCENE = ROOT / "src/aic/aic_utils/aic_mujoco/mjcf/scene.xml"
WEIGHTS = ROOT / "data/outputs/act_policy.pt"
STATS = ROOT / "data/outputs/act_norm_stats.npz"

HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
CAMERAS = ["left_camera", "center_camera", "right_camera"]

CONTROL_HZ = 20
MAX_TIME = 14.0
VISION_INTERVAL = 4
CHUNK_SIZE = 16
DINOV2_INPUT = 224
DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TIP_OFFSET_TCP_FRAME = np.array([-0.0018, -0.0189, 0.0547])
NOMINAL_PORT_POS = np.array([0.220, -0.013, 1.273])

BOARD_YAW_RANGE = 0.15
BOARD_XY_RANGE = 0.015


def setup_indices(m):
    tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_tip_link")
    port_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_port_0_link")
    board_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "task_board_base_link")
    qids, dids, aids = [], [], []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qids.append(m.jnt_qposadr[jid])
        dids.append(m.jnt_dofadr[jid])
        aids.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_motor"))
    return {
        "tcp_site": tcp_site, "tip_id": tip_id, "port_id": port_id,
        "board_id": board_id,
        "qids": np.array(qids), "dids": np.array(dids), "aids": np.array(aids),
    }


def randomize_board(m, idx, rng, nominal_pos, nominal_quat):
    dx = rng.uniform(-BOARD_XY_RANGE, BOARD_XY_RANGE)
    dy = rng.uniform(-BOARD_XY_RANGE, BOARD_XY_RANGE)
    dyaw = rng.uniform(-BOARD_YAW_RANGE, BOARD_YAW_RANGE)
    m.body_pos[idx["board_id"]] = nominal_pos + np.array([dx, dy, 0])
    cyaw, syaw = math.cos(dyaw / 2), math.sin(dyaw / 2)
    dq = np.array([cyaw, 0, 0, syaw])
    new_quat = np.zeros(4)
    mujoco.mju_mulQuat(new_quat, dq, nominal_quat)
    m.body_quat[idx["board_id"]] = new_quat
    return dx, dy, dyaw


def preprocess(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(DINOV2_INPUT, DINOV2_INPUT),
                      mode="bilinear", align_corners=False)
    return (t - DINOV2_MEAN) / DINOV2_STD


def get_state(d, idx, progress, port_override=None):
    tcp_pos = d.site_xpos[idx["tcp_site"]].copy()
    tcp_mat = d.site_xmat[idx["tcp_site"]].reshape(3, 3)
    tcp_quat = np.zeros(4)
    mujoco.mju_mat2Quat(tcp_quat, tcp_mat.flatten())
    port = port_override if port_override is not None else d.xpos[idx["port_id"]].copy()
    return np.concatenate([
        d.qpos[idx["qids"]].copy(),
        d.qvel[idx["dids"]].copy(),
        tcp_pos, tcp_quat,
        d.xpos[idx["tip_id"]].copy(),
        port,
        [progress],
    ])


def run_trial(m, d, idx, observer, dinov2, processor, act, stats, device,
              gt_port=False, mode="cls", zero_vision=False):
    s_mean = torch.tensor(stats["state_mean"], dtype=torch.float32, device=device)
    s_std = torch.tensor(stats["state_std"], dtype=torch.float32, device=device)
    a_mean = torch.tensor(stats["action_mean"], dtype=torch.float32, device=device)
    a_std = torch.tensor(stats["action_std"], dtype=torch.float32, device=device)

    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    dt = m.opt.timestep
    control_interval = int(1.0 / (CONTROL_HZ * dt))
    max_steps = int(MAX_TIME / dt)

    ctrl = HOME.copy()
    chunk = None
    chunk_idx = 0

    n_control = 0
    for s in range(max_steps):
        if s % control_interval == 0:
            progress = (s * dt) / MAX_TIME

            if chunk is None or chunk_idx >= chunk.shape[0] or n_control % VISION_INTERVAL == 0:
                # Encode vision
                obs = observer.get_observation()
                cam_tensors = [preprocess(obs.images[c]) for c in CAMERAS]
                pixel_values = torch.cat(cam_tensors, dim=0).to(device)
                with torch.no_grad():
                    hs = dinov2(pixel_values=pixel_values).last_hidden_state
                    if mode == "cls":
                        feats = hs[:, 0, :]                       # (3, 384)
                        vision_tokens = feats.unsqueeze(0)        # (1, 3, 384)
                    else:
                        # Patch tokens: drop CLS, pool 16x16 → PATCH_GRID
                        patches = hs[:, 1:, :]                    # (3, 256, 384)
                        Bv, _, D = patches.shape
                        grid = patches.reshape(Bv, 16, 16, D).permute(0, 3, 1, 2)
                        pooled = torch.nn.functional.adaptive_avg_pool2d(grid, PATCH_GRID)
                        feats = pooled.permute(0, 2, 3, 1).reshape(
                            Bv, PATCH_GRID * PATCH_GRID, D
                        )                                          # (3, 16, 384)
                        vision_tokens = feats.unsqueeze(0)        # (1, 3, 16, 384)
                    if zero_vision:
                        vision_tokens = torch.zeros_like(vision_tokens)

                port_override = None if gt_port else NOMINAL_PORT_POS
                state = get_state(d, idx, progress, port_override=port_override)
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                state_norm = (state_t - s_mean) / s_std

                with torch.no_grad():
                    action_norm = act(vision_tokens, state_norm).squeeze(0)
                    chunk = (action_norm * a_std + a_mean).cpu().numpy()
                chunk_idx = 0

            ctrl = chunk[chunk_idx]
            chunk_idx += 1
            n_control += 1

        for i, ai in enumerate(idx["aids"]):
            d.ctrl[ai] = ctrl[i]
        d.ctrl[6] = 0.0
        mujoco.mj_step(m, d)

    tip = d.xpos[idx["tip_id"]]
    port = d.xpos[idx["port_id"]]
    xy_err = np.linalg.norm(tip[:2] - port[:2])
    z_rel = (tip[2] - port[2]) * 1000
    success = xy_err < 0.005 and z_rel < -5.0
    return success, xy_err, z_rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gt-port", action="store_true",
                        help="Inject ground-truth port_pos into state (debug)")
    parser.add_argument("--zero-vision", action="store_true",
                        help="Replace vision features with zeros (debug — does model use vision?)")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    idx = setup_indices(m)

    observer = MuJoCoObserver(m, d, image_scale=0.25)

    print("Loading DINOv2...")
    processor, dinov2 = load_dinov2(device)

    print("Loading ACT...")
    payload = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        cfg = payload["config"]
        sd = payload["state_dict"]
    else:
        # Legacy bare state_dict — infer n_spatial from input_pos shape:
        # input_pos shape = (n_cams * n_spatial + 1, d_model)
        n_pos = payload["input_pos"].shape[0]
        n_spatial = max(1, (n_pos - 1) // 3)
        cfg = {"mode": "cls" if n_spatial == 1 else "patches", "n_spatial": n_spatial}
        sd = payload
    print(f"ACT config: mode={cfg.get('mode')}, n_spatial={cfg['n_spatial']}")
    act = ACTHead(n_spatial=cfg["n_spatial"]).to(device).eval()
    act.load_state_dict(sd)
    stats = np.load(STATS)
    mode = cfg.get("mode", "cls")

    nominal_pos = m.body_pos[idx["board_id"]].copy()
    nominal_quat = m.body_quat[idx["board_id"]].copy()
    rng = np.random.default_rng(args.seed)

    successes = 0
    t_start = time.time()
    for trial in range(args.trials):
        if args.trials > 1:
            randomize_board(m, idx, rng, nominal_pos, nominal_quat)
        success, xy_err, z_rel = run_trial(
            m, d, idx, observer, dinov2, processor, act, stats, device,
            gt_port=args.gt_port, mode=mode, zero_vision=args.zero_vision,
        )
        successes += success
        status = "OK" if success else "FAIL"
        print(f"  [{trial+1}/{args.trials}] {status}  xy={xy_err*1000:5.1f}mm  z={z_rel:6.1f}mm")

    m.body_pos[idx["board_id"]] = nominal_pos
    m.body_quat[idx["board_id"]] = nominal_quat
    observer.close()

    elapsed = time.time() - t_start
    print(f"\n{successes}/{args.trials} succeeded ({successes/args.trials:.0%}) in {elapsed:.0f}s "
          f"({elapsed/args.trials:.1f}s/trial)")


if __name__ == "__main__":
    main()
