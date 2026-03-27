#!/usr/bin/env python3
"""Verify the observation pipeline by saving camera images and printing state.

Usage:
    .venv/bin/python3 scripts/test_obs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src/aic/aic_utils/aic_mujoco"))

import math
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from mujoco_obs import MuJoCoObserver

SCENE = Path(__file__).resolve().parent.parent / "src/aic/aic_utils/aic_mujoco/mjcf/scene.xml"
OUT = Path(__file__).resolve().parent / "obs_verify"
OUT.mkdir(exist_ok=True)

HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

m = mujoco.MjModel.from_xml_path(str(SCENE))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

observer = MuJoCoObserver(m, d, image_scale=0.5)  # 0.5x for verification

# Capture at home
obs_home = observer.get_observation()

# IK to approach position (30mm above port)
tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
port_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_port_0_link")
tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_tip_link")
qids = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS])
dids = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS])
aids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_motor") for n in JOINTS])

port_pos = d.xpos[port_id].copy()
tip_off = d.xpos[tip_id] - d.site_xpos[tcp_site]
tcp_qh = np.zeros(4)
mujoco.mju_mat2Quat(tcp_qh, d.site_xmat[tcp_site].flatten())
tcp_target = port_pos + np.array([0, 0, 0.030]) - tip_off

d_ik = mujoco.MjData(m)
d_ik.qpos[:] = d.qpos[:]
d_ik.qpos[qids] = HOME.copy()
jp = np.zeros((3, m.nv))
jr = np.zeros((3, m.nv))
for _ in range(200):
    mujoco.mj_forward(m, d_ik)
    pe = tcp_target - d_ik.site_xpos[tcp_site]
    sm = d_ik.site_xmat[tcp_site].reshape(3, 3)
    tm = np.zeros(9)
    mujoco.mju_quat2Mat(tm, tcp_qh)
    tm = tm.reshape(3, 3)
    Re = tm @ sm.T
    eq = np.zeros(4)
    mujoco.mju_mat2Quat(eq, Re.flatten())
    if eq[0] < 0:
        eq = -eq
    re = 2.0 * eq[1:4]
    if np.linalg.norm(pe) < 3e-4:
        break
    err = np.concatenate([pe, 0.5 * re])
    mujoco.mj_jacSite(m, d_ik, jp, jr, tcp_site)
    J = np.vstack([jp[:, dids], jr[:, dids]])
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
    d_ik.qpos[qids] += 0.15 * dq

# Simulate to approach
mujoco.mj_resetDataKeyframe(m, d, 0)
q_approach = d_ik.qpos[qids].copy()
for s in range(5000):
    alpha = min(s / 3000, 1.0)
    alpha = 0.5 * (1 - math.cos(math.pi * alpha))
    ctrl = (1 - alpha) * HOME + alpha * q_approach
    for i, a in enumerate(aids):
        d.ctrl[a] = ctrl[i]
    d.ctrl[6] = 0.0
    mujoco.mj_step(m, d)

obs_approach = observer.get_observation()

# Save images
print("=== Observation Pipeline Verification ===\n")

for label, obs in [("home", obs_home), ("approach", obs_approach)]:
    print(f"--- {label} ---")
    print(f"  TCP:    {obs.tcp_pos}")
    print(f"  Joints: {np.array2string(obs.joint_positions, precision=2)}")
    print(f"  Wrench: F={np.linalg.norm(obs.wrench[:3]):.2f}N")

    imgs = []
    for name, img in obs.images.items():
        path = OUT / f"{name}_{label}.png"
        Image.fromarray(img).save(path)
        imgs.append((name, img))
        print(f"  {name}: [{img.min()},{img.max()}] mean={img.mean():.0f}")

    # Composite
    W, H = imgs[0][1].shape[1], imgs[0][1].shape[0]
    comp = Image.new("RGB", (W * 3, H + 25), (30, 30, 30))
    draw = ImageDraw.Draw(comp)
    for i, (name, img) in enumerate(imgs):
        comp.paste(Image.fromarray(img), (i * W, 25))
        draw.text((i * W + 5, 4), name, fill=(255, 255, 255))
    comp.save(OUT / f"all_cameras_{label}.png")

observer.close()
print(f"\nImages saved to {OUT}")
print("Open all_cameras_home.png and all_cameras_approach.png to verify.")
