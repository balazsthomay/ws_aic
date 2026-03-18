#!/usr/bin/env python3
"""Relax the cable to equilibrium and save the state as a MuJoCo keyframe.

Uses eq_active to deactivate the weld during free settle, then activates it
and lets the cable adjust. Robot is held at home pose throughout.

Usage:
    python relax_cable.py [--settle-time 15] [--weld-time 15] [--verify-time 10]
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MJCF_DIR = SCRIPT_DIR.parent / "mjcf"
SCENE_PATH = MJCF_DIR / "scene.xml"
WORLD_PATH = MJCF_DIR / "aic_world.xml"

# UR5e home joint positions from aic_bringup/scripts/home_robot.py
HOME_JOINTS = {
    "shoulder_pan_joint": 0.6,
    "shoulder_lift_joint": -1.3,
    "elbow_joint": -1.9,
    "wrist_1_joint": -1.57,
    "wrist_2_joint": 1.57,
    "wrist_3_joint": 0.6,
}

# Damping values: original vs relaxation
DAMPING_FINAL = 'damping="2.0" armature="0.1"'
DAMPING_HIGH = 'damping="5.0" armature="0.1"'
WELD_SOLREF_FINAL = 'solref="0.1 1"'


def get_joint_indices(model, names):
    """Get (qpos_indices, dof_indices) for named joints."""
    qpos_idx, dof_idx = [], []
    for i in range(model.njnt):
        if model.joint(i).name in names:
            adr_q = model.jnt_qposadr[i]
            adr_v = model.jnt_dofadr[i]
            nq = {0: 7, 1: 4, 2: 1, 3: 1}[model.jnt_type[i]]
            nv = {0: 6, 1: 3, 2: 1, 3: 1}[model.jnt_type[i]]
            qpos_idx.extend(range(adr_q, adr_q + nq))
            dof_idx.extend(range(adr_v, adr_v + nv))
    return qpos_idx, dof_idx


def set_home_pose(model, data):
    """Set robot joints to home and return (home_qpos_values, ctrl_values)."""
    for i in range(model.njnt):
        name = model.joint(i).name
        if name in HOME_JOINTS:
            data.qpos[model.jnt_qposadr[i]] = HOME_JOINTS[name]

    ctrl = np.zeros(model.nu)
    for i in range(model.nu):
        jnt_id = model.actuator(i).trnid[0]
        jnt_name = model.joint(jnt_id).name
        ctrl[i] = HOME_JOINTS.get(jnt_name, 0.0)
    data.ctrl[:] = ctrl
    return ctrl


def find_weld_index(model):
    """Find the equality constraint index for the cable weld."""
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
            b1 = model.body(model.eq_obj1id[i]).name
            b2 = model.body(model.eq_obj2id[i]).name
            if "tool_link" in b1 or "tool_link" in b2:
                return i
    return None


def xml_set_damping(world_path, value):
    """Set cable default damping in the XML."""
    text = world_path.read_text()
    import re
    text = re.sub(r'damping="[\d.]+" armature="[\d.]+"', value, text, count=1)
    world_path.write_text(text)


def write_keyframe(scene_path, qpos, ctrl):
    """Write <key> element into scene.xml."""
    tree = ET.parse(scene_path)
    root = tree.getroot()
    for kf in root.findall("keyframe"):
        root.remove(kf)
    kf_elem = ET.SubElement(root, "keyframe")
    key = ET.SubElement(kf_elem, "key")
    key.set("name", "home")
    key.set("qpos", " ".join(f"{v:.8g}" for v in qpos))
    key.set("ctrl", " ".join(f"{v:.8g}" for v in ctrl))
    tree.write(str(scene_path), xml_declaration=False, encoding="unicode")
    print(f"  Keyframe written (nq={len(qpos)}, nu={len(ctrl)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settle-time", type=float, default=15.0,
                        help="Free settle time (weld off)")
    parser.add_argument("--weld-time", type=float, default=15.0,
                        help="Constrained settle time (weld on, high damping)")
    parser.add_argument("--verify-time", type=float, default=10.0)
    args = parser.parse_args()

    # Use high damping during relaxation
    xml_set_damping(WORLD_PATH, DAMPING_HIGH)

    try:
        m = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        d = mujoco.MjData(m)
        print(f"Model: nq={m.nq}, nv={m.nv}, nu={m.nu}, neq={m.neq}")

        robot_names = set(HOME_JOINTS.keys()) | {"gripper/left_finger_joint", "gripper/right_finger_joint"}
        robot_qpos_idx, robot_dof_idx = get_joint_indices(m, robot_names)

        # Set home pose
        ctrl = set_home_pose(m, d)
        robot_home_qpos = d.qpos[robot_qpos_idx].copy()
        print(f"Home ctrl: {ctrl}")

        # Find and disable weld
        weld_idx = find_weld_index(m)
        if weld_idx is not None:
            print(f"Weld constraint index: {weld_idx}")
            m.eq_active0[weld_idx] = 0  # disable in model defaults
            d.eq_active[weld_idx] = 0   # disable in data
        else:
            print("WARNING: No weld constraint found")

        # Forward to initialize
        mujoco.mj_forward(m, d)

        # --- Phase 1: Free settle (weld off) ---
        print(f"\n=== Phase 1: Free settle ({args.settle_time}s, weld off) ===")
        n_steps = int(args.settle_time / m.opt.timestep)
        for step in range(n_steps):
            mujoco.mj_step(m, d)
            # Freeze robot at home
            d.qpos[robot_qpos_idx] = robot_home_qpos
            d.qvel[robot_dof_idx] = 0.0

            if np.any(np.isnan(d.qpos)):
                print(f"  NaN at step {step}")
                return False
            if step % 5000 == 4999:
                cv = np.max(np.abs(d.qvel))
                print(f"  t={d.time:.1f}s  max_vel={cv:.4f}")

        cv = np.max(np.abs(d.qvel))
        print(f"  Phase 1 done: max_vel={cv:.4f}")

        # --- Phase 2: Enable weld, keep high damping ---
        print(f"\n=== Phase 2: Weld active ({args.weld_time}s, high damping) ===")
        if weld_idx is not None:
            d.eq_active[weld_idx] = 1

        n_steps = int(args.weld_time / m.opt.timestep)
        for step in range(n_steps):
            mujoco.mj_step(m, d)
            d.qpos[robot_qpos_idx] = robot_home_qpos
            d.qvel[robot_dof_idx] = 0.0

            if np.any(np.isnan(d.qpos)):
                print(f"  NaN at step {step}")
                return False
            if step % 5000 == 4999:
                cv = np.max(np.abs(d.qvel))
                print(f"  t={d.time:.1f}s  max_vel={cv:.4f}")

        cv = np.max(np.abs(d.qvel))
        print(f"  Phase 2 done: max_vel={cv:.4f}")

        # Set robot exactly at home, zero all velocities
        d.qpos[robot_qpos_idx] = robot_home_qpos
        d.qvel[:] = 0.0
        final_qpos = d.qpos.copy()
        final_ctrl = ctrl.copy()

    finally:
        # Set final production damping
        xml_set_damping(WORLD_PATH, DAMPING_FINAL)

    # --- Write keyframe ---
    print("\n=== Writing keyframe ===")
    write_keyframe(SCENE_PATH, final_qpos, final_ctrl)

    # --- Phase 3: Verify from keyframe with final params ---
    print(f"\n=== Phase 3: Verify ({args.verify_time}s) ===")
    m3 = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    d3 = mujoco.MjData(m3)
    mujoco.mj_resetDataKeyframe(m3, d3, 0)

    n_steps = int(args.verify_time / m3.opt.timestep)
    max_vel_settled = 0
    for step in range(n_steps):
        mujoco.mj_step(m3, d3)
        if np.any(np.isnan(d3.qpos)):
            print(f"  NaN at t={d3.time:.4f}s")
            return False
        if step > n_steps // 2:
            v = np.max(np.abs(d3.qvel))
            max_vel_settled = max(max_vel_settled, v)
        if step % 5000 == 4999:
            print(f"  t={d3.time:.1f}s  max_vel={np.max(np.abs(d3.qvel)):.4f}")

    max_err = 0.0
    for name, target in HOME_JOINTS.items():
        for i in range(m3.njnt):
            if m3.joint(i).name == name:
                q = d3.qpos[m3.jnt_qposadr[i]]
                err = abs(q - target) * 180 / np.pi
                max_err = max(max_err, err)
                tag = "OK" if err < 3 else "SAG" if err < 10 else "BAD"
                print(f"  {name:30s} err={err:.2f} deg {tag}")

    print(f"\n  Max vel (settled): {max_vel_settled:.4f} rad/s")
    print(f"  Max robot err: {max_err:.2f} deg")

    if max_vel_settled < 5 and max_err < 5:
        print("\n  SUCCESS")
        return True
    elif max_vel_settled < 50 and max_err < 15:
        print("\n  MARGINAL")
        return True
    else:
        print("\n  FAILED")
        return False


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
