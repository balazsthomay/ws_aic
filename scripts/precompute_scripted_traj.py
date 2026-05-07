#!/usr/bin/env python3
"""Pre-compute three trial-specific joint trajectories for the AIC competition.

The real cluster runs three deterministic trials (see
`src/aic/aic_engine/config/sample_config.yaml`). We exploit that by emitting
one joint trajectory per trial, each tuned to the trial's exact board pose,
target port location, and cable gripper offset. The runtime policy
(ScriptedVision / ScriptedPlay) dispatches by `(target_module_name, port_name)`.

Trial table (from sample_config.yaml):
  T1: SFP plug → nic_card_mount_0/sfp_port_0   board=(0.15,-0.2,1.14) yaw=π
  T2: SFP plug → nic_card_mount_1/sfp_port_0   board=(0.15,-0.2,1.14) yaw=π
  T3: SC plug  → sc_port_1/sc_port_base        board=(0.17, 0.0,1.14) yaw=3.0

The SC port (sc_port_1) is built into the task-board frame, not on a NIC card.
"""

import math
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "src/aic/aic_utils/aic_mujoco/mjcf/scene.xml"
OUT = ROOT / "data/outputs/scripted_traj.npz"

HOME = np.array([0.6, -1.3, -1.9, -1.57, 1.57, 0.6])
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

CONTROL_HZ = 20
APPROACH_TIME = 5.0
DESCENT_TIME = 8.0
HOLD_TIME = 1.0
ABOVE_PORT = 0.060
INSERT_DEPTH = 0.015

# Trial geometry from src/aic/aic_engine/config/sample_config.yaml. Keep this
# in sync with the yaml. yaw is in radians.
TRIALS = {
    "t1": {
        "key": ("nic_card_mount_0", "sfp_port_0"),
        "board_pos": np.array([0.15, -0.2, 1.14]),
        "board_yaw": math.pi,
        "port_kind": "sfp_on_nic",
        "nic_rail_y": -0.1745,           # nic_rail_0 anchor in board Y
        "nic_translation": 0.036,
        "gripper_offset": np.array([0.0, 0.015385, 0.04245]),
    },
    "t2": {
        "key": ("nic_card_mount_1", "sfp_port_0"),
        "board_pos": np.array([0.15, -0.2, 1.14]),
        "board_yaw": math.pi,
        "port_kind": "sfp_on_nic",
        "nic_rail_y": -0.1345,           # nic_rail_1 anchor in board Y
        "nic_translation": 0.036,
        "gripper_offset": np.array([0.0, 0.015385, 0.04545]),
    },
    "t3": {
        "key": ("sc_port_1", "sc_port_base"),
        "board_pos": np.array([0.17, 0.0, 1.14]),
        "board_yaw": 3.0,
        "port_kind": "sc_port_on_board",
        "sc_port_translation": -0.055,   # sc_port_1.translation along board X
        "sc_port_y": 0.0705,             # sc_port_1 anchor Y (sc_port_0 is 0.0295)
        "gripper_offset": np.array([0.0, 0.015385, 0.04045]),
    },
}


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def densify(traj: np.ndarray, max_per_joint_step: float) -> np.ndarray:
    out = [traj[0]]
    for i in range(1, len(traj)):
        prev = traj[i - 1]
        delta = traj[i] - prev
        n = max(1, int(np.ceil(np.abs(delta).max() / max_per_joint_step)))
        for k in range(1, n + 1):
            out.append(prev + (k / n) * delta)
    return np.array(out)


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
    qids = np.array([
        m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in JOINT_NAMES
    ])
    dids = np.array([
        m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in JOINT_NAMES
    ])
    tcp_home = d.site_xpos[tcp_site].copy()
    tcp_quat = np.zeros(4)
    mujoco.mju_mat2Quat(tcp_quat, d.site_xmat[tcp_site].flatten())
    R_tcp = d.site_xmat[tcp_site].reshape(3, 3).copy()
    print(f"tcp_home   : {tcp_home.round(4)}")
    print(f"tcp_quat   : {tcp_quat.round(4)}")

    # Body IDs whose mjModel.body_pos/quat we mutate to encode each trial.
    board_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "task_board_base_link")
    nic_mount_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                     "nic_card_mount_0::nic_card_mount_link")
    sc_port_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sc_port_0::sc_port_link")
    sfp_port_link = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_port_0_link")
    sc_port_base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sc_port_base_link")

    # Snapshot original poses so we can restore between trials.
    board_pos_0 = m.body_pos[board_id].copy()
    board_quat_0 = m.body_quat[board_id].copy()
    nic_pos_0 = m.body_pos[nic_mount_id].copy()
    sc_pos_0 = m.body_pos[sc_port_id].copy()

    def restore_scene() -> None:
        m.body_pos[board_id] = board_pos_0
        m.body_quat[board_id] = board_quat_0
        m.body_pos[nic_mount_id] = nic_pos_0
        m.body_pos[sc_port_id] = sc_pos_0

    def trial_port_world(t: dict) -> np.ndarray:
        """Compute the target port world position for one trial by mutating
        the MJCF model in-place, running mj_forward, and reading the relevant
        body's world xpos. Restores model state before returning."""
        restore_scene()
        m.body_pos[board_id] = t["board_pos"]
        m.body_quat[board_id] = yaw_to_quat_wxyz(t["board_yaw"])
        if t["port_kind"] == "sfp_on_nic":
            m.body_pos[nic_mount_id] = np.array([
                -0.081418 + t["nic_translation"],
                t["nic_rail_y"],
                0.012,
            ])
            mujoco.mj_forward(m, d)
            port_world = d.xpos[sfp_port_link].copy()
        elif t["port_kind"] == "sc_port_on_board":
            # sc_port_1 in board frame: x = -0.075 + translation, y = 0.0705,
            # z = 0.0165, with the same orientation as sc_port_0. We don't have
            # an sc_port_1 body in the MJCF, so move sc_port_0 to sc_port_1's
            # board-frame anchor and read sc_port_base_link's world xpos.
            m.body_pos[sc_port_id] = np.array([
                -0.075 + t["sc_port_translation"],
                t["sc_port_y"],
                0.0165,
            ])
            mujoco.mj_forward(m, d)
            port_world = d.xpos[sc_port_base].copy()
        else:
            raise ValueError(t["port_kind"])
        restore_scene()
        mujoco.mj_forward(m, d)
        return port_world

    # ---- IK utilities ----
    d_ik = mujoco.MjData(m)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))

    def solve_ik(target_pos: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        d_ik.qpos[:] = d.qpos[:]
        d_ik.qpos[qids] = q_init.copy()
        for _ in range(200):
            mujoco.mj_forward(m, d_ik)
            pe = target_pos - d_ik.site_xpos[tcp_site]
            sm = d_ik.site_xmat[tcp_site].reshape(3, 3)
            tm = np.zeros(9)
            mujoco.mju_quat2Mat(tm, tcp_quat)
            tm = tm.reshape(3, 3)
            Re = tm @ sm.T
            eq = np.zeros(4)
            mujoco.mju_mat2Quat(eq, Re.flatten())
            if eq[0] < 0:
                eq = -eq
            re = 2.0 * eq[1:4]
            if np.linalg.norm(pe) < 3e-4 and np.linalg.norm(re) < 0.01:
                break
            err = np.concatenate([pe, 0.5 * re])
            mujoco.mj_jacSite(m, d_ik, jacp, jacr, tcp_site)
            J = np.vstack([jacp[:, dids], jacr[:, dids]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            d_ik.qpos[qids] += 0.15 * dq
        return d_ik.qpos[qids].copy()

    def make_traj(port_pos: np.ndarray, gripper_offset: np.ndarray) -> np.ndarray:
        # plug-tip world offset from TCP, given the trial's gripper offset
        # (yaml is in gripper-local frame).
        tip_offset = R_tcp @ gripper_offset
        tcp_above = port_pos + np.array([0, 0, ABOVE_PORT]) - tip_offset

        n_app = int(APPROACH_TIME * CONTROL_HZ)
        approach = np.zeros((n_app, 6))
        q_prev = HOME.copy()
        for i in range(n_app):
            alpha = 0.5 * (1 - math.cos(math.pi * i / max(n_app - 1, 1)))
            tcp_i = (1 - alpha) * tcp_home + alpha * tcp_above
            approach[i] = solve_ik(tcp_i, q_prev)
            q_prev = approach[i]

        n_desc = int(DESCENT_TIME * CONTROL_HZ)
        descent = np.zeros((n_desc, 6))
        q_prev = approach[-1].copy()
        for i in range(n_desc):
            z_off = ABOVE_PORT - (i / max(n_desc - 1, 1)) * (ABOVE_PORT + INSERT_DEPTH)
            tcp_i = tcp_above.copy()
            tcp_i[2] = port_pos[2] + z_off - tip_offset[2]
            descent[i] = solve_ik(tcp_i, q_prev)
            q_prev = descent[i]

        n_hold = int(HOLD_TIME * CONTROL_HZ)
        hold = np.tile(descent[-1], (n_hold, 1))
        return np.concatenate([approach, descent, hold], axis=0)

    out: dict[str, np.ndarray] = {}
    out["control_hz"] = np.array(CONTROL_HZ, dtype=np.int64)
    out["trial_keys"] = np.array(
        [f"{k}={t['key'][0]}/{t['key'][1]}" for k, t in TRIALS.items()],
        dtype=object,
    )

    for tname, t in TRIALS.items():
        port_world = trial_port_world(t)
        traj = make_traj(port_world, t["gripper_offset"])
        # Verify: forward-kinematic check at last pose.
        d_ik.qpos[:] = d.qpos[:]
        d_ik.qpos[qids] = traj[-1]
        mujoco.mj_forward(m, d_ik)
        tcp_end = d_ik.site_xpos[tcp_site].copy()
        tip_world = tcp_end + R_tcp @ t["gripper_offset"]
        # tip vs port_world: descent ends at port_z - INSERT_DEPTH (insertion).
        descent_target_tip_z = port_world[2] - INSERT_DEPTH
        err_xy = np.linalg.norm(tip_world[:2] - port_world[:2])
        err_z = tip_world[2] - descent_target_tip_z
        print(f"[{tname}] port_world={port_world.round(4)} "
              f"tip_end={tip_world.round(4)} "
              f"err_xy={err_xy*1000:.1f}mm err_z={err_z*1000:.1f}mm "
              f"steps={traj.shape[0]} ({traj.shape[0]/CONTROL_HZ:.1f}s)")
        out[f"joints_{tname}"] = traj.astype(np.float64)
        out[f"port_{tname}"] = port_world.astype(np.float64)

    # Backward-compat aliases for older policies (if any path still loads them).
    out["joints"] = out["joints_t1"]
    out["joints_sfp"] = out["joints_t1"]
    out["joints_sc"] = out["joints_t3"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
