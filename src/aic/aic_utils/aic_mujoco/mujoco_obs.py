"""MuJoCo observation pipeline matching the AIC competition interface.

Captures the same data as the ROS2 Observation message:
- 3 wrist camera images (left, center, right) at configurable resolution
- Joint state (positions, velocities)
- TCP pose and velocity
- Wrist wrench (force/torque)

Usage:
    obs = MuJoCoObserver(model, data)
    observation = obs.get_observation()
    # observation.images: dict of camera_name -> np.ndarray (H, W, 3) uint8
    # observation.joint_positions: np.ndarray (6,)
    # observation.joint_velocities: np.ndarray (6,)
    # observation.tcp_pos: np.ndarray (3,)
    # observation.tcp_quat: np.ndarray (4,) wxyz
    # observation.tcp_vel: np.ndarray (6,) linear + angular
    # observation.wrench: np.ndarray (6,) force + torque
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np


CAMERA_NAMES = ["left_camera", "center_camera", "right_camera"]
JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def enhance_scene_visibility(model: "mujoco.MjModel") -> None:
    """Assign distinct bright colors to each body for camera visibility.

    The original visual meshes (group 3) are dark/black. Instead we show
    collision primitives (group 0) with per-body semantic colors, giving
    cameras the contrast needed for a learned policy.
    """
    # Strip all material references so geom_rgba takes effect
    for i in range(model.ngeom):
        model.geom_matid[i] = -1

    # Assign a unique hue per body (semantic coloring)
    rng = np.random.RandomState(0)  # deterministic colors
    body_colors = {}
    for i in range(model.nbody):
        # Generate a bright, saturated color via HSV
        hue = (i * 0.618033988) % 1.0  # golden ratio spacing
        sat = 0.6 + rng.random() * 0.3
        val = 0.5 + rng.random() * 0.3
        # HSV to RGB
        h = hue * 6.0
        c = val * sat
        x = c * (1 - abs(h % 2 - 1))
        m = val - c
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        body_colors[i] = [r + m, g + m, b + m, 1.0]

    # Override key bodies with semantically meaningful colors
    _KEY_COLORS = {
        "nic_card_link": [0.1, 0.8, 0.2, 1.0],          # vivid green PCB
        "sfp_module_link": [0.2, 0.3, 1.0, 1.0],        # bright blue plug
        "sfp_tip_link": [0.2, 0.3, 1.0, 1.0],
        "lc_plug_link": [0.8, 0.2, 0.9, 1.0],           # vivid purple
        "task_board_base_link": [0.7, 0.5, 0.3, 1.0],   # warm tan
        "sc_port_0::sc_port_link": [1.0, 0.15, 0.1, 1.0],  # vivid red
        "enclosure_link": [0.25, 0.28, 0.35, 1.0],      # dark blue-steel
        "tabletop": [0.45, 0.35, 0.25, 1.0],            # medium wood
        "floor_link": [0.15, 0.18, 0.22, 1.0],          # dark blue floor
    }
    for name, rgba in _KEY_COLORS.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            body_colors[bid] = rgba

    # Cable → orange
    for i in range(model.nbody):
        bname = model.body(i).name
        if bname.startswith("link_") or bname.startswith("cable_"):
            body_colors[i] = [0.9, 0.45, 0.1, 1.0]

    # Robot → light blue-gray
    for name in ["shoulder_link", "upper_arm_link", "forearm_link",
                 "wrist_1_link", "wrist_2_link", "wrist_3_link"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            body_colors[bid] = [0.6, 0.65, 0.75, 1.0]

    # Apply colors to all geoms
    for i in range(model.ngeom):
        bid = model.geom_bodyid[i]
        if bid in body_colors and model.geom_rgba[i, 3] > 0:
            model.geom_rgba[i] = body_colors[bid]

    # Hide robot geoms that block camera view (wrist, tool, gripper)
    _HIDE_BODIES = {
        "wrist_3_link", "ati/tool_link",
        "center_camera/sensor_link", "center_camera/optical",
        "left_camera/sensor_link", "left_camera/optical",
        "right_camera/sensor_link", "right_camera/optical",
        "gripper/hande_base_link",
        "gripper/hande_finger_link_l", "gripper/hande_finger_link_r",
    }
    for i in range(model.ngeom):
        bname = model.body(model.geom_bodyid[i]).name
        if bname in _HIDE_BODIES:
            model.geom_rgba[i, 3] = 0.0  # fully transparent

    # Make enclosure geoms vary in shade to break uniformity
    enc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "enclosure_link")
    if enc_id >= 0:
        rng2 = np.random.RandomState(7)
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == enc_id:
                base = np.array([0.25, 0.28, 0.35])
                jitter = rng2.uniform(-0.08, 0.08, 3)
                model.geom_rgba[i, :3] = np.clip(base + jitter, 0.1, 0.5)

    # Tabletop → alternating warm/cool patches
    tab_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tabletop")
    if tab_id >= 0:
        idx = 0
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == tab_id:
                if idx % 2 == 0:
                    model.geom_rgba[i] = [0.45, 0.38, 0.3, 1.0]
                else:
                    model.geom_rgba[i] = [0.3, 0.35, 0.42, 1.0]
                idx += 1

    # Balanced lighting
    model.vis.headlight.ambient[:] = [0.15, 0.15, 0.15]
    model.vis.headlight.diffuse[:] = [0.35, 0.35, 0.35]
    model.vis.headlight.specular[:] = [0.15, 0.15, 0.15]
    for i in range(model.nlight):
        model.light_diffuse[i] = np.clip(model.light_diffuse[i] * 0.25, 0, 1)
        model.light_specular[i] = np.clip(model.light_specular[i] * 0.2, 0, 1)


@dataclass
class Observation:
    images: dict[str, np.ndarray] = field(default_factory=dict)
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(6))
    tcp_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tcp_quat: np.ndarray = field(default_factory=lambda: np.zeros(4))
    tcp_vel: np.ndarray = field(default_factory=lambda: np.zeros(6))
    wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))
    time: float = 0.0


class MuJoCoObserver:
    """Captures observations from MuJoCo matching the AIC sensor interface."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        image_scale: float = 0.25,
        enhance_visibility: bool = True,
    ):
        self.model = model
        self.data = data

        if enhance_visibility:
            enhance_scene_visibility(model)

        # Camera setup
        self.cam_ids = {}
        full_res = None
        for name in CAMERA_NAMES:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                raise ValueError(f"Camera '{name}' not found in model")
            self.cam_ids[name] = cid
            if full_res is None:
                full_res = model.cam_resolution[cid]

        self.img_w = int(full_res[0] * image_scale)
        self.img_h = int(full_res[1] * image_scale)
        self.renderer = mujoco.Renderer(model, height=self.img_h, width=self.img_w)

        # Joint indices
        self.qpos_ids = np.array([
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES
        ])
        self.dof_ids = np.array([
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINT_NAMES
        ])

        # TCP site and F/T sensor site
        self.tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
        self.ft_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "AtiForceTorqueSensor")

        # Jacobian workspace for TCP velocity
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def get_observation(self) -> Observation:
        obs = Observation()
        obs.time = self.data.time

        # Camera images
        for name in CAMERA_NAMES:
            self.renderer.update_scene(self.data, camera=name)
            obs.images[name] = self.renderer.render().copy()

        # Joint state
        obs.joint_positions = self.data.qpos[self.qpos_ids].copy()
        obs.joint_velocities = self.data.qvel[self.dof_ids].copy()

        # TCP pose
        obs.tcp_pos = self.data.site_xpos[self.tcp_site].copy()
        tcp_mat = self.data.site_xmat[self.tcp_site].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, tcp_mat.flatten())
        obs.tcp_quat = quat

        # TCP velocity via site Jacobian
        mujoco.mj_jacSite(
            self.model, self.data, self._jacp, self._jacr, self.tcp_site
        )
        qvel = self.data.qvel
        obs.tcp_vel = np.concatenate([
            self._jacp @ qvel,  # linear velocity
            self._jacr @ qvel,  # angular velocity
        ])

        # Wrench: approximate from joint torques at F/T sensor
        # Use contact forces as proxy for wrist wrench
        obs.wrench = self._compute_wrench()

        return obs

    def _compute_wrench(self) -> np.ndarray:
        """Compute approximate wrist wrench from contact forces on gripper bodies."""
        ft_pos = self.data.site_xpos[self.ft_site]
        ft_mat = self.data.site_xmat[self.ft_site].reshape(3, 3)

        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        # Sum contact forces on bodies below the F/T sensor
        ee_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ati/tool_link")
        ee_bodies = set()
        for i in range(self.model.nbody):
            # Walk up the parent chain to see if this body is under tool_link
            bid = i
            while bid > 0:
                if bid == ee_body:
                    ee_bodies.add(i)
                    break
                bid = self.model.body_parentid[bid]

        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1_body = self.model.geom_bodyid[con.geom1]
            g2_body = self.model.geom_bodyid[con.geom2]
            if g1_body not in ee_bodies and g2_body not in ee_bodies:
                continue

            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, c, force)
            contact_force = con.frame.reshape(3, 3).T @ force[:3]

            # Sign: force on the EE body
            if g2_body in ee_bodies:
                contact_force = -contact_force

            total_force += contact_force
            moment_arm = con.pos - ft_pos
            total_torque += np.cross(moment_arm, contact_force)

        # Express in F/T sensor frame
        wrench = np.zeros(6)
        wrench[:3] = ft_mat.T @ total_force
        wrench[3:] = ft_mat.T @ total_torque
        return wrench

    def close(self):
        self.renderer.close()
