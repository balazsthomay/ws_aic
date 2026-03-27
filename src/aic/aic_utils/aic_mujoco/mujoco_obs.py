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

# Distinct colors for key scene components (applied at model load)
_BODY_COLORS = {
    "nic_card_link": [0.1, 0.6, 0.2, 1.0],           # green PCB
    "sfp_module_link": [0.2, 0.3, 0.9, 1.0],          # blue plug
    "sfp_tip_link": [0.2, 0.3, 0.9, 1.0],
    "lc_plug_link": [0.6, 0.2, 0.7, 1.0],             # purple connector
    "task_board_base_link": [0.4, 0.35, 0.3, 1.0],    # dark wood
    "sc_port_0::sc_port_link": [0.8, 0.1, 0.1, 1.0],  # red port
    "enclosure_link": [0.2, 0.2, 0.22, 1.0],          # dark enclosure
    "tabletop": [0.2, 0.2, 0.22, 1.0],
}


def enhance_scene_visibility(model: "mujoco.MjModel") -> None:
    """Strip materials/textures and apply distinct colors for camera visibility.

    MuJoCo rendering precedence: material > geom_rgba.
    We remove all material references so geom_rgba controls color directly.
    """
    # Step 1: Remove material reference from ALL geoms, set warm gray default
    # Use slightly warm gray so backgrounds have color (not R=G=B gray)
    for i in range(model.ngeom):
        model.geom_matid[i] = -1
        if model.geom_rgba[i, 3] > 0:
            model.geom_rgba[i] = [0.30, 0.28, 0.26, 1.0]  # warm brown-gray

    # Step 2: Color specific bodies
    for body_name, rgba in _BODY_COLORS.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            continue
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == bid:
                model.geom_rgba[i] = rgba

    # Step 3: Floor → dark blue-gray
    for i in range(model.ngeom):
        if "floor" in model.geom(i).name.lower():
            model.geom_rgba[i] = [0.12, 0.12, 0.18, 1.0]

    # Step 4: Cable links → orange
    for i in range(model.ngeom):
        bname = model.body(model.geom_bodyid[i]).name
        if bname.startswith("link_") or bname.startswith("cable_"):
            model.geom_rgba[i] = [0.9, 0.4, 0.1, 1.0]

    # Step 5: Robot links → light blue-gray
    robot_bodies = {"shoulder_link", "upper_arm_link", "forearm_link",
                    "wrist_1_link", "wrist_2_link", "wrist_3_link"}
    for i in range(model.ngeom):
        bname = model.body(model.geom_bodyid[i]).name
        if bname in robot_bodies:
            model.geom_rgba[i] = [0.55, 0.58, 0.65, 1.0]

    # Step 6: Tune lighting — reduce all lights to avoid blowout
    model.vis.headlight.ambient[:] = [0.05, 0.05, 0.05]
    model.vis.headlight.diffuse[:] = [0.15, 0.15, 0.15]
    model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]
    for i in range(model.nlight):
        model.light_diffuse[i] = np.clip(model.light_diffuse[i] * 0.3, 0, 1)
        model.light_specular[i] = np.clip(model.light_specular[i] * 0.3, 0, 1)
        model.light_ambient[i] = np.clip(model.light_ambient[i] * 0.2, 0, 1)


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
