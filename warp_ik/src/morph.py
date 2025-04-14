from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
import importlib
import json
import logging
import math
import os
import sys
import time
import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@dataclass
class MorphConfig:
    morph: str = 'template' # unique identifier for the morph (when testing use "blank" morph)
    dockerfile: str = os.environ.get("DOCKERFILE") # dockerfile variant
    root_dir: str = os.environ.get("WARP_IK_ROOT") # root directory of the warp-ik project
    assets_dir: str = f"{root_dir}/assets" # assets directory for the morphs
    output_dir: str = f"{root_dir}/output" # output directory for the morphs
    morph_dir: str = f"{root_dir}/warp_ik/morphs" # directory for the morphs
    morph_output_dir: str = f"{output_dir}/{morph}" # output directory for this unique morph
    seed: int = 42 # random seed
    device: str = None # nvidia device to run the simulation on
    headless: bool = False # turns off rendering
    num_envs: int = 16 # number of parallel environments
    num_rollouts: int = 2 # number of rollouts to perform
    train_iters: int = 64 # number of training iterations per rollout
    track: bool = False # turns on tracking with wandb
    wandb_entity: str = os.environ.get("WANDB_ENTITY", "hug")
    wandb_project: str = os.environ.get("WANDB_PROJECT", "warp-ik")
    created_on: str = datetime.now().strftime("%Y%m%d%H%M%S")
    start_time: float = 0.0 # start time for the simulation
    fps: int = 60 # frames per second
    step_size: float = 1.0 # step size in q space for updates
    urdf_path: str = f"{assets_dir}/trossen_arm_description/urdf/generated/wxai/wxai_follower.urdf" # path to the urdf file
    usd_output_path: str = f"{output_dir}/{morph}-recording.usd" # path to the usd file to save the model
    ee_link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0) # offset from the ee_gripper_link to the end effector
    gizmo_radius: float = 0.005 # radius of the gizmo (used for arrow base radius)
    gizmo_length: float = 0.05 # total length of the gizmo arrow
    gizmo_color_x_ee: tuple[float, float, float] = (1.0, 0.0, 0.0) # color of the x gizmo for the ee
    gizmo_color_y_ee: tuple[float, float, float] = (0.0, 1.0, 0.0) # color of the y gizmo for the ee
    gizmo_color_z_ee: tuple[float, float, float] = (0.0, 0.0, 1.0) # color of the z gizmo for the ee
    gizmo_color_x_target: tuple[float, float, float] = (1.0, 0.5, 0.5) # color of the x gizmo for the target
    gizmo_color_y_target: tuple[float, float, float] = (0.5, 1.0, 0.5) # color of the y gizmo for the target
    gizmo_color_z_target: tuple[float, float, float] = (0.5, 0.5, 1.0) # color of the z gizmo for the target
    arm_spacing_xz: float = 1.0 # spacing between arms in the x-z plane
    arm_height: float = 0.0 # height of the arm off the floor
    target_offset: tuple[float, float, float] = (0.3, 0, 0.1) # offset of the target from the base (in robot coordinates)
    target_spawn_pos_noise: float = 0.1 # maximum position noise to add when spawning targets
    target_spawn_rot_noise: float = np.pi/8  # maximum rotation angle noise to add when spawning targets
    joint_limits: list[tuple[float, float]] = field(default_factory=lambda: [
        (-3.054, 3.054),    # base
        (0.0, 3.14),        # shoulder
        (0.0, 2.356),       # elbow
        (-1.57, 1.57),      # wrist 1
        (-1.57, 1.57),      # wrist 2
        (-3.14, 3.14),      # wrist 3
        (0.0, 0.044),       # right finger
        (0.0, 0.044),       # left finger
    ]) # joint limits for arm
    arm_rot_offset: list[tuple[tuple[float, float, float], float]] = field(default_factory=lambda: [
        ((1.0, 0.0, 0.0), -math.pi * 0.5), # quarter turn about x-axis
        ((0.0, 0.0, 1.0), -math.pi * 0.5), # quarter turn about z-axis
    ]) # list of axis angle rotations for initial arm orientation offset
    # Home position for the arm joints
    # Format: [base, shoulder, elbow, wrist1, wrist2, wrist3, right_finger, left_finger]
    qpos_home: list[float] = field(default_factory=lambda: [
        0,          # base joint
        np.pi/12,   # shoulder joint
        np.pi/12,   # elbow joint
        0,          # wrist 1 joint
        0,          # wrist 2 joint
        0,          # wrist 3 joint
        0,          # right finger joint
        0           # left finger joint
    ])
    
    # Amount of random noise to add to the arm joint angles during initialization
    # This helps with exploration and prevents getting stuck in local minima
    q_angle_shuffle: list[float] = field(default_factory=lambda: [
        np.pi/2,    # base joint noise range
        np.pi/4,    # shoulder joint noise range
        np.pi/4,    # elbow joint noise range
        np.pi/4,    # wrist 1 joint noise range
        np.pi/4,    # wrist 2 joint noise range
        np.pi/4,    # wrist 3 joint noise range
        0.01,       # right finger joint noise range
        0.01        # left finger joint noise range
    ])
    config_extras: dict = field(default_factory=dict) # additional morph-specific config values
    joint_q_requires_grad: bool = True # whether to require grad for the joint q
    body_q_requires_grad: bool = True # whether to require grad for the body q
    joint_attach_ke: float = 1600.0 # stiffness for the joint attach
    joint_attach_kd: float = 20.0 # damping for the joint attach


class MorphState(Enum):
    NOT_RUN_YET = auto()
    ALREADY_RAN = auto()
    ERRORED_OUT = auto()


@dataclass(order=True)
class ActiveMorph:
    score: float
    name: str
    state: MorphState = MorphState.NOT_RUN_YET


def quat_to_rot_matrix_np(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2),     2 * (x * y - z * w),         2 * (x * z + y * w)],
        [2 * (x * y + z * w),           1 - 2 * (x ** 2 + z ** 2),   2 * (y * z - x * w)],
        [2 * (x * z - y * w),           2 * (y * z + x * w),         1 - 2 * (x ** 2 + y ** 2)]
    ], dtype=np.float32)

def quat_from_axis_angle_np(axis, angle):
    return np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])

def apply_transform_np(translation, quat, point):
    R = quat_to_rot_matrix_np(quat)
    return translation + R.dot(point)

def quat_mul_np(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)

def get_body_quaternions(state_body_q, num_links, ee_link_index):
    num_envs = state_body_q.shape[0] // num_links
    quats = np.empty((num_envs, 4), dtype=np.float32)
    for e in range(num_envs):
        t = state_body_q[e * num_links + ee_link_index]
        quats[e, :] = t[3:7]
    return quats

@wp.func
def quat_mul(q1: wp.quat, q2: wp.quat) -> wp.quat:
    return wp.quat(
        q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
        q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0],
        q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3],
        q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2],
    )

@wp.func
def quat_conjugate(q: wp.quat) -> wp.quat:
    return wp.quat(-q[0], -q[1], -q[2], q[3])

@wp.func
def quat_orientation_error(target: wp.quat, current: wp.quat) -> wp.vec3:
    q_err = quat_mul(target, quat_conjugate(current))
    qx = wp.select(q_err[3] < 0.0, -q_err[0], q_err[0])
    qy = wp.select(q_err[3] < 0.0, -q_err[1], q_err[1])
    qz = wp.select(q_err[3] < 0.0, -q_err[2], q_err[2])
    return wp.vec3(qx, qy, qz) * 2.0


@wp.kernel
def compute_ee_error_kernel(
    body_q: wp.array(dtype=wp.transform),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    target_pos: wp.array(dtype=wp.vec3),
    target_ori: wp.array(dtype=wp.quat),
    current_ori: wp.array(dtype=wp.quat),  # Precomputed current orientation
    error_out: wp.array(dtype=wp.float32)  # Flattened array (num_envs*6)
):
    tid = wp.tid()
    t = body_q[tid * num_links + ee_link_index]
    pos = wp.transform_point(t, ee_link_offset)
    ori = current_ori[tid]
    pos_err = target_pos[tid] - pos
    ori_err = quat_orientation_error(target_ori[tid], ori)
    base = tid * 6
    error_out[base + 0] = pos_err.x
    error_out[base + 1] = pos_err.y
    error_out[base + 2] = pos_err.z
    error_out[base + 3] = ori_err.x
    error_out[base + 4] = ori_err.y
    error_out[base + 5] = ori_err.z


class BaseMorph:

    def _update_config(self):
        raise NotImplementedError("Morphs must implement their own update_config function")

    def __init__(self, config: MorphConfig):

        self.config = config
        self._update_config()
        log.debug(f"config:{json.dumps(self.config.__dict__, indent=4)}")
        
        self.config.morph_output_dir = os.path.join(self.config.output_dir, self.config.morph)
        os.makedirs(config.morph_output_dir, exist_ok=True)
        log.debug(f"morph specific output_dir: {config.morph_output_dir}")

        config_filepath = os.path.join(config.morph_output_dir, "config.json")
        with open(config_filepath, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)

        if self.config.track:
            import wandb
            wandb.login()
            wandb.init(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project,
                name=f"{self.config.dockerfile}/{self.config.morph}/seed{self.config.seed}",
                config=self.config.__dict__
            )
            wandb.save(config_filepath)

        self.rng = np.random.default_rng(self.config.seed)
        self.num_envs = self.config.num_envs
        self.render_time = self.config.start_time
        self.fps = self.config.fps
        self.frame_dt = 1.0 / self.fps

        # Parse URDF and build model.
        articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_urdf(
            os.path.expanduser(self.config.urdf_path),
            articulation_builder,
            xform=wp.transform_identity(),
            floating=False
        )
        builder = wp.sim.ModelBuilder()
        self.step_size = self.config.step_size
        self.num_links = len(articulation_builder.joint_type)
        self.dof = len(articulation_builder.joint_q)
        self.joint_limits = self.config.joint_limits

        log.info(f"Parsed URDF with {self.num_links} links and {self.dof} dof")

        # Locate ee_gripper link.
        self.ee_link_offset = wp.vec3(self.config.ee_link_offset)
        self.ee_link_index = -1
        for i, joint in enumerate(articulation_builder.joint_name):
            if joint == "ee_gripper":
                self.ee_link_index = articulation_builder.joint_child[i]
                break
        if self.ee_link_index == -1:
            raise ValueError("Could not find ee_gripper joint in URDF")

        # Compute initial arm orientation.
        _initial_arm_orientation = None
        for axis, angle in self.config.arm_rot_offset:
            if _initial_arm_orientation is None:
                _initial_arm_orientation = wp.quat_from_axis_angle(wp.vec3(axis), angle)
            else:
                _initial_arm_orientation *= wp.quat_from_axis_angle(wp.vec3(axis), angle)
        self.initial_arm_orientation = _initial_arm_orientation
        log.debug(f"Initial arm orientation: {[self.initial_arm_orientation.x, self.initial_arm_orientation.y, self.initial_arm_orientation.z, self.initial_arm_orientation.w]}")

        # --- Revised target origin computation ---
        # Instead of using a raw grid, we compute each arm's target relative to its base transform.
        self.target_origin = []
        self.arm_spacing_xz = self.config.arm_spacing_xz
        self.arm_height = self.config.arm_height
        self.num_rows = int(math.sqrt(self.num_envs))
        log.info(f"Spawning {self.num_envs} arms in a grid of {self.num_rows}x{self.num_rows}")
        for e in range(self.num_envs):
            x = (e % self.num_rows) * self.arm_spacing_xz
            z = (e // self.num_rows) * self.arm_spacing_xz
            base_transform = wp.transform(wp.vec3(x, self.arm_height, z), self.initial_arm_orientation)
            # Convert base transform to numpy (for target computation).
            base_translation = np.array([x, self.arm_height, z], dtype=np.float32)
            base_quat = np.array([self.initial_arm_orientation.x, self.initial_arm_orientation.y,
                                   self.initial_arm_orientation.z, self.initial_arm_orientation.w], dtype=np.float32)
            # Define the target offset in robot coordinates and transform to world
            target_offset_local = np.array(self.config.target_offset, dtype=np.float32)
            target_world = apply_transform_np(base_translation, base_quat, target_offset_local)
            self.target_origin.append(target_world)
            builder.add_builder(articulation_builder, xform=wp.transform(wp.vec3(x, self.arm_height, z), self.initial_arm_orientation))
            num_joints_in_arm = len(self.config.qpos_home)
            for i in range(num_joints_in_arm):
                value = self.config.qpos_home[i] + self.rng.uniform(-self.config.q_angle_shuffle[i], self.config.q_angle_shuffle[i])
                builder.joint_q[-num_joints_in_arm + i] = np.clip(value, self.config.joint_limits[i][0], self.config.joint_limits[i][1])
        self.target_origin = np.array(self.target_origin)
        # Target orientation: default identity.
        self.target_ori = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (self.num_envs, 1))
        log.debug(f"Initial target orientations (first env): {self.target_ori[0]}")

        # Finalize model.
        self.model = builder.finalize()
        self.model.ground = False
        self.model.joint_q.requires_grad = self.config.joint_q_requires_grad
        self.model.body_q.requires_grad = self.config.body_q_requires_grad
        self.model.joint_attach_ke = self.config.joint_attach_ke
        self.model.joint_attach_kd = self.config.joint_attach_kd
        self.integrator = wp.sim.SemiImplicitIntegrator()
        if not self.config.headless:
            self.usd_output_path = os.path.join(self.config.morph_output_dir, "recording.usd")
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.expanduser(self.usd_output_path))
        else:
            self.renderer = None

        # Simulation state.
        self.ee_pos = wp.zeros(self.num_envs, dtype=wp.vec3, requires_grad=True)
        self.ee_error = wp.zeros(self.num_envs * 6, dtype=wp.float32, requires_grad=True)
        self.state = self.model.state(requires_grad=True)
        self.targets = self.target_origin.copy()
        self.profiler = {}

    def compute_ee_error(self):
        with wp.ScopedTimer("forward_kinematics", print=False, active=True, dict=self.profiler):
            wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)
            host_quats = get_body_quaternions(self.state.body_q.numpy(), self.num_links, self.ee_link_index)
            device_quats = wp.array(host_quats, dtype=wp.quat)
        
        with wp.ScopedTimer("error_compute", print=False, active=True, dict=self.profiler):
            wp.launch(
                compute_ee_error_kernel,
                dim=self.num_envs,
                inputs=[
                    self.state.body_q,
                    self.num_links,
                    self.ee_link_index,
                    self.ee_link_offset,
                    wp.array(self.targets, dtype=wp.vec3),
                    wp.array(self.target_ori, dtype=wp.quat),
                    device_quats,
                ],
                outputs=[self.ee_error],
            )
            error_np = self.ee_error.numpy()
            log.debug(f"EE error (env 0): Pos [{error_np[0]:.4f}, {error_np[1]:.4f}, {error_np[2]:.4f}], Ori [{error_np[3]:.4f}, {error_np[4]:.4f}, {error_np[5]:.4f}]")
        return self.ee_error

    def _step(self):
        raise NotImplementedError("Morphs must implement their own step function")

    def step(self):
        ee_error_flat = self.compute_ee_error().numpy()
        error = ee_error_flat.reshape(self.num_envs, 6, 1)
        with wp.ScopedTimer("_step", print=False, active=True, dict=self.profiler):
            self._step()

    def render_gizmos(self):
        if self.renderer is None:
            return

        # Ensure FK is up-to-date for rendering the current state
        # This might be redundant if called right after step(), but safe to include.
        # wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

        radius = self.config.gizmo_radius
        # Warp cones are defined by height along Y axis, so half_height is used for centering later
        cone_height = self.config.gizmo_length
        cone_half_height = cone_height / 2.0

        # --- Define Base Rotations for Gizmo Axes ---
        # We want the cone's length (local Y) to align with the world X, Y, or Z axis.
        # render_cone uses local Y as the height axis.
        # Rotate cone's Y axis to align with world X: Rotate +90 deg around Z
        rot_x_axis = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
        # Rotate cone's Y axis to align with world Y: Identity (no rotation needed)
        rot_y_axis = wp.quat_identity()
        # Rotate cone's Y axis to align with world Z: Rotate -90 deg around X
        rot_z_axis = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0)

        # Convert base rotations to numpy [x,y,z,w] format
        rot_x_np = np.array([rot_x_axis.x, rot_x_axis.y, rot_x_axis.z, rot_x_axis.w], dtype=np.float32)
        rot_y_np = np.array([rot_y_axis.x, rot_y_axis.y, rot_y_axis.z, rot_y_axis.w], dtype=np.float32)
        rot_z_np = np.array([rot_z_axis.x, rot_z_axis.y, rot_z_axis.z, rot_z_axis.w], dtype=np.float32)

        # Get current body transforms (needed for EE pose)
        # Ensure the data is on the host for processing
        current_body_q = self.state.body_q.numpy() # shape: (num_envs * num_links, 8) [px,py,pz, qx,qy,qz,qw, scale=1]

        for e in range(self.num_envs):
            # --- Target Gizmos ---
            target_pos_np = self.targets[e]
            target_ori_np = self.target_ori[e] # Shape [x,y,z,w]
            log.debug(f"Env {e} - Target pos: {target_pos_np}, Target ori: {target_ori_np}")

            # Calculate final orientation for each target axis gizmo
            target_rot_x_np = quat_mul_np(target_ori_np, rot_x_np)
            target_rot_y_np = quat_mul_np(target_ori_np, rot_y_np)
            target_rot_z_np = quat_mul_np(target_ori_np, rot_z_np)
            log.debug(f"Env {e} - Target gizmo orientations - X: {target_rot_x_np}, Y: {target_rot_y_np}, Z: {target_rot_z_np}")

            # Calculate position offset for cone base (tip is at origin before transform)
            # We want the *center* of the gizmo line at the target position.
            # Transform the offset vector (0, cone_half_height, 0) by the gizmo's rotation
            # and subtract it from the target position.
            offset_x = apply_transform_np(np.zeros(3), target_rot_x_np, np.array([0, cone_half_height, 0]))
            offset_y = apply_transform_np(np.zeros(3), target_rot_y_np, np.array([0, cone_half_height, 0]))
            offset_z = apply_transform_np(np.zeros(3), target_rot_z_np, np.array([0, cone_half_height, 0]))

            # Render target cones
            self.renderer.render_cone(
                name=f"target_x_{e}",
                pos=tuple(target_pos_np - offset_x),
                rot=tuple(target_rot_x_np),
                radius=radius,
                half_height=cone_half_height,
                color=self.config.gizmo_color_x_target
            )
            self.renderer.render_cone(
                name=f"target_y_{e}",
                pos=tuple(target_pos_np - offset_y),
                rot=tuple(target_rot_y_np),
                radius=radius,
                half_height=cone_half_height,
                color=self.config.gizmo_color_y_target
            )
            self.renderer.render_cone(
                name=f"target_z_{e}",
                pos=tuple(target_pos_np - offset_z),
                rot=tuple(target_rot_z_np),
                radius=radius,
                half_height=cone_half_height,
                color=self.config.gizmo_color_z_target
            )

            # --- End-Effector Gizmos ---
            # Get the transform of the EE link for the current environment
            ee_link_transform_flat = current_body_q[e * self.num_links + self.ee_link_index]
            ee_link_pos_np = ee_link_transform_flat[0:3]
            ee_link_ori_np = ee_link_transform_flat[3:7] # Shape [x,y,z,w]
            log.debug(f"Env {e} - EE tip pos: {ee_link_pos_np}, EE ori: {ee_link_ori_np}")

            # Calculate the world position of the EE tip (applying offset)
            ee_tip_pos_np = apply_transform_np(ee_link_pos_np, ee_link_ori_np, self.ee_link_offset)

            # Calculate final orientation for each EE axis gizmo
            ee_rot_x_np = quat_mul_np(ee_link_ori_np, rot_x_np)
            ee_rot_y_np = quat_mul_np(ee_link_ori_np, rot_y_np)
            ee_rot_z_np = quat_mul_np(ee_link_ori_np, rot_z_np)
            log.debug(f"Env {e} - EE gizmo orientations - X: {ee_rot_x_np}, Y: {ee_rot_y_np}, Z: {ee_rot_z_np}")

            # Calculate position offset for cone base
            ee_offset_x = apply_transform_np(np.zeros(3), ee_rot_x_np, np.array([0, cone_half_height, 0]))
            ee_offset_y = apply_transform_np(np.zeros(3), ee_rot_y_np, np.array([0, cone_half_height, 0]))
            ee_offset_z = apply_transform_np(np.zeros(3), ee_rot_z_np, np.array([0, cone_half_height, 0]))


            # Render EE cones
            self.renderer.render_cone(
                name=f"ee_pos_x_{e}",
                pos=tuple(ee_tip_pos_np - ee_offset_x), # Use tip position
                rot=tuple(ee_rot_x_np),             # Use calculated EE axis orientation
                radius=radius,
                half_height=cone_half_height,
                color=self.config.gizmo_color_x_ee
            )
            self.renderer.render_cone(
                name=f"ee_pos_y_{e}",
                pos=tuple(ee_tip_pos_np - ee_offset_y),
                rot=tuple(ee_rot_y_np),
                radius=radius,
                half_height=cone_half_height,
                color=self.config.gizmo_color_y_ee
            )
            self.renderer.render_cone(
                name=f"ee_pos_z_{e}",
                pos=tuple(ee_tip_pos_np - ee_offset_z),
                rot=tuple(ee_rot_z_np),
                radius=radius,
                half_height=cone_half_height,
                color=self.config.gizmo_color_z_ee
            )

    def render_error_bars(self):
        # TODO: render error bars on top of each robot for visualization
        pass

    def render(self):
        if self.renderer is None:
            return
        with wp.ScopedTimer("render", print=False, active=True, dict=self.profiler):
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.state)
            self.render_gizmos()
            self.render_error_bars()
            self.renderer.end_frame()
            self.render_time += self.frame_dt

def run_morph(config: MorphConfig) -> dict:
    morph_path = os.path.join(config.morph_dir, f"{config.morph}.py")
    log.info(f"running morph {config.morph} from {morph_path}")
    spec = importlib.util.spec_from_file_location(f"morphs.{config.morph}", morph_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MorphClass = module.Morph
    morph = MorphClass(config)
    # copy the morph python file to output directory
    morph_output_filepath = os.path.join(config.morph_output_dir, "morph.py")
    with open(morph_output_filepath, 'w') as f:
        f.write(module.__str__())
    wp.init()
    log.info(f"gpu enabled: {wp.get_device().is_cuda}")
    log.info("starting simulation")
    results = {}
    with wp.ScopedDevice(config.device):
        for i in range(config.num_rollouts):
            with wp.ScopedTimer("target_update", print=False, active=True, dict=morph.profiler):
                morph.targets = morph.target_origin.copy()
                morph.targets[:, :] += morph.rng.uniform(
                    -config.target_spawn_pos_noise / 2,
                    config.target_spawn_pos_noise / 2,
                    size=(morph.num_envs, 3)
                )
                random_axes = morph.rng.uniform(-1, 1, size=(morph.num_envs, 3))
                random_angles = morph.rng.uniform(-config.target_spawn_rot_noise, 
                                              config.target_spawn_rot_noise, 
                                              size=(morph.num_envs,))
                target_orientations = np.empty((morph.num_envs, 4), dtype=np.float32)
                for e in range(morph.num_envs):
                    base_quat = np.array([morph.initial_arm_orientation.x,
                                        morph.initial_arm_orientation.y,
                                        morph.initial_arm_orientation.z,
                                        morph.initial_arm_orientation.w], dtype=np.float32)
                    random_rot = quat_from_axis_angle_np(random_axes[e], random_angles[e])
                    target_orientations[e] = quat_mul_np(base_quat, random_rot)
                morph.target_ori = target_orientations
                log.debug(f"Updated target orientation (env 0, rollout {i}): {morph.target_ori[0]}")

            for j in range(config.train_iters):
                morph.step()
                morph.render()
                log.debug(f"rollout {i}, iter: {j}, error: {morph.compute_ee_error().numpy().mean()}")

        if not config.headless and morph.renderer is not None:
            morph.renderer.save()

        # Log profiling results
        log.info("Performance Profile:")
        for key, times in morph.profiler.items():
            avg_time = np.array(times).mean()
            avg_steps_second = 1000.0 * float(morph.num_envs) / avg_time if key != "target_update" else 1000.0 / avg_time
            log.info(f"  {key}: {avg_time:.3f} ms ({avg_steps_second:.2f} steps/s)")

        # save results
        results_filepath = os.path.join(config.morph_output_dir, "results.json")
        with open(results_filepath, 'w') as f:
            json.dump(results, f, indent=4)
        if config.track:
            wandb.save(results_filepath)
            wandb.finish()

    log.info(f"simulation complete!")
    log.info(f"performed {config.num_rollouts * config.train_iters} steps")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dockerfile", type=str, default=MorphConfig.dockerfile, help="Override default dockerfile variant.")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode.")
    parser.add_argument("--track", action='store_true', help="Turn on tracking with wandb.")
    parser.add_argument("--morph", type=str, default=MorphConfig.morph, help="Unique identifier for the morph.")
    parser.add_argument("--seed", type=int, default=MorphConfig.seed, help="Random seed.")
    parser.add_argument("--device", type=str, default=MorphConfig.device, help="Override default Warp device.")
    parser.add_argument("--num_envs", type=int, default=MorphConfig.num_envs, help="Number of environments to simulate.")
    parser.add_argument("--num_rollouts", type=int, default=MorphConfig.num_rollouts, help="Number of rollouts to perform.")
    parser.add_argument("--train_iters", type=int, default=MorphConfig.train_iters, help="Training iterations per rollout.")
    args = parser.parse_known_args()[0]
    config = MorphConfig(
        morph=args.morph,
        device=args.device,
        dockerfile=args.dockerfile,
        headless=args.headless,
        track=args.track,
        seed=args.seed,
        num_envs=args.num_envs,
        num_rollouts=args.num_rollouts,
        train_iters=args.train_iters,
    )
    run_morph(config)