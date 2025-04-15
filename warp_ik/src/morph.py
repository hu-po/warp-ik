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
try:
    import wandb
except ImportError:
    wandb = None
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
    backend: str = os.environ.get("BACKEND") # compute backend variant
    root_dir: str = os.environ.get("WARP_IK_ROOT") # root directory of the warp-ik project
    assets_dir: str = f"{root_dir}/assets" # assets directory for the morphs
    output_dir: str = f"{root_dir}/output" # output directory for the morphs
    morph_dir: str = f"{root_dir}/warp_ik/morphs" # directory for the morphs
    morph_output_dir: str = f"{output_dir}/{morph}" # output directory for this unique morph
    seed: int = 42 # random seed
    device: str = None # nvidia device to run the simulation on
    headless: bool = False # turns off rendering
    num_envs: int = 4 # number of parallel environments
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
    qpos_home: list[float] = field(default_factory=lambda: [
        0,          # base joint
        np.pi/12,   # shoulder joint
        np.pi/12,   # elbow joint
        0,          # wrist 1 joint
        0,          # wrist 2 joint
        0,          # wrist 3 joint
        0,          # right finger joint
        0           # left finger joint
    ]) # home position for the arm joints
    q_angle_shuffle: list[float] = field(default_factory=lambda: [
        np.pi/2,    # base joint noise range
        np.pi/4,    # shoulder joint noise range
        np.pi/4,    # elbow joint noise range
        np.pi/4,    # wrist 1 joint noise range
        np.pi/4,    # wrist 2 joint noise range
        np.pi/4,    # wrist 3 joint noise range
        0.01,       # right finger joint noise range
        0.01        # left finger joint noise range
    ]) # amount of random noise to add to the arm joint angles during initialization
    config_extras: dict = field(default_factory=dict) # additional morph-specific config values
    body_q_requires_grad: bool = True # whether to require gradients for the body positions
    joint_attach_ke: float = 1600.0 # stiffness of the joint attachment
    joint_attach_kd: float = 20.0 # damping of the joint attachment

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
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6: return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) # Return identity if axis is zero
    axis = axis / axis_norm
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)], dtype=np.float32)

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

def get_body_quaternions(state_body_q_np: np.ndarray, num_links: int, ee_link_index: int) -> np.ndarray:
    """Extracts EE quaternions from the flattened body_q numpy array."""
    num_envs = state_body_q_np.shape[0] // num_links
    # Quaternions are elements 3, 4, 5, 6 (qx, qy, qz, qw)
    ee_indices = np.arange(num_envs) * num_links + ee_link_index
    return state_body_q_np[ee_indices, 3:7]

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
    """Computes 3D orientation error vector from target and current quaternions."""
    q_err = quat_mul(target, quat_conjugate(current))
    # Ensure scalar part is non-negative for consistency
    qw = wp.select(q_err[3] < 0.0, -q_err[3], q_err[3])
    qx = wp.select(q_err[3] < 0.0, -q_err[0], q_err[0])
    qy = wp.select(q_err[3] < 0.0, -q_err[1], q_err[1])
    qz = wp.select(q_err[3] < 0.0, -q_err[2], q_err[2])
    # Return axis * angle (scaled by 2), which is approx 2 * axis * sin(angle/2)
    # Magnitude is related to the angle error
    return wp.vec3(qx, qy, qz) * 2.0


@wp.kernel
def compute_ee_error_kernel(
    body_q: wp.array(dtype=wp.transform),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    target_pos: wp.array(dtype=wp.vec3),
    target_ori: wp.array(dtype=wp.quat),
    error_out: wp.array(dtype=wp.float32)  # Flattened array (num_envs*6)
):
    tid = wp.tid()
    # Get EE link transform [px,py,pz, qx,qy,qz,qw, scale]
    t_flat = body_q[tid * num_links + ee_link_index]
    t_pos = wp.vec3(t_flat[0], t_flat[1], t_flat[2])
    t_ori = wp.quat(t_flat[3], t_flat[4], t_flat[5], t_flat[6])

    # Calculate current EE tip position
    current_pos = wp.transform_point(wp.transform(t_pos, t_ori), ee_link_offset)
    # Current EE orientation is just t_ori
    current_ori = t_ori

    # Calculate errors
    pos_err = target_pos[tid] - current_pos
    ori_err = quat_orientation_error(target_ori[tid], current_ori)

    # Write to output array
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
            # Convert fields that might not be JSON serializable (like numpy arrays in defaults)
            config_dict = {}
            for key, value in self.config.__dict__.items():
                if isinstance(value, np.ndarray):
                    config_dict[key] = value.tolist() # Convert numpy arrays to lists
                elif isinstance(value, (wp.vec3, wp.quat, wp.transform)):
                     config_dict[key] = list(value) # Convert warp types if needed
                else:
                    config_dict[key] = value
            json.dump(config_dict, f, indent=4)

        self.wandb_run = None
        if self.config.track:
            wandb.login()
            self.wandb_run = wandb.init(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project,
                name=f"{self.config.backend}/{self.config.morph}",
                config=config_dict
            )
            self.wandb_run.save(config_filepath)
            morph_code_path = os.path.join(self.config.morph_dir, f"{self.config.morph}.py")
            if os.path.exists(morph_code_path):
                    self.wandb_run.save(morph_code_path)
            log.info(f"WandB tracking enabled for run: {self.wandb_run.name}")

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
        self.num_links = len(articulation_builder.joint_type)
        self.dof = len(articulation_builder.joint_q)
        self.joint_limits = np.array(self.config.joint_limits, dtype=np.float32)
        log.info(f"Parsed URDF with {self.num_links} links and {self.dof} dof")

        # Locate ee_gripper link.
        self.ee_link_offset = wp.vec3(self.config.ee_link_offset)
        self.ee_link_index = -1
        for i, name in enumerate(articulation_builder.body_name): # Iterate through bodies to find link index
             # The joint child gives the *joint* index, not the *body* index directly connected.
             # We need the index of the body specified as the joint's child.
             # Let's find the body name associated with the joint child index.
             # This seems overly complex, let's assume the joint name convention is reliable for now.
             # TODO: Verify robust EE link index finding if URDFs change structure
             if name == "ee_gripper_link": # Assuming the link name matches the convention often used with joints
                 self.ee_link_index = i
                 break
             # Fallback using joint info (less direct)
             # joint_idx = -1
             # for j_idx, j_name in enumerate(articulation_builder.joint_name):
             #     if j_name == "ee_gripper": # Find the joint named 'ee_gripper'
             #         joint_idx = j_idx
             #         break
             # if joint_idx != -1:
             #      self.ee_link_index = articulation_builder.joint_child[joint_idx] # This is the child *joint* index, need body
                 # Need mapping from joint child index to body index...

        if self.ee_link_index == -1:
             # Let's try finding the joint first and getting its child body index
             found_joint_idx = -1
             for j_idx, j_name in enumerate(articulation_builder.joint_name):
                  if j_name == "ee_gripper":
                       found_joint_idx = j_idx
                       break
             if found_joint_idx != -1:
                  # joint_child gives the index of the child *link/body*
                  self.ee_link_index = articulation_builder.joint_child[found_joint_idx]
                  log.info(f"Found ee_gripper joint, using child link index: {self.ee_link_index} (Name: {articulation_builder.body_name[self.ee_link_index]})")
             else:
                  raise ValueError("Could not find ee_gripper joint or link in URDF")


        # Compute initial arm orientation.
        _initial_arm_orientation_np = np.array([0.,0.,0.,1.], dtype=np.float32) # Start with identity
        for axis, angle in self.config.arm_rot_offset:
            rot_quat_np = quat_from_axis_angle_np(np.array(axis), angle)
            _initial_arm_orientation_np = quat_mul_np(rot_quat_np, _initial_arm_orientation_np) # Apply rotation
        self.initial_arm_orientation = wp.quat(_initial_arm_orientation_np) # Store as Warp quat
        log.debug(f"Initial arm orientation: {list(self.initial_arm_orientation)}")


        # --- Target origin computation and initial joint angles ---
        self.target_origin = []
        self.arm_spacing_xz = self.config.arm_spacing_xz
        self.arm_height = self.config.arm_height
        self.num_rows = int(math.sqrt(self.num_envs)) if self.num_envs > 0 else 0
        if self.num_rows * self.num_rows != self.num_envs:
             log.warning(f"num_envs ({self.num_envs}) is not a perfect square. Grid layout might be uneven.")
             self.num_rows = int(np.ceil(math.sqrt(self.num_envs))) # Adjust for layout

        log.info(f"Spawning {self.num_envs} arms in a grid approximating {self.num_rows}x{self.num_rows}")

        initial_joint_q = [] # Store initial joint angles for all envs

        for e in range(self.num_envs):
            row = e // self.num_rows
            col = e % self.num_rows
            x = col * self.arm_spacing_xz
            z = row * self.arm_spacing_xz
            base_translation = np.array([x, self.arm_height, z], dtype=np.float32)
            base_quat = _initial_arm_orientation_np # Use the final computed numpy orientation

            # Define the target offset in robot coordinates and transform to world
            target_offset_local = np.array(self.config.target_offset, dtype=np.float32)
            target_world = apply_transform_np(base_translation, base_quat, target_offset_local)
            self.target_origin.append(target_world)

            # Add arm instance to the main builder
            builder.add_builder(articulation_builder, xform=wp.transform(wp.vec3(x, self.arm_height, z), self.initial_arm_orientation))

            # Set initial joint angles for this environment
            env_joint_q = []
            num_joints_in_arm = len(self.config.qpos_home)
            if num_joints_in_arm != self.dof:
                 log.warning(f"Mismatch: len(qpos_home)={num_joints_in_arm} != dof={self.dof}. Check config.")
                 # Adjust loop range defensively
                 num_joints_to_set = min(num_joints_in_arm, self.dof, len(self.config.q_angle_shuffle), len(self.joint_limits))
            else:
                 num_joints_to_set = self.dof

            for i in range(num_joints_to_set):
                value = self.config.qpos_home[i] + self.rng.uniform(-self.config.q_angle_shuffle[i], self.config.q_angle_shuffle[i])
                # Clip value based on joint limits before adding
                clipped_value = np.clip(value, self.joint_limits[i, 0], self.joint_limits[i, 1])
                env_joint_q.append(clipped_value)

            # Handle potential DOF mismatch if warning occurred
            if len(env_joint_q) < self.dof:
                 env_joint_q.extend([0.0] * (self.dof - len(env_joint_q))) # Pad with zeros

            initial_joint_q.extend(env_joint_q)

        self.target_origin = np.array(self.target_origin)
        # Initial target orientation (identity relative to base)
        self.target_ori = np.tile(_initial_arm_orientation_np, (self.num_envs, 1))
        log.debug(f"Initial target orientations (first env): {self.target_ori[0]}")

        # Finalize model.
        self.model = builder.finalize(device=self.config.device) # Specify device
        self.model.ground = False
        self.model.joint_q.assign(wp.array(initial_joint_q, dtype=wp.float32, device=self.config.device))
        self.model.body_q.requires_grad = self.config.body_q_requires_grad
        self.model.joint_attach_ke = self.config.joint_attach_ke
        self.model.joint_attach_kd = self.config.joint_attach_kd

        self.integrator = wp.sim.SemiImplicitIntegrator()

        # --- Renderer Setup ---
        self.renderer = None
        if not self.config.headless:
             try:
                 self.usd_output_path = os.path.join(self.config.morph_output_dir, "recording.usd")
                 self.renderer = wp.sim.render.SimRenderer(self.model, self.usd_output_path)
                 log.info(f"Initialized renderer, outputting to {self.usd_output_path}")
                 # Log USD path to wandb if tracking
                 if self.wandb_run:
                     self.wandb_run.summary['usd_output_path'] = self.usd_output_path
             except Exception as e:
                 log.error(f"Failed to initialize renderer: {e}. Running headless.")
                 self.config.headless = True # Force headless if renderer fails
        # --- End Renderer Setup ---

        # --- Simulation State ---
        # ee_pos is mainly used by jacobian_geom_3d, maybe initialize there?
        # For now, keep it here for potential broader use.
        self.ee_pos = wp.zeros(self.num_envs, dtype=wp.vec3, requires_grad=True, device=self.config.device)
        # ee_error stores the flattened 6D error [err_px, py, pz, ox, oy, oz] * num_envs
        self.ee_error = wp.zeros(self.num_envs * 6, dtype=wp.float32, requires_grad=True, device=self.config.device)
        # Get initial state from model
        self.state = self.model.state(requires_grad=True) # Get state AFTER setting initial joint_q
        # Target positions (world frame)
        self.targets = self.target_origin.copy() # Current target positions for each env
        # Profiler dictionary
        self.profiler = {}
        # --- End Simulation State ---


    def compute_ee_error(self) -> wp.array:
        """Computes the 6D end-effector error and returns the flattened Warp array."""
        with wp.ScopedTimer("eval_fk", print=False, active=True, dict=self.profiler, color="blue"):
            wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)
            # Note: self.state is updated in-place by eval_fk
        with wp.ScopedTimer("error_kernel", print=False, active=True, dict=self.profiler, color="green"):
            # Ensure target arrays are on the correct device
            targets_pos_wp = wp.array(self.targets, dtype=wp.vec3, device=self.config.device)
            targets_ori_wp = wp.array(self.target_ori, dtype=wp.quat, device=self.config.device)
            wp.launch(
                compute_ee_error_kernel,
                dim=self.num_envs,
                inputs=[
                    self.state.body_q, # body_q is updated by eval_fk
                    self.num_links,
                    self.ee_link_index,
                    self.ee_link_offset,
                    targets_pos_wp,
                    targets_ori_wp,
                    # Removed precomputed current_ori
                ],
                outputs=[self.ee_error], # ee_error is updated in-place
                device=self.config.device
            )
        return self.ee_error # Return the updated Warp array


    def _step(self):
        """Placeholder for the morph-specific IK update logic."""
        raise NotImplementedError("Morphs must implement their own _step function")


    def step(self):
        """
        Performs one simulation step, including error calculation and the morph-specific update.
        Also handles profiling of the morph's _step implementation.
        """
        # Profile the morph-specific implementation
        with wp.ScopedTimer("_step", print=False, active=True, dict=self.profiler, color="red"):
            self._step() # Execute the morph's core logic
        # Clipping joint angles after the step to respect limits
        current_q = self.model.joint_q.numpy()
        clipped_q = np.clip(current_q, self.joint_limits[:, 0], self.joint_limits[:, 1])
        self.model.joint_q.assign(wp.array(clipped_q, dtype=wp.float32, device=self.config.device))


    # --- render_gizmos and render_error_bars remain the same ---
    def render_gizmos(self):
        if self.renderer is None:
            return
        # Ensure FK is up-to-date for rendering the current state
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

        radius = self.config.gizmo_radius
        cone_height = self.config.gizmo_length
        cone_half_height = cone_height / 2.0

        rot_x_axis = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
        rot_y_axis = wp.quat_identity()
        rot_z_axis = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0)

        rot_x_np = np.array(list(rot_x_axis), dtype=np.float32)
        rot_y_np = np.array(list(rot_y_axis), dtype=np.float32)
        rot_z_np = np.array(list(rot_z_axis), dtype=np.float32)

        current_body_q = self.state.body_q.numpy()

        for e in range(self.num_envs):
            # --- Target Gizmos ---
            target_pos_np = self.targets[e]
            target_ori_np = self.target_ori[e]

            target_rot_x_np = quat_mul_np(target_ori_np, rot_x_np)
            target_rot_y_np = quat_mul_np(target_ori_np, rot_y_np)
            target_rot_z_np = quat_mul_np(target_ori_np, rot_z_np)

            offset_x = apply_transform_np(np.zeros(3), target_rot_x_np, np.array([0, cone_half_height, 0]))
            offset_y = apply_transform_np(np.zeros(3), target_rot_y_np, np.array([0, cone_half_height, 0]))
            offset_z = apply_transform_np(np.zeros(3), target_rot_z_np, np.array([0, cone_half_height, 0]))

            self.renderer.render_cone(name=f"target_x_{e}", pos=tuple(target_pos_np - offset_x), rot=tuple(target_rot_x_np), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_x_target)
            self.renderer.render_cone(name=f"target_y_{e}", pos=tuple(target_pos_np - offset_y), rot=tuple(target_rot_y_np), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_y_target)
            self.renderer.render_cone(name=f"target_z_{e}", pos=tuple(target_pos_np - offset_z), rot=tuple(target_rot_z_np), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_z_target)

            # --- End-Effector Gizmos ---
            if e * self.num_links + self.ee_link_index >= len(current_body_q):
                 log.error(f"Index out of bounds accessing current_body_q for env {e}. Skipping EE gizmo.")
                 continue # Skip this environment's EE gizmo if index is invalid

            ee_link_transform_flat = current_body_q[e * self.num_links + self.ee_link_index]
            ee_link_pos_np = ee_link_transform_flat[0:3]
            ee_link_ori_np = ee_link_transform_flat[3:7]
            ee_tip_pos_np = apply_transform_np(ee_link_pos_np, ee_link_ori_np, np.array(self.config.ee_link_offset)) # Use numpy offset

            ee_rot_x_np = quat_mul_np(ee_link_ori_np, rot_x_np)
            ee_rot_y_np = quat_mul_np(ee_link_ori_np, rot_y_np)
            ee_rot_z_np = quat_mul_np(ee_link_ori_np, rot_z_np)

            ee_offset_x = apply_transform_np(np.zeros(3), ee_rot_x_np, np.array([0, cone_half_height, 0]))
            ee_offset_y = apply_transform_np(np.zeros(3), ee_rot_y_np, np.array([0, cone_half_height, 0]))
            ee_offset_z = apply_transform_np(np.zeros(3), ee_rot_z_np, np.array([0, cone_half_height, 0]))

            self.renderer.render_cone(name=f"ee_pos_x_{e}", pos=tuple(ee_tip_pos_np - ee_offset_x), rot=tuple(ee_rot_x_np), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_x_ee)
            self.renderer.render_cone(name=f"ee_pos_y_{e}", pos=tuple(ee_tip_pos_np - ee_offset_y), rot=tuple(ee_rot_y_np), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_y_ee)
            self.renderer.render_cone(name=f"ee_pos_z_{e}", pos=tuple(ee_tip_pos_np - ee_offset_z), rot=tuple(ee_rot_z_np), radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_z_ee)

    def render_error_bars(self):
        # TODO: implement error bar rendering if desired
        pass

    def render(self):
        if self.renderer is None:
            return
        # Timer for rendering
        with wp.ScopedTimer("render", print=False, active=True, dict=self.profiler, color="purple"):
            self.renderer.begin_frame(self.render_time)
            # State should be up-to-date from FK calls before rendering
            self.renderer.render(self.state)
            self.render_gizmos()
            self.render_error_bars()
            self.renderer.end_frame()
        self.render_time += self.frame_dt # Increment render time *after* frame


def run_morph(config: MorphConfig) -> dict:
    """Runs the simulation for a given morph configuration."""
    morph_path = os.path.join(config.morph_dir, f"{config.morph}.py")
    log.info(f"Running morph {config.morph} from {morph_path}")

    try:
        # Dynamically import the specified morph class
        spec = importlib.util.spec_from_file_location(f"morphs.{config.morph}", morph_path)
        if spec is None:
            raise ImportError(f"Could not find spec for morph: {config.morph} at {morph_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"morphs.{config.morph}"] = module # Add to sys.modules for potential relative imports within morph
        spec.loader.exec_module(module)
        MorphClass = module.Morph
    except Exception as e:
        log.exception(f"Failed to load morph '{config.morph}': {e}")
        return {"error": f"Failed to load morph: {e}"}

    # Initialize Warp
    try:
        wp.init()
    except Exception as e:
        log.exception(f"Failed to initialize Warp: {e}")
        return {"error": f"Warp initialization failed: {e}"}

    log.info(f"Warp device: {wp.get_device().name} (CUDA: {wp.get_device().is_cuda})")

    # Instantiate the morph (handles WandB init inside)
    try:
        morph = MorphClass(config)
    except Exception as e:
        log.exception(f"Failed to initialize morph '{config.morph}': {e}")
        # Clean up wandb run if init failed after wandb.init
        if config.track and wandb and wandb.run:
            wandb.finish(exit_code=1)
        return {"error": f"Morph initialization failed: {e}"}


    # --- Simulation Loop ---
    log.info("Starting simulation loop...")
    results = {} # To store final results
    global_step = 0

    try:
        with wp.ScopedDevice(config.device):
            for i in range(config.num_rollouts):
                log.info(f"--- Starting Rollout {i+1}/{config.num_rollouts} ---")

                # --- Update Targets for New Rollout ---
                with wp.ScopedTimer("target_update", print=False, active=True, dict=morph.profiler):
                    morph.targets = morph.target_origin.copy()
                    # Add positional noise
                    pos_noise = morph.rng.uniform(-config.target_spawn_pos_noise / 2, config.target_spawn_pos_noise / 2, size=(morph.num_envs, 3))
                    morph.targets += pos_noise

                    # Add rotational noise
                    target_orientations = np.empty((morph.num_envs, 4), dtype=np.float32)
                    base_quat_np = np.array(list(morph.initial_arm_orientation), dtype=np.float32)
                    for e in range(morph.num_envs):
                        axis = morph.rng.uniform(-1, 1, size=3)
                        angle = morph.rng.uniform(-config.target_spawn_rot_noise, config.target_spawn_rot_noise)
                        random_rot_np = quat_from_axis_angle_np(axis, angle)
                        # Apply random rotation relative to the base orientation
                        target_orientations[e] = quat_mul_np(base_quat_np, random_rot_np)

                    morph.target_ori = target_orientations

                # --- Training Iterations within Rollout ---
                for j in range(config.train_iters):
                    # 1. Calculate Error *before* the step for logging
                    ee_error_flat_wp = morph.compute_ee_error()
                    ee_error_flat_np = ee_error_flat_wp.numpy() # Transfer for logging calculations
                    error_reshaped_np = ee_error_flat_np.reshape(morph.num_envs, 6)

                    # Calculate position error magnitude per env
                    pos_err_mag = np.linalg.norm(error_reshaped_np[:, 0:3], axis=1)

                    # Calculate orientation error magnitude per env
                    ori_err_mag = np.linalg.norm(error_reshaped_np[:, 3:6], axis=1)

                    # 2. Perform the actual IK step
                    morph.step() # This internally profiles _step

                    # 3. Log metrics to WandB (if enabled)
                    if morph.wandb_run:
                        log_data = {
                            "rollout": i,
                            "train_iter": j,
                            "error/pos/mean": np.mean(pos_err_mag),
                            "error/pos/min": np.min(pos_err_mag),
                            "error/pos/max": np.max(pos_err_mag),
                            "error/ori/mean": np.mean(ori_err_mag),
                            "error/ori/min": np.min(ori_err_mag),
                            "error/ori/max": np.max(ori_err_mag),
                        }
                        # Get the last _step time from the profiler's list
                        if '_step' in morph.profiler and morph.profiler['_step']:
                             log_data["perf/_step_time_ms"] = morph.profiler['_step'][-1]
                        # Log using global_step as the x-axis
                        morph.wandb_run.log(log_data, step=global_step)

                    # 4. Render the frame (optional)
                    morph.render()

                    # 5. Increment global step counter
                    global_step += 1

                log.info(f"--- Finished Rollout {i+1} ---")
                log.info(f"  Final Mean Pos Error: {np.mean(pos_err_mag):.5f}")
                log.info(f"  Final Mean Ori Error: {np.mean(ori_err_mag):.5f}")


        # --- End of Simulation ---
        log.info("Simulation loop completed.")

        # Save the USD rendering
        if not config.headless and morph.renderer is not None:
            try:
                with wp.ScopedTimer("usd_save", print=False, active=True, dict=morph.profiler):
                     morph.renderer.save()
                     log.info(f"USD recording saved to {morph.usd_output_path}")
            except Exception as e:
                log.error(f"Failed to save USD file: {e}")

        # --- Final Results & Profiling Summary ---
        log.info("--- Performance Profile Summary ---")
        total_steps = config.num_rollouts * config.train_iters
        summary_results = {}

        for key, times in morph.profiler.items():
            if not times: continue # Skip empty timers
            avg_time_ms = np.mean(times)
            std_time_ms = np.std(times)
            total_time_s = np.sum(times) / 1000.0
            log.info(f"  {key}: Avg: {avg_time_ms:.3f} ms, Std: {std_time_ms:.3f} ms, Total: {total_time_s:.3f} s")
            summary_results[f"perf/{key}_avg_ms"] = avg_time_ms
            summary_results[f"perf/{key}_std_ms"] = std_time_ms
            summary_results[f"perf/{key}_total_s"] = total_time_s

            # Calculate steps/sec specifically for _step
            if key == '_step' and avg_time_ms > 0:
                steps_per_sec = (1000.0 / avg_time_ms) * morph.num_envs
                log.info(f"    -> Approx. Steps/sec: {steps_per_sec:.2f} (env*steps/sec)")
                summary_results["perf/steps_per_sec"] = steps_per_sec

        # Store final error metrics in results
        # Use the last calculated errors before loop exit
        results["final_pos_error_mean"] = np.mean(pos_err_mag)
        results["final_pos_error_min"] = np.min(pos_err_mag)
        results["final_pos_error_max"] = np.max(pos_err_mag)
        results["final_ori_error_mean"] = np.mean(ori_err_mag)
        results["final_ori_error_min"] = np.min(ori_err_mag)
        results["final_ori_error_max"] = np.max(ori_err_mag)

        # Update wandb summary
        if morph.wandb_run:
            morph.wandb_run.summary.update(summary_results)
            morph.wandb_run.summary.update({
                 "final/pos_error_mean": results["final_pos_error_mean"],
                 "final/pos_error_min": results["final_pos_error_min"],
                 "final/pos_error_max": results["final_pos_error_max"],
                 "final/ori_error_mean": results["final_ori_error_mean"],
                 "final/ori_error_min": results["final_ori_error_min"],
                 "final/ori_error_max": results["final_ori_error_max"],
            })

        # Save results dictionary to file
        results_filepath = os.path.join(config.morph_output_dir, "results.json")
        try:
            with open(results_filepath, 'w') as f:
                 # Combine summary stats with final errors for the json file
                 results.update(summary_results)
                 json.dump(results, f, indent=4)
            log.info(f"Results saved to {results_filepath}")
            # Save to wandb as artifact
            if morph.wandb_run:
               results_artifact = wandb.Artifact(f"results_{morph.wandb_run.id}", type="results")
               results_artifact.add_file(results_filepath)
               morph.wandb_run.log_artifact(results_artifact)

        except Exception as e:
             log.error(f"Failed to save results JSON: {e}")

    except Exception as e:
        log.exception(f"Simulation loop failed: {e}")
        results["error"] = f"Simulation loop failed: {e}"
        # Ensure wandb run is finished with error status if it exists
        if config.track and wandb and wandb.run:
             wandb.finish(exit_code=1)
             morph.wandb_run = None # Prevent finishing again in finally
        return results # Return error state
    finally:
        # Ensure wandb run is finished cleanly if simulation completes or fails after init
        if config.track and morph and morph.wandb_run:
            log.info("Finishing WandB run...")
            wandb.finish()

    log.info(f"Simulation complete for morph {config.morph}!")
    log.info(f"Performed {total_steps} total training steps across {config.num_rollouts} rollouts.")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default=MorphConfig.backend, help="Override default compute backend variant.")
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
        backend=args.backend,
        headless=args.headless,
        track=args.track,
        seed=args.seed,
        num_envs=args.num_envs,
        num_rollouts=args.num_rollouts,
        train_iters=args.train_iters,
    )
    run_morph(config)