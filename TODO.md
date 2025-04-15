Here is a detailed implementation plan to optimize the script by moving CPU-bound operations to the GPU using Warp:

Goal: Reduce the single-core CPU bottleneck observed during simulation by migrating NumPy operations and Python loops (especially those iterating per-environment or per-step) into parallel Warp kernels executed on the GPU.

Phase 1: Preparation and Data Conversion (BaseMorph.__init__)

Convert Joint Limits to Warp Array:
In BaseMorph.__init__, after self.joint_limits = np.array(...), create corresponding Warp arrays.
Add:
Python

# Ensure joint_limits is correctly shaped (num_joints, 2)
joint_limits_np = self.joint_limits # Keep numpy version if needed elsewhere
self.joint_limits_min_wp = wp.array(joint_limits_np[:, 0], dtype=wp.float32, device=self.config.device)
self.joint_limits_max_wp = wp.array(joint_limits_np[:, 1], dtype=wp.float32, device=self.config.device)
# We might need tiled versions later for the kernel, depending on kernel implementation
# Or pass the self.dof and calculate index inside kernel
Convert Target/State Data to Warp Arrays:
self.target_origin: Keep the NumPy version (self.target_origin_np) calculated during init. Create a persistent Warp array for runtime targets.
self.target_ori: Similarly, keep the initial NumPy version (self.initial_arm_orientation_np). Create a persistent Warp array.
Modify the initialization of self.targets and self.target_ori at the end of __init__ (after the loop):
Python

# Existing calculation of self.target_origin (list of np arrays)
self.target_origin_np = np.array(self.target_origin, dtype=np.float32)
# Existing calculation of _initial_arm_orientation_np

# --- Simulation State ---
# ... (ee_pos, ee_error setup) ...
self.state = self.model.state(requires_grad=True)

# *** NEW: Initialize runtime target arrays on GPU ***
self.targets_wp = wp.array(self.target_origin_np, dtype=wp.vec3, device=self.config.device, requires_grad=False) # Runtime targets
# Use the initial orientation computed earlier (_initial_arm_orientation_np)
initial_target_ori_np = np.tile(_initial_arm_orientation_np, (self.num_envs, 1))
self.target_ori_wp = wp.array(initial_target_ori_np, dtype=wp.quat, device=self.config.device, requires_grad=False) # Runtime target orientations

self.profiler = {}
# --- End Simulation State ---
Replace self.targets = self.target_origin.copy() and morph.target_ori = target_orientations in the run_morph target update block later. The kernel will update self.targets_wp and self.target_ori_wp.
Convert Config Values:
Ensure self.ee_link_offset is stored as wp.vec3 if used directly in kernels (it's already passed to compute_ee_error_kernel).
Phase 2: GPU Joint Clipping (BaseMorph.step)

Define Clipping Kernel: Add this kernel definition somewhere accessible, e.g., near other kernels.
Python

@wp.kernel
def clip_joints_kernel(
    joint_q: wp.array(dtype=wp.float32), # Input/Output array
    joint_limits_min: wp.array(dtype=wp.float32), # Min limits per DOF type
    joint_limits_max: wp.array(dtype=wp.float32), # Max limits per DOF type
    num_envs: int,
    dof: int
):
    tid = wp.tid() # Unique index for each joint DOF across all envs

    # Calculate which DOF this thread corresponds to
    joint_idx = tid % dof

    # Clamp the value
    current_val = joint_q[tid]
    min_val = joint_limits_min[joint_idx]
    max_val = joint_limits_max[joint_idx]
    joint_q[tid] = wp.clamp(current_val, min_val, max_val)
Modify BaseMorph.step: Replace the NumPy clipping block.
Python

def step(self):
    # Profile the morph-specific implementation
    with wp.ScopedTimer("_step", print=False, active=True, dict=self.profiler, color="red"):
        self._step() # Execute the morph's core logic

    # --- REMOVE THIS BLOCK ---
    # current_q = self.model.joint_q.numpy()
    # min_limits = np.tile(self.joint_limits[:, 0], self.num_envs)
    # max_limits = np.tile(self.joint_limits[:, 1], self.num_envs)
    # clipped_q = np.clip(current_q, min_limits, max_limits)
    # self.model.joint_q.assign(wp.array(clipped_q, dtype=wp.float32, device=self.config.device))
    # --- END REMOVE ---

    # +++ ADD THIS BLOCK +++
    # Clipping joint angles after the step to respect limits (on GPU)
    with wp.ScopedTimer("clip_joints", print=False, active=True, dict=self.profiler): # Optional profiling
        wp.launch(
            kernel=clip_joints_kernel,
            dim=self.num_envs * self.dof, # Launch one thread per DOF instance
            inputs=[
                self.model.joint_q,
                self.joint_limits_min_wp,
                self.joint_limits_max_wp,
                self.num_envs,
                self.dof
            ],
            outputs=[self.model.joint_q], # Indicate joint_q is modified
            device=self.config.device
        )
    # +++ END ADD +++
Phase 3: GPU Target Update (run_morph)

Define Target Update Kernel: Add this kernel definition.
Python

@wp.kernel
def update_targets_kernel(
    target_origins: wp.array(dtype=wp.vec3),
    initial_base_orientation: wp.quat, # The common base rotation
    pos_noise_scale: float,
    rot_noise_scale: float,
    seed: int, # Seed for RNG state
    # Outputs:
    out_target_pos: wp.array(dtype=wp.vec3),
    out_target_ori: wp.array(dtype=wp.quat)
):
    tid = wp.tid() # Environment index

    # Initialize random state for this thread
    state = wp.rand_init(seed, tid)

    # Positional Noise
    noise_x = wp.randf(state, -0.5, 0.5) * pos_noise_scale
    noise_y = wp.randf(state, -0.5, 0.5) * pos_noise_scale
    noise_z = wp.randf(state, -0.5, 0.5) * pos_noise_scale
    pos_noise_vec = wp.vec3(noise_x, noise_y, noise_z)

    out_target_pos[tid] = target_origins[tid] + pos_noise_vec

    # Rotational Noise (Axis-Angle)
    axis_x = wp.randf(state, -1.0, 1.0)
    axis_y = wp.randf(state, -1.0, 1.0)
    axis_z = wp.randf(state, -1.0, 1.0)
    noise_axis = wp.normalize(wp.vec3(axis_x, axis_y, axis_z))
    noise_angle = wp.randf(state, -rot_noise_scale, rot_noise_scale)

    random_rot = wp.quat_from_axis_angle(noise_axis, noise_angle)

    # Apply random rotation relative to the initial base orientation
    out_target_ori[tid] = wp.quat_mul(initial_base_orientation, random_rot)

Modify run_morph Target Update Block: Replace the NumPy block inside the for i in range(config.num_rollouts): loop.
Python

# --- Update Targets for New Rollout ---
with wp.ScopedTimer("target_update", print=False, active=True, dict=morph.profiler):
    # --- REMOVE THIS BLOCK ---
    # morph.targets = morph.target_origin.copy()
    # pos_noise = morph.rng.uniform(-config.target_spawn_pos_noise / 2, config.target_spawn_pos_noise / 2, size=(morph.num_envs, 3))
    # morph.targets += pos_noise
    # target_orientations = np.empty((morph.num_envs, 4), dtype=np.float32)
    # base_quat_np = np.array(list(morph.initial_arm_orientation), dtype=np.float32)
    # for e in range(morph.num_envs):
    #     axis = morph.rng.uniform(-1, 1, size=3)
    #     angle = morph.rng.uniform(-config.target_spawn_rot_noise, config.target_spawn_rot_noise)
    #     random_rot_np = quat_from_axis_angle_np(axis, angle)
    #     target_orientations[e] = quat_mul_np(base_quat_np, random_rot_np)
    # morph.target_ori = target_orientations
    # --- END REMOVE ---

    # +++ ADD THIS BLOCK +++
    # Update targets directly on GPU
    wp.launch(
        kernel=update_targets_kernel,
        dim=morph.num_envs,
        inputs=[
            wp.array(morph.target_origin_np, dtype=wp.vec3, device=morph.config.device), # Pass base origins
            morph.initial_arm_orientation, # Pass base orientation
            config.target_spawn_pos_noise,
            config.target_spawn_rot_noise,
            config.seed + i # Simple way to vary seed per rollout
        ],
        outputs=[
            morph.targets_wp, # Update the persistent Warp array
            morph.target_ori_wp # Update the persistent Warp array
        ],
        device=morph.config.device
    )
    # Ensure targets_pos_wp and targets_ori_wp used in compute_ee_error_kernel now read from morph.targets_wp and morph.target_ori_wp
    # Modify compute_ee_error call if necessary:
    # targets_pos_wp = morph.targets_wp
    # targets_ori_wp = morph.target_ori_wp
    # (No need to create wp.array from numpy inside compute_ee_error anymore)

    # Update compute_ee_error to accept these directly
    # In BaseMorph.compute_ee_error: REMOVE the lines creating targets_pos_wp/targets_ori_wp
    # and ensure the kernel launch uses self.targets_wp and self.target_ori_wp
    # Example change in compute_ee_error:
    # def compute_ee_error(self) -> wp.array:
    #     # ... eval_fk ...
    #     with wp.ScopedTimer(...):
    #         # REMOVED: targets_pos_wp = wp.array(...)
    #         # REMOVED: targets_ori_wp = wp.array(...)
    #         wp.launch(
    #             compute_ee_error_kernel,
    #             dim=self.num_envs,
    #             inputs=[..., self.targets_wp, self.target_ori_wp], # Use existing wp arrays
    #             outputs=[self.ee_error],
    #             device=self.config.device
    #         )
    #     return self.ee_error

    # +++ END ADD +++

Phase 4: GPU Error Magnitude Calculation & Logging (run_morph)

Prepare Output Arrays: In BaseMorph.__init__, add arrays to store magnitudes.
Python

# --- Simulation State ---
# ... ee_pos, ee_error ...
self.ee_pos_err_mag_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.config.device)
self.ee_ori_err_mag_wp = wp.zeros(self.num_envs, dtype=wp.float32, device=self.config.device)
# ... state, targets_wp, target_ori_wp ...
Define Magnitude Kernel:
Python

@wp.kernel
def calculate_error_magnitudes_kernel(
    flat_errors: wp.array(dtype=float), # Input: num_envs * 6
    num_envs: int,
    # Outputs:
    out_pos_mag: wp.array(dtype=float),
    out_ori_mag: wp.array(dtype=float)
):
    tid = wp.tid() # Environment index
    base = tid * 6

    # Extract pos and ori error vectors
    pos_err = wp.vec3(flat_errors[base + 0], flat_errors[base + 1], flat_errors[base + 2])
    ori_err = wp.vec3(flat_errors[base + 3], flat_errors[base + 4], flat_errors[base + 5])

    # Calculate and store magnitudes
    out_pos_mag[tid] = wp.length(pos_err)
    out_ori_mag[tid] = wp.length(ori_err)
Modify Logging Block in run_morph:
Python

# Inside the for j in range(config.train_iters): loop

# 1. Calculate Error *before* the step for logging
ee_error_flat_wp = morph.compute_ee_error() # This already calls the kernel

# +++ ADD: Calculate magnitudes on GPU +++
with wp.ScopedTimer("error_mag_kernel", print=False, active=True, dict=morph.profiler):
    wp.launch(
        kernel=calculate_error_magnitudes_kernel,
        dim=morph.num_envs,
        inputs=[ee_error_flat_wp, morph.num_envs],
        outputs=[morph.ee_pos_err_mag_wp, morph.ee_ori_err_mag_wp],
        device=morph.config.device
    )

# --- REMOVE / REPLACE ---
# ee_error_flat_np = ee_error_flat_wp.numpy() # Transfer for logging calculations
# error_reshaped_np = ee_error_flat_np.reshape(morph.num_envs, 6)
# pos_err_mag = np.linalg.norm(error_reshaped_np[:, 0:3], axis=1)
# ori_err_mag = np.linalg.norm(error_reshaped_np[:, 3:6], axis=1)
# --- END REMOVE ---

# 2. Perform the actual IK step
morph.step() # This internally profiles _step

# 3. Log metrics to WandB (if enabled)
if morph.wandb_run:
    # *** Option A: Transfer full arrays (Simpler, more transfer) ***
    pos_err_mag_np = morph.ee_pos_err_mag_wp.numpy()
    ori_err_mag_np = morph.ee_ori_err_mag_wp.numpy()
    log_data = {
        # ... rollout, train_iter ...
        "error/pos/mean": np.mean(pos_err_mag_np),
        "error/pos/min": np.min(pos_err_mag_np),
        "error/pos/max": np.max(pos_err_mag_np),
        "error/ori/mean": np.mean(ori_err_mag_np),
        "error/ori/min": np.min(ori_err_mag_np),
        "error/ori/max": np.max(ori_err_mag_np),
        # ... perf data ...
    }
    # *** End Option A ***

    # *** Option B: GPU Reduction (More complex, less transfer - REQUIRES ATOMICS KERNEL) ***
    # Requires another kernel using wp.atomic_min/max/add to compute reductions
    # reduced_vals_wp = wp.zeros(6, dtype=float, device=morph.config.device) # [pos_min, pos_max, pos_sum, ori_min, ori_max, ori_sum]
    # wp.launch(reduction_kernel, ..., outputs=[reduced_vals_wp])
    # reduced_vals_np = reduced_vals_wp.numpy()
    # log_data = { ...
    #    "error/pos/mean": reduced_vals_np[2] / morph.num_envs,
    #    "error/pos/min": reduced_vals_np[0], ... etc ... }
    # For now, implement Option A. Option B is an advanced optimization.

    morph.wandb_run.log(log_data, step=global_step)

# ... rest of loop ...
Also update the final logging after the loop to use the last computed _np versions if using Option A.
Phase 5: Optimize Rendering (BaseMorph.render_gizmos)

Condition: Only apply if self.renderer is not None (or not self.config.headless).
Approach (Option 2 - GPU Calc, CPU Loop):
Define Transform Kernel:
Python

@wp.kernel
def calculate_gizmo_transforms_kernel(
    body_q: wp.array(dtype=wp.transform),
    targets_pos: wp.array(dtype=wp.vec3),
    targets_ori: wp.array(dtype=wp.quat),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    num_envs: int,
    rot_x_axis_q: wp.quat, # Precomputed axis rotations
    rot_y_axis_q: wp.quat,
    rot_z_axis_q: wp.quat,
    cone_half_height: float,
    # Outputs (flat array, order: TgtX,TgtY,TgtZ, EEX,EEY,EEZ per env)
    out_gizmo_pos: wp.array(dtype=wp.vec3),
    out_gizmo_rot: wp.array(dtype=wp.quat)
):
    tid = wp.tid() # Environment index

    # --- Target Gizmos ---
    target_pos = targets_pos[tid]
    target_ori = targets_ori[tid]

    target_rot_x = wp.quat_mul(target_ori, rot_x_axis_q)
    target_rot_y = wp.quat_mul(target_ori, rot_y_axis_q)
    target_rot_z = wp.quat_mul(target_ori, rot_z_axis_q)

    offset_vec = wp.vec3(0.0, cone_half_height, 0.0)
    offset_x = wp.quat_rotate(target_rot_x, offset_vec)
    offset_y = wp.quat_rotate(target_rot_y, offset_vec)
    offset_z = wp.quat_rotate(target_rot_z, offset_vec)

    base_idx = tid * 6
    out_gizmo_pos[base_idx + 0] = target_pos - offset_x
    out_gizmo_rot[base_idx + 0] = target_rot_x
    out_gizmo_pos[base_idx + 1] = target_pos - offset_y
    out_gizmo_rot[base_idx + 1] = target_rot_y
    out_gizmo_pos[base_idx + 2] = target_pos - offset_z
    out_gizmo_rot[base_idx + 2] = target_rot_z

    # --- End-Effector Gizmos ---
    t_flat = body_q[tid * num_links + ee_link_index]
    ee_link_pos = wp.vec3(t_flat[0], t_flat[1], t_flat[2])
    ee_link_ori = wp.quat(t_flat[3], t_flat[4], t_flat[5], t_flat[6])
    ee_tip_pos = wp.transform_point(wp.transform(ee_link_pos, ee_link_ori), ee_link_offset)

    ee_rot_x = wp.quat_mul(ee_link_ori, rot_x_axis_q)
    ee_rot_y = wp.quat_mul(ee_link_ori, rot_y_axis_q)
    ee_rot_z = wp.quat_mul(ee_link_ori, rot_z_axis_q)

    ee_offset_x = wp.quat_rotate(ee_rot_x, offset_vec)
    ee_offset_y = wp.quat_rotate(ee_rot_y, offset_vec)
    ee_offset_z = wp.quat_rotate(ee_rot_z, offset_vec)

    out_gizmo_pos[base_idx + 3] = ee_tip_pos - ee_offset_x
    out_gizmo_rot[base_idx + 3] = ee_rot_x
    out_gizmo_pos[base_idx + 4] = ee_tip_pos - ee_offset_y
    out_gizmo_rot[base_idx + 4] = ee_rot_y
    out_gizmo_pos[base_idx + 5] = ee_tip_pos - ee_offset_z
    out_gizmo_rot[base_idx + 5] = ee_rot_z

Prepare Kernel Inputs in __init__:
Python

# In __init__
self.rot_x_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
self.rot_y_axis_q_wp = wp.quat_identity()
self.rot_z_axis_q_wp = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi / 2.0)
# Add arrays to hold calculated transforms
self.gizmo_pos_wp = wp.zeros(self.num_envs * 6, dtype=wp.vec3, device=self.config.device)
self.gizmo_rot_wp = wp.zeros(self.num_envs * 6, dtype=wp.quat, device=self.config.device)
Modify render_gizmos:
Python

def render_gizmos(self):
    if self.renderer is None: return

    # Ensure FK is up-to-date for rendering the current state
    wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

    radius = self.config.gizmo_radius
    cone_height = self.config.gizmo_length
    cone_half_height = cone_height / 2.0

    # +++ Launch Kernel to calculate all transforms +++
    with wp.ScopedTimer("gizmo_kernel", print=False, active=True, dict=self.profiler):
        wp.launch(
            kernel=calculate_gizmo_transforms_kernel,
            dim=self.num_envs,
            inputs=[
                self.state.body_q, self.targets_wp, self.target_ori_wp, # Use Warp arrays
                self.num_links, self.ee_link_index, self.ee_link_offset, self.num_envs,
                self.rot_x_axis_q_wp, self.rot_y_axis_q_wp, self.rot_z_axis_q_wp,
                cone_half_height
            ],
            outputs=[self.gizmo_pos_wp, self.gizmo_rot_wp],
            device=self.config.device
        )

    # +++ Transfer results to CPU +++
    gizmo_pos_np = self.gizmo_pos_wp.numpy()
    gizmo_rot_np = self.gizmo_rot_wp.numpy()

    # --- REMOVE NumPy axis rotations ---
    # rot_x_np = ...
    # rot_y_np = ...
    # rot_z_np = ...
    # --- REMOVE Transfer ---
    # current_body_q = self.state.body_q.numpy()

    # --- Modify CPU Loop ---
    for e in range(self.num_envs):
        base_idx = e * 6
        # Target Gizmos (Read precomputed values)
        tgt_pos_x, tgt_rot_x = tuple(gizmo_pos_np[base_idx + 0]), tuple(gizmo_rot_np[base_idx + 0])
        tgt_pos_y, tgt_rot_y = tuple(gizmo_pos_np[base_idx + 1]), tuple(gizmo_rot_np[base_idx + 1])
        tgt_pos_z, tgt_rot_z = tuple(gizmo_pos_np[base_idx + 2]), tuple(gizmo_rot_np[base_idx + 2])

        # --- REMOVE NumPy calculations inside loop ---
        # target_pos_np = self.targets[e]
        # ... all quat_mul_np, apply_transform_np ...

        self.renderer.render_cone(name=f"target_x_{e}", pos=tgt_pos_x, rot=tgt_rot_x, radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_x_target)
        self.renderer.render_cone(name=f"target_y_{e}", pos=tgt_pos_y, rot=tgt_rot_y, radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_y_target)
        self.renderer.render_cone(name=f"target_z_{e}", pos=tgt_pos_z, rot=tgt_rot_z, radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_z_target)

        # End-Effector Gizmos (Read precomputed values)
        ee_pos_x, ee_rot_x = tuple(gizmo_pos_np[base_idx + 3]), tuple(gizmo_rot_np[base_idx + 3])
        ee_pos_y, ee_rot_y = tuple(gizmo_pos_np[base_idx + 4]), tuple(gizmo_rot_np[base_idx + 4])
        ee_pos_z, ee_rot_z = tuple(gizmo_pos_np[base_idx + 5]), tuple(gizmo_rot_np[base_idx + 5])

        # --- REMOVE NumPy calculations inside loop ---
        # ee_link_transform_flat = ...
        # ... all quat_mul_np, apply_transform_np ...

        self.renderer.render_cone(name=f"ee_pos_x_{e}", pos=ee_pos_x, rot=ee_rot_x, radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_x_ee)
        self.renderer.render_cone(name=f"ee_pos_y_{e}", pos=ee_pos_y, rot=ee_rot_y, radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_y_ee)
        self.renderer.render_cone(name=f"ee_pos_z_{e}", pos=ee_pos_z, rot=ee_rot_z, radius=radius, half_height=cone_half_height, color=self.config.gizmo_color_z_ee)
Phase 6: Cleanup

Remove Unused NumPy Helpers: Delete quat_to_rot_matrix_np, quat_from_axis_angle_np, apply_transform_np, quat_mul_np.
Remove get_body_quaternions: This function doesn't appear to be called anywhere and is no longer needed.