import warp as wp
import numpy as np
import logging as log

from warp_ik.src.morph import BaseMorph

@wp.kernel
def jacobian_transpose_multiply_kernel(
    jacobians: wp.array3d(dtype=wp.float32),  # Shape: (num_envs, 6, dof)
    error: wp.array(dtype=wp.float32),      # Shape: (num_envs * 6)
    alpha: float,                           # Step size
    damping: float,                         # Damping factor
    num_envs: int,
    dof: int,
    delta_q: wp.array(dtype=wp.float32)     # Output: (num_envs * dof)
):
    tid = wp.tid()  # Thread per DOF per environment
    env_idx = tid // dof
    dof_idx = tid % dof
    
    if env_idx < num_envs:
        # Compute (J^T * error)_i = sum_j J_ji * error_j
        sum = 0.0
        for j in range(6):  # 6D error
            sum += jacobians[env_idx, j, dof_idx] * error[env_idx * 6 + j]
        
        # Apply damping factor to avoid large movements in singular configurations
        delta_q[tid] = -alpha * sum / (1.0 + damping * wp.abs(sum))

@wp.kernel
def add_delta_q_kernel(
    joint_q: wp.array(dtype=wp.float32),
    delta_q: wp.array(dtype=wp.float32),
    joint_limits_min: wp.array(dtype=wp.float32),
    joint_limits_max: wp.array(dtype=wp.float32),
    out_joint_q: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    # Apply delta_q with joint limit constraints
    new_q = joint_q[tid] + delta_q[tid]
    # Clamp to joint limits
    if new_q < joint_limits_min[tid]:
        new_q = joint_limits_min[tid]
    if new_q > joint_limits_max[tid]:
        new_q = joint_limits_max[tid]
    out_joint_q[tid] = new_q

class Morph(BaseMorph):
    """
    Improved Inverse Kinematics Morph using Finite Difference Jacobian.

    Enhancements:
    1. Added adaptive step size based on error magnitude
    2. Introduced damping to improve stability near singularities
    3. Added joint limit enforcement
    4. Implemented adaptive finite difference epsilon
    5. Optimized Jacobian computation with batch processing
    """

    def _update_config(self):
        """
        Updates the configuration with improved parameters for the IK solver.
        """
        self.config.step_size = 0.5  # Base step size for joint angle updates
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {
            "base_eps": 1e-4,           # Base epsilon value for finite difference
            "damping_factor": 0.1,      # Damping factor for stability near singularities
            "max_step_size": 1.0,       # Maximum step size
            "min_step_size": 0.1,       # Minimum step size
            "error_threshold": 1e-2,    # Threshold for adaptive step size adjustment
        }

    def _step(self):
        """
        Performs one improved IK step using the Finite Difference Jacobian method.
        """
        # Get initial error and joint angles
        ee_error_flat_wp = self.compute_ee_error()
        ee_error_flat_initial = ee_error_flat_wp.numpy().copy()
        q0 = self.model.joint_q.numpy().copy()
        
        # Get configuration parameters
        base_eps = self.config.config_extras["base_eps"]
        damping_factor = self.config.config_extras["damping_factor"]
        max_step_size = self.config.config_extras["max_step_size"]
        min_step_size = self.config.config_extras["min_step_size"]
        error_threshold = self.config.config_extras["error_threshold"]
        
        # Compute adaptive step size based on error magnitude
        error_norm = np.mean(np.linalg.norm(ee_error_flat_initial.reshape(self.num_envs, 6), axis=1))
        adaptive_step_size = np.clip(
            self.config.step_size * (1.0 + error_norm / error_threshold),
            min_step_size,
            max_step_size
        )
        
        # Adaptive epsilon based on joint values to handle different scales
        joint_range = np.max(np.abs(q0)) + 1e-6
        adaptive_eps = base_eps * min(1.0, joint_range / 10.0)

        # Initialize Jacobian on GPU
        jacobians_wp = wp.zeros((self.num_envs, 6, self.dof), dtype=wp.float32, device=self.config.device)
        jacobians = np.zeros((self.num_envs, 6, self.dof), dtype=np.float32)

        # Compute Jacobian using finite differences with batch processing for efficiency
        for i in range(self.dof):
            # Perturb positive for all environments at once
            q_plus = q0.copy()
            for e in range(self.num_envs):
                q_plus[e * self.dof + i] += adaptive_eps
            self.model.joint_q.assign(wp.array(q_plus, dtype=wp.float32, device=self.config.device))
            f_plus = self.compute_ee_error().numpy().reshape(self.num_envs, 6)

            # Perturb negative for all environments at once
            q_minus = q0.copy()
            for e in range(self.num_envs):
                q_minus[e * self.dof + i] -= adaptive_eps
            self.model.joint_q.assign(wp.array(q_minus, dtype=wp.float32, device=self.config.device))
            f_minus = self.compute_ee_error().numpy().reshape(self.num_envs, 6)

            # Compute column of Jacobian using central difference
            for e in range(self.num_envs):
                jacobians[e, :, i] = (f_plus[e] - f_minus[e]) / (2 * adaptive_eps)

        # Restore original joint angles
        self.model.joint_q.assign(wp.array(q0, dtype=wp.float32, device=self.config.device))

        # Transfer Jacobian to GPU
        jacobians_wp = wp.array(jacobians, dtype=wp.float32, device=self.config.device)

        # Initialize delta_q on GPU
        delta_q_wp = wp.zeros(self.num_envs * self.dof, dtype=wp.float32, device=self.config.device)

        # Compute joint updates using Jacobian transpose method with damping on GPU
        wp.launch(
            jacobian_transpose_multiply_kernel,
            dim=self.num_envs * self.dof,
            inputs=[jacobians_wp, ee_error_flat_wp, adaptive_step_size, damping_factor, self.num_envs, self.dof],
            outputs=[delta_q_wp],
            device=self.config.device
        )

        # Prepare joint limits arrays
        # Assuming the model has joint_q_limits attribute with shape (num_envs*dof, 2)
        if hasattr(self.model, 'joint_q_limits'):
            joint_limits = self.model.joint_q_limits.numpy()
            joint_limits_min = joint_limits[:, 0]
            joint_limits_max = joint_limits[:, 1]
        else:
            # Default to large limits if not available
            joint_limits_min = np.ones(self.num_envs * self.dof) * -10.0
            joint_limits_max = np.ones(self.num_envs * self.dof) * 10.0
        
        joint_limits_min_wp = wp.array(joint_limits_min, dtype=wp.float32, device=self.config.device)
        joint_limits_max_wp = wp.array(joint_limits_max, dtype=wp.float32, device=self.config.device)

        # Update joint angles with limits on GPU
        wp.launch(
            add_delta_q_kernel,
            dim=self.num_envs * self.dof,
            inputs=[self.model.joint_q, delta_q_wp, joint_limits_min_wp, joint_limits_max_wp],
            outputs=[self.model.joint_q],
            device=self.config.device
        )