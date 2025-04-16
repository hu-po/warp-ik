import warp as wp
import numpy as np
import logging as log

from warp_ik.src.morph import BaseMorph

# Kernel to copy gradient components into the full Jacobian matrix
@wp.kernel
def copy_grad_to_jacobian_kernel(
    grad_k: wp.array(dtype=wp.float32),        # Gradient for component k (num_envs * dof)
    k_idx: int,                                # The component index (0-5)
    num_envs: int,
    dof: int,
    jacobians: wp.array3d(dtype=wp.float32)    # Output Jacobian (num_envs, 6, dof)
):
    tid = wp.tid() # Thread per DOF per environment
    env_idx = tid // dof
    dof_idx = tid % dof

    if env_idx < num_envs:
        # grad_k[tid] = d(sum_env(error_k_env)) / d(q_env,dof_idx)
        # Since loss is sum over envs, gradient w.r.t q_env,dof_idx only comes from error_k_env
        # grad_k[tid] = d(error_k_env) / d(q_env,dof_idx) which is J[env_idx, k_idx, dof_idx]
        jacobians[env_idx, k_idx, dof_idx] = grad_k[tid]

# Kernel for Damped Least Squares update step
@wp.kernel
def dls_update_kernel(
    jacobians: wp.array3d(dtype=wp.float32),      # Input: (num_envs, 6, dof)
    error: wp.array(dtype=wp.float32),           # Input: (num_envs * 6)
    lambda_damping: float,                       # Input: Damping factor
    num_envs: int,
    dof: int,
    delta_q: wp.array(dtype=wp.float32)          # Output: (num_envs * dof)
):
    env_idx = wp.tid() # Thread per environment

    if env_idx >= num_envs:
        return

    # --- Manually construct matrices/vectors for this environment ---
    # Jacobian J_env (6 x dof)
    # Note: Direct slicing might be possible in future Warp versions,
    # but manual construction is robust.
    J_env_data = wp.zeros(shape=(6, dof), dtype=wp.float32)
    for i in range(6):
        for j in range(dof):
            J_env_data[i, j] = jacobians[env_idx, i, j]
    J_env = wp.mat(J_env_data) # Convert data to matrix type

    # Error vector e_env (6x1)
    e_env_data = wp.zeros(shape=(6,), dtype=wp.float32)
    for i in range(6):
        e_env_data[i] = error[env_idx*6 + i]
    e_env = wp.vec(e_env_data) # Convert data to vector type

    # --- DLS Calculation: delta_q = J^T * (J * J^T + lambda * I)^(-1) * e ---

    # 1. Compute J * J^T (6x6)
    # Ensure correct matrix multiplication order for dimensions (6 x dof) * (dof x 6) = (6 x 6)
    JJT = wp.matmul(J_env, wp.transpose(J_env))

    # 2. Add damping: JJT + lambda * I (6x6)
    I6 = wp.identity(n=6, dtype=wp.float32)
    # Add a small epsilon times identity for robustness even if lambda_damping is zero
    eps_ident = 1e-6 * I6 
    JJT_damped = JJT + lambda_damping * I6 + eps_ident

    # 3. Invert (J * J^T + lambda * I) (6x6)
    # Use wp.inverse() which computes the Moore-Penrose pseudoinverse if singular,
    # though damping should generally prevent singularity.
    JJT_damped_inv = wp.inverse(JJT_damped)

    # 4. Compute J^T * (JJT_damped_inv) (dof x 6)
    # Ensure correct matrix multiplication order: (dof x 6) * (6 x 6) = (dof x 6)
    JT_x_inv = wp.matmul(wp.transpose(J_env), JJT_damped_inv)

    # 5. Compute delta_q_env = JT_x_inv * e (dof x 1)
    # Ensure correct matrix multiplication order: (dof x 6) * (6 x 1) = (dof x 1)
    delta_q_env_vec = wp.matmul(JT_x_inv, e_env) # Result is a wp.vec of size dof

    # --- Store result ---
    # Write delta_q_env_vec (dof x 1) into the flat delta_q array
    for i in range(dof):
        delta_q[env_idx * dof + i] = delta_q_env_vec[i]


# Kernel to add delta_q (redefined for clarity, same as example)
@wp.kernel
def add_delta_q_kernel(
    joint_q: wp.array(dtype=wp.float32),    # Input: current joint angles
    delta_q: wp.array(dtype=wp.float32),  # Input: computed update step
    out_joint_q: wp.array(dtype=wp.float32) # Output: updated joint angles
):
    tid = wp.tid()
    # DLS directly computes the update step delta_q
    out_joint_q[tid] = joint_q[tid] + delta_q[tid]


class Morph(BaseMorph):
    """
    Inverse Kinematics Morph using Automatic Differentiation (AD) Jacobian
    and Damped Least Squares (DLS).

    This morph computes the exact end-effector Jacobian using Warp's
    automatic differentiation features by running backward passes on each
    error component. It then utilizes the Damped Least Squares (DLS)
    method, expressed as delta_q = J^T * (J * J^T + lambda * I)^(-1) * error,
    to compute joint angle updates. This provides potentially higher accuracy
    than finite differences and better robustness near singularities compared
    to the simple Jacobian transpose method.
    """

    def _update_config(self):
        """
        Updates the configuration specific to the AD Jacobian + DLS Morph.
        Requires joint_q gradients for AD and sets the DLS damping factor.
        """
        self.config.joint_q_requires_grad = True # Crucial for automatic differentiation
        self.config.config_extras = {
            "lambda_damping": 0.05,  # Damping factor for DLS (tune based on robot/task)
            "jacobian_method": "autodiff_dls", # Identifier for this method
        }
        # Step size is implicitly handled by the DLS formulation, not needed here.

        # Initialize buffer for the Jacobian matrix (once)
        self.jacobians_wp = None
        # Buffer for temporary gradients during Jacobian calculation
        self.grads_k_wp = None


    def _compute_ad_jacobian(self):
        """
        Computes the Jacobian d(error)/d(q) using Automatic Differentiation.

        This method iterates through each of the 6 components of the end-effector
        error vector. For each component 'k', it uses wp.Tape to compute the
        gradient of the sum of this component across all environments with respect
        to all joint angles. This gradient corresponds to the k-th row of the
        Jacobian transpose (or k-th column of the Jacobian) for each environment.
        These gradients are assembled into the full Jacobian matrix.

        Requires self.model.joint_q.requires_grad == True.
        Assumes self.compute_ee_error() is differentiable w.r.t. self.model.joint_q.
        """
        if self.jacobians_wp is None:
            # Initialize buffers on the correct device the first time
            log.info("Initializing AD Jacobian buffers...")
            self.jacobians_wp = wp.zeros((self.num_envs, 6, self.dof), dtype=wp.float32, device=self.config.device)
            self.grads_k_wp = wp.zeros_like(self.model.joint_q, device=self.config.device)

        # Perform 6 backward passes to compute the full Jacobian
        for k in range(6): # For each error component (pos x, y, z, rot x, y, z)
            tape = wp.Tape()
            with tape:
                # Recompute error inside tape context to track operations
                # This implicitly calls model.forward() which must be differentiable
                ee_error_flat_wp = self.compute_ee_error() # Shape: (num_envs * 6)

                # Create a scalar loss for AD: sum of the k-th error component across all envs
                # Advanced slicing or a dedicated kernel could optimize this sum,
                # but wp.sum(ee_error_flat_wp[k::6]) should work functionally.
                # Use a simple loop for clarity and compatibility if slicing is complex:
                loss_k = wp.constant(0.0, dtype=wp.float32)
                for env_i in range(self.num_envs):
                     loss_k = loss_k + ee_error_flat_wp[env_i * 6 + k]

            # Compute gradient: d(loss_k) / d(joint_q)
            tape.backward(loss=loss_k)

            # Retrieve the gradient tape.gradients[self.model.joint_q]
            # This gradient (shape num_envs * dof) contains the k-th column
            # of the Jacobian for each environment, stacked together.
            # grads_k_wp contains d(error_k)/d(q) for each env/dof
            self.grads_k_wp = tape.gradients[self.model.joint_q]

            # Copy this gradient column into the appropriate slice of the main Jacobian matrix
            wp.launch(
                kernel=copy_grad_to_jacobian_kernel,
                dim=self.num_envs * self.dof,
                inputs=[self.grads_k_wp, k, self.num_envs, self.dof],
                outputs=[self.jacobians_wp], # Update the Jacobian matrix in place
                device=self.config.device
            )
            # No need to zero gradients explicitly as a new tape is created each iteration.
            # tape.reset() or similar might be useful if reusing tapes.

        # self.jacobians_wp now holds the computed Jacobian (num_envs, 6, dof)


    def _step(self):
        """
        Performs one Inverse Kinematics step using the AD Jacobian + DLS method.

        1. Computes the exact Jacobian