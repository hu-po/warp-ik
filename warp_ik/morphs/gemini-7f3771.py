import warp as wp
import numpy as np
import logging as log

from warp_ik.src.morph import BaseMorph

@wp.kernel
def jacobian_transpose_multiply_kernel(
    jacobians: wp.array3d(dtype=wp.float32),  # Shape: (num_envs, 6, dof)
    error: wp.array(dtype=wp.float32),      # Shape: (num_envs * 6)
    alpha: float,                           # Step size
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
        delta_q[tid] = -alpha * sum

@wp.kernel
def add_delta_q_kernel(
    joint_q: wp.array(dtype=wp.float32),
    delta_q: wp.array(dtype=wp.float32),
    out_joint_q: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    out_joint_q[tid] = joint_q[tid] + delta_q[tid]

class Morph(BaseMorph):
    """
    Inverse Kinematics Morph using Forward Difference Jacobian (Improved Efficiency).

    This morph implements an IK solver that approximates the 6D (position + orientation)
    end-effector Jacobian using numerical *forward* differences. It perturbs each joint angle
    positively, observes the change in the end-effector pose error (computed by the base class),
    and uses this information to estimate the Jacobian matrix. The joint angles are then
    updated iteratively using the Jacobian transpose method to minimize the pose error.

    Compared to the central difference approach, this reduces the number of required
    error computations per step by half, potentially improving performance at the cost
    of slightly lower theoretical accuracy for the Jacobian approximation.
    """

    def _update_config(self):
        """
        Updates the configuration specific to the Forward Difference Jacobian Morph.

        Sets the step size for joint updates, the epsilon value for finite difference
        calculations, and ensures gradients are disabled for joint angles, as this
        method computes the Jacobian numerically.
        """
        self.config.step_size = 0.5  # Step size for joint angle updates
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {
            "eps": 1e-4,  # Epsilon value for finite difference calculation
        }

    def _step(self):
        """
        Performs one IK step using the Forward Difference Jacobian method.

        It approximates the Jacobian matrix J by perturbing each joint angle q_i
        by +epsilon, computing the resulting end-effector pose error, and using
        the forward difference formula: J[:, i] = (error(q + eps) - error(q)) / eps.

        Once the Jacobian is approximated, it computes the change in joint angles delta_q
        using the formula: delta_q = -step_size * J^T * error.

        The computed delta_q is then added to the current joint angles.
        """
        # Get initial error and joint angles
        ee_error_flat_wp = self.compute_ee_error() # Error at current q0
        ee_error_flat_initial_np = ee_error_flat_wp.numpy().copy()
        q0_np = self.model.joint_q.numpy().copy()
        eps = self.config.config_extras["eps"]

        # Initialize Jacobian on CPU (to be filled)
        jacobians_np = np.zeros((self.num_envs, 6, self.dof), dtype=np.float32)

        # Compute Jacobian using forward differences
        # Create a mutable copy for perturbations
        q_perturbed_np = q0_np.copy()
        q_perturbed_wp = wp.array(q_perturbed_np, dtype=wp.float32, device=self.config.device)

        for e in range(self.num_envs):
            # Get the initial error slice for this environment
            f0_e = ee_error_flat_initial_np[e * 6:(e + 1) * 6]

            for i in range(self.dof):
                # Perturb positive
                original_qi = q_perturbed_np[e * self.dof + i] # Store original value
                q_perturbed_np[e * self.dof + i] += eps
                q_perturbed_wp.assign(q_perturbed_np) # Update GPU array

                # Compute error at perturbed state
                f_plus_wp = self.compute_ee_error()
                f_plus_e = f_plus_wp.numpy()[e * 6:(e + 1) * 6] # Get slice for env e

                # Compute column of Jacobian using forward difference
                jacobians_np[e, :, i] = (f_plus_e - f0_e) / eps

                # Restore original joint angle in numpy array for next iteration
                q_perturbed_np[e * self.dof + i] = original_qi

        # Restore original joint angles in the model state *after* Jacobian computation
        # The model's joint_q should be q0 when calculating delta_q = -alpha * J^T * error(q0)
        self.model.joint_q.assign(wp.array(q0_np, dtype=wp.float32, device=self.config.device))

        # Transfer Jacobian to GPU
        jacobians_wp = wp.array(jacobians_np, dtype=wp.float32, device=self.config.device)

        # Initialize delta_q on GPU
        delta_q_wp = wp.zeros(self.num_envs * self.dof, dtype=wp.float32, device=self.config.device)

        # Compute joint updates using Jacobian transpose method on GPU
        # Note: Uses ee_error_flat_wp which is the error evaluated at the *start* of the step (at q0)
        wp.launch(
            jacobian_transpose_multiply_kernel,
            dim=self.num_envs * self.dof,
            inputs=[jacobians_wp, ee_error_flat_wp, self.config.step_size, self.num_envs, self.dof],
            outputs=[delta_q_wp],
            device=self.config.device
        )

        # Update joint angles on GPU
        wp.launch(
            add_delta_q_kernel,
            dim=self.num_envs * self.dof,
            inputs=[self.model.joint_q, delta_q_wp], # Input is current q0
            outputs=[self.model.joint_q], # Output is q0 + delta_q
            device=self.config.device
        )
