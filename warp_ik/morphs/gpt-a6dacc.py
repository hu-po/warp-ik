import warp as wp
import numpy as np
import logging as log

from warp_ik.src.morph import BaseMorph

@wp.kernel
def jacobian_transpose_multiply_kernel(
    jacobians: wp.array3d(dtype=wp.float32),  # Shape: (num_envs, 6, dof)
    error: wp.array(dtype=wp.float32),         # Shape: (num_envs * 6)
    alpha: float,                              # Step size
    damping: float,                            # Damping factor to prevent overshoot
    num_envs: int,
    dof: int,
    delta_q: wp.array(dtype=wp.float32)        # Output: (num_envs * dof)
):
    tid = wp.tid()  # Thread per DOF per environment
    env_idx = tid // dof
    dof_idx = tid % dof

    if env_idx < num_envs:
        sum = 0.0
        for j in range(6):  # 6D error
            sum += jacobians[env_idx, j, dof_idx] * error[env_idx * 6 + j]
        delta_q[tid] = -alpha * damping * sum

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
    Inverse Kinematics Morph using Finite Difference Jacobian with damping.

    This morph implements an IK solver that approximates the 6D (position + orientation)
    end-effector Jacobian using numerical finite differences. In addition to the base algorithm,
    it includes an extra damping factor to moderate updates and prevent overshooting,
    as well as logging of the iteration error.
    """

    def _update_config(self):
        """
        Updates the configuration for the Finite Difference Jacobian Morph with damping.

        Sets the step size for joint updates, disables autograd for joint angles,
        and adds custom parameters: epsilon for finite differences and damping factor.
        """
        self.config.step_size = 0.5  # Base step size for joint updates
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {
            "eps": 1e-4,         # Epsilon value for finite difference calculations
            "damping": 0.8,      # Damping factor to scale updates and improve stability
            "tol": 1e-3          # Tolerance for logging convergence (6D error norm)
        }

    def _step(self):
        """
        Performs one IK step using the Finite Difference Jacobian method with damping.

        The Jacobian matrix J is approximated by perturbing each joint angle by +/- epsilon
        and using the central difference formula: J[:, i] = (error(q + eps) - error(q - eps)) / (2 * eps).

        The joint update is computed using:
            delta_q = -step_size * damping * J^T * error
        to improve numerical stability. The error norm is logged for debugging.
        """
        # Compute current end-effector error and joint angles.
        ee_error_flat_wp = self.compute_ee_error()
        ee_error_flat_initial = ee_error_flat_wp.numpy().copy()
        error_norm = np.linalg.norm(ee_error_flat_initial.reshape(self.num_envs, 6), axis=1).mean()
        log.info("IK iteration error norm: %f", error_norm)
        q0 = self.model.joint_q.numpy().copy()

        eps = self.config.config_extras["eps"]
        damping = self.config.config_extras["damping"]

        # Initialize Jacobian on GPU and CPU
        jacobians = np.zeros((self.num_envs, 6, self.dof), dtype=np.float32)

        # Compute Jacobian using finite differences.
        # Loop over environments and degrees of freedom.
        for e in range(self.num_envs):
            for i in range(self.dof):
                # Perturb positive.
                q_plus = q0.copy()
                q_plus[e * self.dof + i] += eps
                self.model.joint_q.assign(wp.array(q_plus, dtype=wp.float32, device=self.config.device))
                f_plus = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6].copy()

                # Perturb negative.
                q_minus = q0.copy()
                q_minus[e * self.dof + i] -= eps
                self.model.joint_q.assign(wp.array(q_minus, dtype=wp.float32, device=self.config.device))
                f_minus = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6].copy()

                # Compute Jacobian column using central difference.
                jacobians[e, :, i] = (f_plus - f_minus) / (2 * eps)

        # Restore original joint angles.
        self.model.joint_q.assign(wp.array(q0, dtype=wp.float32, device=self.config.device))

        # Transfer the Jacobian to GPU.
        jacobians_wp = wp.array(jacobians, dtype=wp.float32, device=self.config.device)

        # Initialize delta_q on GPU.
        delta_q_wp = wp.zeros(self.num_envs * self.dof, dtype=wp.float32, device=self.config.device)

        # Compute joint angle updates using the damped Jacobian transpose method.
        wp.launch(
            jacobian_transpose_multiply_kernel,
            dim=self.num_envs * self.dof,
            inputs=[jacobians_wp, ee_error_flat_wp, self.config.step_size, damping, self.num_envs, self.dof],
            outputs=[delta_q_wp],
            device=self.config.device
        )

        # Update joint angles on GPU.
        wp.launch(
            add_delta_q_kernel,
            dim=self.num_envs * self.dof,
            inputs=[self.model.joint_q, delta_q_wp],
            outputs=[self.model.joint_q],
            device=self.config.device
        )