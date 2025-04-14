import warp as wp
import numpy as np

from warp_ik.src.morph import BaseMorph

class Morph(BaseMorph):
    """
    Inverse Kinematics Morph using Finite Difference Jacobian.

    This morph implements an IK solver that approximates the 6D (position + orientation)
    end-effector Jacobian using numerical finite differences. It perturbs each joint angle
    slightly, observes the change in the end-effector pose error (computed by the base class),
    and uses this information to estimate the Jacobian matrix. The joint angles are then
    updated iteratively to minimize the pose error.
    """

    def _update_config(self):
        """
        Updates the configuration specific to the Finite Difference Jacobian Morph.

        Sets the step size for joint updates, the epsilon value for finite difference
        calculations, and ensures gradients are disabled for joint angles, as this
        method computes the Jacobian numerically.
        """
        self.config.step_size = 0.5 # Step size for joint angle updates
        self.config.joint_q_requires_grad = False # Gradients not needed, Jacobian is computed numerically
        self.config.config_extras = {
            "eps": 1e-4, # Epsilon value for finite difference calculation
        }

    def _step(self):
        """
        Performs one IK step using the Finite Difference Jacobian method.

        It approximates the Jacobian matrix J by perturbing each joint angle q_i
        by +/- epsilon, computing the resulting end-effector pose error, and using
        the central difference formula: J[:, i] = (error(q + eps) - error(q - eps)) / (2 * eps).

        Once the Jacobian is approximated, it computes the change in joint angles delta_q
        using the formula: delta_q = -step_size * J^T * error.

        The computed delta_q is then added to the current joint angles.
        """
        # Note: self.compute_ee_error() is called by the base `step` method before this `_step`
        # We need the initial error *before* perturbations.
        ee_error_flat_initial = self.compute_ee_error().numpy().copy() # Get initial error

        jacobians = np.zeros((self.num_envs, 6, self.dof), dtype=np.float32)
        q0 = self.model.joint_q.numpy().copy() # Store original joint angles
        eps = self.config.config_extras["eps"]

        for e in range(self.num_envs):
            for i in range(self.dof):
                # Perturb positive
                q_plus = q0.copy()
                q_plus[e * self.dof + i] += eps
                self.model.joint_q.assign(q_plus) # Assign perturbed angles
                self.compute_ee_error() # Recompute error with perturbed angles
                f_plus = self.ee_error.numpy()[e * 6:(e + 1) * 6].copy() # Get error for this env

                # Perturb negative
                q_minus = q0.copy()
                q_minus[e * self.dof + i] -= eps
                self.model.joint_q.assign(q_minus) # Assign perturbed angles
                self.compute_ee_error() # Recompute error with perturbed angles
                f_minus = self.ee_error.numpy()[e * 6:(e + 1) * 6].copy() # Get error for this env

                # Compute column of Jacobian using central difference
                jacobians[e, :, i] = (f_plus - f_minus) / (2 * eps)

        # Restore original joint angles before applying the update
        self.model.joint_q.assign(q0)

        # Reshape initial error for calculation
        error = ee_error_flat_initial.reshape(self.num_envs, 6, 1)

        # Compute joint angle update: delta_q = -alpha * J^T * error
        # Using J^T (Jacobian transpose) method
        delta_q = -self.config.step_size * np.matmul(jacobians.transpose(0, 2, 1), error)

        # Update joint angles
        self.model.joint_q = wp.array(
            q0 + delta_q.flatten(), # Apply update to original angles
            dtype=wp.float32,
            requires_grad=self.config.joint_q_requires_grad, # Respect config setting
        )