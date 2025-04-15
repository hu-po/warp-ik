import warp as wp
import numpy as np

from warp_ik.src.morph import BaseMorph

class Morph(BaseMorph):
    """
    Inverse Kinematics Morph using Autodiff Geometric Jacobian (6D Pose).

    This morph implements an IK solver that uses Warp's automatic differentiation
    feature (`wp.Tape`) to compute the full 6D Geometric Jacobian relating joint
    angle changes to the end-effector's pose (position and orientation) change.

    The Jacobian is computed by backpropagating a unit gradient through the 6D pose
    error calculation (handled by `compute_ee_error` in the base class) for each
    of the 6 error dimensions (3 position, 3 orientation).

    Joint angles are updated iteratively using the Jacobian transpose method:
    delta_q = -step_size * J^T * error_6d.
    """

    def _update_config(self):
        """
        Updates the configuration specific to the 6D Autodiff Jacobian Morph.

        Sets the step size for joint updates and ensures gradients are enabled
        for joint angles, as this method relies on automatic differentiation.
        """
        self.config.step_size = 1.0 # Step size for joint angle updates
        self.config.config_extras = {
            "joint_q_requires_grad": True, # Gradients required for autodiff
        }

    def _step(self):
        """
        Performs one IK step using the 6D Autodiff Geometric Jacobian method.

        It uses `wp.Tape` to record the computation of the 6D end-effector pose error
        (position and orientation), which is calculated by `self.compute_ee_error()`.
        Then, for each of the 6 error dimensions, it backpropagates a unit gradient
        to compute the corresponding row of the Jacobian (J).

        The change in joint angles is computed using the Jacobian transpose method:
        delta_q = -step_size * J^T * error_6d.

        The computed delta_q is added to the current joint angles.
        """
        jacobians = np.empty((self.num_envs, 6, self.dof), dtype=np.float32)
        tape = wp.Tape()

        # Record computation graph for the 6D EE error
        with tape:
            # self.ee_error is computed and stored by this call
            ee_error_wp = self.compute_ee_error()

        # Compute Jacobian rows via backpropagation
        for o in range(6): # Iterate through 6 error dimensions (px, py, pz, ox, oy, oz)
            # Create a gradient vector with 1.0 for the current dimension, 0.0 otherwise
            select_gradient = np.zeros(6, dtype=np.float32)
            select_gradient[o] = 1.0
            # Tile this gradient for all environments (flattened error array)
            e_grad_flat = wp.array(np.tile(select_gradient, self.num_envs), dtype=wp.float32)

            # Backpropagate the gradient through the recorded tape
            # The gradient source is the flattened ee_error array
            tape.backward(grads={ee_error_wp: e_grad_flat})

            # Extract the gradient with respect to joint angles (this is one row of the Jacobian)
            q_grad_i = tape.gradients[self.model.joint_q]
            jacobians[:, o, :] = q_grad_i.numpy().reshape(self.num_envs, self.dof)

            # Reset gradients for the next dimension
            tape.zero()

        # Get the current 6D error *after* tape evaluation (recompute for consistency)
        # This ensures we use the error corresponding to the joint angles at the *start* of the step
        ee_error_flat_np = self.compute_ee_error().numpy()
        error_6d_reshaped = ee_error_flat_np.reshape(self.num_envs, 6, 1)

        # Compute joint angle update using Jacobian transpose: delta_q = -alpha * J^T * error
        delta_q = -self.config.step_size * np.matmul(jacobians.transpose(0, 2, 1), error_6d_reshaped)

        # Update joint angles
        self.model.joint_q = wp.array(
            self.model.joint_q.numpy() + delta_q.flatten(),
            dtype=wp.float32,
            requires_grad=self.config.config_extras["joint_q_requires_grad"],
        )