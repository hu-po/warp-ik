import warp as wp
import numpy as np

from warp_ik.src.morph import BaseMorph

@wp.kernel
def forward_kinematics(
    body_q: wp.array(dtype=wp.transform),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    ee_pos: wp.array(dtype=wp.vec3), # Output: end-effector positions
):
    """
    Warp kernel to compute the end-effector position using forward kinematics.

    Args:
        body_q: Array of body transforms (position and orientation) for all links in all envs.
        num_links: Number of links per robot arm.
        ee_link_index: Index of the end-effector link within an arm's links.
        ee_link_offset: Offset vector from the end-effector link's origin to the tip.
        ee_pos: Output array to store the computed end-effector world positions.
    """
    tid = wp.tid() # Thread ID, corresponds to environment index
    # Calculate the flat index for the end-effector link of the current environment
    ee_link_flat_index = tid * num_links + ee_link_index
    # Get the transform (position and orientation) of the end-effector link
    ee_link_transform = body_q[ee_link_flat_index]
    # Transform the offset vector from link space to world space and add to link position
    ee_pos[tid] = wp.transform_point(ee_link_transform, ee_link_offset)

class Morph(BaseMorph):
    """
    Inverse Kinematics Morph using Autodiff Geometric Jacobian (3D Position).

    This morph implements an IK solver that uses Warp's automatic differentiation
    feature (`wp.Tape`) to compute the Geometric Jacobian relating joint angle changes
    to the end-effector's 3D position change. It only considers the position error,
    ignoring orientation.

    The Jacobian is computed by backpropagating a unit gradient through the forward
    kinematics calculation for each (x, y, z) dimension of the end-effector position.
    Joint angles are updated iteratively using the Jacobian transpose method.
    """
    def _update_config(self):
        """
        Updates the configuration specific to the 3D Autodiff Jacobian Morph.

        Sets the step size for joint updates and ensures gradients are enabled
        for joint angles, as this method relies on automatic differentiation.
        """
        self.config.step_size = 1.0 # Step size for joint angle updates
        self.config.config_extras = {
            "joint_q_requires_grad": True, # Gradients required for autodiff
        }

    def compute_ee_position(self) -> wp.array:
        """
        Performs forward kinematics to compute the 3D end-effector position.

        Uses `wp.sim.eval_fk` to update the simulation state based on current joint angles
        and then launches the `forward_kinematics` kernel to calculate the world position
        of the end-effector tip for each environment.

        Returns:
            A Warp array containing the (x, y, z) position of the end-effector for each environment.
        """
        # Ensure simulation state (body transforms) is updated based on current joint angles
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)
        # Launch kernel to compute EE positions based on updated state
        wp.launch(
            forward_kinematics,
            dim=self.num_envs,
            inputs=[self.state.body_q, self.num_links, self.ee_link_index, self.ee_link_offset],
            outputs=[self.ee_pos], # self.ee_pos is initialized in BaseMorph
            device=self.config.device
        )
        return self.ee_pos

    def _step(self):
        """
        Performs one IK step using the 3D Autodiff Geometric Jacobian method.

        It uses `wp.Tape` to record the computation of the end-effector position.
        Then, for each dimension (x, y, z), it backpropagates a unit gradient
        to compute the corresponding row of the Jacobian (J).

        The positional error (target_pos - current_pos) is calculated.
        The change in joint angles is computed using the Jacobian transpose method:
        delta_q = step_size * J^T * error_pos.

        The computed delta_q is added to the current joint angles.
        """
        jacobians = np.empty((self.num_envs, 3, self.dof), dtype=np.float32)
        tape = wp.Tape()

        # Record computation graph for EE position
        with tape:
            ee_pos_wp = self.compute_ee_position() # Get current EE positions

        # Compute Jacobian rows via backpropagation
        for o in range(3): # Iterate through x, y, z dimensions
            # Create a gradient vector with 1.0 for the current dimension, 0.0 otherwise
            select_gradient = np.zeros(3, dtype=np.float32)
            select_gradient[o] = 1.0
            # Tile this gradient for all environments
            e_grad = wp.array(np.tile(select_gradient, self.num_envs), dtype=wp.vec3)

            # Backpropagate the gradient through the recorded tape
            tape.backward(grads={ee_pos_wp: e_grad})

            # Extract the gradient with respect to joint angles (this is one row of the Jacobian)
            q_grad_i = tape.gradients[self.model.joint_q]
            jacobians[:, o, :] = q_grad_i.numpy().reshape(self.num_envs, self.dof)

            # Reset gradients for the next dimension
            tape.zero()

        # Get current EE position *after* potential tape evaluation (if needed, depends on tape caching)
        # Recomputing ensures we have the position corresponding to the *start* of the step
        current_ee_pos_np = self.compute_ee_position().numpy()

        # Calculate 3D position error
        error_pos = self.targets - current_ee_pos_np # self.targets is from BaseMorph
        error_pos_reshaped = error_pos.reshape(self.num_envs, 3, 1)

        # Compute joint angle update using Jacobian transpose: delta_q = alpha * J^T * error
        delta_q = self.config.step_size * np.matmul(jacobians.transpose(0, 2, 1), error_pos_reshaped)

        # Update joint angles
        self.model.joint_q = wp.array(
            self.model.joint_q.numpy() + delta_q.flatten(),
            dtype=wp.float32,
            requires_grad=self.config.config_extras["joint_q_requires_grad"],
        )