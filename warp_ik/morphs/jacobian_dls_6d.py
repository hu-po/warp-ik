# warp_ik/morphs/jacobian_dls_6d.py
import warp as wp
import numpy as np
import time # For potential debug timing

from warp_ik.src.morph import BaseMorph

class Morph(BaseMorph):
    """
    Inverse Kinematics Morph using Damped Least Squares (DLS) Jacobian.

    This morph implements the Damped Least Squares IK method. It computes the
    full 6D Geometric Jacobian (J) using Warp's autodifferentiation (`wp.Tape`)
    similar to jacobian_geom_6d.

    Instead of the simple transpose, it uses the DLS update rule:
      delta_q = step_size * J^T * (J * J^T + lambda^2 * I)^(-1) * error_6d
    where lambda is a damping factor.

    This method is generally more robust near singularities than the pseudoinverse
    or simple transpose methods.

    Note: The matrix inversion/solve step `(J*J^T + lambda^2*I)^(-1) * error`
          is currently performed using NumPy on the CPU per step, which may
          impact performance due to GPU-CPU data transfers.
    """

    def _update_config(self):
        """
        Updates the configuration specific to the DLS Jacobian Morph.

        Sets the step size, enables gradients for autodiff, and adds the
        DLS damping factor lambda to config_extras.
        """
        self.config.step_size = 1.0  # Step size (learning rate) for joint angle updates
        self.config.joint_q_requires_grad = True # Gradients required for autodiff Jacobian
        self.config.config_extras = {
            "lambda": 0.1,  # Damping factor lambda
        }

    def _step(self):
        """
        Performs one IK step using the Damped Least Squares method.

        1. Computes the 6D pose error `e`.
        2. Computes the 6D Jacobian `J` using `wp.Tape`.
        3. Transfers `J` and `e` to NumPy arrays.
        4. Calculates the DLS update `delta_q` using NumPy linear algebra:
           - Computes `JJT = J @ J.T`
           - Forms the damped matrix `A = JJT + lambda^2 * I`
           - Solves the system `A * x = e` for `x`
           - Computes `delta_q = step_size * J.T @ x`
        5. Updates the joint angles `q = q + delta_q`.
        """
        lambda_val = self.config.config_extras["lambda"]
        lambda_sq = lambda_val * lambda_val
        step_size = self.config.step_size

        jacobians_wp = wp.zeros((self.num_envs, 6, self.dof), dtype=wp.float32, device=self.config.device)
        tape = wp.Tape()

        # Record computation graph for the 6D EE error
        with tape:
            ee_error_wp = self.compute_ee_error() # Shape (num_envs * 6)

        # Compute Jacobian rows via backpropagation
        # This part can be time-consuming due to repeated backward passes
        # start_jacobian_time = time.time()
        for o in range(6): # Iterate through 6 error dimensions
            select_gradient = np.zeros(6, dtype=np.float32)
            select_gradient[o] = 1.0
            e_grad_flat = wp.array(np.tile(select_gradient, self.num_envs), dtype=wp.float32, device=self.config.device)
            tape.backward(grads={ee_error_wp: e_grad_flat})
            q_grad_i = tape.gradients[self.model.joint_q] # Gradient for all joints, all envs

            if q_grad_i: # Check if gradient exists
                # Reshape the flat gradient array using the array's method
                q_grad_reshaped = q_grad_i.reshape((self.num_envs, self.dof))
                # Assign the reshaped gradient directly to the Jacobian slice
                jacobians_wp[:, o, :] = q_grad_reshaped
            else:
                # Handle case where gradient might be None (though unlikely here if q affects error)
                log.warning(f"Gradient computation returned None for dimension {o} in DLS step.")
                # jacobians_wp[:, o, :] remains zero from initialization
            tape.zero()
        # end_jacobian_time = time.time()
        # print(f"Jacobian computation time: {end_jacobian_time - start_jacobian_time:.4f} s")


        # Get current error and Jacobian as NumPy arrays (GPU -> CPU transfer)
        # start_transfer_time = time.time()
        jacobians_np = jacobians_wp.numpy() # Shape (num_envs, 6, dof)
        error_flat_np = self.compute_ee_error().numpy() # Recompute ensures consistency, Shape (num_envs*6)
        error_np = error_flat_np.reshape(self.num_envs, 6, 1) # Shape (num_envs, 6, 1)
        # end_transfer_time = time.time()
        # print(f"Data transfer time: {end_transfer_time - start_transfer_time:.4f} s")


        # --- Perform DLS update using NumPy ---
        # start_numpy_time = time.time()
        delta_q_np = np.zeros((self.num_envs, self.dof, 1), dtype=np.float32)
        identity_6 = np.identity(6, dtype=np.float32)

        # Process each environment individually (easier for NumPy solve)
        # A batched solve might be possible but adds complexity
        for e in range(self.num_envs):
            J = jacobians_np[e] # Shape (6, dof)
            J_T = J.T           # Shape (dof, 6)
            err = error_np[e]   # Shape (6, 1)

            JJT = J @ J_T                             # Shape (6, 6)
            A = JJT + lambda_sq * identity_6          # Shape (6, 6)

            try:
                # Solve (JJT + lambda^2*I) * x = error
                solve_result = np.linalg.solve(A, err) # Shape (6, 1)

                # Compute delta_q = J^T * solve_result
                delta_q_np[e] = J_T @ solve_result     # Shape (dof, 1)

            except np.linalg.LinAlgError:
                # Handle cases where solving fails (should be rare with damping)
                # print(f"Warning: NumPy linalg.solve failed for env {e}. Skipping update.")
                pass # Keep delta_q_np[e] as zeros

        # Scale by step size
        delta_q_np *= step_size
        # end_numpy_time = time.time()
        # print(f"NumPy computation time: {end_numpy_time - start_numpy_time:.4f} s")

        # Update joint angles (CPU -> GPU transfer)
        current_q = self.model.joint_q.numpy()
        new_q = current_q + delta_q_np.flatten() # Flatten delta_q to match joint_q shape

        self.model.joint_q = wp.array(
            new_q,
            dtype=wp.float32,
            requires_grad=self.config.joint_q_requires_grad,
            device=self.config.device
        )