# warp_ik/morphs/optim_adam_6d.py
import warp as wp
import numpy as np

from warp_ik.src.morph import BaseMorph

@wp.kernel
def adam_update_kernel(
    q: wp.array(dtype=float),
    grad_q: wp.array(dtype=float),
    m: wp.array(dtype=float),
    v: wp.array(dtype=float),
    t: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    q_out: wp.array(dtype=float),
    m_out: wp.array(dtype=float),
    v_out: wp.array(dtype=float),
):
    """Performs the Adam optimization update step element-wise for each joint."""
    tid = wp.tid()

    # Compute bias-corrected learning rate for this step
    # Avoid division by zero for t=0 (although t starts at 1 in practice)
    lr_t = learning_rate
    if t > 0.0:
         lr_t = learning_rate * wp.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)


    # Update biased first moment estimate
    m_new = beta1 * m[tid] + (1.0 - beta1) * grad_q[tid]

    # Update biased second raw moment estimate
    v_new = beta2 * v[tid] + (1.0 - beta2) * (grad_q[tid] * grad_q[tid])

    # Compute update step
    update = lr_t * m_new / (wp.sqrt(v_new) + epsilon)

    # Apply update (gradient descent)
    q_out[tid] = q[tid] - update
    m_out[tid] = m_new
    v_out[tid] = v_new


class Morph(BaseMorph):
    """
    Inverse Kinematics Morph using Adam Optimization.

    This morph frames IK as minimizing the squared 6D pose error magnitude.
    It computes the gradient of this scalar loss with respect to the joint
    angles `q` using Warp's autodifferentiation (`wp.Tape`).

    The joint angles are then updated using the Adam optimization algorithm,
    which maintains adaptive learning rates for each parameter based on
    estimates of first and second moments of the gradients.
    """

    def __init__(self, config):
        """Initializes the Adam morph, including optimizer state."""
        super().__init__(config) # Call BaseMorph init first

        # Initialize Adam state variables (persistent across steps)
        # Ensure they are on the same device as the model
        q_shape = self.model.joint_q.shape
        self.adam_m = wp.zeros(q_shape, dtype=wp.float32, device=self.config.device)
        self.adam_v = wp.zeros(q_shape, dtype=wp.float32, device=self.config.device)
        self.adam_t = 0 # Timestep counter

    def _update_config(self):
        """
        Updates the configuration specific to the Adam Optimization Morph.

        Sets the base learning rate, Adam hyperparameters (beta1, beta2, epsilon),
        and ensures gradients are enabled.
        """
        self.config.joint_q_requires_grad = True # Gradients required for loss calculation
        # Use step_size as the base learning rate, or define a new config param
        self.config.step_size = 0.01 # Adam often needs smaller learning rates
        self.config.config_extras = {
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        }

    def _step(self):
        """
        Performs one IK step using the Adam optimization algorithm.

        1. Computes the scalar loss: sum of squared 6D pose errors.
        2. Computes the gradient of the loss w.r.t. joint angles `q` using `wp.Tape`.
        3. Increments the Adam timestep `t`.
        4. Updates the Adam moment estimates `m` and `v`.
        5. Computes the Adam update step for `q`.
        6. Updates the joint angles `q` using the computed step.
        """
        beta1 = self.config.config_extras["beta1"]
        beta2 = self.config.config_extras["beta2"]
        epsilon = self.config.config_extras["epsilon"]
        learning_rate = self.config.step_size # Use step_size as base LR

        self.adam_t += 1 # Increment timestep counter

        tape = wp.Tape()
        grad_q = None # To store the gradient

        # Record computation graph for the scalar loss
        with tape:
            ee_error_wp = self.compute_ee_error() # Shape (num_envs * 6)
            # Calculate squared L2 norm (dot product with itself)
            # Need to sum this over all environments and error dimensions for a scalar loss
            loss = wp.sum(ee_error_wp * ee_error_wp)

        # Compute gradient of the total loss w.r.t. joint angles
        tape.backward(loss=loss)
        grad_q = tape.gradients[self.model.joint_q] # Shape (num_envs * dof)

        if grad_q is None:
             log.warning("Gradient computation failed in Adam step.")
             return # Skip update if gradient is None

        # Create output array for the kernel
        q_new = wp.zeros_like(self.model.joint_q)

        # Launch Adam update kernel
        wp.launch(
            kernel=adam_update_kernel,
            dim=self.model.joint_q.shape[0], # Launch one thread per joint angle
            inputs=[
                self.model.joint_q,
                grad_q,
                self.adam_m,
                self.adam_v,
                float(self.adam_t), # Pass timestep as float
                learning_rate,
                beta1,
                beta2,
                epsilon,
            ],
            outputs=[
                q_new,      # Output: updated joint angles
                self.adam_m, # Output: updated m (written in-place)
                self.adam_v, # Output: updated v (written in-place)
            ],
            device=self.config.device
        )

        # Assign the updated values back to the model state
        self.model.joint_q.assign(q_new)
        # self.adam_m and self.adam_v were updated in-place by the kernel outputs