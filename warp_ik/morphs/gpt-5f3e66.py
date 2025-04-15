import time
import math
import numpy as np
import warp as wp
from warp_ik.src.morph import BaseMorph

# A helper Warp kernel that computes a dummy forward kinematics and error.
# This kernel is designed to simulate a 6D end‐effector pose from the joint angles.
# The “forward kinematics” here is a contrived function for demonstration purposes.
@wp.kernel
def compute_fk(q: wp.array(dtype=wp.float32), 
               error: wp.array(dtype=wp.float32),
               target: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    # We only run on one thread since the FK computation is small.
    if tid == 0:
        pos0 = 0.0
        pos1 = 0.0
        pos2 = 0.0
        ori0 = 0.0
        ori1 = 0.0
        ori2 = 0.0
        n = q.shape[0]
        for i in range(n):
            qi = q[i]
            pos0 += qi
            pos1 += qi * wp.cos(qi)
            pos2 += qi * wp.sin(qi)
            ori0 += wp.sin(qi)
            ori1 += wp.cos(qi)
            ori2 += qi
        # Compute 6D error: first 3 for position and next 3 for orientation.
        error[0] = pos0 - target[0]
        error[1] = pos1 - target[1]
        error[2] = pos2 - target[2]
        error[3] = ori0 - target[3]
        error[4] = ori1 - target[4]
        error[5] = ori2 - target[5]

class Morph(BaseMorph):
    """
    Morph class implementing a novel IK algorithm that uses an adaptive momentum‐based
    gradient descent with a simple line search mechanism to minimize a 6D end-effector pose error.
    
    The algorithm computes a normalized gradient via automatic differentiation using Warp’s Tape,
    then applies a momentum-integrated update. An adaptive step size is adjusted based on whether
    the cost decreases or increases, effectively performing a simple line search without the typical
    Jacobian‐based updates.
    """

    def _update_config(self):
        # Use a moderately large number of iterations.
        self.config.train_iters = 100

        # Enable gradient on the joint configuration.
        self.config.joint_q_requires_grad = True

        # Config extras: step size for the update, damping factor for reducing step size 
        # when the cost increases, momentum factor for smoothing updates, and the target 6D pose.
        self.config.config_extras = {
            "step_size": 0.1,      # initial update step size
            "damping": 0.5,        # reduction factor if cost goes up (line-search like behavior)
            "momentum": 0.9,       # momentum factor to smooth updates
            "step_increase": 1.05, # factor to slightly increase step size if cost decreases
            # target_pose: [pos0, pos1, pos2, ori0, ori1, ori2]
            "target_pose": wp.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=wp.float32)
        }

        # Initialize variables to store previous cost and momentum update.
        self.prev_cost = None
        self.prev_dq = None

    def _step(self):
        # Allocate memory for the error vector (6D error).
        error = wp.empty(shape=(6,), dtype=wp.float32, device=self.device)
        target = self.config.config_extras["target_pose"]

        # Use Warp's Tape for automatic differentiation.
        with wp.Tape() as tape:
            # Compute forward kinematics error using our custom kernel.
            wp.launch(kernel=compute_fk, dim=1, inputs=[self.joint_q, error, target])
            # Compute scalar cost as the sum of squared errors.
            cost = 0.0
            for i in range(6):
                cost += error[i] * error[i]

        # Retrieve the gradient of the cost with respect to joint angles.
        # (The gradient here is 2 * J^T * error.)
        grad_q = tape.gradients[self.joint_q]

        # Convert the gradient from a Warp array to a NumPy array for easy norm computation.
        grad_np = wp.to_numpy(grad_q)
        grad_norm = np.linalg.norm(grad_np)

        # If the gradient is almost zero, no update is needed.
        if grad_norm < 1e-6:
            return

        # Normalize the gradient.
        grad_normalized = grad_np / grad_norm

        # Retrieve the current step size and momentum from configuration.
        step_size = self.config.config_extras["step_size"]
        momentum  = self.config.config_extras["momentum"]

        # Compute the momentum-based update.
        # If a previous update exists, blend it in using the momentum factor.
        if self.prev_dq is None:
            dq_np = -step_size * grad_normalized
        else:
            dq_np = momentum * self.prev_dq + (1 - momentum) * (-step_size * grad_normalized)

        # Create a Warp array for the computed update.
        dq = wp.array(dq_np, dtype=wp.float32, device=self.device)

        # Apply the update tentatively.
        new_joint_q = wp.clone(self.joint_q)
        # Assuming self.joint_q is a 1D Warp array, update each element.
        n = self.joint_q.shape[0]
        for i in range(n):
            new_joint_q[i] = self.joint_q[i] + dq[i]

        # Evaluate the new cost at updated joint angles.
        new_error = wp.empty(shape=(6,), dtype=wp.float32, device=self.device)
        wp.launch(kernel=compute_fk, dim=1, inputs=[new_joint_q, new_error, target])
        new_cost = 0.0
        new_error_np = wp.to_numpy(new_error)
        for v in new_error_np:
            new_cost += v*v

        # Adjust the step size based on the cost improvement.
        if self.prev_cost is not None and new_cost > self.prev_cost:
            # If cost increased, reduce step size.
            step_size *= self.config.config_extras["damping"]
        else:
            # Cost decreased; gently increase step size.
            step_size *= self.config.config_extras["step_increase"]

        # Update the stored step size.
        self.config.config_extras["step_size"] = step_size

        # Accept the update by writing the new joint configuration.
        # (In Warp, we assume self.joint_q is mutable; otherwise, use an assign method.)
        for i in range(n):
            self.joint_q[i] = new_joint_q[i]

        # Store the cost and update for momentum in the next iteration.
        self.prev_cost = new_cost
        self.prev_dq = dq_np

        # Optional: sleep a tiny duration to simulate computation latency.
        time.sleep(0.001)