import warp as wp
import numpy as np

from warp_ik.src.morph import BaseMorph

class Morph(BaseMorph):

    def _step(self):
        """
            Performs IK by computing the Geometric Jacobian of the end effector in 3D space
            and solving for the joint angles that minimize the error.
        """
        jacobians = np.empty((self.num_envs, 3, self.dof), dtype=np.float32)
        tape = wp.Tape()
        with tape:
            self.compute_ee_position()
        for o in range(3):
            # select which row of the Jacobian we want to compute
            select_index = np.zeros(3)
            select_index[o] = 1.0
            e = wp.array(np.tile(select_index, self.num_envs), dtype=wp.vec3)
            tape.backward(grads={self.ee_pos: e})
            q_grad_i = tape.gradients[self.model.joint_q]
            jacobians[:, o, :] = q_grad_i.numpy().reshape(self.num_envs, self.dof)
            tape.zero()
        self.ee_pos_np = self.compute_ee_position().numpy()
        error = self.targets - self.ee_pos_np
        self.error = error.reshape(self.num_envs, 3, 1)
        delta_q = np.matmul(jacobians.transpose(0, 2, 1), self.error)
        self.model.joint_q = wp.array(
            self.model.joint_q.numpy() + self.step_size * delta_q.flatten(),
            dtype=wp.float32,
            requires_grad=True,
        )