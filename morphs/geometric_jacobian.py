import warp as wp
import numpy as np

from warp_ik.src.morph import BaseMorph

class Morph(BaseMorph):

    def _step(self):
        jacobians = np.empty((self.num_envs, 6, self.dof), dtype=np.float32)
        tape = wp.Tape()
        with tape:
            self.compute_ee_error()
        for o in range(6):
            select_index = np.zeros(6, dtype=np.float32)
            select_index[o] = 1.0
            e = wp.array(np.tile(select_index, self.num_envs), dtype=wp.float32)
            tape.backward(grads={self.ee_error: e})
            q_grad_i = tape.gradients[self.model.joint_q]
            jacobians[:, o, :] = q_grad_i.numpy().reshape(self.num_envs, self.dof)
            tape.zero()
        ee_error_flat = self.compute_ee_error().numpy()
        error = ee_error_flat.reshape(self.num_envs, 6, 1)
        delta_q = -self.step_size * np.matmul(jacobians.transpose(0, 2, 1), error)
        self.model.joint_q = wp.array(
            self.model.joint_q.numpy() + delta_q.flatten(),
            dtype=wp.float32,
            requires_grad=True,
        )