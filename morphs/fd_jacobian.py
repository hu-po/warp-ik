import warp as wp
import numpy as np

from warp_ik.src.morph import BaseMorph

class Morph(BaseMorph):

    def _step(self):
        jacobians = np.zeros((self.num_envs, 6, self.dof), dtype=np.float32)
        q0 = self.model.joint_q.numpy().copy()
        for e in range(self.num_envs):
            for i in range(self.dof):
                q = q0.copy()
                q[e * self.dof + i] += eps
                self.model.joint_q.assign(q)
                self.compute_ee_error()
                f_plus = self.ee_error.numpy()[e * 6:(e + 1) * 6].copy()
                q[e * self.dof + i] -= 2 * eps
                self.model.joint_q.assign(q)
                self.compute_ee_error()
                f_minus = self.ee_error.numpy()[e * 6:(e + 1) * 6].copy()
                jacobians[e, :, i] = (f_plus - f_minus) / (2 * eps)
        self.model.joint_q.assign(q0)
        ee_error_flat = self.compute_ee_error().numpy()
        error = ee_error_flat.reshape(self.num_envs, 6, 1)
        delta_q = -self.step_size * np.matmul(jacobians.transpose(0, 2, 1), error)
        self.model.joint_q = wp.array(
            self.model.joint_q.numpy() + delta_q.flatten(),
            dtype=wp.float32,
            requires_grad=True,
        )