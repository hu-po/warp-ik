import warp as wp
import numpy as np

from src.morph import Morph

class TestMorph(Morph):

    def compute_geometric_jacobian(self):
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
        return jacobians

    def compute_fd_jacobian(self, eps=1e-4):
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
        return jacobians

    def _step(self):
        with wp.ScopedTimer("jacobian", print=False, active=True, dict=self.profiler):
            jacobians = self.compute_geometric_jacobian()
        ee_error_flat = self.compute_ee_error().numpy()
        error = ee_error_flat.reshape(self.num_envs, 6, 1)
        delta_q = -self.step_size * np.matmul(jacobians.transpose(0, 2, 1), error)
        self.model.joint_q = wp.array(
            self.model.joint_q.numpy() + delta_q.flatten(),
            dtype=wp.float32,
            requires_grad=True,
        )