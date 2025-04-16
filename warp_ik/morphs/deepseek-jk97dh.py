import warp as wp
import numpy as np

from warp_ik.src.morph import BaseMorph

wp.init()

@wp.kernel
def jacobian_transpose_multiply_kernel(
    jacobians: wp.array3d(dtype=wp.float32),
    error: wp.array(dtype=wp.float32),
    alpha: float,
    num_envs: int,
    dof: int,
    delta_q: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    env_idx = tid // dof
    dof_idx = tid % dof
    
    if env_idx < num_envs:
        sum = 0.0
        for j in range(6):
            sum += jacobians[env_idx, j, dof_idx] * error[env_idx * 6 + j]
        delta_q[tid] = -alpha * sum

@wp.kernel
def add_delta_q_kernel(
    joint_q: wp.array(dtype=wp.float32),
    delta_q: wp.array(dtype=wp.float32),
    out_joint_q: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    out_joint_q[tid] = joint_q[tid] + delta_q[tid]

class Morph(BaseMorph):
    def _update_config(self):
        self.config.step_size = 0.5
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {"eps": 1e-4}

    def _step(self):
        ee_error_flat_wp = self.compute_ee_error()
        ee_error_flat_initial = ee_error_flat_wp.numpy().copy()
        q0 = self.model.joint_q.numpy().copy()
        eps = self.config.config_extras["eps"]

        jacobians = np.zeros((self.num_envs, 6, self.dof), dtype=np.float32)

        for e in range(self.num_envs):
            for i in range(self.dof):
                q_plus = q0.copy()
                q_plus[e * self.dof + i] += eps
                self.model.joint_q.assign(wp.array(q_plus, dtype=wp.float32, device=self.config.device))
                f_plus = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6].copy()

                q_minus = q0.copy()
                q_minus[e * self.dof + i] -= eps
                self.model.joint_q.assign(wp.array(q_minus, dtype=wp.float32, device=self.config.device))
                f_minus = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6].copy()

                jacobians[e, :, i] = (f_plus - f_minus) / (2 * eps)

        self.model.joint_q.assign(wp.array(q0, dtype=wp.float32, device=self.config.device))
        jacobians_wp = wp.array(jacobians, dtype=wp.float32, device=self.config.device)
        delta_q_wp = wp.zeros(self.num_envs * self.dof, dtype=wp.float32, device=self.config.device)

        wp.launch(
            jacobian_transpose_multiply_kernel,
            dim=self.num_envs * self.dof,
            inputs=[jacobians_wp, ee_error_flat_wp, self.config.step_size, self.num_envs, self.dof],
            outputs=[delta_q_wp],
            device=self.config.device
        )

        original_joint_q = wp.array(q0, dtype=wp.float32, device=self.config.device)
        wp.launch(
            add_delta_q_kernel,
            dim=self.num_envs * self.dof,
            inputs=[self.model.joint_q, delta_q_wp],
            outputs=[self.model.joint_q],
            device=self.config.device
        )

        new_ee_error_flat_wp = self.compute_ee_error()
        new_ee_error_flat = new_ee_error_flat_wp.numpy()
        
        initial_total_error = np.sum(ee_error_flat_initial**2)
        new_total_error = np.sum(new_ee_error_flat**2)

        if new_total_error < initial_total_error:
            self.config.step_size = min(self.config.step_size * 1.1, 1.0)
        else:
            self.model.joint_q.assign(original_joint_q)
            self.config.step_size = max(self.config.step_size * 0.5, 0.01)