import warp as wp
import numpy as np

from warp_ik.src.morph import BaseMorph

@wp.kernel
def levenberg_marquardt_kernel(
    jacobians: wp.array3d(dtype=wp.float32),  # (num_envs, 6, dof)
    error: wp.array(dtype=wp.float32),        # (num_envs * 6)
    lambda_: float,
    num_envs: int,
    dof: int,
    delta_q: wp.array(dtype=wp.float32)       # (num_envs * dof)
):
    env_idx = wp.tid()  # One thread per environment
    
    if env_idx >= num_envs:
        return
    
    # Load Jacobian and error for this environment
    J = wp.mat(6, dof, 0.0)
    for i in range(6):
        for j in range(dof):
            J[i,j] = jacobians[env_idx, i, j]
    
    err = wp.vec6()
    base = env_idx*6
    for i in range(6):
        err[i] = error[base + i]
    
    # Compute J^T*J and J^T*error
    JTJ = wp.mat(dof, dof, 0.0)
    JTe = wp.vec(dof, 0.0)
    
    for i in range(dof):
        for k in range(6):
            JTe[i] += J[k,i] * err[k]
            for j in range(dof):
                JTJ[i,j] += J[k,i] * J[k,j]
    
    # Add damping (lambda * diag(JTJ))
    for i in range(dof):
        JTJ[i,i] += lambda_ * JTJ[i,i]
    
    # Solve (JTJ + damping) * delta = -JTe using Cholesky
    L = wp.mat(dof, dof, 0.0)
    for i in range(dof):
        for j in range(i+1):
            sum_ = JTJ[i,j]
            for k in range(j):
                sum_ -= L[i,k] * L[j,k]
            if i == j:
                L[i,j] = wp.sqrt(wp.max(sum_, 1e-6))
            else:
                L[i,j] = sum_ / L[j,j]
    
    # Solve L*y = -JTe
    y = wp.vec(dof, 0.0)
    for i in range(dof):
        sum_ = -JTe[i]
        for j in range(i):
            sum_ -= L[i,j] * y[j]
        y[i] = sum_ / L[i,i]
    
    # Solve L^T*delta = y
    delta = wp.vec(dof, 0.0)
    for i in reversed(range(dof)):
        sum_ = y[i]
        for j in range(i+1, dof):
            sum_ -= L[j,i] * delta[j]
        delta[i] = sum_ / L[i,i]
    
    # Store delta_q
    for i in range(dof):
        delta_q[env_idx*dof + i] = delta[i]

@wp.kernel
def add_delta_q_kernel(
    joint_q: wp.array(dtype=wp.float32),
    delta_q: wp.array(dtype=wp.float32),
    out_joint_q: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    out_joint_q[tid] = joint_q[tid] + delta_q[tid]

class Morph(BaseMorph):
    """
    Levenberg-Marquardt IK Solver with Adaptive Damping.
    
    Uses finite-difference Jacobian approximation combined with
    Levenberg-Marquardt optimization for robust convergence.
    Implements Cholesky decomposition for matrix inversion in
    parallel across all environments.
    """
    
    def _update_config(self):
        self.config.step_size = 1.0  # Overall step scaling
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {
            "eps": 1e-4,    # Finite difference epsilon
            "lambda": 0.01, # Initial damping parameter
        }
    
    def _step(self):
        # Compute initial error
        ee_error_flat_wp = self.compute_ee_error()
        q0 = self.model.joint_q.numpy().copy()
        eps = self.config.config_extras["eps"]
        lambda_ = self.config.config_extras["lambda"]
        
        # Compute Jacobian using finite differences
        jacobians = np.zeros((self.num_envs, 6, self.dof), dtype=np.float32)
        
        for e in range(self.num_envs):
            for i in range(self.dof):
                # Positive perturbation
                q_plus = q0.copy()
                q_plus[e*self.dof + i] += eps
                self.model.joint_q.assign(wp.array(q_plus, dtype=wp.float32, device=self.config.device))
                f_plus = self.compute_ee_error().numpy()[e*6:(e+1)*6]
                
                # Negative perturbation
                q_minus = q0.copy()
                q_minus[e*self.dof + i] -= eps
                self.model.joint_q.assign(wp.array(q_minus, dtype=wp.float32, device=self.config.device))
                f_minus = self.compute_ee_error().numpy()[e*6:(e+1)*6]
                
                # Central difference
                jacobians[e, :, i] = (f_plus - f_minus) / (2*eps)
        
        # Restore original state
        self.model.joint_q.assign(wp.array(q0, dtype=wp.float32, device=self.config.device))
        jacobians_wp = wp.array(jacobians, dtype=wp.float32, device=self.config.device)
        
        # Compute LM updates
        delta_q_wp = wp.zeros(self.num_envs*self.dof, dtype=wp.float32, device=self.config.device)
        wp.launch(
            levenberg_marquardt_kernel,
            dim=self.num_envs,
            inputs=[jacobians_wp, ee_error_flat_wp, lambda_, self.num_envs, self.dof],
            outputs=[delta_q_wp],
            device=self.config.device
        )
        
        # Apply updates with global step size
        scaled_delta = wp.zeros_like(delta_q_wp)
        wp.launch(
            add_delta_q_kernel,
            dim=self.num_envs*self.dof,
            inputs=[wp.zeros_like(delta_q_wp), delta_q_wp, scaled_delta],
            outputs=[scaled_delta],
            device=self.config.device
        )
        wp.launch(
            add_delta_q_kernel,
            dim=self.num_envs*self.dof,
            inputs=[self.model.joint_q, scaled_delta],
            outputs=[self.model.joint_q],
            device=self.config.device
        )