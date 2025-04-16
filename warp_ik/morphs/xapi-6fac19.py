import warp as wp
import numpy as np
from warp_ik.src.morph import BaseMorph

@wp.kernel
def compute_particle_weights_kernel(
    errors: wp.array(dtype=wp.float32),  # Shape: (num_envs * num_particles * 6)
    weights: wp.array(dtype=wp.float32),  # Shape: (num_envs * num_particles)
    num_envs: int,
    num_particles: int
):
    tid = wp.tid()
    env_idx = tid // num_particles
    particle_idx = tid % num_particles
    
    if env_idx < num_envs:
        # Compute squared norm of 6D error for this particle
        error_norm_sq = 0.0
        for i in range(6):
            error_val = errors[env_idx * num_particles * 6 + particle_idx * 6 + i]
            error_norm_sq += error_val * error_val
        # Weight is inversely proportional to error (Gaussian-like)
        weights[tid] = wp.exp(-0.5 * error_norm_sq)

@wp.kernel
def normalize_weights_kernel(
    weights: wp.array(dtype=wp.float32),  # Shape: (num_envs * num_particles)
    num_envs: int,
    num_particles: int
):
    env_idx = wp.tid()
    if env_idx < num_envs:
        total_weight = 0.0
        for p in range(num_particles):
            total_weight += weights[env_idx * num_particles + p]
        if total_weight > 0.0:
            for p in range(num_particles):
                weights[env_idx * num_particles + p] /= total_weight

@wp.kernel
def resample_particles_kernel(
    weights: wp.array(dtype=wp.float32),
    particles: wp.array2d(dtype=wp.float32),  # Shape: (num_envs * num_particles, dof)
    new_particles: wp.array2d(dtype=wp.float32),
    num_envs: int,
    num_particles: int,
    dof: int,
    seed: int
):
    env_idx = wp.tid()
    if env_idx < num_envs:
        # Cumulative sum for systematic resampling
        cumulative_weights = wp.zeros(num_particles, dtype=wp.float32)
        cumulative_weights[0] = weights[env_idx * num_particles]
        for p in range(1, num_particles):
            cumulative_weights[p] = cumulative_weights[p - 1] + weights[env_idx * num_particles + p]
        
        step = 1.0 / float(num_particles)
        u = wp.randf(seed + env_idx) * step
        current_idx = 0
        
        for p in range(num_particles):
            while u > cumulative_weights[current_idx] and current_idx < num_particles - 1:
                current_idx += 1
            selected_idx = env_idx * num_particles + current_idx
            target_idx = env_idx * num_particles + p
            for d in range(dof):
                new_particles[target_idx, d] = particles[selected_idx, d]
            u += step
            if u > 1.0:
                u -= 1.0

@wp.kernel
def update_joint_q_from_particles_kernel(
    particles: wp.array2d(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    num_envs: int,
    num_particles: int,
    dof: int
):
    env_idx = wp.tid()
    if env_idx < num_envs:
        for d in range(dof):
            sum_q = 0.0
            for p in range(num_particles):
                sum_q += particles[env_idx * num_particles + p, d]
            joint_q[env_idx * dof + d] = sum_q / float(num_particles)

@wp.kernel
def perturb_particles_kernel(
    particles: wp.array2d(dtype=wp.float32),
    sigma: float,
    num_envs: int,
    num_particles: int,
    dof: int,
    seed: int
):
    tid = wp.tid()
    env_idx = tid // (num_particles * dof)
    particle_idx = (tid // dof) % num_particles
    dof_idx = tid % dof
    
    if env_idx < num_envs:
        rand_val = wp.randn(seed + tid)
        particles[env_idx * num_particles + particle_idx, dof_idx] += sigma * rand_val

class Morph(BaseMorph):
    """
    Inverse Kinematics Morph using a Particle Filter Approach.

    This morph implements a novel IK solver using a particle filter to estimate
    the optimal joint configuration. It maintains a set of particles representing
    possible joint configurations, evaluates their fitness based on end-effector
    pose error, and iteratively resamples and perturbs them to converge to a solution.
    This probabilistic method handles multimodal solutions and avoids local minima
    better than traditional gradient-based approaches.
    """
    def _update_config(self):
        """
        Updates the configuration specific to the Particle Filter Morph.
        Sets parameters for particle filter operation and disables gradients
        since this is a sampling-based method.
        """
        self.config.step_size = 0.0  # Not used in particle filter
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {
            "num_particles": 50,  # Number of particles per environment
            "sigma": 0.05,        # Standard deviation for particle perturbation
            "seed": 42            # Random seed for reproducibility
        }
        # Initialize particle-related data structures
        self.num_particles = self.config.config_extras["num_particles"]
        self.particles = wp.zeros((self.num_envs * self.num_particles, self.dof), dtype=wp.float32, device=self.config.device)
        self.new_particles = wp.zeros((self.num_envs * self.num_particles, self.dof), dtype=wp.float32, device=self.config.device)
        self.weights = wp.zeros(self.num_envs * self.num_particles, dtype=wp.float32, device=self.config.device)
        self.particle_errors = wp.zeros(self.num_envs * self.num_particles * 6, dtype=wp.float32, device=self.config.device)
        self._initialize_particles()

    def _initialize_particles(self):
        """
        Initializes particles by sampling around the current joint configuration.
        """
        initial_q = self.model.joint_q.numpy().copy()
        particles_np = np.zeros((self.num_envs * self.num_particles, self.dof), dtype=np.float32)
        for e in range(self.num_envs):
            for p in range(self.num_particles):
                particles_np[e * self.num_particles + p] = initial_q[e * self.dof:(e + 1) * self.dof] + np.random.normal(0, 0.1, self.dof)
        self.particles = wp.array(particles_np, dtype=wp.float32, device=self.config.device)

    def _step(self):
        """
        Performs one IK step using the Particle Filter method.
        1. Evaluates the end-effector error for each particle.
        2. Computes weights based on error (lower error = higher weight).
        3. Resamples particles based on weights.
        4. Perturbs particles to maintain diversity.
        5. Updates joint configuration as the mean of particles.
        """
        # Step 1: Compute errors for all particles
        for p in range(self.num_particles):
            for e in range(self.num_envs):
                particle_idx = e * self.num_particles + p
                q_particle = self.particles[particle_idx].numpy().copy()
                self.model.joint_q.assign(wp.array(q_particle, dtype=wp.float32, device=self.config.device))
                ee_error = self.compute_ee_error().numpy()[e * 6:(e + 1) * 6]
                self.particle_errors[particle_idx * 6:(particle_idx + 1) * 6].assign(wp.array(ee_error, dtype=wp.float32, device=self.config.device))

        # Step 2: Compute weights based on errors
        wp.launch(
            compute_particle_weights_kernel,
            dim=self.num_envs * self.num_particles,
            inputs=[self.particle_errors, self.weights, self.num_envs, self.num_particles],
            device=self.config.device
        )

        # Step 3: Normalize weights
        wp.launch(
            normalize_weights_kernel,
            dim=self.num_envs,
            inputs=[self.weights, self.num_envs, self.num_particles],
            device=self.config.device
        )

        # Step 4: Resample particles based on weights
        wp.launch(
            resample_particles_kernel,
            dim=self.num_envs,
            inputs=[self.weights, self.particles, self.new_particles, self.num_envs, self.num_particles, self.dof, self.config.config_extras["seed"]],
            device=self.config.device
        )
        # Swap buffers
        self.particles, self.new_particles = self.new_particles, self.particles

        # Step 5: Perturb particles to maintain diversity
        wp.launch(
            perturb_particles_kernel,
            dim=self.num_envs * self.num_particles * self.dof,
            inputs=[self.particles, self.config.config_extras["sigma"], self.num_envs, self.num_particles, self.dof, self.config.config_extras["seed"]],
            device=self.config.device
        )

        # Step 6: Update joint angles as mean of particles
        wp.launch(
            update_joint_q_from_particles_kernel,
            dim=self.num_envs,
            inputs=[self.particles, self.model.joint_q, self.num_envs, self.num_particles, self.dof],
            device=self.config.device
        )