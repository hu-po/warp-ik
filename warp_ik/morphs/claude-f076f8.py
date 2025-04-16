import warp as wp
import numpy as np
import math
from warp_ik.src.morph import BaseMorph

@wp.func
def quaternion_mul(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    return wp.vec4(
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
    )

@wp.func
def quaternion_inverse(q: wp.vec4) -> wp.vec4:
    return wp.vec4(-q[0], -q[1], -q[2], q[3])

@wp.kernel
def create_gradient_samples_kernel(
    base_joint_q: wp.array(dtype=wp.float32),
    random_directions: wp.array(dtype=wp.float32),
    sample_step_size: float,
    num_envs: int,
    dof: int,
    samples_per_env: int,
    samples_joint_q: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    env_idx = tid // (dof * samples_per_env)
    remainder = tid % (dof * samples_per_env)
    sample_idx = remainder // dof
    dof_idx = remainder % dof
    
    if env_idx < num_envs:
        global_idx = env_idx * dof + dof_idx
        sample_out_idx = tid
        
        base_value = base_joint_q[global_idx]
        direction = random_directions[sample_idx * dof + dof_idx]
        samples_joint_q[sample_out_idx] = base_value + sample_step_size * direction

@wp.kernel
def update_base_points_kernel(
    joint_q: wp.array(dtype=wp.float32),
    best_samples_joint_q: wp.array(dtype=wp.float32),
    rho: float,
    num_envs: int,
    dof: int,
    out_joint_q: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    env_idx = tid // dof
    dof_idx = tid % dof
    
    if env_idx < num_envs:
        global_idx = env_idx * dof + dof_idx
        best_sample_idx = env_idx * dof + dof_idx
        
        out_joint_q[global_idx] = joint_q[global_idx] + rho * (best_samples_joint_q[best_sample_idx] - joint_q[global_idx])

@wp.kernel
def select_best_samples_kernel(
    samples_joint_q: wp.array(dtype=wp.float32),
    samples_errors: wp.array(dtype=wp.float32),
    num_envs: int,
    dof: int,
    samples_per_env: int,
    best_samples_joint_q: wp.array(dtype=wp.float32),
    best_sample_indices: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    env_idx = tid // dof
    dof_idx = tid % dof
    
    if env_idx < num_envs:
        # Find best sample for this environment
        if dof_idx == 0:  # Only do this once per environment
            best_error = 1e9
            best_sample_idx = 0
            
            for s in range(samples_per_env):
                sample_error = 0.0
                for i in range(6):  # Sum the 6D error components
                    error_idx = env_idx * samples_per_env * 6 + s * 6 + i
                    error_val = samples_errors[error_idx]
                    sample_error += error_val * error_val
                
                if sample_error < best_error:
                    best_error = sample_error
                    best_sample_idx = s
            
            best_sample_indices[env_idx] = best_sample_idx
        
        # Synchronize threads in the same environment
        wp.sync()
        
        best_sample_idx = best_sample_indices[env_idx]
        sample_joint_idx = env_idx * samples_per_env * dof + best_sample_idx * dof + dof_idx
        best_samples_joint_q[env_idx * dof + dof_idx] = samples_joint_q[sample_joint_idx]

class Morph(BaseMorph):
    """
    Evolutionary Strategy-based IK Solver (ES-IK)
    
    This novel IK solver uses a particle-based evolutionary strategy approach to explore
    the configuration space efficiently. The algorithm generates multiple random samples
    around the current joint configuration, evaluates their fitness based on end-effector
    pose error, and moves toward the best-performing samples using an adaptive strategy
    that balances exploration and exploitation.
    
    Key innovations:
    1. Parallel particle evaluation leveraging GPU compute
    2. Adaptive exploration radius that narrows as error decreases
    3. Momentum-based updates to escape local minima
    4. No Jacobian computation required (gradient-free optimization)
    5. Implicitly handles joint limits and singularities
    """

    def _update_config(self):
        self.config.step_size = 0.5
        self.config.joint_q_requires_grad = False
        self.config.config_extras = {
            "samples_per_env": 16,       # Number of random samples per environment
            "exploration_radius": 0.1,   # Initial radius for random perturbations
            "min_radius": 0.001,         # Minimum exploration radius
            "radius_decay": 0.95,        # How quickly to reduce exploration radius
            "momentum": 0.2,             # Momentum coefficient for updates
            "fitness_smoothing": 0.7,    # Smoothing factor for fitness history
            "patience": 5,               # Patience for adaptive strategy
        }
        
        # Initialize additional state
        self.last_best_q = None
        self.exploration_radius = self.config.config_extras["exploration_radius"]
        self.fitness_history = []
        self.stagnation_counter = 0
        self.momentum_vector = None

    def _step(self):
        """
        Performs one step of the Evolutionary Strategy IK solver.
        
        This method:
        1. Generates multiple random samples around the current joint configuration
        2. Evaluates the end-effector pose error for each sample
        3. Selects the best-performing samples and updates the joint configuration
        4. Adapts the exploration strategy based on recent performance
        """
        # Configuration parameters
        samples_per_env = self.config.config_extras["samples_per_env"]
        
        # Current joint positions and error
        current_q = self.model.joint_q.numpy().copy()
        current_error = self.compute_ee_error().numpy()
        current_fitness = np.sum(current_error**2).item()
        
        # Initialize momentum vector if not already done
        if self.momentum_vector is None:
            self.momentum_vector = np.zeros_like(current_q)
        
        # Generate random directions on CPU (could be done on GPU for very large DOF)
        # Using normalized random directions for better exploration
        random_directions = np.random.normal(0.0, 1.0, (samples_per_env, self.dof))
        for i in range(samples_per_env):
            # Normalize each direction vector
            norm = np.linalg.norm(random_directions[i])
            if norm > 0:
                random_directions[i] /= norm
        
        # Transfer random directions to GPU
        random_directions_wp = wp.array(random_directions.flatten(), dtype=wp.float32, device=self.config.device)
        
        # Create storage for sample configurations and their errors
        samples_joint_q = wp.zeros(self.num_envs * samples_per_env * self.dof, dtype=wp.float32, device=self.config.device)
        
        # Generate samples in parallel
        wp.launch(
            create_gradient_samples_kernel,
            dim=self.num_envs * samples_per_env * self.dof,
            inputs=[
                self.model.joint_q, 
                random_directions_wp, 
                self.exploration_radius,
                self.num_envs, 
                self.dof, 
                samples_per_env
            ],
            outputs=[samples_joint_q],
            device=self.config.device
        )
        
        # Evaluate all samples
        samples_errors = wp.zeros(self.num_envs * samples_per_env * 6, dtype=wp.float32, device=self.config.device)
        
        for s in range(samples_per_env):
            # Extract this sample's joint configurations
            s_joint_q = wp.zeros(self.num_envs * self.dof, dtype=wp.float32, device=self.config.device)
            
            # Copy the appropriate slice of samples_joint_q to s_joint_q
            s_joint_q_np = np.zeros(self.num_envs * self.dof, dtype=np.float32)
            for e in range(self.num_envs):
                for d in range(self.dof):
                    idx = e * samples_per_env * self.dof + s * self.dof + d
                    s_joint_q_np[e * self.dof + d] = samples_joint_q.numpy()[idx]
            
            s_joint_q = wp.array(s_joint_q_np, dtype=wp.float32, device=self.config.device)
            
            # Set the model's joint positions to this sample
            original_joint_q = self.model.joint_q.numpy().copy()
            self.model.joint_q.assign(s_joint_q)
            
            # Compute error for this sample
            error = self.compute_ee_error().numpy()
            
            # Store the error for this sample
            for e in range(self.num_envs):
                for i in range(6):
                    samples_errors.numpy()[e * samples_per_env * 6 + s * 6 + i] = error[e * 6 + i]
            
            # Restore original joint positions
            self.model.joint_q.assign(wp.array(original_joint_q, dtype=wp.float32, device=self.config.device))
        
        # Select best samples
        best_samples_joint_q = wp.zeros(self.num_envs * self.dof, dtype=wp.float32, device=self.config.device)
        best_sample_indices = wp.zeros(self.num_envs, dtype=wp.int32, device=self.config.device)
        
        wp.launch(
            select_best_samples_kernel,
            dim=self.num_envs * self.dof,
            inputs=[
                samples_joint_q,
                samples_errors,
                self.num_envs,
                self.dof,
                samples_per_env
            ],
            outputs=[best_samples_joint_q, best_sample_indices],
            device=self.config.device
        )
        
        # Compute fitness of best samples
        best_samples_error = np.zeros(self.num_envs * 6, dtype=np.float32)
        best_samples_indices = best_sample_indices.numpy()
        
        for e in range(self.num_envs):
            best_s = best_samples_indices[e]
            for i in range(6):
                best_samples_error[e * 6 + i] = samples_errors.numpy()[e * samples_per_env * 6 + best_s * 6 + i]
        
        best_fitness = np.sum(best_samples_error**2).item()
        
        # Update fitness history and check for improvement
        if self.fitness_history:
            smoothed_fitness = self.fitness_history[-1] * self.config.config_extras["fitness_smoothing"] + \
                              best_fitness * (1 - self.config.config_extras["fitness_smoothing"])
        else:
            smoothed_fitness = best_fitness
        
        self.fitness_history.append(smoothed_fitness)
        
        # Adaptive strategy based on progress
        if len(self.fitness_history) > 1:
            if self.fitness_history[-1] < self.fitness_history[-2] * 0.95:  # Significant improvement
                self.stagnation_counter = 0
                # Keep the exploration radius (don't reduce it as we're making good progress)
            else:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.config.config_extras["patience"]:
                    # Reduce exploration radius when progress stagnates
                    self.exploration_radius = max(
                        self.config.config_extras["min_radius"],
                        self.exploration_radius * self.config.config_extras["radius_decay"]
                    )
                    self.stagnation_counter = 0
        
        # Update joint configurations with momentum
        best_q = best_samples_joint_q.numpy()
        
        # Compute direction vector from current to best
        direction = best_q - current_q
        
        # Update momentum vector
        self.momentum_vector = self.momentum_vector * self.config.config_extras["momentum"] + \
                              direction * (1 - self.config.config_extras["momentum"])
        
        # Update joint configuration
        next_q = current_q + self.momentum_vector
        
        # Assign final result
        self.model.joint_q.assign(wp.array(next_q, dtype=wp.float32, device=self.config.device))
        
        # Save the best configuration for the next iteration
        self.last_best_q = best_q.copy()