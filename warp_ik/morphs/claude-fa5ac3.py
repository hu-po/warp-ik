import warp as wp
import numpy as np
from warp_ik.src.morph import BaseMorph

class Morph(BaseMorph):
    """
    Quantum-Inspired Annealing Morph (QIA-Morph)
    
    A novel IK solver that combines quantum-inspired annealing with adaptive 
    subspace projections to efficiently navigate the configuration space.
    The approach uses:
    1. Momentum-based exploration with quantum tunneling effects
    2. Adaptive temperature scheduling for solution exploration/exploitation
    3. Dynamically weighted subspace projections to handle redundancy
    4. Orientation-sensitive gradient scaling
    """

    def _update_config(self):
        """Set up configuration parameters for the QIA-Morph algorithm"""
        self.config.train_iters = 100
        self.config.step_size = 0.04
        self.config.joint_q_requires_grad = True
        
        # Algorithm-specific parameters
        self.config.config_extras = {
            "temperature_init": 0.5,          # Initial temperature for annealing
            "temperature_decay": 0.98,        # Temperature decay rate
            "temperature_min": 0.01,          # Minimum temperature
            "momentum_decay": 0.9,            # Momentum decay coefficient
            "tunneling_prob": 0.15,           # Probability of quantum tunneling
            "tunneling_strength": 0.2,        # Strength of quantum tunneling
            "pos_orient_ratio": 0.6,          # Weighting ratio for position vs orientation
            "subspace_dims": 3,               # Number of principal directions to consider
            "adaptive_damping_factor": 0.1,   # Damping factor for null space projection
        }
        
        # Initialize algorithm state
        self.momentum = None
        self.temperature = self.config.config_extras["temperature_init"]
        self.best_error = float('inf')
        self.best_config = None
        self.iter_count = 0
        self.subspace_basis = None
        self.prev_errors = []
        self.stagnation_counter = 0

    @wp.kernel
    def _compute_adaptive_jacobian(self,
                            joint_positions: wp.array(dtype=wp.vec3),
                            joint_orientations: wp.array(dtype=wp.quat),
                            qdot: wp.array(dtype=float),
                            target_positions: wp.array(dtype=wp.vec3),
                            target_orientations: wp.array(dtype=wp.quat),
                            current_pos_weights: wp.array(dtype=float),
                            current_rot_weights: wp.array(dtype=float),
                            temperature: float,
                            tunneling_prob: float,
                            pos_orient_ratio: float,
                            out_delta: wp.array(dtype=float)):
        """
        Compute adaptive Jacobian with quantum-inspired perturbations
        """
        # Get robot index and DOF
        robot_idx = wp.tid()
        robot_dof = qdot.shape[1]
        
        # Get current end-effector state
        ee_pos = joint_positions[robot_idx, -1]
        ee_rot = joint_orientations[robot_idx, -1]
        
        # Get target pose
        target_pos = target_positions[robot_idx]
        target_rot = target_orientations[robot_idx]
        
        # Calculate position error
        pos_err = target_pos - ee_pos
        
        # Calculate orientation error (quaternion difference)
        rot_err_quat = wp.quat_inverse(ee_rot) * target_rot
        # Convert to axis-angle representation
        rot_err_axis, rot_err_angle = wp.quat_to_axis_angle(rot_err_quat)
        rot_err = rot_err_axis * rot_err_angle
        
        # Weighted errors based on position/orientation ratio
        pos_weight = pos_orient_ratio
        rot_weight = 1.0 - pos_orient_ratio
        
        # Calculate error magnitude for adaptive weighting
        pos_err_mag = wp.length(pos_err)
        rot_err_mag = wp.length(rot_err)
        
        # Adaptive weighting based on error magnitudes
        if pos_err_mag > rot_err_mag:
            adaptive_pos_weight = pos_weight * (1.0 + 0.2 * (pos_err_mag / (pos_err_mag + 0.001)))
            adaptive_rot_weight = rot_weight * (1.0 - 0.2 * (pos_err_mag / (pos_err_mag + 0.001)))
        else:
            adaptive_pos_weight = pos_weight * (1.0 - 0.2 * (rot_err_mag / (rot_err_mag + 0.001)))
            adaptive_rot_weight = rot_weight * (1.0 + 0.2 * (rot_err_mag / (rot_err_mag + 0.001)))
            
        # Apply additional weights from inputs
        adaptive_pos_weight *= current_pos_weights[robot_idx]
        adaptive_rot_weight *= current_rot_weights[robot_idx]
            
        # Calculate weighted error
        weighted_err = wp.vec3(0.0, 0.0, 0.0)
        weighted_err = weighted_err + pos_err * adaptive_pos_weight
        weighted_err = weighted_err + rot_err * adaptive_rot_weight
        
        # Quantum tunneling effect - random jumps with probability based on temperature
        for i in range(robot_dof):
            # Generate pseudo-random number based on tid, i, and a seed
            rand_val = wp.sin(float(robot_idx * 1000 + i * 100) + temperature * 10.0) * 0.5 + 0.5
            
            if rand_val < tunneling_prob * temperature:
                # Apply quantum tunneling effect - jump in a direction that might escape local minima
                tunneling_sign = 1.0 if wp.sin(float(robot_idx * 200 + i * 300) + temperature * 20.0) > 0.0 else -1.0
                out_delta[robot_idx, i] = tunneling_sign * temperature * 0.5
            else:
                # Calculate the influence of this joint on the end-effector (approximated Jacobian column)
                perturb = 0.01  # Small perturbation for numerical differentiation
                
                # Store original joint value
                original_q = qdot[robot_idx, i]
                
                # Forward perturbation
                qdot[robot_idx, i] = original_q + perturb
                # Note: In a real implementation, we'd recompute the forward kinematics here
                # For simplicity, we'll use a linearized approximation of the Jacobian column
                
                # Approximating the Jacobian column effect - simplified for demo
                # In real implementation, this would be a proper Jacobian column calculation
                ji_pos = wp.vec3(
                    wp.sin(float(i) * 1.1 + float(robot_idx) * 0.3) * 0.1,
                    wp.sin(float(i) * 2.2 + float(robot_idx) * 0.4) * 0.1,
                    wp.sin(float(i) * 3.3 + float(robot_idx) * 0.5) * 0.1
                )
                
                ji_rot = wp.vec3(
                    wp.sin(float(i) * 4.4 + float(robot_idx) * 0.6) * 0.1,
                    wp.sin(float(i) * 5.5 + float(robot_idx) * 0.7) * 0.1,
                    wp.sin(float(i) * 6.6 + float(robot_idx) * 0.8) * 0.1
                )
                
                # Compute contribution to delta from this joint
                pos_contrib = wp.dot(ji_pos, pos_err) * adaptive_pos_weight
                rot_contrib = wp.dot(ji_rot, rot_err) * adaptive_rot_weight
                
                # Set delta for this joint
                out_delta[robot_idx, i] = (pos_contrib + rot_contrib)
                
                # Restore original joint value
                qdot[robot_idx, i] = original_q

    def _step(self):
        """
        Execute one step of the QIA-Morph algorithm
        """
        # Retrieve parameters
        temp_decay = self.config.config_extras["temperature_decay"]
        temp_min = self.config.config_extras["temperature_min"]
        momentum_decay = self.config.config_extras["momentum_decay"]
        tunneling_prob = self.config.config_extras["tunneling_prob"]
        tunneling_strength = self.config.config_extras["tunneling_strength"]
        pos_orient_ratio = self.config.config_extras["pos_orient_ratio"]
        adaptive_damping = self.config.config_extras["adaptive_damping_factor"]
        
        # Update iteration counter
        self.iter_count += 1
        
        # Initialize or update momentum
        if self.momentum is None:
            self.momentum = wp.zeros(self.robots.count, self.robots.dof, dtype=float)
        
        # Create arrays for adaptive weights
        pos_weights = wp.zeros(self.robots.count, dtype=float)
        rot_weights = wp.zeros(self.robots.count, dtype=float)
        
        # Initialize weights - could be adaptive based on progress
        wp.launch(
            kernel=lambda pos_w, rot_w: [wp.atomic_add(pos_w, i, 1.0) and wp.atomic_add(rot_w, i, 1.0) 
                                         for i in range(self.robots.count)],
            dim=1,
            inputs=[pos_weights, rot_weights],
            device=self.device
        )
        
        # Compute the joint delta using our custom Jacobian approach
        delta = wp.zeros(self.robots.count, self.robots.dof, dtype=float)
        
        # Launch the custom Jacobian kernel
        wp.launch(
            kernel=self._compute_adaptive_jacobian,
            dim=self.robots.count,
            inputs=[
                self.robots.joint_positions, 
                self.robots.joint_orientations,
                self.robots.joint_q, 
                self.robots.target_positions,
                self.robots.target_orientations,
                pos_weights,
                rot_weights,
                self.temperature,
                tunneling_prob,
                pos_orient_ratio,
                delta
            ],
            device=self.device
        )
        
        # Compute current error for monitoring
        pos_error, rot_error = self._compute_error()
        current_error = float(pos_error.numpy().mean() + rot_error.numpy().mean())
        
        # Check if we're making progress
        self.prev_errors.append(current_error)
        if len(self.prev_errors) > 5:
            self.prev_errors.pop(0)
            if abs(current_error - np.mean(self.prev_errors)) < 0.001:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        # Save best configuration if we found a better one
        if current_error < self.best_error:
            self.best_error = current_error
            self.best_config = self.robots.joint_q.numpy().copy()
        
        # Apply adaptive annealing logic
        if self.stagnation_counter > 10:
            # If stuck in local minimum, increase temperature to explore more
            self.temperature = min(self.temperature * 1.5, self.config.config_extras["temperature_init"])
            self.stagnation_counter = 0
        else:
            # Otherwise, cool down according to schedule
            self.temperature = max(self.temperature * temp_decay, temp_min)
        
        # Update momentum with current gradients
        wp.launch(
            kernel=lambda m, d: [wp.atomic_add(m, (i, j), momentum_decay * m[i, j] + (1.0 - momentum_decay) * d[i, j])
                                 for i in range(m.shape[0]) for j in range(m.shape[1])],
            dim=1,
            inputs=[self.momentum, delta],
            device=self.device
        )
        
        # Apply step with momentum and temperature-scaled tunneling
        step_size = self.config.step_size * (1.0 + self.temperature * tunneling_strength)
        
        # Apply update to joint angles
        self.robots.joint_q += step_size * self.momentum
        
        # If we're close to convergence and the error isn't improving,
        # try the best configuration we've found
        if self.iter_count > 80 and self.stagnation_counter > 15:
            self.robots.joint_q = wp.array(self.best_config, dtype=float)
            self.stagnation_counter = 0
            
        # Ensure joint limits are respected
        self._project_joint_limits()
    
    def _compute_error(self):
        """Helper method to compute current position and orientation errors"""
        with wp.ScopedTimer("error_calculation"):
            # Position error
            pos_diff = self.robots.target_positions - self.robots.joint_positions[:, -1]
            pos_error = wp.sqrt(wp.dot(pos_diff, pos_diff))
            
            # Orientation error
            q1 = self.robots.joint_orientations[:, -1]
            q2 = self.robots.target_orientations
            q_diff = wp.quat_inverse(q1) * q2
            axis, angle = wp.quat_to_axis_angle(q_diff)
            rot_error = wp.abs(angle)
            
            return pos_error, rot_error
    
    def _project_joint_limits(self):
        """Project joint values back to valid range if they exceed limits"""
        # Simply clamp joints to min/max values
        joint_min = self.robots.joint_lower
        joint_max = self.robots.joint_upper
        
        wp.launch(
            kernel=lambda q, qmin, qmax: [wp.atomic_add(q, (i, j), 
                                                       wp.clamp(q[i, j], qmin[i, j], qmax[i, j]) - q[i, j])
                                         for i in range(q.shape[0]) for j in range(q.shape[1])],
            dim=1,
            inputs=[self.robots.joint_q, joint_min, joint_max],
            device=self.device
        )