"""
Cognitive Radio Environment for 6G IoT Spectrum Management

This Gymnasium environment simulates dynamic spectrum access for massive IoT deployments.
The agent learns to select optimal channels while avoiding collisions with other devices.

Environment Design:
- Observation: Historical spectrum occupancy (binary matrix)
- Action: Discrete channel selection (0 to n_channels-1)
- Reward: Multi-objective function balancing collision avoidance, throughput, and energy

References:
- ETSI TR 103 511: SmartBAN standards for body area networks
- ITU-R M.2083-0: IMT-2020 requirements (URLLC, mMTC)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any


class CognitiveRadioEnv(gym.Env):
    """
    Cognitive Radio Environment for Dynamic Spectrum Access
    
    The agent observes recent spectrum occupancy and selects a channel for transmission.
    Ground truth data from IoT traffic simulations determines collision outcomes.
    
    Args:
        data_path: Path to .npy file with spectrum occupancy data
        history_length: Number of past time steps to include in observation
        w_collision: Weight for collision penalty in reward function
        w_throughput: Weight for successful transmission reward
        w_energy: Weight for energy cost penalty (channel switching)
        max_episode_steps: Maximum steps per episode (None = full dataset)
        seed: Random seed for reproducibility
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        data_path: str = "data/generated/spectrum_train.npy",
        history_length: int = 20,  # INCREASED: More context for pattern recognition
        w_collision: float = 5.0,   # OPTIMIZED: Better balance
        w_throughput: float = 10.0, # INCREASED: Strong success incentive
        w_energy: float = 0.2,      # INCREASED: Penalize unnecessary switches
        max_episode_steps: Optional[int] = None,
        seed: Optional[int] = None,
        use_enhanced_features: bool = True,  # NEW: Enhanced feature extraction
    ):
        super().__init__()
        
        # Load ground truth spectrum data
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        self.ground_truth = np.load(self.data_path)  # Shape: (time_steps, n_channels)
        self.n_timesteps, self.n_channels = self.ground_truth.shape
        
        # Environment parameters
        self.history_length = history_length
        self.w_collision = w_collision
        self.w_throughput = w_throughput
        self.w_energy = w_energy
        self.max_episode_steps = max_episode_steps or self.n_timesteps
        self.use_enhanced_features = use_enhanced_features
        
        # Initialize advanced tracking for better learning
        self.channel_success_history = np.zeros(self.n_channels)  # Success rate per channel
        self.channel_collision_history = np.zeros(self.n_channels)  # Collision rate per channel
        self.recent_channel_usage = np.zeros(self.n_channels)      # Recent usage frequency
        self.action_diversity_bonus = 0  # Track exploration diversity
        self.stuck_penalty_counter = 0   # Counter for being stuck on same channel
        
        # Define observation and action spaces with enhanced features
        if self.use_enhanced_features:
            # Enhanced observation: [spectrum_history, channel_stats, temporal_features]
            obs_size = (history_length + 3) * self.n_channels  # +3 for success/collision/usage stats
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(obs_size,),  # Flattened for CNN processing
                dtype=np.float32
            )
        else:
            # Original observation space
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(history_length, self.n_channels),
                dtype=np.float32
            )
        
        self.action_space = spaces.Discrete(self.n_channels)
        
        # Episode tracking
        self.current_step = 0
        self.observation_buffer = np.zeros((history_length, self.n_channels), dtype=np.float32)
        self.last_action = None
        
        # Statistics tracking
        self.episode_collisions = 0
        self.episode_successes = 0
        self.episode_energy_cost = 0
        
        # Set random seed
        if seed is not None:
            self.seed(seed)
    
    def seed(self, seed: int) -> list:
        """Set random seed for reproducibility."""
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def _compute_channel_quality(self) -> np.ndarray:
        """Compute channel quality metrics based on recent occupancy patterns."""
        window_start = max(0, self.current_step - 10)
        window_end = self.current_step
        
        if window_end <= window_start:
            return 1.0 - self.ground_truth[self.current_step].astype(np.float32)
        
        # Channel quality = 1 - recent_occupancy_rate
        recent_window = self.ground_truth[window_start:window_end]
        channel_quality = 1.0 - np.mean(recent_window, axis=0).astype(np.float32)
        
        return channel_quality
    
    def _compute_collision_risk(self) -> np.ndarray:
        """Compute collision risk per channel based on historical performance."""
        total_attempts = self.channel_success_history + self.channel_collision_history + 1e-6
        collision_risk = self.channel_collision_history / total_attempts
        return collision_risk.astype(np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial observation (history_length x n_channels)
            info: Metadata dictionary
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Initialize observation buffer with first history_length timesteps
        self.observation_buffer = self.ground_truth[:self.history_length].astype(np.float32)
        
        # Reset tracking variables
        self.last_action = None
        self.episode_collisions = 0
        self.episode_successes = 0
        self.episode_energy_cost = 0
        
        # Initialize observation buffer with enhanced features
        if hasattr(self, 'use_enhanced_features') and self.use_enhanced_features:
            # EXPLORATION INCENTIVE: Bonus for using different channels
            unique_channels_used = np.count_nonzero(self.recent_channel_usage)
            exploration_bonus = unique_channels_used * 0.1  # Reward diversity
        
        # Reset tracking for new episode to prevent learning bad habits
        self.channel_success_history = np.zeros(self.n_channels)
        self.channel_collision_history = np.zeros(self.n_channels) 
        self.recent_channel_usage = np.zeros(self.n_channels)
        self.stuck_penalty_counter = 0
        
        info = {
            'episode_length': 0,
            'collision_rate': 0.0,
            'success_rate': 0.0,
        }
        
        # Return appropriate observation based on feature settings
        if hasattr(self, 'use_enhanced_features') and self.use_enhanced_features:
            enhanced_obs = self._get_enhanced_observation()
            return enhanced_obs, info
        else:
            return self.observation_buffer.copy(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Channel to transmit on (0 to n_channels-1)
        
        Returns:
            observation: Updated observation after action
            reward: Reward signal (higher is better)
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional information dictionary
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Get current ground truth (what's actually happening in the spectrum)
        current_timestep = self.history_length + self.current_step
        
        # Check if episode should end
        terminated = False
        truncated = False
        
        if current_timestep >= self.n_timesteps:
            terminated = True
        elif self.current_step >= self.max_episode_steps:
            truncated = True
        
        if not (terminated or truncated):
            ground_truth_occupancy = self.ground_truth[current_timestep]
            
            # Check for collision
            collision = bool(ground_truth_occupancy[action])
            
            # DOMAIN-AWARE COGNITIVE RADIO REWARD SYSTEM
            # Initialize tracking if needed
            if not hasattr(self, 'consecutive_collisions'):
                self.consecutive_collisions = 0
            if not hasattr(self, 'recent_actions'):
                self.recent_actions = []
            if not hasattr(self, 'channel_pattern_memory'):
                self.channel_pattern_memory = np.zeros(self.n_channels)
            
            # Base collision/success reward
            if collision:
                # Severe collision penalty with escalation
                reward = -10.0
                self.episode_collisions += 1
                self.channel_collision_history[action] += 1
                self.consecutive_collisions += 1
                
                # Exponential penalty for repeated collisions
                consecutive_penalty = -5.0 * (1.5 ** min(self.consecutive_collisions, 5))
                reward += consecutive_penalty
                
                # Channel-specific penalty (avoid persistently bad channels)
                collision_rate = self.channel_collision_history[action] / max(1, self.recent_channel_usage[action])
                if collision_rate > 0.3:  # >30% collision rate on this channel
                    reward -= 5.0
                    
            else:
                # Success reward with efficiency bonuses
                reward = +15.0
                self.episode_successes += 1
                self.channel_success_history[action] += 1
                self.consecutive_collisions = 0
                
                # Channel efficiency bonus
                success_rate = self.channel_success_history[action] / max(1, self.recent_channel_usage[action])
                if success_rate > 0.8:  # >80% success rate
                    reward += 3.0
                elif success_rate > 0.6:  # >60% success rate
                    reward += 1.0
            
            # Cognitive pattern recognition bonus
            self.channel_pattern_memory[action] = 0.9 * self.channel_pattern_memory[action] + 0.1 * (1.0 - float(collision))
            pattern_bonus = 2.0 * self.channel_pattern_memory[action] if not collision else 0.0
            reward += pattern_bonus
            
            # Track recent actions for diversity analysis
            self.recent_actions.append(int(action))
            if len(self.recent_actions) > 10:
                self.recent_actions.pop(0)
            
            # Exploration vs exploitation balance
            if len(self.recent_actions) >= 5:
                unique_actions = len(set(self.recent_actions[-5:]))
                if unique_actions >= 4:  # Good exploration
                    reward += 1.0
                elif unique_actions <= 2:  # Too repetitive
                    reward -= 2.0
            
            # Anti-oscillation penalty
            if len(self.recent_actions) >= 4:
                last_4 = self.recent_actions[-4:]
                if last_4[0] == last_4[2] and last_4[1] == last_4[3] and last_4[0] != last_4[1]:
                    reward -= 3.0  # Penalize A-B-A-B patterns
            
            # Update channel usage tracking
            self.recent_channel_usage[action] += 1
            
            # Smart energy penalty (only for unnecessary switches)
            if self.last_action is not None and self.last_action != action:
                # Only penalize if previous channel was actually good
                prev_success_rate = self.channel_success_history[self.last_action] / max(1, self.recent_channel_usage[self.last_action])
                if prev_success_rate > 0.7:  # Was doing well on previous channel
                    reward -= 1.0
                else:
                    reward += 0.5  # Good switch from bad channel
                self.episode_energy_cost += 1
            
            # Update observation buffer with enhanced features
            self.observation_buffer = np.roll(self.observation_buffer, shift=-1, axis=0)
            self.observation_buffer[-1] = ground_truth_occupancy.astype(np.float32)
            
            # Update state
            self.last_action = action
            self.current_step += 1
        else:
            reward = 0.0
            self.observation_buffer = np.zeros_like(self.observation_buffer, dtype=np.float32)
        
        # Prepare info dictionary
        total_transmissions = self.episode_collisions + self.episode_successes
        info = {
            'collision': collision if not (terminated or truncated) else False,
            'channel_selected': action,
            'episode_length': self.current_step,
            'collision_rate': self.episode_collisions / max(total_transmissions, 1),
            'success_rate': self.episode_successes / max(total_transmissions, 1),
            'energy_switches': self.episode_energy_cost,
        }
        
        # Return enhanced observation if enabled
        if self.use_enhanced_features:
            enhanced_obs = self._get_enhanced_observation()
            return enhanced_obs, reward, terminated, truncated, info
        else:
            return self.observation_buffer.copy(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get enhanced observation with spectrum history and channel intelligence."""
        # Current spectrum state
        current_spectrum = self.ground_truth[self.current_step].astype(np.float32)
        
        # Update observation buffer
        self.observation_buffer[:-1] = self.observation_buffer[1:]
        self.observation_buffer[-1] = current_spectrum
        
        if not self.use_enhanced_features:
            return self.observation_buffer.copy()
        
        # Enhanced features: channel quality and collision risk
        channel_quality = self._compute_channel_quality()
        collision_risk = self._compute_collision_risk()
        
        # Usage frequency normalization
        usage_norm = self.recent_channel_usage / (np.sum(self.recent_channel_usage) + 1e-6)
        
        # Flatten and combine all features
        spectrum_flat = self.observation_buffer.flatten()  # shape: (history_length * n_channels,)
        enhanced_features = np.concatenate([
            spectrum_flat,
            channel_quality,
            collision_risk,
            usage_norm
        ])
        
        return enhanced_features

    def _get_enhanced_observation(self) -> np.ndarray:
        """Create enhanced observation with spectrum + statistical features."""
        # Flatten spectrum history
        spectrum_features = self.observation_buffer.flatten()
        
        # Channel statistics (normalized)
        total_attempts = self.channel_success_history + self.channel_collision_history + 1e-8
        success_rates = self.channel_success_history / total_attempts
        collision_rates = self.channel_collision_history / total_attempts
        usage_rates = self.recent_channel_usage / (self.recent_channel_usage.sum() + 1e-8)
        
        # Combine all features
        enhanced_obs = np.concatenate([
            spectrum_features,      # Raw spectrum data
            success_rates,          # Historical success rates per channel
            collision_rates,        # Historical collision rates per channel  
            usage_rates            # Recent usage distribution
        ]).astype(np.float32)
        
        return enhanced_obs
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment (optional for visualization).
        
        For hackathon demo, this can be extended to show:
        - Current spectrum waterfall
        - Agent's selected channel (highlighted)
        - Collision markers
        """
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Last action: {self.last_action}")
            print(f"Collisions: {self.episode_collisions}")
            print(f"Successes: {self.episode_successes}")
            print(f"Current observation:")
            print(self.observation_buffer[-1])  # Show latest timestep
        
        return None
    
    def close(self):
        """Clean up resources."""
        pass


class RandomAgent:
    """
    Baseline random agent for comparison.
    
    Selects channels uniformly at random without learning.
    """
    
    def __init__(self, n_channels: int, seed: Optional[int] = None):
        self.n_channels = n_channels
        self.rng = np.random.RandomState(seed)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """Select random channel."""
        action = self.rng.randint(0, self.n_channels)
        return action, None


class GreedyAgent:
    """
    Greedy baseline agent that selects least-occupied channel from history.
    
    This provides a simple heuristic baseline for comparison.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        """
        Select channel with lowest average occupancy in observation history.
        
        Args:
            observation: (history_length, n_channels) binary occupancy matrix
        
        Returns:
            action: Channel index with minimum historical occupancy
        """
        channel_occupancy = observation.mean(axis=0)  # Average over time
        
        # Find channels with minimum occupancy
        min_occupancy = channel_occupancy.min()
        candidate_channels = np.where(channel_occupancy == min_occupancy)[0]
        
        # Break ties randomly
        action = self.rng.choice(candidate_channels)
        return action, None


if __name__ == "__main__":
    """
    Test the environment with a random agent.
    """
    print("🧪 Testing Cognitive Radio Environment\n")
    
    # Create environment
    env = CognitiveRadioEnv(
        data_path="data/generated/spectrum_train.npy",
        history_length=10,
        w_collision=10.0,
        w_throughput=1.0,
        w_energy=0.1,
        max_episode_steps=1000,
        seed=42
    )
    
    print(f"✅ Environment created successfully")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Dataset shape: {env.ground_truth.shape}")
    
    # Test with random agent
    print("\n🎲 Testing Random Agent (100 steps)...\n")
    agent = RandomAgent(n_channels=env.n_channels, seed=42)
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(100):
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"Step {step}: Action={action}, Reward={reward:.2f}, "
                  f"Collision Rate={info['collision_rate']:.2%}")
        
        if terminated or truncated:
            break
    
    print(f"\n📊 Episode Summary:")
    print(f"   - Total Reward: {total_reward:.2f}")
    print(f"   - Collision Rate: {info['collision_rate']:.2%}")
    print(f"   - Success Rate: {info['success_rate']:.2%}")
    print(f"   - Channel Switches: {info['energy_switches']}")
    
    # Test with greedy agent
    print("\n🎯 Testing Greedy Agent (100 steps)...\n")
    agent = GreedyAgent(seed=42)
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(100):
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"Step {step}: Action={action}, Reward={reward:.2f}, "
                  f"Collision Rate={info['collision_rate']:.2%}")
        
        if terminated or truncated:
            break
    
    print(f"\n📊 Episode Summary:")
    print(f"   - Total Reward: {total_reward:.2f}")
    print(f"   - Collision Rate: {info['collision_rate']:.2%}")
    print(f"   - Success Rate: {info['success_rate']:.2%}")
    print(f"   - Channel Switches: {info['energy_switches']}")
    
    print("\n✅ Environment test complete!")
