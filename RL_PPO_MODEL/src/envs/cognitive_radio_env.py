"""
Cognitive Radio Gymnasium Environment

A custom Gymnasium environment for training RL agents to perform
dynamic spectrum access in a 6G cognitive radio network.

Uses the multi-class occupancy grid from dataset_pipeline.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from pathlib import Path


class CognitiveRadioEnv(gym.Env):
    """
    Gymnasium environment for cognitive radio spectrum access.
    
    The agent observes spectrum occupancy over time and must decide
    which channel to transmit on to avoid interference.
    
    Observation Space:
        Box(0, 1, shape=(history_length, n_channels))
        - Normalized multi-class occupancy history
        
    Action Space:
        Discrete(n_channels)
        - Select which channel to transmit on
        
    Reward Structure:
        +10: Successful transmission (chose free channel)
        -100: Collision (chose occupied channel with Primary User)
        -50: Collision with Secondary User
        -10: Collision with IoT device
        +1: Staying on same free channel (stability bonus)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self,
        spectrum_data: Optional[np.ndarray] = None,
        data_path: Optional[str] = None,
        n_channels: int = 20,
        history_length: int = 10,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the cognitive radio environment.
        
        Parameters
        ----------
        spectrum_data : np.ndarray, optional
            Pre-loaded spectrum occupancy data [time_steps, n_channels]
            Values: 0=Noise, 1=FM, 2=BPSK, 3=QPSK
        data_path : str, optional
            Path to .npy file with spectrum data
        n_channels : int
            Number of spectrum channels
        history_length : int
            Number of past observations to include in state
        render_mode : str, optional
            Rendering mode ("human" or "ansi")
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.history_length = history_length
        self.render_mode = render_mode
        
        # Load spectrum data
        if spectrum_data is not None:
            self.spectrum_data = spectrum_data
        elif data_path is not None:
            self.spectrum_data = np.load(data_path)
        else:
            # Generate random data for testing
            print("⚠️  No data provided, generating random spectrum")
            self.spectrum_data = np.random.randint(0, 4, (10000, n_channels))
        
        self.total_steps = len(self.spectrum_data)
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(history_length, n_channels),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(n_channels)
        
        # State variables
        self.current_step = 0
        self.current_channel = 0
        self.obs_buffer = np.zeros((history_length, n_channels), dtype=np.float32)
        self.episode_rewards = []
        
        # Reward configuration
        self.rewards = {
            'success': 10.0,           # Chose free channel
            'collision_primary': -100.0,  # Hit Primary User (FM)
            'collision_secondary': -50.0, # Hit Secondary User (QPSK)
            'collision_iot': -10.0,       # Hit IoT device (BPSK)
            'stability_bonus': 1.0,       # Stayed on same free channel
            'switch_penalty': -0.5        # Unnecessary switch cost
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Returns
        -------
        observation : np.ndarray
            Initial observation
        info : dict
            Additional info
        """
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = self.np_random.integers(0, max(1, self.total_steps - 1000))
        self.current_channel = self.np_random.integers(0, self.n_channels)
        self.obs_buffer = np.zeros((self.history_length, self.n_channels), dtype=np.float32)
        self.episode_rewards = []
        
        # Fill initial observation buffer
        for i in range(self.history_length):
            step_idx = min(self.current_step + i, self.total_steps - 1)
            # Normalize class IDs to [0, 1]
            self.obs_buffer[i] = self.spectrum_data[step_idx] / 3.0
        
        self.current_step += self.history_length
        
        info = {
            'current_channel': self.current_channel,
            'step': self.current_step
        }
        
        return self.obs_buffer.copy(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : int
            Channel to transmit on (0 to n_channels-1)
            
        Returns
        -------
        observation : np.ndarray
            New observation
        reward : float
            Reward for this step
        terminated : bool
            Whether episode is done
        truncated : bool
            Whether episode was truncated
        info : dict
            Additional info
        """
        # Get ground truth for selected channel
        if self.current_step >= self.total_steps:
            # Episode done - ran out of data
            return self.obs_buffer.copy(), 0.0, True, False, {}
        
        ground_truth = self.spectrum_data[self.current_step, action]
        previous_channel = self.current_channel
        self.current_channel = action
        
        # Calculate reward based on collision type
        if ground_truth == 0:
            # FREE CHANNEL - Success!
            reward = self.rewards['success']
            if action == previous_channel:
                reward += self.rewards['stability_bonus']
            collision = False
            collision_type = None
        elif ground_truth == 1:
            # PRIMARY USER (FM) - Worst collision
            reward = self.rewards['collision_primary']
            collision = True
            collision_type = 'primary'
        elif ground_truth == 2:
            # IoT (BPSK) - Minor collision
            reward = self.rewards['collision_iot']
            collision = True
            collision_type = 'iot'
        else:  # ground_truth == 3
            # SECONDARY USER (QPSK) - Medium collision
            reward = self.rewards['collision_secondary']
            collision = True
            collision_type = 'secondary'
        
        # Penalize unnecessary switching
        if action != previous_channel and ground_truth != 0:
            reward += self.rewards['switch_penalty']
        
        self.episode_rewards.append(reward)
        
        # Update observation buffer
        self.obs_buffer = np.roll(self.obs_buffer, -1, axis=0)
        self.obs_buffer[-1] = self.spectrum_data[self.current_step] / 3.0
        
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= self.total_steps
        truncated = False
        
        info = {
            'collision': collision,
            'collision_type': collision_type,
            'ground_truth': int(ground_truth),
            'selected_channel': action,
            'step': self.current_step,
            'episode_reward': sum(self.episode_rewards)
        }
        
        return self.obs_buffer.copy(), reward, terminated, truncated, info
    
    def render(self) -> Optional[str]:
        """Render current state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
        return None
    
    def _render_ansi(self) -> str:
        """Generate ASCII representation of current state."""
        lines = []
        lines.append(f"Step: {self.current_step} | Channel: {self.current_channel}")
        lines.append("-" * 42)
        
        # Current spectrum state
        if self.current_step < self.total_steps:
            current_state = self.spectrum_data[self.current_step]
            symbols = {0: '·', 1: '█', 2: '▒', 3: '░'}
            state_str = ''.join(symbols.get(int(s), '?') for s in current_state)
            lines.append(f"Spectrum: [{state_str}]")
            lines.append(f"Legend: ·=Free █=FM ▒=BPSK ░=QPSK")
        
        return '\n'.join(lines)
    
    def get_optimal_action(self) -> int:
        """
        Get the optimal action for current state (oracle).
        Useful for debugging and baseline comparison.
        
        Returns
        -------
        optimal_channel : int
            Best channel to select (first free channel, or least bad)
        """
        if self.current_step >= self.total_steps:
            return 0
        
        current_state = self.spectrum_data[self.current_step]
        
        # Find free channels
        free_channels = np.where(current_state == 0)[0]
        if len(free_channels) > 0:
            # Prefer staying on current channel if it's free
            if self.current_channel in free_channels:
                return self.current_channel
            return int(free_channels[0])
        
        # No free channels - find least bad option (IoT > Secondary > Primary)
        iot_channels = np.where(current_state == 2)[0]
        if len(iot_channels) > 0:
            return int(iot_channels[0])
        
        secondary_channels = np.where(current_state == 3)[0]
        if len(secondary_channels) > 0:
            return int(secondary_channels[0])
        
        # All primary users - just pick channel 0
        return 0
