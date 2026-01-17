"""
Cognitive Radio Environment for RL Training

Gymnasium environment for training PPO agent on dynamic spectrum access.
This environment simulates a multi-channel cognitive radio scenario where
the agent must learn to select unoccupied channels.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class CognitiveRadioEnv(gym.Env):
    """
    Gymnasium environment for cognitive radio channel allocation.
    
    The agent observes channel occupancy over time and learns to select
    channels that minimize interference with primary users.
    
    Observation Space:
        Box of shape (history_length, n_channels) containing channel states
        over the past `history_length` timesteps.
        
    Action Space:
        Discrete(n_channels) - select which channel to use
        
    Rewards:
        +1.0: Selected an unoccupied channel
        -1.0: Selected an occupied channel (collision)
        +0.5: Stayed on same channel when it's still free
        -0.5: Failed to switch away from newly occupied channel
    
    Parameters
    ----------
    n_channels : int
        Number of spectrum channels (default: 20)
    history_length : int
        Number of past observations to include in state (default: 10)
    occupancy_prob : float
        Base probability of a channel being occupied (default: 0.3)
    transition_prob : float
        Probability of channel state changing each step (default: 0.1)
    episode_length : int
        Maximum steps per episode (default: 200)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        n_channels: int = 20,
        history_length: int = 10,
        occupancy_prob: float = 0.3,
        transition_prob: float = 0.1,
        episode_length: int = 200,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.history_length = history_length
        self.occupancy_prob = occupancy_prob
        self.transition_prob = transition_prob
        self.episode_length = episode_length
        self.render_mode = render_mode
        
        # Observation: history of channel states
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(history_length, n_channels),
            dtype=np.float32
        )
        
        # Action: select a channel
        self.action_space = spaces.Discrete(n_channels)
        
        # Internal state
        self.channel_states = None  # Current occupancy (0=free, 1=occupied)
        self.obs_buffer = None      # History buffer
        self.current_channel = None # Agent's current channel
        self.step_count = 0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize channel states randomly
        self.channel_states = (
            self.np_random.random(self.n_channels) < self.occupancy_prob
        ).astype(np.float32)
        
        # Initialize observation buffer with zeros
        self.obs_buffer = np.zeros(
            (self.history_length, self.n_channels),
            dtype=np.float32
        )
        
        # Start on a random channel
        self.current_channel = self.np_random.integers(0, self.n_channels)
        self.step_count = 0
        
        # Update buffer with initial observation
        self._update_observation()
        
        return self.obs_buffer.copy(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : int
            Channel to switch to (0 to n_channels-1)
            
        Returns
        -------
        observation : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        self.step_count += 1
        previous_channel = self.current_channel
        self.current_channel = action
        
        # Calculate reward based on action and channel state
        reward = self._calculate_reward(action, previous_channel)
        
        # Evolve channel states (Markov transition)
        self._evolve_channels()
        
        # Update observation buffer
        self._update_observation()
        
        # Check termination
        terminated = False
        truncated = self.step_count >= self.episode_length
        
        info = {
            "channel_states": self.channel_states.copy(),
            "selected_channel": action,
            "collision": self.channel_states[action] > 0.5
        }
        
        if self.render_mode == "human":
            self.render()
        
        return self.obs_buffer.copy(), reward, terminated, truncated, info
    
    def _calculate_reward(self, action: int, previous_channel: int) -> float:
        """Calculate reward based on action and channel state."""
        is_occupied = self.channel_states[action] > 0.5
        was_occupied = self.channel_states[previous_channel] > 0.5
        stayed = action == previous_channel
        
        if is_occupied:
            # Collision - bad
            return -1.0
        elif stayed and not was_occupied:
            # Stayed on free channel - good but encourage exploration
            return 0.5
        elif not stayed and was_occupied:
            # Successfully switched away from occupied channel - great
            return 1.0
        else:
            # Switched to free channel - good
            return 1.0
    
    def _evolve_channels(self) -> None:
        """Evolve channel states using Markov transition."""
        for i in range(self.n_channels):
            if self.np_random.random() < self.transition_prob:
                # Flip state with some probability
                if self.channel_states[i] > 0.5:
                    # Occupied -> Free (with higher prob)
                    if self.np_random.random() < 0.6:
                        self.channel_states[i] = 0.0
                else:
                    # Free -> Occupied
                    if self.np_random.random() < self.occupancy_prob:
                        self.channel_states[i] = 1.0
    
    def _update_observation(self) -> None:
        """Roll buffer and add current channel states."""
        self.obs_buffer = np.roll(self.obs_buffer, -1, axis=0)
        self.obs_buffer[-1] = self.channel_states
    
    def render(self) -> Optional[str]:
        """Render the environment state."""
        if self.render_mode == "ansi":
            lines = []
            lines.append(f"Step: {self.step_count}")
            lines.append(f"Current Channel: {self.current_channel}")
            lines.append("Channels: " + "".join(
                "█" if s > 0.5 else "░" for s in self.channel_states
            ))
            return "\n".join(lines)
        elif self.render_mode == "human":
            print(self.render())
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass


class MultiClassCognitiveRadioEnv(CognitiveRadioEnv):
    """
    Extended environment with multi-class channel states.
    
    Instead of binary (free/occupied), channels have modulation classes:
    - 0: Noise (free)
    - 1: FM
    - 2: BPSK
    - 3: QPSK
    
    Rewards are scaled based on interference severity.
    """
    
    def __init__(
        self,
        n_channels: int = 20,
        history_length: int = 10,
        n_classes: int = 4,
        episode_length: int = 200,
        render_mode: Optional[str] = None
    ):
        super().__init__(
            n_channels=n_channels,
            history_length=history_length,
            episode_length=episode_length,
            render_mode=render_mode
        )
        
        self.n_classes = n_classes
        
        # Override observation space for multi-class
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(history_length, n_channels),
            dtype=np.float32
        )
        
        # Class probabilities (Noise is most common)
        self.class_probs = [0.5, 0.2, 0.15, 0.15]  # Noise, FM, BPSK, QPSK
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with multi-class states."""
        super().reset(seed=seed)
        
        # Initialize with random classes
        self.channel_classes = self.np_random.choice(
            self.n_classes,
            size=self.n_channels,
            p=self.class_probs
        )
        
        # Normalize to [0, 1] for observation
        self.channel_states = self.channel_classes.astype(np.float32) / (self.n_classes - 1)
        
        self._update_observation()
        return self.obs_buffer.copy(), {}
    
    def _evolve_channels(self) -> None:
        """Evolve multi-class channel states."""
        for i in range(self.n_channels):
            if self.np_random.random() < self.transition_prob:
                self.channel_classes[i] = self.np_random.choice(
                    self.n_classes,
                    p=self.class_probs
                )
        
        self.channel_states = self.channel_classes.astype(np.float32) / (self.n_classes - 1)
    
    def _calculate_reward(self, action: int, previous_channel: int) -> float:
        """Reward based on class severity."""
        class_id = self.channel_classes[action]
        
        if class_id == 0:  # Noise (free)
            return 1.0
        elif class_id == 1:  # FM - moderate interference
            return -0.5
        else:  # BPSK, QPSK - severe interference
            return -1.0


# Register environments with Gymnasium
gym.register(
    id="CognitiveRadio-v0",
    entry_point="envs.cognitive_radio_env:CognitiveRadioEnv",
)

gym.register(
    id="CognitiveRadioMultiClass-v0",
    entry_point="envs.cognitive_radio_env:MultiClassCognitiveRadioEnv",
)
