"""
RL Bridge (Cognitive Brain)

Connects the SmartSpectrumSensor output to the RL Agent's input.
Maintains observation history buffer and translates sensor readings
into the observation format expected by the PPO agent.
"""

import numpy as np
from typing import Optional, Tuple


class CognitiveBrain:
    """
    RL Agent wrapper that bridges sensor output to policy input.
    
    Maintains a sliding window of channel observations and uses
    the trained PPO model to decide which channel to use.
    
    Parameters
    ----------
    model_path : str
        Path to trained RL model (.zip file from stable-baselines3)
    n_channels : int
        Number of spectrum channels (default: 20)
    history_length : int
        Number of past observations to keep (default: 10)
    """
    
    def __init__(
        self,
        model_path: str,
        n_channels: int = 20,
        history_length: int = 10
    ):
        self.n_channels = n_channels
        self.history_length = history_length
        
        # Observation buffer: [history_length, n_channels]
        # Each row is a snapshot of all channel states at time t
        self.obs_buffer = np.zeros((history_length, n_channels), dtype=np.float32)
        
        # Load RL model
        print(f"🧠 Loading RL Agent from {model_path}...")
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            self.loaded = True
            print("   └── ✅ Model loaded successfully!")
        except FileNotFoundError:
            print(f"   └── ⚠️  Model not found at {model_path}")
            print("   └── 🎲 Using random action fallback")
            self.model = None
            self.loaded = False
        except Exception as e:
            print(f"   └── ⚠️  Load error: {e}")
            print("   └── 🎲 Using random action fallback")
            self.model = None
            self.loaded = False
    
    def update_observation(
        self, 
        channel_idx: int, 
        is_occupied: int,
        class_id: Optional[int] = None
    ) -> None:
        """
        Update the observation buffer with new sensing result.
        
        Parameters
        ----------
        channel_idx : int
            Channel that was just scanned (0 to n_channels-1)
        is_occupied : int
            Binary occupancy state (0 = free, 1 = occupied)
        class_id : int, optional
            If provided, use multi-class encoding instead of binary
        """
        # Create new snapshot (all zeros except scanned channel)
        new_snapshot = np.zeros(self.n_channels, dtype=np.float32)
        
        if class_id is not None:
            # Multi-class encoding (normalized to [0, 1])
            new_snapshot[channel_idx] = class_id / 3.0  # Assuming 4 classes (0-3)
        else:
            # Binary encoding
            new_snapshot[channel_idx] = float(is_occupied)
        
        # Roll buffer and add new snapshot
        self.obs_buffer = np.roll(self.obs_buffer, -1, axis=0)
        self.obs_buffer[-1] = new_snapshot
    
    def decide(
        self, 
        current_channel_idx: int, 
        is_occupied: int,
        class_id: Optional[int] = None,
        deterministic: bool = True
    ) -> int:
        """
        Make channel selection decision based on current state.
        
        Parameters
        ----------
        current_channel_idx : int
            Currently tuned channel
        is_occupied : int
            Binary occupancy of current channel
        class_id : int, optional
            Multi-class ID for enhanced state representation
        deterministic : bool
            If True, use greedy action selection. If False, sample from policy.
            
        Returns
        -------
        recommended_channel : int
            Channel ID to switch to (0 to n_channels-1)
        """
        # Update observation buffer
        self.update_observation(current_channel_idx, is_occupied, class_id)
        
        # Get action from policy
        if self.loaded and self.model is not None:
            action, _ = self.model.predict(self.obs_buffer, deterministic=deterministic)
            return int(action)
        else:
            # Random fallback (avoid current channel if occupied)
            if is_occupied:
                available = list(range(self.n_channels))
                available.remove(current_channel_idx)
                return np.random.choice(available)
            else:
                return current_channel_idx  # Stay on current channel
    
    def get_observation(self) -> np.ndarray:
        """
        Get current observation buffer.
        
        Returns
        -------
        obs : np.ndarray
            Current observation state [history_length, n_channels]
        """
        return self.obs_buffer.copy()
    
    def reset(self) -> None:
        """Reset observation buffer to zeros."""
        self.obs_buffer = np.zeros(
            (self.history_length, self.n_channels), 
            dtype=np.float32
        )
        print("🔄 Observation buffer reset")
    
    def get_action_probabilities(self, channel_idx: int, is_occupied: int) -> np.ndarray:
        """
        Get action probability distribution (for analysis/debugging).
        
        Returns
        -------
        probs : np.ndarray
            Probability of selecting each channel [n_channels,]
        """
        if not self.loaded or self.model is None:
            # Uniform distribution for random fallback
            return np.ones(self.n_channels) / self.n_channels
        
        # Update observation
        self.update_observation(channel_idx, is_occupied)
        
        # Get action distribution from policy
        obs_tensor = self.model.policy.obs_to_tensor(self.obs_buffer)[0]
        distribution = self.model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy()
        
        return probs.flatten()
