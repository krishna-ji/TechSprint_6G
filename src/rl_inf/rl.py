"""
RL Agent - PPO Inference

Reinforcement Learning agent for cognitive radio channel allocation.
Uses trained PPO model from stable-baselines3.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class RLAgent:
    """
    RL Agent for channel allocation using trained PPO policy.
    
    Maintains observation history and predicts optimal channel selection.
    
    Parameters
    ----------
    model_path : str | Path
        Path to trained PPO model (.zip file)
    n_channels : int
        Number of spectrum channels (default: 20)
    history_length : int
        Number of past observations to keep (default: 10)
    """
    
    def __init__(
        self,
        model_path: str | Path,
        n_channels: int = 20,
        history_length: int = 10
    ):
        self.model_path = Path(model_path)
        self.n_channels = n_channels
        self.history_length = history_length
        
        # Observation buffer: [history_length, n_channels]
        self.obs_buffer = np.zeros((history_length, n_channels), dtype=np.float32)
        
        # Load model
        self.model = None
        self.loaded = False
        self._load_model()
    
    def _load_model(self) -> None:
        """Load PPO model from file."""
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(str(self.model_path))
            self.loaded = True
            print(f" RL Agent loaded from {self.model_path}")
        except FileNotFoundError:
            print(f"⚠️  Model not found: {self.model_path}")
            print("   Using random action fallback")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            print("   Using random action fallback")
    
    def update_observation(
        self,
        channel_idx: int,
        is_occupied: int,
        class_id: Optional[int] = None
    ) -> None:
        """
        Update observation buffer with new sensing result.
        
        Parameters
        ----------
        channel_idx : int
            Channel that was scanned (0 to n_channels-1)
        is_occupied : int
            Binary occupancy (0=free, 1=occupied)
        class_id : int, optional
            Multi-class encoding (0-3 for modulation type)
        """
        new_snapshot = np.zeros(self.n_channels, dtype=np.float32)
        
        if class_id is not None:
            # Multi-class encoding normalized to [0, 1]
            new_snapshot[channel_idx] = class_id / 3.0
        else:
            new_snapshot[channel_idx] = float(is_occupied)
        
        # Roll buffer and add new observation
        self.obs_buffer = np.roll(self.obs_buffer, -1, axis=0)
        self.obs_buffer[-1] = new_snapshot
    
    def decide(
        self,
        current_channel: int,
        is_occupied: int,
        class_id: Optional[int] = None,
        deterministic: bool = True
    ) -> int:
        """
        Decide which channel to use based on current state.
        
        Parameters
        ----------
        current_channel : int
            Currently tuned channel
        is_occupied : int
            Binary occupancy of current channel
        class_id : int, optional
            Modulation class ID
        deterministic : bool
            If True, use greedy selection
        
        Returns
        -------
        int
            Recommended channel (0 to n_channels-1)
        """
        self.update_observation(current_channel, is_occupied, class_id)
        
        if self.loaded and self.model is not None:
            action, _ = self.model.predict(self.obs_buffer, deterministic=deterministic)
            return int(action)
        else:
            # Random fallback
            if is_occupied:
                available = [i for i in range(self.n_channels) if i != current_channel]
                return int(np.random.choice(available))
            return current_channel
    
    def get_observation(self) -> np.ndarray:
        """Get current observation buffer."""
        return self.obs_buffer.copy()
    
    def reset(self) -> None:
        """Reset observation buffer."""
        self.obs_buffer = np.zeros(
            (self.history_length, self.n_channels),
            dtype=np.float32
        )
    
    def get_action_probs(self, channel_idx: int, is_occupied: int) -> np.ndarray:
        """
        Get action probability distribution.
        
        Returns
        -------
        np.ndarray
            Probability for each channel [n_channels,]
        """
        if not self.loaded or self.model is None:
            return np.ones(self.n_channels) / self.n_channels
        
        self.update_observation(channel_idx, is_occupied)
        obs_tensor = self.model.policy.obs_to_tensor(self.obs_buffer)[0]
        distribution = self.model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy()
        return probs.flatten()
