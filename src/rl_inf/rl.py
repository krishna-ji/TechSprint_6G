"""
RL Agent - PPO Inference

Reinforcement Learning agent for cognitive radio channel allocation.
Uses trained PPO model from stable-baselines3.

NOTE: The trained model expects a flat observation vector of shape (n_channels,)
representing current channel occupancy, NOT a history buffer.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class RLAgent:
    """
    RL Agent for channel allocation using trained PPO policy.
    
    Uses current channel occupancy state (not history) to predict optimal channel.
    
    Parameters
    ----------
    model_path : str | Path
        Path to trained PPO model (.zip file)
    n_channels : int
        Number of spectrum channels (default: 20)
    """
    
    def __init__(
        self,
        model_path: str | Path,
        n_channels: int = 20,
    ):
        self.model_path = Path(model_path)
        self.n_channels = n_channels
        
        # Observation: current channel occupancy [n_channels,]
        # Values normalized: 0=free, 0.33=FM, 0.67=BPSK, 1.0=QPSK
        self.obs = np.zeros(n_channels, dtype=np.float32)
        
        # Track channel states for spectrum hole detection
        self.channel_states = np.zeros(n_channels, dtype=np.float32)
        self.last_scanned_channel = 0
        
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
            # Verify observation space matches
            expected_shape = self.model.observation_space.shape
            print(f"✅ RL Model: Loaded from {self.model_path.name}")
            print(f"   └─ Policy: PPO | Obs shape: {expected_shape} | Actions: {self.model.action_space.n}")
        except FileNotFoundError:
            print(f"⚠️  RL Model: Not found at {self.model_path}")
            print("   └─ Using random action fallback")
        except Exception as e:
            print(f"⚠️  RL Model: Failed to load - {e}")
            print("   └─ Using random action fallback")
    
    @property
    def observation_space(self):
        """Get the model's observation space."""
        if self.model is not None:
            return self.model.observation_space
        return None
    
    @property
    def action_space(self):
        """Get the model's action space."""
        if self.model is not None:
            return self.model.action_space
        return None
    
    def update_observation(
        self,
        channel_idx: int,
        is_occupied: int,
        class_id: Optional[int] = None
    ) -> None:
        """
        Update observation with sensing result for a channel.
        
        Parameters
        ----------
        channel_idx : int
            Channel that was scanned (0 to n_channels-1)
        is_occupied : int
            Binary occupancy (0=free, 1=occupied)
        class_id : int, optional
            Multi-class encoding (0=Noise/Free, 1=FM, 2=BPSK, 3=QPSK)
        """
        if class_id is not None:
            # Normalize class ID to [0, 1] range as model expects
            # 0=Free, 1=FM, 2=BPSK, 3=QPSK -> 0.0, 0.33, 0.67, 1.0
            normalized_value = class_id / 3.0
        else:
            normalized_value = float(is_occupied)
        
        self.obs[channel_idx] = normalized_value
        self.channel_states[channel_idx] = normalized_value
        self.last_scanned_channel = channel_idx
    
    def decide(
        self,
        current_channel: int,
        is_occupied: int,
        class_id: Optional[int] = None,
        deterministic: bool = True
    ) -> int:
        """
        Decide which channel to use based on current spectrum state.
        
        Parameters
        ----------
        current_channel : int
            Currently tuned channel
        is_occupied : int
            Binary occupancy of current channel
        class_id : int, optional
            Modulation class ID (0=Noise, 1=FM, 2=BPSK, 3=QPSK)
        deterministic : bool
            If True, use greedy selection
        
        Returns
        -------
        int
            Recommended channel (0 to n_channels-1)
        """
        # Update observation for the scanned channel
        self.update_observation(current_channel, is_occupied, class_id)
        
        if self.loaded and self.model is not None:
            # Model expects shape (n_channels,) = (20,)
            action, _ = self.model.predict(self.obs, deterministic=deterministic)
            # Handle various output formats from predict
            if isinstance(action, np.ndarray):
                action = action.item() if action.ndim == 0 else action[0]
            return int(action)
        else:
            # Random fallback - prefer free channels
            return self._random_fallback(current_channel, is_occupied)
    
    def _random_fallback(self, current_channel: int, is_occupied: int) -> int:
        """Fallback when model not available - find spectrum holes."""
        if is_occupied:
            # Find free channels (value close to 0)
            free_channels = np.where(self.channel_states < 0.1)[0]
            if len(free_channels) > 0:
                return int(np.random.choice(free_channels))
            # No free channels found, try random
            available = [i for i in range(self.n_channels) if i != current_channel]
            return int(np.random.choice(available))
        return current_channel
    
    def get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        return self.obs.copy()
    
    def get_channel_states(self) -> np.ndarray:
        """Get current channel occupancy states for spectrum hole detection."""
        return self.channel_states.copy()
    
    def find_spectrum_holes(self, threshold: float = 0.1) -> np.ndarray:
        """
        Find free channels (spectrum holes).
        
        Parameters
        ----------
        threshold : float
            Channels with occupancy below this are considered free
        
        Returns
        -------
        np.ndarray
            Indices of free channels
        """
        return np.where(self.channel_states < threshold)[0]
    
    def reset(self) -> None:
        """Reset observation state."""
        self.obs = np.zeros(self.n_channels, dtype=np.float32)
        self.channel_states = np.zeros(self.n_channels, dtype=np.float32)
    
    def update_observations(self, channel_states: np.ndarray) -> None:
        """
        Update all channel observations at once.
        
        Parameters
        ----------
        channel_states : np.ndarray
            Full channel occupancy vector [n_channels,]
        """
        self.obs = np.asarray(channel_states, dtype=np.float32)
        self.channel_states = self.obs.copy()
    
    def get_action(
        self,
        channel_states: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[int, np.ndarray]:
        """
        Get recommended action given current spectrum state.
        
        Parameters
        ----------
        channel_states : np.ndarray, optional
            Full channel occupancy. If provided, updates internal state.
        deterministic : bool
            Use greedy action selection
        
        Returns
        -------
        Tuple[int, np.ndarray]
            (action, action_probabilities)
        """
        if channel_states is not None:
            self.update_observations(channel_states)
        
        if self.loaded and self.model is not None:
            action, _ = self.model.predict(self.obs, deterministic=deterministic)
            if isinstance(action, np.ndarray):
                action = action.item() if action.ndim == 0 else action[0]
            probs = self.get_action_probs_from_state()
            return int(action), probs
        else:
            probs = np.ones(self.n_channels) / self.n_channels
            return int(np.random.choice(self.n_channels)), probs
    
    def get_spectrum_holes(self, threshold: float = 0.5) -> np.ndarray:
        """
        Find free channels based on current observations.
        
        Parameters
        ----------
        threshold : float
            Channels below this value considered free
        
        Returns
        -------
        np.ndarray
            Indices of free channels
        """
        return np.where(self.channel_states < threshold)[0]
    
    def get_best_free_channel(self, threshold: float = 0.5) -> Optional[int]:
        """
        Get best free channel according to policy preferences.
        
        Returns
        -------
        Optional[int]
            Best free channel, or None if all occupied
        """
        holes = self.get_spectrum_holes(threshold)
        if len(holes) == 0:
            return None
        
        if self.loaded and self.model is not None:
            # Get action probs and pick highest-rated free channel
            probs = self.get_action_probs_from_state()
            hole_probs = probs[holes]
            return int(holes[np.argmax(hole_probs)])
        else:
            return int(holes[0])
    
    def get_action_probs_from_state(self) -> np.ndarray:
        """Get action probabilities from current observation state."""
        if not self.loaded or self.model is None:
            return np.ones(self.n_channels) / self.n_channels
        
        obs_tensor = self.model.policy.obs_to_tensor(self.obs)[0]
        distribution = self.model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy()
        return probs.flatten()
    
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
        obs_tensor = self.model.policy.obs_to_tensor(self.obs)[0]
        distribution = self.model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.detach().cpu().numpy()
        return probs.flatten()
