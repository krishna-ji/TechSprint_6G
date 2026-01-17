"""
RL Inference Module

Provides inference wrappers for:
- AMC (Automatic Modulation Classification) using ONNX
- RL (Reinforcement Learning) channel allocation using PPO
"""

from .amc import AMCClassifier
from .rl import RLAgent

__all__ = ["AMCClassifier", "RLAgent"]
