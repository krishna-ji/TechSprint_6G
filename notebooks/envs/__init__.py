"""
RL Training Environments

Gymnasium environments for training cognitive radio RL agents.
"""

from .cognitive_radio_env import CognitiveRadioEnv, MultiClassCognitiveRadioEnv

__all__ = ["CognitiveRadioEnv", "MultiClassCognitiveRadioEnv"]
