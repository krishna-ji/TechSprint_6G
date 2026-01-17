"""
6G Cognitive Radio Environments
"""
from gymnasium.envs.registration import register

register(
    id='CognitiveRadio-v0',
    entry_point='src.envs.cognitive_radio_env:CognitiveRadioEnv',
    max_episode_steps=10000,
)
