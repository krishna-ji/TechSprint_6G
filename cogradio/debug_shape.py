#!/usr/bin/env python3
"""
Quick debug script to check observation shape issues.
"""
import numpy as np
from src.envs.cognitive_radio_env import CognitiveRadioEnv

def debug_observation_shapes():
    print("🔍 Debugging observation shapes...")
    
    # Create environment with same config as training
    env = CognitiveRadioEnv(
        data_path="data/generated/spectrum_train.npy",
        history_length=10,
        w_collision=15.0,
        w_throughput=8.0, 
        w_energy=0.1,
        max_episode_steps=2000,
        use_enhanced_features=False,  # Same as training config
        seed=42
    )
    
    print(f"Environment created successfully!")
    print(f"📊 Observation space: {env.observation_space}")
    print(f"📊 Observation shape: {env.observation_space.shape}")
    print(f"🎯 Action space: {env.action_space}")
    
    # Test reset and step
    obs, info = env.reset()
    print(f"📍 Reset observation shape: {obs.shape}")
    print(f"📍 Reset observation dtype: {obs.dtype}")
    
    action = 0  # Test with first channel
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"📍 Step observation shape: {obs.shape}")
    print(f"📍 Step observation dtype: {obs.dtype}")
    print(f"🎁 Step reward: {reward}")
    
    print("✅ Environment shape check completed!")

if __name__ == "__main__":
    debug_observation_shapes()