#!/usr/bin/env python3
"""
Quick script to compare training and test datasets.
"""
import numpy as np

def compare_datasets():
    print("🔍 Comparing training vs test datasets...")
    
    # Load both datasets
    train_data = np.load("data/generated/spectrum_train.npy")
    test_data = np.load("data/generated/spectrum_test.npy")
    
    print(f"📊 Training data shape: {train_data.shape}")
    print(f"📊 Test data shape: {test_data.shape}")
    
    # Calculate occupancy statistics
    train_occupancy = np.mean(train_data)
    test_occupancy = np.mean(test_data)
    
    print(f"🎯 Training occupancy rate: {train_occupancy:.3f} ({train_occupancy*100:.1f}%)")
    print(f"🎯 Test occupancy rate: {test_occupancy:.3f} ({test_occupancy*100:.1f}%)")
    
    # Channel-wise occupancy
    train_channel_occ = np.mean(train_data, axis=0)
    test_channel_occ = np.mean(test_data, axis=0)
    
    print(f"🔢 Training channel occupancy (min/max/std): {train_channel_occ.min():.3f}/{train_channel_occ.max():.3f}/{train_channel_occ.std():.3f}")
    print(f"🔢 Test channel occupancy (min/max/std): {test_channel_occ.min():.3f}/{test_channel_occ.max():.3f}/{test_channel_occ.std():.3f}")
    
    # Temporal patterns
    train_temporal = np.mean(train_data, axis=1)
    test_temporal = np.mean(test_data, axis=1)
    
    print(f"⏰ Training temporal occupancy (min/max/std): {train_temporal.min():.3f}/{train_temporal.max():.3f}/{train_temporal.std():.3f}")
    print(f"⏰ Test temporal occupancy (min/max/std): {test_temporal.min():.3f}/{test_temporal.max():.3f}/{test_temporal.std():.3f}")
    
    # Overall difference
    diff = np.abs(train_data.astype(float) - test_data.astype(float))
    print(f"🔄 Dataset difference: {np.mean(diff):.3f} (0=identical, 1=completely different)")
    
    if np.mean(diff) > 0.1:
        print("❌ SIGNIFICANT DIFFERENCE between train and test datasets!")
    else:
        print("✅ Train and test datasets are similar")

if __name__ == "__main__":
    compare_datasets()