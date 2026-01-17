"""Quick CUDA test for training"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.train_agent import TrainingConfig

print("=" * 60)
print("CUDA Configuration Test")
print("=" * 60)

# Test direct torch
print("\n1. Direct PyTorch CUDA Test:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

# Test config
print("\n2. TrainingConfig Device Detection:")
config = TrainingConfig()
print(f"   Config device: {config.device}")

print("\n" + "=" * 60)
