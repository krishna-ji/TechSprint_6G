"""
6G Cognitive Radio Spectrum Allocation - Main Entry Point

Run this to generate IoT-optimized datasets and train the RL agent.
"""

from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Main entry point for the 6G cognitive radio project."""
    print("=" * 70)
    print("🚀 6G Cognitive Radio: IoT-Optimized Spectrum Allocation")
    print("=" * 70)
    print("\nAvailable commands:")
    print("  1. Generate dataset: python src/data_pipeline.py")
    print("  2. Train RL agent: (coming soon)")
    print("  3. Run evaluation: (coming soon)")
    print("\n💡 Quick Start:")
    print("  Run: python src/data_pipeline.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
