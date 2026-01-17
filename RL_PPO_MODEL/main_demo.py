#!/usr/bin/env python3
"""
6G Cognitive Radio System - Main Demo

This script ties everything together for live demonstration.
Run in simulation mode to test logic, or live mode with RTL-SDR.

Usage:
    python main_demo.py              # Simulation mode
    python main_demo.py --live       # RTL-SDR hardware mode
    python main_demo.py --steps 100  # Limit to 100 steps
"""
import time
import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.hardware.smart_sensor import SmartSpectrumSensor
from src.hardware.rl_bridge import CognitiveBrain


def print_banner(live_mode: bool) -> None:
    """Print startup banner."""
    mode_str = "REAL HARDWARE (RTL-SDR)" if live_mode else "PHYSICS SIMULATION"
    print("=" * 60)
    print("🚀 6G COGNITIVE RADIO SYSTEM")
    print(f"📡 MODE: {mode_str}")
    print("=" * 60)
    print()
    print("Components:")
    print("  └── 👁️  SmartSpectrumSensor (AMC: CNN-LSTM)")
    print("  └── 🧠  CognitiveBrain (RL: PPO)")
    print("  └── 📊  Multi-class detection (Noise/FM/BPSK/QPSK)")
    print()
    print("-" * 60)


def run_demo(
    live_mode: bool = False,
    max_steps: int = None,
    amc_model_path: str = "models/amc_best.pth",
    rl_model_path: str = "models/rl_agent_best.zip"
) -> None:
    """
    Run the cognitive radio demo.
    
    Parameters
    ----------
    live_mode : bool
        Use RTL-SDR hardware if True
    max_steps : int, optional
        Maximum steps to run (None = infinite)
    amc_model_path : str
        Path to AMC model
    rl_model_path : str
        Path to RL agent model
    """
    print_banner(live_mode)
    
    # Initialize Components
    sensor = SmartSpectrumSensor(
        live_mode=live_mode, 
        model_path=amc_model_path
    )
    brain = CognitiveBrain(
        model_path=rl_model_path,
        n_channels=20
    )
    
    print("-" * 60)
    print("🎮 Starting cognitive radio loop...")
    print("   Press Ctrl+C to stop")
    print("-" * 60)
    
    # State tracking
    current_channel = 5  # Start on channel 5
    stats = {
        'total_scans': 0,
        'interference_detected': 0,
        'channel_switches': 0,
        'class_counts': {0: 0, 1: 0, 2: 0, 3: 0}
    }
    
    try:
        step = 0
        while max_steps is None or step < max_steps:
            step += 1
            stats['total_scans'] += 1
            
            # --- STEP 1: SENSING (Eyes) ---
            # In simulation mode, generate traffic pattern
            # In live mode, sim_ground_truth is ignored
            if not live_mode:
                # Simulate realistic traffic: mostly noise, occasional signals
                p = np.random.random()
                if p < 0.60:
                    sim_truth = 0  # 60% Noise
                elif p < 0.75:
                    sim_truth = 1  # 15% FM (Primary User)
                elif p < 0.90:
                    sim_truth = 2  # 15% BPSK (IoT)
                else:
                    sim_truth = 3  # 10% QPSK (Secondary User)
            else:
                sim_truth = None
            
            occupancy, class_id, class_name = sensor.scan(sim_ground_truth=sim_truth)
            stats['class_counts'][class_id] += 1
            
            # --- STEP 2: DECISION (Brain) ---
            recommended_channel = brain.decide(
                current_channel, 
                occupancy,
                class_id=class_id
            )
            
            # --- STEP 3: ACTUATION & LOGGING ---
            status_icons = {
                0: "🟢",  # Noise = Free
                1: "🔴",  # FM = Occupied (Primary)
                2: "🟡",  # BPSK = Occupied (IoT)
                3: "🟠"   # QPSK = Occupied (Secondary)
            }
            status_icon = status_icons.get(class_id, "⚪")
            
            print(f"\n[Step {step:4d}] Ch {current_channel:2d} │ {status_icon} {class_name:20s}", end="")
            
            if occupancy == 1:
                stats['interference_detected'] += 1
                if recommended_channel != current_channel:
                    stats['channel_switches'] += 1
                    print(f" │ ⚠️  → Switching to Ch {recommended_channel}")
                    current_channel = recommended_channel
                else:
                    print(f" │ 🔄 Staying (no better option)")
            else:
                print(f" │ ✅ Clear")
            
            # Rate limit for readability
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n")
        print("=" * 60)
        print("🛑 System Shutdown")
        print("=" * 60)
        print()
        print("📊 Session Statistics:")
        print(f"   └── Total Scans: {stats['total_scans']}")
        print(f"   └── Interference Detected: {stats['interference_detected']} ({100*stats['interference_detected']/max(1,stats['total_scans']):.1f}%)")
        print(f"   └── Channel Switches: {stats['channel_switches']}")
        print(f"   └── Class Distribution:")
        for cid, count in stats['class_counts'].items():
            pct = 100 * count / max(1, stats['total_scans'])
            names = ['Noise', 'FM', 'BPSK', 'QPSK']
            print(f"       └── {names[cid]}: {count} ({pct:.1f}%)")
        
        sensor.close()


def main():
    parser = argparse.ArgumentParser(
        description="6G Cognitive Radio Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_demo.py                    # Simulation mode (infinite)
  python main_demo.py --live             # RTL-SDR hardware mode
  python main_demo.py --steps 50         # Run 50 steps then stop
  python main_demo.py --live --steps 20  # 20 steps with hardware
        """
    )
    parser.add_argument(
        "--live", 
        action="store_true", 
        help="Use RTL-SDR hardware instead of simulation"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum steps to run (default: infinite)"
    )
    parser.add_argument(
        "--amc-model",
        type=str,
        default="models/amc_best.pth",
        help="Path to AMC model file"
    )
    parser.add_argument(
        "--rl-model",
        type=str,
        default="models/rl_agent_best.zip",
        help="Path to RL agent model file"
    )
    
    args = parser.parse_args()
    
    run_demo(
        live_mode=args.live,
        max_steps=args.steps,
        amc_model_path=args.amc_model,
        rl_model_path=args.rl_model
    )


if __name__ == "__main__":
    main()
