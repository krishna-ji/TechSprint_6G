"""
Convergence Plot Generator for GA-RL Comparison

Reads training logs from multiple runs and generates publication-quality
convergence plots comparing:
- Baseline PPO (default hyperparameters)
- GA-Optimized PPO (hybrid approach)

Usage:
    # Compare all runs in logs/
    python src/plot_convergence.py

    # Compare specific runs
    python src/plot_convergence.py --baseline logs/train_baseline --ga logs/train_ga

    # Custom output path
    python src/plot_convergence.py --output docs/presentation/convergence.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_monitor_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse Stable-Baselines3 Monitor CSV file.
    
    Args:
        csv_path: Path to monitor.csv
    
    Returns:
        Tuple of (timesteps, episode_rewards)
    """
    data = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()[2:]  # Skip header lines
        
        for line in lines:
            if line.strip():
                parts = line.strip().split(',')
                reward = float(parts[0])
                length = int(parts[1])
                data.append((reward, length))
    
    if not data:
        return np.array([]), np.array([])
    
    rewards, lengths = zip(*data)
    timesteps = np.cumsum(lengths)
    
    return np.array(timesteps), np.array(rewards)


def parse_tensorboard_events(log_dir: Path, tag: str = 'rollout/ep_rew_mean') -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse TensorBoard event files for a specific metric.
    
    Args:
        log_dir: Directory containing TensorBoard event files
        tag: Metric tag to extract (default: episode reward mean)
    
    Returns:
        Tuple of (steps, values)
    """
    event_files = list(log_dir.glob('events.out.tfevents.*'))
    if not event_files:
        return np.array([]), np.array([])
    
    # Use the most recent event file
    event_file = max(event_files, key=lambda p: p.stat().st_mtime)
    
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']:
        print(f"⚠️  Warning: Tag '{tag}' not found in {event_file}")
        return np.array([]), np.array([])
    
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    
    return steps, values


def smooth_curve(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """
    Apply exponential moving average smoothing.
    
    Args:
        values: Raw values
        weight: Smoothing factor (0-1, higher = smoother)
    
    Returns:
        Smoothed values
    """
    smoothed = []
    last = values[0]
    
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    
    return np.array(smoothed)


def create_convergence_plot(
    baseline_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ga_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path
):
    """
    Create comprehensive convergence comparison plot.
    
    Args:
        baseline_data: Dict of metric_name -> (steps, values) for baseline
        ga_data: Dict of metric_name -> (steps, values) for GA-optimized
        output_path: Where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('GA-RL Hybrid vs Baseline PPO: Training Convergence', 
                 fontsize=16, fontweight='bold')
    
    colors = {'baseline': '#E74C3C', 'ga': '#2ECC71'}
    
    # Plot 1: Episode Reward
    ax = axes[0, 0]
    if 'reward' in baseline_data and len(baseline_data['reward'][0]) > 0:
        steps, rewards = baseline_data['reward']
        smoothed = smooth_curve(rewards)
        ax.plot(steps, rewards, alpha=0.2, color=colors['baseline'])
        ax.plot(steps, smoothed, label='Baseline PPO', linewidth=2, color=colors['baseline'])
    
    if 'reward' in ga_data and len(ga_data['reward'][0]) > 0:
        steps, rewards = ga_data['reward']
        smoothed = smooth_curve(rewards)
        ax.plot(steps, rewards, alpha=0.2, color=colors['ga'])
        ax.plot(steps, smoothed, label='GA-Optimized PPO', linewidth=2, color=colors['ga'])
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('A) Episode Reward Convergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 2: Episode Length (proxy for stability)
    ax = axes[0, 1]
    if 'length' in baseline_data and len(baseline_data['length'][0]) > 0:
        steps, lengths = baseline_data['length']
        smoothed = smooth_curve(lengths)
        ax.plot(steps, smoothed, label='Baseline PPO', linewidth=2, color=colors['baseline'])
    
    if 'length' in ga_data and len(ga_data['length'][0]) > 0:
        steps, lengths = ga_data['length']
        smoothed = smooth_curve(lengths)
        ax.plot(steps, smoothed, label='GA-Optimized PPO', linewidth=2, color=colors['ga'])
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Length', fontsize=12)
    ax.set_title('B) Episode Length (Stability)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 3: Value Loss (from TensorBoard)
    ax = axes[1, 0]
    if 'value_loss' in baseline_data and len(baseline_data['value_loss'][0]) > 0:
        steps, loss = baseline_data['value_loss']
        smoothed = smooth_curve(loss, weight=0.95)
        ax.plot(steps, smoothed, label='Baseline PPO', linewidth=2, color=colors['baseline'])
    
    if 'value_loss' in ga_data and len(ga_data['value_loss'][0]) > 0:
        steps, loss = ga_data['value_loss']
        smoothed = smooth_curve(loss, weight=0.95)
        ax.plot(steps, smoothed, label='GA-Optimized PPO', linewidth=2, color=colors['ga'])
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Value Loss', fontsize=12)
    ax.set_title('C) Value Function Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Policy Loss
    ax = axes[1, 1]
    if 'policy_loss' in baseline_data and len(baseline_data['policy_loss'][0]) > 0:
        steps, loss = baseline_data['policy_loss']
        smoothed = smooth_curve(loss, weight=0.95)
        ax.plot(steps, smoothed, label='Baseline PPO', linewidth=2, color=colors['baseline'])
    
    if 'policy_loss' in ga_data and len(ga_data['policy_loss'][0]) > 0:
        steps, loss = ga_data['policy_loss']
        smoothed = smooth_curve(loss, weight=0.95)
        ax.plot(steps, smoothed, label='GA-Optimized PPO', linewidth=2, color=colors['ga'])
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Policy Loss', fontsize=12)
    ax.set_title('D) Policy Gradient Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add performance annotations
    if 'reward' in baseline_data and 'reward' in ga_data:
        baseline_final = smooth_curve(baseline_data['reward'][1])[-1]
        ga_final = smooth_curve(ga_data['reward'][1])[-1]
        improvement = ((ga_final - baseline_final) / abs(baseline_final)) * 100
        
        fig.text(0.5, 0.02, 
                 f'Final Performance: GA-Optimized = {ga_final:.1f}, Baseline = {baseline_final:.1f} '
                 f'({improvement:+.1f}% improvement)',
                 ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Convergence plot saved to: {output_path}")


def main():
    """Main convergence plotting entry point."""
    parser = argparse.ArgumentParser(
        description="Generate convergence comparison plots"
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        default='logs/train',
        help='Path to baseline training logs (default: logs/train)'
    )
    parser.add_argument(
        '--ga',
        type=str,
        default='logs/train',
        help='Path to GA-optimized training logs (default: logs/train)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/generated/convergence_comparison.png',
        help='Output path for convergence plot'
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Use TensorBoard logs instead of monitor.csv'
    )
    
    args = parser.parse_args()
    
    print("📊 Convergence Plot Generator\n")
    
    baseline_dir = Path(args.baseline)
    ga_dir = Path(args.ga)
    
    # Load baseline data
    print(f"📂 Loading baseline data from: {baseline_dir}")
    baseline_data = {}
    
    if args.tensorboard:
        # Try to load from TensorBoard
        tb_dirs = list(baseline_dir.parent.glob('tensorboard/*'))
        if tb_dirs:
            steps, rewards = parse_tensorboard_events(tb_dirs[0], 'rollout/ep_rew_mean')
            if len(steps) > 0:
                baseline_data['reward'] = (steps, rewards)
            
            steps, loss = parse_tensorboard_events(tb_dirs[0], 'train/value_loss')
            if len(steps) > 0:
                baseline_data['value_loss'] = (steps, loss)
            
            steps, loss = parse_tensorboard_events(tb_dirs[0], 'train/policy_gradient_loss')
            if len(steps) > 0:
                baseline_data['policy_loss'] = (steps, loss)
    else:
        # Load from monitor.csv
        monitor_path = baseline_dir / 'monitor.csv'
        if monitor_path.exists():
            steps, rewards = parse_monitor_csv(monitor_path)
            baseline_data['reward'] = (steps, rewards)
            baseline_data['length'] = (steps, np.ones_like(rewards) * 5001)  # Placeholder
            print(f"   ✅ Loaded {len(rewards)} episodes")
        else:
            print(f"   ⚠️  No monitor.csv found at {monitor_path}")
    
    # Load GA data (for now, same as baseline - will be different after GA run)
    print(f"📂 Loading GA-optimized data from: {ga_dir}")
    ga_data = {}
    
    if args.tensorboard:
        tb_dirs = list(ga_dir.parent.glob('tensorboard/*'))
        if tb_dirs and len(tb_dirs) > 1:
            # Use second run as GA run
            steps, rewards = parse_tensorboard_events(tb_dirs[1], 'rollout/ep_rew_mean')
            if len(steps) > 0:
                ga_data['reward'] = (steps, rewards)
    else:
        monitor_path = ga_dir / 'monitor.csv'
        if monitor_path.exists():
            steps, rewards = parse_monitor_csv(monitor_path)
            ga_data['reward'] = (steps, rewards)
            ga_data['length'] = (steps, np.ones_like(rewards) * 5001)
            print(f"   ✅ Loaded {len(rewards)} episodes")
    
    # If no separate GA data, use baseline as placeholder
    if not ga_data and baseline_data:
        print("   ℹ️  Using baseline data as placeholder (run --ga-optimize to generate real comparison)")
        ga_data = baseline_data.copy()
    
    # Generate plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_convergence_plot(baseline_data, ga_data, output_path)
    
    print("\n✅ Convergence analysis complete!")


if __name__ == "__main__":
    main()
