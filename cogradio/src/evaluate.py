"""
Evaluation Script for Cognitive Radio Agent

Evaluates trained PPO agent performance on test dataset and compares against baselines.
Calculates key performance indicators (KPIs) for hackathon presentation.

Usage:
    # Evaluate best model
    python src/evaluate.py

    # Evaluate specific model
    python src/evaluate.py --model models/ppo_cognitive_radio_20260114_120000_final.zip

    # Run extended evaluation
    python src/evaluate.py --episodes 10

KPIs Calculated:
    - Collision Rate (%)
    - Spectrum Utilization (%)
    - Average Reward
    - Throughput (successful transmissions)
    - Energy Efficiency (channel switches per success)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from src.envs.cognitive_radio_env import CognitiveRadioEnv, RandomAgent, GreedyAgent


class EvaluationResults:
    """Container for evaluation metrics."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.episode_rewards: List[float] = []
        self.collision_rates: List[float] = []
        self.success_rates: List[float] = []
        self.energy_switches: List[int] = []
        self.episode_lengths: List[int] = []
    
    def add_episode(self, total_reward: float, info: Dict):
        """Record episode results."""
        self.episode_rewards.append(total_reward)
        self.collision_rates.append(info['collision_rate'])
        self.success_rates.append(info['success_rate'])
        self.energy_switches.append(info['energy_switches'])
        self.episode_lengths.append(info['episode_length'])
    
    def get_summary(self) -> Dict:
        """Calculate summary statistics."""
        return {
            'agent': self.agent_name,
            'mean_reward': float(np.mean(self.episode_rewards)),
            'std_reward': float(np.std(self.episode_rewards)),
            'mean_collision_rate': float(np.mean(self.collision_rates) * 100),
            'std_collision_rate': float(np.std(self.collision_rates) * 100),
            'mean_success_rate': float(np.mean(self.success_rates) * 100),
            'mean_energy_switches': float(np.mean(self.energy_switches)),
            'mean_episode_length': float(np.mean(self.episode_lengths)),
        }
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"📊 {summary['agent']} Performance")
        print(f"{'='*60}")
        print(f"  Avg Reward:        {summary['mean_reward']:>10.2f} ± {summary['std_reward']:.2f}")
        print(f"  Collision Rate:    {summary['mean_collision_rate']:>10.2f}% ± {summary['std_collision_rate']:.2f}%")
        print(f"  Success Rate:      {summary['mean_success_rate']:>10.2f}%")
        print(f"  Channel Switches:  {summary['mean_energy_switches']:>10.1f}")
        print(f"  Episode Length:    {summary['mean_episode_length']:>10.0f}")
        print(f"{'='*60}\n")


def evaluate_agent(
    agent,
    env: CognitiveRadioEnv,
    n_episodes: int = 5,
    agent_name: str = "Agent",
    deterministic: bool = True
) -> EvaluationResults:
    """
    Evaluate an agent over multiple episodes.
    
    Args:
        agent: Agent to evaluate (PPO model or baseline)
        env: Cognitive Radio environment
        n_episodes: Number of evaluation episodes
        agent_name: Name for reporting
        deterministic: Use deterministic policy (for PPO)
    
    Returns:
        EvaluationResults object with statistics
    """
    results = EvaluationResults(agent_name)
    
    print(f"🧪 Evaluating {agent_name} over {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        results.add_episode(episode_reward, info)
        
        print(f"  Episode {episode + 1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Collision Rate={info['collision_rate']:.2%}")
    
    return results


def create_comparison_plot(results_dict: Dict[str, EvaluationResults], output_path: Path):
    """
    Create comparison visualization of all agents.
    
    Args:
        results_dict: Dictionary mapping agent names to EvaluationResults
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('6G Cognitive Radio: Agent Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    agents = list(results_dict.keys())
    colors = sns.color_palette("husl", len(agents))
    
    # Plot 1: Collision Rate
    ax = axes[0, 0]
    collision_data = [results_dict[agent].get_summary()['mean_collision_rate'] 
                      for agent in agents]
    collision_std = [results_dict[agent].get_summary()['std_collision_rate'] 
                     for agent in agents]
    bars = ax.bar(agents, collision_data, yerr=collision_std, capsize=5, color=colors, alpha=0.7)
    ax.set_ylabel('Collision Rate (%)', fontsize=12)
    ax.set_title('A) Collision Rate (Lower is Better)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, collision_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Average Reward
    ax = axes[0, 1]
    reward_data = [results_dict[agent].get_summary()['mean_reward'] 
                   for agent in agents]
    reward_std = [results_dict[agent].get_summary()['std_reward'] 
                  for agent in agents]
    bars = ax.bar(agents, reward_data, yerr=reward_std, capsize=5, color=colors, alpha=0.7)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('B) Average Reward (Higher is Better)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Success Rate
    ax = axes[1, 0]
    success_data = [results_dict[agent].get_summary()['mean_success_rate'] 
                    for agent in agents]
    bars = ax.bar(agents, success_data, color=colors, alpha=0.7)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('C) Successful Transmissions (Higher is Better)', 
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Energy Efficiency
    ax = axes[1, 1]
    energy_data = [results_dict[agent].get_summary()['mean_energy_switches'] 
                   for agent in agents]
    bars = ax.bar(agents, energy_data, color=colors, alpha=0.7)
    ax.set_ylabel('Channel Switches', fontsize=12)
    ax.set_title('D) Energy Efficiency (Lower is Better)', 
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {output_path}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Cognitive Radio Agent"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (default: models/best/best_model.zip)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of evaluation episodes (default: 5)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/generated/spectrum_test.npy',
        help='Path to test dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/generated/evaluation_results.png',
        help='Output path for comparison plot'
    )
    parser.add_argument(
        '--no-baselines',
        action='store_true',
        help='Skip baseline comparisons (only evaluate PPO agent)'
    )
    
    args = parser.parse_args()
    
    print("🎯 6G Cognitive Radio Agent Evaluation\n")
    
    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        # Default to best model
        model_path = Path("models/final_overhaul/model.zip")  # Use final model
        if not model_path.exists():
            # Try to find most recent model
            models_dir = Path("models")
            model_files = list(models_dir.glob("*_final.zip"))
            if model_files:
                model_path = max(model_files, key=lambda p: p.stat().st_mtime)
                print(f"📂 Using most recent model: {model_path}")
            else:
                print("❌ No trained model found. Please train a model first:")
                print("   python src/train_agent.py")
                return
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"✅ Loading model: {model_path}\n")
    
    # Create evaluation environment with GA-optimized simple features
    env = CognitiveRadioEnv(
        data_path=args.data,
        history_length=10,        # FIXED: Match training config exactly
        w_collision=15.0,         # FIXED: Match training config exactly
        w_throughput=8.0,         # FIXED: Match training config exactly
        w_energy=0.1,             # FIXED: Match training config exactly
        max_episode_steps=2000,   # Use full test dataset
        use_enhanced_features=False,  # REVERT: Enhanced features failed!
        seed=42
    )
    
    print(f"📡 Evaluation Environment:")
    print(f"   - Dataset: {args.data}")
    print(f"   - Shape: {env.ground_truth.shape}")
    print(f"   - Episodes: {args.episodes}\n")
    
    # Load trained agent
    ppo_agent = PPO.load(model_path)
    
    # Evaluate agents
    results = {}
    
    # Evaluate PPO agent
    results['PPO Agent'] = evaluate_agent(
        ppo_agent, env, args.episodes, "PPO Agent (Trained)", deterministic=True
    )
    
    # Evaluate baselines
    if not args.no_baselines:
        results['Random Agent'] = evaluate_agent(
            RandomAgent(env.n_channels, seed=42),
            env, args.episodes, "Random Baseline"
        )
        
        results['Greedy Agent'] = evaluate_agent(
            GreedyAgent(seed=42),
            env, args.episodes, "Greedy Baseline"
        )
    
    # Print summaries
    print("\n" + "="*60)
    print("📈 EVALUATION SUMMARY")
    print("="*60)
    
    for agent_name, result in results.items():
        result.print_summary()
    
    # Calculate improvement over random baseline
    if 'Random Agent' in results:
        ppo_collision = results['PPO Agent'].get_summary()['mean_collision_rate']
        random_collision = results['Random Agent'].get_summary()['mean_collision_rate']
        improvement = ((random_collision - ppo_collision) / random_collision) * 100
        
        print(f"\n🎯 Key Insights:")
        print(f"   - PPO reduces collisions by {improvement:.1f}% vs Random")
        print(f"   - Collision rate: {ppo_collision:.2f}% (PPO) vs {random_collision:.2f}% (Random)")
    
    # Create comparison plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_comparison_plot(results, output_path)
    
    # Save results to JSON
    json_path = output_path.with_suffix('.json')
    results_dict = {name: result.get_summary() for name, result in results.items()}
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"📄 Results saved to: {json_path}")
    
    print("\n✅ Evaluation Complete!")
    print("\n🎯 Next Steps:")
    print("   - Run demo: streamlit run app.py")
    print(f"   - View results: {output_path}")


if __name__ == "__main__":
    main()
