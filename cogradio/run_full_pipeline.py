"""
Full Training Pipeline for 6G Cognitive Radio

Runs the complete workflow:
1. Generate datasets (if not exists)
2. Train baseline PPO agent
3. Run GA optimization and train GA-optimized agent
4. Evaluate both agents
5. Generate all comparison plots and figures

Usage:
    # Run full pipeline
    python run_full_pipeline.py

    # Quick mode (shorter training)
    python run_full_pipeline.py --quick

    # Skip data generation
    python run_full_pipeline.py --skip-data

    # Custom timesteps
    python run_full_pipeline.py --timesteps 200000
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


class PipelineRunner:
    """Orchestrates the full training and evaluation pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'timestamp': self.timestamp,
            'steps': {}
        }
    
    def run_command(self, cmd: list, step_name: str, critical: bool = True) -> bool:
        """
        Execute a command and track results.
        
        Args:
            cmd: Command and arguments as list
            step_name: Name for logging
            critical: If True, stop pipeline on failure
        
        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*70}")
        print(f"🚀 Step: {step_name}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                cwd=str(self.project_root),
                capture_output=False,
                text=True
            )
            
            self.results['steps'][step_name] = {
                'status': 'success',
                'returncode': result.returncode
            }
            
            print(f"\n✅ {step_name} completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            self.results['steps'][step_name] = {
                'status': 'failed',
                'returncode': e.returncode,
                'error': str(e)
            }
            
            print(f"\n❌ {step_name} failed with return code {e.returncode}")
            
            if critical:
                print(f"\n⚠️  Critical step failed. Stopping pipeline.")
                self.save_results()
                sys.exit(1)
            
            return False
    
    def save_results(self):
        """Save pipeline execution results."""
        results_path = self.project_root / f"logs/pipeline_run_{self.timestamp}.json"
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Pipeline results saved to: {results_path}")
    
    def run(self):
        """Execute full pipeline."""
        print("="*70)
        print("🎯 6G Cognitive Radio - Full Training Pipeline")
        print("="*70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Project Root: {self.project_root}")
        print(f"Quick Mode: {self.args.quick}")
        print("="*70)
        
        # Step 1: Generate datasets
        if not self.args.skip_data:
            data_train = self.project_root / "data/generated/spectrum_train.npy"
            data_test = self.project_root / "data/generated/spectrum_test.npy"
            
            if data_train.exists() and data_test.exists() and not self.args.regenerate_data:
                print("\n✅ Datasets already exist, skipping generation")
                print(f"   - {data_train}")
                print(f"   - {data_test}")
            else:
                self.run_command(
                    [sys.executable, "src/data_pipeline.py"],
                    "Dataset Generation",
                    critical=True
                )
        
        # Determine training timesteps
        timesteps = self.args.timesteps
        if self.args.quick:
            timesteps = 50_000
            print(f"\n⚡ Quick mode: Using {timesteps:,} timesteps")
        
        # Step 2: Train baseline PPO
        if not self.args.skip_baseline:
            baseline_cmd = [
                sys.executable, "src/train_agent.py",
                "--timesteps", str(timesteps),
                "--seed", "42"
            ]
            self.run_command(
                baseline_cmd,
                "Baseline PPO Training",
                critical=False  # Allow continuing even if baseline fails
            )
        
        # Step 3: Train GA-optimized PPO
        if not self.args.skip_ga:
            ga_generations = 5 if self.args.quick else 10
            
            ga_cmd = [
                sys.executable, "src/train_agent.py",
                "--ga-optimize",
                "--ga-generations", str(ga_generations),
                "--timesteps", str(timesteps),
                "--seed", "43"
            ]
            self.run_command(
                ga_cmd,
                "GA-Optimized PPO Training",
                critical=True
            )
        
        # Step 4: Evaluate trained agents
        if not self.args.skip_eval:
            eval_episodes = 3 if self.args.quick else 5
            
            eval_cmd = [
                sys.executable, "src/evaluate.py",
                "--episodes", str(eval_episodes)
            ]
            self.run_command(
                eval_cmd,
                "Agent Evaluation",
                critical=False
            )
        
        # Step 5: Generate convergence plots
        if not self.args.skip_plots:
            plot_cmd = [
                sys.executable, "src/plot_convergence.py",
                "--output", f"data/generated/convergence_{self.timestamp}.png"
            ]
            self.run_command(
                plot_cmd,
                "Convergence Plot Generation",
                critical=False
            )
        
        # Final summary
        self.save_results()
        
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETE!")
        print("="*70)
        
        successful = sum(1 for s in self.results['steps'].values() if s['status'] == 'success')
        total = len(self.results['steps'])
        
        print(f"\n📊 Summary: {successful}/{total} steps completed successfully")
        print("\n📁 Generated Files:")
        print(f"   - Models: models/")
        print(f"   - Logs: logs/")
        print(f"   - Plots: data/generated/")
        
        print("\n🎯 Next Steps:")
        print("   1. View training curves: tensorboard --logdir logs/tensorboard")
        print("   2. Check evaluation results: cat data/generated/evaluation_results.json")
        print("   3. Run interactive demo: streamlit run app.py")
        print("   4. Review convergence: open data/generated/convergence_*.png")


def main():
    """Main pipeline entry point."""
    parser = argparse.ArgumentParser(
        description="Run full 6G Cognitive Radio training pipeline"
    )
    
    # Pipeline control
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: reduced timesteps and generations'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100_000,
        help='Training timesteps per agent (default: 100k)'
    )
    
    # Step control
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip dataset generation'
    )
    parser.add_argument(
        '--regenerate-data',
        action='store_true',
        help='Force regenerate datasets even if they exist'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline PPO training'
    )
    parser.add_argument(
        '--skip-ga',
        action='store_true',
        help='Skip GA-optimized training'
    )
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip evaluation step'
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip convergence plot generation'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    runner = PipelineRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
