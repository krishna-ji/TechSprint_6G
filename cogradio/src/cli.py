"""
Command-line interface entry points for 6G Cognitive Radio.

These functions are registered as console scripts in pyproject.toml,
allowing them to be called directly from the command line after installation.
"""

import sys
import os
import subprocess
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    # Go up two levels from scripts/cli.py to reach project root
    return Path(__file__).parent.parent.absolute()


def run_python_script(script_path: str, args: list = None):
    """
    Run a Python script with optional arguments.
    
    Args:
        script_path: Relative path from project root
        args: Additional command-line arguments
    """
    root = get_project_root()
    script = root / script_path
    
    if not script.exists():
        print(f"❌ Error: Script not found: {script}")
        sys.exit(1)
    
    cmd = [sys.executable, str(script)]
    if args:
        cmd.extend(args)
    
    try:
        subprocess.run(cmd, check=True, cwd=str(root))
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


# ============================================================================
# Dataset Generation
# ============================================================================

def gen_data():
    """Generate IoT spectrum datasets using MMPP traffic model."""
    print("🚀 Generating datasets...")
    run_python_script("src/data_pipeline.py", sys.argv[1:])


# ============================================================================
# Training Commands
# ============================================================================

def train_baseline():
    """Train baseline PPO agent (without GA optimization)."""
    print("🧠 Training baseline PPO agent...")
    args = sys.argv[1:] if len(sys.argv) > 1 else ["--timesteps", "100000"]
    run_python_script("src/train_agent.py", args)


def train_ga():
    """Train GA-optimized PPO agent (hybrid approach)."""
    print("🧬 Training GA-optimized PPO agent...")
    args = ["--ga-optimize", "--ga-generations", "10"]
    args.extend(sys.argv[1:])
    run_python_script("src/train_agent.py", args)


def train_quick():
    """Quick training test (50K steps, useful for debugging)."""
    print("⚡ Quick training test...")
    args = ["--timesteps", "50000"]
    args.extend(sys.argv[1:])
    run_python_script("src/train_agent.py", args)


# ============================================================================
# Evaluation & Analysis
# ============================================================================

def evaluate():
    """Evaluate trained agent against baselines."""
    print("📊 Evaluating agents...")
    args = sys.argv[1:] if len(sys.argv) > 1 else ["--episodes", "5"]
    run_python_script("src/evaluate.py", args)


def plot_convergence():
    """Generate convergence comparison plots."""
    print("📈 Generating convergence plots...")
    run_python_script("src/plot_convergence.py", sys.argv[1:])


# ============================================================================
# Pipeline & Monitoring
# ============================================================================

def run_pipeline():
    """Run full training pipeline (data → baseline → GA → eval → plots)."""
    print("🎯 Running full pipeline...")
    run_python_script("run_full_pipeline.py", sys.argv[1:])


def run_pipeline_quick():
    """Run quick pipeline test (~15 minutes)."""
    print("⚡ Running quick pipeline...")
    run_python_script("run_full_pipeline.py", ["--quick"] + sys.argv[1:])


def tensorboard():
    """Launch TensorBoard to view training logs."""
    print("📊 Launching TensorBoard...")
    root = get_project_root()
    log_dir = root / "logs" / "tensorboard"
    
    if not log_dir.exists():
        print(f"❌ Error: TensorBoard logs not found at {log_dir}")
        print("   Run training first: uv run train-baseline")
        sys.exit(1)
    
    try:
        subprocess.run(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(log_dir)],
            check=True,
            cwd=str(root)
        )
    except subprocess.CalledProcessError:
        print("❌ TensorBoard failed to start. Install with: uv add tensorboard")
        sys.exit(1)


def demo():
    """Launch Streamlit demo dashboard."""
    print("🎬 Launching demo...")
    root = get_project_root()
    app_path = root / "app.py"
    
    if not app_path.exists():
        print(f"❌ Error: app.py not found")
        sys.exit(1)
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=True,
            cwd=str(root)
        )
    except subprocess.CalledProcessError:
        print("❌ Streamlit failed to start. Install with: uv add streamlit")
        sys.exit(1)


# ============================================================================
# Utility Commands
# ============================================================================

def clean():
    """Clean generated files and caches."""
    print("🧹 Cleaning project...")
    root = get_project_root()
    
    import shutil
    
    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    ]
    
    cleaned = 0
    for pattern in patterns:
        for path in root.rglob(pattern.replace("**/", "")):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  🗑️  Removed: {path.relative_to(root)}")
                cleaned += 1
            elif path.is_file():
                path.unlink()
                print(f"  🗑️  Removed: {path.relative_to(root)}")
                cleaned += 1
    
    print(f"\n✅ Cleaned {cleaned} items")


def status():
    """Show project status (models, logs, datasets)."""
    print("📊 Project Status\n")
    root = get_project_root()
    
    # Check datasets
    print("📁 Datasets:")
    data_dir = root / "data" / "generated"
    if data_dir.exists():
        datasets = list(data_dir.glob("*.npy"))
        if datasets:
            for ds in datasets:
                size_mb = ds.stat().st_size / (1024 * 1024)
                print(f"  ✅ {ds.name} ({size_mb:.1f} MB)")
        else:
            print("  ⚠️  No datasets found. Run: uv run gen-data")
    else:
        print("  ⚠️  data/generated/ not found")
    
    # Check models
    print("\n🤖 Models:")
    models_dir = root / "models"
    if models_dir.exists():
        models = list(models_dir.glob("*.zip")) + list(models_dir.glob("best/*.zip"))
        if models:
            for model in models[:5]:  # Show first 5
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"  ✅ {model.relative_to(models_dir)} ({size_mb:.1f} MB)")
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")
        else:
            print("  ⚠️  No models found. Run: uv run train-baseline")
    else:
        print("  ⚠️  models/ not found")
    
    # Check logs
    print("\n📊 Training Logs:")
    tb_dir = root / "logs" / "tensorboard"
    if tb_dir.exists():
        runs = list(tb_dir.glob("PPO_*"))
        if runs:
            print(f"  ✅ {len(runs)} training run(s)")
            print("     View with: uv run tensorboard")
        else:
            print("  ⚠️  No training logs found")
    else:
        print("  ⚠️  logs/tensorboard/ not found")
    
    # Check results
    print("\n📈 Results:")
    results = [
        "data/generated/evaluation_results.json",
        "data/generated/convergence_comparison.png",
        "models/best_params.json",
    ]
    for result_path in results:
        path = root / result_path
        if path.exists():
            print(f"  ✅ {result_path}")
        else:
            print(f"  ❌ {result_path}")
    
    print("\n" + "="*60)


def help_cmd():
    """Show available commands and usage examples."""
    print("""
🚀 6G Cognitive Radio - Available Commands

═══════════════════════════════════════════════════════════════

📊 DATA GENERATION
  uv run gen-data              Generate datasets using MMPP model

🧠 TRAINING
  uv run train-baseline        Train baseline PPO (no GA)
  uv run train-ga              Train GA-optimized PPO (hybrid)
  uv run train-quick           Quick test training (50K steps)

📈 EVALUATION & ANALYSIS  
  uv run evaluate              Evaluate trained agents
  uv run plot-convergence      Generate convergence plots

🎯 FULL PIPELINE
  uv run pipeline              Run complete workflow (~2 hours)
  uv run pipeline-quick        Quick pipeline test (~15 min)

📊 MONITORING & DEMO
  uv run tensorboard           Launch TensorBoard (view losses)
  uv run demo                  Launch Streamlit dashboard

🛠️ UTILITIES
  uv run status                Show project status
  uv run clean                 Remove caches and temp files
  uv run help                  Show this help message

═══════════════════════════════════════════════════════════════

💡 EXAMPLES:

  # Quick start (generate data + train + evaluate)
  uv run pipeline-quick

  # Train with custom settings
  uv run train-baseline --timesteps 200000 --learning-rate 0.0001

  # Evaluate specific model
  uv run evaluate --model models/best/best_model.zip --episodes 10

  # View training in real-time
  uv run tensorboard

  # Clean up before fresh run
  uv run clean

═══════════════════════════════════════════════════════════════

📚 Documentation:
  - Full guide: TRAINING-GUIDE.md
  - Methodology: docs/implementation/methodology.md
  - Status: todo.md

For more help: python <script>.py --help
    """)


# Entry point for main CLI
def main():
    """Main CLI dispatcher."""
    if len(sys.argv) < 2:
        help_cmd()
        return
    
    commands = {
        "gen-data": gen_data,
        "train-baseline": train_baseline,
        "train-ga": train_ga,
        "train-quick": train_quick,
        "evaluate": evaluate,
        "plot-convergence": plot_convergence,
        "pipeline": run_pipeline,
        "pipeline-quick": run_pipeline_quick,
        "tensorboard": tensorboard,
        "demo": demo,
        "status": status,
        "clean": clean,
        "help": help_cmd,
    }
    
    cmd = sys.argv[1]
    if cmd in commands:
        # Remove the command name from sys.argv
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        commands[cmd]()
    else:
        print(f"❌ Unknown command: {cmd}")
        print("Run 'uv run help' for available commands")
        sys.exit(1)
