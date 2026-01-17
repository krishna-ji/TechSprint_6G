# 🚀 UV Command Reference

Quick reference for all `uv run` commands in the 6G Cognitive Radio project.

## 📋 Quick Start

```powershell
# See all available commands
uv run help

# Check project status
uv run status

# Run full pipeline (recommended first run)
uv run pipeline-quick
```

---

## 📊 Data Generation

### `uv run gen-data`
Generate IoT spectrum datasets using MMPP traffic model.

**Output:**
- `data/generated/spectrum_train.npy` (training dataset)
- `data/generated/spectrum_test.npy` (test dataset)
- `data/generated/data_verification_report.png`

**Time:** ~2 minutes

**Example:**
```powershell
uv run gen-data
```

---

## 🧠 Training Commands

### `uv run train-baseline`
Train baseline PPO agent without GA optimization.

**Output:**
- `models/*_final.zip` (trained model)
- `logs/tensorboard/PPO_*/` (training logs)

**Time:** ~30 minutes (100K steps)

**Examples:**
```powershell
# Default training
uv run train-baseline

# Custom timesteps
uv run train-baseline --timesteps 200000

# Custom hyperparameters
uv run train-baseline --learning-rate 0.0001 --gamma 0.995

# Resume from checkpoint
uv run train-baseline --resume models/checkpoints/ppo_*_steps.zip
```

---

### `uv run train-ga`
Train GA-optimized PPO agent (hybrid approach).

**What it does:**
1. Runs genetic algorithm to find optimal hyperparameters
2. Trains PPO with GA-optimized settings

**Output:**
- `models/best_params.json` (GA-optimized hyperparameters)
- `models/best/best_model.zip` (best model)
- `logs/tensorboard/PPO_*/`

**Time:** ~1.5 hours (10 GA generations + 100K PPO steps)

**Examples:**
```powershell
# Default GA training
uv run train-ga

# More GA generations (better optimization, slower)
uv run train-ga --ga-generations 20

# Fewer generations (faster test)
uv run train-ga --ga-generations 5

# Custom timesteps after GA
uv run train-ga --timesteps 200000
```

---

### `uv run train-quick`
Quick training test (50K steps, for debugging).

**Examples:**
```powershell
# Quick test run
uv run train-quick

# Quick test with custom seed
uv run train-quick --seed 123
```

---

## 📈 Evaluation & Analysis

### `uv run evaluate`
Evaluate trained agent against Random and Greedy baselines.

**Output:**
- `data/generated/evaluation_results.png` (comparison chart)
- `data/generated/evaluation_results.json` (metrics)

**Time:** ~5 minutes

**Examples:**
```powershell
# Evaluate best model
uv run evaluate

# Evaluate specific model
uv run evaluate --model models/ppo_cognitive_radio_20260115_091534_final.zip

# More evaluation episodes (more accurate)
uv run evaluate --episodes 10

# Skip baseline comparisons
uv run evaluate --no-baselines
```

---

### `uv run plot-convergence`
Generate convergence comparison plots.

**Output:**
- `data/generated/convergence_comparison.png`

**Examples:**
```powershell
# Generate convergence plots
uv run plot-convergence

# Custom output path
uv run plot-convergence --output docs/presentation/convergence.png

# Use TensorBoard logs instead of monitor.csv
uv run plot-convergence --tensorboard
```

---

## 🎯 Full Pipeline

### `uv run pipeline`
Run complete workflow: data → baseline → GA → eval → plots.

**Time:** ~2 hours

**Examples:**
```powershell
# Full pipeline
uv run pipeline

# Skip data generation (if already exists)
uv run pipeline --skip-data

# Skip baseline training
uv run pipeline --skip-baseline

# Custom timesteps
uv run pipeline --timesteps 200000
```

---

### `uv run pipeline-quick`
Quick pipeline test with reduced steps (~15 minutes).

**Examples:**
```powershell
# Quick test of entire pipeline
uv run pipeline-quick

# Quick test, skip data generation
uv run pipeline-quick --skip-data
```

---

## 📊 Monitoring & Demo

### `uv run tensorboard`
Launch TensorBoard to view training logs.

**View at:** http://localhost:6006

**Available metrics:**
- Episode reward (mean, std)
- Value loss, policy loss
- Entropy, KL divergence
- Learning rate, clip fraction

**Examples:**
```powershell
# Launch TensorBoard
uv run tensorboard
```

---

### `uv run demo`
Launch Streamlit interactive dashboard.

**View at:** http://localhost:8501

**Features:**
- Real-time spectrum waterfall
- Agent decision visualization
- Interactive controls (device count, traffic load)

**Examples:**
```powershell
# Launch demo
uv run demo
```

---

## 🛠️ Utilities

### `uv run status`
Show project status (datasets, models, logs, results).

**Examples:**
```powershell
# Check project status
uv run status
```

**Output:**
```
📊 Project Status

📁 Datasets:
  ✅ spectrum_train.npy (0.2 MB)
  ✅ spectrum_test.npy (0.0 MB)

🤖 Models:
  ✅ best/best_model.zip (2.0 MB)
  ✅ ppo_*_final.zip (0.4 MB)

📊 Training Logs:
  ✅ 3 training run(s)

📈 Results:
  ✅ evaluation_results.json
  ❌ convergence_comparison.png
```

---

### `uv run clean`
Remove caches and temporary files.

**Removes:**
- `__pycache__/` directories
- `*.pyc`, `*.pyo` files
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- `*.egg-info/` directories

**Examples:**
```powershell
# Clean project
uv run clean
```

---

### `uv run help`
Show available commands and usage examples.

**Examples:**
```powershell
# Show help
uv run help
```

---

## 📚 Common Workflows

### First-Time Setup
```powershell
# 1. Install dependencies
uv sync

# 2. Run quick pipeline test
uv run pipeline-quick

# 3. Check results
uv run status

# 4. Launch demo
uv run demo
```

---

### Production Training
```powershell
# 1. Generate datasets
uv run gen-data

# 2. Train baseline for comparison
uv run train-baseline --timesteps 100000

# 3. Train GA-optimized model
uv run train-ga --ga-generations 10 --timesteps 100000

# 4. Evaluate both
uv run evaluate --episodes 10

# 5. Generate plots
uv run plot-convergence

# 6. Monitor training
uv run tensorboard
```

---

### Quick Testing
```powershell
# Quick test cycle
uv run train-quick
uv run evaluate --episodes 3
uv run plot-convergence
```

---

### Debugging
```powershell
# Check what's available
uv run status

# Clean up before fresh run
uv run clean

# Quick test run
uv run pipeline-quick

# View logs
uv run tensorboard
```

---

## 🔧 Advanced Usage

### Passing Arguments
All commands accept standard Python arguments:

```powershell
# Training arguments
uv run train-baseline --timesteps 200000 --learning-rate 0.0001 --gamma 0.995

# Evaluation arguments
uv run evaluate --model models/best/best_model.zip --episodes 10 --data data/generated/spectrum_test.npy

# GA arguments
uv run train-ga --ga-generations 20 --timesteps 150000 --seed 42
```

---

### Environment Variables
```powershell
# Train on CPU only
$env:CUDA_VISIBLE_DEVICES="-1"
uv run train-baseline

# Disable TensorBoard logging
$env:DISABLE_TENSORBOARD="1"
uv run train-baseline
```

---

## ❓ Troubleshooting

### Command not found
```powershell
# Reinstall package with entry points
uv sync
```

### Module import errors
```powershell
# Ensure src/__init__.py exists
# Check that tool.uv.package = true in pyproject.toml
uv sync
```

### Training too slow
```powershell
# Use quick mode
uv run train-quick

# Or reduce timesteps
uv run train-baseline --timesteps 50000
```

### Out of memory
```powershell
# Train on CPU
$env:CUDA_VISIBLE_DEVICES="-1"
uv run train-baseline

# Reduce batch size (edit src/train_agent.py)
```

---

## 📁 File Locations

| Command            | Output Location                             |
| ------------------ | ------------------------------------------- |
| `gen-data`         | `data/generated/*.npy`                      |
| `train-*`          | `models/*.zip`, `logs/tensorboard/`         |
| `evaluate`         | `data/generated/evaluation_results.*`       |
| `plot-convergence` | `data/generated/convergence_comparison.png` |

---

## 📖 See Also

- **Full Training Guide:** [TRAINING-GUIDE.md](TRAINING-GUIDE.md)
- **Methodology:** [docs/implementation/methodology.md](docs/implementation/methodology.md)
- **Project Status:** [todo.md](todo.md)
- **Main README:** [README.md](README.md)

---

**Last Updated:** January 15, 2026
