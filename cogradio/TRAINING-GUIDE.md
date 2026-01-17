# 🚀 COMPLETE TRAINING WORKFLOW GUIDE

## 🎯 Overview

This guide explains how to train the **GA-RL Hybrid System** for 6G cognitive radio with **all loss curves, figures, and evaluation metrics** automatically generated.

---

## ✅ Quick Answer: How to Train

### **Option 1: One-Command Full Pipeline** ⭐ RECOMMENDED

```powershell
# This runs EVERYTHING: data → baseline → GA → evaluation → plots
python run_full_pipeline.py
```

**What it does:**
1. ✅ Generates datasets (if missing)
2. ✅ Trains baseline PPO
3. ✅ Runs GA optimization (10 generations)
4. ✅ Trains GA-optimized PPO
5. ✅ Evaluates both agents
6. ✅ Generates convergence plots
7. ✅ Saves all results to `logs/` and `data/generated/`

**Expected Time:**
- Full mode: ~2 hours
- Quick mode: ~15 minutes (`python run_full_pipeline.py --quick`)

---

### **Option 2: Step-by-Step Manual Control**

If you want to run each step separately:

#### **Step 1: Generate Datasets**
```powershell
python src/data_pipeline.py
```

**Output:**
- `data/generated/spectrum_train.npy` (10,000 × 20 binary occupancy matrix)
- `data/generated/spectrum_test.npy` (2,000 × 20, high load)
- `data/generated/data_verification_report.png` (4 subplots: arrival times, durations, spectrum waterfall, statistics)

**Time:** ~2 minutes

---

#### **Step 2: Train Baseline PPO (No GA)**
```powershell
python src/train_agent.py --timesteps 100000 --seed 42
```

**Output:**
- `models/ppo_cognitive_radio_YYYYMMDD_HHMMSS_final.zip`
- `models/ppo_cognitive_radio_YYYYMMDD_HHMMSS_config.json`
- `logs/tensorboard/PPO_1/events.out.tfevents.*` (TensorBoard logs)
- `logs/train/monitor.csv` (episode rewards)

**Logged Metrics (view in TensorBoard):**
- `rollout/ep_rew_mean`: Episode reward (moving average)
- `rollout/ep_len_mean`: Episode length
- `train/value_loss`: Value function loss
- `train/policy_gradient_loss`: Policy gradient loss
- `train/entropy_loss`: Entropy bonus (exploration)
- `train/approx_kl`: KL divergence (policy change magnitude)
- `train/clip_fraction`: Fraction of clipped policy updates

**Time:** ~30 minutes (100K steps)

---

#### **Step 3: Train GA-Optimized PPO**
```powershell
python src/train_agent.py --ga-optimize --ga-generations 10 --timesteps 100000 --seed 43
```

**What happens:**
1. **GA Phase** (~1 hour):
   - Population: 10 chromosomes (hyperparameter sets)
   - Each chromosome trains a mini-PPO for 10K steps
   - Fitness = negative collision rate
   - Best genes saved to `models/best_params.json`

2. **Full Training Phase** (~30 minutes):
   - Uses GA-optimized hyperparameters
   - Trains for full 100K steps
   - Auto-saves best model to `models/best/best_model.zip`

**Output:**
- `models/best_params.json` (learning_rate, gamma, w_collision, etc.)
- `models/best/best_model.zip` (best model based on eval reward)
- `logs/tensorboard/PPO_2/` (separate TensorBoard logs)
- `logs/eval/evaluations.npz` (evaluation results during training)

**Time:** ~1.5 hours total

---

#### **Step 4: Evaluate Both Agents**
```powershell
python src/evaluate.py --episodes 5
```

**What it tests:**
- Loads best model from `models/best/best_model.zip`
- Runs 5 episodes on test dataset (`spectrum_test.npy`)
- Compares against Random and Greedy baselines

**Output:**
- `data/generated/evaluation_results.png`:
  - 4 subplots: Collision Rate, Avg Reward, Success Rate, Energy Efficiency
  - Bar chart comparison with error bars
- `data/generated/evaluation_results.json`:
  ```json
  {
    "PPO Agent": {
      "mean_collision_rate": 16.48,
      "mean_reward": 9618.2,
      "mean_success_rate": 83.52
    },
    "Random Agent": {...},
    "Greedy Agent": {...}
  }
  ```

**Terminal Output:**
```
📊 PPO Agent (Trained) Performance
══════════════════════════════════════════════════════════════
  Avg Reward:        9618.20 ± 143.45
  Collision Rate:       16.48% ± 2.31%
  Success Rate:         83.52%
  Channel Switches:      127.4
  Episode Length:       5001
══════════════════════════════════════════════════════════════

🎯 Key Insights:
   - PPO reduces collisions by 72.5% vs Random
   - Collision rate: 16.48% (PPO) vs 60.04% (Random)
```

**Time:** ~5 minutes

---

#### **Step 5: Generate Convergence Plots**
```powershell
python src/plot_convergence.py
```

**What it plots:**
- **Panel A**: Episode Reward (raw + smoothed)
- **Panel B**: Episode Length (stability indicator)
- **Panel C**: Value Loss (log scale)
- **Panel D**: Policy Loss (convergence)

**Output:**
- `data/generated/convergence_comparison.png` (2×2 subplot figure)

**Time:** <1 minute

---

## 📊 Viewing Training Curves in Real-Time

### TensorBoard (Best for Real-Time Monitoring)
```powershell
tensorboard --logdir logs/tensorboard
```

Then open: **http://localhost:6006**

**Available Tabs:**
- **SCALARS**: All loss curves, rewards, metrics
- **GRAPHS**: Neural network architecture visualization
- **DISTRIBUTIONS**: Weight/gradient distributions
- **HISTOGRAMS**: Parameter evolution over time

**Key Plots to Monitor:**
1. `rollout/ep_rew_mean`: Should increase over time (convergence)
2. `train/value_loss`: Should decrease then stabilize
3. `train/approx_kl`: Should stay < 0.01 (stable policy updates)
4. `train/entropy_loss`: Should decrease slowly (exploration → exploitation)

---

## 🎨 All Generated Figures

After running the full pipeline, you'll have:

### 1. **Data Verification Report** (`data/generated/data_verification_report.png`)
- **Purpose**: Validate MMPP traffic model
- **Subplots**:
  - Top-left: Inter-arrival time distribution (exponential fit)
  - Top-right: Duration distribution (Pareto fit)
  - Bottom-left: Spectrum waterfall (heatmap showing burstiness)
  - Bottom-right: Statistics table (mean rates, occupancy %)

### 2. **Evaluation Results** (`data/generated/evaluation_results.png`)
- **Purpose**: Compare PPO vs Random vs Greedy
- **Subplots**:
  - A) Collision Rate (lower is better)
  - B) Average Reward (higher is better)
  - C) Success Rate (higher is better)
  - D) Energy Efficiency (channel switches, lower is better)

### 3. **Convergence Comparison** (`data/generated/convergence_comparison.png`)
- **Purpose**: Show GA-RL trains faster than baseline
- **Subplots**:
  - A) Episode Reward (raw + EMA smoothed)
  - B) Episode Length (stability)
  - C) Value Loss (log scale)
  - D) Policy Loss (gradient updates)
- **Annotation**: Final performance improvement % at bottom

### 4. **TensorBoard Logs** (view with `tensorboard --logdir logs/tensorboard`)
- Interactive plots of all metrics
- Zoomable, downloadable, shareable

---

## 🔧 Customizing Training

### Change Training Duration
```powershell
# Train longer (better convergence)
python src/train_agent.py --timesteps 200000

# Train shorter (quick test)
python src/train_agent.py --timesteps 50000
```

### Adjust GA Parameters
```powershell
# More GA generations (better hyperparameters, slower)
python src/train_agent.py --ga-optimize --ga-generations 20

# Fewer generations (faster, less optimal)
python src/train_agent.py --ga-optimize --ga-generations 5
```

### Override Reward Weights
```powershell
# Higher collision penalty (more conservative)
python src/train_agent.py --w-collision 20.0

# Higher throughput reward (more aggressive)
python src/train_agent.py --w-throughput 5.0

# Lower energy cost (more channel switching)
python src/train_agent.py --w-energy 0.01
```

### Resume from Checkpoint
```powershell
# Continue training from saved model
python src/train_agent.py --resume models/checkpoints/ppo_cognitive_radio_TIMESTAMP_100000_steps.zip --timesteps 50000
```

---

## 📁 File Organization

```
6g-cognitive-radio/
├── data/generated/          # All outputs
│   ├── spectrum_train.npy
│   ├── spectrum_test.npy
│   ├── data_verification_report.png
│   ├── evaluation_results.png
│   ├── evaluation_results.json
│   └── convergence_comparison.png
│
├── models/                   # Trained agents
│   ├── best/
│   │   └── best_model.zip   # Auto-saved best model
│   ├── checkpoints/         # Periodic checkpoints
│   │   └── ppo_*_steps.zip
│   ├── best_params.json     # GA-optimized hyperparameters
│   └── *_final.zip          # Final models from each run
│
└── logs/                     # Training logs
    ├── tensorboard/
    │   ├── PPO_1/           # Baseline run
    │   └── PPO_2/           # GA-optimized run
    ├── train/
    │   └── monitor.csv      # Episode-by-episode results
    └── eval/
        ├── evaluations.npz  # Evaluation during training
        └── monitor.csv
```

---

## 🎯 Next Steps After Training

### 1. Run Interactive Demo
```powershell
streamlit run app.py
```
- **Live spectrum waterfall visualization**
- **Agent decision-making in real-time**
- **Interactive controls** (device count, traffic load)

### 2. Analyze Results
```python
import json
with open('data/generated/evaluation_results.json') as f:
    results = json.load(f)

ppo_collision = results['PPO Agent']['mean_collision_rate']
random_collision = results['Random Agent']['mean_collision_rate']
improvement = (random_collision - ppo_collision) / random_collision * 100

print(f"PPO is {improvement:.1f}% better than Random!")
```

### 3. Export for Presentation
```powershell
# Copy key figures to presentation folder
Copy-Item data\generated\*.png docs\presentation\
```

---

## ❓ Troubleshooting

### "No module named 'tensorboard'"
```powershell
uv add tensorboard
```

### "CUDA out of memory"
```powershell
# Train on CPU instead
$env:CUDA_VISIBLE_DEVICES="-1"
python src/train_agent.py
```

### "Training not converging"
- Increase `--timesteps` to 200K
- Lower learning rate: `--learning-rate 0.0001`
- Increase entropy: `--ent-coef 0.1` (more exploration)

### "GA taking too long"
- Reduce generations: `--ga-generations 5`
- Use quick mode: `python run_full_pipeline.py --quick`

---

## 📚 Further Reading

- **Methodology**: See [docs/implementation/methodology.md](docs/implementation/methodology.md)
- **Strategy**: See [docs/strategy/ONE-PAGE-STRATEGY.md](docs/strategy/ONE-PAGE-STRATEGY.md)
- **Architecture**: See [.github/copilot-instructions.md](.github/copilot-instructions.md)

---

**Last Updated:** January 15, 2026  
**Support:** Open an issue on GitHub or check [todo.md](todo.md) for project status
