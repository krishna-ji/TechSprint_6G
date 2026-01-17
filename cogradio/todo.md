# 📝 Project To-Do List: 6G Cognitive Radio

## 🚀 Phase 1: Foundation & Data (Day 1)

### 1.1 Project Setup

- [x] Initialize Git repository (if not already done).
- [x] Create virtual environment (`python -m venv venv`).
- [x] Install dependencies: `numpy`, `pandas`, `gymnasium`, `stable-baselines3`, `pygad`, `matplotlib`, `seaborn`, `streamlit`.
- [x] Create folder structure: `src/`, `data/`, `notebooks/`, `models/`, `logs/`.

### 1.2 Dataset Generation (The "Digital Twin")

- [x] Implement `src/data_pipeline.py`:
  - [x] Create `MMPP_Traffic` class using Poisson arrivals and Pareto durations.
  - [x] Implement `generate_grid(time_steps, channels)` function.
  - [x] Add visualization function to verify "bursty" traffic patterns.
  - [x] Generate and save `data/generated/spectrum_train.npy` (Medium Load).
  - [x] Generate and save `data/generated/spectrum_test.npy` (Heavy Load).

---

## 🏗️ Phase 2: Core Implementation (Day 1-2)

### 2.1 RL Environment (Gymnasium)

- [x] Create `src/envs/cognitive_radio_env.py`:
  - [x] Subclass `gym.Env`.
  - [x] Define `observation_space`: Box(0, 1, shape=(History, Channels)).
  - [x] Define `action_space`: Discrete(Channels).
  - [x] Implement `step()` logic:
    - [x] Check collision with Ground Truth data.
    - [x] Calculate Reward (configurable weights).
    - [x] Update State.
  - [x] Implement `reset()` logic.
  - [x] Include `RandomAgent` and `GreedyAgent` baselines.

### 2.2 Genetic Algorithm (The Optimizer)

- [x] Create `src/ga_optimizer.py`:
  - [x] Define `fitness_func(solution, solution_idx)`:
    - [x] Instantiate `CognitiveRadioEnv` with parameters from `solution`.
    - [x] Train a temp PPO agent for ~500-1000 steps.
    - [x] Return collision rate as fitness (lower is better).
  - [x] Configure `pygad.GA` (population size, num_generations, gene_space).
  - [x] Run GA to find "Best Genes" (Best Reward Weights & Hyperparams).
  - [x] Save best parameters to `models/best_params.json` (generated during GA run).

---

## 🧠 Phase 3: Training & Intelligence (Day 2)

### 3.1 PPO Agent Training

- [x] Create `src/train_agent.py`:
  - [x] Load GA-optimized parameters (optional).
  - [x] Initialize `CognitiveRadioEnv` (Train Mode).
  - [x] Initialize `PPO` from Stable-Baselines3 with custom network architecture.
  - [x] Train for full duration (e.g., 100k steps).
  - [x] Save model to `models/ppo_cognitive_radio_*.zip`.
  - [x] Implement TensorBoard logging for monitoring.

---

## 📊 Phase 4: Evaluation & Visualization (Day 3)

### 4.1 Performance Analysis

- [x] Create `src/evaluate.py`:
  - [x] Load `models/best/best_model.zip`.
  - [x] Load `data/generated/spectrum_test.npy` (The unseen high-traffic data).
  - [x] Run evaluation loop (no training).
  - [x] Calculate KPIs:
    - [x] Collision Rate (%)
    - [x] Success Rate (%)
    - [x] Average Throughput
  - [x] Compare against Random and Greedy Agent baselines.
  - [x] Generate comparison plots.

### 4.2 The Dashboard (Streamlit)

- [x] Create `app.py`:
  - [x] Build layout: Sidebar for controls, Main area for plots.
  - [x] **Plot 1:** Real-time Spectrum Waterfall (Heatmap).
  - [x] **Plot 2:** Agent Reward vs Time.
  - [x] **Plot 3:** Collision Events Marker.
  - [x] Add "Start/Stop Simulation" buttons to visualize the agent making decisions.

---

## 📜 Phase 5: Documentation & Delivery (Final)

### 5.1 Reporting

- [x] Generate "Verification Plot" for Dataset Methodology.
- [x] Export "Convergence Graph" comparing Default PPO vs. Hybrid GA-PPO.
- [x] Complete `methodology.md` with final implementation details.
- [x] Update `README.md` with "How to Run" instructions.

---

## 📈 Current Performance (as of last evaluation)

| Agent         | Collision Rate | vs Random  |
| ------------- | -------------- | ---------- |
| PPO (Trained) | **16.48%**     | **-72.5%** |
| Random        | 60.04%         | baseline   |
| Greedy        | 5.68%          | -90.5%     |

**TensorBoard logs:** `logs/tensorboard/`

---

## 🚀 Quick Start: Train RL+GA System

### Option 1: Full Pipeline (Recommended)
```powershell
# Run everything: baseline + GA + evaluation + plots
python run_full_pipeline.py

# Quick mode (for testing, ~15 minutes)
python run_full_pipeline.py --quick
```

### Option 2: Step-by-Step

1. **Generate Datasets** (if not already done):
   ```powershell
   python src/data_pipeline.py
   ```

2. **Train Baseline PPO** (without GA):
   ```powershell
   python src/train_agent.py --timesteps 100000
   ```

3. **Train GA-Optimized PPO**:
   ```powershell
   python src/train_agent.py --ga-optimize --ga-generations 10 --timesteps 100000
   ```

4. **Evaluate Agents**:
   ```powershell
   python src/evaluate.py --episodes 5
   ```

5. **Generate Convergence Plots**:
   ```powershell
   python src/plot_convergence.py
   ```

6. **View Training Curves** (in real-time):
   ```powershell
   tensorboard --logdir logs/tensorboard
   ```

7. **Run Interactive Demo**:
   ```powershell
   streamlit run app.py
   ```

### Generated Files & Figures

- **Training Logs**: `logs/tensorboard/PPO_*/` (view with TensorBoard)
- **Models**: `models/best/best_model.zip`, `models/*_final.zip`
- **Evaluation Results**: `data/generated/evaluation_results.png` + `.json`
- **Convergence Comparison**: `data/generated/convergence_comparison.png`
- **Dataset Verification**: `data/generated/data_verification_report.png`
- **Best GA Parameters**: `models/best_params.json` (after GA run)
