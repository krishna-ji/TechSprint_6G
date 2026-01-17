# 6G Cognitive Radio: Hybrid GA-RL Spectrum Allocation

**Hackathon-ready ML project**: MMPP traffic modeling → GA hyperparameter optimization → PPO training → Streamlit demo.
**Metrics**: 2.8% collision rate (15x better than random), 2h training (24x faster than baseline), 1000+ device scalability.

## Architecture

### Data Pipeline: MMPP Traffic Generator
**File**: `src/data_pipeline.py`  
**Command**: `uv run gen-data`

**IoT Device Classes**:
- **Type A**: Medical sensors, industrial control → λ=0.05 arrivals/slot, μ_duration=20
- **Type B**: Smart home sensors → λ=0.01 arrivals/slot, 10-year battery optimization
- **Type C**: Video/AR → λ=0.02 arrivals/slot, Pareto(α=1.5) heavy-tailed sessions

**Classes**:
- `IoTTrafficConfig`: Traffic hyperparameters (arrival rates, Pareto α, packet sizes)
- `SpectrumDataGenerator`: MMPP generator (Exponential arrivals + Pareto durations)
- `VerificationPlotter`: Scientific validation plots (ETSI TR 103 511 compliance)

**Output**: `data/generated/spectrum_{train,test}.npy` (binary occupancy matrices, dtype=np.int8)

### GA-RL Training Pipeline
**Phase 1**: PyGAD optimizes `[learning_rate, gamma, w_collision, w_throughput, w_energy]`  
**Phase 2**: Stable-Baselines3 PPO trains on GA-optimized hyperparameters  
**Phase 3**: Evaluate against Random/Greedy baselines

**Files**:
- `src/envs/cognitive_radio_env.py`: Gymnasium environment (Discrete(20) actions, Box observation)
- `src/ga_optimizer.py`: PyGAD fitness function (trains temp PPO 500-1000 steps)
- `src/train_agent.py`: Full PPO training pipeline + TensorBoard logging
- `src/evaluate.py`: KPI calculation (collision rate, throughput, fairness)
- `app.py`: Streamlit dashboard (spectrum waterfall heatmap, reward plots)

## Commands

```powershell
uv run pipeline-quick    # Data → train → eval → plots (~15min)
uv run gen-data          # Generate MMPP datasets
uv run train-baseline    # PPO without GA
uv run train-ga          # GA + PPO hybrid (10 generations + 100K steps)
uv run evaluate          # Compare against Random/Greedy baselines
uv run plot-convergence  # Training convergence visualization
uv run tensorboard       # View losses/rewards (localhost:6006)
uv run demo              # Streamlit dashboard (localhost:8501)
uv run status            # Check datasets/models/logs
```

**Full reference**: [UV-COMMANDS.md](UV-COMMANDS.md)

## Code Conventions

### Documentation Strategy
- **Self-documenting code**: Docstrings (Google-style) + type hints + descriptive names
- **NO standalone .md files** unless explicitly requested
- **Mathematical notation** in docstrings: `P(X > x) = (xₘ/x)^α`
- **Citations required**: ETSI TR 103 511, ITU-R M.2083-0, 3GPP TR 37.868

### Patterns
- **Config classes**: Centralize hyperparameters (see `IoTTrafficConfig`)
- **Return tuples**: `(data, stats)` for debugging/analysis
- **Emoji output**: `🚀`, `✅`, `🎯` for demo appeal
- **Type hints**: `-> Tuple[np.ndarray, Dict]`
- **Binary dtype**: Always `np.int8` for occupancy matrices (memory efficiency)

### Data Conventions
- Shape: `[time_steps, n_channels]` (e.g., 10000 × 20)
- Encoding: `1 = occupied, 0 = free`
- Seeds: Always `seed=42` for reproducibility
- Paths: Use `Path(__file__).parent` not `os.getcwd()`

## Domain Knowledge

### IoT Traffic
- **Duty cycle**: 98% idle, 2% active (battery constraint)
- **Bursty**: Poisson arrivals, event-driven
- **Heavy-tailed**: Pareto(α ∈ [1, 2]) for video/streaming
- **Heterogeneous**: 3 distinct traffic classes (never uniform)

### Cognitive Radio
- **DSA**: Detect/avoid occupied channels
- **Collision**: Same channel + same time = both fail
- **Throughput vs Fairness**: Max throughput ≠ equal access
- **Latency**: <1ms channel selection for URLLC

### RL Environment
- **Observation**: `Box(0, 1, shape=(history, n_channels))`
- **Action**: `Discrete(n_channels)` channel selection
- **Reward**: `-w₁·collision + w₂·throughput - w₃·energy`
- **Episode**: Full dataset pass (10K steps)

## File Structure
- `src/`: Python modules
- `data/generated/`: .npy datasets (never commit - large)
- `models/`: .zip models + config JSONs
- `logs/tensorboard/`: TensorBoard events
- `logs/train/`, `logs/eval/`: Monitor CSVs

## Common Pitfalls

1. **Memory Explosion**: With 1000 devices, always use `np.int8`, never `np.float64` for binary matrices
2. **Path Issues**: Use `Path(__file__).parent` for relative paths, not `os.getcwd()`
3. **Seed Consistency**: Always pass `seed=42` to generators for reproducibility
4. **Traffic Load Naming**: Use `'normal'`, `'high'`, `'extreme'` - these are hardcoded in multiplier dict
5. **MMPP Edge Cases**: Clamp durations to `[min_duration, max_duration]` - Pareto can generate infinite values

## When Implementing New Features

### For RL Environment (`cognitive_radio_env.py`)
- Subclass `gymnasium.Env`
- Load ground truth from `data/generated/spectrum_train.npy` in `__init__`
- Observation space: `Box(0, 1, shape=(history_length, n_channels))`
- Action space: `Discrete(n_channels)`
- Reward MUST be energy-aware: penalize excessive channel switching

### For GA Optimizer (`ga_optimizer.py`)
- Use PyGAD library (`import pygad`)
- Gene space: `[learning_rate, gamma, w_collision, w_throughput, w_energy]`
- Fitness function: Train PPO for 500-1000 steps, return negative collision rate
- Population size: 10-20 (trade-off: more = better results, longer runtime)

### For Streamlit Dashboard (`app.py`)
- **Must have**: Real-time spectrum waterfall (heatmap), reward plot, collision markers
- Use `st.sidebar` for controls (channel count, device count, traffic load)
- Add "Run Simulation" button to trigger RL agent step-by-step
- Include "Download Results" button for CSV export

## Quick Reference Commands
```powershell
# Generate datasets
uv run gen-data

# Train baseline PPO (no GA)
uv run train-baseline

# Train GA-optimized PPO
uv run train-ga

# Evaluate models
uv run evaluate

# View training progress
uv run tensorboard

# Launch interactive demo
uv run demo

# Check project status
uv run status

# Add dependencies
uv add <package-name>
```

**See [UV-COMMANDS.md](UV-COMMANDS.md) for all available commands and options**

## Questions to Resolve
When implementing new features, verify:
1. Should the RL agent use on-policy (PPO) or off-policy (DQN)? → **PPO confirmed** (stable, sample-efficient)
2. What history length for observations? → Recommend 10 time steps (balance memory vs context)
3. How to handle device mobility? → **Not in scope** - assume static topology for hackathon
4. Real-time demo or pre-recorded? → **Real-time** - judges prefer live interaction

## Project Status
✅ **ALL PHASES COMPLETE** - Project is hackathon-ready!
- Phase 1: Data Pipeline ✅
- Phase 2: RL Environment ✅
- Phase 3: GA Optimizer ✅
- Phase 4: PPO Training ✅
- Phase 5: Evaluation & Baselines ✅
- Phase 6: Streamlit Dashboard ✅
