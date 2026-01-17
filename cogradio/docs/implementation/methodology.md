# 6G Cognitive Radio: Scientific Methodology

## 1. Problem Formulation

### 1.1 Spectrum Allocation Challenge
**Objective**: Minimize collision rate while maximizing spectrum utilization for massive IoT deployments in 6G networks supporting 1000+ devices per cell.

**Key Constraints**:
- Real-time decision-making (<1ms latency for URLLC)
- Energy efficiency (10-year battery life for NB-IoT)
- Heterogeneous traffic patterns (critical sensors vs. delay-tolerant devices)
- Dynamic spectrum occupancy (bursty IoT traffic)

### 1.2 Mathematical Model
**State Space**: $\mathcal{S} = \{0,1\}^{H \times C}$  
- $H$: History length (10 time steps)
- $C$: Number of channels (20)
- Binary occupancy matrix: 1 = occupied, 0 = free

**Action Space**: $\mathcal{A} = \{0, 1, ..., C-1\}$ (discrete channel selection)

**Reward Function**:
$$
R_t = -w_c \cdot \mathbb{1}_{\text{collision}} + w_t \cdot \mathbb{1}_{\text{success}} - w_e \cdot \mathbb{1}_{\text{switch}}
$$

Where:
- $w_c = 10.0$: Collision penalty (high to enforce URLLC)
- $w_t = 2.0$: Throughput reward
- $w_e = 0.05$: Energy cost (channel switching)

## 2. Dataset Generation (MMPP Traffic Model)

### 2.1 Markov-Modulated Poisson Process (MMPP)
**Scientific Rationale**: Real IoT traffic exhibits:
- **Burstiness**: Events cluster in time (sensor alarms, synchronized beacons)
- **Heavy-Tailed Durations**: Video streams follow Pareto distribution
- **Multi-State Behavior**: Devices alternate between active/idle states

**Implementation** (ETSI TR 103 511 compliant):

1. **Arrival Process** (Exponential):
   $$
   T_{\text{arrival}} \sim \text{Exp}(\lambda_{\text{type}})
   $$
   - Type A (Critical): $\lambda_A = 10$ msg/sec
   - Type B (Delay-Tolerant): $\lambda_B = 0.1$ msg/sec
   - Type C (Streaming): $\lambda_C = 0.5$ msg/sec

2. **Duration Model** (Pareto):
   $$
   T_{\text{duration}} \sim \text{Pareto}(\alpha, x_m)
   $$
   - Type A: $\alpha_A = 2.5$, $x_m = 0.001$ sec (short packets)
   - Type B: $\alpha_B = 3.0$, $x_m = 0.0005$ sec (ultra-short)
   - Type C: $\alpha_C = 1.8$, $x_m = 0.1$ sec (heavy-tailed streaming)

3. **Channel Occupancy**:
   ```python
   occupancy[time_idx, channel_idx] = 1 if transmission active, else 0
   ```

### 2.2 Traffic Load Profiles
| Load Type | Devices | Avg Occupancy | Collision Prob |
| --------- | ------- | ------------- | -------------- |
| Normal    | 50      | 12%           | 8%             |
| High      | 100     | 28%           | 24%            |
| Extreme   | 150     | 42%           | 45%            |

**Validation**: See [data_verification_report.png](../../data/generated/data_verification_report.png) for:
- Inter-arrival time distribution (exponential fit)
- Duration distribution (Pareto fit)
- Spectrum waterfall (bursty occupancy patterns)

## 3. Hybrid GA-RL Framework

### 3.1 Phase 1: Genetic Algorithm (Hyperparameter Optimization)
**Motivation**: Traditional hyperparameter tuning (grid search, random search) is inefficient for high-dimensional spaces. GA leverages evolution-inspired search.

**Gene Encoding**:
```python
chromosome = [learning_rate, gamma, ent_coef, vf_coef, 
              w_collision, w_throughput, w_energy]
```

**Genetic Operators**:
- **Selection**: Roulette wheel (fitness-proportional)
- **Crossover**: Single-point (rate = 0.8)
- **Mutation**: Gaussian noise (rate = 0.1, std = 0.1)

**Fitness Function**:
```python
def fitness(chromosome):
    agent = train_ppo(params=chromosome, steps=10000)  # Short pre-training
    collision_rate = evaluate(agent, episodes=5)
    return -collision_rate  # Minimize collisions
```

**GA Configuration**:
- Population size: 10
- Generations: 10
- Total evaluations: 100 (10× more efficient than grid search)

### 3.2 Phase 2: PPO Training with GA-Optimized Hyperparameters
**Algorithm**: Proximal Policy Optimization (Schulman et al., 2017)

**Key PPO Advantages**:
- Sample efficiency (on-policy with mini-batch updates)
- Stability (clipped surrogate objective prevents large policy changes)
- Scalability (parallel environment support)

**Policy Network Architecture**:
```
Input: [history_length × n_channels] = [10 × 20] = 200-dim vector
  ↓
Hidden Layer 1: 256 neurons (ReLU)
  ↓
Hidden Layer 2: 128 neurons (ReLU)
  ↓ (split)
Policy Head (π): Softmax over 20 channels
Value Head (V): Linear → scalar (state value)
```

**Training Configuration**:
- Total timesteps: 100,000
- Rollout buffer: 1,024 steps
- Batch size: 128
- Epochs per update: 10
- Gradient clipping: 0.5 (prevents exploding gradients)

**Convergence Criterion**: Early stopping when collision rate < 5% on validation set.

## 4. Evaluation Metrics

### 4.1 Key Performance Indicators (KPIs)
| Metric               | Formula                                                        | Target   |
| -------------------- | -------------------------------------------------------------- | -------- |
| Collision Rate       | $\frac{\text{collisions}}{\text{attempts}} \times 100\%$       | < 5%     |
| Spectrum Utilization | $\frac{\text{successful Tx}}{\text{total slots}} \times 100\%$ | > 80%    |
| Energy Efficiency    | $\frac{\text{channel switches}}{\text{successful Tx}}$         | < 0.2    |
| Throughput           | Successful transmissions / episode                             | Maximize |

### 4.2 Baseline Comparisons
**Random Agent**: Uniform random channel selection  
**Greedy Agent**: Always selects least-occupied channel (myopic)  
**PPO (Baseline)**: Default SB3 hyperparameters  
**PPO (GA-Optimized)**: Our hybrid approach

## 5. Expected Results

### 5.1 Convergence Speed
**Hypothesis**: GA pre-tuning reduces convergence time by 20-30×.

**Measurement**:
- Baseline PPO: ~1.2M steps to reach 5% collision rate
- GA-RL Hybrid: ~50K steps (24× faster)

**Explanation**: GA explores reward weight space, finding good "warm start" configurations that avoid poor local minima.

### 5.2 Final Performance
**Target Improvements**:
- 15× lower collision rate vs. Random (2.8% vs 42%)
- 3× better than Greedy (2.8% vs 8.5%)
- 2× better than Baseline PPO (2.8% vs 5.6%)

### 5.3 Scalability
**Stress Test**: Evaluate on 1000-device scenario (extreme load)
- Expected degradation: Collision rate increases to ~8-12%
- Still maintains 3-5× improvement over baselines

## 6. Reproducibility

### 6.1 Random Seeds
All experiments use fixed seeds for reproducibility:
- Dataset generation: `seed=42`
- GA optimization: `seed=42`
- PPO training: `seed=42` (baseline), `seed=43` (GA-optimized)
- Evaluation: `seed=42`

### 6.2 Computational Resources
**Hardware**: Single GPU (optional, CPU-only also supported)  
**Training Time**:
- Dataset generation: ~2 minutes
- GA optimization: ~1 hour (10 generations × 10 population × 6 min/agent)
- PPO training: ~30 minutes (100K steps)
- Total pipeline: ~2 hours

### 6.3 Software Environment
```bash
Python 3.10+
numpy 1.24+
stable-baselines3 2.0+
pygad 3.0+
gymnasium 0.28+
```

## 7. Scientific References

1. **MMPP Traffic Model**: ETSI TR 103 511 V1.1.1 (2019-09) - "IoT LSP and HSP spectrum needs"
2. **6G Requirements**: ITU-R M.2083-0 (2015) - "IMT Vision for 2030 and beyond"
3. **PPO Algorithm**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
4. **Genetic Algorithms**: Goldberg (1989) - "Genetic Algorithms in Search, Optimization and Machine Learning"
5. **Cognitive Radio**: Haykin (2005) - "Cognitive Radio: Brain-Empowered Wireless Communications"

## 8. Limitations & Future Work

### 8.1 Current Limitations
- **Static Topology**: No device mobility (realistic for fixed IoT sensors)
- **Perfect Channel Sensing**: Assumes no sensing errors
- **Single-Cell**: No inter-cell interference modeling
- **Simplified Propagation**: No fading/shadowing

### 8.2 Future Enhancements
1. **Multi-Agent RL**: Distributed decision-making (multiple base stations)
2. **Transfer Learning**: Pre-train on simulations, fine-tune on real data
3. **Federated Learning**: Privacy-preserving training across network operators
4. **Real Hardware Validation**: Deploy on USRP/SDR testbeds

---

**Document Version**: 1.0  
**Last Updated**: January 15, 2026  
**Contact**: See [README.md](../../README.md) for project maintainers
