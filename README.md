# TechSprint 6G - Cognitive Radio System

A complete 6G Cognitive Radio system with Automatic Modulation Classification (AMC) and Reinforcement Learning-based Channel Allocation.

---

## Abstract

The exponential growth of wireless devices and the emergence of 6G networks have intensified the demand for efficient and adaptive use of the radio spectrum. Traditional static spectrum allocation schemes are increasingly inadequate for handling the heterogeneous and dynamic traffic patterns of next-generation networks. This project presents an end-to-end **Cognitive Radio System** that combines **deep learning-based Automatic Modulation Classification (AMC)** with a **Reinforcement Learning (RL) agent for dynamic channel allocation**. A Convolutional Neural Network (CNN) is trained on synthetic in-phase/quadrature (IQ) samples to identify the modulation scheme of incoming signals, while a Proximal Policy Optimization (PPO) agent learns to allocate channels across primary and secondary users in a custom Gymnasium environment. The system is wrapped in a real-time PyQt6 dashboard that supports both simulated signals and live RTL-SDR captures, enabling visualization of the spectrum, modulation predictions, and RL-driven allocation decisions. Experimental results demonstrate that the proposed pipeline can reliably classify common digital modulations and learn allocation policies that improve spectrum utilization while reducing primary-user interference.

## Keywords

6G, Cognitive Radio, Automatic Modulation Classification, Deep Learning, Convolutional Neural Networks, Reinforcement Learning, Proximal Policy Optimization, Software-Defined Radio, Dynamic Spectrum Access.

## 1. Introduction

The radio frequency (RF) spectrum is a finite and increasingly congested resource. With the rollout of 5G and ongoing research into 6G, future networks are expected to support massive device densities, ultra-low latency, and high data rates across diverse application domains. **Cognitive Radio (CR)** is a key enabling paradigm that allows unlicensed (secondary) users to opportunistically access portions of the spectrum that are not currently in use by licensed (primary) users.

A practical CR system must address two coupled problems:

1. **Spectrum sensing and signal understanding** — identifying what kind of signal occupies a given band.
2. **Decision making** — deciding when and where secondary users should transmit without harming primary users.

This project tackles both problems jointly by integrating a CNN-based AMC module with a PPO-based RL allocation agent, exposed through a unified dashboard for analysis and demonstration.

## 2. Problem Statement

Given a stream of complex IQ samples captured from an SDR (or generated synthetically), the system must:

- Detect the presence and modulation type of signals occupying a set of monitored channels.
- Allocate available channels to a set of secondary users in real time, subject to constraints on primary-user interference and quality-of-service (QoS) requirements.
- Provide an interpretable, real-time visualization of the spectrum, the AMC predictions, and the RL agent's allocation decisions.

## 3. Methodology

The system is organized into four cooperating subsystems: dataset generation, AMC model training, RL agent training, and a real-time dashboard.

### 3.1 Dataset Generation

Synthetic IQ datasets are generated in `notebooks/dataset_generation.ipynb` using a configurable pipeline (`notebooks/dataset_pipeline.py`). The pipeline:

- Generates baseband symbols for a set of common digital modulations (e.g., BPSK, QPSK, 8PSK, QAM variants, FSK).
- Applies channel impairments such as additive white Gaussian noise (AWGN), frequency/phase offsets, and multipath effects via the physics models in `src/sdr/physics.py`.
- Stores fixed-length IQ frames together with modulation labels and SNR metadata, which are later consumed by the AMC training notebook.

### 3.2 Automatic Modulation Classification (AMC)

A 1D Convolutional Neural Network is trained in `notebooks/amc_training.ipynb` to classify the modulation scheme of an IQ frame:

- **Input:** complex IQ frame represented as a 2-channel (I, Q) tensor.
- **Architecture:** stacked 1D convolutional blocks with batch normalization and ReLU activations, followed by global pooling and a fully connected classification head.
- **Training:** cross-entropy loss with the Adam optimizer; checkpoints are written to `notebooks/models/` (e.g., `best_model_epoch_9.pth`).
- **Deployment:** the best model is exported to ONNX and loaded at runtime via `src/rl_inf/amc.py` for fast CPU inference inside the dashboard.

### 3.3 Reinforcement Learning for Channel Allocation

The channel allocation problem is formulated as a Markov Decision Process and solved using **Proximal Policy Optimization (PPO)** from `stable-baselines3`. The custom Gymnasium environment is defined in `notebooks/envs/cognitive_radio_env.py`.

- **State:** per-channel occupancy, AMC-derived modulation classes, SNR estimates, and pending secondary-user demands.
- **Action:** discrete assignment of channels to secondary users (or a no-op).
- **Reward:** positive for successful secondary-user transmissions and met QoS targets; negative for collisions with primary users and unmet demands.
- **Training:** `notebooks/rl_training.ipynb` trains the PPO policy and exports it to `notebooks/models/rl_agent.zip`. At runtime, `src/rl_inf/rl.py` wraps the policy for inference.

### 3.4 Real-Time Dashboard

The PyQt6 dashboard in `src/ui/` orchestrates the full pipeline:

- `src/radio/capture.py` and `src/radio/simulation.py` provide a hardware abstraction layer that supports both RTL-SDR captures and fully simulated signals.
- `src/ui/core/system.py` coordinates the AMC and RL inference modules.
- Custom widgets in `src/ui/widgets/charts/` render the time-domain signal, frequency-domain spectrum, waterfall, constellation, modulation probabilities, RL allocation state, and QoS metrics.
- The interface (menus, toolbars, action handlers) lets the user start/stop capture, switch sources, and inspect per-channel decisions in real time.

## 4. System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   SDR/Sensor    │────▶│   AMC Model     │────▶│   RL Agent      │
│  (IQ Samples)   │     │ (Modulation ID) │     │(Channel Alloc)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                              │
         └──────────────────────┬───────────────────────┘
                                ▼
                    ┌─────────────────────┐
                    │     Dashboard UI    │
                    │ (Real-time Display) │
                    └─────────────────────┘
```

## 5. Experimental Setup

- **Language / Runtime:** Python 3.10+
- **Deep Learning:** PyTorch (training) and ONNX Runtime (deployment).
- **Reinforcement Learning:** `stable-baselines3` (PPO) with a custom Gymnasium environment.
- **UI:** PyQt6 with `pyqtgraph`-based custom chart widgets.
- **Signal Source:** RTL-SDR for live captures; internal IQ generator for reproducible simulation.
- **Datasets:** synthetic IQ frames covering several digital modulation schemes across a range of SNR values.

## 6. Results and Discussion

- The trained AMC CNN achieves strong classification accuracy on the held-out synthetic test set, with the expected degradation at low SNR values.
- The PPO agent converges to allocation policies that prefer idle channels and avoid bands where the AMC module reports active primary-user modulations, leading to higher secondary-user throughput compared to a random-allocation baseline.
- The integrated dashboard runs the full sense → classify → decide loop in real time, confirming that the ONNX-exported AMC model and the lightweight PPO policy are suitable for interactive use on commodity hardware.

## 7. Conclusion and Future Work

This project demonstrates a fully integrated cognitive radio prototype that couples deep-learning-based modulation classification with reinforcement-learning-based channel allocation, all driven from a real-time SDR-aware dashboard. The modular design — separating signal acquisition, inference, environment, and UI — makes it straightforward to extend.

Promising directions for future work include:

- Training the AMC model on real-world recordings (e.g., RadioML-style datasets) to improve robustness.
- Exploring multi-agent RL for distributed allocation across many secondary users.
- Adding richer QoS modeling (latency, jitter, fairness) to the reward function.
- Hardening the RTL-SDR capture path for continuous, long-duration deployments.

## 8. References

1. J. Mitola and G. Q. Maguire, "Cognitive radio: making software radios more personal," *IEEE Personal Communications*, 1999.
2. T. J. O'Shea, J. Corgan, and T. C. Clancy, "Convolutional Radio Modulation Recognition Networks," *EANN*, 2016.
3. J. Schulman et al., "Proximal Policy Optimization Algorithms," *arXiv:1707.06347*, 2017.
4. A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *JMLR*, 2021.
5. G. Brockman et al., "OpenAI Gym," *arXiv:1606.01540*, 2016.

---

## Project Structure

```
TechSprint_6G/
├── notebooks/                      # Jupyter Notebooks & Data
│   ├── dataset_generation.ipynb    # Generate synthetic RF datasets
│   ├── amc_training.ipynb          # Train AMC CNN model
│   ├── rl_training.ipynb           # Train PPO RL agent
│   ├── amc_inference.ipynb         # Standalone inference tests
│   ├── dataset_pipeline.py         # Dataset utilities
│   ├── models/                     # Trained model weights
│   │   └── best_model_epoch_9.pth
│   └── results/                    # Training logs & metrics
│
└── src/                            # Source Code
    ├── core/                       # Shared business logic
    ├── envs/                       # Gymnasium environments
    │   └── cognitive_radio_env.py  # RL training environment
    ├── sdr/                        # Hardware abstraction layer
    │   ├── smart_sensor.py         # SDR interface (RTL-SDR / Simulation)
    │   ├── iq_generator.py         # Synthetic IQ generation
    │   ├── physics.py              # Channel physics models
    │   └── rl_bridge.py            # RL agent interface
    └── ui/                         # PyQt6 Dashboard Application
        ├── main.py                 # UI entry point
        ├── core/                   # UI system controller
        ├── widgets/                # Custom chart widgets
        ├── interface/              # Menus & Toolbars
        ├── logic/                  # Action handlers
        ├── radio/                  # Radio capture logic
        └── style.qss               # Visual theme
```

## Quick Start

### 1. Generate Dataset
```bash
cd notebooks
jupyter notebook dataset_generation.ipynb
# Run all cells
```

### 2. Train AMC Model
```bash
jupyter notebook amc_training.ipynb
# Run all cells -> Exports to notebooks/models/amc_model.onnx
```

### 3. Train RL Agent
```bash
jupyter notebook rl_training.ipynb
# Run all cells -> Exports to notebooks/models/rl_agent.zip
```

### 4. Launch Dashboard
```bash
cd src/ui
python main.py
```

## Requirements

- Python 3.10+
- PyQt6
- PyTorch
- stable-baselines3
- onnxruntime
- numpy, scipy, matplotlib

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   SDR/Sensor    │────▶│   AMC Model     │────▶│   RL Agent      │
│  (IQ Samples)   │     │ (Modulation ID) │     │(Channel Alloc)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                              │
         └──────────────────────┬───────────────────────┘
                                ▼
                    ┌─────────────────────┐
                    │     Dashboard UI    │
                    │ (Real-time Display) │
                    └─────────────────────┘
```
