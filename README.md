# TechSprint 6G - Cognitive Radio System

A complete 6G Cognitive Radio system with Automatic Modulation Classification (AMC) and Reinforcement Learning-based Channel Allocation.

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
