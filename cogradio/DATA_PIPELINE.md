# IoT-Optimized 6G Spectrum Dataset Generator

## Overview

The [data_pipeline.py](src/data_pipeline.py) file is a comprehensive, unified script that generates realistic IoT spectrum occupancy datasets for 6G cognitive radio research. It implements a **Markov-Modulated Poisson Process (MMPP)** traffic model compliant with industry standards and generates both datasets and scientific validation visualizations in a single execution.

## Scientific Foundation

### Standards Compliance
- **ETSI EN 303 645 V2.1.1**: Cyber Security for Consumer IoT (Traffic characteristics)
- **ITU-R M.2083-0**: IMT Vision for 6G (Massive IoT traffic models)  
- **ETSI TR 103 511 V1.1.1**: Cognitive Radio techniques for 5G/6G networks
- **3GPP TR 38.817**: NR for Non-Terrestrial Networks (Bursty/Sporadic Models)
- **3GPP TR 37.868**: MTC (Machine-Type Communications) improvements

### Mathematical Model

The dataset generator implements **heterogeneous IoT traffic** using MMPP with three distinct device classes representing real-world IoT deployments:

## Core Classes and Components

### 1. `IoTTrafficConfig`

**Purpose**: Centralized configuration for IoT device classes and traffic parameters.

**Key Attributes**:
- `n_channels`: Spectrum channels (default: 20)
- `time_steps_train`: Training dataset length (default: 10,000)
- `time_steps_test`: Test dataset length (default: 2,000)
- `n_devices`: Total IoT devices (default: 60, optimized for ~70% occupancy)

**Device Classes**:

#### Type A: Critical IoT (URLLC - Ultra-Reliable Low-Latency Communications)
- **Use Cases**: Medical sensors, industrial control systems
- **Characteristics**: Frequent beacons + event-driven alarms
- **Parameters**:
  - `avg_inter_arrival`: 8 time slots (periodic beacons)
  - `pareto_shape`: 0.8 (light tail for predictable bursts)
  - `min_duration`: 1, `max_duration`: 10
  - `priority`: 3 (highest)
  - `packet_size_mean`: 100 bytes (small control packets)

#### Type B: Delay-Tolerant IoT (mMTC - massive Machine-Type Communications)
- **Use Cases**: Smart home sensors, environmental monitoring
- **Characteristics**: Long intervals, ultra-short packets, 10-year battery life optimization
- **Parameters**:
  - `avg_inter_arrival`: 60 time slots (realistic sensor reporting)
  - `pareto_shape`: 0.5 (very light tail)
  - `min_duration`: 1, `max_duration`: 5
  - `priority`: 1 (lowest)
  - `packet_size_mean`: 50 bytes (sensor readings)

#### Type C: High-Throughput IoT (eMBB-IoT - enhanced Mobile BroadBand IoT)
- **Use Cases**: Video surveillance, AR/VR applications
- **Characteristics**: Near-continuous streaming, heavy-tailed sessions
- **Parameters**:
  - `avg_inter_arrival`: 12 time slots (frequent arrivals)
  - `pareto_shape`: 1.5 (heavy tail for long streaming sessions)
  - `min_duration`: 5, `max_duration`: 40
  - `priority`: 2 (medium)
  - `packet_size_mean`: 1500 bytes (video/audio streams)

### 2. `SpectrumDataGenerator`

**Purpose**: Core MMPP traffic generator implementing scientific traffic models.

#### Key Methods

##### `generate_device_traffic(n_steps, device_type)`
Implements **two-state Markov chain** for each device class:

- **State 1 (IDLE)**: Inter-arrival times follow **Exponential distribution** (Poisson arrivals)
  ```
  T_arrival ~ Exponential(λ) where λ = 1/avg_inter_arrival
  ```
- **State 2 (BUSY)**: Service times follow **Pareto distribution** (heavy-tailed)
  ```
  T_service ~ Pareto(α, x_m) where P(X > x) = (x_m/x)^α
  ```

**Returns**: Binary occupancy matrix `[time_steps × n_channels]` where:
- `1` = channel occupied
- `0` = channel free
- `dtype=np.int8` for memory efficiency

##### `generate_heterogeneous_traffic(n_steps, traffic_load)`
Combines all device classes with configurable traffic intensity:

- **Normal Load**: λ = λ₀ (~70% occupancy)
- **High Load**: λ = 1.5λ₀ (~85% occupancy, realistic 6G load)
- **Extreme Load**: λ = 2.5λ₀ (~95% occupancy, stress test)

**Algorithm**:
1. Generate independent traffic for each device type
2. Combine using **logical OR** (channel busy if ANY device transmits)
3. Track statistics for verification

##### `generate_training_data()` & `generate_test_data()`
- **Training**: Normal load for stable learning
- **Testing**: High load for stress testing and generalization

### 3. `VerificationPlotter`

**Purpose**: Generate scientific validation plots proving statistical compliance.

#### `create_verification_report()`
Creates **4-panel verification visualization**:

1. **Spectrum Occupancy Heatmap**: Visual validation of bursty block structure
2. **Duration Distribution vs Pareto Theory**: Mathematical validation of heavy-tail behavior
3. **Inter-Arrival Distribution vs Exponential Theory**: Validation of Poisson arrival process
4. **Device Class Occupancy Comparison**: Verification of heterogeneous traffic mix

**Output**: `data_verification_report.png` (publication-quality figure for presentations)

### 4. `EnhancedPlotter`

**Purpose**: Advanced analytics suite for hackathon presentation and technical defense.

#### Advanced Visualization Methods

##### `plot_collision_heatmap(grid, window=500)`
- Identifies collision-prone channels (>90% occupancy)
- Highlights risk zones with red dashed lines
- Generates per-channel occupancy bar chart

##### `plot_autocorrelation(grid, max_lag=100, n_channels=3)`
- Analyzes temporal correlation structure
- Proves dataset has temporal dependency (justifies recurrent RL models)
- Shows correlation persistence across multiple time lags

##### `plot_waterfall_spectrogram(grid, window=1000)`
- Classic RF engineering time-frequency representation
- Professional spectrogram using `viridis` colormap
- Shows spectrum usage evolution over time

##### `plot_cumulative_collisions(grid, threshold=0.7)`
- Dramatic visualization of collision problem severity
- Shows steady rise in collision events without intelligent allocation
- Includes collision rate per time window analysis

##### `plot_device_trajectory(grid, channel_id=0, window=500)`
- Traces single channel occupancy pattern
- Shows bursty IoT device behavior
- Includes burst duration histogram analysis

##### `plot_occupancy_timeline(grid, window_size=100)`
- Smooth timeline showing congestion patterns
- Color-coded occupancy levels (green/yellow/red)
- Statistical summary with mean, std dev, min/max

## Data Format and Output

### Generated Files
```
data/generated/
├── spectrum_train.npy          # Training data [10000 × 20]
├── spectrum_train_metadata.json    # Training statistics
├── spectrum_test.npy           # Test data [2000 × 20]
├── spectrum_test_metadata.json     # Test statistics
├── data_verification_report.png    # Scientific validation (4-panel)
├── occupancy_summary.png          # Channel distribution analysis
├── enhanced_collision_heatmap.png # Collision risk visualization
├── enhanced_autocorrelation.png   # Temporal correlation analysis
├── enhanced_waterfall_spectrogram.png # RF engineering plot
├── enhanced_cumulative_collisions.png # Problem severity visualization
├── enhanced_trajectory_channel_0.png  # Single channel behavior
└── enhanced_occupancy_timeline.png    # Congestion patterns
```

### Data Specifications
- **Shape**: `[time_steps, n_channels]`
- **Encoding**: Binary (1 = occupied, 0 = free)
- **Data Type**: `np.int8` for memory efficiency
- **Format**: NumPy `.npy` files for fast loading
- **Metadata**: JSON files with statistics and configuration

## Execution Pipeline

### Main Workflow (`main()` function)

```python
# 1. Initialize configuration and generator
config = IoTTrafficConfig()
generator = SpectrumDataGenerator(config, seed=42)

# 2. Generate training data (normal load)
train_grid, train_stats = generator.generate_training_data()

# 3. Generate test data (high load)
test_grid, test_stats = generator.generate_test_data()

# 4. Scientific verification plots
VerificationPlotter.create_verification_report(...)

# 5. Enhanced analytics plots
enhanced_plotter = EnhancedPlotter(output_dir)
enhanced_plotter.generate_all_plots(train_grid, train_stats)
```

### Usage

```bash
# Direct execution
python src/data_pipeline.py

# Using UV package manager
uv run gen-data

# As part of full pipeline
uv run pipeline-quick
```

## Key Features and Innovations

### 1. **Scientific Rigor**
- Standards-compliant traffic models (ETSI, ITU-R, 3GPP)
- Mathematically validated distributions (Exponential + Pareto)
- Statistical verification with autocorrelation analysis

### 2. **Realistic IoT Heterogeneity**
- Three distinct device classes reflecting real deployments
- Configurable device distribution ratios
- Priority-based access modeling

### 3. **Memory Optimization**
- `np.int8` data type for binary occupancy (8x memory savings vs. `float64`)
- Efficient channel assignment with replacement
- Optimized for 1000+ device scenarios

### 4. **Comprehensive Visualization**
- Scientific validation plots for academic defense
- Professional RF engineering visualizations
- Interactive-ready plots for live demonstration

### 5. **Reproducible Research**
- Fixed seed (`seed=42`) for consistent results
- Detailed metadata tracking
- JSON configuration export

## Hackathon Defense Points

### Technical Rigor
- *"Our data follows ETSI TR 103 511 standards for cognitive radio"*
- *"Pareto-distributed durations capture real IoT traffic heterogeneity"*
- *"Three device classes represent URLLC, mMTC, and eMBB-IoT use cases"*

### Performance Validation
- *"Autocorrelation analysis proves temporal dependency, justifying recurrent RL models"*
- *"Collision heatmaps identify specific problem zones for targeted optimization"*
- *"High-load test data ensures robustness under 6G massive IoT conditions"*

### Implementation Quality
- *"Memory-optimized binary encoding supports 1000+ device simulations"*
- *"Comprehensive visualization suite provides both scientific validation and presentation materials"*
- *"Standards compliance ensures real-world applicability"*

## Dependencies

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import json
import warnings
import multiprocessing as mp
from functools import partial
```

## Integration with RL Pipeline

The generated datasets integrate seamlessly with the cognitive radio environment:

```python
# Load in RL environment
from src.envs.cognitive_radio_env import CognitiveRadioEnv

env = CognitiveRadioEnv(
    data_path="data/generated/spectrum_train.npy",
    history_length=10,
    w_collision=10.0,
    w_throughput=2.0,
    w_energy=0.05
)
```

This data pipeline forms the foundation of the entire 6G cognitive radio project, providing scientifically validated, standards-compliant IoT traffic data for training and evaluating intelligent spectrum allocation algorithms.