# AMC-APP (Automated RF / Modulation Classification UI)

Desktop application for visualizing live (or simulated) IQ samples and running an ONNX model to estimate modulation/probability outputs.

The UI is built with PyQt6 and pyqtgraph and shows multiple dockable views:
- Constellation plot (IQ scatter)
- Time-domain plot
- Frequency-domain plot (FFT)
- Waterfall
- Probability bar chart (model inference)

## How it works

1. A capture flowgraph produces blocks of complex IQ samples (1024 samples per update).
	 - Live capture: GNU Radio + SoapySDR (RTL-SDR)
	 - Dummy capture: a generated complex sine wave for UI testing
2. The constellation widget emits IQ data to other widgets.
3. The ONNX model (see `models/combined_model.onnx`) is run via ONNX Runtime to produce logits/probabilities that drive the probability bar chart.

## Requirements

### Python packages

At minimum (UI + inference):
- PyQt6
- pyqtgraph
- qtawesome
- numpy
- onnxruntime

For live SDR capture (optional):
- GNU Radio
- SoapySDR + RTL-SDR support (SoapyRTLSDR)

Notes:
- Installing GNU Radio/SoapySDR is platform-specific. If you only want to run the UI, use the dummy capture mode below.

## Setup

Create a virtual environment and install Python dependencies:

PowerShell (Windows):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install PyQt6 pyqtgraph qtawesome numpy onnxruntime
```

macOS/Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install PyQt6 pyqtgraph qtawesome numpy onnxruntime
```

If you want SDR capture, install GNU Radio + SoapySDR for your OS and ensure an RTL-SDR is available.

## Run

From the repo root:

```bash
python main.py
```

In VS Code you can also run the task: **Run Main.py**.

### Real-world classification UI

![Real-world classification UI](realworld-classify-ui.png)

### Dummy (no hardware) mode

To run without an RTL-SDR/GNU Radio hardware setup, switch the constellation capture backend to the dummy flowgraph:

- Edit `widgets/charts/constellation.py`
- Change:
	- `from radio.capture import FullCaptureFlowgraph`
	- to `from radio.capture_dummy import FullCaptureFlowgraph`

### Simulation UI

![Simulation UI](simulation-ui.png)

## Configuration

Common settings are in `config.py`:
- Window sizing / refresh rates
- Plot colors
- `MODEL_PATH` (defaults to `./models/combined_model.onnx`)
- SDR variables (sample rate, center frequency, gain, bandwidth)

## Project layout (high level)

- `main.py`: Main PyQt window + dock layout and signal wiring
- `interface/`: menu bar + toolbar UI
- `widgets/`: UI widgets
	- `widgets/charts/`: plot widgets (constellation, frequency, time, waterfall, probability, simulation)
- `radio/`: GNU Radio capture flowgraphs (live + dummy)
- `logic/`: app actions and ONNX inference helpers
- `models/`: ONNX model and notebooks
