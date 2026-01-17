# src/ui/__init__.py
"""
UI Package - PyQt6 Presentation Layer

This package provides the graphical user interface for the
6G Cognitive Radio Spectrum Sensing system.

Architecture
------------
The UI follows a clean separation pattern:
- src/radio/     : RTL-SDR hardware capture
- src/rl_inf/    : ML inference only (ONNX + PPO)
- src/ui/        : Visualization and user interaction

Modules
-------
main.py : Application entry point and main window
config.py : Centralized constants and configuration
core/system.py : Central controller coordinating radio + inference + UI
widgets/charts/ : Signal visualization widgets (constellation, spectrum, etc.)
interface/ : Menu bar and toolbar components
logic/ : Event handlers and business logic

Data Flow
---------
1. SystemController starts DataWorker thread
2. DataWorker captures IQ → runs AMC → runs RL → emits unified signal
3. Main window receives signals and updates all chart widgets
4. Single source of truth: inference happens once in DataWorker

Dependencies
------------
- PyQt6 : GUI framework
- pyqtgraph : High-performance plotting
- numpy : Numerical processing
- onnxruntime : AMC classification
- stable-baselines3 : RL channel allocation
"""

__version__ = "0.1.0"
