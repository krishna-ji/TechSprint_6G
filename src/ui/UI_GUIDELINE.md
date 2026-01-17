# TechSprint 6G - UI Guidelines & Architecture

## Overview
This document outlines the architecture and design guidelines for the **amcRFIdentifyUI**, a Cognitive Radio dashboard integrating Artificial Intelligence (AMC) and Reinforcement Learning (RL) components.

## 1. System Architecture

The application follows a **Three-Layer Architecture**:

### A. Frontend (View)
*   **Framework:** PyQt6
*   **Main Entry:** `main.py`
*   **Components:**
    *   **Docks:** Re-arrangeable windows for different visualizations (Spectrum, Time Domain, etc.).
    *   **Heads-Up Display (HUD):** A fixed panel at the top of the main window for critical real-time stats (Modulation, Channel).
    *   **Widgets:** Custom plotting widgets in `widgets/charts/`.

### B. Controller (Core)
*   **Location:** `core/system.py`
*   **Role:** The "Brain" of the application. It acts as the bridge between the raw data source and the UI.
*   **Threading:** Uses a dedicated `QThread` (`DataWorker`) to prevent UI freezing during model inference.
*   **Signals:**
    *   `update_plots`: Emits raw IQ data for drawing.
    *   `update_status`: Emits text for the HUD.

### C. Backend (Models & Hardware)
*   **Location:** `core/` wrapper files.
*   **RL Agent:** `rl_wrapper.py` executes the PPO model for channel allocation.
*   **AMC Model:** `amc_wrapper.py` executes the ONNX model for modulation classification.
*   **Sensor:** `sensor_backend.py` (abstracts RTL-SDR or Simulation).

---

## 2. UI Design Guidelines ("Cyber-Industrial")

The interface uses a dark, high-contrast theme suitable for engineering environments.

### Color Palette
*   **Background:** `#1E1E2E` (Deep Gunmetal) - Reduces eye strain compared to pure black.
*   **Accent 1 (Modulation):** `#FF2A6D` (Neon Pink/Red) - Used for classification results.
*   **Accent 2 (Channel):** `#00E5FF` (Neon Cyan) - Used for allocation status.
*   **Panels:** `#252526` - Slightly lighter than background to differentiate zones.

### Layout Principles
1.  **Tiled Default:** New windows should be initialized in a tiled, non-overlapping structure.
2.  **Tabbed Groups:** Secondary views (e.g., Time Domain vs Frequency Domain) should be tabbed together to save pixel space for the primary Waterfall or Spectrum view.
3.  **Heads-Up Status:** Critical indicators (Modulation, Channel) must be visible at a glance and not hidden in foldable menus.

### Extending the UI (How-To)

**To add a new Plot Widget:**
1.  Create the widget class in `widgets/charts/`.
2.  In `main.py`, instantiate it and wrap it in a `QDockWidget`.
3.  Connect it to the central data signal:
    ```python
    self.system.update_plots.connect(self.my_new_widget.update_method)
    ```

**To add a new Status Indicator:**
1.  Add a `QLabel` to the `self.status_layout` in `main.py`.
2.  Give it a unique `objectName` (e.g., `lbl_snr`).
3.  Style it in `style.qss` using ID selection (`#lbl_snr`).
4.  Update `core/system.py` to emit the new metric and connect it in `main.py`.

---

## 3. Configuration
*   **`style.qss`**: Central stylesheet. Edit this to change colors or fonts globally.
*   **`config.py`**: application constants (Window size, FFT size).
