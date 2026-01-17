"""
System Controller

Central controller that coordinates data flow between:
- Radio capture (RTL-SDR or simulation)
- RL inference (channel allocation)
- AMC inference (modulation classification)
- UI updates

Single source of truth for all inference - no duplicate processing.
"""

from PyQt6.QtCore import QObject, pyqtSignal, QThread
import numpy as np
import time
from pathlib import Path

# Import from inference module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rl_inf import AMCClassifier, RLAgent
from radio import FullCaptureFlowgraph

# Import constants
from config import SAMPLE_SIZE, N_CHANNELS


class DataWorker(QObject):
    """
    Worker thread for data acquisition and ML inference.
    Keeps UI responsive by running processing in background.
    
    Emits all data needed by UI widgets in a single signal to avoid
    duplicate inference calls.
    """
    # Single unified signal: iq, probs, mod_class, channel, occupancy
    data_ready = pyqtSignal(np.ndarray, np.ndarray, str, int, np.ndarray)

    def __init__(self, radio, amc: AMCClassifier, rl: RLAgent, use_hardware: bool = False):
        super().__init__()
        self.radio = radio
        self.amc = amc
        self.rl = rl
        self.use_hardware = use_hardware
        self.running = True
        self.current_channel = 0
        self.n_channels = N_CHANNELS
        
        # EWMA smoothing for probabilities
        self.alpha = 0.1
        self.ewma_probs = None

    def run(self):
        while self.running:
            # 1. Get IQ Data
            if self.use_hardware and self.radio is not None:
                iq_data = self.radio.get_iq_sample()
            else:
                iq_data = self._simulate_iq()
            
            # 2. AMC Classification (SINGLE inference point)
            probs = self.amc.predict_proba(iq_data)
            avg_probs = probs.mean(axis=0)  # Average across windows
            
            # Apply EWMA smoothing
            if self.ewma_probs is None:
                self.ewma_probs = avg_probs
            else:
                self.ewma_probs = self.alpha * avg_probs + (1 - self.alpha) * self.ewma_probs
            
            # Get predicted class
            class_id = int(np.argmax(self.ewma_probs))
            mod_class = self.amc.CLASSES[class_id]
            
            # 3. Determine occupancy (non-noise = occupied)
            is_occupied = 1 if class_id > 0 else 0
            
            # 4. RL Channel Allocation
            allocated_channel = self.rl.decide(
                self.current_channel, 
                is_occupied, 
                class_id
            )
            self.current_channel = allocated_channel
            
            # 5. Simulate channel occupancy for visualization
            occupancy = self._simulate_occupancy()
            
            # 6. Emit ALL results in single signal
            self.data_ready.emit(
                iq_data, 
                self.ewma_probs.copy(),
                mod_class, 
                allocated_channel,
                occupancy
            )
            
            time.sleep(0.1)
    
    def _simulate_iq(self) -> np.ndarray:
        """Generate simulated IQ data for testing."""
        N = SAMPLE_SIZE
        t = np.arange(N)
        bits = np.random.randint(0, 2, N)
        symbols = 2 * bits - 1
        fc = 0.1
        carrier = np.exp(1j * 2 * np.pi * fc * t)
        sig = symbols * carrier
        noise = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.5
        return (sig + noise).astype(np.complex64)
    
    def _simulate_occupancy(self) -> np.ndarray:
        """Generate random channel occupancy for visualization."""
        return (np.random.random(self.n_channels) > 0.8).astype(np.float32)
    
    def stop(self):
        self.running = False


class SystemController(QObject):
    """
    Main system controller coordinating radio, inference, and UI.
    
    Single source of truth for all ML inference.
    """
    # Signals for UI updates
    update_plots = pyqtSignal(np.ndarray)           # IQ data for charts
    update_probs = pyqtSignal(np.ndarray)           # Probabilities for bar graph
    update_status = pyqtSignal(str, str)            # Mod class, channel
    update_spectrum = pyqtSignal(np.ndarray)        # Channel occupancy
    
    def __init__(self, use_hardware: bool = False):
        super().__init__()
        
        # Model paths - all models stored in notebooks/models/
        project_root = Path(__file__).parent.parent.parent.parent
        models_path = project_root / "notebooks" / "models"
        self.amc_path = models_path / "amc_model.onnx"
        self.rl_path = models_path / "rl_model.zip"
        
        # Initialize radio
        self.use_hardware = use_hardware
        self.radio = None
        if use_hardware:
            try:
                self.radio = FullCaptureFlowgraph()
                self.radio.start()
            except Exception as e:
                print(f"⚠️  Hardware init failed: {e}")
                self.use_hardware = False
        
        # Initialize inference models
        self.amc = AMCClassifier(self.amc_path)
        self.rl = RLAgent(self.rl_path)
        
        # Worker thread
        self.thread = QThread()
        self.worker = DataWorker(self.radio, self.amc, self.rl, self.use_hardware)
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.data_ready.connect(self._handle_data)
    
    def start_system(self):
        if not self.thread.isRunning():
            self.thread.start()
    
    def stop_system(self):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        if self.radio is not None:
            self.radio.stop()
    
    def _handle_data(self, iq_data, probs, predicted_mod, allocated_channel, occupancy):
        """Distribute data to all UI components."""
        self.update_plots.emit(iq_data)
        self.update_probs.emit(probs)
        self.update_status.emit(
            f"Detected: {predicted_mod}",
            f"Channel: {allocated_channel}"
        )
        self.update_spectrum.emit(occupancy)
