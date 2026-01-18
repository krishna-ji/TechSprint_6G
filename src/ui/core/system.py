"""
System Controller

Central controller that coordinates data flow between:
- Radio capture (RTL-SDR or simulation)
- Spectrum Sweeper (sequential channel scanning)
- RL inference (channel allocation)
- AMC inference (modulation classification)
- UI updates

Single source of truth for all inference - no duplicate processing.

Data Flow:
----------
1. Sweeper scans all channels → builds occupancy map
2. RL agent receives REAL occupancy → recommends best channel
3. Radio tunes to best channel → captures IQ
4. AMC classifies IQ → updates UI
5. Repeat
"""

from PyQt6.QtCore import QObject, pyqtSignal, QThread
import numpy as np
import time
from pathlib import Path

# Import from inference module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rl_inf import AMCClassifier, RLAgent
from radio import FullCaptureFlowgraph, SpectrumSweeper, SweepMode

# Import constants
from config import (
    SAMPLE_SIZE, N_CHANNELS, 
    SWEEP_START_FREQ, SWEEP_END_FREQ, SWEEP_DWELL_TIME
)


class DataWorker(QObject):
    """
    Worker thread for spectrum sweeping, data acquisition, and ML inference.
    
    Strategy:
    ---------
    1. Perform spectrum sweep to get REAL channel occupancy
    2. Use RL to decide best channel from real data
    3. Tune to recommended channel and capture IQ
    4. Classify with AMC and emit to UI
    
    Emits all data needed by UI widgets in a single signal.
    """
    # Signal: iq, probs, mod_class, channel, occupancy, sweep_info
    data_ready = pyqtSignal(np.ndarray, np.ndarray, str, int, np.ndarray, dict)
    sweep_progress = pyqtSignal(int, int)  # current_channel, total_channels

    def __init__(
        self, 
        radio, 
        amc: AMCClassifier, 
        rl: RLAgent, 
        sweeper: SpectrumSweeper,
        use_hardware: bool = False
    ):
        super().__init__()
        self.radio = radio
        self.amc = amc
        self.rl = rl
        self.sweeper = sweeper
        self.use_hardware = use_hardware
        self.running = True
        self.current_channel = 0
        self.n_channels = N_CHANNELS
        
        # Sweep configuration
        self.sweep_interval = 5  # Full sweep every N iterations
        self.iteration_count = 0
        self.last_sweep_result = None
        
        # EWMA smoothing for probabilities
        self.alpha = 0.15
        self.ewma_probs = None

    def run(self):
        """Main worker loop with spectrum sweeping."""
        while self.running:
            self.iteration_count += 1
            
            # =================================================================
            # STEP 1: Spectrum Sweep (get REAL channel occupancy)
            # =================================================================
            do_full_sweep = (
                self.iteration_count % self.sweep_interval == 1 or 
                self.last_sweep_result is None
            )
            
            if do_full_sweep:
                print(f"\n{'='*50}")
                print(f"🔍 SPECTRUM SWEEP #{self.sweeper.sweep_count + 1}")
                print(f"{'='*50}")
                
                # Perform sweep - this updates sweeper.channel_states
                sweep_result = self.sweeper.sweep(mode=SweepMode.SEQUENTIAL)
                self.last_sweep_result = sweep_result
                
                # Print spectrum map
                self.sweeper.print_spectrum_map()
                
                # Get REAL channel states from sweep
                channel_states = sweep_result.channel_states
            else:
                # Use cached sweep result
                channel_states = self.sweeper.get_channel_states()
            
            # =================================================================
            # STEP 2: RL Decision (based on REAL data)
            # =================================================================
            # Update RL with real channel states
            self.rl.update_observations(channel_states)
            
            # Get RL recommendation
            recommended_channel, action_probs = self.rl.get_action(channel_states)
            
            # Get spectrum holes
            spectrum_holes = self.sweeper.find_spectrum_holes(threshold=0.3)
            
            # Validate recommendation - if RL picks occupied, override
            if len(spectrum_holes) > 0:
                if channel_states[recommended_channel] >= 0.5:
                    # RL picked occupied channel - override with best hole
                    recommended_channel = self.sweeper._select_best_channel(spectrum_holes)
                    print(f"⚠️  RL override: Switched to free channel {recommended_channel}")
            
            prev_channel = self.current_channel
            self.current_channel = recommended_channel
            
            # =================================================================
            # STEP 3: Tune to Recommended Channel & Capture IQ
            # =================================================================
            if self.radio is not None and self.use_hardware:
                # Tune to recommended channel
                freq = self.sweeper.get_channel_frequency(recommended_channel)
                self.radio.set_frequency(freq)
                time.sleep(0.01)  # Let tuner settle
                iq_data = self.radio.get_iq_sample()
            else:
                # Simulation mode
                iq_data = self._simulate_iq(recommended_channel, channel_states)
            
            # =================================================================
            # STEP 4: AMC Classification of current channel
            # =================================================================
            try:
                probs = self.amc.predict_proba(iq_data)
                avg_probs = probs.mean(axis=0)
                
                # Handle shape mismatch
                n_classes = len(self.amc.CLASSES)
                if len(avg_probs) != n_classes:
                    if len(avg_probs) < n_classes:
                        avg_probs = np.pad(avg_probs, (0, n_classes - len(avg_probs)))
                    else:
                        avg_probs = avg_probs[:n_classes]
                
                # EWMA smoothing
                if self.ewma_probs is None:
                    self.ewma_probs = avg_probs
                else:
                    self.ewma_probs = self.alpha * avg_probs + (1 - self.alpha) * self.ewma_probs
                
                class_id = int(np.argmax(self.ewma_probs))
                class_id = min(class_id, n_classes - 1)
                mod_class = self.amc.CLASSES[class_id]
            except Exception as e:
                print(f"AMC error: {e}")
                self.ewma_probs = np.zeros(10)
                self.ewma_probs[0] = 1.0
                mod_class = "Unknown"
            
            # =================================================================
            # STEP 5: Build sweep info and emit
            # =================================================================
            is_free = channel_states[recommended_channel] < 0.3
            action_str = "STAY" if prev_channel == recommended_channel else f"SWITCH {prev_channel}→{recommended_channel}"
            status = "🟢 FREE" if is_free else "🔴 BUSY"
            
            print(f"[RL] Ch:{recommended_channel:2d} @ {self.sweeper.get_channel_frequency(recommended_channel)/1e6:.1f}MHz | {status} | Mod:{mod_class:8s} | {action_str}")
            
            # Build comprehensive sweep info for dashboard
            channel_frequencies = [
                self.sweeper.get_channel_frequency(ch) / 1e6 
                for ch in range(self.n_channels)
            ]
            
            # Get simulation-specific data if available
            channel_info_list = []
            try:
                if not self.use_hardware:
                    from radio.simulation import get_simulator
                    simulator = get_simulator()
                    channel_info_list = simulator.get_all_channel_info()
            except Exception:
                pass
            
            # Get QoS metrics from simulation
            qos_summary = {}
            try:
                if not self.use_hardware:
                    from radio.simulation import get_simulator
                    simulator = get_simulator()
                    qos_summary = simulator.get_qos_summary()
            except Exception:
                pass
            
            sweep_info = {
                "sweep_count": self.sweeper.sweep_count,
                "spectrum_holes": spectrum_holes.tolist(),
                "n_free": len(spectrum_holes),
                "n_occupied": self.n_channels - len(spectrum_holes),
                "recommended_channel": recommended_channel,
                "channel_freq_mhz": self.sweeper.get_channel_frequency(recommended_channel) / 1e6,
                "is_free": is_free,
                # New fields for dashboard
                "channel_states": channel_states.tolist(),
                "channel_frequencies": channel_frequencies,
                "action_probs": action_probs.tolist() if action_probs is not None else None,
                "modulation": mod_class,
                "prev_channel": prev_channel,
                "action": "STAY" if prev_channel == recommended_channel else "SWITCH",
                # Simulation-specific data
                "simulation_mode": not self.use_hardware,
                "channel_info": channel_info_list,
                # 6G QoS metrics
                "qos_summary": qos_summary,
            }
            
            # Emit ALL data
            self.data_ready.emit(
                iq_data,
                self.ewma_probs.copy(),
                mod_class,
                recommended_channel,
                channel_states.copy(),
                sweep_info
            )
            
            time.sleep(0.1)
    
    def _simulate_iq(self, channel: int, channel_states: np.ndarray) -> np.ndarray:
        """
        Generate simulated IQ based on actual simulation traffic.
        
        Uses the simulation engine's generate_iq() method which creates
        realistic IQ signals based on:
        - Service class (URLLC/mMTC/eMBB/PU/FREE)
        - Modulation type (QPSK/BPSK/64QAM/FM/Noise)
        - Power levels
        - Channel fading effects
        """
        try:
            from radio.simulation import get_simulator
            simulator = get_simulator()
            # Use the simulation's proper IQ generator
            return simulator.generate_iq(channel)
        except Exception as e:
            # Fallback to basic simulation if simulator unavailable
            N = SAMPLE_SIZE
            t = np.arange(N)
            fc = 0.1
            carrier = np.exp(1j * 2 * np.pi * fc * t)
            
            if channel_states[channel] > 0.5:
                # Occupied channel - generate QPSK-like signal
                bits = np.random.randint(0, 4, N)
                symbols = np.exp(1j * np.pi/4 * (2*bits + 1))
                signal = symbols * carrier * 0.5
                noise = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.1
            else:
                # Free channel - just noise
                signal = np.zeros(N, dtype=np.complex64)
                noise = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.05
            
            return (signal + noise).astype(np.complex64)
    
    def stop(self):
        self.running = False


class SystemController(QObject):
    """
    Main system controller coordinating radio, sweeper, inference, and UI.
    
    Architecture:
    -------------
    Radio (RTL-SDR) ──► Sweeper (scan channels) ──► RL (decide channel)
                                                          │
    UI ◄────────────────── AMC (classify) ◄──────────────┘
    
    Single source of truth for all ML inference.
    """
    # Signals for UI updates
    update_plots = pyqtSignal(np.ndarray)           # IQ data for charts
    update_probs = pyqtSignal(np.ndarray)           # Probabilities for bar graph
    update_status = pyqtSignal(str, str)            # Mod class, channel
    update_spectrum = pyqtSignal(np.ndarray)        # Channel occupancy
    update_sweep_info = pyqtSignal(dict)            # Sweep metadata
    
    def __init__(self, use_hardware: bool = False):
        super().__init__()
        
        # Model paths - all models stored in notebooks/models/
        project_root = Path(__file__).parent.parent.parent.parent
        models_path = project_root / "notebooks" / "models"
        self.amc_path = models_path / "amc_model.onnx"
        self.rl_path = models_path / "best_ppo_spectrum.zip"
        
        # Initialize radio
        self.use_hardware = use_hardware
        self.radio = None
        
        print("\n" + "="*60)
        print("🚀 COGNITIVE RADIO SYSTEM WITH SPECTRUM SWEEPING")
        print("="*60)
        
        if use_hardware:
            try:
                self.radio = FullCaptureFlowgraph()
                self.radio.start()
                print("✅ RTL-SDR: Connected and streaming")
                print(f"   └─ Using real hardware data")
            except Exception as e:
                print(f"⚠️  RTL-SDR: Hardware init failed - {e}")
                print(f"   └─ Falling back to simulation mode")
                self.use_hardware = False
        else:
            print("ℹ️  RTL-SDR: Simulation mode (no hardware)")
        
        # Initialize inference models
        print("\n📦 Loading ML Models...")
        self.amc = AMCClassifier(self.amc_path)
        if self.amc.session is not None:
            print(f"✅ AMC Model: Loaded from {self.amc_path.name}")
            print(f"   └─ Classes: {self.amc.CLASSES}")
        else:
            print(f"⚠️  AMC Model: Failed to load")
        
        self.rl = RLAgent(self.rl_path)
        
        # Initialize Spectrum Sweeper
        print("\n📡 Initializing Spectrum Sweeper...")
        self.sweeper = SpectrumSweeper(
            radio=self.radio,
            amc=self.amc,
            start_freq=SWEEP_START_FREQ,
            end_freq=SWEEP_END_FREQ,
            n_channels=N_CHANNELS,
            dwell_time=SWEEP_DWELL_TIME,
        )
        
        print("\n" + "="*60)
        print("📡 System Ready - Starting spectrum sweeping...")
        print("="*60 + "\n")
        
        # Worker thread
        self.thread = QThread()
        self.worker = DataWorker(
            self.radio, 
            self.amc, 
            self.rl, 
            self.sweeper,
            self.use_hardware
        )
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
    
    def _handle_data(self, iq_data, probs, predicted_mod, allocated_channel, occupancy, sweep_info):
        """Distribute data to all UI components."""
        self.update_plots.emit(iq_data)
        self.update_probs.emit(probs)
        
        # Format status with frequency
        freq_mhz = sweep_info.get("channel_freq_mhz", 0)
        status = "FREE" if sweep_info.get("is_free", True) else "BUSY"
        self.update_status.emit(
            f"Detected: {predicted_mod}",
            f"Ch {allocated_channel} @ {freq_mhz:.1f} MHz [{status}]"
        )
        
        self.update_spectrum.emit(occupancy)
        self.update_sweep_info.emit(sweep_info)
    
    def get_sweep_summary(self) -> dict:
        """Get current sweep summary."""
        return self.sweeper.get_sweep_summary()
    
    def set_sweep_range(self, start_freq: float, end_freq: float):
        """Update sweep frequency range."""
        self.sweeper.start_freq = start_freq
        self.sweeper.end_freq = end_freq
        self.sweeper.bandwidth = end_freq - start_freq
        self.sweeper.channel_spacing = self.sweeper.bandwidth / self.sweeper.n_channels
        self.sweeper.channel_freqs = np.array([
            start_freq + (i + 0.5) * self.sweeper.channel_spacing 
            for i in range(self.sweeper.n_channels)
        ])
        print(f"📡 Sweep range updated: {start_freq/1e6:.1f} - {end_freq/1e6:.1f} MHz")
