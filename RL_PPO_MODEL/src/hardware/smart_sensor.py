"""
Smart Spectrum Sensor

The "God Switch" - decides if data comes from simulation or real hardware.
This is the source-agnostic interface that makes your system work with:
    - Simulated physics (IQGenerator) during development/training
    - Real RTL-SDR hardware during live demos

Architecture:
    [RTL-SDR] ─┐
               ├──► SmartSpectrumSensor ──► AMC Model ──► RL Agent
    [IQGen]  ──┘
"""

import numpy as np
from typing import Tuple, Optional
import sys

# Try importing pyrtlsdr, handle gracefully if not available
try:
    from rtlsdr import RtlSdr
    HAS_SDR = True
except ImportError:
    HAS_SDR = False

from .iq_generator import IQGenerator


class MockAMCModel:
    """
    Placeholder AMC model for testing system flow.
    Replace with your actual CNN-LSTM model.
    """
    def __init__(self):
        self.class_names = ['Noise', 'FM_PrimaryUser', 'BPSK_IoT', 'QPSK_SecondaryUser']
    
    def predict(self, iq_samples: np.ndarray) -> Tuple[int, str]:
        """
        Mock prediction based on signal energy.
        
        Parameters
        ----------
        iq_samples : np.ndarray
            Complex IQ samples
            
        Returns
        -------
        class_id : int
            Predicted class (0-3)
        class_name : str
            Human-readable class name
        """
        energy = np.mean(np.abs(iq_samples) ** 2)
        
        # Simple energy-based thresholds (replace with real model!)
        if energy < 0.01:
            return 0, 'Noise'
        elif energy > 0.5:
            return 1, 'FM_PrimaryUser'
        elif energy > 0.1:
            return 3, 'QPSK_SecondaryUser'
        else:
            return 2, 'BPSK_IoT'


class SmartSpectrumSensor:
    """
    Source-agnostic spectrum sensing interface.
    
    Can seamlessly switch between:
        - Simulation Mode: Uses IQGenerator + WirelessChannel
        - Live Mode: Uses RTL-SDR hardware
        
    The AMC model receives the same input format regardless of source.
    
    Parameters
    ----------
    live_mode : bool
        If True, use RTL-SDR hardware. If False, use simulation.
    model_path : str, optional
        Path to trained AMC model (.pth file)
    center_freq : float
        Center frequency for RTL-SDR (default: 100 MHz = FM band)
    sample_rate : float
        Sampling rate (default: 2.4 MHz)
    n_samples : int
        Samples per observation (default: 1024)
    """
    
    def __init__(
        self,
        live_mode: bool = False,
        model_path: Optional[str] = None,
        center_freq: float = 100e6,
        sample_rate: float = 2.4e6,
        n_samples: int = 1024
    ):
        self.live_mode = live_mode and HAS_SDR
        self.n_samples = n_samples
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        
        # Initialize IQ Generator (always available for simulation fallback)
        self.iq_gen = IQGenerator(n_samples=n_samples, sample_rate=sample_rate)
        
        # Initialize Hardware (if requested and available)
        if self.live_mode:
            print("📡 Initializing RTL-SDR Hardware...")
            try:
                self.sdr = RtlSdr()
                self.sdr.sample_rate = sample_rate
                self.sdr.center_freq = center_freq
                self.sdr.gain = 'auto'
                print(f"   └── ✅ Connected! Tuned to {center_freq/1e6:.1f} MHz")
            except Exception as e:
                print(f"   └── ⚠️  RTL-SDR Error: {e}")
                print(f"   └── 🔄 Falling back to simulation mode")
                self.live_mode = False
        else:
            print("💻 Initializing Physics Simulation Mode...")
        
        # Load AMC Model
        if model_path is not None:
            try:
                import torch
                print(f"🧠 Loading AMC model from {model_path}...")
                # Uncomment when you have your trained model:
                # self.model = torch.load(model_path)
                # self.model.eval()
                self.model = MockAMCModel()  # Placeholder
                print("   └── ⚠️  Using MockAMCModel (replace with real model)")
            except Exception as e:
                print(f"   └── ⚠️  Model load failed: {e}")
                self.model = MockAMCModel()
        else:
            self.model = MockAMCModel()
    
    def scan(
        self, 
        sim_ground_truth: Optional[int] = None
    ) -> Tuple[int, int, str]:
        """
        Perform spectrum sensing operation.
        
        Parameters
        ----------
        sim_ground_truth : int, optional
            Ground truth class ID for simulation mode (0-3).
            Ignored in live mode.
            
        Returns
        -------
        occupancy_state : int
            Binary occupancy: 0 = Free, 1 = Occupied
        class_id : int
            Detected modulation class (0-3)
        class_name : str
            Human-readable class name
        """
        # 1. ACQUIRE RAW DATA
        if self.live_mode:
            # PATH A: REAL HARDWARE
            samples = self.sdr.read_samples(self.n_samples)
            # Normalize real hardware data (crucial for model compatibility)
            samples = samples / (np.max(np.abs(samples)) + 1e-9)
        else:
            # PATH B: SIMULATION
            if sim_ground_truth is None:
                sim_ground_truth = 0  # Default to noise
            samples = self.iq_gen.get_iq_samples(sim_ground_truth)
        
        # 2. INFERENCE (AMC Model)
        # The model doesn't know/care if samples are real or simulated!
        class_id, class_name = self.model.predict(samples)
        
        # 3. TRANSLATE TO BINARY OCCUPANCY STATE (for RL)
        # Noise = Free (0), Everything else = Occupied (1)
        occupancy_state = 0 if class_id == 0 else 1
        
        return occupancy_state, class_id, class_name
    
    def get_raw_iq(
        self, 
        sim_ground_truth: Optional[int] = None
    ) -> np.ndarray:
        """
        Get raw IQ samples without AMC inference.
        Useful for debugging or AMC training.
        
        Returns
        -------
        iq_samples : np.ndarray
            Complex IQ samples (n_samples,)
        """
        if self.live_mode:
            samples = self.sdr.read_samples(self.n_samples)
            return samples / (np.max(np.abs(samples)) + 1e-9)
        else:
            if sim_ground_truth is None:
                sim_ground_truth = 0
            return self.iq_gen.get_iq_samples(sim_ground_truth)
    
    def tune(self, freq_hz: float) -> None:
        """
        Change RTL-SDR center frequency.
        
        Parameters
        ----------
        freq_hz : float
            New center frequency in Hz
        """
        self.center_freq = freq_hz
        if self.live_mode:
            self.sdr.center_freq = freq_hz
            print(f"📡 Retuned to {freq_hz/1e6:.1f} MHz")
    
    def close(self) -> None:
        """Release hardware resources."""
        if self.live_mode and hasattr(self, 'sdr'):
            self.sdr.close()
            print("📡 RTL-SDR disconnected")
