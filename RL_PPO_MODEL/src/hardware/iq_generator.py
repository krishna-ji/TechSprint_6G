"""
IQ Sample Generator

Generates modulated baseband signals for each traffic class:
    Class 0: Noise (empty channel)
    Class 1: FM-like continuous wave (Primary User - high power)
    Class 2: BPSK (IoT device - low power)  
    Class 3: QPSK (Secondary User - medium power)

These map directly to the multi-class occupancy grid from dataset_pipeline.py
"""

import numpy as np
from typing import Tuple, Optional
from .physics import WirelessChannel


class IQGenerator:
    """
    The Signal Source: Generates specific modulations on demand.
    
    Maps occupancy grid class IDs to actual IQ waveforms that can be
    used for CNN-LSTM AMC training.
    
    Parameters
    ----------
    n_samples : int
        Number of IQ samples per observation (default: 1024)
    sample_rate : float
        Sampling rate in Hz (default: 2.4 MHz)
    seed : int, optional
        Random seed for reproducibility
        
    Class Mapping
    -------------
    0 → Noise (SNR = -10 dB)
    1 → FM/CW Primary User (SNR = 25 dB, continuous sine wave)
    2 → BPSK IoT (SNR = 8 dB, digital modulation)
    3 → QPSK Secondary User (SNR = 15 dB, digital modulation)
    """
    
    # SNR levels for each modulation type (dB)
    SNR_MAP = {
        0: -10,  # Noise floor
        1: 25,   # FM/Primary User (high power)
        2: 8,    # BPSK/IoT (low power)
        3: 15    # QPSK/Secondary User (medium power)
    }
    
    # Human-readable class names
    CLASS_NAMES = {
        0: 'Noise',
        1: 'FM_PrimaryUser',
        2: 'BPSK_IoT', 
        3: 'QPSK_SecondaryUser'
    }
    
    def __init__(
        self, 
        n_samples: int = 1024,
        sample_rate: float = 2.4e6,
        seed: Optional[int] = None
    ):
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.channel = WirelessChannel(sample_rate=sample_rate, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.t = np.arange(n_samples)
    
    def get_iq_samples(self, class_id: int) -> np.ndarray:
        """
        Generate IQ samples for a given class ID.
        
        Parameters
        ----------
        class_id : int
            0=Noise, 1=FM, 2=BPSK, 3=QPSK
            
        Returns
        -------
        iq_samples : np.ndarray
            Complex baseband samples (n_samples,), dtype=complex64
        """
        snr_db = self.SNR_MAP.get(class_id, -10)
        
        # --- CLASS 0: NOISE (Empty Channel) ---
        if class_id == 0:
            # Pure noise floor - no signal present
            tx_signal = np.zeros(self.n_samples, dtype=np.complex64)
            return self.channel.apply_channel_effects(tx_signal, snr_db=snr_db)
        
        # --- CLASS 1: FM RADIO / CW (Primary User) ---
        elif class_id == 1:
            # Continuous Wave (complex sinusoid)
            # Represents licensed primary user (e.g., FM broadcast)
            freq_offset = 0.05  # Normalized frequency
            tx_signal = np.exp(1j * 2 * np.pi * freq_offset * self.t)
            return self.channel.apply_channel_effects(tx_signal, snr_db=snr_db)
        
        # --- CLASS 2: BPSK (IoT Device - Low Power) ---
        elif class_id == 2:
            # Binary Phase Shift Keying: {+1, -1}
            # Low power IoT sensors (mMTC)
            n_symbols = self.n_samples // 8  # 8 samples per symbol
            symbols = self.rng.choice([1.0, -1.0], n_symbols)
            tx_signal = np.repeat(symbols, 8).astype(np.complex64)
            # Pad/truncate to exact length
            tx_signal = tx_signal[:self.n_samples]
            if len(tx_signal) < self.n_samples:
                tx_signal = np.pad(tx_signal, (0, self.n_samples - len(tx_signal)))
            return self.channel.apply_channel_effects(tx_signal, snr_db=snr_db)
        
        # --- CLASS 3: QPSK (Secondary User - Medium Power) ---
        elif class_id == 3:
            # Quadrature Phase Shift Keying: {1+1j, 1-1j, -1+1j, -1-1j}
            # Secondary cognitive radio user
            n_symbols = self.n_samples // 4  # 4 samples per symbol
            qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
            symbol_indices = self.rng.integers(0, 4, n_symbols)
            symbols = qpsk_constellation[symbol_indices]
            tx_signal = np.repeat(symbols, 4)
            return self.channel.apply_channel_effects(tx_signal, snr_db=snr_db)
        
        # Fallback: Return noise
        return np.zeros(self.n_samples, dtype=np.complex64)
    
    def get_iq_with_label(self, class_id: int) -> Tuple[np.ndarray, int, str]:
        """
        Generate IQ samples with associated labels.
        
        Parameters
        ----------
        class_id : int
            Class ID from occupancy grid
            
        Returns
        -------
        iq_samples : np.ndarray
            Complex IQ samples
        class_id : int
            Numeric class label (for training)
        class_name : str
            Human-readable class name
        """
        iq = self.get_iq_samples(class_id)
        class_name = self.CLASS_NAMES.get(class_id, 'Unknown')
        return iq, class_id, class_name
    
    def generate_dataset_from_grid(
        self, 
        occupancy_grid: np.ndarray,
        channel_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert multi-class occupancy grid to IQ dataset.
        
        Parameters
        ----------
        occupancy_grid : np.ndarray
            Shape [time_steps, n_channels], values in {0, 1, 2, 3}
        channel_idx : int, optional
            If provided, only generate for this channel.
            Otherwise generates for a random channel per timestep.
            
        Returns
        -------
        X : np.ndarray
            IQ samples, shape [n_samples, n_timesteps, 2] (I/Q as 2 channels)
        y : np.ndarray  
            Class labels, shape [n_timesteps,]
        """
        n_timesteps = occupancy_grid.shape[0]
        
        X_list = []
        y_list = []
        
        for t in range(n_timesteps):
            if channel_idx is not None:
                class_id = occupancy_grid[t, channel_idx]
            else:
                # Random channel selection (simulates random scanning)
                ch = self.rng.integers(0, occupancy_grid.shape[1])
                class_id = occupancy_grid[t, ch]
            
            iq = self.get_iq_samples(int(class_id))
            
            # Convert complex to 2-channel real (I, Q)
            iq_real = np.stack([iq.real, iq.imag], axis=-1)
            
            X_list.append(iq_real)
            y_list.append(class_id)
        
        X = np.array(X_list, dtype=np.float32)  # [n_timesteps, n_samples, 2]
        y = np.array(y_list, dtype=np.int64)
        
        return X, y
