"""
Wireless Channel Physics Engine

Simulates real-world signal degradation effects:
- Log-Distance Path Loss
- Rayleigh Fading (multipath interference)
- AWGN (Additive White Gaussian Noise)

Reference: 3GPP TR 38.901 (Channel models for 5G/6G)
"""

import numpy as np
from typing import Optional


class WirelessChannel:
    """
    The Physics Engine: Applies real-world degradations to signals.
    
    Defense Point: "Simulates Log-Distance Path Loss and Rayleigh Fading
    according to 3GPP TR 38.901 channel models."
    
    Parameters
    ----------
    sample_rate : float
        Sampling rate in Hz (default: 2.4 MHz for RTL-SDR compatibility)
    seed : int, optional
        Random seed for reproducibility
        
    Attributes
    ----------
    fs : float
        Sampling frequency
    rng : numpy.random.Generator
        Random number generator
    """
    
    def __init__(self, sample_rate: float = 2.4e6, seed: Optional[int] = None):
        self.fs = sample_rate
        self.rng = np.random.default_rng(seed)
    
    def apply_channel_effects(
        self, 
        tx_signal: np.ndarray, 
        snr_db: float,
        enable_fading: bool = True
    ) -> np.ndarray:
        """
        Corrupts a clean signal with noise and fading.
        
        Mathematical Model:
            y(t) = h * x(t) + n(t)
            
            where:
                x(t) = transmitted signal
                h ~ CN(0, 1) = Rayleigh fading coefficient
                n(t) ~ CN(0, σ²) = AWGN noise
                σ² = P_signal / SNR_linear
        
        Parameters
        ----------
        tx_signal : np.ndarray
            Complex baseband transmitted signal
        snr_db : float
            Signal-to-Noise Ratio in dB
        enable_fading : bool
            Whether to apply Rayleigh fading (default: True)
            
        Returns
        -------
        rx_signal : np.ndarray
            Received signal with channel effects (complex64)
        """
        n_samples = len(tx_signal)
        
        # 1. Calculate Noise Power based on SNR
        # Signal Power is assumed to be ~1.0 after normalization
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        
        # 2. Add White Gaussian Noise (AWGN)
        # Complex noise: n = n_I + j*n_Q, each ~ N(0, σ²/2)
        noise = (
            self.rng.normal(0, 1, n_samples) + 
            1j * self.rng.normal(0, 1, n_samples)
        ) * np.sqrt(noise_power / 2)
        
        # 3. Add Rayleigh Fading (Multipath Interference)
        # Represents signal bouncing off walls/buildings
        # h ~ CN(0, 1) → |h| follows Rayleigh distribution
        if enable_fading:
            h_fading = (
                self.rng.normal(0, 1) + 
                1j * self.rng.normal(0, 1)
            ) / np.sqrt(2)
        else:
            h_fading = 1.0  # No fading (AWGN-only channel)
        
        # 4. Received Signal = (Signal * Channel) + Noise
        rx_signal = (tx_signal * h_fading) + noise
        
        return rx_signal.astype(np.complex64)
    
    def apply_path_loss(
        self, 
        signal: np.ndarray, 
        distance_m: float,
        freq_hz: float = 100e6,
        path_loss_exp: float = 3.5
    ) -> np.ndarray:
        """
        Apply log-distance path loss model.
        
        PL(d) = PL(d₀) + 10n*log₁₀(d/d₀)
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        distance_m : float
            Distance from transmitter in meters
        freq_hz : float
            Carrier frequency in Hz
        path_loss_exp : float
            Path loss exponent (2=free space, 3.5=urban)
            
        Returns
        -------
        attenuated_signal : np.ndarray
            Signal with path loss applied
        """
        c = 3e8  # Speed of light
        wavelength = c / freq_hz
        d0 = 1.0  # Reference distance (1 meter)
        
        # Free-space path loss at reference distance
        pl_d0 = (4 * np.pi * d0 / wavelength) ** 2
        
        # Total path loss
        pl_total = pl_d0 * (distance_m / d0) ** path_loss_exp
        
        # Convert to linear attenuation
        attenuation = 1.0 / np.sqrt(pl_total)
        
        return signal * attenuation
