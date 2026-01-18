"""
Spectrum Sweeper Module

Implements sequential spectrum sweeping for cognitive radio.
Scans across a frequency range, classifies each channel using AMC,
and identifies spectrum holes for opportunistic access.

Strategy:
---------
1. Sequential Sweep: Tune to each channel, capture IQ, classify
2. Occupancy Mapping: Build real-time channel state vector
3. Hole Detection: Find contiguous free channels
4. RL Integration: Feed real spectrum data to RL agent
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum


class SweepMode(Enum):
    """Sweep operation modes."""
    SEQUENTIAL = "sequential"   # Full sweep with AMC classification
    POWER_ONLY = "power"        # Quick power detection only
    ADAPTIVE = "adaptive"       # Skip known-occupied channels


@dataclass
class ChannelInfo:
    """Information about a single channel."""
    index: int
    frequency: float                    # Center frequency in Hz
    occupancy: float = 0.0              # 0.0 = free, 1.0 = fully occupied
    modulation: str = "Unknown"         # Detected modulation type
    confidence: float = 0.0             # Classification confidence
    power_db: float = -100.0            # Signal power in dB
    last_update: float = 0.0            # Timestamp of last measurement
    is_primary_user: bool = False       # True if strong/licensed signal
    
    def is_free(self, threshold: float = 0.3) -> bool:
        """Check if channel is considered free."""
        return self.occupancy < threshold
    
    def is_occupied(self, threshold: float = 0.5) -> bool:
        """Check if channel is definitely occupied."""
        return self.occupancy >= threshold


@dataclass  
class SweepResult:
    """Result of a complete spectrum sweep."""
    channel_states: np.ndarray          # Occupancy vector [n_channels]
    channel_info: List[ChannelInfo]     # Detailed info per channel
    spectrum_holes: np.ndarray          # Indices of free channels
    best_channel: int                   # Recommended channel
    sweep_time: float                   # Time taken for sweep (seconds)
    timestamp: float                    # When sweep completed
    
    def get_hole_ranges(self) -> List[Tuple[int, int]]:
        """Get contiguous ranges of free channels."""
        if len(self.spectrum_holes) == 0:
            return []
        
        ranges = []
        start = self.spectrum_holes[0]
        prev = start
        
        for ch in self.spectrum_holes[1:]:
            if ch == prev + 1:
                prev = ch
            else:
                ranges.append((start, prev))
                start = ch
                prev = ch
        ranges.append((start, prev))
        
        return ranges


class SpectrumSweeper:
    """
    Sequential spectrum sweeper for cognitive radio.
    
    Sweeps through frequency range, uses AMC to classify each channel,
    and builds real-time spectrum occupancy map for RL agent.
    
    Parameters
    ----------
    radio : FullCaptureFlowgraph
        RTL-SDR radio interface
    amc : AMCClassifier
        Automatic modulation classifier
    start_freq : float
        Start frequency in Hz
    end_freq : float
        End frequency in Hz
    n_channels : int
        Number of channels to divide spectrum into
    dwell_time : float
        Time to stay on each channel (seconds)
    """
    
    # Modulation classes that indicate an occupied channel
    OCCUPIED_MODULATIONS = {
        "FM", "AM-DSB", "AM-SSB", "BPSK", "QPSK", 
        "8PSK", "16QAM", "4ASK", "8ASK", "OOK"
    }
    
    # Modulations that indicate a free/noise channel
    FREE_MODULATIONS = {"Noise", "Unknown"}
    
    def __init__(
        self,
        radio,
        amc,
        start_freq: float = 88e6,
        end_freq: float = 108e6,
        n_channels: int = 20,
        dwell_time: float = 0.02,
    ):
        self.radio = radio
        self.amc = amc
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.n_channels = n_channels
        self.dwell_time = dwell_time
        
        # Calculate channel parameters
        self.bandwidth = end_freq - start_freq
        self.channel_spacing = self.bandwidth / n_channels
        
        # Channel frequencies (center of each channel)
        self.channel_freqs = np.array([
            start_freq + (i + 0.5) * self.channel_spacing 
            for i in range(n_channels)
        ])
        
        # Initialize channel info
        self.channels = [
            ChannelInfo(index=i, frequency=self.channel_freqs[i])
            for i in range(n_channels)
        ]
        
        # Current state
        self.channel_states = np.zeros(n_channels, dtype=np.float32)
        self.last_sweep_time = 0.0
        self.sweep_count = 0
        
        # EWMA smoothing for occupancy (reduce noise)
        self.alpha = 0.3  # Smoothing factor (higher = more responsive)
        
        print(f"📡 SpectrumSweeper initialized:")
        print(f"   └─ Range: {start_freq/1e6:.1f} - {end_freq/1e6:.1f} MHz")
        print(f"   └─ Channels: {n_channels} @ {self.channel_spacing/1e6:.2f} MHz spacing")
        print(f"   └─ Dwell time: {dwell_time*1000:.1f} ms/channel")
    
    def get_channel_frequency(self, channel_idx: int) -> float:
        """Get center frequency for a channel."""
        return self.channel_freqs[channel_idx]
    
    def _compute_power(self, iq_data: np.ndarray) -> float:
        """Compute signal power in dB."""
        power = np.mean(np.abs(iq_data) ** 2)
        return 10 * np.log10(power + 1e-10)
    
    def _modulation_to_occupancy(self, mod_class: str, confidence: float, power_db: float) -> float:
        """
        Convert modulation classification to occupancy level.
        
        Returns
        -------
        float
            Occupancy level [0.0 = free, 1.0 = occupied]
        """
        # Power-based thresholds (empirically tuned for RTL-SDR)
        NOISE_FLOOR = -35  # dB - below this is definitely noise
        WEAK_SIGNAL = -25  # dB - weak but detectable
        STRONG_SIGNAL = -15  # dB - definitely occupied
        
        # Very low power = noise regardless of classification
        if power_db < NOISE_FLOOR:
            return 0.0
        
        # High power = definitely occupied
        if power_db > STRONG_SIGNAL:
            return min(1.0, 0.7 + confidence * 0.3)
        
        # Check if modulation indicates clear signal
        strong_modulations = {"FM", "AM-DSB", "AM-SSB"}
        digital_modulations = {"BPSK", "QPSK", "8PSK", "16QAM", "4ASK", "8ASK", "OOK"}
        
        if mod_class in strong_modulations and confidence > 0.6:
            # Strong analog signal - high occupancy
            base = 0.6 + (power_db - WEAK_SIGNAL) / 30
            return min(1.0, base + confidence * 0.2)
        
        elif mod_class in digital_modulations and confidence > 0.5:
            # Digital signal - medium-high occupancy
            base = 0.4 + (power_db - NOISE_FLOOR) / 40
            return min(0.9, base + confidence * 0.2)
        
        elif mod_class in {"Noise", "Unknown"}:
            # Low confidence noise
            return max(0.0, min(0.2, (power_db - NOISE_FLOOR) / 50))
        
        else:
            # Default: scale by power and confidence
            power_factor = (power_db - NOISE_FLOOR) / 40
            return min(0.8, max(0.1, power_factor + confidence * 0.2))
    
    def scan_channel(self, channel_idx: int) -> ChannelInfo:
        """
        Scan a single channel and update its info.
        
        Parameters
        ----------
        channel_idx : int
            Channel index to scan
            
        Returns
        -------
        ChannelInfo
            Updated channel information
        """
        freq = self.channel_freqs[channel_idx]
        channel = self.channels[channel_idx]
        
        # =====================================================================
        # SIMULATION MODE: Use simulation's REAL occupancy data directly
        # =====================================================================
        if self.radio is None:
            try:
                from radio.simulation import get_simulator
                simulator = get_simulator()
                
                # Step the simulation forward ONCE at the start of each sweep
                # This ensures all channels see the same time snapshot
                if channel_idx == 0:
                    simulator.step()
                    # Cache all channel info at this snapshot
                    self._sim_snapshot = simulator.get_all_channel_info()
                
                # Use cached snapshot (ensures consistency)
                if hasattr(self, '_sim_snapshot') and channel_idx < len(self._sim_snapshot):
                    sim_info = self._sim_snapshot[channel_idx]
                else:
                    sim_info = simulator.get_channel_info(channel_idx)
                
                # Use simulation's occupancy directly (no IQ-based estimation)
                channel.occupancy = sim_info['occupancy']
                channel.modulation = sim_info['modulation']
                channel.power_db = sim_info['power_db']
                channel.confidence = 0.95  # Simulation is ground truth
                channel.last_update = time.time()
                channel.is_primary_user = sim_info['is_primary']
                
                # Update state vector with REAL simulation occupancy
                self.channel_states[channel_idx] = sim_info['occupancy']
                
                return channel
                
            except ImportError:
                # Fall through to IQ-based approach
                pass
        
        # =====================================================================
        # HARDWARE MODE: Use IQ capture and AMC classification
        # =====================================================================
        # Tune radio to channel frequency
        if self.radio is not None:
            self.radio.set_frequency(freq)
            time.sleep(0.005)  # Let tuner settle
            iq_data = self.radio.get_iq_sample()
        else:
            # Fallback simulation
            iq_data = self._simulate_channel_iq_basic(channel_idx)
        
        # Compute power
        power_db = self._compute_power(iq_data)
        
        # Run AMC classification
        try:
            mod_class, confidence = self.amc.predict(iq_data)
        except Exception as e:
            mod_class, confidence = "Unknown", 0.0
        
        # Compute occupancy based on power primarily
        occupancy = self._modulation_to_occupancy(mod_class, confidence, power_db)
        
        # Update channel info with EWMA smoothing
        old_occupancy = channel.occupancy
        
        # CRITICAL: If power is very low, channel is definitely free
        # Override AMC classification when signal is below noise floor
        if power_db < -35:
            # Definitely noise - fast reset to free
            smoothed_occupancy = 0.0
        elif occupancy < 0.2:
            # Probably free - decay toward zero
            smoothed_occupancy = occupancy * self.alpha + old_occupancy * (1 - self.alpha) * 0.5
        else:
            # Occupied - normal EWMA
            smoothed_occupancy = self.alpha * occupancy + (1 - self.alpha) * old_occupancy
        
        channel.occupancy = smoothed_occupancy
        channel.modulation = mod_class
        channel.confidence = confidence
        channel.power_db = power_db
        channel.last_update = time.time()
        channel.is_primary_user = (confidence > 0.8 and mod_class in {"FM", "AM-DSB"})
        
        # Update state vector
        self.channel_states[channel_idx] = smoothed_occupancy
        
        return channel
    
    def _simulate_channel_iq(self, channel_idx: int) -> np.ndarray:
        """
        Generate simulated IQ data for testing without hardware.
        
        Uses the SimulationTrafficGenerator for realistic, time-varying
        IoT traffic patterns with proper user classification.
        """
        try:
            from radio.simulation import get_simulator
            simulator = get_simulator()
            
            # Step the simulation forward periodically
            if channel_idx == 0:
                simulator.step()
            
            # Get IQ for this channel
            return simulator.generate_iq(channel_idx)
        
        except ImportError:
            # Fallback to basic simulation if module not available
            return self._simulate_channel_iq_basic(channel_idx)
    
    def _simulate_channel_iq_basic(self, channel_idx: int) -> np.ndarray:
        """Basic simulation fallback without the full traffic generator."""
        N = 1024
        t = np.arange(N)
        fc = 0.1
        carrier = np.exp(1j * 2 * np.pi * fc * t)
        
        # DYNAMIC occupancy simulation - Primary Users appear/disappear over time
        # Base stations (always on): 2, 10, 14
        # Intermittent stations: 5, 17 (50% duty cycle), 8 (30% duty cycle)
        always_occupied = {2, 10, 14}
        
        # Time-varying occupancy (changes every ~5 seconds)
        time_slot = int(time.time() / 5) % 10  # 0-9 cycle
        
        # Channels that vary with time
        if time_slot < 5:
            intermittent_occupied = {5, 17}  # First half: 5,17 are on
        else:
            intermittent_occupied = {3, 7}   # Second half: 3,7 are on instead
        
        # Random weak signals (simulates interference)
        weak_channels = {8} if time_slot % 3 == 0 else {12}
        
        occupied_channels = always_occupied | intermittent_occupied
        
        if channel_idx in occupied_channels:
            # Strong FM-like signal (high SNR)
            mod_freq = 0.03  # Fixed modulation frequency for consistency
            fm_signal = np.exp(1j * 2 * np.pi * (fc * t + 10 * np.sin(2 * np.pi * mod_freq * t)))
            signal = fm_signal * 1.0  # High power
            noise = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.05
            return (signal + noise).astype(np.complex64)
        
        elif channel_idx in weak_channels:
            # Weak QPSK-like signal
            bits = np.random.randint(0, 4, N)
            symbols = np.exp(1j * np.pi/4 * (2*bits + 1))
            signal = symbols * carrier * 0.3
            noise = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.15
            return (signal + noise).astype(np.complex64)
        
        else:
            # Free channel - very low power noise only
            # Use very small amplitude to ensure it reads as noise
            return (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64) * 0.01
    
    def sweep(self, mode: SweepMode = SweepMode.SEQUENTIAL) -> SweepResult:
        """
        Perform a complete spectrum sweep.
        
        Parameters
        ----------
        mode : SweepMode
            Sweep operation mode
            
        Returns
        -------
        SweepResult
            Complete sweep results including channel states and holes
        """
        start_time = time.time()
        
        if mode == SweepMode.SEQUENTIAL:
            # Scan all channels sequentially
            for i in range(self.n_channels):
                self.scan_channel(i)
                if self.dwell_time > 0:
                    time.sleep(self.dwell_time)
        
        elif mode == SweepMode.ADAPTIVE:
            # Skip channels that were recently scanned and stable
            for i in range(self.n_channels):
                channel = self.channels[i]
                time_since_update = time.time() - channel.last_update
                
                # Skip if recently updated and high-confidence occupied
                if time_since_update < 1.0 and channel.is_primary_user:
                    continue
                
                self.scan_channel(i)
                if self.dwell_time > 0:
                    time.sleep(self.dwell_time)
        
        sweep_time = time.time() - start_time
        self.last_sweep_time = sweep_time
        self.sweep_count += 1
        
        # Find spectrum holes
        spectrum_holes = self.find_spectrum_holes()
        
        # Determine best channel
        best_channel = self._select_best_channel(spectrum_holes)
        
        return SweepResult(
            channel_states=self.channel_states.copy(),
            channel_info=self.channels.copy(),
            spectrum_holes=spectrum_holes,
            best_channel=best_channel,
            sweep_time=sweep_time,
            timestamp=time.time()
        )
    
    def find_spectrum_holes(self, threshold: float = 0.3) -> np.ndarray:
        """
        Find free channels (spectrum holes).
        
        Parameters
        ----------
        threshold : float
            Occupancy threshold below which channel is considered free
            
        Returns
        -------
        np.ndarray
            Indices of free channels
        """
        return np.where(self.channel_states < threshold)[0]
    
    def _select_best_channel(self, spectrum_holes: np.ndarray) -> int:
        """
        Select the best channel from available holes.
        
        Strategy:
        - Prefer channels in the middle of contiguous free ranges
        - Avoid edges near occupied channels
        - Consider power levels (lower = better)
        """
        if len(spectrum_holes) == 0:
            # No free channels - return least occupied
            return int(np.argmin(self.channel_states))
        
        # Find contiguous ranges
        ranges = []
        start = spectrum_holes[0]
        prev = start
        
        for ch in spectrum_holes[1:]:
            if ch == prev + 1:
                prev = ch
            else:
                ranges.append((start, prev))
                start = ch
                prev = ch
        ranges.append((start, prev))
        
        # Find largest contiguous range
        best_range = max(ranges, key=lambda r: r[1] - r[0])
        
        # Select middle of best range
        best_channel = (best_range[0] + best_range[1]) // 2
        
        return int(best_channel)
    
    def get_channel_states(self) -> np.ndarray:
        """Get current channel occupancy states."""
        return self.channel_states.copy()
    
    def get_sweep_summary(self) -> Dict[str, Any]:
        """Get summary of current spectrum state."""
        holes = self.find_spectrum_holes()
        occupied = np.where(self.channel_states >= 0.5)[0]
        
        return {
            "n_channels": self.n_channels,
            "n_free": len(holes),
            "n_occupied": len(occupied),
            "free_channels": holes.tolist(),
            "occupied_channels": occupied.tolist(),
            "avg_occupancy": float(self.channel_states.mean()),
            "sweep_count": self.sweep_count,
            "last_sweep_time": self.last_sweep_time,
        }
    
    def print_spectrum_map(self) -> str:
        """Print ASCII spectrum map for debugging with 6G service classes."""
        lines = []
        lines.append("="*70)
        lines.append("📡 6G SPECTRUM OCCUPANCY MAP")
        lines.append("="*70)
        
        # Use cached snapshot if available, otherwise query fresh
        service_info = {}
        try:
            if hasattr(self, '_sim_snapshot') and self._sim_snapshot:
                for ch_idx, info in enumerate(self._sim_snapshot):
                    service_info[ch_idx] = info.get('service_class', 'FREE')
            else:
                from radio.simulation import get_simulator
                simulator = get_simulator()
                for ch_idx in range(self.n_channels):
                    info = simulator.get_channel_info(ch_idx)
                    service_info[ch_idx] = info.get('service_class', 'FREE')
        except:
            pass
        
        # Build visual bar with 6G service class colors
        bar = ""
        for i, ch in enumerate(self.channels):
            sc = service_info.get(i, 'FREE')
            if sc == 'FREE' or ch.occupancy < 0.3:
                bar += "🟢"  # Free
            elif sc == 'PU':
                bar += "📻"  # Primary User (broadcast)
            elif sc == 'URLLC':
                bar += "⚡"  # URLLC (ultra-reliable)
            elif sc == 'mMTC':
                bar += "📡"  # mMTC (IoT sensors)
            elif sc == 'eMBB':
                bar += "📺"  # eMBB (broadband)
            else:
                bar += "🔴"  # Unknown occupied
        
        lines.append(f"Channels: {bar}")
        lines.append(f"          {''.join([str(i%10) for i in range(self.n_channels)])}")
        lines.append("")
        
        # Legend
        lines.append("Legend: 🟢FREE  📻PU  ⚡URLLC  📡mMTC  📺eMBB")
        lines.append("")
        
        holes = self.find_spectrum_holes()
        lines.append(f"🔍 Free channels: {holes.tolist()}")
        lines.append(f"⭐ Recommended: CH {self._select_best_channel(holes)}")
        lines.append("="*70)
        
        result = "\n".join(lines)
        print(result)
        return result
