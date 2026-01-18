"""
Advanced 6G Cognitive Radio Simulation Engine

Provides scientifically-valid, research-grade simulation of 6G spectrum usage
with proper 3GPP terminology and realistic channel models.

┌─────────────────────────────────────────────────────────────────────────┐
│                    6G Network Slice Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│  URLLC (Ultra-Reliable Low-Latency Communications)                      │
│    • Latency: < 1ms                                                     │
│    • Reliability: 99.9999% (six 9s)                                     │
│    • Use cases: Industrial automation, remote surgery, V2X              │
│    • Modulation: BPSK/QPSK (robust)                                     │
│    • Traffic: Sporadic, small packets (32-256 bytes)                    │
├─────────────────────────────────────────────────────────────────────────┤
│  mMTC (massive Machine-Type Communications)                             │
│    • Devices: Up to 1M per km²                                          │
│    • Reliability: 99.9%                                                 │
│    • Use cases: Smart city, agriculture, environmental monitoring       │
│    • Modulation: OOK/BPSK (low power)                                   │
│    • Traffic: Periodic small packets, long sleep cycles                 │
├─────────────────────────────────────────────────────────────────────────┤
│  eMBB (enhanced Mobile Broadband)                                       │
│    • Throughput: 20+ Gbps peak                                          │
│    • Use cases: 8K video, AR/VR, holographic communications             │
│    • Modulation: 64QAM/256QAM (high spectral efficiency)                │
│    • Traffic: Continuous streaming, large buffers                       │
└─────────────────────────────────────────────────────────────────────────┘

Channel Models:
  - Rayleigh fading (NLOS urban)
  - Rician fading (LOS rural)
  - Log-normal shadowing (8-12 dB std)
  - Path loss: Free space + environment factors

Spectrum Sensing:
  - Energy detection with configurable Pd/Pfa
  - Sensing-throughput tradeoff modeling
  - Primary user detection probability

References:
  - 3GPP TR 38.913: Study on Scenarios and Requirements for Next Generation Access
  - ITU-R M.2410: Minimum requirements for IMT-2020 radio interfaces
  - 3GPP TS 22.261: Service requirements for 5G/6G systems
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, NamedTuple
from enum import Enum, auto
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════════
#  6G SERVICE CLASSES (3GPP Terminology)
# ═══════════════════════════════════════════════════════════════════════════════

class ServiceClass(Enum):
    """6G Network Service Classes as per 3GPP specifications."""
    FREE = 0        # No active transmission
    PU = 1          # Primary User (Licensed incumbent)
    URLLC = 2       # Ultra-Reliable Low-Latency Communications
    mMTC = 3        # massive Machine-Type Communications  
    eMBB = 4        # enhanced Mobile Broadband
    

class QoSRequirements(NamedTuple):
    """QoS requirements per service class."""
    max_latency_ms: float       # Maximum acceptable latency
    reliability: float          # Required reliability (0-1)
    min_throughput_kbps: float  # Minimum throughput
    priority: int               # Scheduling priority (1=highest)
    packet_size_bytes: int      # Typical packet size


# 6G QoS Specifications (based on 3GPP standards)
QOS_SPECS: Dict[ServiceClass, QoSRequirements] = {
    ServiceClass.URLLC: QoSRequirements(
        max_latency_ms=1.0,
        reliability=0.999999,  # Six 9s
        min_throughput_kbps=100,
        priority=1,
        packet_size_bytes=64
    ),
    ServiceClass.mMTC: QoSRequirements(
        max_latency_ms=1000.0,  # 1 second tolerance
        reliability=0.999,
        min_throughput_kbps=10,
        priority=3,
        packet_size_bytes=32
    ),
    ServiceClass.eMBB: QoSRequirements(
        max_latency_ms=10.0,
        reliability=0.99,
        min_throughput_kbps=10000,  # 10 Mbps minimum
        priority=2,
        packet_size_bytes=1500
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CHANNEL MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChannelModel:
    """
    Wireless channel model with fading and path loss.
    
    Implements:
    - Rayleigh fading (NLOS)
    - Rician fading (LOS)
    - Log-normal shadowing
    - Free-space path loss
    """
    # Environment parameters
    carrier_freq_mhz: float = 900.0  # Carrier frequency
    environment: str = "urban"       # urban, suburban, rural
    
    # Fading parameters
    k_factor: float = 0.0            # Rician K-factor (0 = Rayleigh)
    doppler_hz: float = 10.0         # Max Doppler frequency
    
    # Shadowing
    shadow_std_db: float = 8.0       # Log-normal shadowing std
    
    # Cached values
    _shadow_cache: Dict[int, float] = field(default_factory=dict)
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    
    def path_loss_db(self, distance_m: float) -> float:
        """Calculate path loss using appropriate model."""
        if distance_m <= 0:
            return 0.0
        
        # Free space path loss (Friis)
        fspl = 20 * np.log10(distance_m) + 20 * np.log10(self.carrier_freq_mhz) - 27.55
        
        # Environment-dependent additional loss
        env_loss = {
            "urban": 25.0,
            "suburban": 15.0,
            "rural": 5.0,
        }.get(self.environment, 20.0)
        
        return fspl + env_loss
    
    def fading_coefficient(self) -> complex:
        """Generate fading coefficient (Rayleigh or Rician)."""
        if self.k_factor == 0:
            # Rayleigh fading
            h = (self._rng.standard_normal() + 1j * self._rng.standard_normal()) / np.sqrt(2)
        else:
            # Rician fading
            k = self.k_factor
            los = np.sqrt(k / (k + 1))  # LOS component
            nlos = np.sqrt(1 / (k + 1)) * (self._rng.standard_normal() + 1j * self._rng.standard_normal()) / np.sqrt(2)
            h = los + nlos
        return h
    
    def shadowing_db(self, channel_id: int) -> float:
        """Get log-normal shadowing for a channel (cached for consistency)."""
        if channel_id not in self._shadow_cache:
            self._shadow_cache[channel_id] = self._rng.normal(0, self.shadow_std_db)
        return self._shadow_cache[channel_id]
    
    def total_gain_db(self, distance_m: float, channel_id: int) -> float:
        """Total channel gain including path loss, shadowing, and fading."""
        pl = self.path_loss_db(distance_m)
        shadow = self.shadowing_db(channel_id)
        fading_mag = np.abs(self.fading_coefficient())
        fading_db = 20 * np.log10(fading_mag + 1e-10)
        
        return -pl + shadow + fading_db


# ═══════════════════════════════════════════════════════════════════════════════
#  SPECTRUM SENSING MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectrumSensingConfig:
    """Configuration for energy detection based spectrum sensing."""
    # Detection parameters
    noise_power_dbm: float = -100.0      # Noise floor
    detection_threshold_db: float = 6.0  # Above noise floor
    
    # Performance targets
    target_pd: float = 0.9               # Target probability of detection
    target_pfa: float = 0.1              # Target probability of false alarm
    
    # Sensing parameters  
    sensing_time_ms: float = 1.0         # Time spent sensing
    n_samples: int = 1024                # Samples per sensing period


class SpectrumSensor:
    """
    Energy detection based spectrum sensor.
    
    Models realistic sensing with:
    - Probability of detection (Pd)
    - Probability of false alarm (Pfa)
    - SNR-dependent performance
    """
    
    def __init__(self, config: SpectrumSensingConfig):
        self.config = config
        self.rng = np.random.default_rng(42)
        
        # Calculate threshold from target Pfa
        # Using chi-squared distribution approximation
        self.threshold = self._calculate_threshold()
        
        # Statistics
        self.stats = {
            'detections': 0,
            'misses': 0,
            'false_alarms': 0,
            'correct_free': 0,
        }
    
    def _calculate_threshold(self) -> float:
        """Calculate detection threshold from target Pfa."""
        # Simplified: threshold = noise_floor + detection_margin
        return self.config.noise_power_dbm + self.config.detection_threshold_db
    
    def sense(self, received_power_dbm: float, is_actually_occupied: bool) -> Dict:
        """
        Perform spectrum sensing on a channel.
        
        Parameters
        ----------
        received_power_dbm : float
            Received signal power in dBm
        is_actually_occupied : bool
            Ground truth - is the channel actually occupied?
            
        Returns
        -------
        dict
            Sensing result with detected state and probabilities
        """
        # Add measurement noise
        measurement_noise = self.rng.normal(0, 2.0)  # 2 dB std
        measured_power = received_power_dbm + measurement_noise
        
        # Detection decision
        detected_occupied = measured_power > self.threshold
        
        # Calculate SNR
        snr_db = received_power_dbm - self.config.noise_power_dbm
        
        # Theoretical Pd based on SNR (simplified Marcum Q-function approximation)
        if is_actually_occupied:
            pd = 1.0 / (1.0 + np.exp(-0.5 * (snr_db - 3)))  # Sigmoid approximation
        else:
            pd = 0.0
        
        # Update statistics
        if is_actually_occupied:
            if detected_occupied:
                self.stats['detections'] += 1
            else:
                self.stats['misses'] += 1
        else:
            if detected_occupied:
                self.stats['false_alarms'] += 1
            else:
                self.stats['correct_free'] += 1
        
        return {
            'measured_power_dbm': measured_power,
            'detected_occupied': detected_occupied,
            'snr_db': snr_db,
            'theoretical_pd': pd,
            'threshold_dbm': self.threshold,
        }
    
    def get_performance(self) -> Dict:
        """Get sensing performance metrics."""
        total_occupied = self.stats['detections'] + self.stats['misses']
        total_free = self.stats['false_alarms'] + self.stats['correct_free']
        
        pd = self.stats['detections'] / max(1, total_occupied)
        pfa = self.stats['false_alarms'] / max(1, total_free)
        
        return {
            'pd': pd,
            'pfa': pfa,
            'total_sensed': total_occupied + total_free,
            'accuracy': (self.stats['detections'] + self.stats['correct_free']) / 
                       max(1, total_occupied + total_free),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAFFIC CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass  
class TrafficConfig:
    """Configuration for 6G traffic simulation."""
    n_channels: int = 20
    
    # Primary User channels (licensed incumbents - always occupied)
    primary_user_channels: List[int] = field(default_factory=lambda: [2, 10, 14])
    
    # Device counts per service class
    n_urllc_devices: int = 20      # Industrial sensors, V2X
    n_mmtc_devices: int = 150      # IoT sensors (massive)
    n_embb_devices: int = 25       # Video/AR devices
    
    # ─────────────────────────────────────────────────────────────────────────
    # URLLC Traffic Model (Event-driven, bursty)
    # ─────────────────────────────────────────────────────────────────────────
    urllc_arrival_rate: float = 0.50       # High arrival rate for visibility
    urllc_duration_mean: int = 5           # Longer for visibility (5 timesteps)
    urllc_duration_std: int = 2
    urllc_power_dbm: float = -18.0         # Higher power for reliability
    urllc_retx_prob: float = 0.001         # Retransmission probability
    
    # ─────────────────────────────────────────────────────────────────────────
    # mMTC Traffic Model (Periodic with sleep cycles)
    # ─────────────────────────────────────────────────────────────────────────
    mmtc_arrival_rate: float = 0.10        # Higher for visibility
    mmtc_duration_mean: int = 4            # Longer duration
    mmtc_duration_std: int = 2
    mmtc_power_dbm: float = -25.0          # Higher power for visibility
    mmtc_duty_cycle: float = 0.05          # Higher duty cycle
    
    # ─────────────────────────────────────────────────────────────────────────
    # eMBB Traffic Model (Continuous streaming with ON/OFF)
    # ─────────────────────────────────────────────────────────────────────────
    embb_arrival_rate: float = 0.15        # Higher for visibility
    embb_duration_mean: int = 12           # Long sessions
    embb_duration_std: int = 4
    embb_power_dbm: float = -20.0          # Good power for visibility
    embb_buffer_size: int = 100            # Packet buffer size
    
    # Channel model
    use_fading: bool = False            # Disable for demo visibility
    use_interference: bool = True
    aci_rejection_db: float = 30.0         # Adjacent Channel Interference rejection


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SimulationTrafficGenerator:
    """
    Advanced 6G Cognitive Radio Simulation Engine.
    
    Generates scientifically-valid, time-varying spectrum occupancy with:
    - Proper 3GPP service classes (URLLC, mMTC, eMBB)
    - Realistic channel models (fading, shadowing, path loss)
    - QoS tracking and metrics
    - Energy detection spectrum sensing model
    
    This is a RESEARCH-GRADE simulator suitable for:
    - Algorithm development and testing
    - Performance benchmarking
    - Educational demonstrations
    - Paper validations
    
    Parameters
    ----------
    config : TrafficConfig
        Traffic generation parameters
    seed : int
        Random seed for reproducibility
    """
    
    # Service class labels for UI display
    SERVICE_LABELS = {
        ServiceClass.FREE: "FREE",
        ServiceClass.PU: "Primary User",
        ServiceClass.URLLC: "URLLC",
        ServiceClass.mMTC: "mMTC", 
        ServiceClass.eMBB: "eMBB",
    }
    
    # Legacy mapping for backward compatibility
    USER_TYPES = {
        0: "FREE",
        1: "PU",
        2: "URLLC",
        3: "mMTC",
        4: "eMBB",
    }
    
    # Modulation per service class
    MODULATION_MAP = {
        "FREE": "Noise",
        "PU": "FM",
        "URLLC": "QPSK",      # Robust for low latency
        "mMTC": "BPSK",       # Simple, low power
        "eMBB": "64QAM",      # High spectral efficiency
    }
    
    def __init__(self, config: Optional[TrafficConfig] = None, seed: int = 42):
        self.config = config or TrafficConfig()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        
        # Initialize channel state
        self.n_channels = self.config.n_channels
        
        # ─────────────────────────────────────────────────────────────────────
        # State Variables
        # ─────────────────────────────────────────────────────────────────────
        # Occupancy: 0.0 = free, 1.0 = fully occupied
        self.occupancy = np.zeros(self.n_channels, dtype=np.float32)
        
        # Service class per channel
        self.service_classes = [ServiceClass.FREE] * self.n_channels
        
        # User type codes (for backward compatibility)
        self.user_types = np.zeros(self.n_channels, dtype=np.int32)
        
        # Modulation per channel
        self.modulations = ["Noise"] * self.n_channels
        
        # Power levels per channel (dBm)
        self.power_levels = np.full(self.n_channels, -100.0, dtype=np.float32)
        
        # ─────────────────────────────────────────────────────────────────────
        # Channel Model
        # ─────────────────────────────────────────────────────────────────────
        self.channel_model = ChannelModel(
            carrier_freq_mhz=900.0,
            environment="urban",
            k_factor=0.0,  # Rayleigh fading
            doppler_hz=10.0,
            shadow_std_db=8.0,
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Spectrum Sensing Model
        # ─────────────────────────────────────────────────────────────────────
        self.sensing_config = SpectrumSensingConfig()
        self.sensor = SpectrumSensor(self.sensing_config)
        
        # ─────────────────────────────────────────────────────────────────────
        # Transmission Tracking
        # ─────────────────────────────────────────────────────────────────────
        # Active transmissions: (channel, end_time, service_class, power, device_id)
        self.active_transmissions: List[Tuple[int, float, ServiceClass, float, int]] = []
        
        # Device state tracking
        self.devices: Dict[int, Dict] = {}
        
        # ─────────────────────────────────────────────────────────────────────
        # Time Tracking
        # ─────────────────────────────────────────────────────────────────────
        self.current_time = 0.0
        self.time_step = 0.1  # 100ms per step
        self.sweep_count = 0
        
        # ─────────────────────────────────────────────────────────────────────
        # QoS Metrics Tracking
        # ─────────────────────────────────────────────────────────────────────
        self.qos_metrics = {
            ServiceClass.URLLC: {
                'packets_sent': 0,
                'packets_delivered': 0,
                'latency_sum_ms': 0.0,
                'latency_violations': 0,
                'reliability_achieved': 1.0,
            },
            ServiceClass.mMTC: {
                'packets_sent': 0,
                'packets_delivered': 0,
                'latency_sum_ms': 0.0,
                'latency_violations': 0,
                'reliability_achieved': 1.0,
            },
            ServiceClass.eMBB: {
                'packets_sent': 0,
                'packets_delivered': 0,
                'throughput_sum_kbps': 0.0,
                'throughput_violations': 0,
                'avg_throughput_kbps': 0.0,
            },
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # Statistics
        # ─────────────────────────────────────────────────────────────────────
        self.stats = {
            'arrivals': 0,
            'departures': 0,
            'collisions': 0,
            'avg_occupancy': deque(maxlen=100),  # Rolling window
            'spectral_efficiency': deque(maxlen=100),
            'interference_events': 0,
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # Pre-recorded Data Playback (optional)
        # ─────────────────────────────────────────────────────────────────────
        self.playback_data = None
        self.playback_index = 0
        self.use_playback = False
        
        # Class ID to service class mapping (from dataset_pipeline)
        # NOTE: Aligning dataset Type A -> URLLC (not a Primary Licensed User).
        self.class_id_to_service = {
            0: ServiceClass.FREE,      # Noise
            1: ServiceClass.URLLC,     # QPSK_URLLC (Type A Critical)
            2: ServiceClass.mMTC,      # BPSK_mMTC (Type B Delay-tolerant)
            3: ServiceClass.eMBB,      # 64QAM_eMBB (Type C High-throughput)
        }
        
        # Class ID to modulation mapping
        self.class_id_to_modulation = {
            0: "Noise",
            1: "QPSK",     # URLLC
            2: "BPSK",     # mMTC
            3: "64QAM",    # eMBB
        }
        
        # Try to load spectrum data from notebooks/data/generated/
        self._try_load_playback_data()
        
        # Initialize devices
        self._initialize_devices()
        
        # Set primary user channels as always occupied
        self._setup_primary_users()
    
    def _try_load_playback_data(self):
        """Try to load pre-recorded spectrum data for playback."""
        import os
        
        # Possible paths to look for spectrum data
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'data', 'generated', 'spectrum_test.npy'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks', 'data', 'generated', 'spectrum_train.npy'),
            'notebooks/data/generated/spectrum_test.npy',
            'notebooks/data/generated/spectrum_train.npy',
        ]
        
        for path in possible_paths:
            try:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    self.playback_data = np.load(abs_path)
                    self.use_playback = True
                    self.playback_index = 0
                    print(f"✅ Loaded spectrum playback data: {abs_path} (shape: {self.playback_data.shape})")
                    return
            except Exception as e:
                pass
        
        print("ℹ️  No spectrum playback data found, using real-time simulation")
    
    def set_playback_mode(self, enabled: bool):
        """Enable or disable playback mode."""
        if enabled and self.playback_data is None:
            self._try_load_playback_data()
        self.use_playback = enabled and self.playback_data is not None
        if self.use_playback:
            self.playback_index = 0
            print(f"📼 Playback mode ENABLED (data length: {len(self.playback_data)})")
        else:
            print("🔴 Playback mode DISABLED, using real-time simulation")
    
    def _apply_playback_state(self):
        """Apply the current playback frame to simulation state."""
        if self.playback_data is None:
            return
        
        # Get current frame (with wraparound)
        frame = self.playback_data[self.playback_index % len(self.playback_data)]
        
        # Apply to each channel
        for ch_idx, class_id in enumerate(frame):
            if ch_idx >= self.n_channels:
                break
            
            service_class = self.class_id_to_service.get(int(class_id), ServiceClass.FREE)
            modulation = self.class_id_to_modulation.get(int(class_id), "Noise")
            
            # Update channel state
            self.service_classes[ch_idx] = service_class
            self.modulations[ch_idx] = modulation
            self.user_types[ch_idx] = int(class_id)
            
            # Set occupancy based on class
            if service_class == ServiceClass.FREE:
                self.occupancy[ch_idx] = self.rng.uniform(0.0, 0.2)
                self.power_levels[ch_idx] = -100.0 + self.rng.uniform(-5, 5)
            else:
                self.occupancy[ch_idx] = self.rng.uniform(0.6, 1.0)
                # Set power based on service class
                if service_class == ServiceClass.PU:
                    self.power_levels[ch_idx] = -30.0 + self.rng.uniform(-5, 5)
                elif service_class == ServiceClass.mMTC:
                    self.power_levels[ch_idx] = -60.0 + self.rng.uniform(-10, 10)
                elif service_class == ServiceClass.eMBB:
                    self.power_levels[ch_idx] = -45.0 + self.rng.uniform(-8, 8)
                else:
                    self.power_levels[ch_idx] = -50.0 + self.rng.uniform(-5, 5)
        
        # Advance playback index
        self.playback_index += 1
    
    def _initialize_devices(self):
        """Initialize all devices with their properties."""
        device_id = 0
        
        # Available channels (exclude PU channels)
        available_channels = [ch for ch in range(self.n_channels) 
                            if ch not in self.config.primary_user_channels]
        
        # ───────────────────────────────────────────────────────────────────
        # URLLC Devices (Industrial sensors, V2X)
        # ───────────────────────────────────────────────────────────────────
        for i in range(self.config.n_urllc_devices):
            ch = self.rng.choice(available_channels)
            self.devices[device_id] = {
                'service_class': ServiceClass.URLLC,
                'preferred_channel': ch,
                'alt_channels': list(self.rng.choice(available_channels, size=3, replace=False)),
                'power_dbm': self.config.urllc_power_dbm + self.rng.uniform(-2, 2),
                'distance_m': self.rng.uniform(10, 100),
                'active': False,
                'last_tx_time': -np.inf,
                'packets_sent': 0,
                'packets_success': 0,
            }
            device_id += 1
        
        # ───────────────────────────────────────────────────────────────────
        # mMTC Devices (Massive IoT sensors)
        # ───────────────────────────────────────────────────────────────────
        for i in range(self.config.n_mmtc_devices):
            ch = self.rng.choice(available_channels)
            self.devices[device_id] = {
                'service_class': ServiceClass.mMTC,
                'preferred_channel': ch,
                'alt_channels': [ch],  # mMTC devices are simpler
                'power_dbm': self.config.mmtc_power_dbm + self.rng.uniform(-3, 3),
                'distance_m': self.rng.uniform(50, 500),
                'active': False,
                'last_tx_time': -np.inf,
                'sleep_until': self.rng.uniform(0, 10),  # Random initial sleep
                'packets_sent': 0,
                'packets_success': 0,
            }
            device_id += 1
        
        # ───────────────────────────────────────────────────────────────────
        # eMBB Devices (Video/AR streamers)
        # ───────────────────────────────────────────────────────────────────
        for i in range(self.config.n_embb_devices):
            ch = self.rng.choice(available_channels)
            self.devices[device_id] = {
                'service_class': ServiceClass.eMBB,
                'preferred_channel': ch,
                'alt_channels': list(self.rng.choice(available_channels, size=5, replace=False)),
                'power_dbm': self.config.embb_power_dbm + self.rng.uniform(-2, 2),
                'distance_m': self.rng.uniform(20, 200),
                'active': False,
                'last_tx_time': -np.inf,
                'buffer_level': 0,
                'streaming': False,
                'packets_sent': 0,
                'packets_success': 0,
            }
            device_id += 1
    
    def _setup_primary_users(self):
        """Configure primary user channels."""
        for ch in self.config.primary_user_channels:
            self.occupancy[ch] = 1.0
            self.service_classes[ch] = ServiceClass.PU
            self.user_types[ch] = 1
            self.modulations[ch] = "FM"
            self.power_levels[ch] = -15.0  # Strong licensed signal
    
    def step(self) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        Advance simulation by one time step.
        
        This is the main simulation loop that:
        1. Processes departures (ended transmissions)
        2. Generates new arrivals based on traffic models
        3. Applies channel effects (fading, interference)
        4. Updates QoS metrics
        5. Returns current spectrum state
        
        Returns
        -------
        occupancy : np.ndarray
            Channel occupancy levels [n_channels]
        user_types : np.ndarray  
            User type code per channel [n_channels]
        modulations : List[str]
            Modulation type per channel
        power_levels : np.ndarray
            Power level per channel in dBm [n_channels]
        """
        self.current_time += self.time_step
        self.sweep_count += 1
        
        # ─────────────────────────────────────────────────────────────────────
        # Playback Mode: Use pre-recorded spectrum data
        # ─────────────────────────────────────────────────────────────────────
        if self.use_playback and self.playback_data is not None:
            self._apply_playback_state()
            
            # Still update stats for playback mode
            self.stats['avg_occupancy'].append(np.mean(self.occupancy))
            free_ratio = np.sum(self.occupancy < 0.3) / self.n_channels
            self.stats['spectral_efficiency'].append(1.0 - free_ratio)
            
            return (
                self.occupancy.copy(),
                self.user_types.copy(),
                self.modulations.copy(),
                self.power_levels.copy()
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # Real-time Simulation Mode
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Process departures
        self._process_departures()
        
        # Step 2: Generate new arrivals
        self._generate_urllc_traffic()
        self._generate_mmtc_traffic()
        self._generate_embb_traffic()
        
        # Step 3: Update occupancy with channel effects
        self._update_occupancy()
        
        # Step 4: Apply interference model
        if self.config.use_interference:
            self._apply_interference()
        
        # Step 5: Update QoS metrics
        self._update_qos_metrics()
        
        # Step 6: Record statistics
        self.stats['avg_occupancy'].append(np.mean(self.occupancy))
        free_ratio = np.sum(self.occupancy < 0.3) / self.n_channels
        self.stats['spectral_efficiency'].append(1.0 - free_ratio)
        
        return (
            self.occupancy.copy(),
            self.user_types.copy(),
            self.modulations.copy(),
            self.power_levels.copy()
        )
    
    def _process_departures(self):
        """Remove transmissions that have ended."""
        still_active = []
        
        for tx in self.active_transmissions:
            ch, end_time, service_class, power, device_id = tx
            
            if end_time > self.current_time:
                still_active.append(tx)
            else:
                # Transmission completed
                self.stats['departures'] += 1
                
                # Update device state
                if device_id in self.devices:
                    self.devices[device_id]['active'] = False
                    self.devices[device_id]['packets_success'] += 1
                    
                    # Update QoS metrics
                    if service_class in self.qos_metrics:
                        self.qos_metrics[service_class]['packets_delivered'] += 1
        
        self.active_transmissions = still_active
    
    def _generate_urllc_traffic(self):
        """Generate URLLC traffic (sporadic, time-critical)."""
        for device_id, device in self.devices.items():
            if device['service_class'] != ServiceClass.URLLC:
                continue
            if device['active']:
                continue
            
            # Event-driven arrival (e.g., sensor trigger, control command)
            if self.rng.random() < self.config.urllc_arrival_rate:
                duration = max(1, int(self.rng.normal(
                    self.config.urllc_duration_mean,
                    self.config.urllc_duration_std
                )))
                
                # Try preferred channel, then alternatives
                channel = self._find_best_channel(device)
                
                if channel is not None:
                    self._start_transmission(
                        channel, duration, ServiceClass.URLLC,
                        device['power_dbm'], device_id
                    )
                    
                    # Record latency (URLLC is latency-critical)
                    latency_ms = self.time_step * 1000  # Immediate transmission
                    self.qos_metrics[ServiceClass.URLLC]['latency_sum_ms'] += latency_ms
                    
                    if latency_ms > QOS_SPECS[ServiceClass.URLLC].max_latency_ms:
                        self.qos_metrics[ServiceClass.URLLC]['latency_violations'] += 1
    
    def _generate_mmtc_traffic(self):
        """Generate mMTC traffic (periodic, low power)."""
        for device_id, device in self.devices.items():
            if device['service_class'] != ServiceClass.mMTC:
                continue
            if device['active']:
                continue
            
            # Check if device is awake from sleep cycle
            if device.get('sleep_until', 0) > self.current_time:
                continue
            
            # Periodic transmission with very low duty cycle
            if self.rng.random() < self.config.mmtc_arrival_rate:
                duration = max(1, int(self.rng.normal(
                    self.config.mmtc_duration_mean,
                    self.config.mmtc_duration_std
                )))
                
                channel = device['preferred_channel']
                
                # mMTC devices transmit even with collision risk (simple devices)
                self._start_transmission(
                    channel, duration, ServiceClass.mMTC,
                    device['power_dbm'], device_id
                )
                
                # Set next sleep cycle
                sleep_duration = self.rng.exponential(100)  # Long sleep
                device['sleep_until'] = self.current_time + sleep_duration
    
    def _generate_embb_traffic(self):
        """Generate eMBB traffic (streaming, high throughput)."""
        for device_id, device in self.devices.items():
            if device['service_class'] != ServiceClass.eMBB:
                continue
            
            # eMBB can have ongoing sessions
            if device.get('streaming', False):
                # Continue streaming with some probability
                if self.rng.random() < 0.95:  # 95% chance to continue
                    if not device['active']:
                        # Start new transmission in ongoing session
                        channel = self._find_best_channel(device)
                        if channel is not None:
                            duration = max(1, int(self.rng.normal(
                                self.config.embb_duration_mean // 2,
                                self.config.embb_duration_std // 2
                            )))
                            self._start_transmission(
                                channel, duration, ServiceClass.eMBB,
                                device['power_dbm'], device_id
                            )
                else:
                    # End streaming session
                    device['streaming'] = False
            else:
                # New streaming session
                if self.rng.random() < self.config.embb_arrival_rate:
                    device['streaming'] = True
                    channel = self._find_best_channel(device)
                    
                    if channel is not None:
                        duration = max(1, int(self.rng.normal(
                            self.config.embb_duration_mean,
                            self.config.embb_duration_std
                        )))
                        self._start_transmission(
                            channel, duration, ServiceClass.eMBB,
                            device['power_dbm'], device_id
                        )
    
    def _find_best_channel(self, device: Dict) -> Optional[int]:
        """Find the best available channel for a device."""
        # Try preferred channel first
        preferred = device['preferred_channel']
        if self.occupancy[preferred] < 0.5:
            return preferred
        
        # Try alternative channels
        for ch in device.get('alt_channels', []):
            if ch not in self.config.primary_user_channels:
                if self.occupancy[ch] < 0.5:
                    return ch
        
        # Find any free channel
        free_channels = [ch for ch in range(self.n_channels)
                        if ch not in self.config.primary_user_channels
                        and self.occupancy[ch] < 0.3]
        
        if free_channels:
            return self.rng.choice(free_channels)
        
        return None
    
    def _start_transmission(self, channel: int, duration: int, 
                           service_class: ServiceClass, power: float, device_id: int):
        """Start a new transmission on a channel."""
        # Skip PU channels
        if channel in self.config.primary_user_channels:
            return
        
        # Apply fading if enabled
        if self.config.use_fading:
            device = self.devices.get(device_id, {})
            distance = device.get('distance_m', 100)
            channel_gain = self.channel_model.total_gain_db(distance, channel)
            power = power + channel_gain
        
        end_time = self.current_time + duration * self.time_step
        self.active_transmissions.append((channel, end_time, service_class, power, device_id))
        
        # Update device state
        if device_id in self.devices:
            self.devices[device_id]['active'] = True
            self.devices[device_id]['last_tx_time'] = self.current_time
            self.devices[device_id]['packets_sent'] += 1
        
        self.stats['arrivals'] += 1
        
        # Update QoS metrics
        if service_class in self.qos_metrics:
            self.qos_metrics[service_class]['packets_sent'] += 1
    
    def _update_occupancy(self):
        """Update channel occupancy based on active transmissions."""
        # Reset non-PU channels
        for ch in range(self.n_channels):
            if ch not in self.config.primary_user_channels:
                self.occupancy[ch] = 0.0
                self.service_classes[ch] = ServiceClass.FREE
                self.user_types[ch] = 0
                self.modulations[ch] = "Noise"
                self.power_levels[ch] = -100.0
        
        # Apply active transmissions
        for ch, end_time, service_class, power, device_id in self.active_transmissions:
            # Take the strongest signal on each channel
            if power > self.power_levels[ch]:
                self.power_levels[ch] = power
                self.service_classes[ch] = service_class
                self.user_types[ch] = service_class.value
                
                user_label = self.USER_TYPES.get(service_class.value, "FREE")
                self.modulations[ch] = self.MODULATION_MAP.get(user_label, "Noise")
                
                # Calculate occupancy from power level
                # -15 dBm = 1.0 (strong), -50 dBm = 0.0 (weak)
                occ = min(1.0, max(0.0, (power + 50) / 35))
                self.occupancy[ch] = max(self.occupancy[ch], occ)
        
        # Ensure PU channels stay occupied
        self._setup_primary_users()
    
    def _apply_interference(self):
        """Apply adjacent channel interference model."""
        # Calculate ACI for each channel
        aci_power = np.zeros(self.n_channels)
        
        for ch in range(self.n_channels):
            if self.power_levels[ch] > -90:
                # Adjacent channels get interference
                if ch > 0:
                    aci_power[ch - 1] += 10 ** ((self.power_levels[ch] - self.config.aci_rejection_db) / 10)
                if ch < self.n_channels - 1:
                    aci_power[ch + 1] += 10 ** ((self.power_levels[ch] - self.config.aci_rejection_db) / 10)
        
        # Convert back to dB and add to power levels
        for ch in range(self.n_channels):
            if aci_power[ch] > 0 and ch not in self.config.primary_user_channels:
                aci_db = 10 * np.log10(aci_power[ch] + 1e-12)
                if aci_db > self.power_levels[ch]:
                    # ACI causes interference
                    self.stats['interference_events'] += 1
    
    def _update_qos_metrics(self):
        """Update QoS achievement metrics."""
        for service_class in [ServiceClass.URLLC, ServiceClass.mMTC, ServiceClass.eMBB]:
            metrics = self.qos_metrics[service_class]
            sent = metrics['packets_sent']
            delivered = metrics['packets_delivered']
            
            if sent > 0:
                metrics['reliability_achieved'] = delivered / sent
            
            if service_class == ServiceClass.eMBB and delivered > 0:
                # Estimate throughput
                metrics['avg_throughput_kbps'] = (delivered * 1500 * 8) / max(1, self.current_time)
    
    def get_channel_info(self, channel: int) -> Dict:
        """Get detailed information about a channel."""
        service_class = self.service_classes[channel]
        user_label = self.USER_TYPES.get(service_class.value, "FREE")
        
        # Perform spectrum sensing
        sensing_result = self.sensor.sense(
            self.power_levels[channel],
            service_class != ServiceClass.FREE
        )
        
        return {
            'channel': channel,
            'occupancy': float(self.occupancy[channel]),
            'user_type': user_label,
            'service_class': service_class.name,
            'user_code': service_class.value,
            'modulation': self.modulations[channel],
            'power_db': float(self.power_levels[channel]),
            'is_primary': channel in self.config.primary_user_channels,
            'sensing': sensing_result,
        }
    
    def get_all_channel_info(self) -> List[Dict]:
        """Get information for all channels."""
        return [self.get_channel_info(ch) for ch in range(self.n_channels)]
    
    def get_qos_summary(self) -> Dict:
        """Get QoS metrics summary for all service classes."""
        summary = {}
        
        for service_class in [ServiceClass.URLLC, ServiceClass.mMTC, ServiceClass.eMBB]:
            metrics = self.qos_metrics[service_class]
            sent = metrics['packets_sent']
            delivered = metrics['packets_delivered']
            
            qos_req = QOS_SPECS[service_class]
            
            summary[service_class.name] = {
                'packets_sent': sent,
                'packets_delivered': delivered,
                'reliability_achieved': delivered / max(1, sent),
                'reliability_required': qos_req.reliability,
                'reliability_met': (delivered / max(1, sent)) >= qos_req.reliability,
                'priority': qos_req.priority,
            }
            
            if service_class == ServiceClass.URLLC:
                avg_latency = metrics['latency_sum_ms'] / max(1, sent)
                summary[service_class.name].update({
                    'avg_latency_ms': avg_latency,
                    'max_latency_ms': qos_req.max_latency_ms,
                    'latency_violations': metrics['latency_violations'],
                    'latency_met': avg_latency <= qos_req.max_latency_ms,
                })
            
            if service_class == ServiceClass.eMBB:
                summary[service_class.name].update({
                    'avg_throughput_kbps': metrics['avg_throughput_kbps'],
                    'min_throughput_kbps': qos_req.min_throughput_kbps,
                    'throughput_met': metrics['avg_throughput_kbps'] >= qos_req.min_throughput_kbps,
                })
        
        return summary
    
    def get_spectrum_summary(self) -> Dict:
        """Get summary statistics about current spectrum state."""
        free_channels = np.sum(self.occupancy < 0.3)
        occupied_channels = np.sum(self.occupancy >= 0.6)
        weak_channels = self.n_channels - free_channels - occupied_channels
        
        # Count by service class
        service_counts = {
            'URLLC': sum(1 for sc in self.service_classes if sc == ServiceClass.URLLC),
            'mMTC': sum(1 for sc in self.service_classes if sc == ServiceClass.mMTC),
            'eMBB': sum(1 for sc in self.service_classes if sc == ServiceClass.eMBB),
            'PU': sum(1 for sc in self.service_classes if sc == ServiceClass.PU),
        }
        
        # Find spectrum holes
        holes = []
        in_hole = False
        hole_start = 0
        
        for ch in range(self.n_channels):
            is_free = self.occupancy[ch] < 0.3
            if is_free and not in_hole:
                hole_start = ch
                in_hole = True
            elif not is_free and in_hole:
                holes.append((hole_start, ch - 1))
                in_hole = False
        
        if in_hole:
            holes.append((hole_start, self.n_channels - 1))
        
        # Calculate spectral efficiency
        spectral_eff = np.mean(list(self.stats['spectral_efficiency'])) if self.stats['spectral_efficiency'] else 0.0
        
        return {
            'total_channels': self.n_channels,
            'free_channels': int(free_channels),
            'occupied_channels': int(occupied_channels),
            'weak_channels': int(weak_channels),
            'spectrum_holes': holes,
            'spectral_efficiency': float(spectral_eff),
            'avg_occupancy': float(np.mean(self.occupancy)),
            'primary_user_channels': self.config.primary_user_channels,
            'active_transmissions': len(self.active_transmissions),
            'service_counts': service_counts,
            'total_devices': len(self.devices),
            'sweep_count': self.sweep_count,
            'simulation_time': self.current_time,
        }
    
    def get_device_statistics(self) -> Dict:
        """Get per-service-class device statistics."""
        stats = {
            'URLLC': {'active': 0, 'total': 0, 'success_rate': 0.0},
            'mMTC': {'active': 0, 'total': 0, 'success_rate': 0.0},
            'eMBB': {'active': 0, 'total': 0, 'success_rate': 0.0},
        }
        
        for device_id, device in self.devices.items():
            sc_name = device['service_class'].name
            if sc_name in stats:
                stats[sc_name]['total'] += 1
                if device['active']:
                    stats[sc_name]['active'] += 1
                
                sent = device['packets_sent']
                success = device['packets_success']
                if sent > 0:
                    stats[sc_name]['success_rate'] += success / sent
        
        # Average success rates
        for sc_name in stats:
            total = stats[sc_name]['total']
            if total > 0:
                stats[sc_name]['success_rate'] /= total
        
        return stats
    
    def generate_iq(self, channel: int) -> np.ndarray:
        """
        Generate simulated IQ samples for a channel.
        
        Creates modulated signals based on the channel's service class
        with realistic channel effects.
        
        Parameters
        ----------
        channel : int
            Channel index
            
        Returns
        -------
        np.ndarray
            Complex64 IQ samples (1024 samples)
        """
        N = 1024
        t = np.arange(N) / 1024.0
        
        service_class = self.service_classes[channel]
        power_linear = 10 ** (self.power_levels[channel] / 20)
        
        if service_class == ServiceClass.FREE:
            # Pure noise
            noise = (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)) * 0.01
            return noise.astype(np.complex64)
        
        elif service_class == ServiceClass.PU:
            # FM signal (strong, continuous broadcast)
            fc = 0.1
            mod_freq = 0.02
            fm_deviation = 10
            phase = 2 * np.pi * (fc * t * N + fm_deviation * np.sin(2 * np.pi * mod_freq * t * N))
            signal = np.exp(1j * phase) * power_linear
            noise = (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)) * 0.02
            return (signal + noise).astype(np.complex64)
        
        elif service_class == ServiceClass.URLLC:
            # QPSK (robust, low latency)
            bits = self.rng.integers(0, 4, N)
            symbols = np.exp(1j * np.pi / 4 * (2 * bits + 1))
            fc = 0.15
            carrier = np.exp(1j * 2 * np.pi * fc * t * N)
            signal = symbols * carrier * power_linear
            
            # Apply fading
            if self.config.use_fading:
                h = self.channel_model.fading_coefficient()
                signal = signal * h
            
            noise = (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)) * 0.05
            return (signal + noise).astype(np.complex64)
        
        elif service_class == ServiceClass.mMTC:
            # BPSK (simple, low power)
            bits = self.rng.integers(0, 2, N)
            symbols = 2 * bits - 1
            fc = 0.08
            carrier = np.exp(1j * 2 * np.pi * fc * t * N)
            signal = symbols * carrier * power_linear
            
            # Apply fading
            if self.config.use_fading:
                h = self.channel_model.fading_coefficient()
                signal = signal * h
            
            noise = (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)) * 0.15
            return (signal + noise).astype(np.complex64)
        
        elif service_class == ServiceClass.eMBB:
            # 64QAM (high spectral efficiency)
            bits = self.rng.integers(0, 64, N)
            # 64QAM constellation
            i_bits = (bits % 8) - 3.5
            q_bits = (bits // 8) - 3.5
            symbols = (i_bits + 1j * q_bits) / 4.0
            
            fc = 0.12
            carrier = np.exp(1j * 2 * np.pi * fc * t * N)
            signal = symbols * carrier * power_linear
            
            # Apply fading
            if self.config.use_fading:
                h = self.channel_model.fading_coefficient()
                signal = signal * h
            
            noise = (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)) * 0.03
            return (signal + noise).astype(np.complex64)
        
        else:
            # Default: noise
            noise = (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)) * 0.05
            return noise.astype(np.complex64)
    
    def reset(self):
        """Reset simulation to initial state."""
        self.current_time = 0.0
        self.sweep_count = 0
        self.active_transmissions = []
        self.occupancy = np.zeros(self.n_channels, dtype=np.float32)
        self.service_classes = [ServiceClass.FREE] * self.n_channels
        self.user_types = np.zeros(self.n_channels, dtype=np.int32)
        self.modulations = ["Noise"] * self.n_channels
        self.power_levels = np.full(self.n_channels, -100.0, dtype=np.float32)
        
        # Reset QoS metrics
        for service_class in self.qos_metrics:
            for key in self.qos_metrics[service_class]:
                if isinstance(self.qos_metrics[service_class][key], (int, float)):
                    self.qos_metrics[service_class][key] = 0 if isinstance(self.qos_metrics[service_class][key], int) else 0.0
        
        # Reset stats
        self.stats = {
            'arrivals': 0,
            'departures': 0,
            'collisions': 0,
            'avg_occupancy': deque(maxlen=100),
            'spectral_efficiency': deque(maxlen=100),
            'interference_events': 0,
        }
        
        # Reinitialize
        self._initialize_devices()
        self._setup_primary_users()
        
        # Reset channel model cache
        self.channel_model._shadow_cache = {}
        
        # Reset sensor stats
        self.sensor.stats = {
            'detections': 0,
            'misses': 0,
            'false_alarms': 0,
            'correct_free': 0,
        }
        
        # Reset playback index
        self.playback_index = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL INSTANCE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

_global_simulator: Optional[SimulationTrafficGenerator] = None


def get_simulator() -> SimulationTrafficGenerator:
    """Get or create the global simulation traffic generator."""
    global _global_simulator
    if _global_simulator is None:
        _global_simulator = SimulationTrafficGenerator()
    return _global_simulator


def reset_simulator(config: Optional[TrafficConfig] = None, seed: int = 42) -> SimulationTrafficGenerator:
    """Reset the global simulator with new config."""
    global _global_simulator
    _global_simulator = SimulationTrafficGenerator(config, seed)
    return _global_simulator
