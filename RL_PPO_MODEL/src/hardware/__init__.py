# Hardware Interface Package
"""
Hardware abstraction layer for 6G Cognitive Radio.

Modules:
    - physics: Wireless channel simulation (AWGN, Rayleigh fading)
    - iq_generator: Modulation waveform generation (Noise/FM/BPSK/QPSK)
    - smart_sensor: Source-agnostic sensing (Simulation or RTL-SDR)
    - rl_bridge: RL agent interface adapter
"""

from .physics import WirelessChannel
from .iq_generator import IQGenerator
from .smart_sensor import SmartSpectrumSensor
from .rl_bridge import CognitiveBrain

__all__ = ['WirelessChannel', 'IQGenerator', 'SmartSpectrumSensor', 'CognitiveBrain']
