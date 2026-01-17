"""
Radio Module - RTL-SDR IQ Capture

This module provides the hardware interface for capturing IQ samples
from RTL-SDR using GNU Radio with SoapySDR backend.

Features
--------
- 1024-sample IQ blocks for real-time ML inference
- Optional raw IQ recording to file
- FM demodulation for audio playback
- Mock implementation for testing without hardware

Usage
-----
>>> from radio import FullCaptureFlowgraph
>>> flowgraph = FullCaptureFlowgraph(samp_rate=1.024e6, radio_freq=96.5e6)
>>> flowgraph.start()
>>> iq_samples = flowgraph.get_iq_sample()  # Returns (1024,) complex64

Hardware Requirements
---------------------
- RTL-SDR dongle
- GNU Radio with SoapySDR
- Audio output for FM playback (optional)
"""

from .capture import FullCaptureFlowgraph

__all__ = ["FullCaptureFlowgraph"]
