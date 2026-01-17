"""
RTL-SDR Capture Module

GNU Radio flowgraph for capturing IQ samples from RTL-SDR hardware.
Provides 1024-sample blocks for real-time processing.
"""

try:
    from gnuradio import blocks, filter, gr, soapy, analog, audio
    from gnuradio.filter import firdes
    GNURADIO_AVAILABLE = True
except ImportError:
    GNURADIO_AVAILABLE = False
    print("Warning: gnuradio module not found. Hardware capture unavailable.")


if GNURADIO_AVAILABLE:
    import numpy as np
    import time

    class FullCaptureFlowgraph(gr.top_block):
        """
        RTL-SDR capture flowgraph for real-time IQ sample acquisition.
        
        Features:
        - Captures 1024-sample blocks for ML inference
        - Optional raw IQ recording to file
        - FM demodulation for audio playback
        
        Parameters
        ----------
        samp_rate : float
            Sample rate in Hz (default: 1.024 MHz)
        radio_freq : float
            Center frequency in Hz (default: 96.5 MHz)
        """

        def __init__(self, samp_rate: float = 1.024e6, radio_freq: float = 96.5e6):
            super().__init__("FullCaptureFlowgraph")
            self.samp_rate = samp_rate
            self.radio_freq = radio_freq
            self.file_sink_connected = False
            self.file_path = None

            # RTL-SDR Source (via SoapySDR for broader hardware support)
            dev = 'driver=rtlsdr'
            stream_args = 'bufflen=16384'
            tune_args = ['']
            settings = ['']
            
            self.src = soapy.source(
                dev, "fc32", 1, '', stream_args, tune_args, settings
            )
            self.src.set_sample_rate(0, samp_rate)
            self.src.set_frequency(0, radio_freq)
            self.src.set_gain_mode(0, False)
            self.src.set_gain(0, 'TUNER', 15)

            # Constellation capture branch (1024 samples)
            self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)
            self.low_pass_filter = filter.fir_filter_ccf(
                1,
                firdes.low_pass(1.0, samp_rate, 80e3, 10e3)
            )
            self.stream_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex, 1024)
            self.vector_sink = blocks.vector_sink_c(1024)

            # File recording branch (optional)
            self.file_sink = None
            self.file_sink_connected = False

            # FM demodulation branch (optional audio output)
            self.wfm_demod = analog.wfm_rcv(
                quad_rate=int(samp_rate),
                audio_decimation=int(samp_rate // 48000),
            )
            self.audio_sink = audio.sink(48000, "", True)

            # Connect constellation branch
            self.connect(self.src, self.throttle, self.low_pass_filter)
            self.connect(self.low_pass_filter, self.stream_to_vector, self.vector_sink)

        def get_iq_sample(self) -> np.ndarray:
            """
            Get a 1024-sample IQ block for processing.
            
            Returns
            -------
            np.ndarray
                Complex64 array of shape (1024,)
            """
            while len(self.vector_sink.data()) < 1024:
                time.sleep(0.01)
            data = np.array(self.vector_sink.data()[:1024], dtype=np.complex64)
            self.vector_sink.reset()
            return data

        def open_file(self, file_path: str) -> None:
            """Start recording raw IQ to file."""
            self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, file_path, False)
            self.file_path = file_path
            self.file_sink.set_unbuffered(False)
            if not self.file_sink_connected:
                self.connect(self.low_pass_filter, self.file_sink)
                self.file_sink_connected = True
            self.file_sink.open(self.file_path)

        def close_file(self) -> None:
            """Stop recording."""
            if self.file_sink:
                self.file_sink.close()
            if self.file_sink_connected:
                self.disconnect(self.low_pass_filter, self.file_sink)
                self.file_sink_connected = False

        def mute_fm(self) -> None:
            """Mute FM audio output."""
            self.lock()
            try:
                self.disconnect(self.src, self.wfm_demod, self.audio_sink)
            except RuntimeError:
                pass
            finally:
                self.unlock()

        def unmute_fm(self) -> None:
            """Enable FM audio output."""
            self.lock()
            try:
                self.connect(self.src, self.wfm_demod, self.audio_sink)
            except RuntimeError:
                pass
            finally:
                self.unlock()

        def set_frequency(self, freq: float) -> None:
            """Set tuner frequency."""
            self.radio_freq = freq
            self.src.set_frequency(0, freq)

        def set_gain(self, gain: float) -> None:
            """Set tuner gain."""
            self.src.set_gain(0, 'TUNER', gain)

else:
    # Mock implementation when GNU Radio is not available
    import numpy as np
    
    class FullCaptureFlowgraph:
        """Mock flowgraph for testing without hardware."""
        
        def __init__(self, samp_rate: float = 1.024e6, radio_freq: float = 96.5e6):
            self.samp_rate = samp_rate
            self.radio_freq = radio_freq
            print("⚠️  Running in mock mode (no RTL-SDR hardware)")
        
        def start(self) -> None:
            pass
        
        def stop(self) -> None:
            pass
        
        def wait(self) -> None:
            pass
        
        def get_iq_sample(self) -> np.ndarray:
            """Return random noise samples."""
            return (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64) * 0.1
        
        def set_frequency(self, freq: float) -> None:
            self.radio_freq = freq
        
        def set_gain(self, gain: float) -> None:
            pass
        
        def open_file(self, file_path: str) -> None:
            pass
        
        def close_file(self) -> None:
            pass
        
        def mute_fm(self) -> None:
            pass
        
        def unmute_fm(self) -> None:
            pass
