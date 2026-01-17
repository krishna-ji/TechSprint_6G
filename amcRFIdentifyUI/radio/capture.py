
try:
    from gnuradio import blocks, filter, gr, soapy, analog, audio
    from gnuradio.filter import firdes
except ImportError:
    print("Warning: gnuradio module not found. Using mock capture flowgraph.")
    from radio.capture_mock import FullCaptureFlowgraph
else:
    import numpy as np
    import time

    class FullCaptureFlowgraph(gr.top_block):
        """
        A single flowgraph that:
          - Captures 1024-sample blocks for plotting.
          - Can record raw IQ to file (on demand).
          - Can play FM radio (on demand).
        """

        def __init__(self, samp_rate=1.024e6, radio_freq=96.5e6):
            super().__init__("FullCaptureFlowgraph")
            self.samp_rate = samp_rate
            self.radio_freq = radio_freq
            self.file_sink_connected = False
            self.file_path = None

            # ---------- Shared RTL-SDR Source ----------
            # Note: We use soapy source for broader hardware support (RTL-SDR, HackRF, etc)
            dev = 'driver=rtlsdr'
            stream_args = 'bufflen=16384'
            tune_args = ['']
            settings = ['']
            
            try:
                self.src = soapy.source(
                    dev, "fc32", 1, '', stream_args, tune_args, settings
                )
            except Exception as e:
                 print(f"Error creating Soapy Source: {e}")
                 # Fallback or re-raise? for now let it crash if hardware fails but module exists
                 raise e

            self.src.set_sample_rate(0, samp_rate)
            self.src.set_frequency(0, radio_freq)
            self.src.set_gain_mode(0, False)
            self.src.set_gain(0, 'TUNER', 15)

            # ---------- (A) 1024-Sample Constellation Branch ----------
            self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)
            self.low_pass_filter = filter.fir_filter_ccf(
                1,
                firdes.low_pass(
                    1.0, samp_rate, 80e3, 10e3
                )
            )
            self.stream_to_vector = blocks.stream_to_vector(
                gr.sizeof_gr_complex, 1024)
            self.vector_sink = blocks.vector_sink_c(1024)

            # ---------- (B) Raw IQ to File Branch ----------
            self.file_sink = None
            self.file_sink_connected = False

            # ---------- (C) FM Demod Branch ----------
            self.wfm_demod = analog.wfm_rcv(
                quad_rate=int(samp_rate),
                audio_decimation=int(samp_rate // 48000),
            )
            self.audio_sink = audio.sink(48000, "", True)

            # Connect for FM demod
            # self.connect(self.src, self.wfm_demod, self.audio_sink)

            # ---------- Connect Constellation Branch ----------
            self.connect(self.src, self.throttle, self.low_pass_filter)
            self.connect(self.low_pass_filter,
                        self.stream_to_vector, self.vector_sink)

        def get_iq_sample(self):
            """
            Returns a 1024-sample block from the vector sink
            (for constellation plotting, etc.).
            """
            while len(self.vector_sink.data()) < 1024:
                time.sleep(0.01)
            data = np.array(self.vector_sink.data()[:1024], dtype=np.complex64)
            self.vector_sink.reset()
            return data

        # ---------- File Operations ----------
        def open_file(self, file_path):
            """
            Opens or reopens the file sink so data is written to disk.
            """
            print("Opening file sink for full IQ capture...")
            self.file_sink = blocks.file_sink(
                gr.sizeof_gr_complex, file_path, False)
            print("File path:", file_path)
            self.file_path = file_path
            self.file_sink.set_unbuffered(False)
            if not self.file_sink_connected:
                self.connect(self.low_pass_filter, self.file_sink)
                self.file_sink_connected = True
            self.file_sink.open(self.file_path)

        def close_file(self):
            """
            Closes the file sink so data is no longer written.
            """
            print("Closing file sink...")
            if self.file_sink:
                self.file_sink.close()
            if self.file_sink_connected:
                self.disconnect(self.low_pass_filter, self.file_sink)
                self.file_sink_connected = False

        # ---------- FM Demod Controls ----------
        def mute_fm(self):
            """
            Simple approach: Disconnect the audio sink, effectively muting radio.
            (We lock/unlock to avoid race conditions.)
            """
            self.lock()
            try:
                self.disconnect(self.src,
                                self.wfm_demod, self.audio_sink)
            except RuntimeError:
                pass # Already disconnected
            finally:
                self.unlock()
            print("FM audio muted.")

        def unmute_fm(self):
            """
            Reconnect the audio sink to hear radio again.
            """
            self.lock()
            try:
                self.connect(self.src, self.wfm_demod, self.audio_sink)
            except RuntimeError:
                pass # Already connected
            finally:
                self.unlock()
            print("FM audio unmuted.")

        def set_frequency(self, freq):
            """
            Sets the frequency of the RTL-SDR source.
            """
            self.radio_freq = freq
            self.src.set_frequency(0, freq)
            print(f"Updated frequency to {freq} Hz")

        def set_sample_rate(self, samp_rate):
            """
            Sets the sample rate of the RTL-SDR source.
            """
            self.samp_rate = samp_rate
            self.src.set_sample_rate(0, samp_rate)
            self.throttle.set_sample_rate(samp_rate)
            print(f"Updated sample rate to {samp_rate} Sps")
