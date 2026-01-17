from gnuradio.filter import firdes
from gnuradio import blocks, filter, gr, analog
import numpy as np
import time
from config import *


class FullCaptureFlowgraph(gr.top_block):
    """
    A dummy flowgraph that:
      - Generates 1024-sample blocks of a sine wave (instead of real RTL-SDR data).
      - Can record raw IQ to file (on demand).
      - Prints messages for FM start/stop (no audio sink).
    """

    def __init__(self, samp_rate=VARS["SAMPLE_RATE"], wave_freq=3e3, file_path=None):
        """
        :param samp_rate: Sample rate to pretend we're using (default 1.024e6)
        :param wave_freq: Frequency of the dummy sine wave
        :param file_path: Default path for saving IQ data
        """
        super().__init__("DummyNoAudioFlowgraph")
        self.samp_rate = samp_rate
        self.wave_freq = wave_freq
        self.file_path = file_path

        # ---------- (A) Sine wave source instead of RTL-SDR ----------
        self.src = analog.sig_source_c(
            samp_rate,
            analog.GR_SIN_WAVE,
            wave_freq,
            1.0,
            0.0
        )

        # ---------- (A) 1024-Sample Constellation Branch ----------
        self.throttle = blocks.throttle(gr.sizeof_gr_complex, samp_rate, True)
        self.low_pass_filter = filter.fir_filter_ccf(
            1,
            firdes.low_pass(
                1.0,
                samp_rate,
                80e3,
                10e3
            )
        )
        self.stream_to_vector = blocks.stream_to_vector(
            gr.sizeof_gr_complex, 1024)
        self.vector_sink = blocks.vector_sink_c(1024)

        # ---------- (B) Raw IQ to File Branch ----------
        self.file_sink = None
        self.file_sink_connected = False

        # ---------- (C) "FM Demod" Stub (no audio sink) ----------
        self.wfm_demod = analog.wfm_rcv(
            quad_rate=int(samp_rate),
            audio_decimation=int(samp_rate // 48000),
        )
        # NOTE: We do not connect to an audio sink, we only print messages in mute_fm/unmute_fm.

        # ---------- Connect Constellation Branch ----------
        self.connect(self.src, self.throttle, self.low_pass_filter)
        self.connect(self.low_pass_filter,
                     self.stream_to_vector, self.vector_sink)

    def get_iq_sample(self):
        """
        Returns 1024-sample dummy data from the vector sink
        (for constellation plotting, etc.).
        """
        while len(self.vector_sink.data()) < 1024:
            time.sleep(0.01)
        raw_data = self.vector_sink.data()[:1024]
        self.vector_sink.reset()
        return np.array(raw_data, dtype=np.complex64)

    # ---------- File Operations ----------
    def open_file(self):
        """
        Opens or reopens the file sink so data is written to disk.
        """
        print("Opening dummy file sink (no audio).")
        if self.file_sink is None:
            self.file_sink = blocks.file_sink(
                gr.sizeof_gr_complex,
                self.file_path,
                False
            )
            self.file_sink.set_unbuffered(False)
        if not self.file_sink_connected:
            self.connect(self.src, self.file_sink)
            self.file_sink_connected = True
        self.file_sink.open(self.file_path)

    def close_file(self):
        """
        Closes the file sink so data is no longer written.
        """
        print("Closing dummy file sink (no audio).")
        if self.file_sink is not None:
            self.file_sink.close()
        if self.file_sink_connected:
            self.disconnect(self.src, self.file_sink)
            self.file_sink_connected = False

    # ---------- FM Demod Controls ----------
    def mute_fm(self):
        """
        Just a message indicating FM is "muted" (no audio sink used).
        """
        print("FM audio muted (dummy, no audio sink).")

    def unmute_fm(self):
        """
        Just a message indicating FM is "unmuted" (no audio sink used).
        """
        print("FM audio unmuted (dummy, no audio sink).")
