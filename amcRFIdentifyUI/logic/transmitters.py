import numpy as np
from gnuradio import gr, blocks, digital, analog, filter
from gnuradio.filter import firdes


class TransmitterBase(gr.hier_block2):
    def __init__(self, txname, constellation_points, constellation_mapping, rotational_symmetry, real_sectors, imag_sectors, width_real_sectors, width_imag_sectors, samples_per_symbol=2, excess_bw=0.35):
        gr.hier_block2.__init__(self, txname,
                                gr.io_signature(1, 1, gr.sizeof_char),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))

        self.constellation = digital.constellation_rect(
            constellation_points, constellation_mapping,
            len(constellation_points), rotational_symmetry, real_sectors, imag_sectors, width_real_sectors, width_imag_sectors
        ).base()

        self.modulator = digital.generic_mod(
            constellation=self.constellation,
            differential=True,
            samples_per_symbol=samples_per_symbol,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=False,
            log=False,
            truncate=False
        )

        # nfilts = 32
        # ntaps = nfilts * 11 * int(samples_per_symbol)
        # rrc_taps = firdes.root_raised_cosine(nfilts, nfilts, 1.0, excess_bw, ntaps)
        # self.rrc_filter = filter.fft_filter_ccc(1, rrc_taps)

        # self.connect(self, self.modulator, self.rrc_filter, self)
        self.connect(self, self.modulator, self)


class transmitter_bpsk(TransmitterBase):
    modname = "BPSK"

    def __init__(self, samples_per_symbol=2, excess_bw=0.35):
        super().__init__("transmitter_bpsk", [
            1, -1], [0, 1], 1, 1, 1, 1, 1, samples_per_symbol, excess_bw)


class transmitter_qpsk(TransmitterBase):
    modname = "QPSK"

    def __init__(self, samples_per_symbol=2, excess_bw=0.35):
        super().__init__("transmitter_qpsk",
                         [0.707+0.707j, -0.707+0.707j, -
                             0.707-0.707j, 0.707-0.707j],
                         [0, 1, 2, 3],
                         4, 2, 2, 1, 1,
                         samples_per_symbol, excess_bw)


class transmitter_8psk(TransmitterBase):
    modname = "8PSK"

    def __init__(self, samples_per_symbol=2, excess_bw=0.35):
        #     points = [0.383+0.924j, 0.924+0.383j, 0.924-0.383j, 0.383-0.924j, -0.383-0.924j,
        #   -0.924-0.383j, -0.924+0.383j, -0.383+0.924j]
        # angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        # points = np.exp(1j * angles)
        # symbols = [1,0,7,6,5,4,3,2]
        points = [0.707+0.707j, 1+0j, 0.707-0.707j, 0 -
                  1j, -0.707-0.707j, -1+0j, -0.707+0.707j, 0+1j]
        symbols = [0, 1, 2, 3, 4, 5, 6, 7]
        super().__init__("transmitter_8psk", points, symbols,
                         8, 2, 2, 1, 1, samples_per_symbol, excess_bw)


class transmitter_qam16(TransmitterBase):
    modname = "QAM16"

    def __init__(self, samples_per_symbol=2, excess_bw=0.35):
        real = [-3, -1, 1, 3]
        imag = [-3, -1, 1, 3]
        points = [r + 1j*i for r in real for i in imag]
        super().__init__("transmitter_qam16", points, list(range(16)),
                         16, 4, 4, 1, 1, samples_per_symbol, excess_bw)


class transmitter_qam64(TransmitterBase):
    modname = "QAM64"

    def __init__(self, samples_per_symbol=2, excess_bw=0.35):
        real = [-7, -5, -3, -1, 1, 3, 5, 7]
        imag = [-7, -5, -3, -1, 1, 3, 5, 7]
        points = [r + 1j*i for r in real for i in imag]
        super().__init__("transmitter_qam64", points, list(range(64)),
                         64, 8, 8, 1, 1, samples_per_symbol, excess_bw)


class transmitter_gfsk(gr.hier_block2):
    modname = "GFSK"

    def __init__(self, samples_per_symbol=2, sensitivity=1.0, bt=0.3):
        gr.hier_block2.__init__(self, "transmitter_gfsk",
                                gr.io_signature(1, 1, gr.sizeof_char),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.mod = digital.gfsk_mod(
            samples_per_symbol, sensitivity, bt, verbose=False, log=False, do_unpack=True)
        self.connect(self, self.mod, self)


class transmitter_gmsk(gr.hier_block2):
    modname = "GMSK"

    def __init__(self, samples_per_symbol=2, bt=0.3):
        gr.hier_block2.__init__(self, "transmitter_gmsk",
                                gr.io_signature(1, 1, gr.sizeof_char),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.mod = digital.gmsk_mod(
            samples_per_symbol, bt, verbose=False, log=False, do_unpack=True)
        self.connect(self, self.mod, self)


class transmitter_cpfsk(gr.hier_block2):
    modname = "CPFSK"

    def __init__(self, mod_index, samples_per_symbol):
        gr.hier_block2.__init__(self, "transmitter_cpfsk",
                                gr.io_signature(1, 1, gr.sizeof_char),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.mod = analog.cpfsk_bc(mod_index, 1.0, samples_per_symbol)
        self.connect(self, self.mod, self)


class transmitter_fm(gr.hier_block2):
    modname = "WBFM"

    def __init__(self, audio_rate, output_stream_rate, tau, max_dev):
        gr.hier_block2.__init__(self, "transmitter_fm",
                                gr.io_signature(1, 1, gr.sizeof_float),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        # self.mod = analog.wfm_tx( audio_rate, output_stream_rate, tau, max_dev)
        self.mod = analog.wfm_tx(
            audio_rate=audio_rate,
            quad_rate=output_stream_rate,
            tau=tau,
            max_dev=max_dev,
            fh=(-1.0)
        )
        self.connect(self, self.mod, self)
        # self.rate = 200e3/44.1e3


class transmitter_am(gr.hier_block2):
    modname = "AM-DSB"

    def __init__(self, audio_rate, samp_rate):
        gr.hier_block2.__init__(self, "transmitter_am",
                                gr.io_signature(1, 1, gr.sizeof_float),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.rate = audio_rate/samp_rate
        # self.rate = 200e3/44.1e3
        self.interp = filter.mmse_resampler_ff(0.0, self.rate)
        self.cnv = blocks.float_to_complex()
        self.mul = blocks.multiply_const_cc(1.0)
        self.add = blocks.add_const_cc(1.0)
        self.src = analog.sig_source_c(
            samp_rate, analog.GR_SIN_WAVE, 50e3, 1.0)
        # self.src = analog.sig_source_c(200e3, analog.GR_SIN_WAVE, 50e3, 1.0)
        self.mod = blocks.multiply_cc()
        self.connect(self, self.interp, self.cnv,
                     self.mul, self.add, self.mod, self)
        self.connect(self.src, (self.mod, 1))


class transmitter_amssb(gr.hier_block2):
    modname = "AM-SSB"

    def __init__(self, audio_rate, samp_rate):
        gr.hier_block2.__init__(self, "transmitter_amssb",
                                gr.io_signature(1, 1, gr.sizeof_float),
                                gr.io_signature(1, 1, gr.sizeof_gr_complex))
        self.rate = audio_rate/samp_rate
        # self.rate = 200e3/44.1e3
        self.interp = filter.mmse_resampler_ff(0.0, self.rate)
#        self.cnv = blocks.float_to_complex()
        self.mul = blocks.multiply_const_ff(1.0)
        self.add = blocks.add_const_ff(1.0)
        self.src = analog.sig_source_f(
            samp_rate, analog.GR_SIN_WAVE, 50e3, 1.0)
        # self.src = analog.sig_source_c(200e3, analog.GR_SIN_WAVE, 50e3, 1.0)
        self.mod = blocks.multiply_ff()
        # self.filt = filter.fir_filter_ccf(1, firdes.band_pass(1.0, 200e3, 10e3, 60e3, 0.25e3, firdes.WIN_HAMMING, 6.76))
        self.filt = filter.hilbert_fc(401)
        self.connect(self, self.interp, self.mul,
                     self.add, self.mod, self.filt, self)
        self.connect(self.src, (self.mod, 1))
