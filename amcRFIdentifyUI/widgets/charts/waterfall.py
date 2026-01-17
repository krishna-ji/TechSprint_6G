import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer
import numpy as np
from config import FIGURE_REFRESH_RATE


class WaterfallPlotter:
    def __init__(self, fft_size, num_rows):
        self.fft_size = fft_size
        self.num_rows = num_rows
        print(self.fft_size, self.num_rows)
        self.spectrogram = np.zeros((num_rows, fft_size))

    def update_waterfall(self, PSD, imageitem):
        self.spectrogram = np.roll(self.spectrogram, 1, axis=0)
        self.spectrogram[0, :] = PSD
        transposed_spectrogram = self.spectrogram.T
        imageitem.setImage(transposed_spectrogram, autoLevels=True)


class PlotWaterfallDiagram(QWidget):
    def __init__(self, fft_size, num_rows):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        self.waterfall_plotter = WaterfallPlotter(fft_size, num_rows)

        self.imageitem = pg.ImageItem(np.zeros((num_rows, fft_size)))
        self.plot_widget.addItem(self.imageitem)
        self.plot_widget.setMouseEnabled(x=False, y=False)

        colormap = pg.colormap.get('viridis')
        lut = colormap.getLookupTable(nPts=256)
        self.imageitem.setLookupTable(lut)

        self.imageitem.setRect(pg.QtCore.QRectF(0, 0, fft_size, num_rows))
        # self.plot_widget.invertY(True)
        self.plot_widget.getAxis('left').setTicks([])

        # Set labels for the axes
        self.plot_widget.setLabel('bottom', 'Frequency (Hz)')
        self.plot_widget.setLabel('left', 'Time (s)')

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(FIGURE_REFRESH_RATE)  # Update every second

        self.latest_psd = None

    def update_plot(self, iq_data):
        # Perform FFT on the IQ data to get the Power Spectral Density (PSD)
        iq_data = iq_data.flatten()
        iq_data = iq_data - np.mean(iq_data)
        iq_data = iq_data * np.hanning(len(iq_data))
        iq_data_fft = np.fft.fft(iq_data, self.waterfall_plotter.fft_size)
        iq_data_fft = np.fft.fftshift(iq_data_fft)
        # Convert to dB scale
        self.latest_psd = 20 * np.log10(np.abs(iq_data_fft) + 1e-6)

    def refresh_plot(self):
        if self.latest_psd is not None:
            self.waterfall_plotter.update_waterfall(
                self.latest_psd, self.imageitem)


