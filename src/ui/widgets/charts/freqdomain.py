"""
Frequency Domain Chart Widget

Displays the power spectral density (PSD) of IQ signals as a bar graph.
Shows signal strength across frequency bins.
"""

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QMenu
from PyQt6.QtCore import pyqtSignal, QObject
import numpy as np
from config import FREQUENCY_DOMAIN_PLOT_COLOR


class FrequencyDomainSignal(QObject):
    """
    Signal processor for frequency domain analysis.
    
    Converts IQ samples to power spectral density using FFT.
    Emits frequency bins and magnitude data.
    
    Signals
    -------
    data_generated : pyqtSignal(np.ndarray, np.ndarray)
        Emits (frequency_bins, magnitude_dB)
    """
    data_generated = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self.data = np.array([])
        self.freq_bins = np.fft.fftfreq(1024, 1 / 1024)
        self.freq_bins = self.freq_bins[:len(self.freq_bins) // 2]

    def generate_frequency_domain_data(self, iq_data: np.ndarray) -> None:
        """
        Process IQ data and emit frequency domain representation.
        
        Parameters
        ----------
        iq_data : np.ndarray
            Complex IQ samples to transform
        """
        # Ensure the IQ data is a 1D array
        iq_data = iq_data.flatten()

        # Remove the mean to center the data
        iq_data = iq_data - np.mean(iq_data)

        # Apply a Hanning window to reduce spectral leakage
        iq_data = iq_data * np.hanning(len(iq_data))

        # Perform FFT and shift the zero frequency component to the center
        iq_data_fft = np.fft.fft(iq_data)
        iq_data_fft = np.fft.fftshift(iq_data_fft)

        # Take the magnitude of the FFT and convert to dB scale
        iq_data_magnitude = 20 * np.log10(np.abs(iq_data_fft) + 1e-6)

        # Return the frequency bins and the magnitude of the FFT
        freq_bins = np.fft.fftfreq(len(iq_data), 1 / 1024)
        freq_bins = np.fft.fftshift(freq_bins)

        self.data_generated.emit(freq_bins, iq_data_magnitude)


class PlotFrequencyDomain(QWidget):
    """
    Frequency domain visualization widget.
    
    Displays signal power spectrum as a bar graph:
    - X-axis: Frequency in kHz
    - Y-axis: Relative gain in dB
    """
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        # Set x-axis label
        self.plot_widget.setLabel('bottom', 'Frequency (kHz)')

        # Set y-axis label
        self.plot_widget.setLabel('left', 'Relative Gain (dB)')

        # Create bar graph with smaller width
        self.bar_graph = pg.BarGraphItem(
            x=[], height=[], width=1, pen=FREQUENCY_DOMAIN_PLOT_COLOR)
        self.plot_widget.addItem(self.bar_graph)

        # Initialize data generator
        self.data_gen = FrequencyDomainSignal()
        self.data_gen.data_generated.connect(self.update_plot)

        # Set initial plot limits
        self.initial_x_min = 0
        self.initial_x_max = 500
        self.initial_y_min = 0
        self.initial_y_max = 62

        # Set fixed plot limits
        self.plot_widget.setXRange(self.initial_x_min, self.initial_x_max)
        self.plot_widget.setYRange(self.initial_y_min, self.initial_y_max)
        self.plot_widget.setMouseEnabled(x=True, y=False)

    def update_plot(self, freq_bins: np.ndarray, freq_data: np.ndarray) -> None:
        """
        Update the frequency domain bar chart.
        
        Parameters
        ----------
        freq_bins : np.ndarray
            Frequency values for x-axis
        freq_data : np.ndarray
            Power values in dB for bar heights
        """
        self.bar_graph.setOpts(x=freq_bins, height=freq_data)
        self.plot_widget.setLimits(xMin=0, xMax=freq_bins[-1])
        self.plot_widget.setLimits(yMin=0, yMax=62)
        self.plot_widget.setLabel('bottom', 'Frequency (kHz)')
        self.plot_widget.setLabel('left', 'Relative Gain (dB)')
