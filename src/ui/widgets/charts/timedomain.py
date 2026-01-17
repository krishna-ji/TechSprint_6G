"""
Time Domain Chart Widget

Displays IQ signal components (In-phase and Quadrature) over time.
Shows the raw waveform of received signals.
"""

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QMenu
from PyQt6.QtCore import pyqtSignal, QObject
import numpy as np
from config import TIME_DOMAIN_PLOT_COLOR


class TimeDomainSignal(QObject):
    """
    Signal processor for time domain visualization.
    
    Converts IQ samples to time-series data for plotting.
    
    Signals
    -------
    time_domain : pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
        Emits (time_vector, in_phase, quadrature)
    """
    time_domain = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()

    def generate_time_domain_data(self, iq_data: np.ndarray, sampling_rate: float) -> None:
        """
        Process IQ data and emit time domain components.
        
        Parameters
        ----------
        iq_data : np.ndarray
            Complex IQ samples
        sampling_rate : float
            Sample rate in Hz for time axis calculation
        """
        # Calculate time vector
        t = np.arange(len(iq_data)) / sampling_rate
        inphase = np.real(iq_data)
        quadrature = np.imag(iq_data)
        self.time_domain.emit(t, inphase, quadrature)


class PlotTimeDomain(QWidget):
    """
    Time domain visualization widget.
    
    Displays I/Q signal components:
    - Red line: In-phase (I) component
    - Blue line: Quadrature (Q) component
    - X-axis: Time in seconds
    - Y-axis: Amplitude
    """
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        # Initialize data generator
        self.data_gen = TimeDomainSignal()
        self.data_gen.time_domain.connect(self.update_plot)

        # Create line plots for in-phase and quadrature components
        self.inphase_plot = pg.PlotDataItem(
            [], [], pen='r')  # Red for in-phase
        self.quadrature_plot = pg.PlotDataItem(
            [], [], pen='b')  # Blue for quadrature
        self.plot_widget.addItem(self.inphase_plot)
        self.plot_widget.addItem(self.quadrature_plot)

        # Set x-axis label
        self.plot_widget.setLabel('bottom', 'Time (s)')

        # Set y-axis label
        self.plot_widget.setLabel('left', 'Amplitude')

        # Set initial plot limits
        # self.initial_x_min = 0
        # self.initial_x_max = 1e6
        # self.initial_y_min = -1
        # self.initial_y_max = 1
        # self.plot_widget.setLimits(
        #     xMin=self.initial_x_min, xMax=self.initial_x_max)
        # self.plot_widget.setRange(
        #     yRange=(self.initial_y_min, self.initial_y_max))
        self.plot_widget.setMouseEnabled(x=True, y=True)

    def update_plot(self, t: np.ndarray, inphase: np.ndarray, quadrature: np.ndarray) -> None:
        """
        Update the time domain plot with new data.
        
        Parameters
        ----------
        t : np.ndarray
            Time vector in seconds
        inphase : np.ndarray
            In-phase (I) component values
        quadrature : np.ndarray
            Quadrature (Q) component values
        """
        self.inphase_plot.setData(t, inphase)
        self.quadrature_plot.setData(t, quadrature)
