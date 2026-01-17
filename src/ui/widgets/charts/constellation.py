"""
Constellation Diagram Widget

Displays IQ samples as a scatter plot (constellation diagram).
Used for visualizing modulation schemes like BPSK, QPSK, etc.
Supports both RTL-SDR hardware capture and external data driving.
"""

from config import FIGURE_REFRESH_RATE, CONSTELLATION_PLOT_COLOR
import logging
import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
import pyqtgraph as pg
import threading
import time
import sys
from pathlib import Path

# Import from radio module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from radio import FullCaptureFlowgraph

logging.basicConfig(level=logging.DEBUG)


class DataUpdateSignal(QObject):
    """Qt signal wrapper for thread-safe IQ data updates."""
    data_updated = pyqtSignal(np.ndarray)


class PlotConstellation(QWidget):
    """
    Constellation diagram widget for IQ visualization.
    
    Displays complex IQ samples as a 2D scatter plot where:
    - X-axis: In-phase (I) component
    - Y-axis: Quadrature (Q) component
    
    Supports two modes:
    1. Self-driven: Captures IQ from RTL-SDR hardware directly
    2. External-driven: Receives IQ from SystemController
    
    Parameters
    ----------
    external_drive : bool
        If True, waits for external data via update_from_external().
        If False, creates own capture thread.
    
    Signals
    -------
    iq_data_captured : pyqtSignal(np.ndarray)
        Emits captured IQ samples for downstream widgets
    """
    iq_data_captured = pyqtSignal(np.ndarray)  # Signal to emit IQ data

    def __init__(self, external_drive: bool = False):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        # Disable auto range and mouse interaction for zooming/panning
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setXRange(-1, 1)
        self.plot_widget.setYRange(-1, 1)
        
        # Scatter plot item
        self.scatter = self.plot_widget.plot([], [], pen=None,
                                             symbol='o',
                                             symbolBrush=CONSTELLATION_PLOT_COLOR,
                                             symbolPen='w')

        self.external_drive = external_drive
        if not self.external_drive:
            # Create the unified flowgraph
            self.flowgraph = FullCaptureFlowgraph()
            self.flowgraph.start()  # Start capturing AND enabling the FM branch

            # Set up a signal for updating the plot
            self.data_update_signal = DataUpdateSignal()
            self.data_update_signal.data_updated.connect(self.update_plot)

            # Create a thread to fetch IQ data regularly
            self.capture_thread = threading.Thread(target=self.run_capture)
            self.capture_thread.start()

    def update_from_external(self, iq_data: np.ndarray) -> None:
        """
        Receive IQ data from SystemController.
        
        Parameters
        ----------
        iq_data : np.ndarray
            Complex IQ samples from external source
        """
        self.update_plot(iq_data)
        self.iq_data_captured.emit(iq_data)

    def run_capture(self) -> None:
        """Background thread for continuous IQ capture from hardware."""
        while True:
            iq_data = self.flowgraph.get_iq_sample()
            self.data_update_signal.data_updated.emit(iq_data)
            self.iq_data_captured.emit(iq_data)
            time.sleep(FIGURE_REFRESH_RATE / 1000)

    def update_plot(self, iq_data: np.ndarray) -> None:
        """
        Update constellation display with new IQ samples.
        
        Parameters
        ----------
        iq_data : np.ndarray
            Complex IQ samples to display
        """
        x = np.real(iq_data)
        y = np.imag(iq_data)
        x = self.normalize_array(x)
        y = self.normalize_array(y)
        self.scatter.setData(x, y)

    def normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalize array to [-1, 1] range.
        
        Parameters
        ----------
        arr : np.ndarray
            Input array
            
        Returns
        -------
        np.ndarray
            Normalized array in range [-1, 1]
        """
        epsilon = 1e-8
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        if arr_range < epsilon:
            return arr * 0
        return 2 * (arr - arr_min) / (arr_range + epsilon) - 1

    def closeEvent(self, event) -> None:
        """Clean up flowgraph and threads on widget close."""
        logging.debug("Closing, stopping flowgraph...")
        if not self.external_drive and hasattr(self, 'flowgraph'):
            self.flowgraph.stop()
            self.flowgraph.wait()
            if hasattr(self, 'capture_thread'):
                self.capture_thread.join(timeout=1)
            logging.debug("Flowgraph stopped.")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotConstellation()
    window.show()
    sys.exit(app.exec())
