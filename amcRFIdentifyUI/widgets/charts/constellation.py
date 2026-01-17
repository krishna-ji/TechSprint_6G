from radio.capture import FullCaptureFlowgraph
from config import FIGURE_REFRESH_RATE, CONSTELLATION_PLOT_COLOR
import logging
import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
import pyqtgraph as pg
import threading
import time
import sys

logging.basicConfig(level=logging.DEBUG)


class DataUpdateSignal(QObject):
    data_updated = pyqtSignal(np.ndarray)


class PlotConstellation(QWidget):
    iq_data_captured = pyqtSignal(np.ndarray)  # Signal to emit IQ data

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        # Disable auto range and mouse interaction for zooming/panning
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setXRange(-1, 1)
        self.plot_widget.setYRange(-1, 1)

        # Create the unified flowgraph
        self.flowgraph = FullCaptureFlowgraph()
        self.flowgraph.start()  # Start capturing AND enabling the FM branch

        # Set up a signal for updating the plot
        self.data_update_signal = DataUpdateSignal()
        self.data_update_signal.data_updated.connect(self.update_plot)

        # Create a thread to fetch IQ data regularly
        self.capture_thread = threading.Thread(target=self.run_capture)
        self.capture_thread.start()

        # Scatter plot item
        self.scatter = self.plot_widget.plot([], [], pen=None,
                                             symbol='o',
                                             symbolBrush=CONSTELLATION_PLOT_COLOR,
                                             symbolPen='w')

    def run_capture(self):
        while True:
            iq_data = self.flowgraph.get_iq_sample()
            self.data_update_signal.data_updated.emit(iq_data)
            self.iq_data_captured.emit(iq_data)
            time.sleep(FIGURE_REFRESH_RATE / 1000)

    def update_plot(self, iq_data):
        # Example normalization for display
        x = np.real(iq_data)
        y = np.imag(iq_data)
        x = self.normalize_array(x)
        y = self.normalize_array(y)
        self.scatter.setData(x, y)

    def normalize_array(self, arr):
        epsilon = 1e-8
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_range = arr_max - arr_min
        if arr_range < epsilon:
            return arr * 0
        return 2 * (arr - arr_min) / (arr_range + epsilon) - 1

    def closeEvent(self, event):
        logging.debug("Closing, stopping flowgraph...")
        self.flowgraph.stop()
        self.flowgraph.wait()
        self.capture_thread.join(timeout=1)
        logging.debug("Flowgraph stopped.")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotConstellation()
    window.show()
    sys.exit(app.exec())
