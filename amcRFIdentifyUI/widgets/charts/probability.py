from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication
import pyqtgraph as pg
import numpy as np
from config import PROBABILITY_BAR_COLOR, PROBABILITY_REFRESH_RATE, MODEL_PATH
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from logic.onnx_inference import ONNXInference
import sys
import time


class ProbabilityOfModulation(QObject):
    numbers_generated = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.data = []
        self.onnx_inference = ONNXInference(MODEL_PATH)
        self.alpha = 0.1  # Smoothing factor for EWMA
        self.ewma = None  # To store the exponentially weighted moving average

    def infer_from_iq_data(self, iq_data):
        # Perform ONNX inference using the captured IQ data
        probabilities = self.onnx_inference.infer_probabilities(iq_data)
        new_data = probabilities[0]

        # Calculate EWMA
        if self.ewma is None:
            self.ewma = new_data
        else:
            self.ewma = self.alpha * new_data + (1 - self.alpha) * self.ewma

        self.data = self.ewma
        self.numbers_generated.emit(self.data)


class BarGraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.prob_mod = ProbabilityOfModulation()
        self.prob_mod.numbers_generated.connect(self.update_data)

        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        self.bars = None
        self.text_items = []
        self.initialized = False  # To handle first data reception

        # Timer to refresh the plot
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_plot)
        self.timer.start(PROBABILITY_REFRESH_RATE)

        # Disable mouse interaction
        self.plot_widget.setMouseEnabled(x=False, y=False)

    def setup_chart(self, data):
        """Initializes the bar chart on first data reception"""
        y_positions = np.arange(len(data))  # Category positions on y-axis
        self.bars = pg.BarGraphItem(
            x0=0, x1=data, y0=y_positions - 0.4, y1=y_positions + 0.4, brush=PROBABILITY_BAR_COLOR)
        self.plot_widget.addItem(self.bars)

        # Configure axes
        plot_item = self.plot_widget.getPlotItem()
        plot_item.getAxis('left').setTicks(
            [[(i, f'Class {i}') for i in range(len(data))]])
        plot_item.getAxis('bottom').setLabel("Probability")
        plot_item.getAxis('left').setLabel("Modulation Classes")

        # Set fixed ranges for the axes
        plot_item.setXRange(0, 1, padding=0)
        plot_item.setYRange(-0.5, len(data) - 0.5, padding=0)

        self.add_labels(data)
        self.initialized = True  # Chart is now initialized

    def update_data(self, new_data):
        """Updates the bar chart dynamically when new data is received"""
        if not self.initialized:
            self.setup_chart(new_data)
            return

        y_positions = np.arange(len(new_data))  # Ensure correct y-positions
        self.bars.setOpts(x0=0, x1=new_data, y0=y_positions -
                          0.4, y1=y_positions + 0.4)
        # self.add_labels(new_data)

    def add_labels(self, data):
        """Adds probability value labels on the bars"""
        # Remove previous labels
        for text in self.text_items:
            self.plot_widget.removeItem(text)
        self.text_items.clear()

    def refresh_plot(self):
        # Only update if there is nonempty data.
        if self.prob_mod.data is not None and len(self.prob_mod.data) > 0:
            self.update_data(self.prob_mod.data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BarGraphWidget()
    window.show()
    sys.exit(app.exec())
