"""
Probability Bar Graph Widget

Displays AMC classification probabilities.
Receives data from SystemController - does NOT do its own inference.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication
import pyqtgraph as pg
import numpy as np
from config import PROBABILITY_BAR_COLOR
from PyQt6.QtCore import pyqtSignal, pyqtSlot
import sys


class BarGraphWidget(QWidget):
    """
    Bar graph widget for displaying modulation classification probabilities.
    
    Receives probability data from SystemController via update_probabilities slot.
    No longer does its own inference - single source of truth pattern.
    """
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        self.bars = None
        self.initialized = False
        self.data = np.array([])
        
        # Class labels
        self.class_labels = ["Noise", "FM", "BPSK", "QPSK"]

        # Disable mouse interaction
        self.plot_widget.setMouseEnabled(x=False, y=False)

    def setup_chart(self, data: np.ndarray) -> None:
        """Initialize the bar chart on first data reception."""
        y_positions = np.arange(len(data))
        self.bars = pg.BarGraphItem(
            x0=0, x1=data, 
            y0=y_positions - 0.4, y1=y_positions + 0.4, 
            brush=PROBABILITY_BAR_COLOR
        )
        self.plot_widget.addItem(self.bars)

        # Configure axes
        plot_item = self.plot_widget.getPlotItem()
        plot_item.getAxis('left').setTicks([
            [(i, self.class_labels[i]) for i in range(len(data))]
        ])
        plot_item.getAxis('bottom').setLabel("Probability")
        plot_item.getAxis('left').setLabel("Modulation Classes")

        # Set fixed ranges
        plot_item.setXRange(0, 1, padding=0)
        plot_item.setYRange(-0.5, len(data) - 0.5, padding=0)

        self.initialized = True

    @pyqtSlot(np.ndarray)
    def update_probabilities(self, probs: np.ndarray) -> None:
        """
        Receive probability data from SystemController.
        
        Parameters
        ----------
        probs : np.ndarray
            Probability distribution over classes
        """
        self.data = probs
        
        if not self.initialized:
            self.setup_chart(probs)
            return

        y_positions = np.arange(len(probs))
        self.bars.setOpts(
            x0=0, x1=probs, 
            y0=y_positions - 0.4, y1=y_positions + 0.4
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BarGraphWidget()
    # Test with dummy data
    window.update_probabilities(np.array([0.1, 0.6, 0.2, 0.1]))
    window.show()
    sys.exit(app.exec())
