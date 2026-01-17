"""
Channel Spectrum Widget for Cognitive Radio

Displays channel occupancy as a bar chart showing:
- Which channels are occupied (red) vs free (green)
- Current RL agent's selected channel (highlighted)
- Real-time spectrum sensing results
"""

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSlot
import numpy as np


class ChannelSpectrumWidget(QWidget):
    """
    Channel spectrum display for cognitive radio.
    
    Shows N channels as colored bars:
    - Green: Free channel
    - Red: Occupied channel  
    - Cyan border: Currently selected channel by RL agent
    """
    
    def __init__(self, n_channels: int = 20):
        super().__init__()
        self.n_channels = n_channels
        self.current_channel = 0
        self.occupancy = np.zeros(n_channels)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1E1E2E')
        self.layout.addWidget(self.plot_widget)
        
        # Configure axes
        self.plot_widget.setLabel('bottom', 'Channel Index')
        self.plot_widget.setLabel('left', 'Occupancy')
        self.plot_widget.setTitle('Spectrum Channels', color='#E0E0E0')
        
        # Create bar items for each channel
        self.bars = []
        for i in range(n_channels):
            bar = pg.BarGraphItem(
                x=[i], height=[0.1], width=0.8,
                brush='#2ECC71', pen='#1E1E2E'
            )
            self.plot_widget.addItem(bar)
            self.bars.append(bar)
        
        # Selected channel indicator
        self.selection_bar = pg.BarGraphItem(
            x=[0], height=[1.1], width=0.9,
            brush=None, pen=pg.mkPen('#00E5FF', width=3)
        )
        self.plot_widget.addItem(self.selection_bar)
        
        # Set fixed ranges
        self.plot_widget.setXRange(-0.5, n_channels - 0.5)
        self.plot_widget.setYRange(0, 1.2)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        
        # Add channel labels
        x_axis = self.plot_widget.getAxis('bottom')
        x_axis.setTicks([[(i, str(i)) for i in range(0, n_channels, 2)]])
    
    @pyqtSlot(np.ndarray)
    def update_occupancy(self, occupancy: np.ndarray) -> None:
        """
        Update channel occupancy display.
        
        Parameters
        ----------
        occupancy : np.ndarray
            Array of shape (n_channels,) with values 0-1
            Higher values = more likely occupied
        """
        self.occupancy = occupancy
        for i, (bar, occ) in enumerate(zip(self.bars, occupancy)):
            # Color based on occupancy: green (free) to red (occupied)
            if occ > 0.5:
                color = '#E74C3C'  # Red - occupied
            else:
                color = '#2ECC71'  # Green - free
            
            bar.setOpts(height=[max(0.1, occ)], brush=color)
    
    def set_selected_channel(self, channel: int) -> None:
        """Highlight the RL agent's selected channel."""
        self.current_channel = channel
        self.selection_bar.setOpts(x=[channel])
    
    @pyqtSlot(int)
    def update_selection(self, channel: int) -> None:
        """Slot for updating selected channel."""
        self.set_selected_channel(channel)
