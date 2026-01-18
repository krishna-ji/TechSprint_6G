"""
Spectrum Occupancy Matrix Widget

A visual grid showing real-time spectrum occupancy across channels and time.
Displays Primary/Secondary/Tertiary user identification with color coding.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont
import numpy as np
from collections import deque


class SpectrumMatrixWidget(QWidget):
    """
    Real-time spectrum occupancy matrix visualization.
    
    Shows a time-frequency grid where:
    - X-axis: Channels (frequency)
    - Y-axis: Time (history)
    - Colors: Occupancy level and user type
    """
    
    # Color scheme for occupancy
    COLORS = {
        'free': QColor(46, 204, 113),      # Green - available
        'weak': QColor(241, 196, 15),      # Yellow - weak signal
        'occupied': QColor(231, 76, 60),   # Red - primary user
        'secondary': QColor(52, 152, 219), # Blue - secondary user (us)
        'background': QColor(30, 30, 46),  # Dark background
        'grid': QColor(60, 60, 80),        # Grid lines
        'text': QColor(200, 200, 200),     # Text color
    }
    
    def __init__(self, n_channels: int = 20, history_depth: int = 20):
        super().__init__()
        self.n_channels = n_channels
        self.history_depth = history_depth
        
        # Occupancy history (deque for efficient rolling)
        self.occupancy_history = deque(maxlen=history_depth)
        
        # Initialize with empty history
        for _ in range(history_depth):
            self.occupancy_history.append(np.zeros(n_channels))
        
        # Current state
        self.current_channel = -1  # Our selected channel
        self.channel_info = {}     # Dict of channel -> info
        
        # Widget setup
        self.setMinimumSize(400, 250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("SPECTRUM OCCUPANCY MATRIX")
        title.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 12px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.setSpacing(15)
        
        for label, color_key in [("🟢 FREE", 'free'), ("🟡 WEAK", 'weak'), 
                                   ("🔴 PU", 'occupied'), ("🔵 SU (You)", 'secondary')]:
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {self.COLORS[color_key].name()}; font-size: 10px;")
            legend_layout.addWidget(lbl)
        
        legend_layout.addStretch()
        layout.addLayout(legend_layout)
        
        # The matrix will be drawn in paintEvent
        self.matrix_area = QWidget()
        self.matrix_area.setMinimumHeight(180)
        layout.addWidget(self.matrix_area, stretch=1)
        
        # Channel labels at bottom
        self.channel_labels = QLabel("Ch: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19")
        self.channel_labels.setStyleSheet("color: #888; font-size: 9px; font-family: monospace;")
        self.channel_labels.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.channel_labels)
    
    def paintEvent(self, event):
        """Draw the spectrum matrix."""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate cell dimensions
        margin_left = 40
        margin_top = 60
        margin_right = 10
        margin_bottom = 30
        
        width = self.width() - margin_left - margin_right
        height = self.height() - margin_top - margin_bottom
        
        cell_width = width / self.n_channels
        cell_height = height / self.history_depth
        
        # Draw background
        painter.fillRect(self.rect(), self.COLORS['background'])
        
        # Draw cells
        for t_idx, occupancy_row in enumerate(self.occupancy_history):
            for ch_idx, occ in enumerate(occupancy_row):
                x = margin_left + ch_idx * cell_width
                y = margin_top + t_idx * cell_height
                
                # Determine color based on occupancy
                if ch_idx == self.current_channel:
                    color = self.COLORS['secondary']  # Our channel
                elif occ < 0.3:
                    color = self.COLORS['free']
                elif occ < 0.6:
                    color = self.COLORS['weak']
                else:
                    color = self.COLORS['occupied']
                
                # Fade older entries
                alpha = int(255 * (0.4 + 0.6 * (t_idx / self.history_depth)))
                color.setAlpha(alpha)
                
                # Draw cell
                painter.fillRect(int(x)+1, int(y)+1, 
                               int(cell_width)-2, int(cell_height)-2, 
                               color)
        
        # Draw grid
        painter.setPen(QPen(self.COLORS['grid'], 1))
        for i in range(self.n_channels + 1):
            x = margin_left + i * cell_width
            painter.drawLine(int(x), margin_top, int(x), margin_top + height)
        
        for i in range(self.history_depth + 1):
            y = margin_top + i * cell_height
            painter.drawLine(margin_left, int(y), margin_left + width, int(y))
        
        # Draw axis labels
        painter.setPen(QPen(self.COLORS['text'], 1))
        painter.setFont(QFont("Consolas", 8))
        
        # Y-axis label (Time)
        painter.save()
        painter.translate(15, int(margin_top + height/2))
        painter.rotate(-90)
        painter.drawText(-30, 0, "Time →")
        painter.restore()
        
        # X-axis label (Frequency)
        painter.drawText(int(margin_left + width/2 - 30), int(self.height() - 5), "Frequency (Channels) →")
        
        # Draw current time marker
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        y = margin_top + (self.history_depth - 1) * cell_height
        painter.drawLine(int(margin_left - 5), int(y + cell_height/2), 
                        int(margin_left), int(y + cell_height/2))
        painter.drawText(5, int(y + cell_height/2 + 4), "NOW")
    
    @pyqtSlot(np.ndarray)
    def update_occupancy(self, occupancy: np.ndarray):
        """Update with new occupancy data."""
        if len(occupancy) == self.n_channels:
            self.occupancy_history.append(occupancy.copy())
            self.update()
    
    def set_current_channel(self, channel: int):
        """Set the channel we're currently using."""
        self.current_channel = channel
        self.update()
    
    def set_channel_info(self, channel: int, info: dict):
        """Set info for a specific channel."""
        self.channel_info[channel] = info
        self.update()


class SpectrumHeatmapWidget(QWidget):
    """
    Compact heatmap showing current spectrum state.
    Single row of colored cells with hover info.
    """
    
    COLORS = {
        'free': "#2ECC71",
        'weak': "#F1C40F", 
        'occupied': "#E74C3C",
        'secondary': "#3498DB",
    }
    
    def __init__(self, n_channels: int = 20):
        super().__init__()
        self.n_channels = n_channels
        self.occupancy = np.zeros(n_channels)
        self.modulations = ["--"] * n_channels
        self.current_channel = -1
        
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        # Title
        title = QLabel("CURRENT SPECTRUM STATE")
        title.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 11px;")
        layout.addWidget(title)
        
        # Heatmap row (using labels)
        self.cell_layout = QHBoxLayout()
        self.cell_layout.setSpacing(2)
        
        self.cells = []
        for i in range(n_channels):
            cell = QLabel(f"{i}")
            cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cell.setMinimumSize(25, 30)
            cell.setStyleSheet(f"""
                background-color: {self.COLORS['free']};
                color: black;
                font-size: 9px;
                font-weight: bold;
                border-radius: 3px;
            """)
            self.cells.append(cell)
            self.cell_layout.addWidget(cell)
        
        layout.addLayout(self.cell_layout)
    
    @pyqtSlot(np.ndarray)
    def update_occupancy(self, occupancy: np.ndarray):
        """Update cell colors based on occupancy."""
        self.occupancy = occupancy
        
        for i, (cell, occ) in enumerate(zip(self.cells, occupancy)):
            if i == self.current_channel:
                color = self.COLORS['secondary']
                text_color = "white"
            elif occ < 0.3:
                color = self.COLORS['free']
                text_color = "black"
            elif occ < 0.6:
                color = self.COLORS['weak']
                text_color = "black"
            else:
                color = self.COLORS['occupied']
                text_color = "white"
            
            cell.setStyleSheet(f"""
                background-color: {color};
                color: {text_color};
                font-size: 9px;
                font-weight: bold;
                border-radius: 3px;
            """)
    
    def set_current_channel(self, channel: int):
        """Highlight our current channel."""
        self.current_channel = channel
        self.update_occupancy(self.occupancy)
    
    def set_modulations(self, modulations: list):
        """Update modulation labels for hover."""
        self.modulations = modulations
