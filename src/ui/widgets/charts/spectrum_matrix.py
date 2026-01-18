"""
6G Spectrum Occupancy Matrix Widget

A visual grid showing real-time spectrum occupancy across channels and time.
Displays 6G service classes: URLLC, mMTC, eMBB, Primary User, FREE.

Features:
- Time-frequency grid visualization
- 6G service class color coding
- Rolling history buffer
- Real-time updates from simulation
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont
import numpy as np
from collections import deque


class MatrixCanvas(QWidget):
    """Canvas widget that actually draws the spectrum matrix grid."""
    
    # 6G-aware color scheme
    COLORS = {
        'FREE': QColor(46, 204, 113),       # Green - available
        'PU': QColor(231, 76, 60),           # Red - primary user (licensed)
        'URLLC': QColor(255, 165, 0),        # Orange - ultra-reliable
        'mMTC': QColor(0, 200, 200),         # Cyan - massive IoT
        'eMBB': QColor(186, 85, 211),        # Magenta - broadband
        'secondary': QColor(52, 152, 219),   # Blue - our channel
        'background': QColor(30, 30, 46),    # Dark background
        'grid': QColor(60, 60, 80),          # Grid lines
        'text': QColor(200, 200, 200),       # Text color
        'weak': QColor(241, 196, 15),        # Yellow - weak/unknown
    }
    
    def __init__(self, n_channels: int, history_depth: int, parent=None):
        super().__init__(parent)
        self.n_channels = n_channels
        self.history_depth = history_depth
        self.current_channel = -1
        
        # Data storage
        self.occupancy_history = deque(maxlen=history_depth)
        self.service_class_history = deque(maxlen=history_depth)
        
        # Initialize with empty data
        for _ in range(history_depth):
            self.occupancy_history.append(np.zeros(n_channels))
            self.service_class_history.append(['FREE'] * n_channels)
        
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)
    
    def paintEvent(self, event):
        """Draw the spectrum matrix grid."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), self.COLORS['background'])
        
        # Margins for labels
        margin_left = 30
        margin_top = 5
        margin_right = 5
        margin_bottom = 20
        
        # Calculate drawing area
        draw_width = self.width() - margin_left - margin_right
        draw_height = self.height() - margin_top - margin_bottom
        
        if draw_width <= 0 or draw_height <= 0:
            return
        
        cell_width = draw_width / self.n_channels
        cell_height = draw_height / self.history_depth
        
        # Draw cells
        for t_idx, (occupancy_row, service_row) in enumerate(zip(
            self.occupancy_history, self.service_class_history
        )):
            for ch_idx, (occ, service_class) in enumerate(zip(occupancy_row, service_row)):
                x = margin_left + ch_idx * cell_width
                y = margin_top + t_idx * cell_height
                
                # Determine color
                if ch_idx == self.current_channel:
                    color = QColor(self.COLORS['secondary'])
                elif occ < 0.2 or service_class == 'FREE':
                    color = QColor(self.COLORS['FREE'])
                elif service_class == 'PU':
                    color = QColor(self.COLORS['PU'])
                elif service_class == 'URLLC':
                    color = QColor(self.COLORS['URLLC'])
                elif service_class == 'mMTC':
                    color = QColor(self.COLORS['mMTC'])
                elif service_class == 'eMBB':
                    color = QColor(self.COLORS['eMBB'])
                else:
                    if occ < 0.3:
                        color = QColor(self.COLORS['FREE'])
                    elif occ < 0.6:
                        color = QColor(self.COLORS['weak'])
                    else:
                        color = QColor(self.COLORS['PU'])
                
                # Fade older entries
                alpha = int(255 * (0.4 + 0.6 * (t_idx / self.history_depth)))
                color.setAlpha(alpha)
                
                # Draw cell
                painter.fillRect(int(x)+1, int(y)+1, 
                               max(1, int(cell_width)-2), max(1, int(cell_height)-2), 
                               color)
        
        # Draw grid lines
        painter.setPen(QPen(self.COLORS['grid'], 1))
        for i in range(self.n_channels + 1):
            x = margin_left + i * cell_width
            painter.drawLine(int(x), margin_top, int(x), margin_top + draw_height)
        
        for i in range(self.history_depth + 1):
            y = margin_top + i * cell_height
            painter.drawLine(margin_left, int(y), margin_left + draw_width, int(y))
        
        # Draw labels
        painter.setPen(QPen(self.COLORS['text'], 1))
        painter.setFont(QFont("Consolas", 7))
        
        # Y-axis: Time arrow
        painter.save()
        painter.translate(12, int(margin_top + draw_height/2))
        painter.rotate(-90)
        painter.drawText(-20, 0, "Time →")
        painter.restore()
        
        # NOW marker
        y_now = margin_top + (self.history_depth - 1) * cell_height
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawText(3, int(y_now + cell_height/2 + 4), "NOW")
        
        # X-axis: Channel numbers (every 5)
        painter.setPen(QPen(self.COLORS['text'], 1))
        for ch in range(0, self.n_channels, 5):
            x = margin_left + ch * cell_width + cell_width/2
            painter.drawText(int(x)-4, self.height()-3, str(ch))
        
        painter.end()
    
    def update_data(self, occupancy: np.ndarray, service_classes: list):
        """Update the matrix data."""
        if len(occupancy) == self.n_channels:
            self.occupancy_history.append(occupancy.copy())
            if service_classes and len(service_classes) == self.n_channels:
                self.service_class_history.append(service_classes.copy())
            else:
                derived = ['FREE' if o < 0.3 else 'PU' for o in occupancy]
                self.service_class_history.append(derived)
            self.update()
    
    def set_current_channel(self, channel: int):
        """Set the currently selected channel."""
        self.current_channel = channel
        self.update()


class SpectrumMatrixWidget(QWidget):
    """
    Real-time 6G spectrum occupancy matrix visualization.
    
    Shows a time-frequency grid where:
    - X-axis: Channels (frequency)
    - Y-axis: Time (history)
    - Colors: 6G service class identification
    
    Color Coding:
    - 🟢 GREEN: FREE (no active transmission)
    - 📻 RED: Primary User (broadcast/licensed)
    - ⚡ ORANGE: URLLC (ultra-reliable low-latency)
    - 📡 CYAN: mMTC (massive IoT)
    - 📺 MAGENTA: eMBB (enhanced broadband)
    - 🔵 BLUE: Our secondary user
    """
    
    # 6G-aware color scheme (for compatibility)
    COLORS = MatrixCanvas.COLORS
    
    def __init__(self, n_channels: int = 20, history_depth: int = 20):
        super().__init__()
        self.n_channels = n_channels
        self.history_depth = history_depth
        
        # Current state
        self.current_channel = -1
        self.channel_info = {}
        self.current_service_classes = ['FREE'] * n_channels
        
        # Keep references for compatibility
        self.occupancy_history = None  # Will be accessed via canvas
        self.service_class_history = None
        
        # Widget setup
        self.setMinimumSize(250, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Title
        title = QLabel("📡 6G SPECTRUM OCCUPANCY MATRIX")
        title.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 6G Legend
        legend_layout = QHBoxLayout()
        legend_layout.setSpacing(4)
        
        legend_items = [
            ("🟢FREE", 'FREE'),
            ("📻PU", 'PU'),
            ("⚡URLLC", 'URLLC'),
            ("📡mMTC", 'mMTC'),
            ("📺eMBB", 'eMBB'),
            ("🔵YOU", 'secondary'),
        ]
        
        for label, color_key in legend_items:
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {self.COLORS[color_key].name()}; font-size: 8px; font-weight: bold;")
            legend_layout.addWidget(lbl)
        
        legend_layout.addStretch()
        layout.addLayout(legend_layout)
        
        # The canvas widget that draws the actual matrix
        self.canvas = MatrixCanvas(n_channels, history_depth, self)
        layout.addWidget(self.canvas, stretch=1)
        
        # Link to canvas data for compatibility
        self.occupancy_history = self.canvas.occupancy_history
        self.service_class_history = self.canvas.service_class_history
    
    @pyqtSlot(np.ndarray)
    def update_occupancy(self, occupancy: np.ndarray, service_classes: list = None):
        """Update with new occupancy data and optional service class info."""
        if service_classes is None:
            service_classes = ['FREE' if o < 0.3 else 'PU' for o in occupancy]
        self.canvas.update_data(occupancy, service_classes)
        self.current_service_classes = service_classes.copy() if service_classes else ['FREE'] * self.n_channels
    
    def update_with_channel_info(self, channel_info_list: list):
        """Update using detailed channel info from simulation."""
        if len(channel_info_list) == self.n_channels:
            occupancy = np.array([ch['occupancy'] for ch in channel_info_list])
            service_classes = [ch.get('service_class', 'FREE') for ch in channel_info_list]
            self.canvas.update_data(occupancy, service_classes)
            self.current_service_classes = service_classes
    
    def set_current_channel(self, channel: int):
        """Set the channel we're currently using."""
        self.current_channel = channel
        self.canvas.set_current_channel(channel)
    
    def set_channel_info(self, channel: int, info: dict):
        """Set info for a specific channel."""
        self.channel_info[channel] = info
        self.canvas.update()


class SpectrumHeatmapWidget(QWidget):
    """
    Compact 6G-aware heatmap showing current spectrum state.
    Single row of colored cells representing each channel's 6G service class.
    """
    
    # 6G color scheme
    COLORS = {
        'FREE': "#2ECC71",       # Green
        'PU': "#E74C3C",         # Red  
        'URLLC': "#FFA500",      # Orange
        'mMTC': "#00CED1",       # Dark Cyan
        'eMBB': "#BA55D3",       # Medium Orchid
        'secondary': "#3498DB",  # Blue (our channel)
        'weak': "#F1C40F",       # Yellow
    }
    
    def __init__(self, n_channels: int = 20):
        super().__init__()
        self.n_channels = n_channels
        self.occupancy = np.zeros(n_channels)
        self.service_classes = ['FREE'] * n_channels
        self.modulations = ["--"] * n_channels
        self.current_channel = -1
        
        self.setMinimumHeight(40)
        self.setMaximumHeight(50)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 2, 3, 2)
        layout.setSpacing(1)
        
        # Title
        title = QLabel("📶 6G SPECTRUM STATE")
        title.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 9px;")
        layout.addWidget(title)
        
        # Heatmap row (using labels)
        self.cell_layout = QHBoxLayout()
        self.cell_layout.setSpacing(1)
        
        self.cells = []
        for i in range(n_channels):
            cell = QLabel(f"{i}")
            cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cell.setMinimumSize(18, 20)
            cell.setStyleSheet(f"""
                background-color: {self.COLORS['FREE']};
                color: black;
                font-size: 7px;
                font-weight: bold;
                border-radius: 2px;
            """)
            self.cells.append(cell)
            self.cell_layout.addWidget(cell)
        
        layout.addLayout(self.cell_layout)
    
    @pyqtSlot(np.ndarray)
    def update_occupancy(self, occupancy: np.ndarray, service_classes: list = None):
        """Update cell colors based on 6G service classes."""
        self.occupancy = occupancy
        if service_classes is not None:
            self.service_classes = service_classes
        
        for i, (cell, occ) in enumerate(zip(self.cells, occupancy)):
            sc = self.service_classes[i] if i < len(self.service_classes) else 'FREE'
            
            # Determine color based on service class or current channel
            if i == self.current_channel:
                color = self.COLORS['secondary']
                text_color = "white"
            elif occ < 0.2 or sc == 'FREE':
                color = self.COLORS['FREE']
                text_color = "black"
            elif sc in self.COLORS:
                color = self.COLORS[sc]
                text_color = "white" if sc in ['PU', 'eMBB'] else "black"
            else:
                # Fallback to occupancy-based
                if occ < 0.6:
                    color = self.COLORS['weak']
                    text_color = "black"
                else:
                    color = self.COLORS['PU']
                    text_color = "white"
            
            cell.setStyleSheet(f"""
                background-color: {color};
                color: {text_color};
                font-size: 7px;
                font-weight: bold;
                border-radius: 2px;
            """)
    
    def set_current_channel(self, channel: int):
        """Highlight our current channel."""
        self.current_channel = channel
        self.update_occupancy(self.occupancy, self.service_classes)
    
    def set_modulations(self, modulations: list):
        """Update modulation labels for hover."""
        self.modulations = modulations
    
    def set_service_classes(self, service_classes: list):
        """Update service class info."""
        self.service_classes = service_classes
        self.update_occupancy(self.occupancy, service_classes)
