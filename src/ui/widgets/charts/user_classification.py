"""
User Classification Panel

Displays channel-by-channel classification showing:
- Modulation type (via AMC)
- User type (Primary/Secondary/Tertiary)
- Power level
- Status
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor, QFont
import numpy as np


class UserClassificationPanel(QWidget):
    """
    Table showing user classification for each channel.
    
    Columns:
    - Channel number
    - Frequency (MHz)
    - Modulation (via AMC)
    - User Type (PU/SU/TU/FREE)
    - Power (dB)
    - Status
    """
    
    USER_TYPES = {
        'PU': ('Primary User', '#E74C3C', 'Licensed broadcaster'),
        'SU': ('Secondary User', '#3498DB', 'Opportunistic access (You)'),
        'TU': ('Tertiary User', '#9B59B6', 'Low-priority access'),
        'FREE': ('Available', '#2ECC71', 'Spectrum hole'),
    }
    
    def __init__(self, n_channels: int = 20, start_freq: float = 88e6):
        super().__init__()
        self.n_channels = n_channels
        self.start_freq = start_freq
        self.channel_spacing = 1e6  # 1 MHz
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title with AMC label
        title_layout = QHBoxLayout()
        title = QLabel("USER CLASSIFICATION")
        title.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 12px;")
        
        amc_label = QLabel("(via AMC)")
        amc_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        
        title_layout.addWidget(title)
        title_layout.addWidget(amc_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.setSpacing(10)
        for code, (name, color, _) in self.USER_TYPES.items():
            lbl = QLabel(f"● {code}")
            lbl.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")
            lbl.setToolTip(name)
            legend_layout.addWidget(lbl)
        legend_layout.addStretch()
        layout.addLayout(legend_layout)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "CH", "FREQ", "MODULATION", "USER", "PWR", "STATUS"
        ])
        
        # Style table
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E2E;
                color: #E0E0E0;
                gridline-color: #333;
                border: 1px solid #333;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #3498DB;
            }
            QHeaderView::section {
                background-color: #252526;
                color: #00E5FF;
                padding: 5px;
                border: 1px solid #333;
                font-weight: bold;
                font-size: 10px;
            }
        """)
        
        # Column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        
        self.table.setColumnWidth(0, 35)
        self.table.setColumnWidth(1, 55)
        self.table.setColumnWidth(3, 45)
        self.table.setColumnWidth(4, 45)
        
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(True)
        
        layout.addWidget(self.table)
        
        # Summary stats
        self.summary_label = QLabel("PU: 0 | SU: 0 | FREE: 0")
        self.summary_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.summary_label)
        
        # Initialize with empty data
        self._init_table()
    
    def _init_table(self):
        """Initialize table with channel data."""
        self.table.setRowCount(self.n_channels)
        
        for i in range(self.n_channels):
            freq = (self.start_freq + (i + 0.5) * self.channel_spacing) / 1e6
            
            self.table.setItem(i, 0, QTableWidgetItem(f"{i:02d}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{freq:.1f}"))
            self.table.setItem(i, 2, QTableWidgetItem("--"))
            self.table.setItem(i, 3, QTableWidgetItem("--"))
            self.table.setItem(i, 4, QTableWidgetItem("--"))
            self.table.setItem(i, 5, QTableWidgetItem("Scanning..."))
            
            # Center align
            for col in range(6):
                item = self.table.item(i, col)
                if item:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def classify_user(self, modulation: str, occupancy: float, power_db: float) -> str:
        """
        Classify user type based on modulation and signal characteristics.
        
        Primary User (PU): Strong FM/AM broadcasts, licensed services
        Secondary User (SU): Our cognitive radio transmissions
        Tertiary User (TU): Weak/intermittent signals
        FREE: No significant signal
        """
        if occupancy < 0.3:
            return 'FREE'
        
        # Strong analog broadcasts = Primary Users
        if modulation in {'FM', 'AM-DSB', 'AM-SSB'} and power_db > -20:
            return 'PU'
        
        # Digital modulations with strong signal = might be PU or SU
        if modulation in {'BPSK', 'QPSK', '8PSK', '16QAM'}:
            if power_db > -15:
                return 'PU'
            elif power_db > -25:
                return 'SU'
            else:
                return 'TU'
        
        # Weak signals = Tertiary
        if occupancy < 0.6 or power_db < -25:
            return 'TU'
        
        return 'PU'
    
    @pyqtSlot(list)
    def update_channels(self, channel_data: list):
        """
        Update table with channel information.
        
        Parameters
        ----------
        channel_data : list of dict
            Each dict contains: channel, freq, modulation, occupancy, power_db
        """
        pu_count = su_count = tu_count = free_count = 0
        
        for data in channel_data:
            ch = data.get('channel', 0)
            if ch >= self.n_channels:
                continue
            
            freq = data.get('freq', 0) / 1e6
            mod = data.get('modulation', '--')
            occ = data.get('occupancy', 0)
            power = data.get('power_db', -40)
            is_ours = data.get('is_ours', False)
            
            # Classify user type
            if is_ours:
                user_type = 'SU'
            else:
                user_type = self.classify_user(mod, occ, power)
            
            # Count users
            if user_type == 'PU':
                pu_count += 1
            elif user_type == 'SU':
                su_count += 1
            elif user_type == 'TU':
                tu_count += 1
            else:
                free_count += 1
            
            # Update table
            color = self.USER_TYPES[user_type][1]
            
            # Channel
            item = self.table.item(ch, 0)
            if item:
                item.setText(f"{ch:02d}")
            
            # Frequency
            item = self.table.item(ch, 1)
            if item:
                item.setText(f"{freq:.1f}")
            
            # Modulation (via AMC)
            item = self.table.item(ch, 2)
            if item:
                item.setText(mod)
                if mod in {'FM', 'AM-DSB', 'AM-SSB'}:
                    item.setForeground(QColor('#E74C3C'))
                elif mod in {'BPSK', 'QPSK', '8PSK', '16QAM'}:
                    item.setForeground(QColor('#3498DB'))
                else:
                    item.setForeground(QColor('#888'))
            
            # User type
            item = self.table.item(ch, 3)
            if item:
                item.setText(user_type)
                item.setForeground(QColor(color))
            
            # Power
            item = self.table.item(ch, 4)
            if item:
                item.setText(f"{power:.0f}")
                if power > -15:
                    item.setForeground(QColor('#E74C3C'))
                elif power > -25:
                    item.setForeground(QColor('#F1C40F'))
                else:
                    item.setForeground(QColor('#2ECC71'))
            
            # Status
            item = self.table.item(ch, 5)
            if item:
                if is_ours:
                    item.setText("◆ ACTIVE")
                    item.setForeground(QColor('#3498DB'))
                elif user_type == 'FREE':
                    item.setText("● Available")
                    item.setForeground(QColor('#2ECC71'))
                elif user_type == 'PU':
                    item.setText("✖ Licensed")
                    item.setForeground(QColor('#E74C3C'))
                else:
                    item.setText("○ Occupied")
                    item.setForeground(QColor('#F1C40F'))
        
        # Update summary
        self.summary_label.setText(
            f"<span style='color:#E74C3C'>PU: {pu_count}</span> | "
            f"<span style='color:#3498DB'>SU: {su_count}</span> | "
            f"<span style='color:#9B59B6'>TU: {tu_count}</span> | "
            f"<span style='color:#2ECC71'>FREE: {free_count}</span>"
        )
    
    def highlight_channel(self, channel: int):
        """Highlight the currently selected channel."""
        self.table.selectRow(channel)
