"""
User Classification Panel - 6G Network Slicing

Displays channel-by-channel classification showing:
- Modulation type (via AMC)
- 6G Service Class (URLLC/mMTC/eMBB/PU)
- Power level
- QoS Status

6G Service Classes (3GPP):
  - URLLC: Ultra-Reliable Low-Latency Communications (<1ms, 99.9999%)
  - mMTC: massive Machine-Type Communications (1M devices/km²)
  - eMBB: enhanced Mobile Broadband (20+ Gbps)
  - PU: Primary User (Licensed incumbent)
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
    Table showing 6G service classification for each channel.
    
    Columns:
    - Channel number
    - Frequency (MHz)
    - Modulation (via AMC)
    - Service Class (6G network slice type)
    - Power (dB)
    - Status
    """
    
    # 6G Service Class definitions with colors and tooltips
    USER_TYPES = {
        'PU': ('Primary User', '#E74C3C', 'Licensed incumbent (FM broadcast)'),
        'URLLC': ('URLLC', '#FF6B35', 'Ultra-Reliable Low-Latency (<1ms)'),
        'mMTC': ('mMTC', '#9B59B6', 'massive Machine-Type Comm'),
        'eMBB': ('eMBB', '#3498DB', 'enhanced Mobile Broadband'),
        'SU': ('Secondary', '#17A2B8', 'Secondary User'),
        'TU': ('Tertiary', '#6C757D', 'Tertiary/Low-priority'),
        'FREE': ('Available', '#2ECC71', 'Spectrum hole - available'),
    }
    
    def __init__(self, n_channels: int = 20, start_freq: float = 88e6):
        super().__init__()
        self.n_channels = n_channels
        self.start_freq = start_freq
        self.channel_spacing = 1e6  # 1 MHz
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title with 6G label
        title_layout = QHBoxLayout()
        title = QLabel("6G SERVICE CLASSIFICATION")
        title.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 12px;")
        
        amc_label = QLabel("(via AMC + Network Slicing)")
        amc_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        
        title_layout.addWidget(title)
        title_layout.addWidget(amc_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # Legend - 6G service classes
        legend_layout = QHBoxLayout()
        legend_layout.setSpacing(8)
        # Show key service classes
        key_types = ['PU', 'URLLC', 'mMTC', 'eMBB', 'FREE']
        for code in key_types:
            if code in self.USER_TYPES:
                name, color, tooltip = self.USER_TYPES[code]
                lbl = QLabel(f"● {code}")
                lbl.setStyleSheet(f"color: {color}; font-size: 9px; font-weight: bold;")
                lbl.setToolTip(f"{name}: {tooltip}")
                legend_layout.addWidget(lbl)
        legend_layout.addStretch()
        layout.addLayout(legend_layout)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "CH", "FREQ", "MODULATION", "SERVICE", "PWR", "STATUS"
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
        self.summary_label = QLabel("PU: 0 | URLLC: 0 | mMTC: 0 | eMBB: 0 | FREE: 0")
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
        Classify 6G service type based on modulation and signal characteristics.
        
        Service Classes:
        - PU: Primary User - Strong FM/AM broadcasts, licensed services
        - URLLC: Ultra-Reliable Low-Latency - QPSK with strong signal
        - mMTC: massive MTC - BPSK/OOK with weak signal (battery-constrained)
        - eMBB: enhanced Mobile Broadband - High-order modulation (64QAM, 16QAM)
        - FREE: No significant signal (spectrum hole)
        """
        if occupancy < 0.3:
            return 'FREE'
        
        # Strong analog broadcasts = Primary Users (licensed FM)
        if modulation in {'FM', 'AM-DSB', 'AM-SSB'} and power_db > -20:
            return 'PU'
        
        # High-order digital modulation = eMBB (high throughput)
        if modulation in {'64QAM', '16QAM', '256QAM'}:
            if power_db > -25:
                return 'eMBB'
            else:
                return 'mMTC'  # Weak high-order = likely compressed mMTC
        
        # QPSK with strong signal = URLLC (robust, low latency)
        if modulation in {'QPSK', '8PSK'}:
            if power_db > -20:
                return 'URLLC'
            elif power_db > -30:
                return 'eMBB'
            else:
                return 'mMTC'
        
        # BPSK/OOK = mMTC (simple, low power IoT)
        if modulation in {'BPSK', 'OOK', '4ASK'}:
            return 'mMTC'
        
        # Weak signals = mMTC (battery-constrained IoT)
        if occupancy < 0.6 or power_db < -30:
            return 'mMTC'
        
        return 'PU'
    
    @pyqtSlot(list)
    def update_channels(self, channel_data: list):
        """
        Update table with channel information.
        
        Parameters
        ----------
        channel_data : list of dict
            Each dict contains: channel, freq, modulation, occupancy, power_db
            Optionally: user_type (from simulation), is_primary, service_class
        """
        # Count by 6G service class
        counts = {'PU': 0, 'URLLC': 0, 'mMTC': 0, 'eMBB': 0, 'SU': 0, 'TU': 0, 'FREE': 0}
        
        for data in channel_data:
            ch = data.get('channel', 0)
            if ch >= self.n_channels:
                continue
            
            freq = data.get('freq', 0) / 1e6
            mod = data.get('modulation', '--')
            occ = data.get('occupancy', 0)
            power = data.get('power_db', -40)
            is_ours = data.get('is_ours', False)
            is_primary = data.get('is_primary', False)
            sim_user_type = data.get('user_type', None)
            service_class = data.get('service_class', None)
            
            # Determine 6G service class
            if is_ours:
                user_type = 'SU'  # Our cognitive radio = Secondary User
            elif service_class:
                # Direct service class from simulation (URLLC, mMTC, eMBB)
                user_type = service_class
            elif sim_user_type:
                # Map simulation user types to 6G service classes
                user_type_map = {
                    'FREE': 'FREE',
                    'PU': 'PU',
                    'URLLC': 'URLLC',
                    'mMTC': 'mMTC',
                    'eMBB': 'eMBB',
                    # Legacy mappings for backward compatibility
                    'SU_CRIT': 'URLLC',    # Critical IoT = URLLC
                    'SU_DELAY': 'mMTC',    # Delay tolerant = mMTC
                    'SU_HIGH': 'eMBB',     # High throughput = eMBB
                    'TU': 'mMTC',
                    'SU': 'SU',
                }
                user_type = user_type_map.get(sim_user_type, 
                                             self.classify_user(mod, occ, power))
                # Override: primary user channels from simulation
                if is_primary:
                    user_type = 'PU'
            else:
                user_type = self.classify_user(mod, occ, power)
            
            # Ensure valid user type
            if user_type not in self.USER_TYPES:
                user_type = 'FREE'
            
            # Count by service class
            counts[user_type] = counts.get(user_type, 0) + 1
            
            # Get color for this service class
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
                # Color by modulation type
                if mod in {'FM', 'AM-DSB', 'AM-SSB'}:
                    item.setForeground(QColor('#E74C3C'))  # Red for analog
                elif mod in {'64QAM', '256QAM'}:
                    item.setForeground(QColor('#3498DB'))  # Blue for high-order
                elif mod in {'QPSK', '8PSK', '16QAM'}:
                    item.setForeground(QColor('#FF6B35'))  # Orange for URLLC-type
                elif mod in {'BPSK', 'OOK', '4ASK'}:
                    item.setForeground(QColor('#9B59B6'))  # Purple for mMTC
                else:
                    item.setForeground(QColor('#888'))
            
            # Service class (6G)
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
                elif user_type == 'URLLC':
                    item.setText("⚡ Critical")
                    item.setForeground(QColor('#FF6B35'))
                elif user_type == 'mMTC':
                    item.setText("📡 IoT")
                    item.setForeground(QColor('#9B59B6'))
                elif user_type == 'eMBB':
                    item.setText("📺 Stream")
                    item.setForeground(QColor('#3498DB'))
                else:
                    item.setText("○ Occupied")
                    item.setForeground(QColor('#F1C40F'))
        
        # Update summary with 6G service class counts
        self.summary_label.setText(
            f"<span style='color:#E74C3C'>PU:{counts['PU']}</span> │ "
            f"<span style='color:#FF6B35'>URLLC:{counts['URLLC']}</span> │ "
            f"<span style='color:#9B59B6'>mMTC:{counts['mMTC']}</span> │ "
            f"<span style='color:#3498DB'>eMBB:{counts['eMBB']}</span> │ "
            f"<span style='color:#2ECC71'>FREE:{counts['FREE']}</span>"
        )
    
    def highlight_channel(self, channel: int):
        """Highlight the currently selected channel."""
        self.table.selectRow(channel)
