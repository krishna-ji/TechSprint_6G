"""
QoS Metrics Panel - 6G Network Slicing Metrics

Displays real-time QoS metrics for 6G service classes:
- URLLC: Latency, reliability (99.9999%)
- mMTC: Device count, packet success rate
- eMBB: Throughput, spectral efficiency

Reference: 3GPP TS 22.261 Service requirements for 5G/6G
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QFrame, QGridLayout, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor, QFont


class QoSMeter(QFrame):
    """A styled meter for displaying a single QoS metric."""
    
    def __init__(self, title: str, unit: str, target: float, color: str):
        super().__init__()
        self.target = target
        self.color = color
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #1E1E2E;
                border: 1px solid {color};
                border-radius: 4px;
                padding: 3px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 3, 5, 3)
        layout.setSpacing(2)
        
        # Title row
        title_layout = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {color}; font-size: 9px; font-weight: bold;")
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.value_label)
        layout.addLayout(title_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: #333;
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress)
        
        # Target label
        self.target_label = QLabel(f"Target: {target}{unit}")
        self.target_label.setStyleSheet("color: #666; font-size: 8px;")
        layout.addWidget(self.target_label)
    
    def set_value(self, value: float, unit: str = ""):
        """Update the displayed value and progress."""
        self.value_label.setText(f"{value:.2f}{unit}")
        
        # Calculate progress relative to target
        if self.target > 0:
            progress = min(100, int((value / self.target) * 100))
        else:
            progress = 0
        
        self.progress.setValue(progress)
        
        # Color based on achievement
        if progress >= 100:
            self.progress.setStyleSheet("""
                QProgressBar { background-color: #333; border: none; border-radius: 3px; }
                QProgressBar::chunk { background-color: #2ECC71; border-radius: 3px; }
            """)
        elif progress >= 80:
            self.progress.setStyleSheet("""
                QProgressBar { background-color: #333; border: none; border-radius: 3px; }
                QProgressBar::chunk { background-color: #F1C40F; border-radius: 3px; }
            """)
        else:
            self.progress.setStyleSheet(f"""
                QProgressBar {{ background-color: #333; border: none; border-radius: 3px; }}
                QProgressBar::chunk {{ background-color: {self.color}; border-radius: 3px; }}
            """)


class QoSMetricsPanel(QWidget):
    """
    Panel showing 6G QoS metrics for all service classes.
    
    Displays:
    - URLLC: Latency (<1ms), Reliability (99.9999%)
    - mMTC: Device count, Packet success rate
    - eMBB: Throughput, Spectral efficiency
    """
    
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        title = QLabel("6G QoS METRICS")
        title.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # ─────────────────────────────────────────────────────────────────────
        # URLLC Section
        # ─────────────────────────────────────────────────────────────────────
        urllc_frame = QFrame()
        urllc_frame.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 1px solid #FF6B35;
                border-radius: 4px;
            }
        """)
        urllc_layout = QVBoxLayout(urllc_frame)
        urllc_layout.setContentsMargins(5, 5, 5, 5)
        urllc_layout.setSpacing(3)
        
        urllc_header = QLabel("⚡ URLLC - Ultra-Reliable Low-Latency")
        urllc_header.setStyleSheet("color: #FF6B35; font-weight: bold; font-size: 10px;")
        urllc_layout.addWidget(urllc_header)
        
        urllc_metrics = QHBoxLayout()
        self.urllc_latency = QoSMeter("Latency", "ms", 1.0, "#FF6B35")
        self.urllc_reliability = QoSMeter("Reliability", "%", 99.9999, "#FF6B35")
        urllc_metrics.addWidget(self.urllc_latency)
        urllc_metrics.addWidget(self.urllc_reliability)
        urllc_layout.addLayout(urllc_metrics)
        
        layout.addWidget(urllc_frame)
        
        # ─────────────────────────────────────────────────────────────────────
        # mMTC Section
        # ─────────────────────────────────────────────────────────────────────
        mmtc_frame = QFrame()
        mmtc_frame.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 1px solid #9B59B6;
                border-radius: 4px;
            }
        """)
        mmtc_layout = QVBoxLayout(mmtc_frame)
        mmtc_layout.setContentsMargins(5, 5, 5, 5)
        mmtc_layout.setSpacing(3)
        
        mmtc_header = QLabel("📡 mMTC - massive Machine-Type Comm")
        mmtc_header.setStyleSheet("color: #9B59B6; font-weight: bold; font-size: 10px;")
        mmtc_layout.addWidget(mmtc_header)
        
        mmtc_metrics = QHBoxLayout()
        self.mmtc_devices = QoSMeter("Active Devices", "", 50, "#9B59B6")
        self.mmtc_success = QoSMeter("Success Rate", "%", 99.9, "#9B59B6")
        mmtc_metrics.addWidget(self.mmtc_devices)
        mmtc_metrics.addWidget(self.mmtc_success)
        mmtc_layout.addLayout(mmtc_metrics)
        
        layout.addWidget(mmtc_frame)
        
        # ─────────────────────────────────────────────────────────────────────
        # eMBB Section
        # ─────────────────────────────────────────────────────────────────────
        embb_frame = QFrame()
        embb_frame.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 1px solid #3498DB;
                border-radius: 4px;
            }
        """)
        embb_layout = QVBoxLayout(embb_frame)
        embb_layout.setContentsMargins(5, 5, 5, 5)
        embb_layout.setSpacing(3)
        
        embb_header = QLabel("📺 eMBB - enhanced Mobile Broadband")
        embb_header.setStyleSheet("color: #3498DB; font-weight: bold; font-size: 10px;")
        embb_layout.addWidget(embb_header)
        
        embb_metrics = QHBoxLayout()
        self.embb_throughput = QoSMeter("Throughput", " Mbps", 10.0, "#3498DB")
        self.embb_spectral = QoSMeter("Spectral Eff", "%", 80.0, "#3498DB")
        embb_metrics.addWidget(self.embb_throughput)
        embb_metrics.addWidget(self.embb_spectral)
        embb_layout.addLayout(embb_metrics)
        
        layout.addWidget(embb_frame)
        
        # ─────────────────────────────────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────────────────────────────────
        self.summary_label = QLabel("Waiting for simulation data...")
        self.summary_label.setStyleSheet("color: #666; font-size: 9px;")
        layout.addWidget(self.summary_label)
        
        layout.addStretch()
    
    @pyqtSlot(dict)
    def update_metrics(self, qos_data: dict):
        """
        Update QoS metrics display.
        
        Parameters
        ----------
        qos_data : dict
            QoS summary from simulator with URLLC, mMTC, eMBB metrics
        """
        # URLLC metrics
        if 'URLLC' in qos_data:
            urllc = qos_data['URLLC']
            latency = urllc.get('avg_latency_ms', 0)
            reliability = urllc.get('reliability_achieved', 0) * 100
            self.urllc_latency.set_value(latency, " ms")
            self.urllc_reliability.set_value(reliability, "%")
        
        # mMTC metrics
        if 'mMTC' in qos_data:
            mmtc = qos_data['mMTC']
            devices = mmtc.get('packets_sent', 0)
            success = mmtc.get('reliability_achieved', 0) * 100
            self.mmtc_devices.set_value(devices, "")
            self.mmtc_success.set_value(success, "%")
        
        # eMBB metrics
        if 'eMBB' in qos_data:
            embb = qos_data['eMBB']
            throughput = embb.get('avg_throughput_kbps', 0) / 1000  # Convert to Mbps
            spectral = embb.get('reliability_achieved', 0) * 100
            self.embb_throughput.set_value(throughput, " Mbps")
            self.embb_spectral.set_value(spectral, "%")
        
        # Update summary
        total_met = 0
        total_reqs = 0
        
        for sc in ['URLLC', 'mMTC', 'eMBB']:
            if sc in qos_data:
                data = qos_data[sc]
                total_reqs += 1
                if data.get('reliability_met', False):
                    total_met += 1
                if sc == 'URLLC' and data.get('latency_met', False):
                    total_met += 1
                    total_reqs += 1
                if sc == 'eMBB' and data.get('throughput_met', False):
                    total_met += 1
                    total_reqs += 1
        
        if total_reqs > 0:
            score = int((total_met / total_reqs) * 100)
            color = "#2ECC71" if score >= 80 else "#F1C40F" if score >= 50 else "#E74C3C"
            self.summary_label.setText(
                f"<span style='color:{color}'>QoS Score: {score}%</span> | "
                f"<span style='color:#888'>{total_met}/{total_reqs} requirements met</span>"
            )
