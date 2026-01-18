"""
Cognitive Radio Dashboard - Revamped UI

A modern interface focused on spectrum intelligence visualization:
- Spectrum Occupancy Matrix (time-frequency grid)
- User Classification (PU/SU/TU identification via AMC)
- RL Decision Visualization
- Allocation Intelligence Metrics

No constellation plots - focus on actionable spectrum insights.
"""

import signal
import sys
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFrame, QLabel, QStatusBar, QSplitter, QGroupBox,
    QTabWidget
)
import qtawesome as qta
import numpy as np

from config import *
from core.system import SystemController
from interface.toolbar import create_tool_bar
from interface.menu_bar import create_menu_bar

# New widgets
from widgets.charts.spectrum_matrix import SpectrumMatrixWidget, SpectrumHeatmapWidget
from widgets.charts.user_classification import UserClassificationPanel
from widgets.charts.rl_visualizer import RLDecisionPanel, AllocationIntelligencePanel
from widgets.charts.probability import BarGraphWidget
from widgets.charts.waterfall import PlotWaterfallDiagram


class StatusCard(QFrame):
    """A styled card for displaying a single status value."""
    
    def __init__(self, title: str, initial_value: str, accent_color: str):
        super().__init__()
        self.setObjectName("statusCard")
        self.setStyleSheet(f"""
            QFrame#statusCard {{
                background-color: #252526;
                border: 2px solid {accent_color};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("""
            color: #888888;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
        """)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Value
        self.value_label = QLabel(initial_value)
        self.value_label.setStyleSheet(f"""
            color: {accent_color};
            font-size: 20px;
            font-weight: bold;
            font-family: 'Consolas', 'Monaco', monospace;
        """)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Subtitle
        self.subtitle_label = QLabel("")
        self.subtitle_label.setStyleSheet("color: #666; font-size: 9px;")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.subtitle_label)
    
    def set_value(self, value: str, subtitle: str = ""):
        self.value_label.setText(value)
        self.subtitle_label.setText(subtitle)


class CognitiveRadioDashboard(QMainWindow):
    """
    Revamped Cognitive Radio Dashboard.
    
    Focus on spectrum intelligence visualization:
    - Real-time spectrum occupancy matrix
    - User classification with AMC
    - RL decision visualization
    - Allocation intelligence metrics
    """
    
    def __init__(self, use_hardware: bool = False):
        super().__init__()
        self.use_hardware = use_hardware
        self.setWindowTitle("Cognitive Radio - Spectrum Intelligence Dashboard")
        self.setGeometry(50, 50, 1500, 950)
        self.setWindowIcon(QIcon(qta.icon('mdi.radio-tower').pixmap(32, 32)))
        
        # Dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E2E;
            }
            QWidget {
                color: #E0E0E0;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            QGroupBox {
                background-color: #252526;
                border: 1px solid #333333;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #00E5FF;
            }
            QSplitter::handle {
                background-color: #333333;
            }
            QTabWidget::pane {
                border: 1px solid #333;
                background-color: #252526;
            }
            QTabBar::tab {
                background-color: #1E1E2E;
                color: #888;
                padding: 8px 15px;
                border: 1px solid #333;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #252526;
                color: #00E5FF;
                border-bottom: 2px solid #00E5FF;
            }
        """)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background-color: #1A1A1A; color: #888888;")
        self.setStatusBar(self.status_bar)
        
        # Menu and toolbar
        self.setMenuBar(create_menu_bar(self))
        toolbar = create_tool_bar(self)
        toolbar.setOrientation(Qt.Orientation.Horizontal)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # === SYSTEM CONTROLLER ===
        self.system_controller = SystemController(use_hardware=self.use_hardware)
        
        # === STATUS HUD ===
        hud_layout = QHBoxLayout()
        hud_layout.setSpacing(15)
        
        self.card_modulation = StatusCard("MODULATION (via AMC)", "SCANNING...", "#FF2A6D")
        self.card_channel = StatusCard("ACTIVE CHANNEL", "--", "#00E5FF")
        self.card_user_type = StatusCard("USER TYPE", "SECONDARY", "#3498DB")
        self.card_status = StatusCard("SYSTEM STATUS", "INITIALIZING", "#2ECC71")
        
        hud_layout.addWidget(self.card_modulation)
        hud_layout.addWidget(self.card_channel)
        hud_layout.addWidget(self.card_user_type)
        hud_layout.addWidget(self.card_status)
        
        main_layout.addLayout(hud_layout)
        
        # === MAIN CONTENT AREA ===
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT SIDE: Spectrum Visualization ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        
        # Current spectrum state (heatmap)
        self.spectrum_heatmap = SpectrumHeatmapWidget(N_CHANNELS)
        left_layout.addWidget(self.spectrum_heatmap)
        
        # Spectrum matrix (time-frequency)
        matrix_group = QGroupBox("SPECTRUM OCCUPANCY MATRIX (Time-Frequency)")
        matrix_layout = QVBoxLayout(matrix_group)
        matrix_layout.setContentsMargins(5, 15, 5, 5)
        self.spectrum_matrix = SpectrumMatrixWidget(N_CHANNELS, history_depth=15)
        matrix_layout.addWidget(self.spectrum_matrix)
        left_layout.addWidget(matrix_group, stretch=2)
        
        # Waterfall (optional, in tab)
        waterfall_group = QGroupBox("SPECTRUM WATERFALL")
        waterfall_layout = QVBoxLayout(waterfall_group)
        waterfall_layout.setContentsMargins(5, 15, 5, 5)
        self.waterfall_widget = PlotWaterfallDiagram(FFT_SIZE, 80)
        waterfall_layout.addWidget(self.waterfall_widget)
        left_layout.addWidget(waterfall_group, stretch=1)
        
        content_splitter.addWidget(left_widget)
        
        # --- CENTER: User Classification ---
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        classification_group = QGroupBox("CHANNEL ANALYSIS")
        classification_layout = QVBoxLayout(classification_group)
        classification_layout.setContentsMargins(5, 15, 5, 5)
        self.user_classification = UserClassificationPanel(N_CHANNELS, SWEEP_START_FREQ)
        classification_layout.addWidget(self.user_classification)
        center_layout.addWidget(classification_group)
        
        content_splitter.addWidget(center_widget)
        
        # --- RIGHT SIDE: RL & Intelligence ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        
        # RL Decision Panel
        rl_group = QGroupBox("RL AGENT DECISION")
        rl_layout = QVBoxLayout(rl_group)
        rl_layout.setContentsMargins(5, 15, 5, 5)
        self.rl_panel = RLDecisionPanel(N_CHANNELS)
        rl_layout.addWidget(self.rl_panel)
        right_layout.addWidget(rl_group)
        
        # Allocation Intelligence
        intel_group = QGroupBox("SPECTRUM INTELLIGENCE")
        intel_layout = QVBoxLayout(intel_group)
        intel_layout.setContentsMargins(5, 15, 5, 5)
        self.intel_panel = AllocationIntelligencePanel(N_CHANNELS)
        intel_layout.addWidget(self.intel_panel)
        right_layout.addWidget(intel_group)
        
        # Modulation Probabilities
        probs_group = QGroupBox("MODULATION CLASSIFICATION (via AMC)")
        probs_layout = QVBoxLayout(probs_group)
        probs_layout.setContentsMargins(5, 15, 5, 5)
        self.probability_widget = BarGraphWidget()
        probs_layout.addWidget(self.probability_widget)
        right_layout.addWidget(probs_group)
        
        content_splitter.addWidget(right_widget)
        
        # Set splitter proportions
        content_splitter.setSizes([500, 400, 400])
        
        main_layout.addWidget(content_splitter, stretch=1)
        
        # === CONNECT SIGNALS ===
        self._connect_signals()
        
        # Start system
        self.system_controller.start_system()
        self.card_status.set_value("RUNNING", "Spectrum sweeping active")
    
    def _connect_signals(self):
        """Connect system controller signals to UI widgets."""
        # Basic updates
        self.system_controller.update_plots.connect(self._update_waterfall)
        self.system_controller.update_probs.connect(self._update_probabilities)
        self.system_controller.update_status.connect(self._update_hud)
        self.system_controller.update_spectrum.connect(self._update_spectrum)
        self.system_controller.update_sweep_info.connect(self._update_sweep_info)
    
    def _update_waterfall(self, iq_data):
        """Update waterfall display."""
        self.waterfall_widget.update_plot(iq_data)
    
    def _update_probabilities(self, probs):
        """Update probability bar chart."""
        self.probability_widget.update_probabilities(probs)
    
    def _update_hud(self, mod_text: str, chan_text: str):
        """Update the heads-up display cards."""
        # Parse modulation
        if ":" in mod_text:
            mod_value = mod_text.split(":")[1].strip()
        else:
            mod_value = mod_text
        self.card_modulation.set_value(mod_value, "(via AMC)")
        
        # Parse channel
        if "Ch" in chan_text or "CH" in chan_text:
            import re
            match = re.search(r'(\d+)', chan_text)
            if match:
                ch_num = int(match.group(1))
                freq = (SWEEP_START_FREQ + (ch_num + 0.5) * 1e6) / 1e6
                self.card_channel.set_value(f"CH {ch_num:02d}", f"@ {freq:.1f} MHz")
                self.spectrum_heatmap.set_current_channel(ch_num)
                self.spectrum_matrix.set_current_channel(ch_num)
        
        # Update user type based on channel state
        self.card_user_type.set_value("SECONDARY", "Opportunistic Access")
        
        # Status bar
        mode = "HW" if self.use_hardware else "SIM"
        self.status_bar.showMessage(f"📡 [{mode}] {mod_text} | 📻 {chan_text}")
    
    def _update_spectrum(self, occupancy: np.ndarray):
        """Update spectrum visualizations."""
        self.spectrum_heatmap.update_occupancy(occupancy)
        self.spectrum_matrix.update_occupancy(occupancy)
    
    def _update_sweep_info(self, sweep_info: dict):
        """Update all panels with sweep information."""
        # Extract data from sweep_info
        channel_states = sweep_info.get('channel_states', [0.0] * N_CHANNELS)
        if isinstance(channel_states, list):
            occupancy = np.array(channel_states)
        else:
            occupancy = channel_states
            
        channel = sweep_info.get('recommended_channel', 0)
        action = sweep_info.get('action', 'STAY')
        prev_channel = sweep_info.get('prev_channel', channel)
        channel_freq = sweep_info.get('channel_freq_mhz', 0)
        modulation = sweep_info.get('modulation', '--')
        spectrum_holes = sweep_info.get('spectrum_holes', [])
        action_probs = sweep_info.get('action_probs', None)
        channel_frequencies = sweep_info.get('channel_frequencies', [])
        is_free = sweep_info.get('is_free', True)
        
        # Calculate confidence from action probs
        if action_probs is not None:
            confidence = max(action_probs) if len(action_probs) > 0 else 0.5
        else:
            confidence = 0.8 if is_free else 0.4
        
        # Build channel data for classification panel
        channel_data = []
        for i in range(N_CHANNELS):
            if len(channel_frequencies) > i:
                freq = channel_frequencies[i] * 1e6  # Convert MHz to Hz
            else:
                freq = SWEEP_START_FREQ + (i + 0.5) * 1e6
            
            # Determine power from occupancy (simulated)
            occ = occupancy[i] if i < len(occupancy) else 0.0
            power_db = -20 if occ > 0.6 else (-35 if occ > 0.3 else -50)
            
            channel_data.append({
                'channel': i,
                'freq': freq,
                'modulation': modulation if i == channel else '--',
                'occupancy': occ,
                'power_db': power_db,
                'is_ours': (i == channel)
            })
        
        self.user_classification.update_channels(channel_data)
        self.user_classification.highlight_channel(channel)
        
        # Update RL decision panel
        # Determine reason
        if action == 'SWITCH':
            reason = f"Found better channel (lower occupancy)"
        elif occupancy[channel] < 0.3:
            reason = "Current channel is optimal (free)"
        else:
            reason = "Evaluating alternatives..."
        
        rl_decision = {
            'channel': channel,
            'freq': channel_freq * 1e6,  # Convert to Hz
            'action': action,
            'confidence': confidence,
            'occupancy': occupancy[channel] if channel < len(occupancy) else 0.0,
            'reason': reason,
            'action_probs': action_probs if action_probs is not None else [0.0] * N_CHANNELS,
            'prev_channel': prev_channel
        }
        self.rl_panel.update_decision(rl_decision)
        
        # Update intelligence panel
        free_count = np.sum(occupancy < 0.3)
        occupied_count = np.sum(occupancy >= 0.6)
        
        intel_data = {
            'occupancy_array': occupancy,
            'spectrum_holes': spectrum_holes,
            'free_count': int(free_count),
            'occupied_count': int(occupied_count),
        }
        self.intel_panel.update_intelligence(intel_data)
    
    def closeEvent(self, event):
        """Clean shutdown."""
        print("Shutting down Cognitive Radio Dashboard...")
        self.card_status.set_value("STOPPING", "")
        self.system_controller.stop_system()
        event.accept()


# Alias for compatibility
CognitiveRadioWindow = CognitiveRadioDashboard
MainWindow = CognitiveRadioDashboard


def main():
    """Launch the Cognitive Radio application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cognitive Radio Dashboard")
    parser.add_argument("--use-hardware", action="store_true", 
                       help="Use real RTL-SDR hardware instead of simulation")
    parser.add_argument("--simulate", action="store_true",
                       help="Force simulation mode (default)")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    use_hardware = args.use_hardware and not args.simulate
    
    if use_hardware:
        print("🔌 Starting with HARDWARE mode (RTL-SDR)")
    else:
        print("🖥️  Starting with SIMULATION mode")
    
    window = CognitiveRadioDashboard(use_hardware=use_hardware)
    window.show()
    
    def signal_handler(sig, frame):
        print("\nSignal received, closing...")
        window.close()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
