"""
Cognitive Radio Dashboard - Main Window

A streamlined interface for real-time spectrum sensing and
intelligent channel allocation using RL and AMC.

Layout:
┌─────────────────────────────────────────────────────────┐
│  [Toolbar]                                               │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐    │
│  │         COGNITIVE RADIO STATUS HUD              │    │
│  │   [Modulation: QPSK]  [Channel: 7]  [SNR: 15dB] │    │
│  └─────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐ ┌────────────────────────┐    │
│  │                      │ │                        │    │
│  │   SPECTRUM WATERFALL │ │   CHANNEL OCCUPANCY    │    │
│  │   (Time-Frequency)   │ │   (RL Agent View)      │    │
│  │                      │ │                        │    │
│  └──────────────────────┘ └────────────────────────┘    │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐ ┌────────────────────────┐    │
│  │   CONSTELLATION      │ │   MODULATION PROBS     │    │
│  │   (IQ Diagram)       │ │   (AMC Output)         │    │
│  └──────────────────────┘ └────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
"""

import signal
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFrame, QLabel, QStatusBar, QSplitter, QGroupBox
)
import qtawesome as qta

from config import *
from core.system import SystemController
from interface.toolbar import create_tool_bar
from interface.menu_bar import create_menu_bar
from widgets.charts.waterfall import PlotWaterfallDiagram
from widgets.charts.constellation import PlotConstellation
from widgets.charts.probability import BarGraphWidget
from widgets.charts.spectrum import ChannelSpectrumWidget


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
            font-size: 24px;
            font-weight: bold;
            font-family: 'Consolas', 'Monaco', monospace;
        """)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
    
    def set_value(self, value: str):
        self.value_label.setText(value)


class CognitiveRadioWindow(QMainWindow):
    """
    Main window for the Cognitive Radio Dashboard.
    
    Provides real-time visualization of:
    - Spectrum waterfall (time-frequency analysis)
    - Channel occupancy (RL agent's view of spectrum)
    - Constellation diagram (IQ samples)
    - Modulation classification probabilities
    """
    
    def __init__(self, use_hardware: bool = False):
        super().__init__()
        self.use_hardware = use_hardware
        self.setWindowTitle("Cognitive Radio - Spectrum Intelligence")
        self.setGeometry(50, 50, 1400, 900)
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
        self.system = SystemController(use_hardware=self.use_hardware)
        
        # === STATUS HUD ===
        hud_layout = QHBoxLayout()
        hud_layout.setSpacing(15)
        
        self.card_modulation = StatusCard("DETECTED MODULATION", "SCANNING...", "#FF2A6D")
        self.card_channel = StatusCard("ACTIVE CHANNEL", "--", "#00E5FF")
        self.card_status = StatusCard("SYSTEM STATUS", "INITIALIZING", "#2ECC71")
        
        hud_layout.addWidget(self.card_modulation)
        hud_layout.addWidget(self.card_channel)
        hud_layout.addWidget(self.card_status)
        
        main_layout.addLayout(hud_layout)
        
        # === MAIN VISUALIZATION AREA ===
        viz_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # --- Top Row: Waterfall + Channel Spectrum ---
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Waterfall
        waterfall_group = QGroupBox("SPECTRUM WATERFALL")
        waterfall_layout = QVBoxLayout(waterfall_group)
        waterfall_layout.setContentsMargins(5, 15, 5, 5)
        self.waterfall_widget = PlotWaterfallDiagram(FFT_SIZE, 100)
        waterfall_layout.addWidget(self.waterfall_widget)
        
        # Channel Spectrum
        spectrum_group = QGroupBox("CHANNEL OCCUPANCY (RL View)")
        spectrum_layout = QVBoxLayout(spectrum_group)
        spectrum_layout.setContentsMargins(5, 15, 5, 5)
        self.spectrum_widget = ChannelSpectrumWidget(N_CHANNELS)
        spectrum_layout.addWidget(self.spectrum_widget)
        
        top_splitter.addWidget(waterfall_group)
        top_splitter.addWidget(spectrum_group)
        top_splitter.setSizes([700, 500])
        
        # --- Bottom Row: Constellation + Probability ---
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Constellation
        constellation_group = QGroupBox("CONSTELLATION (IQ)")
        constellation_layout = QVBoxLayout(constellation_group)
        constellation_layout.setContentsMargins(5, 15, 5, 5)
        self.constellation_widget = PlotConstellation(external_drive=True)
        constellation_layout.addWidget(self.constellation_widget)
        
        # Probability
        prob_group = QGroupBox("MODULATION CLASSIFICATION")
        prob_layout = QVBoxLayout(prob_group)
        prob_layout.setContentsMargins(5, 15, 5, 5)
        self.prob_widget = BarGraphWidget()
        prob_layout.addWidget(self.prob_widget)
        
        bottom_splitter.addWidget(constellation_group)
        bottom_splitter.addWidget(prob_group)
        bottom_splitter.setSizes([500, 700])
        
        # Add rows to vertical splitter
        viz_splitter.addWidget(top_splitter)
        viz_splitter.addWidget(bottom_splitter)
        viz_splitter.setSizes([500, 350])
        
        main_layout.addWidget(viz_splitter, stretch=1)
        
        # === CONNECT SIGNALS ===
        self._connect_signals()
        
        # Start system
        self.system.start_system()
        self.card_status.set_value("RUNNING")
    
    def _connect_signals(self):
        """Connect SystemController signals to UI widgets."""
        # IQ data to visualizations
        self.system.update_plots.connect(self.constellation_widget.update_from_external)
        self.system.update_plots.connect(self.waterfall_widget.update_plot)
        
        # Probabilities to bar chart
        self.system.update_probs.connect(self.prob_widget.update_probabilities)
        
        # Spectrum occupancy
        self.system.update_spectrum.connect(self.spectrum_widget.update_occupancy)
        
        # Status updates
        self.system.update_status.connect(self._update_hud)
    
    def _update_hud(self, mod_text: str, chan_text: str):
        """Update the heads-up display cards."""
        # Parse modulation text
        if ":" in mod_text:
            mod_value = mod_text.split(":")[1].strip()
        else:
            mod_value = mod_text
        self.card_modulation.set_value(mod_value)
        
        # Parse channel text
        if ":" in chan_text:
            chan_value = chan_text.split(":")[1].strip()
            try:
                chan_num = int(chan_value)
                self.spectrum_widget.set_selected_channel(chan_num)
            except ValueError:
                pass
        else:
            chan_value = chan_text
        self.card_channel.set_value(chan_value)
        
        # Status bar
        self.status_bar.showMessage(f"📡 {mod_text} | 📻 {chan_text}")
    
    def closeEvent(self, event):
        """Clean shutdown."""
        print("Shutting down Cognitive Radio...")
        self.card_status.set_value("STOPPING")
        self.system.stop_system()
        self.constellation_widget.close()
        event.accept()


# Keep old MainWindow as alias for compatibility
MainWindow = CognitiveRadioWindow


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
    
    # High DPI scaling
    app.setStyle('Fusion')
    
    # Determine hardware mode
    use_hardware = args.use_hardware and not args.simulate
    
    if use_hardware:
        print("🔌 Starting with HARDWARE mode (RTL-SDR)")
    else:
        print("🖥️  Starting with SIMULATION mode")
    
    window = CognitiveRadioWindow(use_hardware=use_hardware)
    window.show()
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nSignal received, closing...")
        window.close()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
