from logic.radio_actions import play_radio, stop_radio
from logic.menu_actions import capture_iq, simulate, close_app
from logic.toolbar_actions import simulate_transmit, show_spectrum_info
import qtawesome as qta
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QToolBar


def create_tool_bar(parent):
    tool_bar = QToolBar(parent)

    # Create actions with icons
    capture_action = QAction(qta.icon('fa5s.camera'), "Capture IQ", parent)
    capture_action.triggered.connect(lambda: capture_iq(parent))
    parent.capture_button = capture_action  # So we can toggle icon later

    load_action = QAction(qta.icon('fa5s.folder-open'), "Load IQ", parent)
    load_action.triggered.connect(lambda: simulate(parent))

    play_radio_action = QAction(qta.icon('mdi.radio'), "Play Radio", parent)
    play_radio_action.triggered.connect(lambda: play_radio(parent))

    stop_radio_action = QAction(qta.icon('fa5s.stop'), "Stop Radio", parent)
    stop_radio_action.triggered.connect(lambda: stop_radio(parent))

    # NEW: Cognitive Radio Actions
    transmit_action = QAction(qta.icon('fa5s.broadcast-tower'), "Simulate TX", parent)
    transmit_action.triggered.connect(lambda: simulate_transmit(parent))
    transmit_action.setToolTip("Simulate transmission on best available channel")
    
    spectrum_info_action = QAction(qta.icon('fa5s.chart-bar'), "Spectrum Info", parent)
    spectrum_info_action.triggered.connect(lambda: show_spectrum_info(parent))
    spectrum_info_action.setToolTip("Show detailed spectrum analysis")

    exit_action = QAction(qta.icon('mdi.location-exit'), "Exit AMC", parent)
    exit_action.triggered.connect(lambda: close_app(parent))

    # Add actions to the toolbar
    tool_bar.addAction(capture_action)
    tool_bar.addAction(load_action)
    tool_bar.addAction(play_radio_action)
    tool_bar.addAction(stop_radio_action)
    tool_bar.addSeparator()
    tool_bar.addAction(transmit_action)
    tool_bar.addAction(spectrum_info_action)
    tool_bar.addSeparator()
    tool_bar.addAction(exit_action)

    return tool_bar
