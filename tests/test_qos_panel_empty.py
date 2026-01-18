import os
import sys
import pytest
from PyQt6.QtWidgets import QApplication

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
UI_ROOT = os.path.join(ROOT, 'src', 'ui')
if UI_ROOT not in sys.path:
    sys.path.insert(0, UI_ROOT)

# Ensure a QApplication exists
app = QApplication.instance() or QApplication(sys.argv)

from src.ui.widgets.charts.qos_metrics import QoSMetricsPanel


def test_qos_panel_empty_shows_no_data():
    panel = QoSMetricsPanel()
    panel.update_metrics({})
    assert 'No QoS data' in panel.summary_label.text() or 'No QoS requirements' in panel.summary_label.text()
    assert panel.urllc_latency.value_label.text() != "--"
    assert panel.mmtc_devices.value_label.text() != "--"
    assert panel.embb_throughput.value_label.text() != "--"
