import os
import sys
import pytest
from PyQt6.QtWidgets import QApplication

# Make project root importable when running tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# Also add src/ui to sys.path because some UI modules import `config` as a top-level module
UI_ROOT = os.path.join(ROOT, 'src', 'ui')
if UI_ROOT not in sys.path:
    sys.path.insert(0, UI_ROOT)

# Ensure PyQt6 QApplication exists for widget tests
app = QApplication.instance() or QApplication(sys.argv)

from src.ui.widgets.charts.qos_metrics import QoSMetricsPanel


def test_qos_panel_updates():
    panel = QoSMetricsPanel()

    sample_qos = {
        'URLLC': {
            'packets_sent': 10,
            'packets_delivered': 9,
            'reliability_achieved': 0.9,
            'avg_latency_ms': 0.8,
            'latency_met': True,
        },
        'mMTC': {
            'packets_sent': 100,
            'packets_delivered': 98,
            'reliability_achieved': 0.98,
        },
        'eMBB': {
            'packets_sent': 5,
            'packets_delivered': 5,
            'avg_throughput_kbps': 15000,
            'reliability_achieved': 1.0,
        }
    }

    # Before update, value labels contain default placeholders
    assert panel.urllc_latency.value_label.text() == "--"

    panel.update_metrics(sample_qos)

    # After update, labels should be populated with formatted numbers
    assert panel.urllc_latency.value_label.text() != "--"
    assert "%" in panel.urllc_reliability.value_label.text() or panel.urllc_reliability.value_label.text() != "--"
    assert panel.mmtc_devices.value_label.text() != "--"
    assert panel.embb_throughput.value_label.text() != "--"