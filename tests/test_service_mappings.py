import os
import sys
import pytest
# Make project root importable when running tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.radio.simulation import SimulationTrafficGenerator, ServiceClass


def test_class_id_mappings():
    sim = SimulationTrafficGenerator()

    # Ensure dataset mapping aligns with 6G service classes
    assert sim.class_id_to_service[1] == ServiceClass.URLLC
    assert sim.class_id_to_modulation[1] == "QPSK"
    assert sim.class_id_to_modulation[3] in {"64QAM", "QPSK"}


def test_qos_summary_structure():
    sim = SimulationTrafficGenerator()

    # Set some QoS counters and verify summary contains expected keys
    sim.qos_metrics[ServiceClass.URLLC]['packets_sent'] = 10
    sim.qos_metrics[ServiceClass.URLLC]['packets_delivered'] = 9
    sim.qos_metrics[ServiceClass.eMBB]['packets_sent'] = 5
    sim.qos_metrics[ServiceClass.eMBB]['packets_delivered'] = 5

    summary = sim.get_qos_summary()
    assert set(summary.keys()) == {"URLLC", "mMTC", "eMBB"}
    assert summary['URLLC']['packets_sent'] == 10
    assert 0 <= summary['URLLC']['reliability_achieved'] <= 1
    assert 'avg_latency_ms' in summary['URLLC']
    assert 'avg_throughput_kbps' in summary['eMBB']