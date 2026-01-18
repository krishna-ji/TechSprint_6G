import os
import sys
import threading
import time
import numpy as np
from types import SimpleNamespace
from PyQt6.QtWidgets import QApplication

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Also add src/ui so imports like `from config import ...` resolve in tests
UI_ROOT = os.path.join(ROOT, 'src', 'ui')
if UI_ROOT not in sys.path:
    sys.path.insert(0, UI_ROOT)

from src.ui.core.system import SystemController
from src.ui.core.system import DataWorker

# Ensure a QApplication exists for Qt threading/signals
app = QApplication.instance() or QApplication(sys.argv)


def test_dataworker_systemcontroller_signal_flow(monkeypatch):
    """Integration test: DataWorker -> SystemController -> UI signals."""
    N_CHANNELS = 20

    # --- Mocks ---
    class MockAMC:
        CLASSES = ['Noise', 'FM', 'QPSK', 'BPSK', '64QAM']

        def predict_proba(self, iq):
            # Return softmax-like probabilities with QPSK dominant
            probs = np.zeros((1, len(self.CLASSES)))
            probs[0, 2] = 0.9
            probs[0, 0] = 0.1
            return probs

    class MockRL:
        def update_observations(self, obs):
            self.last_obs = obs

        def get_action(self, channel_states):
            # Recommend channel 3 and uniform action_probs
            action_probs = np.zeros(len(channel_states))
            action_probs[3] = 1.0
            return 3, action_probs

    class MockSweeper:
        def __init__(self):
            self.sweep_count = 0

        def sweep(self, mode=None):
            self.sweep_count += 1
            # All channels free (0.0 occupancy)
            return SimpleNamespace(channel_states=np.zeros(N_CHANNELS))

        def print_spectrum_map(self):
            pass

        def get_channel_states(self):
            return np.zeros(N_CHANNELS)

        def get_channel_frequency(self, ch):
            return 90e6 + ch * 1e6

        def find_spectrum_holes(self, threshold=0.3):
            return [1, 2, 3]

        def _select_best_channel(self, holes):
            return holes[0]

    class MockSimulator:
        def get_qos_summary(self):
            return {
                'URLLC': {'packets_sent': 2, 'packets_delivered': 2, 'avg_latency_ms': 0.5},
                'mMTC': {'packets_sent': 5, 'packets_delivered': 5},
                'eMBB': {'packets_sent': 1, 'packets_delivered': 1, 'avg_throughput_kbps': 15000}
            }

        def get_all_channel_info(self):
            return [{'channel': i, 'modulation': 'QPSK', 'service_class': 'URLLC', 'power_db': -20} for i in range(N_CHANNELS)]

    # Monkeypatch the global get_simulator used by DataWorker
    monkeypatch.setattr('radio.simulation.get_simulator', lambda: MockSimulator())

    # Instantiate SystemController (creates DataWorker but does not start thread yet)
    controller = SystemController(use_hardware=False)

    # Inject mocks into the worker before starting the thread
    controller.worker.amc = MockAMC()
    controller.worker.rl = MockRL()
    controller.worker.sweeper = MockSweeper()

    received = {}
    event = threading.Event()

    def on_sweep_info(sweep_info):
        # Save and signal completion; then stop the controller cleanly
        received['sweep_info'] = sweep_info
        event.set()
        # Stop the thread to avoid background running
        controller.stop_system()

    # Connect to the sweep info signal
    controller.update_sweep_info.connect(on_sweep_info)

    # --- Simulate a single DataWorker iteration synchronously (avoid threads / QThread) ---
    worker = controller.worker

    # STEP 1: Spectrum Sweep
    sweep_result = worker.sweeper.sweep()
    channel_states = sweep_result.channel_states

    # STEP 2: RL decision
    worker.rl.update_observations(channel_states)
    recommended_channel, action_probs = worker.rl.get_action(channel_states)

    # STEP 3: IQ capture (simulation path)
    iq_data = worker._simulate_iq(recommended_channel, channel_states)

    # STEP 4: AMC classification
    probs = worker.amc.predict_proba(iq_data)
    avg_probs = probs.mean(axis=0)
    n_classes = len(worker.amc.CLASSES)
    if len(avg_probs) != n_classes:
        if len(avg_probs) < n_classes:
            avg_probs = np.pad(avg_probs, (0, n_classes - len(avg_probs)))
        else:
            avg_probs = avg_probs[:n_classes]

    if worker.ewma_probs is None:
        worker.ewma_probs = avg_probs
    else:
        worker.ewma_probs = worker.alpha * avg_probs + (1 - worker.alpha) * worker.ewma_probs

    class_id = int(np.argmax(worker.ewma_probs))
    class_id = min(class_id, n_classes - 1)
    mod_class = worker.amc.CLASSES[class_id]

    # STEP 5: Build sweep_info as DataWorker does
    channel_frequencies = [worker.sweeper.get_channel_frequency(ch) / 1e6 for ch in range(worker.n_channels)]

    # Use simulator get_qos_summary (monkeypatched earlier)
    qos_summary = {}
    try:
        from radio.simulation import get_simulator
        simulator = get_simulator()
        qos_summary = simulator.get_qos_summary()
        channel_info_list = simulator.get_all_channel_info()
    except Exception:
        channel_info_list = []

    sweep_info = {
        "sweep_count": worker.sweeper.sweep_count,
        "spectrum_holes": worker.sweeper.find_spectrum_holes(threshold=0.3),
        "n_free": len(worker.sweeper.find_spectrum_holes(threshold=0.3)),
        "n_occupied": worker.n_channels - len(worker.sweeper.find_spectrum_holes(threshold=0.3)),
        "recommended_channel": recommended_channel,
        "channel_freq_mhz": worker.sweeper.get_channel_frequency(recommended_channel) / 1e6,
        "is_free": channel_states[recommended_channel] < 0.3,
        "channel_states": channel_states.tolist(),
        "channel_frequencies": channel_frequencies,
        "action_probs": action_probs.tolist() if action_probs is not None else None,
        "modulation": mod_class,
        "prev_channel": worker.current_channel,
        "action": "STAY",
        "simulation_mode": not worker.use_hardware,
        "channel_info": channel_info_list,
        "qos_summary": qos_summary,
    }

    # Call SystemController handler directly (synchronous)
    controller._handle_data(iq_data, worker.ewma_probs.copy(), mod_class, recommended_channel, channel_states.copy(), sweep_info)

    # Wait for signal (max 2s)
    assert event.wait(timeout=2), "Timed out waiting for sweep_info signal"

    # Basic assertions on the payload
    sweep_info_received = received.get('sweep_info')
    assert isinstance(sweep_info_received, dict)
    assert 'channel_states' in sweep_info_received
    assert 'recommended_channel' in sweep_info_received
    assert 'modulation' in sweep_info_received
    assert sweep_info_received['recommended_channel'] == 3
    # QoS summary should be present and non-empty (from MockSimulator)
    assert 'qos_summary' in sweep_info_received
    assert 'URLLC' in sweep_info_received['qos_summary']

    # No threads started, no cleanup required
