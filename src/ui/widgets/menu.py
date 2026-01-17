from config import VARS
from PyQt6 import QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel, QLineEdit, QHBoxLayout


class MenuWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Center Frequency Slider and Input
        self.center_freq_label = QLabel(
            f"Center Frequency: {VARS['CENTER_FREQUENCY'] / 1e6:.1f} MHz")
        self.center_freq_slider = QSlider(Qt.Orientation.Horizontal)
        # Range from 40.0 MHz to 160.0 MHz (scaled by 10)
        self.center_freq_slider.setRange(400, 16000)
        self.center_freq_slider.setSingleStep(1)  # Step change of 0.1 MHz
        # Set value: convert Hz -> MHz then scale by 10
        self.center_freq_slider.setValue(
            int((VARS['CENTER_FREQUENCY'] / 1e6) * 10))
        self.center_freq_slider.valueChanged.connect(
            self.update_center_freq_label)

        self.center_freq_input = QLineEdit(
            f"{VARS['CENTER_FREQUENCY'] / 1e6:.1f}")
        self.center_freq_input.setValidator(
            QtGui.QDoubleValidator(40.0, 1600.0, 1))
        self.center_freq_input.returnPressed.connect(
            self.update_center_freq_slider)

        center_freq_layout = QHBoxLayout()
        center_freq_layout.addWidget(self.center_freq_label)
        center_freq_layout.addWidget(self.center_freq_input)

        layout.addLayout(center_freq_layout)
        layout.addWidget(self.center_freq_slider)

        # Sampling Rate Slider and Input (unchanged)
        self.sampling_rate_label = QLabel(
            f"Sampling Rate: {VARS['SAMPLE_RATE'] / 1e6:.3f} MSps")
        self.sampling_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.sampling_rate_slider.setRange(1000, 2000)
        self.sampling_rate_slider.setSingleStep(1)
        self.sampling_rate_slider.setValue(int(VARS['SAMPLE_RATE'] / 1e3))
        self.sampling_rate_slider.valueChanged.connect(
            self.update_sampling_rate_label)

        self.sampling_rate_input = QLineEdit(
            f"{VARS['SAMPLE_RATE'] / 1e6:.3f}")
        self.sampling_rate_input.setValidator(
            QtGui.QDoubleValidator(1.000, 2.000, 3))
        self.sampling_rate_input.returnPressed.connect(
            self.update_sampling_rate_slider)

        sampling_rate_layout = QHBoxLayout()
        sampling_rate_layout.addWidget(self.sampling_rate_label)
        sampling_rate_layout.addWidget(self.sampling_rate_input)

        layout.addLayout(sampling_rate_layout)
        layout.addWidget(self.sampling_rate_slider)

        self.setLayout(layout)

    def update_center_freq_label(self, value):
        # Convert slider integer to MHz with 1 decimal: value/10.0
        center_freq = value / 10.0
        self.center_freq_label.setText(
            f"Center Frequency: {center_freq:.1f} MHz")
        self.center_freq_input.setText(f"{center_freq:.1f}")
        VARS['CENTER_FREQUENCY'] = center_freq * 1e6  # Update in Hz
        print(f"Updated Center Frequency: {VARS['CENTER_FREQUENCY']} Hz")
        self.parent.plot_constellation_widget.flowgraph.set_frequency(
            VARS['CENTER_FREQUENCY'])
        self.parent.plot_constellation_widget.flowgraph.mute_fm()
        self.parent.statusBar().showMessage("FM muted due to frequency change", 2000)

    def update_center_freq_slider(self):
        value = float(self.center_freq_input.text())
        self.center_freq_slider.setValue(int(value * 10))
        VARS['CENTER_FREQUENCY'] = value * 1e6
        print(f"Updated Center Frequency: {VARS['CENTER_FREQUENCY']} Hz")
        self.parent.plot_constellation_widget.flowgraph.set_frequency(
            VARS['CENTER_FREQUENCY'])
        self.parent.plot_constellation_widget.flowgraph.mute_fm()
        self.parent.statusBar().showMessage("FM muted due to frequency change", 2000)

    def update_sampling_rate_label(self, value):
        sampling_rate = value / 1000.0  # Scale back to floating-point value
        self.sampling_rate_label.setText(
            f"Sampling Rate: {sampling_rate:.3f} MSps")
        self.sampling_rate_input.setText(f"{sampling_rate:.3f}")
        VARS['SAMPLE_RATE'] = sampling_rate * 1e6
        print(f"Updated Sampling Rate: {VARS['SAMPLE_RATE']} Sps")
        self.parent.plot_constellation_widget.flowgraph.set_sample_rate(
            VARS['SAMPLE_RATE'])

    def update_sampling_rate_slider(self):
        value = float(self.sampling_rate_input.text())
        self.sampling_rate_slider.setValue(int(value * 1000))
        VARS['SAMPLE_RATE'] = value * 1e6
        print(f"Updated Sampling Rate: {VARS['SAMPLE_RATE']} Sps")
        self.parent.plot_constellation_widget.flowgraph.set_sample_rate(
            VARS['SAMPLE_RATE'])


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = MenuWidget()
    window.show()
    sys.exit(app.exec())
