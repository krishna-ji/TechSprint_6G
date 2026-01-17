import numpy as np
import time

class FullCaptureFlowgraph:
    """
    A purely Mock flowgraph that does not use GNU Radio.
    Generates dummy IQ data using numpy.
    Used when gnuradio is not installed.
    """

    def __init__(self, samp_rate=1.024e6, radio_freq=96.5e6):
        self.samp_rate = samp_rate
        self.radio_freq = radio_freq
        self.file_sink_connected = False
        self.file_path = None
        self._running = False

    def start(self):
        self._running = True
        print("Mock Flowgraph started (No GNU Radio)")

    def stop(self):
        self._running = False
        print("Mock Flowgraph stopped")

    def wait(self):
        pass

    def get_iq_sample(self):
        """
        Returns a 1024-sample block of dummy data.
        """
        # Simulate processing time
        time.sleep(0.01) # ~10ms
        
        # Generate random complex noise
        real = np.random.normal(0, 0.1, 1024)
        imag = np.random.normal(0, 0.1, 1024)
        
        # Add a mock signal (sine wave)
        t = np.arange(1024)
        # Use a changing phase to make it look alive
        phase = (time.time() * 10) % (2*np.pi)
        signal = 0.5 * np.exp(1j * (2 * np.pi * 0.05 * t + phase))
        
        data = (real + 1j * imag + signal).astype(np.complex64)
        return data

    def open_file(self, file_path):
        self.file_path = file_path
        self.file_sink_connected = True
        print(f"Mock: Opening file {file_path}")

    def close_file(self):
        self.file_sink_connected = False
        print("Mock: Closing file")

    def mute_fm(self):
        print("Mock: FM audio muted.")

    def unmute_fm(self):
        print("Mock: FM audio unmuted.")

    def set_frequency(self, freq):
        self.radio_freq = freq
        print(f"Mock: Updated frequency to {freq} Hz")

    def set_sample_rate(self, samp_rate):
        self.samp_rate = samp_rate
        print(f"Mock: Updated sample rate to {samp_rate} Sps")
    
    def lock(self):
        pass
        
    def unlock(self):
        pass
        
    def disconnect(self, *args):
        pass
        
    def connect(self, *args):
        pass
