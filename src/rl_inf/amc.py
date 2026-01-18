"""
AMC Classifier - ONNX Inference

Automatic Modulation Classification using pre-trained ONNX model.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple


class AMCClassifier:
    """
    Automatic Modulation Classifier using ONNX inference.
    
    Preprocesses IQ samples and classifies modulation type.
    The model is trained on 10 modulation classes from RadioML/similar dataset.
    
    Parameters
    ----------
    model_path : str | Path
        Path to ONNX model file
    window_size : int
        Size of sliding window for preprocessing (default: 224)
    """
    
    # Class labels - 9 modulations as trained (output shape is 10 but 9 modulation schemes)
    # The classes are based on the training notebook which uses modulation_schemes = range(9)
    # Mapping based on common RadioML/GOLD_XYZ_OSC datasets:
    CLASSES = [
        "OOK",       # 0 - On-Off Keying
        "4ASK",      # 1 - 4-Level Amplitude Shift Keying  
        "8ASK",      # 2 - 8-Level Amplitude Shift Keying
        "BPSK",      # 3 - Binary Phase Shift Keying
        "QPSK",      # 4 - Quadrature Phase Shift Keying
        "8PSK",      # 5 - 8-Level Phase Shift Keying
        "16QAM",     # 6 - 16-Quadrature Amplitude Modulation
        "AM-SSB",    # 7 - AM Single Sideband
        "AM-DSB",    # 8 - AM Double Sideband
        "FM",        # 9 - Frequency Modulation
    ]
    
    def __init__(self, model_path: str | Path, window_size: int = 224):
        self.model_path = Path(model_path)
        self.window_size = window_size
        self.overlap = window_size // 2
        self.step_size = window_size - self.overlap
        
        # Load ONNX model
        self.session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalize to [-1, 1]."""
        epsilon = 1e-8
        X_min = np.min(X, axis=0, keepdims=True)
        X_max = np.max(X, axis=0, keepdims=True)
        X_range = X_max - X_min
        mask = X_range == 0
        return np.where(mask, 0, 2 * (X - X_min) / (X_range + epsilon) - 1)
    
    def _windowing(self, data: np.ndarray) -> np.ndarray:
        """Create overlapping windows from input."""
        if data.shape != (1024, 2):
            raise ValueError("Input must have shape (1024, 2)")
        windows = [
            data[i:i + self.window_size] 
            for i in range(0, 1024 - self.window_size, self.step_size)
        ]
        return np.array(windows)
    
    def _add_amplitude_phase(self, iq_seq: np.ndarray) -> np.ndarray:
        """Add amplitude and phase channels."""
        amplitude = np.sqrt(np.sum(np.square(iq_seq), axis=2, keepdims=True))
        phase = np.arctan2(iq_seq[:, :, 1], iq_seq[:, :, 0])[..., np.newaxis]
        return np.concatenate([iq_seq, amplitude, phase], axis=2)
    
    def _prepare_alexnet(self, data: np.ndarray) -> np.ndarray:
        """Convert to AlexNet input format (batch, 3, 224, 224)."""
        iq_channel = np.repeat(data[:, :, :2], 112, axis=2)
        amp_channel = np.repeat(data[:, :, 2:3], 224, axis=2)
        phase_channel = np.repeat(data[:, :, 3:], 224, axis=2)
        
        iq_channel = iq_channel[:, np.newaxis, :, :]
        amp_channel = amp_channel[:, np.newaxis, :, :]
        phase_channel = phase_channel[:, np.newaxis, :, :]
        
        return np.concatenate([iq_channel, amp_channel, phase_channel], axis=1)
    
    def preprocess(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline for IQ samples.
        
        Parameters
        ----------
        iq_data : np.ndarray
            Complex IQ samples of shape (1024,)
        
        Returns
        -------
        np.ndarray
            Preprocessed data ready for inference (batch, 3, 224, 224)
        """
        # Convert complex to I/Q channels
        iq_2d = np.stack([iq_data.real, iq_data.imag], axis=1)
        iq_2d = self._normalize(iq_2d)
        windowed = self._windowing(iq_2d)
        augmented = self._add_amplitude_phase(windowed)
        return self._prepare_alexnet(augmented)
    
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def predict_logits(self, iq_data: np.ndarray) -> np.ndarray:
        """Run inference and return raw logits."""
        processed = self.preprocess(iq_data).astype(np.float32)
        return self.session.run(None, {self.input_name: processed})[0]
    
    def predict_proba(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        iq_data : np.ndarray
            Complex IQ samples of shape (1024,)
        
        Returns
        -------
        np.ndarray
            Probability distribution over classes (1, n_classes)
            Note: Model internally aggregates 8 windows into 1 prediction.
        """
        logits = self.predict_logits(iq_data)
        return self._softmax(logits)
    
    def predict(self, iq_data: np.ndarray) -> Tuple[str, float]:
        """
        Predict modulation class.
        
        Parameters
        ----------
        iq_data : np.ndarray
            Complex IQ samples of shape (1024,)
        
        Returns
        -------
        Tuple[str, float]
            (class_name, confidence)
        """
        probs = self.predict_proba(iq_data)
        # Average across windows
        avg_probs = probs.mean(axis=0)
        class_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[class_idx])
        return self.CLASSES[class_idx], confidence
