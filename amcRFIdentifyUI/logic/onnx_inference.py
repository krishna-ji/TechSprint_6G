import numpy as np
import onnxruntime as ort

class ONNXInference:
    """
    Class that preprocesses IQ data and runs ONNX inference.
    """

    def __init__(self, model_path):
        """
        Initialize the ONNX inference class with the model path.
        """
        self.window_size = 224
        self.overlap = self.window_size // 2
        self.step_size = self.window_size - self.overlap
        
        # Load ONNX model
        self.model_path = model_path
        self.sess = ort.InferenceSession(self.model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def normalize_X(self, X):
        epsilon = 1e-8 
        X_min = np.min(X, axis=0, keepdims=True)
        X_max = np.max(X, axis=0, keepdims=True)
        X_range = X_max - X_min
        mask = X_range == 0
        X = np.where(mask, 0, 2 * (X - X_min) / (X_range + epsilon) - 1)
        return X

    def windowing_function(self, input_data):
        if input_data.shape != (1024, 2):
            raise ValueError("Input data must have shape (1024, 2).")
        output_sequences = [input_data[i:i + self.window_size] 
                            for i in range(0, 1024 - self.window_size, self.step_size)]
        out = np.array(output_sequences)
        return out
    
    def amplitude_phase_seq_batch(self, iq_seq):
        amplitude = np.sqrt(np.sum(np.square(iq_seq), axis=2, keepdims=True))
        phase = np.arctan2(iq_seq[:, :, 1], iq_seq[:, :, 0])[..., np.newaxis]
        return np.concatenate([iq_seq, amplitude, phase], axis=2)
    
    def data_preparation_alexnet(self, input_data):
        iq_channel = input_data[:, :, :2]  # Shape: (32, 224, 2)
        amp_channel = input_data[:, :, 2:3]  # Shape: (32, 224, 1)
        phase_channel = input_data[:, :, 3:]  # Shape: (32, 224, 1)

        # Repeat the channels to match the required shape
        iq_channel = np.repeat(iq_channel, 112, axis=2)  # Shape: (32, 224, 224)
        amp_channel = np.repeat(amp_channel, 224, axis=2)  # Shape: (32, 224, 224)
        phase_channel = np.repeat(phase_channel, 224, axis=2)  # Shape: (32, 224, 224)

        # Add a new dimension for the channel
        iq_channel = iq_channel[:, np.newaxis, :, :]  # Shape: (32, 1, 224, 224)
        amp_channel = amp_channel[:, np.newaxis, :, :]  # Shape: (32, 1, 224, 224)
        phase_channel = phase_channel[:, np.newaxis, :, :]  # Shape: (32, 1, 224, 224)

        # Stack the channels along the new dimension
        return np.concatenate([iq_channel, amp_channel, phase_channel], axis=1)  # Shape: (32, 3, 224, 224)
    
    def softmax(self, logits):
        """
        Compute softmax probabilities from logits.
        """
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probabilities

    def preprocess(self, iq_data):
        iq_data = np.array(iq_data)
        # take last 1024 samples
        iq_data = np.stack([iq_data.real, iq_data.imag], axis=1)
        iq_data = self.normalize_X(iq_data)
        windowed_data = self.windowing_function(iq_data)
        augmented_data = self.amplitude_phase_seq_batch(windowed_data)  # Shape: (8, 224, 4)
        return self.data_preparation_alexnet(augmented_data)  # Shape: (8, 3, 224, 224)

    def infer_logits(self, iq_data):
        """
        Run inference and return logits.
        """
        processed_data = self.preprocess(iq_data)  # Ensure correct shape
        # Convert to float32 (ONNX requires specific types)
        input_data = processed_data.astype(np.float32)
        # Run ONNX inference
        ort_inputs = {self.sess.get_inputs()[0].name: input_data}
        result = self.sess.run(None, ort_inputs)
        return result[0]

    def infer_probabilities(self, iq_data):
        """
        Run inference and return probabilities.
        """
        logits = self.infer_logits(iq_data)
        probabilities = self.softmax(logits)
        return probabilities