import os
import urllib.request
import numpy as np
import onnxruntime as ort

class BargeInDetector:
    """
    Implements the Silero VAD frame checking, asynchronous double-state monitoring,
    and audio output cancellation triggers.
    """
    def __init__(self, threshold=0.5, barge_in_frames=4, sample_rate=16000):
        self.threshold = threshold
        self.barge_in_frames = barge_in_frames
        self.sample_rate = sample_rate
        
        self.model_path = "silero_vad.onnx"
        self._ensure_model_exists()
        
        # Load ONNX model
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'], sess_options=opts)
        
        self.reset_states()
        
        # Runtime states
        self.consecutive_voice_frames = 0
        self.is_user_speaking = False

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"[VAD] Downloading Silero VAD ONNX model to {self.model_path}...")
            url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            urllib.request.urlretrieve(url, self.model_path)
            print("[VAD] Download complete.")

    def reset_states(self):
        """Reset internal GRU states for the VAD."""
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def process_frame(self, audio_bytes, is_agent_speaking_flag):
        """
        Process a 32ms audio frame (512 samples @ 16kHz).
        Updates `is_user_speaking` and returns a boolean indicating if a Barge-In just occurred.
        """
        # Convert bytes to float32 normalized
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # Silero VAD expects [batch, sequence]
        audio_in = audio_np[np.newaxis, :]
        
        ort_inputs = {
            'input': audio_in,
            'sr': np.array(self.sample_rate, dtype=np.int64),
            'h': self.h,
            'c': self.c
        }
        
        ort_outs = self.session.run(None, ort_inputs)
        out, self.h, self.c = ort_outs
        
        speech_prob = out[0][0]
        
        if speech_prob > self.threshold:
            self.consecutive_voice_frames += 1
            if self.consecutive_voice_frames >= self.barge_in_frames:
                self.is_user_speaking = True
                
                # Check for Barge-In Condition: User spoke while agent is speaking
                if is_agent_speaking_flag:
                    return True # Barge-In Triggered!
        else:
            # Add a small buffer before immediately toggling off to prevent micro-dropouts
            if self.consecutive_voice_frames > 0:
                self.consecutive_voice_frames -= 1
            else:
                self.is_user_speaking = False
                
        return False
