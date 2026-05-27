import numpy as np
from faster_whisper import WhisperModel

class StreamingTranscriptionModule:
    """
    Converts accumulated voice audio buffer chunks to plain text
    using an int8 quantized baseline faster-whisper model.
    """
    def __init__(self, model_size="base", compute_type="int8"):
        print(f"[ASR] Loading Faster-Whisper '{model_size}' ({compute_type})...")
        # device="auto" picks GPU if available, else CPU.
        self.model = WhisperModel(model_size, device="auto", compute_type=compute_type)
        print("[ASR] Model loaded.")

    def transcribe_buffer(self, audio_buffer_bytes):
        """
        Takes raw accumulated 16-bit PCM bytes, normalizes, and transcribes.
        Returns the transcription text.
        """
        if not audio_buffer_bytes:
            return ""
            
        audio_np = np.frombuffer(audio_buffer_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe using Beam Search (beam_size=5 is standard)
        segments, info = self.model.transcribe(audio_np, beam_size=5, language="en", vad_filter=True)
        
        text = " ".join([segment.text for segment in segments]).strip()
        return text
