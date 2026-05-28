from __future__ import annotations

import logging
import platform
import threading

import numpy as np
from faster_whisper import WhisperModel


LOGGER = logging.getLogger(__name__)


class StreamingTranscriptionModule:
    """Turn-completion ASR over accumulated 16-bit PCM using faster-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        compute_type: str = "int8",
        device: str | None = None,
        language: str = "en",
    ) -> None:
        self.language = language
        self._lock = threading.Lock()
        if device is None:
            device = "cpu" if platform.system() == "Windows" else "auto"
        LOGGER.info("Loading faster-whisper model=%s device=%s compute=%s", model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_buffer(self, audio_buffer_bytes: bytes, sample_rate: int = 16000) -> str:
        if not audio_buffer_bytes:
            return ""
        audio = np.frombuffer(audio_buffer_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size < sample_rate * 0.18:
            return ""
        with self._lock:
            segments, _info = self.model.transcribe(
                audio,
                beam_size=1,
                language=self.language,
                vad_filter=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()
