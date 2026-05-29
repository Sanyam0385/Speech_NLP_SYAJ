from __future__ import annotations

import logging
import platform
import threading

import numpy as np
from faster_whisper import WhisperModel

from speech_processing import (
    SpeechPreprocessor,
    TranscriptCandidate,
    TranscriptPostProcessor,
)


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
        self.preprocessor = SpeechPreprocessor()
        self.postprocessor = TranscriptPostProcessor(
            custom_terms=("Ollama", "Whisper", "Barge-in", "VAD", "TTS", "ASR")
        )
        if device is None:
            device = "cpu" if platform.system() == "Windows" else "auto"
        LOGGER.info("Loading faster-whisper model=%s device=%s compute=%s", model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_buffer(self, audio_buffer_bytes: bytes, sample_rate: int = 16000) -> str:
        if not audio_buffer_bytes:
            return ""
        audio, quality = self.preprocessor.prepare(audio_buffer_bytes, sample_rate)
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
            candidates: list[TranscriptCandidate] = []
            for segment in segments:
                candidates.append(
                    TranscriptCandidate(
                        text=(segment.text or "").strip(),
                        avg_logprob=float(getattr(segment, "avg_logprob", -0.8) or -0.8),
                        no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
                        compression_ratio=float(getattr(segment, "compression_ratio", 1.0) or 1.0),
                        duration_seconds=float(max(0.0, (getattr(segment, "end", 0.0) or 0.0) - (getattr(segment, "start", 0.0) or 0.0))),
                    )
                )
            result = self.postprocessor.process(candidates)
            if result.reason:
                LOGGER.info(
                    "ASR post-process confidence=%s reason=%s rms=%s gain_db=%.2f clipped=%.3f",
                    result.confidence,
                    result.reason,
                    quality.rms,
                    quality.gain_db,
                    quality.clipped_ratio,
                )
            return result.text
