from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AudioQualityReport:
    duration_seconds: float
    rms: int
    peak: int
    gain_db: float
    clipped_ratio: float


@dataclass(frozen=True)
class TranscriptCandidate:
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    duration_seconds: float


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    confidence: str
    reason: str = ""


class SpeechPreprocessor:
    """Small local speech enhancement layer before Whisper."""

    def __init__(
        self,
        target_rms: int = 3600,
        max_gain_db: float = 18.0,
        silence_rms: int = 90,
        edge_padding_ms: int = 180,
    ) -> None:
        self.target_rms = target_rms
        self.max_gain_db = max_gain_db
        self.silence_rms = silence_rms
        self.edge_padding_ms = edge_padding_ms

    def prepare(self, audio_buffer_bytes: bytes, sample_rate: int) -> tuple[np.ndarray, AudioQualityReport]:
        samples = np.frombuffer(audio_buffer_bytes, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return np.array([], dtype=np.float32), AudioQualityReport(0.0, 0, 0, 0.0, 0.0)

        samples = samples - float(np.mean(samples))
        samples = self._trim_silence(samples, sample_rate)
        rms = self._rms(samples)
        gain_db = 0.0

        if rms > 0:
            desired_gain = self.target_rms / max(rms, 1)
            max_gain = 10 ** (self.max_gain_db / 20)
            gain = min(desired_gain, max_gain)
            samples = samples * gain
            gain_db = 20 * math.log10(max(gain, 1e-6))

        samples = np.clip(samples, -32768, 32767)
        clipped_ratio = float(np.mean(np.abs(samples) >= 32760)) if samples.size else 0.0
        normalized = (samples / 32768.0).astype(np.float32)
        report = AudioQualityReport(
            duration_seconds=float(samples.size / sample_rate),
            rms=self._rms(samples),
            peak=int(np.max(np.abs(samples))) if samples.size else 0,
            gain_db=gain_db,
            clipped_ratio=clipped_ratio,
        )
        return normalized, report

    def _trim_silence(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        frame = max(1, int(sample_rate * 0.02))
        if samples.size <= frame:
            return samples
        usable = (samples.size // frame) * frame
        frames = samples[:usable].reshape(-1, frame)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        voiced = np.flatnonzero(rms >= self.silence_rms)
        if voiced.size == 0:
            return samples
        pad = max(1, int((self.edge_padding_ms / 1000) * sample_rate / frame))
        start_frame = max(0, int(voiced[0]) - pad)
        end_frame = min(frames.shape[0], int(voiced[-1]) + pad + 1)
        return samples[start_frame * frame : end_frame * frame]

    @staticmethod
    def _rms(samples: np.ndarray) -> int:
        if samples.size == 0:
            return 0
        return int(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


class TranscriptPostProcessor:
    """Project-owned transcript cleanup and quality gate."""

    FILLER_RE = re.compile(r"\b(?:um+|uh+|erm|ah+)\b", re.IGNORECASE)
    SPACE_RE = re.compile(r"\s+")
    REPEATED_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b){2,}", re.IGNORECASE)

    def __init__(self, custom_terms: Iterable[str] | None = None) -> None:
        self.custom_terms = tuple(custom_terms or ())

    def process(self, candidates: list[TranscriptCandidate]) -> TranscriptResult:
        if not candidates:
            return TranscriptResult("", "empty", "No speech segment survived ASR.")
        usable = [item for item in candidates if item.text.strip()]
        if not usable:
            return TranscriptResult("", "empty", "Whisper returned only empty segments.")

        text = " ".join(item.text for item in usable)
        text = self._clean_text(text)
        worst_no_speech = max(item.no_speech_prob for item in usable)
        avg_logprob = sum(item.avg_logprob for item in usable) / len(usable)
        compression = max(item.compression_ratio for item in usable)

        if not text:
            return TranscriptResult("", "empty", "Transcript cleaned to empty text.")
        if len(text) < 3:
            return TranscriptResult("", "low", "Transcript was too short to trust.")
        if worst_no_speech > 0.78 and avg_logprob < -0.85:
            return TranscriptResult("", "low", "Audio looked like silence or background noise.")
        if compression > 2.8:
            return TranscriptResult("", "low", "Transcript looked repetitive or unstable.")

        confidence = "high" if avg_logprob > -0.45 and worst_no_speech < 0.45 else "medium"
        return TranscriptResult(text, confidence)

    def _clean_text(self, text: str) -> str:
        text = self.FILLER_RE.sub("", text)
        text = self.REPEATED_WORD_RE.sub(r"\1", text)
        text = self.SPACE_RE.sub(" ", text).strip(" .")
        if text:
            text = text[0].upper() + text[1:]
        return self._restore_custom_terms(text)

    def _restore_custom_terms(self, text: str) -> str:
        if not self.custom_terms:
            return text
        terms_lower = [term.lower() for term in self.custom_terms]
        words = text.split()
        restored = []
        for word in words:
            bare = word.strip(".,!?;:")
            match = get_close_matches(bare.lower(), terms_lower, n=1, cutoff=0.88)
            if match:
                term = next(item for item in self.custom_terms if item.lower() == match[0])
                restored.append(word.replace(bare, term))
            else:
                restored.append(word)
        return " ".join(restored)
