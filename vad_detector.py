from __future__ import annotations

import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort


LOGGER = logging.getLogger(__name__)


class BargeInDetector:
    """
    Silero VAD wrapper with asymmetric state tracking.

    It treats user speech as confirmed only after `barge_in_frames` consecutive
    speech-positive 32 ms windows, then triggers barge-in if the agent is active.
    """

    MODEL_URL = (
        "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
        "src/silero_vad/data/silero_vad.onnx"
    )

    def __init__(
        self,
        threshold: float = 0.5,
        negative_threshold: float = 0.35,
        energy_threshold_rms: int = 500,
        barge_in_frames: int = 4,
        sample_rate: int = 16000,
        model_path: str = "silero_vad.onnx",
    ) -> None:
        self.threshold = threshold
        self.negative_threshold = negative_threshold
        self.energy_threshold_rms = energy_threshold_rms
        self.barge_in_frames = barge_in_frames
        self.sample_rate = sample_rate
        self.model_path = Path(model_path)
        self._lock = threading.RLock()

        self._ensure_model_exists()
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.reset_states()

        self.consecutive_voice_frames = 0
        self.consecutive_silence_frames = 0
        self.is_user_speaking = False
        self.last_probability = 0.0
        self.last_rms = 0

    def reset_states(self) -> None:
        with self._lock:
            self.h = np.zeros((2, 1, 64), dtype=np.float32)
            self.c = np.zeros((2, 1, 64), dtype=np.float32)
            self.state = np.zeros((2, 1, 128), dtype=np.float32)

    def process_frame(self, audio_bytes: bytes, is_agent_speaking: bool) -> bool:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return False
        if audio.size != 512:
            audio = self._pad_or_trim(audio, 512)

        with self._lock:
            prob = self._infer_probability(audio)
            rms = int(np.sqrt(np.mean((audio * 32768.0) ** 2)))
            self.last_probability = prob
            self.last_rms = rms
            if prob >= self.threshold or rms >= self.energy_threshold_rms:
                self.consecutive_voice_frames += 1
                self.consecutive_silence_frames = 0
            elif prob <= self.negative_threshold and rms < self.energy_threshold_rms * 0.55:
                self.consecutive_silence_frames += 1
                self.consecutive_voice_frames = 0

            self.is_user_speaking = self.consecutive_voice_frames >= self.barge_in_frames
            return bool(is_agent_speaking and self.is_user_speaking)

    def mark_silence_if_needed(self, release_frames: int = 8) -> bool:
        with self._lock:
            if self.consecutive_silence_frames >= release_frames:
                self.is_user_speaking = False
            return self.is_user_speaking

    def _infer_probability(self, audio: np.ndarray) -> float:
        inputs = {
            "input": audio.reshape(1, -1),
            "sr": np.array(self.sample_rate, dtype=np.int64),
        }
        if "h" in self.input_names and "c" in self.input_names:
            inputs["h"] = self.h
            inputs["c"] = self.c
            outs = self.session.run(None, inputs)
            self.h = outs[1]
            self.c = outs[2]
            return float(np.ravel(outs[0])[0])
        if "state" in self.input_names:
            inputs["state"] = self.state
            outs = self.session.run(None, inputs)
            self.state = outs[1]
            return float(np.ravel(outs[0])[0])
        outs = self.session.run(None, inputs)
        return float(np.ravel(outs[0])[0])

    def _ensure_model_exists(self) -> None:
        if self.model_path.exists():
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Downloading Silero VAD ONNX model to %s", self.model_path)
        try:
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
        except Exception as exc:
            raise RuntimeError(
                "Silero VAD model is missing and could not be downloaded. "
                f"Download it manually to {self.model_path.resolve()}."
            ) from exc

    @staticmethod
    def _pad_or_trim(audio: np.ndarray, size: int) -> np.ndarray:
        if audio.size > size:
            return audio[:size]
        return np.pad(audio, (0, size - audio.size))
