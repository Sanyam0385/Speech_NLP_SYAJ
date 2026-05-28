from __future__ import annotations

import logging
import queue
import threading
import audioop
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pyaudio


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaybackChunk:
    """PCM playback bytes plus the text slice represented by those bytes."""

    pcm: bytes
    text_delta: str = ""
    final: bool = False


class AudioStreamingEngine:
    """
    Owns PyAudio input/output streams and exposes thread-safe queues.

    Input is captured as 16 kHz, 16-bit mono PCM with a 512-sample analysis
    window and 256-sample hop, giving 32 ms VAD windows with 50% overlap.
    """

    def __init__(
        self,
        rate: int = 16000,
        input_rate: Optional[int] = None,
        output_rate: Optional[int] = None,
        frame_size: int = 512,
        hop_size: int = 256,
        channels: int = 1,
        input_channels: Optional[int] = None,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        playback_progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.rate = rate
        self.input_rate = input_rate or rate
        self.output_rate = output_rate or rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.channels = channels
        self.input_channels = input_channels or channels
        self.format = pyaudio.paInt16
        self.sample_width = 2
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.playback_progress_callback = playback_progress_callback

        self.input_queue: queue.Queue[bytes] = queue.Queue(maxsize=2000)
        self.output_queue: queue.Queue[PlaybackChunk] = queue.Queue(maxsize=500)

        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._input_stream: Optional[pyaudio.Stream] = None
        self._output_stream: Optional[pyaudio.Stream] = None
        self._output_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._flush_generation = 0
        self._ratecv_state = None
        self._lock = threading.RLock()

    @staticmethod
    def list_devices() -> list[dict]:
        pyaudio_instance = pyaudio.PyAudio()
        try:
            devices = []
            for index in range(pyaudio_instance.get_device_count()):
                info = pyaudio_instance.get_device_info_by_index(index)
                devices.append(
                    {
                        "index": index,
                        "name": info.get("name", ""),
                        "inputs": int(info.get("maxInputChannels", 0)),
                        "outputs": int(info.get("maxOutputChannels", 0)),
                        "default_rate": int(info.get("defaultSampleRate", 0)),
                    }
                )
            return devices
        finally:
            pyaudio_instance.terminate()

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def _input_callback(self, in_data, frame_count, time_info, status_flags):
        if status_flags:
            LOGGER.warning("PyAudio input status flag: %s", status_flags)
        if not self.is_running:
            return (None, pyaudio.paComplete)
        try:
            normalized = self._normalize_input_pcm(in_data)
            if normalized:
                self.input_queue.put_nowait(normalized)
        except queue.Full:
            try:
                self.input_queue.get_nowait()
                normalized = self._normalize_input_pcm(in_data)
                if normalized:
                    self.input_queue.put_nowait(normalized)
                LOGGER.warning("Input queue full; dropped oldest microphone hop.")
            except queue.Empty:
                pass
        return (None, pyaudio.paContinue)

    def start_streams(self) -> None:
        with self._lock:
            if self.is_running:
                return
            self._pyaudio = pyaudio.PyAudio()
            self._running.set()
            self._resolve_input_format()
            self._input_stream = self._open_input_stream_with_fallbacks()
            output_rate = self._resolve_output_rate()
            self._output_stream = self._pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=output_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.hop_size,
                start=False,
            )
            self.output_rate = output_rate
            self._output_thread = threading.Thread(
                target=self._output_worker,
                name="audio-output",
                daemon=True,
            )
            self._output_thread.start()
            self._output_stream.start_stream()
            self._input_stream.start_stream()
            LOGGER.info(
                "Audio streams started. input_device=%s input=%sHz/%sch -> processing=%sHz/mono; output_device=%s output=%sHz",
                self.input_device_index,
                self.input_rate,
                self.input_channels,
                self.rate,
                self.output_device_index,
                self.output_rate,
            )

    def stop_streams(self) -> None:
        with self._lock:
            self._running.clear()
            self.flush_output()
            self._close_stream(self._input_stream)
            self._close_stream(self._output_stream)
            self._input_stream = None
            self._output_stream = None
            if self._output_thread and self._output_thread.is_alive():
                self._output_thread.join(timeout=1.0)
            self._output_thread = None
            if self._pyaudio:
                self._pyaudio.terminate()
            self._pyaudio = None
            self._drain(self.input_queue)
            self._drain(self.output_queue)
            LOGGER.info("Audio streams stopped.")

    def enqueue_playback(self, chunk: PlaybackChunk, timeout: float = 0.25) -> bool:
        if not self.is_running:
            return False
        try:
            self.output_queue.put(chunk, timeout=timeout)
            return True
        except queue.Full:
            LOGGER.warning("Output queue full; dropping synthesized playback chunk.")
            return False

    def flush_output(self) -> None:
        with self._lock:
            self._flush_generation += 1
            self._drain(self.output_queue)
            if self._output_stream:
                try:
                    self._output_stream.stop_stream()
                    self._output_stream.start_stream()
                except Exception:
                    LOGGER.exception("Could not hard-flush output stream.")

    def _output_worker(self) -> None:
        while self.is_running:
            generation = self._flush_generation
            try:
                chunk = self.output_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if generation != self._flush_generation:
                continue
            if not chunk.pcm:
                continue
            try:
                if self._output_stream:
                    self._output_stream.write(chunk.pcm, exception_on_underflow=False)
                if (
                    chunk.text_delta
                    and self.playback_progress_callback
                    and generation == self._flush_generation
                    and self.is_running
                ):
                    self.playback_progress_callback(chunk.text_delta)
            except OSError:
                LOGGER.exception("Audio output underflow/device error.")
            except Exception:
                LOGGER.exception("Unexpected audio output error.")

    @staticmethod
    def _close_stream(stream: Optional[pyaudio.Stream]) -> None:
        if not stream:
            return
        try:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        except Exception:
            LOGGER.exception("Error while closing audio stream.")

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _resolve_output_rate(self) -> int:
        if not self._pyaudio or self.output_device_index is None:
            return self.output_rate
        info = self._pyaudio.get_device_info_by_index(self.output_device_index)
        default_rate = int(info.get("defaultSampleRate", self.output_rate))
        if default_rate and default_rate != self.output_rate:
            LOGGER.info(
                "Using output device default sample rate %s Hz instead of %s Hz.",
                default_rate,
                self.output_rate,
            )
            return default_rate
        return self.output_rate

    def _resolve_input_format(self) -> None:
        if not self._pyaudio or self.input_device_index is None:
            return
        info = self._pyaudio.get_device_info_by_index(self.input_device_index)
        default_rate = int(info.get("defaultSampleRate", self.input_rate))
        max_channels = int(info.get("maxInputChannels", self.input_channels))
        if default_rate:
            self.input_rate = default_rate
        if max_channels > 0:
            self.input_channels = max_channels

    def _open_input_stream_with_fallbacks(self):
        if not self._pyaudio:
            raise RuntimeError("PyAudio is not initialized.")
        rates = self._dedupe([self.input_rate, self.rate, 48000, 44100])
        channels = self._dedupe([self.input_channels, min(self.input_channels, 2), 1])
        errors = []
        for rate in rates:
            for channel_count in channels:
                try:
                    self.input_rate = int(rate)
                    self.input_channels = int(channel_count)
                    return self._pyaudio.open(
                        format=self.format,
                        channels=self.input_channels,
                        rate=self.input_rate,
                        input=True,
                        input_device_index=self.input_device_index,
                        frames_per_buffer=self._device_hop_size(),
                        stream_callback=self._input_callback,
                        start=False,
                    )
                except OSError as exc:
                    errors.append(f"{rate}Hz/{channel_count}ch -> {exc}")
        joined = "; ".join(errors)
        raise OSError(f"Could not open input device {self.input_device_index}. Tried: {joined}")

    def _device_hop_size(self) -> int:
        return max(1, round(self.hop_size * self.input_rate / self.rate))

    def _normalize_input_pcm(self, pcm: bytes) -> bytes:
        mono = self._downmix_to_mono(pcm, self.input_channels)
        if self.input_rate != self.rate:
            mono, self._ratecv_state = audioop.ratecv(
                mono,
                self.sample_width,
                1,
                self.input_rate,
                self.rate,
                self._ratecv_state,
            )
        return mono

    def _downmix_to_mono(self, pcm: bytes, channels: int) -> bytes:
        if channels <= 1:
            return pcm
        samples = np.frombuffer(pcm, dtype=np.int16)
        usable = (samples.size // channels) * channels
        if usable == 0:
            return b""
        frames = samples[:usable].reshape(-1, channels).astype(np.int32)
        rms_by_channel = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=0))
        loudest_channel = int(np.argmax(rms_by_channel))
        mono = np.clip(frames[:, loudest_channel], -32768, 32767).astype(np.int16)
        return mono.tobytes()

    @staticmethod
    def _dedupe(values: list[int]) -> list[int]:
        seen = set()
        result = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                result.append(int(value))
        return result
