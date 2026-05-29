from __future__ import annotations

import logging
import math
import audioop
import queue
import signal
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional

from audio_streaming import AudioStreamingEngine
from dialogue_memory import DialogueMemoryManager
from llm_tts_pipeline import LLMTTSPipeline
from nlp_layer import LightweightNLPLayer
from performance_metrics import PerformanceMetrics
from transcription import StreamingTranscriptionModule
from vad_detector import BargeInDetector


LOGGER = logging.getLogger(__name__)


class FullDuplexOrchestrator:
    """Coordinates microphone, VAD, ASR, LLM, TTS, playback, and barge-in."""

    def __init__(
        self,
        llm_model: str = "llama3.2:3b",
        whisper_model: str = "base",
        whisper_device: str | None = None,
        tts_voice: str = "en-US-ChristopherNeural",
        tts_backend: str = "auto",
        input_device_index: int | None = None,
        output_device_index: int | None = None,
        output_rate: int | None = None,
        vad_debug: bool = False,
        event_sink: Callable[[str, str, str | None], None] | None = None,
        metrics: PerformanceMetrics | None = None,
    ) -> None:
        self.running = threading.Event()
        self.agent_speaking = threading.Event()
        self.agent_playing = threading.Event()
        self.user_speaking = threading.Event()
        self.barge_in_event = threading.Event()
        self._barge_in_handled = threading.Event()
        self._state_lock = threading.RLock()

        self._spoken_text_lock = threading.RLock()
        self._physically_spoken_text = ""
        self._active_llm_thread: Optional[threading.Thread] = None
        self._active_asr_thread: Optional[threading.Thread] = None
        self._event_sink = event_sink
        self.metrics = metrics or PerformanceMetrics(
            audio_window_ms=32.0,
            audio_hop_ms=16.0,
            barge_in_frames=4,
            sample_rate=16000,
            model_name=llm_model,
            asr_model=whisper_model,
            tts_backend=tts_backend,
        )

        self.audio = AudioStreamingEngine(
            input_device_index=input_device_index,
            output_device_index=output_device_index,
            output_rate=output_rate,
            playback_progress_callback=self._on_playback_progress,
        )
        self.vad = BargeInDetector(sample_rate=self.audio.rate)
        self.asr = StreamingTranscriptionModule(
            model_size=whisper_model,
            compute_type="int8",
            device=whisper_device,
        )
        self.memory = DialogueMemoryManager()
        self.nlp = LightweightNLPLayer()
        self.llm_tts = LLMTTSPipeline(
            audio_engine=self.audio,
            memory_manager=self.memory,
            cancel_event=self.barge_in_event,
            agent_done_callback=self._on_agent_done,
            model=llm_model,
            voice=tts_voice,
            tts_backend=tts_backend,
            event_sink=event_sink,
            metrics=self.metrics,
            transcriber=self.asr,
        )

        self._processing_thread: Optional[threading.Thread] = None
        self._window = bytearray()
        self._pre_roll: Deque[bytes] = deque(maxlen=10)
        self._speech_buffer = bytearray()
        self._silence_frames = 0
        self.silence_release_frames = 8
        self.end_of_turn_silence_frames = 30
        self.min_turn_bytes = int(self.audio.rate * 0.25) * self.audio.sample_width
        self.vad_debug = vad_debug
        self._debug_frame_count = 0

    def start(self) -> None:
        if self.running.is_set():
            return
        self.running.set()
        self.audio.start_streams()
        self._processing_thread = threading.Thread(
            target=self._audio_processing_loop,
            name="audio-processing",
            daemon=True,
        )
        self._processing_thread.start()
        LOGGER.info("System ready. Speak into the microphone.")
        self._emit("status", "Listening")
        self._wait_until_stopped()

    def start_background(self) -> None:
        if self.running.is_set():
            return
        thread = threading.Thread(target=self.start, name="orchestrator", daemon=True)
        thread.start()

    def stop(self) -> None:
        if not self.running.is_set():
            return
        LOGGER.info("Shutting down full-duplex orchestrator.")
        self._emit("status", "Stopping")
        self.running.clear()
        self.barge_in_event.set()
        self.audio.stop_streams()
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.5)
        if self._active_llm_thread and self._active_llm_thread.is_alive():
            self._active_llm_thread.join(timeout=2.0)

    def _wait_until_stopped(self) -> None:
        try:
            while self.running.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.stop()

    def _audio_processing_loop(self) -> None:
        while self.running.is_set():
            try:
                hop = self.audio.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            for frame in self._emit_overlapped_frames(hop):
                self._handle_analysis_frame(frame)

    def _emit_overlapped_frames(self, hop: bytes):
        self._window.extend(hop)
        needed = self.audio.frame_size * self.audio.sample_width
        step = self.audio.hop_size * self.audio.sample_width
        while len(self._window) >= needed:
            frame = bytes(self._window[:needed])
            del self._window[:step]
            yield frame

    def _handle_analysis_frame(self, frame: bytes) -> None:
        self._pre_roll.append(frame)
        try:
            barge_in = self.vad.process_frame(frame, self.agent_playing.is_set())
        except Exception:
            LOGGER.exception("VAD failed for input frame.")
            return
        self._log_vad_debug(frame)

        if barge_in and not self._barge_in_handled.is_set():
            self._handle_barge_in()

        speaking = self.vad.mark_silence_if_needed(self.silence_release_frames)
        if speaking:
            self._on_user_voice_frame(frame)
        elif self.user_speaking.is_set():
            self._on_user_silence_frame(frame)

    def _on_user_voice_frame(self, frame: bytes) -> None:
        if not self.user_speaking.is_set():
            self.user_speaking.set()
            self._silence_frames = 0
            self._speech_buffer = bytearray()
            for pre in list(self._pre_roll):
                self._speech_buffer.extend(pre)
            LOGGER.info("User speech started.")
            self.metrics.start_turn()
            self._emit("status", "Listening to you")
        self._speech_buffer.extend(frame)
        self._silence_frames = 0

    def _on_user_silence_frame(self, frame: bytes) -> None:
        self._speech_buffer.extend(frame)
        self._silence_frames += 1
        if self._silence_frames < self.end_of_turn_silence_frames:
            return
        audio = bytes(self._speech_buffer)
        self._speech_buffer = bytearray()
        self._silence_frames = 0
        self.user_speaking.clear()
        if len(audio) < self.min_turn_bytes:
            return
        self._start_transcription(audio)

    def _start_transcription(self, audio: bytes) -> None:
        if self._active_asr_thread and self._active_asr_thread.is_alive():
            LOGGER.warning("ASR is still busy; dropping overlapping utterance.")
            self._emit("warning", "Skipped overlapping speech")
            self.metrics.mark_dropped_utterance()
            return
        self._emit("status", "Transcribing")
        self.metrics.mark_speech_end()
        self._active_asr_thread = threading.Thread(
            target=self._transcribe_and_respond,
            args=(audio,),
            name="asr",
            daemon=True,
        )
        self._active_asr_thread.start()

    def _transcribe_and_respond(self, audio: bytes) -> None:
        try:
            transcript = self.asr.transcribe_buffer(audio, sample_rate=self.audio.rate)
        except Exception:
            LOGGER.exception("ASR transcription failed.")
            return
        if not transcript:
            LOGGER.info("Ignored empty transcription.")
            self._emit("status", "Listening")
            return
        LOGGER.info("User: %s", transcript)
        semantic_frame = self.nlp.analyze(transcript)
        LOGGER.info("NLP frame intent=%s entities=%s", semantic_frame.intent, semantic_frame.entities)
        self.metrics.mark_transcript(transcript)
        self._emit("message", transcript, "user")
        self.memory.add_user_message(semantic_frame.rewritten_text, nlp_hint=semantic_frame.prompt_hint)
        self._emit("status", f"Intent: {semantic_frame.intent}")
        self._start_agent_response()

    def _start_agent_response(self) -> None:
        with self._state_lock:
            if self._active_llm_thread and self._active_llm_thread.is_alive():
                self.barge_in_event.set()
                self.audio.flush_output()
                self._active_llm_thread.join(timeout=0.5)
            self.barge_in_event.clear()
            self._barge_in_handled.clear()
            self.agent_speaking.set()
            self.agent_playing.clear()
            self._emit("status", "Thinking")
            with self._spoken_text_lock:
                self._physically_spoken_text = ""
            self._active_llm_thread = threading.Thread(
                target=self.llm_tts.generate_and_speak,
                name="llm-tts",
                daemon=True,
            )
            self._active_llm_thread.start()

    def _handle_barge_in(self) -> None:
        with self._state_lock:
            self._barge_in_handled.set()
            self.barge_in_event.set()
            self.audio.flush_output()
            with self._spoken_text_lock:
                spoken = self._physically_spoken_text
                self._physically_spoken_text = ""
            self.memory.truncate_active_agent_response(spoken)
            self.agent_speaking.clear()
            self.agent_playing.clear()
            self.vad.reset_states()
            LOGGER.warning("BARGE-IN detected. Playback cancelled and memory truncated.")
            self.metrics.mark_barge_in()
            self._emit("barge_in", "Interrupted")

    def _on_playback_progress(self, text_delta: str) -> None:
        self.agent_playing.set()
        self.metrics.mark_playback_start()
        with self._spoken_text_lock:
            self._physically_spoken_text += text_delta

    def _on_agent_done(self, interrupted: bool) -> None:
        with self._state_lock:
            if not interrupted and not self.barge_in_event.is_set():
                self.memory.finalize_agent_message()
            self.agent_speaking.clear()
            self.agent_playing.clear()
            self.metrics.mark_agent_done(interrupted)
            self._emit("status", "Listening")
            with self._spoken_text_lock:
                self._physically_spoken_text = ""

    def _emit(self, kind: str, message: str, role: str | None = None) -> None:
        if self._event_sink:
            self._event_sink(kind, message, role)

    def _log_vad_debug(self, frame: bytes) -> None:
        if not self.vad_debug:
            return
        self._debug_frame_count += 1
        if self._debug_frame_count % 10 != 0:
            return
        rms = audioop.rms(frame, self.audio.sample_width)
        dbfs = 20 * math.log10(max(rms, 1) / 32768.0)
        LOGGER.info(
            "mic rms=%s dbfs=%.1f vad_prob=%.3f voice_frames=%s user_speaking=%s",
            rms,
            dbfs,
            self.vad.last_probability,
            self.vad.consecutive_voice_frames,
            self.vad.is_user_speaking,
        )


def install_signal_handlers(orchestrator: FullDuplexOrchestrator) -> None:
    def _handler(signum, frame):
        orchestrator.stop()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
