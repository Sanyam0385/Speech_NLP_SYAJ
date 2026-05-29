from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
import tempfile
import wave
from pathlib import Path
from typing import Callable, Iterable, List

import aiohttp
import edge_tts
import ollama
from ollama import RequestError
from pydub import AudioSegment

from audio_streaming import AudioStreamingEngine, PlaybackChunk
from dialogue_memory import DialogueMemoryManager
from performance_metrics import PerformanceMetrics


LOGGER = logging.getLogger(__name__)


class LLMTTSPipeline:
    """
    Streams LLM text into TTS as soon as phrase-level token groups are complete.

    edge-tts emits compressed audio chunks, so each phrase is synthesized as soon
    as available, converted to 16 kHz PCM, and sent to the playback queue in small
    timed chunks carrying proportional text deltas for interruption memory.
    """

    def __init__(
        self,
        audio_engine: AudioStreamingEngine,
        memory_manager: DialogueMemoryManager,
        cancel_event: threading.Event,
        agent_done_callback,
        model: str = "llama3.2:3b",
        voice: str = "en-US-ChristopherNeural",
        tts_backend: str = "auto",
        event_sink: Callable[[str, str, str | None], None] | None = None,
        metrics: PerformanceMetrics | None = None,
        transcriber: object | None = None,
    ) -> None:
        self.audio_engine = audio_engine
        self.memory = memory_manager
        self.cancel_event = cancel_event
        self.agent_done_callback = agent_done_callback
        self.model = model
        self.voice = voice
        self.tts_backend = tts_backend
        self._event_sink = event_sink
        self.metrics = metrics
        self.transcriber = transcriber
        self.max_phrase_chars = 120
        self._bt_pcm_accumulator = bytearray()
        self._bt_text_accumulator: list[str] = []

    def generate_and_speak(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        interrupted = False
        try:
            loop.run_until_complete(self._async_generate_and_speak())
            interrupted = self.cancel_event.is_set()
        except Exception:
            interrupted = self.cancel_event.is_set()
            self.memory.discard_active_agent_message_if_empty()
            LOGGER.exception("LLM/TTS pipeline failed.")
        finally:
            loop.close()
            self.agent_done_callback(interrupted)

    async def _async_generate_and_speak(self) -> None:
        messages = self.memory.get_messages()
        self.memory.begin_agent_message()
        phrase_buffer = ""
        self._bt_pcm_accumulator = bytearray()
        self._bt_text_accumulator = []

        try:
            LOGGER.info("Prompting Ollama model=%s", self.model)
            if self.metrics:
                self.metrics.mark_llm_prompt()
            stream = ollama.chat(model=self.model, messages=messages, stream=True)
        except RequestError:
            self.memory.discard_active_agent_message_if_empty()
            raise
        except Exception as exc:
            self.memory.discard_active_agent_message_if_empty()
            raise RuntimeError(
                "Could not reach Ollama. Start Ollama and pull a compatible model, "
                f"for example: ollama pull {self.model}"
            ) from exc

        saw_token = False
        for chunk in stream:
            if self.cancel_event.is_set():
                return
            token = chunk.get("message", {}).get("content", "")
            if not token:
                continue
            # record token timestamp for streaming LLM analysis
            if self.metrics:
                try:
                    self.metrics.mark_llm_token()
                except Exception:
                    LOGGER.debug("Failed to mark llm token timestamp", exc_info=True)
            if not saw_token:
                LOGGER.info("Ollama started streaming tokens.")
                if self.metrics:
                    self.metrics.mark_llm_first_token()
                self._emit("status", "Replying")
                saw_token = True
            self.memory.append_agent_generated_text(token)
            phrase_buffer += token
            phrases = self._extract_speakable_phrases(phrase_buffer)
            if phrases:
                if not self._is_complete_phrase(phrases[-1]):
                    phrase_buffer = phrases[-1]
                    speak_now = phrases[:-1]
                else:
                    phrase_buffer = ""
                    speak_now = phrases
                for phrase in speak_now:
                    await self._synthesize_and_queue(phrase.strip())

        if phrase_buffer.strip() and not self.cancel_event.is_set():
            await self._synthesize_and_queue(phrase_buffer.strip())
        self._flush_back_transcription_sample()
        if not self.cancel_event.is_set():
            self.memory.finalize_agent_message()

    async def _synthesize_and_queue(self, text: str) -> None:
        if not text or self.cancel_event.is_set():
            return
        LOGGER.info("TTS phrase: %s", text)
        if self.metrics:
            self.metrics.mark_tts_phrase(text)
        self._emit("message", text, "assistant")
        if self.tts_backend == "pyttsx3":
            await asyncio.to_thread(self._synthesize_pyttsx3_and_queue, text)
            return
        try:
            await self._synthesize_edge_and_queue(text)
        except aiohttp.ClientResponseError as exc:
            if self.metrics:
                self.metrics.mark_tts_failure()
            if self.tts_backend != "auto":
                raise
            LOGGER.warning("edge-tts failed with HTTP %s; falling back to offline pyttsx3.", exc.status)
            await asyncio.to_thread(self._synthesize_pyttsx3_and_queue, text)
        except aiohttp.WSServerHandshakeError as exc:
            if self.metrics:
                self.metrics.mark_tts_failure()
            if self.tts_backend != "auto":
                raise
            LOGGER.warning("edge-tts websocket failed with HTTP %s; falling back to offline pyttsx3.", exc.status)
            await asyncio.to_thread(self._synthesize_pyttsx3_and_queue, text)

    async def _synthesize_edge_and_queue(self, text: str) -> None:
        audio_data = bytearray()
        communicate = edge_tts.Communicate(text, self.voice)
        async for chunk in communicate.stream():
            if self.cancel_event.is_set():
                return
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        if not audio_data or self.cancel_event.is_set():
            return
        segment = AudioSegment.from_file(io.BytesIO(bytes(audio_data)), format="mp3")
        segment = segment.set_frame_rate(self.audio_engine.output_rate).set_channels(1).set_sample_width(2)
        pcm = segment.raw_data
        self._collect_back_transcription_sample(text, pcm)
        for playback_chunk in self._build_playback_chunks(pcm, text):
            if self.cancel_event.is_set():
                return
            self.audio_engine.enqueue_playback(playback_chunk)

    def _synthesize_pyttsx3_and_queue(self, text: str) -> None:
        if self.cancel_event.is_set():
            return
        try:
            import pyttsx3
        except ImportError as exc:
            if self.metrics:
                self.metrics.mark_tts_failure()
            raise RuntimeError("pyttsx3 is required for offline TTS fallback. Run: pip install pyttsx3") from exc

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            engine = pyttsx3.init()
            engine.save_to_file(text, str(temp_path))
            engine.runAndWait()
            engine.stop()
            if self.cancel_event.is_set():
                return
            pcm = self._load_wav_as_output_pcm(temp_path)
            self._collect_back_transcription_sample(text, pcm)
            for playback_chunk in self._build_playback_chunks(pcm, text):
                if self.cancel_event.is_set():
                    return
                self.audio_engine.enqueue_playback(playback_chunk)
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    LOGGER.warning("Could not remove temporary TTS file: %s", temp_path)

    def _load_wav_as_output_pcm(self, wav_path: Path) -> bytes:
        with wave.open(str(wav_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            pcm = wav_file.readframes(wav_file.getnframes())
        segment = AudioSegment(
            data=pcm,
            sample_width=sample_width,
            frame_rate=frame_rate,
            channels=channels,
        )
        segment = segment.set_frame_rate(self.audio_engine.output_rate).set_channels(1).set_sample_width(2)
        return segment.raw_data

    def _build_playback_chunks(self, pcm: bytes, text: str) -> Iterable[PlaybackChunk]:
        bytes_per_sample = 2
        samples_per_chunk = 512
        bytes_per_chunk = samples_per_chunk * bytes_per_sample
        total_chunks = max(1, (len(pcm) + bytes_per_chunk - 1) // bytes_per_chunk)
        cursor = 0
        for idx, start in enumerate(range(0, len(pcm), bytes_per_chunk), start=1):
            end = min(len(pcm), start + bytes_per_chunk)
            char_end = round(len(text) * idx / total_chunks)
            text_delta = text[cursor:char_end]
            cursor = char_end
            if idx == total_chunks:
                text_delta += " "
            yield PlaybackChunk(pcm=pcm[start:end], text_delta=text_delta, final=idx == total_chunks)

    def _extract_speakable_phrases(self, text: str) -> List[str]:
        if len(text) >= self.max_phrase_chars and re.search(r"[,;:]\s+", text):
            parts = re.split(r"(?<=[,;:])\s+", text, maxsplit=1)
            return [parts[0], parts[1]]
        if self._is_complete_phrase(text):
            return [text]
        return []

    @staticmethod
    def _is_complete_phrase(text: str) -> bool:
        return bool(re.search(r"[.!?]\s*$", text.strip()))

    def _emit(self, kind: str, message: str, role: str | None = None) -> None:
        if self._event_sink:
            self._event_sink(kind, message, role)

    def _collect_back_transcription_sample(self, text: str, pcm: bytes) -> None:
        if not self.transcriber or not self.metrics or not pcm:
            return
        self._bt_text_accumulator.append(text)
        self._bt_pcm_accumulator.extend(pcm)
        # wait for a minimum duration so Whisper is less likely to return empty output
        bytes_per_second = self.audio_engine.output_rate * 2
        if len(self._bt_pcm_accumulator) < int(bytes_per_second * 0.75):
            return
        self._flush_back_transcription_sample()

    def _flush_back_transcription_sample(self) -> None:
        if not self.transcriber or not self.metrics or not self._bt_pcm_accumulator:
            return
        try:
            expected_text = " ".join(part.strip() for part in self._bt_text_accumulator if part.strip())
            if not expected_text:
                self._bt_pcm_accumulator.clear()
                self._bt_text_accumulator.clear()
                return
            back_text = self.transcriber.transcribe_buffer(
                bytes(self._bt_pcm_accumulator),
                sample_rate=self.audio_engine.output_rate,
            )
            if back_text and back_text.strip():
                wer_result = self.metrics._word_error_rate(expected_text, back_text)
                wer = wer_result[0] if isinstance(wer_result, tuple) else wer_result
                self.metrics.set_back_transcription_wer(None, wer)
        except Exception:
            LOGGER.debug("Back-transcription flush failed", exc_info=True)
        finally:
            self._bt_pcm_accumulator.clear()
            self._bt_text_accumulator.clear()
