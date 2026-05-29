from __future__ import annotations

import csv
import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List


@dataclass
class TurnMetric:
    turn_id: int
    speech_start_ts: float
    speech_end_ts: float | None = None
    transcript_ts: float | None = None
    llm_prompt_ts: float | None = None
    llm_first_token_ts: float | None = None
    # per-token timestamps (perf_counter values) for streaming token timing analysis
    llm_token_timestamps: List[float] = field(default_factory=list)
    # derived LLM streaming metrics (filled when tokens are available)
    llm_token_count: int | None = None
    llm_median_token_latency_ms: float | None = None
    llm_token_throughput_tps: float | None = None
    first_tts_phrase_ts: float | None = None
    playback_start_ts: float | None = None
    agent_done_ts: float | None = None
    barge_in_ts: float | None = None
    transcript: str = ""
    reference_transcript: str = ""
    word_error_rate: float | None = None
    nlp_relevance_score: int | None = None
    nlp_context_score: int | None = None
    barge_in_label: str = "unlabeled"
    first_assistant_phrase: str = ""
    interrupted: bool = False
    # ASR S/I/D breakdown
    asr_substitutions: int | None = None
    asr_insertions: int | None = None
    asr_deletions: int | None = None
    # back-transcription (ASR on TTS output) WER
    back_transcription_wer: float | None = None
    back_transcription_wer_samples: list[float] = field(default_factory=list)
    # playback / audio engine issues
    audio_drop_count: int = 0
    tts_failure_count: int = 0

    def duration_ms(self, start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return round((end - start) * 1000.0, 2)

    def to_row(self) -> dict[str, Any]:
        # compute derived llm token metrics if timestamps available
        if self.llm_token_timestamps:
            deltas = [
                (t2 - t1) for t1, t2 in zip(self.llm_token_timestamps, self.llm_token_timestamps[1:])
            ]
            median_delta = round(statistics.median(deltas) * 1000.0, 2) if deltas else None
            duration = (self.llm_token_timestamps[-1] - self.llm_token_timestamps[0]) if len(self.llm_token_timestamps) > 1 else None
            throughput = round(len(self.llm_token_timestamps) / duration, 2) if duration and duration > 0 else None
            self.llm_token_count = len(self.llm_token_timestamps)
            self.llm_median_token_latency_ms = median_delta
            self.llm_token_throughput_tps = throughput

        return {
            "turn_id": self.turn_id,
            "transcript": self.transcript,
            "reference_transcript": self.reference_transcript,
            "word_error_rate": self.word_error_rate,
            "nlp_relevance_score": self.nlp_relevance_score,
            "nlp_context_score": self.nlp_context_score,
            "barge_in_label": self.barge_in_label,
            "interrupted": self.interrupted,
            "user_speech_duration_ms": self.duration_ms(self.speech_start_ts, self.speech_end_ts),
            "asr_latency_ms": self.duration_ms(self.speech_end_ts, self.transcript_ts),
            "llm_first_token_latency_ms": self.duration_ms(self.llm_prompt_ts, self.llm_first_token_ts),
            "tts_first_phrase_latency_ms": self.duration_ms(self.llm_first_token_ts, self.first_tts_phrase_ts),
            "response_start_latency_ms": self.duration_ms(self.speech_end_ts, self.playback_start_ts),
            "agent_turn_duration_ms": self.duration_ms(self.playback_start_ts, self.agent_done_ts),
            "barge_in_after_playback_ms": self.duration_ms(self.playback_start_ts, self.barge_in_ts),
            "first_assistant_phrase": self.first_assistant_phrase,
            "llm_token_count": self.llm_token_count,
            "llm_median_token_latency_ms": self.llm_median_token_latency_ms,
            "llm_token_throughput_tps": self.llm_token_throughput_tps,
            "asr_substitutions": self.asr_substitutions,
            "asr_insertions": self.asr_insertions,
            "asr_deletions": self.asr_deletions,
            "back_transcription_wer": self.back_transcription_wer,
            "audio_drop_count": self.audio_drop_count,
            "tts_failure_count": self.tts_failure_count,
        }


@dataclass
class PerformanceMetrics:
    """Thread-safe metrics store for report-ready latency and interaction results."""

    audio_window_ms: float = 32.0
    audio_hop_ms: float = 16.0
    barge_in_frames: int = 4
    sample_rate: int = 16000
    model_name: str = ""
    asr_model: str = ""
    tts_backend: str = ""
    started_ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._turns: list[TurnMetric] = []
        self._active_turn_id: int | None = None
        self._barge_in_count = 0
        self._dropped_utterance_count = 0

    @property
    def theoretical_barge_in_latency_ms(self) -> float:
        return round(self.barge_in_frames * self.audio_hop_ms, 2)

    def start_turn(self) -> int:
        with self._lock:
            turn = TurnMetric(turn_id=len(self._turns) + 1, speech_start_ts=time.perf_counter())
            self._turns.append(turn)
            self._active_turn_id = turn.turn_id
            return turn.turn_id

    def mark_speech_end(self, turn_id: int | None = None) -> None:
        self._update(turn_id, speech_end_ts=time.perf_counter())

    def mark_transcript(self, transcript: str, turn_id: int | None = None) -> None:
        self._update(turn_id, transcript_ts=time.perf_counter(), transcript=transcript)

    def mark_llm_prompt(self, turn_id: int | None = None) -> None:
        self._update(turn_id, llm_prompt_ts=time.perf_counter())

    def mark_llm_first_token(self, turn_id: int | None = None) -> None:
        turn = self._get_turn(turn_id)
        if turn and turn.llm_first_token_ts is None:
            turn.llm_first_token_ts = time.perf_counter()

    def mark_llm_token(self, turn_id: int | None = None, timestamp: float | None = None) -> None:
        """Record a single LLM token timestamp for streaming analysis."""
        turn = self._get_turn(turn_id)
        if not turn:
            return
        ts = timestamp or time.perf_counter()
        turn.llm_token_timestamps.append(ts)

    def mark_tts_phrase(self, phrase: str, turn_id: int | None = None) -> None:
        turn = self._get_turn(turn_id)
        if not turn:
            return
        now = time.perf_counter()
        if turn.first_tts_phrase_ts is None:
            turn.first_tts_phrase_ts = now
            turn.first_assistant_phrase = phrase

    def mark_playback_start(self, turn_id: int | None = None) -> None:
        turn = self._get_turn(turn_id)
        if turn and turn.playback_start_ts is None:
            turn.playback_start_ts = time.perf_counter()

    def mark_agent_done(self, interrupted: bool, turn_id: int | None = None) -> None:
        turn = self._get_turn(turn_id)
        if not turn:
            return
        turn.agent_done_ts = time.perf_counter()
        turn.interrupted = interrupted

    def mark_barge_in(self, turn_id: int | None = None) -> None:
        turn = self._get_turn(turn_id)
        with self._lock:
            self._barge_in_count += 1
        if turn:
            turn.barge_in_ts = time.perf_counter()
            turn.interrupted = True

    def mark_dropped_utterance(self) -> None:
        with self._lock:
            self._dropped_utterance_count += 1

    def increment_audio_drop(self, turn_id: int | None = None, count: int = 1) -> None:
        turn = self._get_turn(turn_id)
        with self._lock:
            if turn:
                turn.audio_drop_count = (turn.audio_drop_count or 0) + count
            else:
                self._dropped_utterance_count += count

    def mark_tts_failure(self, turn_id: int | None = None) -> None:
        turn = self._get_turn(turn_id)
        if turn:
            turn.tts_failure_count = (turn.tts_failure_count or 0) + 1

    def set_asr_counts(self, turn_id: int | None, substitutions: int | None, insertions: int | None, deletions: int | None) -> None:
        turn = self._get_turn(turn_id)
        if not turn:
            return
        turn.asr_substitutions = substitutions
        turn.asr_insertions = insertions
        turn.asr_deletions = deletions

    def set_back_transcription_wer(self, turn_id: int | None, wer: float | None) -> None:
        turn = self._get_turn(turn_id)
        if not turn:
            return
        if wer is None:
            return
        try:
            sample = float(wer)
        except (TypeError, ValueError):
            return
        turn.back_transcription_wer_samples.append(sample)
        turn.back_transcription_wer = round(statistics.mean(turn.back_transcription_wer_samples), 3)

    def update_turn_evaluation(
        self,
        turn_id: int,
        reference_transcript: str | None = None,
        nlp_relevance_score: int | None = None,
        nlp_context_score: int | None = None,
        barge_in_label: str | None = None,
    ) -> dict[str, Any] | None:
        turn = self._get_turn(turn_id)
        if not turn:
            return None
        if reference_transcript is not None:
            turn.reference_transcript = reference_transcript.strip()
            wer_result = self._word_error_rate(turn.reference_transcript, turn.transcript)
            # _word_error_rate now returns (wer, S, I, D)
            if isinstance(wer_result, tuple):
                wer, S, I, D = wer_result
                turn.word_error_rate = wer
                # store S/I/D counts on the turn
                self.set_asr_counts(turn.turn_id, substitutions=S, insertions=I, deletions=D)
            else:
                turn.word_error_rate = wer_result
        if nlp_relevance_score is not None:
            turn.nlp_relevance_score = self._clamp_score(nlp_relevance_score)
        if nlp_context_score is not None:
            turn.nlp_context_score = self._clamp_score(nlp_context_score)
        if barge_in_label in {"unlabeled", "intentional", "false_positive"}:
            turn.barge_in_label = barge_in_label
        return turn.to_row()

    def mark_all_interruptions_intentional(self) -> None:
        with self._lock:
            for turn in self._turns:
                if turn.interrupted:
                    turn.barge_in_label = "intentional"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = [turn.to_row() for turn in self._turns]
            return {
                "session": {
                    "elapsed_seconds": round(time.time() - self.started_ts, 2),
                    "turns": len(rows),
                    "barge_ins": self._barge_in_count,
                    "dropped_utterances": self._dropped_utterance_count,
                    "sample_rate": self.sample_rate,
                    "window_ms": self.audio_window_ms,
                    "hop_ms": self.audio_hop_ms,
                    "barge_in_frames": self.barge_in_frames,
                    "theoretical_barge_in_latency_ms": self.theoretical_barge_in_latency_ms,
                    "llm_model": self.model_name,
                    "asr_model": self.asr_model,
                    "tts_backend": self.tts_backend,
                },
                "summary": self._summary(rows),
                "report": self._report_summary(rows),
                "turns": rows,
            }

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        return target

    def export_csv(self, path: str | Path) -> Path:
        target = Path(path)
        rows = self.snapshot()["turns"]
        fieldnames = list(rows[0].keys()) if rows else [
            "turn_id",
            "transcript",
            "interrupted",
            "user_speech_duration_ms",
            "asr_latency_ms",
            "llm_first_token_latency_ms",
            "tts_first_phrase_latency_ms",
            "response_start_latency_ms",
            "agent_turn_duration_ms",
            "barge_in_after_playback_ms",
            "first_assistant_phrase",
        ]
        with target.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return target

    def _summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        # collect barge-in latencies for percentile calculations
        barge_latencies = [row.get("barge_in_after_playback_ms") for row in rows if isinstance(row.get("barge_in_after_playback_ms"), (int, float))]
        return {
            "avg_asr_latency_ms": self._mean(rows, "asr_latency_ms"),
            "avg_llm_first_token_latency_ms": self._mean(rows, "llm_first_token_latency_ms"),
            "avg_tts_first_phrase_latency_ms": self._mean(rows, "tts_first_phrase_latency_ms"),
            "avg_response_start_latency_ms": self._mean(rows, "response_start_latency_ms"),
            "avg_barge_in_after_playback_ms": self._mean(rows, "barge_in_after_playback_ms"),
            "barge_in_p50_ms": self._percentile_from_list(barge_latencies, 50),
            "barge_in_p90_ms": self._percentile_from_list(barge_latencies, 90),
            "barge_in_p99_ms": self._percentile_from_list(barge_latencies, 99),
            "interruption_rate": self._rate(rows, "interrupted"),
            "avg_word_error_rate": self._mean(rows, "word_error_rate"),
            "avg_nlp_relevance_score": self._mean(rows, "nlp_relevance_score"),
            "avg_nlp_context_score": self._mean(rows, "nlp_context_score"),
            "intentional_barge_in_success_rate": self._intentional_barge_rate(rows),
            "false_barge_in_rate": self._false_barge_rate(rows),
            # LLM streaming metrics
            "avg_llm_token_throughput_tps": self._mean(rows, "llm_token_throughput_tps"),
            "avg_llm_median_token_latency_ms": self._mean(rows, "llm_median_token_latency_ms"),
            "avg_llm_token_count": self._mean(rows, "llm_token_count"),
            # ASR breakdowns & TTS/Audio issues
            "total_asr_substitutions": self._sum(rows, "asr_substitutions"),
            "total_asr_insertions": self._sum(rows, "asr_insertions"),
            "total_asr_deletions": self._sum(rows, "asr_deletions"),
            "avg_back_transcription_wer": self._mean(rows, "back_transcription_wer"),
            "total_audio_drop_count": self._sum(rows, "audio_drop_count"),
            "tts_failure_rate": self._rate(rows, "tts_failure_count"),
        }

    def _report_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        summary = self._summary(rows)
        return {
            "speech_layer": {
                "asr_latency_ms": summary["avg_asr_latency_ms"],
                "word_error_rate": summary["avg_word_error_rate"],
                "sample_rate": self.sample_rate,
                "window_ms": self.audio_window_ms,
                "hop_ms": self.audio_hop_ms,
            },
            "nlp_layer": {
                "llm_first_token_latency_ms": summary["avg_llm_first_token_latency_ms"],
                "relevance_score": summary["avg_nlp_relevance_score"],
                "context_score": summary["avg_nlp_context_score"],
                "median_token_latency_ms": summary.get("avg_llm_median_token_latency_ms"),
                "token_throughput_tps": summary.get("avg_llm_token_throughput_tps"),
                "model": self.model_name,
            },
            "tts_layer": {
                "tts_first_phrase_latency_ms": summary["avg_tts_first_phrase_latency_ms"],
                "backend": self.tts_backend,
            },
            "full_duplex_layer": {
                "response_start_latency_ms": summary["avg_response_start_latency_ms"],
                "theoretical_barge_in_latency_ms": self.theoretical_barge_in_latency_ms,
                "intentional_barge_in_success_rate": summary["intentional_barge_in_success_rate"],
                "false_barge_in_rate": summary["false_barge_in_rate"],
                "barge_in_latency_p50_ms": summary.get("barge_in_p50_ms"),
                "barge_in_latency_p90_ms": summary.get("barge_in_p90_ms"),
                "barge_in_latency_p99_ms": summary.get("barge_in_p99_ms"),
            },
        }

    def _percentile_from_list(self, values: list[float], percentile: int) -> float | None:
        vals = [v for v in values if isinstance(v, (int, float))]
        if not vals:
            return None
        vals.sort()
        k = (len(vals) - 1) * (percentile / 100.0)
        f = int(k)
        c = min(f + 1, len(vals) - 1)
        if f == c:
            return round(vals[int(k)], 2)
        d0 = vals[f] * (c - k)
        d1 = vals[c] * (k - f)
        return round(d0 + d1, 2)

    def _sum(self, rows: list[dict[str, Any]], key: str) -> int:
        return sum(int(row.get(key) or 0) for row in rows)

    def _mean(self, rows: list[dict[str, Any]], key: str) -> float | None:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            return None
        return round(statistics.mean(values), 2)

    def _rate(self, rows: list[dict[str, Any]], key: str) -> float | None:
        if not rows:
            return None
        return round(sum(1 for row in rows if row.get(key)) / len(rows), 3)

    def _intentional_barge_rate(self, rows: list[dict[str, Any]]) -> float | None:
        intentional = [row for row in rows if row.get("barge_in_label") == "intentional"]
        if not intentional:
            return None
        return round(sum(1 for row in intentional if row.get("interrupted")) / len(intentional), 3)

    def _false_barge_rate(self, rows: list[dict[str, Any]]) -> float | None:
        completed_without_intent = [
            row for row in rows if row.get("barge_in_label") != "intentional"
        ]
        if not completed_without_intent:
            return None
        false_count = sum(1 for row in completed_without_intent if row.get("barge_in_label") == "false_positive")
        return round(false_count / len(completed_without_intent), 3)

    def _word_error_rate(self, reference: str, hypothesis: str) -> float | None:
        ref = reference.lower().split()
        hyp = hypothesis.lower().split()
        if not ref:
            return None
        rows = len(ref) + 1
        cols = len(hyp) + 1
        distances = [[0] * cols for _ in range(rows)]
        ops = [[None] * cols for _ in range(rows)]
        for i in range(rows):
            distances[i][0] = i
            ops[i][0] = "D" if i > 0 else None
        for j in range(cols):
            distances[0][j] = j
            ops[0][j] = "I" if j > 0 else None
        for i in range(1, rows):
            for j in range(1, cols):
                if ref[i - 1] == hyp[j - 1]:
                    distances[i][j] = distances[i - 1][j - 1]
                    ops[i][j] = "M"
                else:
                    # substitution
                    sub_cost = distances[i - 1][j - 1] + 1
                    # deletion (ref word deleted)
                    del_cost = distances[i - 1][j] + 1
                    # insertion (extra hyp word)
                    ins_cost = distances[i][j - 1] + 1
                    best = min(sub_cost, del_cost, ins_cost)
                    distances[i][j] = best
                    if best == sub_cost:
                        ops[i][j] = "S"
                    elif best == del_cost:
                        ops[i][j] = "D"
                    else:
                        ops[i][j] = "I"
        # backtrace to count S/I/D
        i = rows - 1
        j = cols - 1
        S = I = D = 0
        while i > 0 or j > 0:
            op = ops[i][j]
            if op == "M":
                i -= 1
                j -= 1
            elif op == "S":
                S += 1
                i -= 1
                j -= 1
            elif op == "D":
                D += 1
                i -= 1
            elif op == "I":
                I += 1
                j -= 1
            else:
                # fallback safety
                break
        wer = round((S + I + D) / len(ref), 3)
        return wer, S, I, D

    def _clamp_score(self, score: int) -> int:
        return max(1, min(5, int(score)))

    def _update(self, turn_id: int | None = None, **updates: Any) -> None:
        turn = self._get_turn(turn_id)
        if not turn:
            return
        for key, value in updates.items():
            setattr(turn, key, value)

    def _get_turn(self, turn_id: int | None = None) -> TurnMetric | None:
        with self._lock:
            resolved_id = turn_id or self._active_turn_id
            if resolved_id is None:
                return None
            for turn in reversed(self._turns):
                if turn.turn_id == resolved_id:
                    return turn
            return None
