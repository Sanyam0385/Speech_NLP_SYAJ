from __future__ import annotations

import json
import random
from typing import List, Tuple

from nlp_layer import LightweightNLPLayer
from performance_metrics import PerformanceMetrics


def _mutate_text(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    # Controlled synthetic ASR noise for WER/S/I/D simulation.
    mode = rng.random()
    if mode < 0.35 and len(words) > 3:
        idx = rng.randrange(1, len(words) - 1)
        words[idx] = "thing"
    elif mode < 0.55 and len(words) > 4:
        idx = rng.randrange(1, len(words) - 1)
        del words[idx]
    elif mode < 0.75:
        idx = rng.randrange(len(words))
        words.insert(idx, "please")
    return " ".join(words)


def _build_text_dataset() -> List[Tuple[str, str]]:
    # 1000 total prompts across 5 intents.
    patterns = {
        "system_debug": [
            "why does speech pipeline not understand my words",
            "debug this speech recognition bug",
            "fix speech system issue quickly",
            "why does this app not understand microphone input",
            "debug barge in behavior for this session",
        ],
        "explain": [
            "explain how vad works",
            "what is whisper tiny model",
            "how does nlp layer classify intent",
            "explain text to speech process",
            "what is ollama token streaming",
        ],
        "summarize": [
            "summarize this report in brief",
            "give me short version of metrics",
            "summarize asr and tts performance",
            "in brief tell me current status",
            "short version of pipeline please",
        ],
        "code_help": [
            "there is a bug in function saveEvaluation",
            "code error in web frontend metrics block",
            "help fix this python function",
            "debug this code for barge in",
            "there is an error in asr pipeline",
        ],
        "conversation": [
            "how are you today",
            "thanks for helping me",
            "can we continue testing",
            "i want to run another turn",
            "this project sounds good",
        ],
    }
    dataset: List[Tuple[str, str]] = []
    for intent, samples in patterns.items():
        for i in range(200):
            base = samples[i % len(samples)]
            dataset.append((base, intent))
    return dataset


def run_simulation(seed: int = 11) -> dict:
    rng = random.Random(seed)
    nlp = LightweightNLPLayer()
    metrics = PerformanceMetrics(
        audio_window_ms=32.0,
        audio_hop_ms=16.0,
        barge_in_frames=4,
        sample_rate=16000,
        model_name="llama3.2:3b",
        asr_model="tiny",
        tts_backend="pyttsx3",
    )

    dataset = _build_text_dataset()
    rng.shuffle(dataset)

    nlp_correct = 0
    entity_hits = 0
    rewrite_hits = 0

    for text, expected_intent in dataset:
        turn_id = metrics.start_turn()
        turn = metrics._get_turn(turn_id)
        if not turn:
            continue

        frame = nlp.analyze(text)
        nlp_correct += 1 if frame.intent == expected_intent else 0
        entity_hits += 1 if frame.entities else 0
        rewrite_hits += 1 if frame.rewritten_text != text else 0

        # Simulated latencies (seconds) based on your observed ranges.
        speech_duration = max(0.45, rng.gauss(1.2, 0.35))
        asr_latency = max(0.25, rng.gauss(0.84, 0.2))
        llm_first = max(0.8, rng.gauss(2.9, 0.55))
        tts_first = max(0.06, rng.gauss(0.24, 0.07))
        response_start = asr_latency + llm_first + tts_first + max(0.1, rng.gauss(0.7, 0.2))
        turn_duration = max(0.6, rng.gauss(2.6, 0.9))

        base = turn.speech_start_ts
        turn.speech_end_ts = base + speech_duration
        turn.transcript_ts = turn.speech_end_ts + asr_latency
        turn.llm_prompt_ts = turn.transcript_ts
        turn.llm_first_token_ts = turn.llm_prompt_ts + llm_first
        turn.first_tts_phrase_ts = turn.llm_first_token_ts + tts_first
        turn.playback_start_ts = turn.speech_end_ts + response_start
        turn.agent_done_ts = turn.playback_start_ts + turn_duration

        # Simulated token stream stats.
        token_count = rng.randint(20, 160)
        token_span = max(0.45, rng.gauss(3.0, 0.9))
        start_t = turn.llm_first_token_ts
        turn.llm_token_timestamps = [start_t + (token_span * idx / max(1, token_count - 1)) for idx in range(token_count)]

        # Simulated ASR output and evaluation.
        transcript = _mutate_text(frame.rewritten_text, rng)
        turn.transcript = transcript
        relevance = 5 if frame.intent == expected_intent else 3
        context = 4 if frame.entities else 3
        metrics.update_turn_evaluation(
            turn_id=turn_id,
            reference_transcript=frame.rewritten_text,
            nlp_relevance_score=relevance,
            nlp_context_score=context,
            barge_in_label="unlabeled",
        )

        # Simulated barge-in labels and events.
        p = rng.random()
        if p < 0.18:
            turn.interrupted = True
            turn.barge_in_ts = turn.playback_start_ts + max(0.05, rng.gauss(0.17, 0.05))
            turn.barge_in_label = "intentional"
            metrics._barge_in_count += 1
        elif p < 0.26:
            turn.interrupted = True
            turn.barge_in_ts = turn.playback_start_ts + max(0.04, rng.gauss(0.11, 0.03))
            turn.barge_in_label = "false_positive"
            metrics._barge_in_count += 1

        # Non-zero diagnostics to prove metric paths.
        if rng.random() < 0.07:
            metrics.increment_audio_drop(turn_id, count=1)
        if rng.random() < 0.035:
            metrics.mark_tts_failure(turn_id)

        # Back-transcription synthetic WER sample.
        bt_wer = round(max(0.0, rng.gauss(0.14, 0.06)), 3)
        metrics.set_back_transcription_wer(turn_id, bt_wer)

    snapshot = metrics.snapshot()
    n = len(dataset)
    nlp_eval = {
        "samples": n,
        "intent_accuracy": round(nlp_correct / n, 3),
        "entity_hit_rate": round(entity_hits / n, 3),
        "rewrite_hit_rate": round(rewrite_hits / n, 3),
    }
    return {
        "nlp_evaluation": nlp_eval,
        "session": snapshot["session"],
        "summary": snapshot["summary"],
        "report": snapshot["report"],
    }


def main() -> None:
    result = run_simulation()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
