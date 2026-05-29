from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SemanticFrame:
    intent: str
    entities: dict[str, str]
    rewritten_text: str
    prompt_hint: str


class LightweightNLPLayer:
    """Config-driven local NLP layer for command shaping and prompt guidance."""

    INTENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("system_debug", (r"\bwhy\b.*\bnot understand", r"\bfix\b.*\bspeech", r"\bdebug\b")),
        ("explain", (r"\bexplain\b", r"\bwhat is\b", r"\bhow does\b")),
        ("summarize", (r"\bsummarize\b", r"\bshort version\b", r"\bin brief\b")),
        ("code_help", (r"\bcode\b", r"\berror\b", r"\bbug\b", r"\bfunction\b")),
        ("conversation", (r".*",)),
    )

    ENTITY_PATTERNS: tuple[tuple[str, str], ...] = (
        ("model", r"\b(?:llama|mistral|deepseek|whisper|ollama)[\w.:-]*\b"),
        ("device", r"\b(?:mic|microphone|speaker|headphone|earphone|realtek|bluetooth)\b"),
        ("project_area", r"\b(?:speech|transcription|tts|nlp|frontend|vad|barge-?in)\b"),
    )

    DOMAIN_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\bspeech to text\b", re.IGNORECASE), "speech-to-text"),
        (re.compile(r"\btext to speech\b", re.IGNORECASE), "text-to-speech"),
        (re.compile(r"\bfront end\b", re.IGNORECASE), "frontend"),
        (re.compile(r"\bbarge in\b", re.IGNORECASE), "barge-in"),
    )

    def analyze(self, text: str) -> SemanticFrame:
        rewritten = self._rewrite(text)
        intent = self._classify(rewritten)
        entities = self._extract_entities(rewritten)
        hint = self._prompt_hint(intent, entities)
        return SemanticFrame(intent=intent, entities=entities, rewritten_text=rewritten, prompt_hint=hint)

    def _rewrite(self, text: str) -> str:
        result = text.strip()
        for pattern, replacement in self.DOMAIN_REWRITES:
            result = pattern.sub(replacement, result)
        return result

    def _classify(self, text: str) -> str:
        lowered = text.lower()
        for intent, patterns in self.INTENT_PATTERNS:
            if any(re.search(pattern, lowered) for pattern in patterns):
                return intent
        return "conversation"

    def _extract_entities(self, text: str) -> dict[str, str]:
        entities: dict[str, str] = {}
        for name, pattern in self.ENTITY_PATTERNS:
            matches = self._unique(re.findall(pattern, text, flags=re.IGNORECASE))
            if matches:
                entities[name] = ", ".join(matches[:4])
        return entities

    def _prompt_hint(self, intent: str, entities: dict[str, str]) -> str:
        entity_text = "; ".join(f"{key}: {value}" for key, value in entities.items()) or "none"
        return (
            f"Local NLP frame: intent={intent}; entities={entity_text}. "
            "Use this as weak guidance, but answer the user's actual words."
        )

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        seen = set()
        result = []
        for value in values:
            normalized = value.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(value)
        return result
