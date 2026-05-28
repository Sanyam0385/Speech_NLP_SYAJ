from __future__ import annotations

import threading
from typing import Dict, List


class DialogueMemoryManager:
    """Thread-safe multi-turn memory with explicit interruption truncation."""

    def __init__(
        self,
        system_prompt: str = (
            "You are a helpful, conversational AI assistant. Keep responses "
            "brief, natural, and suited to a spoken dialogue."
        ),
    ) -> None:
        self._lock = threading.RLock()
        self._history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        self._active_agent_index: int | None = None

    def add_user_message(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._history.append({"role": "user", "content": text})
            self._active_agent_index = None

    def begin_agent_message(self) -> None:
        with self._lock:
            self._drop_empty_active_agent_locked()
            self._history.append({"role": "assistant", "content": ""})
            self._active_agent_index = len(self._history) - 1

    def append_agent_generated_text(self, text_delta: str) -> None:
        if not text_delta:
            return
        with self._lock:
            if self._active_agent_index is None:
                self.begin_agent_message()
            assert self._active_agent_index is not None
            self._history[self._active_agent_index]["content"] += text_delta

    def finalize_agent_message(self) -> None:
        with self._lock:
            if self._active_agent_index is not None:
                content = self._history[self._active_agent_index]["content"].strip()
                if content:
                    self._history[self._active_agent_index]["content"] = content
                else:
                    self._history.pop(self._active_agent_index)
                self._active_agent_index = None

    def discard_active_agent_message_if_empty(self) -> None:
        with self._lock:
            self._drop_empty_active_agent_locked()

    def truncate_active_agent_response(self, physically_spoken_text: str) -> None:
        spoken = physically_spoken_text.strip()
        with self._lock:
            idx = self._active_agent_index
            if idx is None:
                for candidate in range(len(self._history) - 1, -1, -1):
                    if self._history[candidate]["role"] == "assistant":
                        idx = candidate
                        break
            if idx is None:
                return
            if spoken:
                self._history[idx]["content"] = f"{spoken} [INTERRUPTED]"
            else:
                self._history.pop(idx)
            self._active_agent_index = None

    def get_messages(self) -> List[Dict[str, str]]:
        with self._lock:
            return [
                dict(message)
                for message in self._history
                if message.get("content", "").strip()
            ]

    def _drop_empty_active_agent_locked(self) -> None:
        if self._active_agent_index is None:
            return
        if self._active_agent_index >= len(self._history):
            self._active_agent_index = None
            return
        message = self._history[self._active_agent_index]
        if message["role"] == "assistant" and not message["content"].strip():
            self._history.pop(self._active_agent_index)
        self._active_agent_index = None
