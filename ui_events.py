from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class UIEvent:
    id: int
    timestamp: float
    kind: str
    message: str
    role: str | None = None


class UIEventBroker:
    """Small thread-safe event hub for the Flask dashboard."""

    def __init__(self, history_size: int = 80) -> None:
        self._history: deque[UIEvent] = deque(maxlen=history_size)
        self._subscribers: list[queue.Queue[UIEvent]] = []
        self._lock = threading.RLock()
        self._next_id = 1

    def publish(self, kind: str, message: str, role: str | None = None) -> None:
        clean_message = " ".join(message.strip().split())
        if not clean_message:
            return
        with self._lock:
            event = UIEvent(
                id=self._next_id,
                timestamp=time.time(),
                kind=kind,
                message=clean_message,
                role=role,
            )
            self._next_id += 1
            self._history.append(event)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                pass

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [asdict(event) for event in self._history]

    def subscribe(self) -> queue.Queue[UIEvent]:
        subscriber: queue.Queue[UIEvent] = queue.Queue(maxsize=120)
        with self._lock:
            self._subscribers.append(subscriber)
            history = list(self._history)
        for event in history:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                break
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[UIEvent]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)


def encode_sse(event: UIEvent) -> str:
    return f"id: {event.id}\nevent: {event.kind}\ndata: {json.dumps(asdict(event))}\n\n"
