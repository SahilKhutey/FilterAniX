from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, List, Optional


@dataclass
class PipelineEvent:
    """Event emitted during pipeline execution for real-time telemetry."""
    job_id: str
    stage: str
    progress: float = 0.0
    message: str = ""
    frame: int = 0
    total_frames: int = 0
    fps: float = 0.0
    eta_seconds: Optional[float] = None
    substage: str = ""
    timestamp: float = field(default_factory=time.time)
    details: dict = field(default_factory=dict)


class EventBus:
    """Thread-safe event bus for broadcasting pipeline events to telemetry listeners."""

    def __init__(self):
        self._listeners: List[Callable[[PipelineEvent], Any]] = []
        self._lock = Lock()

    def subscribe(self, callback: Callable[[PipelineEvent], Any]) -> None:
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[PipelineEvent], Any]) -> None:
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def emit(self, event: PipelineEvent) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                pass

    def clear(self) -> None:
        with self._lock:
            self._listeners.clear()


# Global default EventBus instance
default_event_bus = EventBus()
