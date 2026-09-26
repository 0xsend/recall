"""Per-session index timing and performance metrics for watch-mode daemon."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IndexEvent:
    """Record of a single session index operation in watch mode."""

    path: str  # resolved file path
    source: str  # parser source value, e.g. "claude_code"
    success: bool
    duration: float  # seconds (monotonic delta)
    timestamp: float  # wall-clock epoch (time.time())
    error: str | None = None


@dataclass
class WatchIndexMetrics:
    """Mutable, thread-safe accumulator for watch-mode index timing.

    Follows the EmbedPhaseState pattern: mutable dataclass with a record()
    method, exposed via RPC status overlay in the daemon.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _events: deque[IndexEvent] = field(default_factory=lambda: deque(maxlen=50), repr=False)

    # Aggregates since daemon start
    total_indexed: int = 0
    total_failed: int = 0
    total_duration: float = 0.0  # sum of successful durations
    min_duration: float | None = None
    max_duration: float = 0.0
    last_event: IndexEvent | None = field(default=None, repr=False)

    def record(self, event: IndexEvent) -> None:
        """Record a completed index event. Thread-safe."""
        with self._lock:
            self._events.append(event)
            self.last_event = event
            if event.success:
                self.total_indexed += 1
                self.total_duration += event.duration
                if self.min_duration is None or event.duration < self.min_duration:
                    self.min_duration = event.duration
                if event.duration > self.max_duration:
                    self.max_duration = event.duration
            else:
                self.total_failed += 1

    @property
    def avg_duration(self) -> float | None:
        """Average duration of successful index operations."""
        with self._lock:
            if self.total_indexed == 0:
                return None
            return self.total_duration / self.total_indexed

    def recent_events(self) -> list[IndexEvent]:
        """Return a snapshot of recent events (oldest first)."""
        with self._lock:
            return list(self._events)

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable summary dict for RPC/status overlay."""
        with self._lock:
            avg = self.total_duration / self.total_indexed if self.total_indexed > 0 else None
            last = self.last_event
            return {
                "watch_total_indexed": self.total_indexed,
                "watch_total_failed": self.total_failed,
                "watch_avg_duration": avg,
                "watch_min_duration": self.min_duration,
                "watch_max_duration": (self.max_duration if self.total_indexed > 0 else None),
                "watch_last_event_at": last.timestamp if last else None,
                "watch_last_event_duration": (last.duration if last and last.success else None),
            }

    def record_index(
        self,
        path: str,
        source: str,
        *,
        success: bool,
        duration: float,
        error: str | None = None,
    ) -> IndexEvent:
        """Convenience: build an IndexEvent and record it. Returns the event."""
        event = IndexEvent(
            path=path,
            source=source,
            success=success,
            duration=duration,
            timestamp=time.time(),
            error=error,
        )
        self.record(event)
        return event
