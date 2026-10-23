"""Request-local search timing shared by the event loop and cursor worker."""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass

from recall.core.rpc_types import SearchPhase


@dataclass(frozen=True)
class SearchTimeoutSnapshot:
    phase: SearchPhase
    phase_seconds: tuple[float, ...]
    elapsed_ms: int
    captured_at: float

    @property
    def largest_phase(self) -> SearchPhase:
        return max(
            SearchPhase, key=lambda phase: self.phase_seconds[list(SearchPhase).index(phase)]
        )

    def details(self, deadline: float, cleanup_ms: int) -> dict[str, object]:
        return {
            "kind": "search_timeout",
            "phase": self.phase.value,
            "largest_phase": self.largest_phase.value,
            "deadline_ms": int(deadline * 1000),
            "elapsed_ms": self.elapsed_ms,
            "cleanup_ms": cleanup_ms,
            "phase_ms": {
                phase.value: int(seconds * 1000)
                for phase, seconds in zip(SearchPhase, self.phase_seconds, strict=True)
            },
        }


class SearchRequest:
    def __init__(self, request_id: object, deadline: float | None) -> None:
        self.request_id = request_id
        self.deadline = deadline
        self.started_at = time.monotonic()
        self._entered_at = self.started_at
        self._phase = SearchPhase.RESOLVE_MODE
        self._seconds = dict.fromkeys(SearchPhase, 0.0)
        self._lock = threading.Lock()
        self._snapshot: SearchTimeoutSnapshot | None = None

    def _enter_locked(self, phase: SearchPhase) -> None:
        if self._snapshot is not None:
            return
        now = time.monotonic()
        self._seconds[self._phase] += now - self._entered_at
        self._entered_at = now
        self._phase = phase

    def enter(self, phase: SearchPhase) -> None:
        with self._lock:
            self._enter_locked(phase)

    def cursor_opened(self) -> None:
        # Mode resolution owns its entire initial read. Only search reads
        # transition here, at actual worker admission rather than a loop callback.
        with self._lock:
            if self._phase is SearchPhase.READ_QUEUE:
                self._enter_locked(SearchPhase.SCAN)

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)

    def freeze(self) -> SearchTimeoutSnapshot:
        with self._lock:
            if self._snapshot is None:
                now = time.monotonic()
                self._seconds[self._phase] += now - self._entered_at
                self._snapshot = SearchTimeoutSnapshot(
                    self._phase,
                    tuple(self._seconds.values()),
                    int((now - self.started_at) * 1000),
                    now,
                )
            return self._snapshot


# A task context carries the request through validation and executor submission;
# worker code receives the same explicit tracker, never a server-wide phase.
search_request: ContextVar[SearchRequest | None] = ContextVar("search_request", default=None)
