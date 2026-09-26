"""Tests for watch-mode per-session index metrics."""

from __future__ import annotations

import threading
import time

from recall.services.watch_metrics import IndexEvent, WatchIndexMetrics


def _make_event(
    *,
    path: str = "/tmp/session.jsonl",
    source: str = "claude_code",
    success: bool = True,
    duration: float = 0.05,
    error: str | None = None,
) -> IndexEvent:
    return IndexEvent(
        path=path,
        source=source,
        success=success,
        duration=duration,
        timestamp=time.time(),
        error=error,
    )


class TestWatchIndexMetrics:
    def test_empty_metrics(self) -> None:
        m = WatchIndexMetrics()
        assert m.total_indexed == 0
        assert m.total_failed == 0
        assert m.avg_duration is None
        assert m.min_duration is None
        assert m.max_duration == 0.0
        assert m.last_event is None
        assert m.recent_events() == []

    def test_record_success(self) -> None:
        m = WatchIndexMetrics()
        m.record(_make_event(duration=0.1))
        m.record(_make_event(duration=0.3))

        assert m.total_indexed == 2
        assert m.total_failed == 0
        assert m.avg_duration is not None
        assert abs(m.avg_duration - 0.2) < 1e-9
        assert m.min_duration == 0.1
        assert m.max_duration == 0.3

    def test_record_failure(self) -> None:
        m = WatchIndexMetrics()
        m.record(_make_event(success=False, error="parse error"))

        assert m.total_indexed == 0
        assert m.total_failed == 1
        # Failures don't contribute to duration aggregates
        assert m.avg_duration is None
        assert m.min_duration is None

    def test_mixed_success_failure(self) -> None:
        m = WatchIndexMetrics()
        m.record(_make_event(duration=0.1))
        m.record(_make_event(success=False, duration=0.5, error="boom"))
        m.record(_make_event(duration=0.3))

        assert m.total_indexed == 2
        assert m.total_failed == 1
        # avg only counts successes
        assert m.avg_duration is not None
        assert abs(m.avg_duration - 0.2) < 1e-9

    def test_last_event_tracks_most_recent(self) -> None:
        m = WatchIndexMetrics()
        m.record(_make_event(path="/a.jsonl"))
        m.record(_make_event(path="/b.jsonl"))
        assert m.last_event is not None
        assert m.last_event.path == "/b.jsonl"

    def test_recent_events_order(self) -> None:
        m = WatchIndexMetrics()
        for i in range(5):
            m.record(_make_event(path=f"/session_{i}.jsonl"))

        recent = m.recent_events()
        assert len(recent) == 5
        # Oldest first
        assert recent[0].path == "/session_0.jsonl"
        assert recent[4].path == "/session_4.jsonl"

    def test_ring_buffer_eviction(self) -> None:
        m = WatchIndexMetrics()
        for i in range(55):
            m.record(_make_event(path=f"/session_{i}.jsonl"))

        recent = m.recent_events()
        assert len(recent) == 50
        # Oldest 5 should have been evicted
        assert recent[0].path == "/session_5.jsonl"
        assert recent[-1].path == "/session_54.jsonl"
        # Aggregates still reflect all 55
        assert m.total_indexed == 55

    def test_snapshot_empty(self) -> None:
        m = WatchIndexMetrics()
        snap = m.snapshot()
        assert snap["watch_total_indexed"] == 0
        assert snap["watch_total_failed"] == 0
        assert snap["watch_avg_duration"] is None
        assert snap["watch_min_duration"] is None
        assert snap["watch_max_duration"] is None
        assert snap["watch_last_event_at"] is None
        assert snap["watch_last_event_duration"] is None

    def test_snapshot_with_events(self) -> None:
        m = WatchIndexMetrics()
        m.record(_make_event(duration=0.1))
        m.record(_make_event(duration=0.3))

        snap = m.snapshot()
        assert snap["watch_total_indexed"] == 2
        assert snap["watch_avg_duration"] is not None
        assert abs(snap["watch_avg_duration"] - 0.2) < 1e-9
        assert snap["watch_min_duration"] == 0.1
        assert snap["watch_max_duration"] == 0.3
        assert snap["watch_last_event_at"] is not None
        assert snap["watch_last_event_duration"] == 0.3

    def test_snapshot_last_event_failure_has_no_duration(self) -> None:
        """When the last event is a failure, watch_last_event_duration is None."""
        m = WatchIndexMetrics()
        m.record(_make_event(duration=0.1))
        m.record(_make_event(success=False, duration=0.5, error="fail"))

        snap = m.snapshot()
        assert snap["watch_last_event_duration"] is None

    def test_record_index_convenience(self) -> None:
        m = WatchIndexMetrics()
        event = m.record_index(
            "/tmp/test.jsonl",
            "claude_code",
            success=True,
            duration=0.042,
        )
        assert isinstance(event, IndexEvent)
        assert event.path == "/tmp/test.jsonl"
        assert event.duration == 0.042
        assert m.total_indexed == 1

    def test_thread_safety(self) -> None:
        """Concurrent record() calls should not corrupt state."""
        m = WatchIndexMetrics()
        n_threads = 8
        n_per_thread = 100
        barrier = threading.Barrier(n_threads)

        def _worker():
            barrier.wait()
            for i in range(n_per_thread):
                m.record(_make_event(duration=0.001 * i))

        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert m.total_indexed == n_threads * n_per_thread
        assert len(m.recent_events()) == 50  # ring buffer cap
