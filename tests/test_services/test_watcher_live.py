from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never

import pytest
from recall.core.config import AppConfig
from recall.core.types import DaemonMode, Source
from recall.services.live_session_set import LiveSessionSet
from recall.services.watcher import (
    LiveWatchRuntime,
    SessionFileHandler,
    _reconcile_observer_subscriptions,
    _run_discovery_tick,
    _seed_live_session_set,
    build_live_watch_runtime,
    get_live_snapshot,
    reset_live_snapshot,
    resolve_daemon_mode,
    start_live_watch_runtime,
    stop_live_watch_runtime,
)


def _touch(path: Path, *, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


@dataclass
class FakeClock:
    wall_now: float
    monotonic_now: float

    def monotonic(self) -> float:
        return self.monotonic_now

    def wall(self) -> float:
        return self.wall_now

    def advance(self, seconds: float) -> None:
        self.wall_now += seconds
        self.monotonic_now += seconds


@dataclass(frozen=True)
class FakeWatch:
    path: str
    recursive: bool


class FakeObserver:
    def __init__(self, *, fail_paths: set[str] | None = None) -> None:
        self.fail_paths = fail_paths or set()
        self.schedule_calls: list[tuple[str, bool]] = []
        self.scheduled: dict[str, FakeWatch] = {}
        self.unscheduled: list[FakeWatch] = []

    def schedule(self, handler: object, path: str, recursive: bool = False) -> FakeWatch:
        self.schedule_calls.append((path, recursive))
        if path in self.fail_paths:
            raise OSError(f"schedule failed for {path}")
        watch = FakeWatch(path=path, recursive=recursive)
        self.scheduled[path] = watch
        return watch

    def unschedule(self, watch: object) -> None:
        if isinstance(watch, FakeWatch):
            self.unscheduled.append(watch)
            self.scheduled.pop(watch.path, None)

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def join(self, timeout: float) -> None:
        return None


class FailingUnscheduleObserver(FakeObserver):
    def unschedule(self, watch: object) -> None:
        raise KeyError("watch missing from backend")


class RuntimeErrorUnscheduleObserver(FakeObserver):
    def unschedule(self, watch: object) -> None:
        raise RuntimeError("observer state corrupted")


class FakeQueue:
    def __init__(self) -> None:
        self.mark_calls: list[str] = []

    def mark(self, path: str, now: float | None = None) -> None:
        self.mark_calls.append(path)


class FakeParser:
    def __init__(self, root: Path, source: Source, candidates: list[Path]) -> None:
        self._root = root
        self.source = source
        self.file_pattern = "*.jsonl"
        self._candidates = list(candidates)

    def discover(self) -> list[Path]:
        raise AssertionError("discover() is not used by watcher-live tests")

    def sidecar_paths(self, path: Path) -> list[Path]:
        _ = path
        return []

    def parse(
        self,
        path: Path,
        *,
        offset: int = 0,
        message_idx_base: int = 0,
        orphan_tool_call_idx_base: int = 0,
        resume_state: Mapping[str, Any] | None = None,
    ) -> Never:
        raise AssertionError("parse() is not used by watcher-live tests")

    roots: tuple[Path, ...] | None = None

    def default_roots(self) -> list[Path]:
        return [self._root]

    def watch_roots(self) -> list[Path]:
        return [self._root]

    def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
        # Same window `default_live_candidates` applies: the threshold the tick
        # passes decides how far back the sweep reaches, so a fake that ignores
        # it cannot grade that choice.
        cutoff = (now - timedelta(seconds=idle_threshold)).timestamp()
        return [path for path in self._candidates if path.stat().st_mtime >= cutoff]

    def set_candidates(self, candidates: list[Path]) -> None:
        self._candidates = list(candidates)


def test_seed_and_schedule_linux_and_macos(tmp_path: Path) -> None:
    alpha = _touch(tmp_path / "alpha" / "one.jsonl", mtime=110.0)
    beta = _touch(tmp_path / "beta" / "two.jsonl", mtime=120.0)

    for is_macos, expected_paths in (
        (False, [str(alpha), str(beta)]),
        (True, [str(alpha.parent), str(beta.parent)]),
    ):
        parser_a = FakeParser(tmp_path / "alpha", Source.CODEX, [alpha])
        parser_b = FakeParser(tmp_path / "beta", Source.CLAUDE_CODE, [beta])
        live_set = LiveSessionSet(
            max_subscriptions=4,
            idle_threshold=300.0,
            is_macos=is_macos,
        )
        observer = FakeObserver()
        scheduled_watches: dict[str, object] = {}
        poll_fallback: set[str] = set()
        clock = FakeClock(wall_now=200.0, monotonic_now=200.0)
        handler = SessionFileHandler(
            FakeQueue(),
            [parser_a, parser_b],
            live_set=live_set,
            clock=clock,
        )

        _seed_live_session_set(
            parsers=[parser_a, parser_b],
            live_set=live_set,
            wall_now=clock.wall(),
            monotonic_now=clock.monotonic(),
            idle_threshold=300.0,
        )
        _reconcile_observer_subscriptions(
            observer=observer,
            handler=handler,
            live_set=live_set,
            scheduled_watches=scheduled_watches,
            poll_fallback=poll_fallback,
            reconcile_lock=threading.Lock(),
        )

        assert observer.schedule_calls == [(path, False) for path in expected_paths]
        assert sorted(scheduled_watches) == sorted(expected_paths)
        assert poll_fallback == set()


def test_promote_on_discovery_adds_new_observer_watch(tmp_path: Path) -> None:
    existing = _touch(tmp_path / "live" / "existing.jsonl", mtime=110.0)
    discovered = _touch(tmp_path / "live" / "discovered.jsonl", mtime=130.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [existing])
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=300.0, is_macos=False)
    observer = FakeObserver()
    queue = FakeQueue()
    lock = threading.Lock()
    clock = FakeClock(wall_now=200.0, monotonic_now=200.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )

    _seed_live_session_set(
        parsers=[parser],
        live_set=live_set,
        wall_now=clock.wall(),
        monotonic_now=clock.monotonic(),
        idle_threshold=300.0,
    )
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()
    _reconcile_observer_subscriptions(
        observer=observer,
        handler=handler,
        live_set=live_set,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=lock,
    )

    parser.set_candidates([existing, discovered])
    _run_discovery_tick(
        parsers=[parser],
        live_set=live_set,
        observer=observer,
        handler=handler,
        queue=queue,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        idle_threshold=300.0,
        discovery_interval=30.0,
        clock=clock,
        reconcile_lock=lock,
    )

    assert sorted(observer.scheduled) == sorted([str(existing), str(discovered)])
    snapshot = get_live_snapshot()
    assert snapshot.live_session_count == 2
    assert snapshot.watcher_subscription_count == 2


def test_demote_idle_unschedules_observer_watch(tmp_path: Path) -> None:
    path = _touch(tmp_path / "live" / "idle.jsonl", mtime=10.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [path])
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=30.0, is_macos=False)
    observer = FakeObserver()
    queue = FakeQueue()
    lock = threading.Lock()
    clock = FakeClock(wall_now=40.0, monotonic_now=40.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()

    _seed_live_session_set(
        parsers=[parser],
        live_set=live_set,
        wall_now=clock.wall(),
        monotonic_now=clock.monotonic(),
        idle_threshold=30.0,
    )
    _reconcile_observer_subscriptions(
        observer=observer,
        handler=handler,
        live_set=live_set,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=lock,
    )

    parser.set_candidates([])
    clock.advance(31.0)
    _run_discovery_tick(
        parsers=[parser],
        live_set=live_set,
        observer=observer,
        handler=handler,
        queue=queue,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        idle_threshold=30.0,
        discovery_interval=30.0,
        clock=clock,
        reconcile_lock=lock,
    )

    assert observer.scheduled == {}
    assert [watch.path for watch in observer.unscheduled] == [str(path)]


def test_demote_idle_tolerates_backend_unschedule_drift(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _touch(tmp_path / "live" / "stale.jsonl", mtime=10.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [path])
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=30.0, is_macos=False)
    observer = FailingUnscheduleObserver()
    queue = FakeQueue()
    lock = threading.Lock()
    clock = FakeClock(wall_now=40.0, monotonic_now=40.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()

    _seed_live_session_set(
        parsers=[parser],
        live_set=live_set,
        wall_now=clock.wall(),
        monotonic_now=clock.monotonic(),
        idle_threshold=30.0,
    )
    _reconcile_observer_subscriptions(
        observer=observer,
        handler=handler,
        live_set=live_set,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=lock,
    )

    parser.set_candidates([])
    clock.advance(31.0)
    with caplog.at_level("WARNING", logger="recall.watcher"):
        _run_discovery_tick(
            parsers=[parser],
            live_set=live_set,
            observer=observer,
            handler=handler,
            queue=queue,
            scheduled_watches=scheduled_watches,
            poll_fallback=poll_fallback,
            idle_threshold=30.0,
            discovery_interval=30.0,
            clock=clock,
            reconcile_lock=lock,
        )

    assert scheduled_watches == {}
    assert get_live_snapshot().watcher_subscription_count == 0
    assert "failed to unschedule stale live watch" in caplog.text


def test_demote_idle_preserves_handle_for_unexpected_unschedule_failure(
    tmp_path: Path,
) -> None:
    path = _touch(tmp_path / "live" / "corrupt.jsonl", mtime=10.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [path])
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=30.0, is_macos=False)
    observer = RuntimeErrorUnscheduleObserver()
    queue = FakeQueue()
    lock = threading.Lock()
    clock = FakeClock(wall_now=40.0, monotonic_now=40.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()

    _seed_live_session_set(
        parsers=[parser],
        live_set=live_set,
        wall_now=clock.wall(),
        monotonic_now=clock.monotonic(),
        idle_threshold=30.0,
    )
    _reconcile_observer_subscriptions(
        observer=observer,
        handler=handler,
        live_set=live_set,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=lock,
    )

    parser.set_candidates([])
    clock.advance(31.0)
    with pytest.raises(RuntimeError, match="observer state corrupted"):
        _run_discovery_tick(
            parsers=[parser],
            live_set=live_set,
            observer=observer,
            handler=handler,
            queue=queue,
            scheduled_watches=scheduled_watches,
            poll_fallback=poll_fallback,
            idle_threshold=30.0,
            discovery_interval=30.0,
            clock=clock,
            reconcile_lock=lock,
        )

    assert list(scheduled_watches) == [str(path)]


def test_overflow_preserves_roster_and_caps_subscriptions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = _touch(tmp_path / "live" / "first.jsonl", mtime=10.0)
    second = _touch(tmp_path / "live" / "second.jsonl", mtime=20.0)
    third = _touch(tmp_path / "live" / "third.jsonl", mtime=30.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [first, second])
    live_set = LiveSessionSet(max_subscriptions=2, idle_threshold=300.0, is_macos=False)
    observer = FakeObserver()
    queue = FakeQueue()
    lock = threading.Lock()
    clock = FakeClock(wall_now=100.0, monotonic_now=100.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()

    _seed_live_session_set(
        parsers=[parser],
        live_set=live_set,
        wall_now=clock.wall(),
        monotonic_now=clock.monotonic(),
        idle_threshold=300.0,
    )
    _reconcile_observer_subscriptions(
        observer=observer,
        handler=handler,
        live_set=live_set,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=lock,
    )

    parser.set_candidates([first, second, third])
    with caplog.at_level("INFO"):
        _run_discovery_tick(
            parsers=[parser],
            live_set=live_set,
            observer=observer,
            handler=handler,
            queue=queue,
            scheduled_watches=scheduled_watches,
            poll_fallback=poll_fallback,
            idle_threshold=300.0,
            discovery_interval=30.0,
            clock=clock,
            reconcile_lock=lock,
        )

    assert len(observer.scheduled) == 2
    assert sorted(observer.scheduled) == sorted([str(second), str(third)])
    assert {member.path for member in live_set.members()} == {first, second, third}


def test_older_overflow_member_remains_observable_without_displacing_subscriptions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    newer_a = _touch(tmp_path / "live" / "newer-a.jsonl", mtime=50.0)
    newer_b = _touch(tmp_path / "live" / "newer-b.jsonl", mtime=60.0)
    rejected = _touch(tmp_path / "live" / "rejected.jsonl", mtime=40.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [newer_a, newer_b])
    live_set = LiveSessionSet(max_subscriptions=2, idle_threshold=300.0, is_macos=False)
    observer = FakeObserver()
    queue = FakeQueue()
    lock = threading.Lock()
    clock = FakeClock(wall_now=100.0, monotonic_now=100.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()

    _seed_live_session_set(
        parsers=[parser],
        live_set=live_set,
        wall_now=clock.wall(),
        monotonic_now=clock.monotonic(),
        idle_threshold=300.0,
    )
    _reconcile_observer_subscriptions(
        observer=observer,
        handler=handler,
        live_set=live_set,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=lock,
    )

    before_watches = dict(scheduled_watches)
    parser.set_candidates([newer_a, newer_b, rejected])
    with caplog.at_level("WARNING"):
        _run_discovery_tick(
            parsers=[parser],
            live_set=live_set,
            observer=observer,
            handler=handler,
            queue=queue,
            scheduled_watches=scheduled_watches,
            poll_fallback=poll_fallback,
            idle_threshold=300.0,
            discovery_interval=30.0,
            clock=clock,
            reconcile_lock=lock,
        )

    assert scheduled_watches == before_watches
    assert {member.path for member in live_set.members()} == {newer_a, newer_b, rejected}


def test_touch_refreshes_member_and_prevents_demote(tmp_path: Path) -> None:
    path = _touch(tmp_path / "live" / "touch.jsonl", mtime=10.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [path])
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=30.0, is_macos=False)
    queue = FakeQueue()
    observer = FakeObserver()
    lock = threading.Lock()
    clock = FakeClock(wall_now=40.0, monotonic_now=40.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()

    _seed_live_session_set(
        parsers=[parser],
        live_set=live_set,
        wall_now=clock.wall(),
        monotonic_now=clock.monotonic(),
        idle_threshold=30.0,
    )
    _reconcile_observer_subscriptions(
        observer=observer,
        handler=handler,
        live_set=live_set,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=lock,
    )

    clock.advance(25.0)
    handler.on_modified(SimpleNamespace(is_directory=False, src_path=str(path)))
    parser.set_candidates([])
    clock.advance(10.0)
    _run_discovery_tick(
        parsers=[parser],
        live_set=live_set,
        observer=observer,
        handler=handler,
        queue=queue,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        idle_threshold=30.0,
        discovery_interval=30.0,
        clock=clock,
        reconcile_lock=lock,
    )

    assert queue.mark_calls == [str(path)]
    assert observer.scheduled == {str(path): observer.scheduled[str(path)]}
    assert live_set.stats().count == 1


def test_graceful_degradation_uses_poll_fallback_until_demote(tmp_path: Path) -> None:
    path = _touch(tmp_path / "live" / "fallback.jsonl", mtime=100.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [path])
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=30.0, is_macos=False)
    observer = FakeObserver(fail_paths={str(path)})
    queue = FakeQueue()
    lock = threading.Lock()
    clock = FakeClock(wall_now=110.0, monotonic_now=110.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()

    _seed_live_session_set(
        parsers=[parser],
        live_set=live_set,
        wall_now=clock.wall(),
        monotonic_now=clock.monotonic(),
        idle_threshold=30.0,
    )
    _reconcile_observer_subscriptions(
        observer=observer,
        handler=handler,
        live_set=live_set,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=lock,
    )

    assert poll_fallback == {str(path)}
    assert scheduled_watches == {}

    _run_discovery_tick(
        parsers=[parser],
        live_set=live_set,
        observer=observer,
        handler=handler,
        queue=queue,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        idle_threshold=30.0,
        discovery_interval=30.0,
        clock=clock,
        reconcile_lock=lock,
    )
    assert queue.mark_calls == [str(path)]

    parser.set_candidates([])
    clock.advance(31.0)
    _run_discovery_tick(
        parsers=[parser],
        live_set=live_set,
        observer=observer,
        handler=handler,
        queue=queue,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        idle_threshold=30.0,
        discovery_interval=30.0,
        clock=clock,
        reconcile_lock=lock,
    )

    assert poll_fallback == set()
    assert live_set.stats().count == 0


def test_auto_resolve_to_watch() -> None:
    # REQ-DAEMON-040/041 make watchdog a required runtime dependency.
    assert resolve_daemon_mode(DaemonMode.AUTO) == DaemonMode.WATCH


def _config_with_temp(tmp_path: Path) -> AppConfig:
    from dataclasses import replace as dc_replace

    data_dir = tmp_path / "recall_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig.load()
    daemon_cfg = dc_replace(
        cfg.daemon,
        mode=DaemonMode.WATCH,
        live_discovery_interval=60,
        live_idle_threshold=300,
        embed=False,
    )
    return dc_replace(cfg, data_dir=data_dir, daemon=daemon_cfg)


def _build_runtime_with_fake_observer(cfg: AppConfig) -> LiveWatchRuntime:
    return build_live_watch_runtime(
        config=cfg,
        source=Source.CODEX,
        observer_factory=lambda: FakeObserver(),
    )


def test_start_live_watch_runtime_marks_runtime_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-DAEMON-060: `runtime_started_at` populates on start so
    `_detect_runtime_mode` reports watch during the startup window on an idle
    host, before the first discovery tick runs."""
    # Isolate the parser from real ~/.claude and ~/.codex state.
    monkeypatch.setenv("HOME", str(tmp_path))
    reset_live_snapshot()
    try:
        assert get_live_snapshot().runtime_started_at is None
        cfg = _config_with_temp(tmp_path)
        runtime = _build_runtime_with_fake_observer(cfg)
        try:
            start_live_watch_runtime(runtime)
            snapshot = get_live_snapshot()
            # All counts zero on an idle host, but runtime_started_at is set.
            assert snapshot.live_session_count == 0
            assert snapshot.watcher_subscription_count == 0
            assert snapshot.discovery_last_run_at is None
            assert snapshot.runtime_started_at is not None
        finally:
            stop_live_watch_runtime(runtime)
        # Shutdown clears runtime_started_at.
        assert get_live_snapshot().runtime_started_at is None
    finally:
        reset_live_snapshot()


def test_fsevent_promote_refreshes_live_snapshot_counts(tmp_path: Path) -> None:
    """REQ-DAEMON-059: fsevent-driven `live_set.promote` must republish the
    snapshot so `get_live_snapshot().live_session_count` reflects new live
    sessions before the next discovery tick."""
    reset_live_snapshot()
    try:
        session_dir = tmp_path / "live"
        first = _touch(session_dir / "first.jsonl", mtime=100.0)

        live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=300.0, is_macos=False)
        observer = FakeObserver()
        queue = FakeQueue()
        lock = threading.Lock()
        scheduled_watches: dict[str, object] = {}
        parser = FakeParser(session_dir, Source.CODEX, [first])

        handler = SessionFileHandler(
            queue,
            [parser],
            live_set=live_set,
            clock=FakeClock(wall_now=200.0, monotonic_now=200.0),
            reconcile_lock=lock,
            scheduled_watches=scheduled_watches,
        )

        _seed_live_session_set(
            parsers=[parser],
            live_set=live_set,
            wall_now=200.0,
            monotonic_now=200.0,
            idle_threshold=300.0,
        )
        _reconcile_observer_subscriptions(
            observer=observer,
            handler=handler,
            live_set=live_set,
            scheduled_watches=scheduled_watches,
            poll_fallback=set(),
            reconcile_lock=lock,
        )
        # Publish an initial snapshot via discovery so refresh has a baseline.
        _run_discovery_tick(
            parsers=[parser],
            live_set=live_set,
            observer=observer,
            handler=handler,
            queue=queue,
            scheduled_watches=scheduled_watches,
            poll_fallback=set(),
            idle_threshold=300.0,
            discovery_interval=30.0,
            clock=FakeClock(wall_now=200.0, monotonic_now=200.0),
            reconcile_lock=lock,
        )
        baseline = get_live_snapshot()
        assert baseline.live_session_count == 1
        baseline_discovery_run_at = baseline.discovery_last_run_at
        assert baseline_discovery_run_at is not None

        # Simulate fsevent: a second live file appears, handler dispatches on_created.
        second = _touch(session_dir / "second.jsonl", mtime=210.0)
        event = SimpleNamespace(is_directory=False, src_path=str(second))
        handler.on_created(event)

        refreshed = get_live_snapshot()
        # Count reflects the new live member immediately.
        assert refreshed.live_session_count == 2
        assert sorted(refreshed.live_session_paths) == sorted([str(first), str(second)])
        # Discovery fields preserved (not rewound by the refresh).
        assert refreshed.discovery_last_run_at == baseline_discovery_run_at
    finally:
        reset_live_snapshot()


def test_a_newly_promoted_path_is_queued_for_indexing(tmp_path: Path) -> None:
    """A file that quiesced before discovery found it has no future write to ride on.

    Found by the U22 bug bash: a write burst that starts and ends between two
    discovery ticks lands on a path with no observer subscription, so no event
    ever fires for it. Discovery promotes the path a tick later and subscribes,
    but the bytes already on disk are never indexed — the row sat at
    `freshness.current: false` for 93 s across sixteen reads, and one session
    reported a turn state eight minutes stale. Promotion is the only moment
    that observes the file exists and is behind, so promotion must enqueue it.
    """
    existing = _touch(tmp_path / "live" / "existing.jsonl", mtime=110.0)
    quiesced = _touch(tmp_path / "live" / "quiesced.jsonl", mtime=130.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [existing])
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=300.0, is_macos=False)
    observer = FakeObserver()
    queue = FakeQueue()
    lock = threading.Lock()
    clock = FakeClock(wall_now=200.0, monotonic_now=200.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )
    _seed_live_session_set(
        parsers=[parser],
        live_set=live_set,
        wall_now=clock.wall(),
        monotonic_now=clock.monotonic(),
        idle_threshold=300.0,
    )
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()
    _reconcile_observer_subscriptions(
        observer=observer,
        handler=handler,
        live_set=live_set,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=lock,
    )
    queue.mark_calls.clear()

    parser.set_candidates([existing, quiesced])
    _run_discovery_tick(
        parsers=[parser],
        live_set=live_set,
        observer=observer,
        handler=handler,
        queue=queue,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        idle_threshold=300.0,
        discovery_interval=30.0,
        clock=clock,
        reconcile_lock=lock,
    )

    assert queue.mark_calls == [str(quiesced)]


def test_a_re_swept_member_is_not_queued_again(tmp_path: Path) -> None:
    """Discovery re-sweeps every member every 30 s; marking them all would re-index the fleet."""
    existing = _touch(tmp_path / "live" / "existing.jsonl", mtime=110.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [existing])
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=300.0, is_macos=False)
    observer = FakeObserver()
    queue = FakeQueue()
    lock = threading.Lock()
    clock = FakeClock(wall_now=200.0, monotonic_now=200.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )
    _seed_live_session_set(
        parsers=[parser],
        live_set=live_set,
        wall_now=clock.wall(),
        monotonic_now=clock.monotonic(),
        idle_threshold=300.0,
    )
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()
    _reconcile_observer_subscriptions(
        observer=observer,
        handler=handler,
        live_set=live_set,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=lock,
    )
    queue.mark_calls.clear()

    for _ in range(3):
        _run_discovery_tick(
            parsers=[parser],
            live_set=live_set,
            observer=observer,
            handler=handler,
            queue=queue,
            scheduled_watches=scheduled_watches,
            poll_fallback=poll_fallback,
            idle_threshold=300.0,
            discovery_interval=30.0,
            clock=clock,
            reconcile_lock=lock,
        )

    assert queue.mark_calls == []


def test_a_path_that_quiesced_past_the_live_window_is_still_indexed(tmp_path: Path) -> None:
    """Discovery is the only observer a cold path ever gets, so its sweep sets the floor.

    Found by the U24 bug bash: `live_idle_threshold` doubles as the sweep's
    reach, and `config` accepts a value below the 30 s discovery interval
    without complaint. A session written entirely between two ticks that then
    went quiet for longer than the threshold was enumerated by nothing, watched
    by nothing, and indexed by nothing -- absent from `recall list` and `recall
    live --all` 208 s and seven ticks later, with `watch_total_failed` still 0.
    The bytes are indexed; the session is still too quiet to be called live.
    """
    quiesced = _touch(tmp_path / "live" / "quiesced.jsonl", mtime=176.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [quiesced])
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=20.0, is_macos=False)
    observer = FakeObserver()
    queue = FakeQueue()
    lock = threading.Lock()
    clock = FakeClock(wall_now=200.0, monotonic_now=200.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )

    _run_discovery_tick(
        parsers=[parser],
        live_set=live_set,
        observer=observer,
        handler=handler,
        queue=queue,
        scheduled_watches={},
        poll_fallback=set(),
        idle_threshold=20.0,
        discovery_interval=30.0,
        clock=clock,
        reconcile_lock=lock,
    )

    assert queue.mark_calls == [str(quiesced)]
    assert [str(member.path) for member in live_set.members()] == []


def test_the_sweep_reaches_back_no_further_than_two_discovery_intervals(
    tmp_path: Path,
) -> None:
    """The widened reach is a bound, not an open catch-up sweep of every root."""
    ancient = _touch(tmp_path / "live" / "ancient.jsonl", mtime=100.0)
    parser = FakeParser(tmp_path / "live", Source.CODEX, [ancient])
    live_set = LiveSessionSet(max_subscriptions=4, idle_threshold=20.0, is_macos=False)
    observer = FakeObserver()
    queue = FakeQueue()
    lock = threading.Lock()
    clock = FakeClock(wall_now=200.0, monotonic_now=200.0)
    handler = SessionFileHandler(
        queue,
        [parser],
        live_set=live_set,
        clock=clock,
        reconcile_lock=lock,
    )

    _run_discovery_tick(
        parsers=[parser],
        live_set=live_set,
        observer=observer,
        handler=handler,
        queue=queue,
        scheduled_watches={},
        poll_fallback=set(),
        idle_threshold=20.0,
        discovery_interval=30.0,
        clock=clock,
        reconcile_lock=lock,
    )

    assert queue.mark_calls == []
