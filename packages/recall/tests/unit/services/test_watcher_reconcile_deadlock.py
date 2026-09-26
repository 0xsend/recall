"""Lock-ordering regression for the live-watch discovery tick.

watchdog's `BaseObserver.dispatch_events` holds `BaseObserver._lock` for the
whole handler callback, and `schedule()`/`unschedule()` take that same lock.
`SessionFileHandler._handle` needs `reconcile_lock`, so any recall code that
holds `reconcile_lock` across a watchdog subscription call inverts the order
and deadlocks the discovery thread against the dispatch thread permanently.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import cast

from recall.parsers import SessionParser
from recall.services.live_session_set import LiveSessionSet
from recall.services.watcher import (
    _SYSTEM_CLOCK,
    IndexQueueLike,
    ObserverLike,
    SessionFileHandler,
    _run_discovery_tick,
    reset_live_snapshot,
)

# Long enough that a healthy run never trips it, short enough that the deadlock
# fails the test instead of hanging the suite.
_HANDOFF_TIMEOUT_SECONDS = 5.0


class _LockOrderObserver:
    """Observer double that reproduces watchdog's lock contract."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.schedule_entered = threading.Event()
        self.dispatch_in_handler = threading.Event()

    def schedule(self, handler: object, path: str, recursive: bool = False) -> object:
        _ = (handler, path, recursive)
        # Announce that the caller reached the inversion point, then hand the
        # dispatch thread a chance to take `_lock` first — the exact interleaving
        # observed in the stalled daemon.
        self.schedule_entered.set()
        self.dispatch_in_handler.wait(timeout=_HANDOFF_TIMEOUT_SECONDS)
        with self._lock:
            return object()

    def unschedule(self, watch: object) -> None:
        _ = watch
        with self._lock:
            return None

    def dispatch(self, handler: SessionFileHandler, event: object) -> None:
        """Mirror `BaseObserver.dispatch_events`: handler runs under `_lock`."""
        with self._lock:
            self.dispatch_in_handler.set()
            handler.on_modified(event)


class _StubParser:
    file_pattern = "*.jsonl"

    def __init__(self, root: Path, candidates: list[Path]) -> None:
        self._root = root
        self._candidates = candidates

    roots: tuple[Path, ...] | None = None

    def default_roots(self) -> list[Path]:
        return [self._root]

    def watch_roots(self) -> list[Path]:
        return [self._root]

    def live_candidates(self, *, now: object, idle_threshold: float) -> list[Path]:
        _ = (now, idle_threshold)
        return list(self._candidates)


class _RecordingQueue:
    def __init__(self) -> None:
        self.marked: list[str] = []

    def mark(self, path: str, now: float | None = None) -> None:
        _ = now
        self.marked.append(path)

    def ready(self, now: float | None = None) -> list[str]:
        _ = now
        return []


class _FakeEvent:
    is_directory = False

    def __init__(self, src_path: str) -> None:
        self.src_path = src_path


def test_discovery_tick_does_not_hold_reconcile_lock_across_observer_schedule(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session.jsonl"
    session_path.write_text("{}\n", encoding="utf-8")

    live_set = LiveSessionSet(max_subscriptions=8, idle_threshold=3600.0, is_macos=False)
    reconcile_lock = threading.Lock()
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()
    parser = cast(SessionParser, _StubParser(tmp_path, [session_path]))
    queue = _RecordingQueue()
    observer = _LockOrderObserver()
    handler = SessionFileHandler(
        cast(IndexQueueLike, queue),
        [parser],
        live_set=live_set,
        reconcile_lock=reconcile_lock,
        scheduled_watches=scheduled_watches,
    )

    dispatch_done = threading.Event()

    def _dispatch_thread() -> None:
        # Wait until the discovery tick is inside `observer.schedule()`, which is
        # where the pre-fix code still held `reconcile_lock`.
        if not observer.schedule_entered.wait(timeout=_HANDOFF_TIMEOUT_SECONDS):
            return
        observer.dispatch(handler, _FakeEvent(str(session_path)))
        dispatch_done.set()

    tick_error: list[BaseException] = []

    def _discovery_thread() -> None:
        try:
            _run_discovery_tick(
                parsers=[parser],
                live_set=live_set,
                observer=cast(ObserverLike, observer),
                handler=handler,
                queue=cast(IndexQueueLike, queue),
                scheduled_watches=scheduled_watches,
                poll_fallback=poll_fallback,
                idle_threshold=3600.0,
                discovery_interval=30.0,
                clock=_SYSTEM_CLOCK,
                reconcile_lock=reconcile_lock,
            )
        except BaseException as err:
            tick_error.append(err)

    dispatcher = threading.Thread(target=_dispatch_thread, name="test-dispatch", daemon=True)
    discovery = threading.Thread(target=_discovery_thread, name="test-discovery", daemon=True)
    try:
        dispatcher.start()
        discovery.start()

        discovery.join(timeout=_HANDOFF_TIMEOUT_SECONDS * 3)
        dispatcher.join(timeout=_HANDOFF_TIMEOUT_SECONDS * 3)

        assert not discovery.is_alive(), (
            "discovery tick deadlocked against the watchdog dispatch thread: "
            "reconcile_lock is held across observer.schedule()"
        )
        assert not dispatcher.is_alive(), "watchdog dispatch thread deadlocked on reconcile_lock"
        assert dispatch_done.is_set()
        assert not tick_error, f"discovery tick raised: {tick_error}"
        assert str(session_path) in scheduled_watches
    finally:
        reset_live_snapshot()


def test_handler_still_serializes_against_reconcile_lock(tmp_path: Path) -> None:
    """The fix narrows the critical section; it must not drop mutual exclusion."""

    session_path = tmp_path / "session.jsonl"
    session_path.write_text("{}\n", encoding="utf-8")

    live_set = LiveSessionSet(max_subscriptions=8, idle_threshold=3600.0, is_macos=False)
    reconcile_lock = threading.Lock()
    scheduled_watches: dict[str, object] = {}
    queue = _RecordingQueue()
    parser = cast(SessionParser, _StubParser(tmp_path, [session_path]))
    handler = SessionFileHandler(
        cast(IndexQueueLike, queue),
        [parser],
        live_set=live_set,
        reconcile_lock=reconcile_lock,
        scheduled_watches=scheduled_watches,
    )

    handled = threading.Event()

    def _handle() -> None:
        handler.on_modified(_FakeEvent(str(session_path)))
        handled.set()

    worker = threading.Thread(target=_handle, name="test-handler", daemon=True)
    try:
        with reconcile_lock:
            worker.start()
            assert not handled.wait(timeout=0.25), (
                "handler mutated the live set while reconcile_lock was held"
            )
        assert handled.wait(timeout=_HANDOFF_TIMEOUT_SECONDS)
        assert queue.marked == [str(session_path)]
    finally:
        worker.join(timeout=_HANDOFF_TIMEOUT_SECONDS)
        reset_live_snapshot()
