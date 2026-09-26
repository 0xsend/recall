from __future__ import annotations

import errno
import fnmatch
import logging
import platform
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import duckdb
import watchdog.observers
from watchdog.events import FileSystemEventHandler

from recall.core.config import AppConfig, ContextConfig
from recall.core.models import ParseResult, Session
from recall.core.types import DaemonMode, Source
from recall.db import (
    advisory_lock,
    connect,
    connect_readonly,
)
from recall.db.fatal import is_fatal_db_invalidation
from recall.parsers import SessionParser, all_parsers, get_parser
from recall.services.indexer import (
    ContextRun,
    DiscoveredSessionPath,
    SessionContextStats,
    _incremental_write_session,
    _load_session_state,
    _prepare_context_run,
    _resolve_parse_offset,
    _write_session,
)
from recall.services.live_events import live_path_key
from recall.services.live_session_set import LiveSessionSet

logger = logging.getLogger("recall.watcher")


def _maybe_harvest_usage(conn: duckdb.DuckDBPyConnection) -> None:
    """Harvest Grok usage log during watch/discovery ticks (REQ-USAGE-010)."""
    try:
        from recall.services.usage_harvest import harvest_grok_unified_log

        harvest_grok_unified_log(conn)
    except Exception as err:
        if is_fatal_db_invalidation(err):
            # A dead instance is not a harvest failure; it never heals in-process,
            # so swallowing it spins the watch loop on errors forever.
            raise
        # Traceback included: `str(err)` alone hid which statement produced
        # `tuple index out of range` on two fleet hosts (REQ-RESIL-023).
        logger.warning("usage harvest failed: %s", err, exc_info=True)


@dataclass
class _ContextCounters:
    messages: int = 0
    reused: int = 0
    mode: str = "off"
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None

    def add(self, stats: SessionContextStats, *, mode: str) -> None:
        self.mode = mode
        self.messages += stats.messages
        self.reused += stats.reused
        self.input_tokens += stats.input_tokens
        self.output_tokens += stats.output_tokens
        if stats.model is not None:
            self.model = stats.model


class ObserverLike(Protocol):
    def schedule(self, handler: object, path: str, recursive: bool = False) -> object: ...

    def unschedule(self, watch: object) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def join(self, timeout: float) -> None: ...


class WatcherClock(Protocol):
    def monotonic(self) -> float: ...

    def wall(self) -> float: ...


class IndexQueueLike(Protocol):
    def mark(self, path: str, now: float | None = None) -> None: ...


@dataclass(frozen=True)
class _SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def wall(self) -> float:
        return time.time()


@dataclass(frozen=True)
class WatcherLiveSnapshot:
    live_session_count: int
    live_session_paths: list[str]
    discovery_interval_seconds: float
    discovery_last_run_at: datetime | None
    discovery_last_promoted: int
    discovery_last_demoted: int
    watcher_subscription_count: int
    # Keys of currently-scheduled observer watches. On Linux, this is the set
    # of file paths; on macOS, the set of parent directories. REQ-DAEMON-050
    # requires status to reflect actual scheduled state, not desired state.
    scheduled_watch_keys: tuple[str, ...] = ()
    # REQ-DAEMON-060: set when `start_live_watch_runtime` publishes the initial
    # snapshot and cleared on `reset_live_snapshot`. Lets a cross-process status
    # query distinguish "watch runtime is live" from "no watch runtime" before
    # the first discovery tick runs, independent of whether any sessions are
    # active. Without it `_detect_runtime_mode` would report poll during the
    # startup window on an idle host.
    runtime_started_at: datetime | None = None
    catchup_in_progress: bool = False
    catchup_total: int = 0
    catchup_done: int = 0


_SYSTEM_CLOCK = _SystemClock()
_live_snapshot_lock = threading.Lock()


def _zero_live_snapshot() -> WatcherLiveSnapshot:
    return WatcherLiveSnapshot(
        live_session_count=0,
        live_session_paths=[],
        discovery_interval_seconds=0.0,
        discovery_last_run_at=None,
        discovery_last_promoted=0,
        discovery_last_demoted=0,
        watcher_subscription_count=0,
        scheduled_watch_keys=(),
        catchup_in_progress=False,
        catchup_total=0,
        catchup_done=0,
    )


_live_snapshot = _zero_live_snapshot()


@dataclass
class LiveWatchRuntime:
    """Shared runtime state for the active-session watcher."""

    live_set: LiveSessionSet
    handler: SessionFileHandler
    queue: DebouncedIndexQueue
    fts_debouncer: FtsRebuildDebouncer
    observer: ObserverLike
    scheduled_watches: dict[str, object]
    poll_fallback: set[str]
    reconcile_lock: threading.Lock
    parsers: list[SessionParser]
    clock: WatcherClock
    config: AppConfig


def get_live_snapshot() -> WatcherLiveSnapshot:
    with _live_snapshot_lock:
        return replace(
            _live_snapshot,
            live_session_paths=list(_live_snapshot.live_session_paths),
            scheduled_watch_keys=tuple(_live_snapshot.scheduled_watch_keys),
        )


def reset_live_snapshot() -> None:
    with _live_snapshot_lock:
        global _live_snapshot
        _live_snapshot = _zero_live_snapshot()


def _update_live_snapshot(
    *,
    live_set: LiveSessionSet,
    scheduled_watches: dict[str, object],
    discovery_interval: float,
    discovery_last_run_at: datetime | None,
    discovery_last_promoted: int,
    discovery_last_demoted: int,
) -> None:
    stats = live_set.stats()
    scheduled_keys = tuple(sorted(scheduled_watches.keys()))
    with _live_snapshot_lock:
        global _live_snapshot
        # REQ-DAEMON-060: preserve runtime_started_at across discovery-tick
        # updates. Only `start_live_watch_runtime` / `reset_live_snapshot`
        # mutate that field.
        _live_snapshot = replace(
            _live_snapshot,
            live_session_count=stats.count,
            live_session_paths=[str(path) for path in stats.top_paths],
            discovery_interval_seconds=discovery_interval,
            discovery_last_run_at=discovery_last_run_at,
            discovery_last_promoted=discovery_last_promoted,
            discovery_last_demoted=discovery_last_demoted,
            # REQ-DAEMON-050: report actual scheduled watches, not desired ones,
            # so poll-fallback paths (REQ-DAEMON-055) don't inflate the count.
            watcher_subscription_count=len(scheduled_keys),
            scheduled_watch_keys=scheduled_keys,
        )


def _refresh_live_counts(
    *,
    live_set: LiveSessionSet,
    scheduled_watches: dict[str, object],
) -> None:
    """Republish live session / subscription counts without touching discovery fields.

    REQ-DAEMON-059: called from the filesystem event handler so fsevent-driven
    `live_set.promote` results are visible in `get_live_snapshot()` before the
    next discovery tick. Preserves `discovery_*` and `runtime_started_at` so a
    fast fsevent cannot rewind them.
    """
    stats = live_set.stats()
    scheduled_keys = tuple(sorted(scheduled_watches.keys()))
    with _live_snapshot_lock:
        global _live_snapshot
        _live_snapshot = replace(
            _live_snapshot,
            live_session_count=stats.count,
            live_session_paths=[str(path) for path in stats.top_paths],
            watcher_subscription_count=len(scheduled_keys),
            scheduled_watch_keys=scheduled_keys,
        )


def _mark_runtime_started(value: datetime) -> None:
    with _live_snapshot_lock:
        global _live_snapshot
        _live_snapshot = replace(_live_snapshot, runtime_started_at=value)


def update_catch_up_progress(*, in_progress: bool, total: int, done: int) -> None:
    """Publish catch-up progress in the live snapshot without disturbing watch state."""

    safe_total = max(0, total)
    safe_done = min(max(0, done), safe_total)
    with _live_snapshot_lock:
        global _live_snapshot
        _live_snapshot = replace(
            _live_snapshot,
            catchup_in_progress=in_progress,
            catchup_total=safe_total,
            catchup_done=safe_done,
        )


def resolve_daemon_mode(mode: DaemonMode) -> DaemonMode:
    if mode == DaemonMode.AUTO:
        # REQ-DAEMON-040/041: watchdog is a required dependency, so AUTO always resolves
        # to watch mode and explicit WATCH never performs optional-dependency checks.
        return DaemonMode.WATCH
    return mode


class DebouncedIndexQueue:
    def __init__(self, debounce: float = 5.0) -> None:
        self._debounce = debounce
        self._events: dict[str, float] = {}
        self._lock = threading.Lock()

    def mark(self, path: str, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        with self._lock:
            self._events[path] = now

    def ready(self, now: float | None = None) -> list[str]:
        now = now if now is not None else time.monotonic()
        result: list[str] = []
        with self._lock:
            for path, last_event in list(self._events.items()):
                if now - last_event >= self._debounce:
                    result.append(path)
            for path in result:
                del self._events[path]
        return sorted(result)

    def flush(self, path: str) -> bool:
        """Take one path off the queue, reporting whether it was pending.

        `index_session_now` indexes that path immediately (REQ-LIVE-011), so
        leaving the debounce entry behind would have the drain loop index it a
        second time a moment later.
        """
        with self._lock:
            return self._events.pop(path, None) is not None

    def flush_all(self) -> list[str]:
        with self._lock:
            result = sorted(self._events.keys())
            self._events.clear()
        return result

    def pending_count(self) -> int:
        with self._lock:
            return len(self._events)


class FtsRebuildDebouncer:
    _BACKOFF_INITIAL_SECONDS = 60.0
    _BACKOFF_CAP_SECONDS = 3600.0

    def __init__(self, fts_debounce: float = 10.0) -> None:
        self._fts_debounce = fts_debounce
        self._last_dirty: float | None = None
        self._last_rebuild: float | None = None
        self._oom_count = 0
        self._last_oom_at_mono: float | None = None
        self._last_oom_at_wall: float | None = None
        self._last_oom_reason: str | None = None
        self._next_retry_at_mono: float | None = None

    def mark_dirty(self, now: float | None = None) -> None:
        self._last_dirty = now if now is not None else time.monotonic()

    def ready(self, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        if self._next_retry_at_mono is not None and now < self._next_retry_at_mono:
            return False
        if self._last_dirty is None:
            return False
        if self._last_rebuild is not None and self._last_rebuild >= self._last_dirty:
            return False
        return now - self._last_dirty >= self._fts_debounce

    def mark_oom(
        self,
        reason: str,
        *,
        now_mono: float | None = None,
        now_wall: float | None = None,
    ) -> float:
        """Record an FTS rebuild OOM and return the retry backoff window."""
        now_mono = now_mono if now_mono is not None else time.monotonic()
        now_wall = now_wall if now_wall is not None else time.time()
        self._oom_count += 1
        backoff = min(
            self._BACKOFF_INITIAL_SECONDS * (2 ** (self._oom_count - 1)),
            self._BACKOFF_CAP_SECONDS,
        )
        self._last_oom_at_mono = now_mono
        self._last_oom_at_wall = now_wall
        self._last_oom_reason = reason
        self._next_retry_at_mono = now_mono + backoff
        return backoff

    def mark_rebuilt(self, now: float | None = None) -> None:
        self._last_rebuild = now if now is not None else time.monotonic()
        self._oom_count = 0
        self._next_retry_at_mono = None

    @property
    def oom_count(self) -> int:
        return self._oom_count

    @property
    def last_oom_at_wall(self) -> float | None:
        return self._last_oom_at_wall

    @property
    def last_oom_reason(self) -> str | None:
        return self._last_oom_reason

    @property
    def next_retry_at_mono(self) -> float | None:
        return self._next_retry_at_mono

    @property
    def next_retry_at_wall(self) -> float | None:
        if (
            self._last_oom_at_wall is None
            or self._last_oom_at_mono is None
            or self._next_retry_at_mono is None
        ):
            return None
        return self._last_oom_at_wall + (self._next_retry_at_mono - self._last_oom_at_mono)


class SessionFileHandler(FileSystemEventHandler):
    """Filters filesystem events for session JSONL files and feeds a debounce queue."""

    def __init__(
        self,
        queue: IndexQueueLike,
        parsers: list[SessionParser],
        *,
        live_set: LiveSessionSet | None = None,
        clock: WatcherClock | None = None,
        reconcile_lock: threading.Lock | None = None,
        scheduled_watches: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self._queue = queue
        self._live_set = live_set
        self._clock = clock or _SYSTEM_CLOCK
        self._reconcile_lock = reconcile_lock
        # REQ-DAEMON-059: handler shares the scheduled_watches dict with the
        # runtime so fsevent-driven refreshes can publish an up-to-date
        # subscription count without forcing a full reconcile pass.
        self._scheduled_watches = scheduled_watches
        self._pattern_map: dict[str, str] = {}
        for parser in parsers:
            for root in parser.watch_roots():
                self._pattern_map[str(root)] = parser.file_pattern

    def on_modified(self, event: object) -> None:
        self._handle(event)

    def on_created(self, event: object) -> None:
        self._handle(event)

    def _handle(self, event: object) -> None:
        if getattr(event, "is_directory", False):
            return
        path: str = getattr(event, "src_path", "")
        if not path.endswith(".jsonl"):
            return
        filename = Path(path).name
        for root, pattern in self._pattern_map.items():
            if path.startswith(root) and fnmatch.fnmatch(filename, pattern):
                monotonic_now = self._clock.monotonic()
                path_obj = Path(path)
                # REQ-DAEMON-045: a fsevent that fires against a .jsonl under an
                # already-watched parent (macOS FSEvents) or a watched file
                # (Linux inotify) counts as live activity. Stat the path so
                # promote() refreshes the wall-clock mtime, and bump_event=True
                # so last_event_at moves forward on the monotonic clock per
                # REQ-DAEMON-046. promote() is idempotent for existing members.
                mtime: float | None
                try:
                    mtime = path_obj.stat().st_mtime
                except OSError as err:
                    logger.debug("fsevent stat failed path=%s reason=%s", path, err)
                    mtime = None
                lock = self._reconcile_lock or nullcontext()
                with lock:
                    if self._live_set is not None and mtime is not None:
                        self._live_set.promote(
                            path_obj,
                            mtime,
                            monotonic_now,
                            bump_event=True,
                        )
                        # REQ-DAEMON-059: republish snapshot counts so
                        # `get_live_snapshot()` reflects fsevent-driven promotes
                        # before the next discovery tick. Without this a live
                        # session remains invisible to status for up to
                        # `live_discovery_interval` seconds.
                        if self._scheduled_watches is not None:
                            _refresh_live_counts(
                                live_set=self._live_set,
                                scheduled_watches=self._scheduled_watches,
                            )
                    self._queue.mark(path, now=monotonic_now)
                return


def _datetime_from_wall(wall_now: float) -> datetime:
    return datetime.fromtimestamp(wall_now)


def _live_member_paths_by_subscription(live_set: LiveSessionSet) -> dict[str, set[str]]:
    # Phase 3 intentionally kept the public API narrow. The watcher still needs the file-level
    # membership behind each subscription key for REQ-DAEMON-055 poll fallback bookkeeping.
    live_lock = getattr(live_set, "_lock", None)
    members = getattr(live_set, "_members", None)
    if live_lock is None or members is None:
        return {}

    with live_lock:
        by_subscription: dict[str, set[str]] = {}
        for member in members.values():
            by_subscription.setdefault(member.subscription_key, set()).add(str(member.path))
        return by_subscription


def _seed_live_session_set(
    *,
    parsers: list[SessionParser],
    live_set: LiveSessionSet,
    wall_now: float,
    monotonic_now: float,
    idle_threshold: float,
) -> None:
    now = _datetime_from_wall(wall_now)
    candidates: list[tuple[Path, float]] = []
    for parser in parsers:
        for path in parser.live_candidates(now=now, idle_threshold=idle_threshold):
            try:
                candidates.append((path, path.stat().st_mtime))
            except OSError as err:
                logger.debug("live seed stat failed path=%s reason=%s", path, err)
    candidates.sort(key=lambda item: (item[1], str(item[0])))
    live_set.seed(candidates, monotonic_now=monotonic_now)


@dataclass(frozen=True)
class _SubscriptionPlan:
    """The watchdog calls one reconcile pass owes, computed under `reconcile_lock`."""

    member_paths_by_key: dict[str, set[str]]
    removals: list[tuple[str, object]]
    additions: list[str]


def _plan_observer_subscriptions_unlocked(
    *,
    live_set: LiveSessionSet,
    scheduled_watches: dict[str, object],
    poll_fallback: set[str],
) -> _SubscriptionPlan:
    """Diff the live set against the scheduled watches. Caller holds `reconcile_lock`."""

    desired = live_set.iter_subscriptions()
    desired_keys = {key for key, _ in desired}
    member_paths_by_key = _live_member_paths_by_subscription(live_set)
    current_live_paths = {path for paths in member_paths_by_key.values() for path in paths}
    poll_fallback.intersection_update(current_live_paths)

    removals = [
        (key, scheduled_watches[key]) for key in sorted(set(scheduled_watches) - desired_keys)
    ]

    for key in sorted(desired_keys & set(scheduled_watches)):
        for path in member_paths_by_key.get(key, set()):
            poll_fallback.discard(path)

    additions = [key for key, _kind in desired if key not in scheduled_watches]

    return _SubscriptionPlan(
        member_paths_by_key=member_paths_by_key,
        removals=removals,
        additions=additions,
    )


def _reconcile_observer_subscriptions(
    *,
    observer: ObserverLike,
    handler: SessionFileHandler,
    live_set: LiveSessionSet,
    scheduled_watches: dict[str, object],
    poll_fallback: set[str],
    reconcile_lock: threading.Lock,
) -> None:
    """Bring watchdog subscriptions in line with the live set.

    `reconcile_lock` must never be held across a watchdog call. watchdog runs
    handler callbacks from `dispatch_events` while holding `BaseObserver._lock`,
    and `schedule()`/`unschedule()` take that same lock; since
    `SessionFileHandler._handle` acquires `reconcile_lock`, holding it across a
    subscription call inverts the order and deadlocks the discovery thread
    against the dispatch thread — permanently, with no timeout and no exception,
    so the daemon keeps running with discovery and fsevents both dead. Plan and
    publish hold the lock; the watchdog calls between them do not.
    """

    with reconcile_lock:
        plan = _plan_observer_subscriptions_unlocked(
            live_set=live_set,
            scheduled_watches=scheduled_watches,
            poll_fallback=poll_fallback,
        )

    removed: list[str] = []
    added: dict[str, object] = {}
    schedule_failed: list[str] = []

    for key, watch in plan.removals:
        try:
            observer.unschedule(watch)
        except KeyError as err:
            # Watchdog may already have dropped an emitter after a delete-self
            # before reconciliation saw it. The desired live set no longer
            # includes this key, so drop only this known-stale local handle.
            logger.warning(
                "failed to unschedule stale live watch path=%s reason=%s",
                key,
                err,
            )
        removed.append(key)

    for key in plan.additions:
        try:
            added[key] = observer.schedule(handler, key, recursive=False)
        except OSError as err:
            # REQ-DAEMON-055: inotify ENOSPC, EACCES, ENOENT and other backend
            # scheduling failures route the path into the poll-fallback set so
            # indexing continues via the debounce queue. Narrow to OSError so
            # logic errors in our own reconcile code still surface loudly.
            logger.warning("failed to schedule live watch path=%s reason=%s", key, err)
            schedule_failed.append(key)

    with reconcile_lock:
        for key in removed:
            scheduled_watches.pop(key, None)
        for key, watch in added.items():
            scheduled_watches[key] = watch
            for path in plan.member_paths_by_key.get(key, set()):
                poll_fallback.discard(path)
        for key in schedule_failed:
            for path in plan.member_paths_by_key.get(key, {key}):
                poll_fallback.add(path)


def _run_discovery_tick(
    *,
    parsers: list[SessionParser],
    live_set: LiveSessionSet,
    observer: ObserverLike,
    handler: SessionFileHandler,
    queue: IndexQueueLike,
    scheduled_watches: dict[str, object],
    poll_fallback: set[str],
    idle_threshold: float,
    discovery_interval: float,
    clock: WatcherClock,
    reconcile_lock: threading.Lock,
) -> None:
    wall_now = clock.wall()
    monotonic_now = clock.monotonic()
    now = _datetime_from_wall(wall_now)
    promoted = 0
    known = {str(member.path) for member in live_set.members()}
    unannounced: list[str] = []
    # The sweep has to reach back at least as far as the previous tick: a file
    # written entirely between two ticks carries no observer subscription while
    # it is being written, so this sweep is the only thing that will ever see
    # its bytes. `live_idle_threshold` alone is the wrong reach — `config`
    # accepts a value below `live_discovery_interval`, and every transcript that
    # quiesced inside that gap was then dropped permanently and silently.
    # Doubled to absorb the tick's own duration and clock jitter; at the shipped
    # defaults (300 s over 30 s) this collapses to the threshold and changes
    # nothing.
    scan_threshold = max(idle_threshold, discovery_interval * 2.0)
    for parser in parsers:
        for path in parser.live_candidates(now=now, idle_threshold=scan_threshold):
            try:
                mtime = path.stat().st_mtime
            except OSError as err:
                logger.debug("live discovery stat failed path=%s reason=%s", path, err)
                continue
            if wall_now - mtime > idle_threshold:
                # Swept but too quiet to be live: index the bytes without
                # claiming the session is running.
                if str(path) not in known:
                    unannounced.append(str(path))
                continue
            # Discovery re-sweeps refresh mtime but must NOT bump last_event_at
            # for existing members — otherwise a quiet file never ages out and
            # REQ-DAEMON-046 is violated.
            if live_set.promote(path, mtime, monotonic_now, bump_event=False) is not None:
                promoted += 1
                if str(path) not in known:
                    unannounced.append(str(path))

    demoted = live_set.demote_idle(wall_now=wall_now, monotonic_now=monotonic_now)

    # Takes and releases `reconcile_lock` internally — it must not be held here,
    # or the watchdog calls inside deadlock against the dispatch thread.
    _reconcile_observer_subscriptions(
        observer=observer,
        handler=handler,
        live_set=live_set,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=reconcile_lock,
    )

    with reconcile_lock:
        # These paths have bytes on disk that no event announced: they were
        # written before any observer was subscribed to them, and a file that
        # has since quiesced has no future write to ride on. The sweep is the
        # moment that observation is made, so it is the moment to enqueue.
        # Only paths the sweep saw for the *first* time mark — discovery
        # re-sweeps every member every tick, and marking those would re-index
        # the whole live set on a timer.
        for path in sorted(poll_fallback.union(unannounced)):
            queue.mark(path, now=monotonic_now)

        _update_live_snapshot(
            live_set=live_set,
            scheduled_watches=scheduled_watches,
            discovery_interval=discovery_interval,
            discovery_last_run_at=now,
            discovery_last_promoted=promoted,
            discovery_last_demoted=len(demoted),
        )


def _default_observer_factory() -> ObserverLike:
    # `watchdog.observers.Observer` is the concrete factory at runtime, but ty
    # does not infer it through the optional fallback expression.
    observer_factory = cast(Callable[[], ObserverLike], watchdog.observers.Observer)
    return observer_factory()


def build_live_watch_runtime(
    *,
    config: AppConfig,
    source: Source | None = None,
    clock: WatcherClock | None = None,
    observer_factory: Callable[[], ObserverLike] | None = None,
) -> LiveWatchRuntime:
    """Build observation independently of optional context/embedding models."""

    parsers = [get_parser(source, config.sources)] if source else all_parsers(config.sources)
    runtime_clock = clock or _SYSTEM_CLOCK
    queue = DebouncedIndexQueue(debounce=config.daemon.debounce)
    fts_debouncer = FtsRebuildDebouncer(fts_debounce=config.daemon.fts_debounce)
    live_set = LiveSessionSet(
        max_subscriptions=config.daemon.live_max_subscriptions,
        idle_threshold=float(config.daemon.live_idle_threshold),
        is_macos=platform.system() == "Darwin",
    )
    reconcile_lock = threading.Lock()
    scheduled_watches: dict[str, object] = {}
    poll_fallback: set[str] = set()
    handler = SessionFileHandler(
        queue,
        parsers,
        live_set=live_set,
        clock=runtime_clock,
        reconcile_lock=reconcile_lock,
        scheduled_watches=scheduled_watches,
    )
    observer = observer_factory() if observer_factory is not None else _default_observer_factory()
    return LiveWatchRuntime(
        live_set=live_set,
        handler=handler,
        queue=queue,
        fts_debouncer=fts_debouncer,
        observer=observer,
        scheduled_watches=scheduled_watches,
        poll_fallback=poll_fallback,
        reconcile_lock=reconcile_lock,
        parsers=parsers,
        clock=runtime_clock,
        config=config,
    )


def start_live_watch_runtime(runtime: LiveWatchRuntime) -> None:
    """Seed the runtime, reconcile subscriptions, start the observer, and publish status."""

    _seed_live_session_set(
        parsers=runtime.parsers,
        live_set=runtime.live_set,
        wall_now=runtime.clock.wall(),
        monotonic_now=runtime.clock.monotonic(),
        idle_threshold=float(runtime.config.daemon.live_idle_threshold),
    )
    _reconcile_observer_subscriptions(
        observer=runtime.observer,
        handler=runtime.handler,
        live_set=runtime.live_set,
        scheduled_watches=runtime.scheduled_watches,
        poll_fallback=runtime.poll_fallback,
        reconcile_lock=runtime.reconcile_lock,
    )
    try:
        runtime.observer.start()
    except OSError as err:
        if err.errno == errno.ENOSPC:
            raise OSError(
                errno.ENOSPC,
                "inotify watch limit exceeded. Increase the limit with:\n"
                "  sudo sysctl fs.inotify.max_user_watches=524288\n"
                "To make it permanent, add to /etc/sysctl.conf:\n"
                "  fs.inotify.max_user_watches=524288",
            ) from err
        raise
    _update_live_snapshot(
        live_set=runtime.live_set,
        scheduled_watches=runtime.scheduled_watches,
        discovery_interval=float(runtime.config.daemon.live_discovery_interval),
        discovery_last_run_at=None,
        discovery_last_promoted=0,
        discovery_last_demoted=0,
    )
    # REQ-DAEMON-060: publish runtime_started_at AFTER the initial snapshot so
    # `_detect_runtime_mode` returns WATCH immediately, even on an idle host
    # where the first discovery tick is still up to `live_discovery_interval`
    # seconds away.
    _mark_runtime_started(_datetime_from_wall(runtime.clock.wall()))


def run_live_discovery_tick(runtime: LiveWatchRuntime) -> None:
    """Run one blocking discovery tick against a shared runtime."""

    _run_discovery_tick(
        parsers=runtime.parsers,
        live_set=runtime.live_set,
        observer=runtime.observer,
        handler=runtime.handler,
        queue=runtime.queue,
        scheduled_watches=runtime.scheduled_watches,
        poll_fallback=runtime.poll_fallback,
        idle_threshold=float(runtime.config.daemon.live_idle_threshold),
        discovery_interval=float(runtime.config.daemon.live_discovery_interval),
        clock=runtime.clock,
        reconcile_lock=runtime.reconcile_lock,
    )


def stop_live_watch_runtime(runtime: LiveWatchRuntime) -> None:
    """Stop the observer and clear the published live snapshot."""

    try:
        runtime.observer.stop()
        runtime.observer.join(timeout=5.0)
    finally:
        reset_live_snapshot()


def _resolve_parser_for_path(path_str: str, parsers: list[SessionParser]) -> SessionParser | None:
    """Find the parser whose watch root contains this path.

    Both spellings are accepted: fsevents deliver the path under the
    unresolved root `watch_roots()` builds from `$HOME`, while a caller reading
    `sessions.source_path` holds the resolved one. On a host whose home is a
    symlink those never match as plain strings.
    """
    canonical = live_path_key(path_str)
    for parser in parsers:
        for root in parser.watch_roots():
            if path_str.startswith(str(root)) or canonical.startswith(live_path_key(str(root))):
                return parser
    return None


def _resolve_watch_session_write_contexts(
    conn: duckdb.DuckDBPyConnection,
    session: Session,
    context_run: ContextRun,
    *,
    is_full_parse: bool,
) -> SessionContextStats:
    """Resolve write contexts for watch indexing with llm-codex batch support."""

    from recall.services.indexer import _resolve_session_write_contexts

    return _resolve_session_write_contexts(conn, session, context_run, is_full_parse=is_full_parse)


def _lightweight_watch_context(context: ContextConfig) -> ContextConfig:
    if context.mode not in {"llm-local", "llm-remote", "llm-codex"}:
        return context
    fallback_mode = "off" if context.fallback == "error" else context.fallback
    return replace(context, mode=fallback_mode)


def _session_state_exists(conn: duckdb.DuckDBPyConnection, session_id: str) -> bool:
    """Whether the row a resumed append would merge into is still there."""
    return (
        conn.execute("SELECT 1 FROM session_state WHERE session_id = ?", [session_id]).fetchone()
        is not None
    )


def index_single_session(
    path: Path,
    parser: SessionParser,
    config: AppConfig,
    *,
    conn: duckdb.DuckDBPyConnection | None = None,
    context_counts: _ContextCounters | None = None,
    lightweight_context: bool = False,
    on_commit: Callable[[Session, ParseResult], None] | None = None,
    prepared_result: ParseResult | None = None,
    host: str | None = None,
) -> bool:

    stat = path.stat()
    discovered = DiscoveredSessionPath(
        parser=parser,
        path=path,
        resolved_path=str(path.resolve()),
        file_mtime=stat.st_mtime,
        file_size=stat.st_size,
    )

    # Load existing state for this single session (not all sessions)
    if conn is not None:
        try:
            state = _load_session_state(conn, discovered.resolved_path)
        except Exception:
            state = None
    else:
        try:
            ro_conn = connect_readonly(config)
            try:
                state = _load_session_state(ro_conn, discovered.resolved_path)
            finally:
                ro_conn.close()
        except Exception:
            state = None
    if prepared_result is None:
        parse_offset = _resolve_parse_offset(state, discovered)
        msg_base = state.message_count if state and parse_offset > 0 else 0
        orphan_base = state.orphan_tool_count if state and parse_offset > 0 else 0
        parse_result = parser.parse(
            path,
            offset=parse_offset,
            message_idx_base=msg_base,
            orphan_tool_call_idx_base=orphan_base,
        )
    else:
        # The coordinator captured complete raw input before taking the writer,
        # and already decided between the whole source and a suffix it verified
        # against the committed prefix. Re-deciding here would discard that
        # proof, which this path cannot rebuild.
        parse_result = prepared_result
        assert conn is not None, "a prepared capture is committed on the writer's connection"
        if not parse_result.is_full_parse and not _session_state_exists(
            conn, parse_result.session.id
        ):
            # The proof describes rows this database no longer holds -- a
            # restored backup, a partial repair, a manual purge. Appending onto
            # nothing would strand the prefix behind an acknowledged offset, so
            # take the reference path, which rebuilds the whole session exactly
            # as the raw path did before it could resume (REQ-INDEX-025).
            logger.warning(
                "resume discarded, session rows missing path=%s session_id=%s",
                path,
                parse_result.session.id,
            )
            parse_result = parser.parse(path, offset=0)
    # A watch write is the same destructive full-normalization boundary as a
    # batch index.  Do not let a torn/malformed/unsupported captured result
    # replace valid history; the next event or reconciliation pass retries it.
    from recall.services.indexer import _require_committable_capture

    try:
        _require_committable_capture(
            path, parse_result, has_indexed_history=state is not None, conn=conn
        )
    except RuntimeError as err:
        logger.warning("watch index deferred path=%s error=%s", path, err)
        return False
    session = parse_result.session

    # When caller provides a connection, they own the lock and lifecycle.
    owned_conn = conn is None
    if owned_conn:
        lock_ctx = advisory_lock(config.lock_path)
    else:
        from contextlib import contextmanager

        @contextmanager
        def _noop():
            yield

        lock_ctx = _noop()

    with lock_ctx:
        if owned_conn:
            conn = connect(config)
        assert conn is not None
        if host is None:
            existing_host = conn.execute(
                "SELECT host FROM session_state WHERE session_id = ?", [session.id]
            ).fetchone()
            if existing_host is not None:
                host = existing_host[0]
        sidecar_conn: sqlite3.Connection | None = None
        try:
            if config.fts.backend == "sqlite_sidecar":
                from recall.db import open_sidecar, sidecar_path

                sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
            context_config = (
                _lightweight_watch_context(config.embedding.context)
                if lightweight_context
                else config.embedding.context
            )
            context_run = _prepare_context_run(context_config)
            session_context = _resolve_watch_session_write_contexts(
                conn,
                session,
                context_run,
                is_full_parse=parse_result.is_full_parse,
            )
            if parse_result.is_full_parse:
                result = _write_session(
                    conn,
                    session,
                    last_byte_offset=parse_result.next_byte_offset,
                    sidecar_conn=sidecar_conn,
                    fts_fields=config.fts.fields,
                    host=host,
                    tail_facts=parse_result.tail_facts,
                    on_commit=(
                        (lambda: on_commit(session, parse_result))
                        if on_commit is not None
                        else None
                    ),
                )
            else:
                result = _incremental_write_session(
                    conn,
                    session,
                    last_byte_offset=parse_result.next_byte_offset,
                    sidecar_conn=sidecar_conn,
                    fts_fields=config.fts.fields,
                    host=host,
                    tail_facts=parse_result.tail_facts,
                    on_commit=(
                        (lambda: on_commit(session, parse_result))
                        if on_commit is not None
                        else None
                    ),
                )
            if context_counts is not None:
                context_counts.add(session_context, mode=context_run.mode)
            if owned_conn:
                conn.execute("CHECKPOINT")
            return result
        finally:
            if sidecar_conn is not None:
                sidecar_conn.close()
            if owned_conn:
                conn.close()
