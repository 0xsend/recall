"""Publish and read the progress of an automatic index repair.

The owning daemon repairs before it binds its socket or writes its pid file, so
for the whole repair a client sees neither. `ProgressPublisher` leaves
`repair-progress.json` for them: one background thread renders an in-memory
snapshot that the repair code updates and never touches the file itself.
Readers decide liveness from the database ownership gate, never from the pid
the file records.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from recall.core.config import AppConfig
from recall.core.daemon_status import RecoveryProgress, RecoveryState, RecoveryStatus
from recall.core.repair_progress import (
    FRESH_SECONDS,
    HEARTBEAT_SECONDS,
    PROGRESS_FILENAME,
    PROGRESS_TEMP_FILENAME,
    Phase,
    ProgressRecord,
    ProgressUnit,
    TriggerKind,
    describe_progress,
    read_progress,
    remove_progress,
)
from recall.db.oracle_sort import ScanReport, TableSortStats
from recall.db.ownership import ownership_busy

logger = logging.getLogger("recall.repair_progress")

FIRST_WRITE_WAIT_SECONDS = 1.0
JOIN_SECONDS_MAX = 5.0
WARNING_INTERVAL_SECONDS = 60.0


class RepairProgress(ScanReport, Protocol):
    """What the repair code reports; every call only updates memory."""

    def begin(self, trigger_kind: TriggerKind, attempt_no: int) -> None: ...

    def phase(self, phase: Phase) -> None: ...

    def last_line(self) -> str | None:
        """The last published progress as one sentence, or None when none was."""
        ...

    def end(self) -> None: ...


class _NoProgress:
    def begin(self, trigger_kind: TriggerKind, attempt_no: int) -> None:
        pass

    def phase(self, phase: Phase) -> None:
        pass

    def tables_planned(self, count: int) -> None:
        pass

    def table_started(self, name: str) -> None:
        pass

    def table_finished(self, name: str, stats: TableSortStats) -> None:
        pass

    def last_line(self) -> str | None:
        return None

    def end(self) -> None:
        pass


NO_PROGRESS: RepairProgress = _NoProgress()


class ProgressPublisher:
    """The only writer of `repair-progress.json`, owned by the daemon that holds the gate."""

    def __init__(
        self,
        data_dir: Path,
        *,
        clock: Callable[[], float] = time.time,
        interval_seconds: float = HEARTBEAT_SECONDS,
    ) -> None:
        self._data_dir = data_dir
        self._clock = clock
        self._interval = interval_seconds
        # Anything already here belongs to an owner that is gone: this process holds the gate.
        remove_progress(data_dir)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._first_write = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._record: ProgressRecord | None = None
        self._last_warning = float("-inf")
        self._last_logged = float("-inf")

    def begin(self, trigger_kind: TriggerKind, attempt_no: int) -> None:
        now = self._clock()
        with self._lock:
            assert self._record is None and not self._stopped, "a repair begins once"
            self._record = ProgressRecord(
                schema=1,
                owner_pid=os.getpid(),
                attempt_no=attempt_no,
                trigger_kind=trigger_kind,
                phase="admitting",
                started_at=now,
                phase_started_at=now,
                updated_at=now,
            )
        self._thread = threading.Thread(target=self._run, name="repair-progress", daemon=True)
        self._thread.start()
        # The file exists before any capture starts whenever the filesystem allows it.
        self._first_write.wait(FIRST_WRITE_WAIT_SECONDS)

    def phase(self, phase: Phase) -> None:
        now = self._clock()
        with self._lock:
            if self._record is None or self._record.phase == phase:
                return
            self._record = self._record.model_copy(
                update={"phase": phase, "phase_started_at": now, "progress": None}
            )
            message = describe_progress(self._record, now=now)
        logger.info("automatic index repair: %s", message)
        self._wake.set()

    def tables_planned(self, count: int) -> None:
        self._update_unit(lambda unit: unit.model_copy(update={"total": unit.total + count}))

    def table_started(self, name: str) -> None:
        self._update_unit(lambda unit: unit.model_copy(update={"current": name}))

    def table_finished(self, name: str, stats: TableSortStats) -> None:
        self._update_unit(
            lambda unit: unit.model_copy(
                update={
                    "done": unit.done + 1,
                    "current": None,
                    "bytes_done": unit.bytes_done + stats.stream_bytes,
                }
            )
        )

    def _update_unit(self, change: Callable[[ProgressUnit], ProgressUnit]) -> None:
        with self._lock:
            if self._record is None:
                return
            unit = self._record.progress or ProgressUnit(done=0, total=0, bytes_done=0)
            self._record = self._record.model_copy(update={"progress": change(unit)})

    def last_line(self) -> str | None:
        with self._lock:
            record = self._record
        if record is None:
            return None
        return f"last repair progress: {describe_progress(record, now=self._clock())}"

    def end(self) -> None:
        """Stop the writer, then remove the file: no replace can follow the removal."""
        with self._lock:
            self._stopped = True
        self._wake.set()
        if self._thread is not None:
            # A write still in flight is abandoned; the stopped flag forbids its replace.
            self._thread.join(JOIN_SECONDS_MAX)
        remove_progress(self._data_dir)

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._stopped or self._record is None:
                    return
                now = self._clock()
                record = self._record.model_copy(update={"updated_at": now})
            try:
                self._write(record)
            except OSError as error:
                self._warn(error)
            self._first_write.set()
            self._log_heartbeat(record)
            self._wake.wait(self._interval)
            self._wake.clear()

    def _write(self, record: ProgressRecord) -> None:
        temporary = self._data_dir / PROGRESS_TEMP_FILENAME
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(record.render())
        with self._lock:
            if self._stopped:
                return
            os.replace(temporary, self._data_dir / PROGRESS_FILENAME)

    def _warn(self, error: OSError) -> None:
        now = self._clock()
        if now - self._last_warning >= WARNING_INTERVAL_SECONDS:
            self._last_warning = now
            logger.warning("could not publish repair progress: %s", error)

    def _log_heartbeat(self, record: ProgressRecord) -> None:
        now = self._clock()
        if now - self._last_logged >= WARNING_INTERVAL_SECONDS:
            self._last_logged = now
            logger.info("automatic index repair: %s", describe_progress(record, now=now))


def _unit(record: ProgressRecord) -> RecoveryProgress | None:
    unit = record.progress
    if unit is None:
        return None
    return {
        "unit": unit.unit,
        "done": unit.done,
        "total": unit.total,
        "current": unit.current,
        "bytes_done": unit.bytes_done,
    }


def _view(
    record: ProgressRecord,
    *,
    state: RecoveryState,
    message: str,
    now: float,
    durable_phase: str | None,
) -> RecoveryStatus:
    return {
        "state": state,
        "phase": record.phase,
        "progress": _unit(record),
        "elapsed_seconds": max(0, round(now - record.started_at)),
        "updated_age_seconds": max(0, round(now - record.updated_at)),
        "recorded_by_pid": record.owner_pid,
        "durable_phase": durable_phase,
        "message": message,
    }


_UNAVAILABLE = "data requests are unavailable until the repair finishes"
_NEXT_START = {
    "idle": "the next start recaptures the reference",
    "required": "the next start resumes from the saved reference",
    "running": (
        "the repair was armed; the next start refuses and names "
        "`recall db rebuild-indexes` (run it offline)"
    ),
}


def recovery_for_busy_gate(data_dir: Path, *, now: float) -> RecoveryStatus | None:
    """Rows 3 to 5: a live owner holds the gate; only its fresh or stale record can say more."""
    record = read_progress(data_dir)
    if record is None:
        return None
    detail = describe_progress(record, now=now)
    if now - record.updated_at <= FRESH_SECONDS:
        message = f"automatic index repair in progress ({detail}); {_UNAVAILABLE}"
        return _view(record, state="running", message=message, now=now, durable_phase=None)
    age = max(0, round(now - record.updated_at))
    message = (
        f"automatic index repair may be stalled: no update for {age} s; "
        f"the owner may be hung or busy ({detail}); {_UNAVAILABLE}"
    )
    return _view(record, state="stalled", message=message, now=now, durable_phase=None)


def recovery_for_free_gate(
    data_dir: Path, *, durable_phase: str | None, now: float
) -> RecoveryStatus | None:
    """Rows 7 to 10: nothing owns the database; the durable phase says what the next start does."""
    record = read_progress(data_dir)
    if record is None:
        return None
    recorded = f"recorded by pid {record.owner_pid}"
    if durable_phase in {"verified", "refused"}:
        message = "recorded progress belongs to a repair that already finished; start normally"
        return _view(record, state="stale", message=message, now=now, durable_phase=durable_phase)
    if durable_phase is None:
        message = (
            f"an automatic index repair was interrupted ({recorded}); "
            "the durable repair state could not be read"
        )
    elif durable_phase == "running":
        message = (
            f"an automatic index repair was interrupted ({recorded}); {_NEXT_START['running']}"
        )
    else:
        message = (
            f"an automatic index repair was interrupted ({recorded}); no process owns the "
            f"database; {_NEXT_START[durable_phase]}"
        )
    return _view(record, state="interrupted", message=message, now=now, durable_phase=durable_phase)


def repair_in_progress_notice(config: AppConfig) -> str | None:
    """The refusal text for a command that needs data or a second daemon, or None.

    Applies only while a live owner holds the gate and a record is readable. A
    refusal marker wins: the commands then already say why the daemon is down.
    """
    from recall.services.self_repair import read_refusal_marker

    # Cheapest first: with no record there is nothing to say, and nothing is touched.
    view = recovery_for_busy_gate(config.data_dir, now=time.time())
    if view is None or read_refusal_marker(config.data_dir) is not None:
        return None
    if not ownership_busy(config.db_path):
        return None
    return f"{view['message']}; see `recall daemon status`"
