"""Failure-signature memory and startup self-repair for the daemon.

A fatally invalidated DuckDB instance rejects every statement, so the run that
dies cannot record why in `runtime_state`. It spools a small JSON marker beside
the database instead (REQ-RESIL-014); the next start reads it before any other
DuckDB work, folds it into `runtime_state`, repairs the one condition with a
known one-second fix -- table/index divergence, healed by rebuilding every
index (REQ-RESIL-015) -- and refuses to start when the same signature recurs
after a repair, so a scheduler's KeepAlive is not handed an identical failure
1,679 times (REQ-RESIL-016). ENOSPC anywhere on the shared connection flags the
database for verification on the next open (REQ-RESIL-019).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast, get_args

import duckdb

from recall.core.config import AppConfig
from recall.core.time import to_naive_utc, utcnow_naive
from recall.db.connection import connect
from recall.db.fatal import (
    is_disk_full_error,
    is_fatal_db_invalidation,
    is_index_divergence_error,
)
from recall.db.maintenance import (
    PROBE_BUDGET_SECONDS_DEFAULT,
    PROBE_SAMPLE_DEFAULT,
    IndexDivergenceReport,
    IndexRebuildResult,
    probe_index_divergence,
    rebuild_indexes,
)
from recall.services.runtime_state import (
    RuntimeStatus,
    clear_fatal_memory_columns,
    load_runtime_status_from_conn,
    record_fatal_failure,
    record_index_repair,
    set_needs_index_verification,
)

logger = logging.getLogger("recall.self_repair")

MARKER_FILENAME = "daemon-failure.json"
# Exists only after a refused start. launchd cannot filter on exit status, so
# its KeepAlive is conditioned on this path being absent (REQ-RESIL-024).
REFUSAL_FILENAME = "daemon-refused"
MESSAGE_MAX_CHARS = 500
REFUSED_EXIT_CODE = 3

FailureClass = Literal["index-divergence", "disk-full", "other"]
StartupAction = Literal["none", "recorded", "rebuilt"]

_REPAIRABLE_CLASSES: frozenset[str] = frozenset({"index-divergence", "disk-full"})
# Order matters: quoted strings (paths) go first so their contents never leak
# into the id/number passes; hex runs need a digit so ordinary words such as
# "decade" survive.
_QUOTED = re.compile(r'"[^"]*"')
_HEX_RUN = re.compile(r"\b(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{6,}\b")
_NUMBER = re.compile(r"\b\d+\b")


# ---- classification and signatures ----


def classify_failure(err: BaseException) -> FailureClass:
    if is_index_divergence_error(err):
        return "index-divergence"
    if is_disk_full_error(err):
        return "disk-full"
    return "other"


def _first_line(text: str) -> str:
    lines = text.splitlines()
    return lines[0] if lines else ""


def _normalize_message(message: str) -> str:
    text = _QUOTED.sub('"<str>"', message)
    text = _HEX_RUN.sub("<id>", text)
    text = _NUMBER.sub("<n>", text)
    return " ".join(text.split())


def normalize_failure_signature(site: str, err: BaseException) -> str:
    """`site:ExceptionClass:message` with ids, numbers, and quoted strings elided.

    Two runs that die on different rows of the same fault compare equal; the
    site and exception class keep distinct faults apart.
    """
    return f"{site}:{type(err).__name__}:{_normalize_message(_first_line(str(err)))}"


@dataclass(frozen=True)
class FailureRecord:
    site: str
    exception_type: str
    message: str
    signature: str
    failure_class: FailureClass
    at: datetime


def build_failure_record(
    err: BaseException, *, site: str, now: datetime | None = None
) -> FailureRecord:
    at = to_naive_utc(now) if now is not None else utcnow_naive()
    return FailureRecord(
        site=site,
        exception_type=type(err).__name__,
        message=_first_line(str(err))[:MESSAGE_MAX_CHARS],
        signature=normalize_failure_signature(site, err),
        failure_class=classify_failure(err),
        at=at,
    )


# ---- marker file ----


@dataclass(frozen=True)
class FailureMarker:
    failure: FailureRecord | None
    needs_index_verification: bool


def marker_path(data_dir: Path) -> Path:
    return Path(data_dir) / MARKER_FILENAME


def write_failure_marker(data_dir: Path, marker: FailureMarker) -> bool:
    """Atomically write the marker; return False (never raise) when it cannot be written.

    Temp file + `os.replace` so a reader sees the old record or the new one,
    never a partial write (INV-RESIL-007). Best-effort by design: on a full disk
    the write itself fails, and shutdown must proceed regardless.
    """
    path = marker_path(data_dir)
    failure_payload = None
    if marker.failure is not None:
        failure_payload = asdict(marker.failure)
        failure_payload["at"] = marker.failure.at.isoformat()
    payload = {
        "version": 1,
        "failure": failure_payload,
        "needs_index_verification": marker.needs_index_verification,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".daemon-failure-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            with suppress(OSError):
                os.unlink(tmp_name)
            raise
    except OSError as err:
        logger.warning("could not write failure marker %s: %s", path, err)
        return False
    return True


def _failure_class(value: object) -> FailureClass:
    text = str(value)
    if text not in get_args(FailureClass):
        raise ValueError(f"unknown failure class {text!r}")
    return cast(FailureClass, text)


def read_failure_marker(data_dir: Path) -> FailureMarker | None:
    """Return the marker, or None when absent or unreadable (logged)."""
    path = marker_path(data_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as err:
        logger.warning("could not read failure marker %s: %s", path, err)
        return None
    try:
        payload = json.loads(text)
        raw_failure = payload.get("failure")
        failure = None
        if raw_failure is not None:
            failure = FailureRecord(
                site=str(raw_failure["site"]),
                exception_type=str(raw_failure["exception_type"]),
                message=str(raw_failure["message"]),
                signature=str(raw_failure["signature"]),
                failure_class=_failure_class(raw_failure["failure_class"]),
                at=datetime.fromisoformat(str(raw_failure["at"])),
            )
        return FailureMarker(
            failure=failure,
            needs_index_verification=bool(payload.get("needs_index_verification", False)),
        )
    except (ValueError, KeyError, TypeError, AttributeError) as err:
        logger.warning("ignoring unreadable failure marker %s: %s", path, err)
        return None


def refusal_marker_path(data_dir: Path) -> Path:
    return Path(data_dir) / REFUSAL_FILENAME


def write_refusal_marker(data_dir: Path, message: str) -> bool:
    """Leave the refusal marker; return False (never raise) when it cannot be written.

    The scheduler keys on the path, not the contents; the message is for a
    human who finds the file before `daemon status`.
    """
    path = refusal_marker_path(data_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(message + "\n", encoding="utf-8")
    except OSError as err:
        logger.warning("could not write refusal marker %s: %s", path, err)
        return False
    return True


def read_refusal_marker(data_dir: Path) -> str | None:
    """The refusal message left by the last refused start, or None."""
    try:
        text = refusal_marker_path(data_dir).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as err:
        logger.warning("could not read refusal marker: %s", err)
        return None
    return text.strip() or None


def clear_refusal_marker(data_dir: Path) -> None:
    """Release the scheduler: a start that serves, or a manual repair, ends the refusal."""
    _unlink_marker(refusal_marker_path(data_dir))


def clear_failure_marker(data_dir: Path) -> None:
    """Remove the failure marker and, with it, the refusal marker.

    A refusal is only ever about the failure it refused on, so clearing the
    memory releases both: that is what lets the scheduler relaunch the daemon
    (REQ-RESIL-024).
    """
    _unlink_marker(marker_path(data_dir))
    clear_refusal_marker(data_dir)


def _unlink_marker(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as err:
        logger.warning("could not remove marker %s: %s", path, err)


def remember_fatal_failure(
    data_dir: Path,
    err: BaseException,
    *,
    site: str,
    now: datetime | None = None,
) -> FailureRecord | None:
    """Spool a fatal failure for the next start. Never raises (REQ-RESIL-020)."""
    try:
        record = build_failure_record(err, site=site, now=now)
        existing = read_failure_marker(data_dir)
        flagged = existing is not None and existing.needs_index_verification
        marker = FailureMarker(
            failure=record,
            needs_index_verification=flagged or record.failure_class == "disk-full",
        )
        if not write_failure_marker(data_dir, marker):
            return None
        return record
    except Exception:
        logger.warning("could not remember fatal failure at site=%s", site, exc_info=True)
        return None


def remember_disk_full(data_dir: Path, conn: duckdb.DuckDBPyConnection | None = None) -> bool:
    """Flag the database for index verification after ENOSPC (REQ-RESIL-019).

    Writes the marker flag (preserving any spooled failure) and, when the
    instance is still alive, the `runtime_state` column too. Never raises.
    """
    written = False
    try:
        existing = read_failure_marker(data_dir)
        failure = existing.failure if existing is not None else None
        written = write_failure_marker(
            data_dir, FailureMarker(failure=failure, needs_index_verification=True)
        )
    except Exception:
        logger.warning("could not flag index verification in marker", exc_info=True)
    if conn is not None:
        try:
            set_needs_index_verification(conn, True)
        except Exception as db_err:
            logger.warning("could not persist needs_index_verification: %s", db_err)
    return written


# ---- startup self-repair ----


class DaemonStartupRefused(RuntimeError):
    """The previous run's failure recurred after a repair; do not serve again."""

    exit_code = REFUSED_EXIT_CODE


@dataclass(frozen=True)
class StartupRepairOutcome:
    action: StartupAction
    record: FailureRecord | None
    probe: IndexDivergenceReport | None
    rebuild: IndexRebuildResult | None


def run_startup_self_repair(
    config: AppConfig,
    *,
    now: datetime | None = None,
    probe_sample: int = PROBE_SAMPLE_DEFAULT,
    probe_budget_seconds: float = PROBE_BUDGET_SECONDS_DEFAULT,
) -> StartupRepairOutcome:
    """Standalone form of `run_startup_self_repair_on` on a short-lived connection.

    For tooling and tests that have no daemon connection of their own; the
    daemon itself passes its shared connection so the step cannot race it.
    """
    conn = connect(config, lenient_schema=True)
    try:
        return run_startup_self_repair_on(
            config,
            conn,
            now=now,
            probe_sample=probe_sample,
            probe_budget_seconds=probe_budget_seconds,
        )
    finally:
        conn.close()


def run_startup_self_repair_on(
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    *,
    now: datetime | None = None,
    probe_sample: int = PROBE_SAMPLE_DEFAULT,
    probe_budget_seconds: float = PROBE_BUDGET_SECONDS_DEFAULT,
) -> StartupRepairOutcome:
    """Consume the failure marker, repair or refuse, and cache a divergence probe.

    Runs on `conn` -- the daemon's shared connection, held exclusively while the
    step runs, so a request arriving mid-repair waits on it instead of opening a
    second connection whose schema bootstrap races this one. Raises
    `DaemonStartupRefused` when the same repairable signature recurred after a
    rebuild; every other problem inside the step is the caller's to log.
    """
    moment = to_naive_utc(now) if now is not None else utcnow_naive()
    marker = read_failure_marker(config.data_dir)
    return _self_repair_on_conn(
        config,
        conn,
        marker,
        moment,
        probe_sample=probe_sample,
        probe_budget_seconds=probe_budget_seconds,
    )


def clear_fatal_memory(config: AppConfig, conn: duckdb.DuckDBPyConnection) -> None:
    """Forget the marker and the runtime_state memory after a manual repair."""
    clear_failure_marker(config.data_dir)
    clear_fatal_memory_columns(conn)


def _self_repair_on_conn(
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    marker: FailureMarker | None,
    moment: datetime,
    *,
    probe_sample: int,
    probe_budget_seconds: float,
) -> StartupRepairOutcome:
    record = marker.failure if marker is not None else None
    status = load_runtime_status_from_conn(conn)
    repeat_count = _repeat_count(status, record)

    if record is not None and _repair_already_failed(status, record):
        message = _refusal_message(record, config, repeat_count)
        _persist_failure(conn, record, message, repeat_count)
        write_refusal_marker(config.data_dir, message)
        logger.error("%s", message)
        raise DaemonStartupRefused(message)

    needs_verification = (
        marker is not None and marker.needs_index_verification
    ) or status.needs_index_verification

    # REQ-RESIL-014: the record is the first DB write. Whatever the rebuild or
    # the probes do next, the failure is already visible in runtime_state; the
    # marker is released only once that write committed, so a failed write
    # leaves it for the next start instead of silently forgetting the fatal.
    if marker is not None:
        _fold_marker_into_runtime_state(config, conn, marker, repeat_count)

    signature = record.signature if record is not None else None
    rebuild: IndexRebuildResult | None = None
    if record is not None and record.failure_class == "index-divergence":
        logger.warning(
            "previous run died on index/table divergence at site=%s; rebuilding indexes",
            record.site,
        )
        rebuild = _rebuild_armed(config, conn, signature, moment)

    probe = probe_index_divergence(
        conn, sample=probe_sample, budget_seconds=probe_budget_seconds, now=lambda: moment
    )
    if rebuild is None and probe.diverged:
        logger.warning(
            "startup probe found %d diverged key(s); rebuilding indexes", probe.diverged_count
        )
        rebuild = _rebuild_armed(config, conn, signature, moment)
        probe = probe_index_divergence(
            conn, sample=probe_sample, budget_seconds=probe_budget_seconds, now=lambda: moment
        )

    action: StartupAction = "none"
    if rebuild is not None:
        action = "rebuilt"
    elif record is not None:
        action = "recorded"
    if needs_verification:
        _persist(conn, lambda: set_needs_index_verification(conn, False))

    # Reaching here means this start serves, so any refusal is over: an
    # orphaned marker (e.g. from an unarmed refusal whose failure marker the
    # fold had already consumed) would otherwise hold launchd down for good.
    clear_refusal_marker(config.data_dir)
    logger.info(
        "startup self-repair action=%s class=%s site=%s repeat=%d verified=%s "
        "probe_diverged=%d probe_checked=%d probe_complete=%s",
        action,
        record.failure_class if record is not None else "-",
        record.site if record is not None else "-",
        repeat_count if record is not None else 0,
        needs_verification,
        probe.diverged_count,
        probe.samples_checked,
        probe.complete,
    )
    return StartupRepairOutcome(action=action, record=record, probe=probe, rebuild=rebuild)


def _rebuild_armed(
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    signature: str | None,
    moment: datetime,
) -> IndexRebuildResult:
    """Record the repair attempt, then rebuild; refuse when it cannot be recorded.

    The refusal of REQ-RESIL-016 keys on `last_index_repair_signature`. Arming
    it before the attempt makes a rebuild that raises count as a repair that
    did not hold; recorded afterwards, a raising rebuild was retried on every
    start. An attempt that cannot be armed is not run for the same reason: it
    would be retried on every start, so the start refuses with the manual fix.

    A schema that lacks the memory columns is the one exception: the lenient
    open let it through so the operator can `recall index --recreate`, which
    needs the daemon listening, so the start rebuilds unarmed and warns instead.
    """
    missing = _missing_repair_memory_columns(conn)
    if missing:
        logger.warning(
            "runtime_state lacks %s (schema predates migration 0023), so this index rebuild "
            "cannot be armed and a recurrence will rebuild again on every start; run "
            "`recall index --recreate --yes` while the daemon is up",
            ", ".join(missing),
        )
        return rebuild_indexes(conn)
    try:
        record_index_repair(conn, signature=signature, repaired_at=moment)
    except duckdb.Error as err:
        if is_fatal_db_invalidation(err):
            # A dead instance is the funnel's case (REQ-RESIL-015), not a
            # refusal: relabelling it would hold the scheduler down on a
            # database that a relaunch could still repair.
            raise
        message = _unarmed_refusal_message(err)
        write_refusal_marker(config.data_dir, message)
        logger.error("%s", message)
        raise DaemonStartupRefused(message) from err
    return rebuild_indexes(conn)


_REPAIR_MEMORY_COLUMNS: tuple[str, ...] = ("last_index_repair_at", "last_index_repair_signature")


def _missing_repair_memory_columns(conn: duckdb.DuckDBPyConnection) -> tuple[str, ...]:
    """Name the REQ-RESIL-016 memory columns absent from `runtime_state`.

    An unreadable catalog reports nothing missing: the arming write then
    decides, and its failure is a refusal rather than a silent unarmed rebuild.
    """
    try:
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = 'runtime_state'"
        ).fetchall()
    except duckdb.Error as err:
        if is_fatal_db_invalidation(err):
            raise
        logger.warning("could not read the runtime_state columns: %s", err)
        return ()
    present = {row[0] for row in rows}
    return tuple(column for column in _REPAIR_MEMORY_COLUMNS if column not in present)


def _fold_marker_into_runtime_state(
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    marker: FailureMarker,
    repeat_count: int,
) -> bool:
    """Persist the spooled failure (and the disk-full flag) before any maintenance.

    Returns True when every write committed and the marker was released; False
    keeps the marker on disk so the next start folds it again (REQ-RESIL-014).
    """
    persisted = True
    if marker.failure is not None:
        record = marker.failure
        message = (
            f"fatal DuckDB invalidation in {record.site}: {record.exception_type}: {record.message}"
        )
        persisted = _persist_failure(conn, record, message, repeat_count)
    if marker.needs_index_verification:
        persisted = _persist(conn, lambda: set_needs_index_verification(conn, True)) and persisted
    if not persisted:
        logger.warning(
            "failure marker %s not folded into runtime_state; keeping it for the next start",
            marker_path(config.data_dir),
        )
        return False
    clear_failure_marker(config.data_dir)
    return True


def _repeat_count(status: RuntimeStatus, record: FailureRecord | None) -> int:
    if record is None:
        return status.fatal_repeat_count
    same_signature = status.last_fatal_signature == record.signature
    healthy_since_last_failure = status.last_failure_at is None or (
        status.last_successful_at is not None and status.last_successful_at > status.last_failure_at
    )
    if same_signature and not healthy_since_last_failure:
        return status.fatal_repeat_count + 1
    return 1


def _repair_already_failed(status: RuntimeStatus, record: FailureRecord) -> bool:
    """True when this signature already got a rebuild and nothing succeeded since."""
    if record.failure_class not in _REPAIRABLE_CLASSES:
        return False
    if status.last_index_repair_signature != record.signature:
        return False
    if status.last_index_repair_at is None or record.at < status.last_index_repair_at:
        return False
    success_since_repair = (
        status.last_successful_at is not None
        and status.last_successful_at > status.last_index_repair_at
    )
    return not success_since_repair


# `recall index --recreate` is deliberately not named: it is served by the
# daemon, whose auto-forked start refuses again while the memory persists.
_MANUAL_FIX = (
    "Manual fix: run `recall daemon stop`, then `recall db rebuild-indexes`; if the failure "
    "persists, `recall compact --yes`."
)


def _unarmed_refusal_message(err: duckdb.Error) -> str:
    return (
        "refusing to start: the index repair attempt could not be recorded in runtime_state "
        f"({type(err).__name__}: {_first_line(str(err))}), and an unrecorded repair would be "
        f"retried on every start. {_MANUAL_FIX}"
    )


def _refusal_message(record: FailureRecord, config: AppConfig, repeat_count: int) -> str:
    if record.failure_class == "disk-full":
        cause = (
            f"the volume holding {config.data_dir} ran out of space again after an index rebuild"
        )
    else:
        cause = "DuckDB index/table divergence recurred after an index rebuild"
    return (
        f"refusing to start: {cause} (site={record.site}, repeat={repeat_count}, "
        f"last error: {record.message}). {_MANUAL_FIX}"
    )


def _persist_failure(
    conn: duckdb.DuckDBPyConnection, record: FailureRecord, message: str, repeat_count: int
) -> bool:
    return _persist(
        conn,
        lambda: record_fatal_failure(
            conn,
            message=message,
            signature=record.signature,
            failed_at=record.at,
            repeat_count=repeat_count,
        ),
    )


def _persist(conn: duckdb.DuckDBPyConnection, write: Callable[[], object]) -> bool:
    """Run a runtime_state write, tolerating a stale schema that lacks the columns.

    Returns False when the write failed, so callers can keep the marker.
    """
    try:
        write()
    except duckdb.Error as err:
        if is_fatal_db_invalidation(err):
            # Tolerating a stale schema is the point; tolerating a dead
            # instance would run the rebuild on a corpse (INV-RESIL-004).
            raise
        logger.warning("could not persist failure memory to runtime_state: %s", err)
        return False
    return True
