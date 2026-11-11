"""Immutable failure diagnostics and the exclusive durable startup driver.

An invalidated instance cannot record its own failure in DuckDB. Files retain
the instruction for the next owner's admission. The durable recovery protocol
owns the single automatic attempt, exact verification and instruction retirement;
timestamps remain diagnostic history. An interrupted armed attempt requires
verified offline manual repair or matched restore.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast, get_args

import duckdb

from recall.core.config import AppConfig
from recall.core.repair_state import PublicationRole
from recall.core.time import to_naive_utc, utcnow_naive
from recall.db.fatal import (
    is_disk_full_error,
    is_index_divergence_error,
)
from recall.db.lease import DbLease, StandaloneLifetime
from recall.db.maintenance import (
    PROBE_BUDGET_SECONDS_DEFAULT,
    PROBE_SAMPLE_DEFAULT,
    IndexDivergenceReport,
    IndexRebuildResult,
    probe_index_divergence,
)
from recall.db.repair import SchemaFence
from recall.services.runtime_state import (
    clear_fatal_memory_columns,
    set_needs_index_verification,
)

if TYPE_CHECKING:
    from recall.services.recovery_admission import RecoveryAdmission

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


def write_failure_marker(
    data_dir: Path,
    marker: FailureMarker,
    *,
    role: PublicationRole = PublicationRole.PRIMARY,
    fence: SchemaFence | None = None,
) -> bool:
    """Publish an immutable instruction; later events use separate spool files.

    EIO/ENOSPC may prevent publication; shutdown must proceed regardless.
    """
    from recall.services.repair_events import (
        EventFailure,
        EventStorageError,
        LegacyEvent,
        RecoveryEvent,
        publish_event,
    )

    failure = (
        EventFailure.model_validate(asdict(marker.failure)) if marker.failure is not None else None
    )
    if fence is None:
        event = LegacyEvent(
            failure=failure, needs_index_verification=marker.needs_index_verification
        )
    else:
        from uuid import uuid4

        event = RecoveryEvent(
            event_id=uuid4(),
            failure=failure,
            needs_index_verification=marker.needs_index_verification,
        )
    try:
        publish_event(data_dir, event, role=role)
    except (OSError, EventStorageError) as err:
        logger.warning("could not publish failure instruction: %s", err)
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
        from recall.services.repair_events import read_events

        retained_verification = any(
            record.needs_index_verification for _, record in read_events(data_dir).secondary
        )
        return FailureMarker(
            failure=failure,
            needs_index_verification=(
                bool(payload.get("needs_index_verification", False)) or retained_verification
            ),
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
    role: PublicationRole = PublicationRole.PRIMARY,
    fence: SchemaFence | None = None,
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
        if not write_failure_marker(data_dir, marker, role=role, fence=fence):
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
        written = write_failure_marker(
            data_dir, FailureMarker(failure=None, needs_index_verification=True)
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


def preflight_recovery_files(config: AppConfig) -> None:
    """Hold before any database opener can recover WAL or run migrations."""
    from recall.services.repair_events import EventStorageError, read_events

    try:
        batch = read_events(config.data_dir)
        if batch.full:
            raise EventStorageError("failure-event spool is full")
    except EventStorageError as err:
        message = (
            f"refusing to start: {err}; correct storage capacity and use matched restore, "
            "including acknowledged post-backup writes. Never use old confirmed recreation."
        )
        write_refusal_marker(config.data_dir, message)
        raise DaemonStartupRefused(message) from err
    try:
        refusal = refusal_marker_path(config.data_dir).read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    except (OSError, UnicodeError) as err:
        raise DaemonStartupRefused(
            "refusing to start: unreadable refusal; use manual repair or matched restore"
        ) from err
    raise DaemonStartupRefused(
        refusal.strip() or "refusing to start: retained refusal requires manual repair"
    )


class DaemonStartupRefused(RuntimeError):
    """The previous run's failure recurred after a repair; do not serve again."""

    exit_code = REFUSED_EXIT_CODE


@dataclass(frozen=True)
class StartupRepairOutcome:
    action: StartupAction
    record: FailureRecord | None
    probe: IndexDivergenceReport | None
    rebuild: IndexRebuildResult | None


class StartupDatabase(Protocol):
    def run[T](self, operation: str, work: Callable[[DbLease], T]) -> T: ...

    def reopen(self, *, opener: Callable[[], duckdb.DuckDBPyConnection] | None = None) -> None: ...

    def close(self) -> None: ...


class _StandaloneStartupDatabase:
    """A synchronous startup tool owns exactly one private parent and no workers."""

    def __init__(self, admission: RecoveryAdmission, lifetime: StandaloneLifetime) -> None:
        self._admission = admission
        self._lifetime = lifetime
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._active = True

    def run[T](self, operation: str, work: Callable[[DbLease], T]) -> T:
        assert self._active, "standalone startup scope expired"
        if self._connection is None:
            self._connection = self._admission.open()
        with self._lifetime.lease(self._connection) as lease:
            return lease.call(operation, lambda _: work(lease))

    def reopen(self, *, opener: Callable[[], duckdb.DuckDBPyConnection] | None = None) -> None:
        assert self._active, "standalone startup scope expired"
        self._close_parent()
        self._connection = (opener or self._admission.open)()

    def _close_parent(self) -> None:
        if self._connection is not None:
            conn, self._connection = self._connection, None
            self._lifetime.cleanup_db("standalone_startup:close", conn.close)

    def close(self) -> None:
        self._active = False
        self._close_parent()


def run_startup_self_repair(
    config: AppConfig,
    *,
    now: datetime | None = None,
    probe_sample: int = PROBE_SAMPLE_DEFAULT,
    probe_budget_seconds: float = PROBE_BUDGET_SECONDS_DEFAULT,
) -> StartupRepairOutcome:
    """Standalone startup tools own their complete private connection lifetime."""
    from recall.services.recovery_admission import RecoveryAdmission

    lifetime = StandaloneLifetime()
    admission = RecoveryAdmission(config, lifetime)
    database = _StandaloneStartupDatabase(admission, lifetime)
    try:
        result = run_startup_self_repair_on(
            config,
            database,
            admission=admission,
            now=now,
            probe_sample=probe_sample,
            probe_budget_seconds=probe_budget_seconds,
        )
    except BaseException as primary:
        lifetime.observe(primary, "standalone_startup")
        try:
            database.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup("startup and close failed", [primary, cleanup]) from None
        raise
    else:
        database.close()
        return result


def run_startup_self_repair_on(
    config: AppConfig,
    database: StartupDatabase,
    *,
    admission: RecoveryAdmission,
    now: datetime | None = None,
    probe_sample: int = PROBE_SAMPLE_DEFAULT,
    probe_budget_seconds: float = PROBE_BUDGET_SECONDS_DEFAULT,
) -> StartupRepairOutcome:
    """Execute the durable protocol inside the caller's exclusive lifetime."""
    from recall.services.repair_protocol import (
        AutomaticStartup,
        RefusalCommitUncertain,
        RepairProtocol,
    )

    moment = to_naive_utc(now) if now is not None else None
    clock = (lambda: moment) if moment is not None else utcnow_naive

    def probe(lease: DbLease) -> IndexDivergenceReport:
        return lease.call(
            "startup_probe",
            lambda conn: probe_index_divergence(
                conn,
                sample=probe_sample,
                budget_seconds=probe_budget_seconds,
                now=clock,
            ),
        )

    protocol = RepairProtocol(config.data_dir, probe=probe, clock=clock)
    try:
        preflight_recovery_files(config)
        start = database.run("startup_admission", protocol.begin_start)
        rebuild = None
        if isinstance(start, AutomaticStartup):
            arm = database.run("startup_arm", protocol.arm)
            rebuilt = database.run("startup_rebuild", lambda lease: protocol.rebuild(lease, arm))
            database.reopen(opener=lambda: admission.reopen(protocol, rebuilt))
            database.run("startup_verify", lambda lease: protocol.verify(lease, rebuilt))
            database.run("startup_retire", protocol.retire)
            rebuild = rebuilt.result
            checked = protocol.last_probe
            assert checked is not None, "verified repair lacks its fresh probe"
        else:
            checked = start.probe
    except RefusalCommitUncertain as error:
        try:
            database.close()
            row = admission.read_retained_state()
        except Exception as resolution_error:
            message = f"refusal commit remains uncertain: {resolution_error}"
            logger.error(message, exc_info=True)
        else:
            message = f"refusal commit fresh read: phase={row.phase} attempt={row.attempt_no}"
            logger.error(message)
        message = (
            f"refusing to start: {message}; retain all recovery files and run "
            "`recall db rebuild-indexes` offline or use matched restore"
        )
        write_refusal_marker(config.data_dir, message)
        raise DaemonStartupRefused(message) from error
    except DaemonStartupRefused:
        raise
    except Exception as error:
        message = (
            f"refusing to start: startup recovery failed: {error}; "
            "retain the recovery files and run `recall db rebuild-indexes` offline "
            "for verified repair or use matched restore"
        )
        write_refusal_marker(config.data_dir, message)
        raise DaemonStartupRefused(message) from error
    records = start.records
    failure = next(
        (record.failure for record in records if record.marker.value == "index"),
        next((record.failure for record in records if record.failure is not None), None),
    )
    record = FailureRecord(**failure.model_dump()) if failure is not None else None
    action: StartupAction = "rebuilt" if rebuild is not None else "recorded" if record else "none"
    logger.info(
        "startup recovery action=%s probe_diverged=%d probe_checked=%d probe_complete=%s",
        action,
        checked.diverged_count,
        checked.samples_checked,
        checked.complete,
    )
    return StartupRepairOutcome(action=action, record=record, probe=checked, rebuild=rebuild)


def clear_fatal_memory(config: AppConfig, conn: duckdb.DuckDBPyConnection) -> None:
    """Forget the marker and runtime memory after an authorized offline repair."""
    clear_failure_marker(config.data_dir)
    clear_fatal_memory_columns(conn)
