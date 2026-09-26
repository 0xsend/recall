from __future__ import annotations

import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, cast

import duckdb

from recall.core.config import AppConfig, create_private_dir
from recall.db.schema import ensure_schema, ensure_schema_lenient

# DuckDB buffers do not bound total RSS. Leave headroom for captures, vectors
# and native allocations independently of host RAM; explicit overrides remain.
_DEFAULT_MEMORY_LIMIT = "2GB"
# Fixed on-disk target selected by the populated upgrade benchmark. Existing
# files retain their header on ordinary opens; maintenance owns conversion.
STORAGE_VERSION = "v1.2.0"
_STORAGE_VERSION_RE = re.compile(r"v(\d+)\.(\d+)\.(\d+)\+?")
# DuckDB's 16 MiB default makes a reconciliation commit also checkpoint the
# accumulated WAL. On a populated multi-gigabyte database that folds seconds of
# unrelated checkpoint work into one source's writer turn. The WAL remains
# durable and bounded; normal auto-checkpointing continues at this larger cap.
_WAL_AUTOCHECKPOINT = "256MiB"
WAL_CHECKPOINT_HIGH_WATER_BYTES = 64 * 1024 * 1024
_MEMORY_LIMIT_RE = re.compile(
    r"^(?P<number>-?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>[kmgt]?i?b)$",
    re.IGNORECASE,
)
_MEMORY_LIMIT_UNITS = {
    "b": "B",
    "kb": "KiB",
    "kib": "KiB",
    "mb": "MiB",
    "mib": "MiB",
    "gb": "GiB",
    "gib": "GiB",
    "tb": "TiB",
    "tib": "TiB",
}


@dataclass(frozen=True)
class WalCheckpointResult:
    attempted: bool
    wal_bytes_before: int
    wal_bytes_after: int
    duration_seconds: float
    contended: bool = False


def storage_version(conn: duckdb.DuckDBPyConnection) -> str | None:
    """Report the attached format; a normal reopen is required to verify persistence."""
    row = conn.execute(
        "SELECT tags['storage_version'] FROM duckdb_databases() "
        "WHERE database_name = current_database()"
    ).fetchone()
    assert row is not None
    return str(row[0]) if row[0] is not None else None


def storage_upgrade_needed(current: str | None, target: str = STORAGE_VERSION) -> bool:
    """Compare explicit format tags; in-memory databases have no storage to migrate."""
    if current is None:
        return False
    versions: list[tuple[int, ...]] = []
    for value in (current, target):
        match = _STORAGE_VERSION_RE.fullmatch(value)
        if match is None:
            raise ValueError(f"unrecognized DuckDB storage version: {value}")
        versions.append(tuple(int(part) for part in match.groups()))
    return versions[0] < versions[1]


def wal_size_bytes(db_path: Path) -> int:
    """Return the durable WAL size without starting a DuckDB transaction."""
    try:
        return Path(f"{db_path}.wal").stat().st_size
    except FileNotFoundError:
        return 0


def checkpoint_wal_if_due(
    conn: duckdb.DuckDBPyConnection,
    db_path: Path,
    *,
    high_water_bytes: int = WAL_CHECKPOINT_HIGH_WATER_BYTES,
) -> WalCheckpointResult:
    """Checkpoint a WAL that crossed its application maintenance boundary."""
    if high_water_bytes <= 0:
        raise ValueError("checkpoint high-water must be positive")
    before = wal_size_bytes(db_path)
    if before < high_water_bytes:
        return WalCheckpointResult(False, before, before, 0.0)
    started = time.monotonic()
    try:
        conn.execute("CHECKPOINT")
    except duckdb.TransactionException as err:
        if "Cannot CHECKPOINT: there are other write transactions active" not in str(err):
            raise
        return WalCheckpointResult(
            True,
            before,
            wal_size_bytes(db_path),
            time.monotonic() - started,
            contended=True,
        )
    return WalCheckpointResult(True, before, wal_size_bytes(db_path), time.monotonic() - started)


def resolve_memory_limit(config: AppConfig) -> str:
    env_override = os.environ.get("RECALL_DUCKDB_MEMORY_LIMIT")
    if env_override is not None:
        return _validate_memory_limit(env_override, source="RECALL_DUCKDB_MEMORY_LIMIT")
    if config.duckdb.memory_limit is not None:
        return _validate_memory_limit(
            config.duckdb.memory_limit,
            source="[duckdb] memory_limit",
        )
    return _DEFAULT_MEMORY_LIMIT


def resolve_temp_directory(config: AppConfig) -> Path:
    env_override = os.environ.get("RECALL_DUCKDB_TEMP_DIR")
    if env_override is not None:
        return _validate_temp_directory(env_override, source="RECALL_DUCKDB_TEMP_DIR")
    if config.duckdb.temp_directory is not None:
        return _validate_temp_directory(
            config.duckdb.temp_directory,
            source="[duckdb] temp_directory",
        )
    return config.data_dir / "duckdb_spill"


def _validate_temp_directory(value: str, *, source: str) -> Path:
    candidate = value.strip()
    if not candidate:
        raise ValueError(f"{source} must not be empty")
    return Path(candidate).expanduser()


def _validate_memory_limit(value: str, *, source: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError(f"{source} must not be empty")
    match = _MEMORY_LIMIT_RE.fullmatch(candidate)
    if match is None:
        raise ValueError(
            f"{source} must be a positive DuckDB memory size like '32GB', '500MB', or '1TB'"
        )
    if float(match.group("number")) <= 0:
        raise ValueError(f"{source} must be positive")
    unit = _MEMORY_LIMIT_UNITS[match.group("unit").lower()]
    return f"{match.group('number')}{unit}"


def _sql_string_literal(value: object) -> str:
    return str(value).replace("'", "''")


def _apply_runtime_pragmas(conn: duckdb.DuckDBPyConnection, config: AppConfig) -> None:
    spill_path = resolve_temp_directory(config)
    spill_path.mkdir(parents=True, exist_ok=True)
    conn.execute(f"SET memory_limit = '{_sql_string_literal(resolve_memory_limit(config))}'")
    conn.execute(f"SET temp_directory = '{_sql_string_literal(spill_path)}'")
    conn.execute(f"SET wal_autocheckpoint = '{_WAL_AUTOCHECKPOINT}'")


class RecallLockError(RuntimeError):
    pass


@dataclass
class _AdvisoryLockEntry:
    handle: TextIO
    count: int


_advisory_lock_local = threading.local()


def _advisory_lock_registry() -> dict[Path, _AdvisoryLockEntry]:
    registry = getattr(_advisory_lock_local, "registry", None)
    if registry is None:
        registry = {}
        _advisory_lock_local.registry = registry
    return cast(dict[Path, _AdvisoryLockEntry], registry)


@contextmanager
def advisory_lock(lock_path: Path):
    # Reentry is intentionally limited to one thread. Watcher and embedding
    # writers use separate threads and must still serialize through flock.
    resolved_lock_path = Path(lock_path).expanduser().resolve(strict=False)
    create_private_dir(resolved_lock_path.parent)
    registry = _advisory_lock_registry()
    entry = registry.get(resolved_lock_path)
    if entry is not None:
        entry.count += 1
    else:
        handle = open(resolved_lock_path, "w", encoding="utf-8")  # noqa: SIM115
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            handle.close()
            raise RecallLockError("another recall index is running") from err
        except Exception:
            handle.close()
            raise
        registry[resolved_lock_path] = _AdvisoryLockEntry(
            handle=handle,
            count=1,
        )
    try:
        yield
    finally:
        entry = registry[resolved_lock_path]
        entry.count -= 1
        if entry.count == 0:
            try:
                import fcntl

                fcntl.flock(entry.handle, fcntl.LOCK_UN)
            except OSError:
                pass
            finally:
                entry.handle.close()
                registry.pop(resolved_lock_path, None)


def connect(
    config: AppConfig,
    *,
    recreate: bool = False,
    lenient_schema: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open (or create) the recall DuckDB database.

    When *lenient_schema* is True, a fresh database still gets the full schema
    applied, but an existing database with a stale schema version is opened
    without raising.  This lets the daemon start and accept the ``--recreate``
    RPC that will fix the mismatch, avoiding the chicken-and-egg deadlock
    where the daemon can't listen until the schema matches but the recreate
    command can't reach the daemon until it's listening.
    """
    create_private_dir(config.data_dir)
    if recreate:
        _backup_database(config.db_path)
    creating = not config.db_path.exists()
    options: dict[str, str | int | float | list[str]] = {}
    if creating:
        options["storage_compatibility_version"] = STORAGE_VERSION
    conn = duckdb.connect(str(config.db_path), config=options)
    try:
        _apply_runtime_pragmas(conn, config)
        if lenient_schema:
            ensure_schema_lenient(conn, embed_dim=config.embedding.dimensions)
        else:
            ensure_schema(conn, embed_dim=config.embedding.dimensions)
        if creating:
            # Creation-only settings participate in DuckDB connection identity.
            # Persist the header, then return the same ordinary configuration
            # used by every later handle (including startup sidecar work).
            conn.close()
            conn = duckdb.connect(str(config.db_path))
            _apply_runtime_pragmas(conn, config)
    except BaseException:
        conn.close()
        raise
    return conn


def is_lock_conflict(err: BaseException) -> bool:
    """True when DuckDB refused the file because another process holds it read-write.

    DuckDB's file lock is exclusive to a read-write holder across processes:
    while the daemon holds the database, even a read-only open from the CLI
    fails with this error (REQ-DAEMON-074).
    """
    message = str(err)
    return "Conflicting lock" in message or "Could not set lock" in message


def connect_readonly(config: AppConfig) -> duckdb.DuckDBPyConnection:
    """Open database in read-only mode.

    Read-only opens coexist with each other, but not with a read-write holder
    in another process: a live daemon makes this raise the `duckdb.IOException`
    that `is_lock_conflict` recognizes. Falls back to a read-write connection
    if the database does not exist yet (read-only mode requires an existing
    file).
    """
    if not config.db_path.exists():
        return connect(config)
    conn = duckdb.connect(str(config.db_path), read_only=True)
    _apply_runtime_pragmas(conn, config)
    return conn


def _backup_database(db_path: Path) -> None:
    if not db_path.exists():
        return
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    backup_path = db_path.with_suffix(f".bak-{timestamp}")
    os.replace(db_path, backup_path)
    wal_path = db_path.with_suffix(db_path.suffix + ".wal")
    if wal_path.exists():
        os.replace(wal_path, wal_path.with_suffix(wal_path.suffix + f".{timestamp}.bak"))
