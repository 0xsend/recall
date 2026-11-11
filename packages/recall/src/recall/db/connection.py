from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TextIO, cast
from weakref import WeakKeyDictionary

import duckdb

from recall.core.config import AppConfig, create_private_dir
from recall.core.memory_budget import DUCKDB_MEMORY_LIMIT
from recall.core.schema_version import SchemaTooNewError
from recall.core.telemetry import Stage, measured
from recall.db.lease import DbLease, DbLifetime, StandaloneLifetime
from recall.db.schema import SchemaChange, check_schema_compatibility, ensure_schema_on

logger = logging.getLogger("recall.connection")

# DuckDB buffers do not bound total RSS. Leave headroom for captures, vectors
# and native allocations independently of host RAM. Oversized overrides require
# an explicit opt-out from the whole-process resource floor.
_DEFAULT_MEMORY_LIMIT = DUCKDB_MEMORY_LIMIT
# Native task allocations are separate from DuckDB's buffer allowance.
_MAX_THREADS = 4
_ALLOCATOR_FLUSH_THRESHOLD = "1MiB"
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
    r"^(?P<number>-?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>(?:[kmgt]i?)?b)$",
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


@measured(Stage.CHECKPOINT_CHECK)
def wal_size_bytes(db_path: Path) -> int:
    """Return the durable WAL size without starting a DuckDB transaction."""
    try:
        return Path(f"{db_path}.wal").stat().st_size
    except FileNotFoundError:
        return 0


@measured(Stage.CHECKPOINT)
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


@dataclass(frozen=True)
class MemoryLimitResolution:
    requested: str | None
    applied: str

    @property
    def clamped(self) -> bool:
        return self.requested is not None and self.requested != self.applied


def _memory_limit_bytes(value: str) -> Decimal:
    match = _MEMORY_LIMIT_RE.fullmatch(value)
    assert match is not None
    unit = match.group("unit").lower()
    exponent = "bkmgt".index(unit[0])
    base = 1024 if "i" in unit else 1000
    return Decimal(match.group("number")) * base**exponent


def resolve_memory_limit_settings(config: AppConfig) -> MemoryLimitResolution:
    """Resolve precedence and clamp against the same allowance the default uses."""
    env_override = os.environ.get("RECALL_DUCKDB_MEMORY_LIMIT")
    if env_override is not None:
        requested = _validate_memory_limit(env_override, source="RECALL_DUCKDB_MEMORY_LIMIT")
    elif config.duckdb.memory_limit is not None:
        requested = _validate_memory_limit(
            config.duckdb.memory_limit,
            source="[duckdb] memory_limit",
        )
    else:
        return MemoryLimitResolution(None, _DEFAULT_MEMORY_LIMIT)
    exceeds_allowance = _memory_limit_bytes(requested) > _memory_limit_bytes(_DEFAULT_MEMORY_LIMIT)
    applied = (
        _DEFAULT_MEMORY_LIMIT
        if exceeds_allowance and not config.duckdb.memory_limit_exceeds_floor
        else requested
    )
    return MemoryLimitResolution(requested, applied)


def resolve_memory_limit(config: AppConfig) -> str:
    return resolve_memory_limit_settings(config).applied


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
    # Fail before schema/data writes if this engine cannot enforce the policy.
    conn.execute("SET disabled_compression_methods = 'roaring'")
    row = conn.execute("SELECT current_setting('threads')").fetchone()
    assert row is not None
    conn.execute(f"SET threads = {min(int(row[0]), _MAX_THREADS)}")
    conn.execute(f"SET allocator_flush_threshold = '{_ALLOCATOR_FLUSH_THRESHOLD}'")
    conn.execute(
        f"SET allocator_bulk_deallocation_flush_threshold = '{_ALLOCATOR_FLUSH_THRESHOLD}'"
    )
    spill_path = resolve_temp_directory(config)
    spill_path.mkdir(parents=True, exist_ok=True)
    conn.execute(f"SET memory_limit = '{_sql_string_literal(resolve_memory_limit(config))}'")
    row = conn.execute("SELECT current_setting('temp_directory')").fetchone()
    assert row is not None
    # DuckDB rejects even a same-path SET once this database instance has spilled.
    if row[0] != str(spill_path):
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


# Keep only live handles, never a persisted version cache. DuckDB cannot open
# read-only alongside a writer in the same process; its independent cursors
# provide the effective catalog for subsequent opens instead.
_writer_handles: WeakKeyDictionary[duckdb.DuckDBPyConnection, Path] = WeakKeyDictionary()
_connection_open_lock = threading.RLock()


def _close_after_failure(
    connection: duckdb.DuckDBPyConnection,
    lifetime: DbLifetime,
    primary: BaseException,
    operation: str,
) -> None:
    lifetime.observe(primary, operation)
    try:
        lifetime.cleanup_db(f"{operation}:close", connection.close)
    except BaseException as cleanup:
        raise BaseExceptionGroup(
            "database operation and close failed", [primary, cleanup]
        ) from None


def _check_and_close(
    connection: duckdb.DuckDBPyConnection,
    lifetime: DbLifetime,
    config: AppConfig,
    admission: Callable[[DbLease, int | None], None] | None,
) -> None:
    with lifetime.lease(connection) as lease:
        try:
            lease.call(
                "schema_precheck:policy", lambda conn: _apply_runtime_pragmas(conn, config)
            )
            version = lease.call("schema_precheck", check_schema_compatibility)
            if admission is not None:
                admission(lease, version)
        except BaseException as primary:
            _close_after_failure(connection, lifetime, primary, "schema_precheck")
            raise
        else:
            lifetime.cleanup_db("schema_precheck:close", connection.close)


def _check_before_write(
    config: AppConfig,
    lifetime: DbLifetime,
    admission: Callable[[DbLease, int | None], None] | None,
) -> None:
    path = config.db_path.resolve()
    for writer, writer_path in list(_writer_handles.items()):
        if writer_path != path:
            continue
        try:
            cursor = lifetime.call_db("schema_precheck:cursor", writer.cursor)
        except duckdb.ConnectionException:
            del _writer_handles[writer]
            continue
        _check_and_close(cursor, lifetime, config, admission)
        return
    reader = lifetime.call_db(
        "schema_precheck:open", lambda: duckdb.connect(str(config.db_path), read_only=True)
    )
    _check_and_close(reader, lifetime, config, admission)


def _open_database(
    config: AppConfig,
    *,
    lifetime: DbLifetime,
    read_only: bool = False,
    storage_target: str | None = None,
    on_newer_schema: Callable[[duckdb.DuckDBPyConnection], None] | None = None,
    admission: Callable[[DbLease, int | None], None] | None = None,
    creation_storage_version: str = STORAGE_VERSION,
) -> duckdb.DuckDBPyConnection:
    """Own compatibility admission before any canonical write open or ATTACH.

    Only the confirmed recreation callback may consume an incompatible writer;
    it must back up and replace the schema before a handle can be returned.
    Existing pathname ownership must span the precheck and open for maintenance.
    """
    with _connection_open_lock:
        creating = not config.db_path.exists()
        newer = False
        if not read_only and not creating:
            try:
                _check_before_write(config, lifetime, admission)
            except SchemaTooNewError:
                if on_newer_schema is None:
                    raise
                newer = True
        options: dict[str, str | int | float | list[str]] = {}
        if creating:
            options["storage_compatibility_version"] = creation_storage_version
        conn = lifetime.call_db(
            "database_writer_open",
            lambda: duckdb.connect(
                ":memory:" if storage_target is not None else str(config.db_path),
                read_only=read_only,
                config=options,
            ),
        )
        try:
            if storage_target is not None:
                lifetime.call_db(
                    "storage_attach",
                    lambda: conn.execute(
                        f"ATTACH '{_sql_string_literal(config.db_path)}' AS maintenance "
                        f"(STORAGE_VERSION '{_sql_string_literal(storage_target)}')"
                    ).fetchone(),
                )
                lifetime.call_db(
                    "storage_select", lambda: conn.execute("USE maintenance").fetchone()
                )
            if read_only:
                with lifetime.lease(conn) as lease:
                    version = lease.call("schema_readonly_check", check_schema_compatibility)
                    if admission is not None:
                        admission(lease, version)
            lifetime.call_db(
                "database_runtime_pragmas", lambda: _apply_runtime_pragmas(conn, config)
            )
            if newer:
                assert on_newer_schema is not None
                lifetime.call_db("confirmed_schema_replacement", lambda: on_newer_schema(conn))
                lifetime.call_db(
                    "replacement_schema_check", lambda: check_schema_compatibility(conn)
                )
            if not read_only:
                _writer_handles[conn] = config.db_path.resolve()
        except BaseException as primary:
            _close_after_failure(conn, lifetime, primary, "database_open")
            raise
        return conn


def connect_storage_transition(config: AppConfig) -> duckdb.DuckDBPyConnection:
    """Open the existing compatible file with the explicit maintenance format."""
    if not config.db_path.is_file():
        raise ValueError(f"database does not exist: {config.db_path}")
    return _open_database(config, lifetime=StandaloneLifetime(), storage_target=STORAGE_VERSION)


def connect(
    config: AppConfig,
    *,
    recreate: bool = False,
    lenient_schema: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open an independently owned library connection; the caller closes it."""
    return _connect(
        config,
        lifetime=StandaloneLifetime(),
        recreate=recreate,
        lenient_schema=lenient_schema,
    )


def connect_owned(
    config: AppConfig,
    *,
    lifetime: DbLifetime,
    admission: Callable[[DbLease, int | None], None],
    lenient_schema: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open through a required owner before precheck, migration or cleanup can fail."""
    return _connect(
        config,
        lifetime=lifetime,
        recreate=False,
        lenient_schema=lenient_schema,
        admission=admission,
    )


def connect_recovery_owned(
    config: AppConfig,
    *,
    lifetime: DbLifetime,
    admission: Callable[[DbLease, int | None], None],
) -> duckdb.DuckDBPyConnection:
    """Fresh reopen of an existing fenced file under an explicit live admission."""
    if not config.db_path.is_file():
        raise ValueError("recovery database is absent; matched restore is required")
    return _open_database(config, lifetime=lifetime, admission=admission)


def create_copy_candidate_owned(
    config: AppConfig, *, lifetime: DbLifetime, storage_version: str
) -> duckdb.DuckDBPyConnection:
    """Create only a new configured copy candidate, preserving its source format."""
    if config.db_path.exists():
        raise ValueError("a copy candidate cannot overwrite an existing database")
    return _open_database(config, lifetime=lifetime, creation_storage_version=storage_version)


def _connect(
    config: AppConfig,
    *,
    lifetime: DbLifetime,
    recreate: bool,
    lenient_schema: bool,
    admission: Callable[[DbLease, int | None], None] | None = None,
) -> duckdb.DuckDBPyConnection:
    from recall.core.keyword_maintenance import require_keyword_writes_enabled

    require_keyword_writes_enabled(config.data_dir)
    create_private_dir(config.data_dir)
    if recreate:
        _backup_database(config.db_path)
    conn = _open_database(config, lifetime=lifetime, admission=admission)
    try:
        with lifetime.lease(conn) as lease:
            schema_change = ensure_schema_on(
                lease, embed_dim=config.embedding.dimensions, lenient=lenient_schema
            )
            if admission is not None:
                version = lease.call("schema_postmigration_admission", check_schema_compatibility)
                admission(lease, version)
        if schema_change is SchemaChange.CHANGED:
            # Persist bootstrap/migration state before using the ordinary
            # identity shared by every subsequent handle on this pathname.
            initial = conn
            conn = None
            lifetime.cleanup_db("database_bootstrap_close", initial.close)
            conn = lifetime.call_db(
                "postmigration_open",
                lambda: _open_database(config, lifetime=lifetime, admission=admission),
            )
        else:
            assert schema_change is SchemaChange.UNCHANGED
    except BaseException as primary:
        if conn is not None:
            _close_after_failure(conn, lifetime, primary, "database_schema_open")
        raise
    assert conn is not None
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
    return _open_database(config, lifetime=StandaloneLifetime(), read_only=True)


def connect_existing(config: AppConfig, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open an existing compatible maintenance target without schema migration."""
    if not config.db_path.is_file():
        raise ValueError(f"database does not exist: {config.db_path}")
    return _open_database(config, lifetime=StandaloneLifetime(), read_only=read_only)


def connect_existing_owned(
    config: AppConfig,
    *,
    lifetime: DbLifetime,
    admission: Callable[[DbLease, int | None], None],
    read_only: bool,
    storage_target: str | None = None,
) -> duckdb.DuckDBPyConnection:
    if not config.db_path.is_file():
        raise ValueError(f"database does not exist: {config.db_path}")
    assert not (read_only and storage_target is not None)
    return _open_database(
        config,
        lifetime=lifetime,
        admission=admission,
        read_only=read_only,
        storage_target=storage_target,
    )


def check_database_schema(config: AppConfig) -> None:
    """Refuse a known newer schema before starting a daemon or maintenance.

    An unavailable snapshot leaves diagnosis to the operation's existing
    ownership/error handling. Its connection still validates the schema.
    """
    from recall.db.ownership import SnapshotUnavailable, read_offline

    try:
        if not config.db_path.is_file():
            return
        result = read_offline(config, check_schema_compatibility)
    except SchemaTooNewError:
        raise
    except (duckdb.Error, OSError, RuntimeError, ValueError) as err:
        logger.debug("schema preflight unavailable for %s: %s", config.db_path, err)
    else:
        if isinstance(result, SnapshotUnavailable):
            logger.debug("schema preflight unavailable: %s", result.reason)


def _backup_database(db_path: Path) -> None:
    if not db_path.exists():
        return
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    backup_path = db_path.with_suffix(f".bak-{timestamp}")
    os.replace(db_path, backup_path)
    wal_path = db_path.with_suffix(db_path.suffix + ".wal")
    if wal_path.exists():
        os.replace(wal_path, wal_path.with_suffix(wal_path.suffix + f".{timestamp}.bak"))
