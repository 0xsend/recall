"""Consistent pre-recreate backups and transactional schema replacement."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import duckdb

from recall.core.config import AppConfig
from recall.core.durability import sync_directory
from recall.core.failure_observation import FailureObserver, observe_before_cleanup
from recall.db import sidecar_path
from recall.db.connection import check_database_schema
from recall.db.lease import DbLease
from recall.db.repair import FENCE_VERSION, MANIFEST_BYTES_MAX, load_repair_row
from recall.db.repair_oracle import OracleReference, read_oracle, read_publication_at
from recall.db.schema import check_schema_compatibility, initialize_schema_in
from recall.services.compaction import _clone_or_copy
from recall.services.migration_backup import BackupInputs

_FAILED_SET_ENTRIES_MAX = 32768
_FAILED_SET_FILES_MAX = 16384
_FAILED_SET_BYTES_MAX = 16 * 1024**4


class RecreationCapability:
    """A single destructive execution bound to its admitted backup and live lease."""

    def __init__(self, lease: DbLease, backup: Path) -> None:
        self.backup = backup
        self._lineage = lease.lineage
        self._epoch = lease.connection_epoch
        self._consumed = False

    def consume(self, lease: DbLease) -> None:
        assert not self._consumed, "destructive reset authority was already consumed"
        assert lease.lineage == self._lineage and lease.connection_epoch is self._epoch
        self._consumed = True


def prepare_recreation_on(lease: DbLease, config: AppConfig) -> RecreationCapability:
    from recall.services.recovery_admission import require_preserving_on
    from recall.services.self_repair import preflight_recovery_files

    preflight_recovery_files(config)
    version = lease.call("recreate_schema_admission", check_schema_compatibility)
    require_preserving_on(lease, config, version)
    backup = lease.call(
        "recreate_backup",
        lambda conn: backup_before_recreate(conn, config, observer=lease.observer),
    )
    return RecreationCapability(lease, backup)


def preflight_confirmed_recreation(config: AppConfig) -> None:
    from recall.services.self_repair import preflight_recovery_files

    preflight_recovery_files(config)
    check_database_schema(config)


def consistent_backup(
    conn: duckdb.DuckDBPyConnection,
    config: AppConfig,
    *,
    kind: str,
    recovery: bool = False,
    observer: FailureObserver,
    destination: Path | None = None,
) -> Path:
    """Copy a checkpointed database and consistent sidecar before destructive work.

    The caller owns the writer and keeps the shared handle open. Concurrent
    read cursors may continue: checkpoint plus the writer lock freezes both
    stores, and SQLite's backup API produces its consistent snapshot.
    An incomplete backup is retained for diagnosis and never authorizes mutation.
    """
    # DuckDB 1.5.5 may reject plain CHECKPOINT while a read cursor retains an
    # older transaction. FORCE waits for that cursor; it does not abort it.
    # The storage ownership tests preserve a real reader across this boundary.
    conn.execute("FORCE CHECKPOINT")
    backup = (
        destination
        if destination is not None
        else config.data_dir / "snapshots" / f"{kind}-{uuid4().hex}"
    )
    backup.mkdir(parents=True, mode=0o700)
    database = backup / "recall.duckdb"
    _clone_or_copy(config.db_path, database)
    copied = [database]
    sidecar = sidecar_path(config.data_dir)
    if sidecar.is_file():
        destination = backup / "recall.fts.sqlite"
        source = sqlite3.connect(sidecar)
        with observe_before_cleanup(observer, "backup:sidecar_source", source.close):
            target = sqlite3.connect(destination)
            with observe_before_cleanup(observer, "backup:sidecar_target", target.close):
                import time

                started = time.monotonic()

                def progress(status, remaining, total):
                    if time.monotonic() - started > 300:
                        raise TimeoutError("matched sidecar backup exceeded five minutes")

                source.backup(target, pages=256, progress=progress, sleep=0.01)
        copied.append(destination)
    files = [path.name for path in copied]
    recovery_set = _copy_recovery_set(conn, config, backup) if recovery else None
    for path in copied:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    manifest = backup / "manifest.json"
    payload = json.dumps(
        {
            "kind": kind,
            "complete": True,
            "created_at": datetime.now(UTC).isoformat(),
            "files": files,
            "checksums": {path.name: _hash(path) for path in copied},
            **({"recovery": recovery_set} if recovery else {}),
        }
    ).encode()
    if len(payload) > MANIFEST_BYTES_MAX:
        raise ValueError("matched backup manifest exceeds 4 MiB")
    with manifest.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(backup, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    sync_directory(backup.parent)
    return backup


def backup_before_recreate(
    conn: duckdb.DuckDBPyConnection, config: AppConfig, *, observer: FailureObserver
) -> Path:
    """Copy a checkpointed database and consistent sidecar before destructive work."""
    return consistent_backup(conn, config, kind="recreate", recovery=True, observer=observer)


def _copy_recovery_set(
    conn: duckdb.DuckDBPyConnection, config: AppConfig, backup: Path
) -> dict[str, object]:
    from recall.services.repair_events import read_events

    retained: dict[str, dict[str, object]] = {}

    def copy(source: Path, relative: str) -> None:
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _clone_or_copy(source, target)
        digest = hashlib.sha256()
        size = 0
        with target.open("rb") as handle:
            while block := handle.read(64 * 1024):
                digest.update(block)
                size += len(block)
            os.fsync(handle.fileno())
        retained[relative] = {"sha256": digest.hexdigest(), "bytes": size}
        sync_directory(target.parent)

    batch = read_events(config.data_dir)
    if batch.primary is not None:
        copy(config.data_dir / "daemon-failure.json", "daemon-failure.json")
    for path, _ in batch.secondary:
        copy(path, f"fatal-events/{path.name}")
    if config.config_path.is_file():
        if config.config_path.stat().st_size > MANIFEST_BYTES_MAX:
            raise ValueError("matched backup config exceeds 4 MiB")
        copy(config.config_path, "config.toml")
    refusal = config.data_dir / "daemon-refused"
    if refusal.exists():
        if refusal.stat().st_size > MANIFEST_BYTES_MAX:
            raise ValueError("matched backup refusal exceeds 4 MiB")
        copy(refusal, "daemon-refused")
    version = check_schema_compatibility(conn)
    references: dict[str, OracleReference] = {}
    if version is not None and version >= FENCE_VERSION:
        row = load_repair_row(conn)
        if row.oracle_ref is not None:
            assert row.trigger_id is not None and row.oracle_hash is not None
            references[row.oracle_ref] = OracleReference(
                trigger_id=row.trigger_id, manifest_hash=row.oracle_hash
            )
    assert len(references) <= 2
    for relative, reference in references.items():
        directory = config.data_dir / relative
        manifest = read_oracle(directory, reference)
        copy(directory / "manifest.json", f"{relative}/manifest.json")
        tables = manifest.tables + (manifest.sidecar.tables if manifest.sidecar is not None else ())
        for table in tables:
            copy(directory / table.filename, f"{relative}/{table.filename}")
    publications: dict[str, str] = {}
    if version is not None and version >= FENCE_VERSION:
        if row.release_certificate is not None:
            assert row.oracle_ref is not None
            publications[row.oracle_ref + "/publication"] = row.release_certificate.publication_hash
        work = None if row.trigger_json is None else row.trigger_json.manual_work
        if work is not None and work.publication is not None:
            publications[work.publication.path] = work.publication.manifest_hash
    assert len(publications) <= 2
    for relative, digest in publications.items():
        publication = read_publication_at(config.data_dir / relative, digest)
        copy(config.data_dir / relative / "manifest.json", relative + "/manifest.json")
        for table in (
            publication.pending,
            *publication.sidecar.tables,
            *publication.native_keywords,
        ):
            copy(config.data_dir / relative / table.filename, relative + "/" + table.filename)
        sync_directory((backup / relative).parent)
        sync_directory((backup / relative).parent.parent)
    if references:
        sync_directory(backup / "repair-oracles")
    return {
        "schema_version": version,
        "files": retained,
        "absent": [
            name
            for name in ("daemon-failure.json", "daemon-refused", "config.toml")
            if name not in retained
        ],
        "spool_present": (config.data_dir / "fatal-events").exists(),
        "post_backup_acknowledged_writes": (
            "recovery and verification required before restore authority"
        ),
    }


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preserve_failed_set(config: AppConfig, destination: Path, inputs: BackupInputs) -> Path:
    """Archive a quiesced failed set without opening or checkpointing either engine.

    This forensic archive retains post-backup work. It grants no restore or
    service authority and is outside automatic snapshot garbage collection.
    """
    root = config.data_dir.resolve()
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(root):
        raise ValueError("failed-set archive must be a unique path outside the data directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[Path, Path] = {}
    for source, name in (
        (config.db_path, "recall.duckdb"),
        (Path(str(config.db_path) + ".wal"), "recall.duckdb.wal"),
        (config.config_path, "config.toml"),
        *(
            (Path(str(sidecar_path(root)) + suffix), "recall.fts.sqlite" + suffix)
            for suffix in ("", "-wal", "-shm", "-journal")
        ),
    ):
        if source.exists():
            paths[source.resolve()] = Path(name)
    for count, source in enumerate(root.rglob("*"), start=1):
        if count > _FAILED_SET_ENTRIES_MAX or source.is_symlink():
            raise ValueError("failed-set archive exceeds its entry bound or contains a link")
        mode = source.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("failed-set archive requires quiesced regular files")
        paths.setdefault(source.resolve(), Path("data") / source.relative_to(root))
    if len(paths) > _FAILED_SET_FILES_MAX:
        raise ValueError("failed-set archive exceeds its file bound")
    matched = ((inputs.database, "matched-input.duckdb"),) + (
        () if inputs.sidecar is None else ((inputs.sidecar, "matched-input.fts.sqlite"),)
    )
    size = sum(source.stat().st_size for source in paths) + sum(
        os.fstat(source.fileno()).st_size for source, _ in matched
    )
    if size > _FAILED_SET_BYTES_MAX or size > shutil.disk_usage(destination.parent).free:
        raise ValueError("insufficient capacity for the complete failed-set archive")
    destination.mkdir(mode=0o700)
    files = {}
    directories = {destination}
    for source, relative in sorted(paths.items()):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _clone_or_copy(source, target)
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
        files[str(relative)] = {"bytes": target.stat().st_size, "sha256": _hash(target)}
        for parent in (target.parent, *target.parent.parents):
            if not parent.is_relative_to(destination):
                break
            directories.add(parent)
    for source, name in matched:
        target = destination / name
        source.seek(0)
        with target.open("wb") as output:
            shutil.copyfileobj(source, output, length=64 * 1024)
            output.flush()
            os.fsync(output.fileno())
        source.seek(0)
        files[name] = {"bytes": target.stat().st_size, "sha256": _hash(target)}
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        sync_directory(directory)
    payload = json.dumps(
        {"authoritative": False, "status": "post-backup recovery ledger unsettled", "files": files}
    ).encode()
    if len(payload) > MANIFEST_BYTES_MAX:
        raise ValueError("failed-set archive manifest exceeds 4 MiB")
    with (destination / "preserved.json").open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(destination)
    sync_directory(destination.parent)
    return destination


def restore_consistent_backup(config: AppConfig, backup: BackupInputs) -> None:
    """Restore validated, pinned inputs without reopening a source pathname."""
    sources = (backup.database,) if backup.sidecar is None else (backup.database, backup.sidecar)
    sidecar = sidecar_path(config.data_dir)
    destinations = (config.db_path,) if backup.sidecar is None else (config.db_path, sidecar)
    # Stream copying must retain copyfile's same-file protection, including
    # cross-file aliases, before truncating either destination or removing WAL.
    for destination in destinations:
        try:
            target_stat = destination.stat()
        except FileNotFoundError:
            continue
        if any(os.path.samestat(os.fstat(source.fileno()), target_stat) for source in sources):
            raise shutil.SameFileError("backup input overlaps a restore destination")
    wal = Path(str(config.db_path) + ".wal")
    wal.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm", "-journal"):
        Path(str(sidecar) + suffix).unlink(missing_ok=True)
    with config.db_path.open("wb") as target:
        shutil.copyfileobj(backup.database, target)
        target.flush()
        os.fsync(target.fileno())
    if backup.sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        with sidecar.open("wb") as target:
            shutil.copyfileobj(backup.sidecar, target)
            target.flush()
            os.fsync(target.fileno())
    else:
        sidecar.unlink(missing_ok=True)
    sync_directory(config.db_path.parent)
    if sidecar.parent != config.db_path.parent:
        sync_directory(sidecar.parent)


def replace_schema_on(lease: DbLease, config: AppConfig, capability: RecreationCapability) -> None:
    """Consume a backup-bound reset and atomically install the fresh fenced schema."""
    from recall.services.recovery_admission import require_preserving_on
    from recall.services.repair_protocol import RepairProtocol
    from recall.services.self_repair import preflight_recovery_files

    preflight_recovery_files(config)
    version = lease.call("recreate_final_admission", check_schema_compatibility)
    require_preserving_on(lease, config, version)
    # Only already-final replay records are admitted here. Their matched copy
    # is in the backup before retirement; no unfinished event can be reset.
    RepairProtocol(config.data_dir).retire(lease)
    capability.consume(lease)
    with lease.transaction("recreate_schema") as transaction:
        transaction.call("recreate_drop", _drop_main_tables)
        initialize_schema_in(transaction, config.embedding.dimensions)
    lease.call("recreate_sidecar", lambda _: _clear_sidecar(config))
    lease.call("recreate_checkpoint", lambda conn: conn.execute("CHECKPOINT").fetchone())


def _drop_main_tables(conn: duckdb.DuckDBPyConnection) -> None:
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    for (table,) in tables:
        quoted = str(table).replace('"', '""')
        conn.execute(f'DROP TABLE IF EXISTS "{quoted}" CASCADE')


def _clear_sidecar(config: AppConfig) -> None:
    sidecar = sidecar_path(config.data_dir)
    if sidecar.is_file():
        with closing(sqlite3.connect(sidecar)) as connection, connection:
            for table in (
                "message_fts",
                "message_fts_rowid",
                "tool_calls_fts",
                "tool_calls_fts_rowid",
            ):
                connection.execute(f"DELETE FROM {table}")
