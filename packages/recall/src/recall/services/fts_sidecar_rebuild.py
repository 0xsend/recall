"""Explicit offline canonical keyword reconstruction, with retained rollback artifacts.

Run ``python -m recall.services.fts_sidecar_rebuild --help``. All targets are
explicit; this module never stops services, parses transcripts or enriches data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import duckdb
from pydantic import BaseModel, ConfigDict, model_validator

from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.keyword_maintenance import MARKER_NAME
from recall.db.connection import advisory_lock, connect_existing
from recall.db.fts_sidecar import (
    FINGERPRINT_FORMAT,
    content_fingerprint,
    fts_fields_signature,
    open_sidecar,
    scope_message_fts_columns,
    set_fts_fields_signature,
    should_index_bash_fts,
)
from recall.services.fts_sidecar_bootstrap import bootstrap_sidecar
from recall.services.recreation import consistent_backup

BATCH_SIZE = 10_000
FORMAT_VERSION = 2
logger = logging.getLogger(__name__)


class CaptureIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    python_version: str
    sqlite_version: str
    duckdb_version: str
    database_bytes: int
    sidecar_bytes: int
    schema_versions: list[int]
    schema_migrations: list[str]
    index_migration_version: int
    index_migration_target: int
    index_migration_phase: str
    sidecar_metadata: dict[str, str]
    source_sessions: dict[str, int]
    pending_count: int
    pending_sha256: str


def _capture_identity(backup: Path) -> CaptureIdentity:
    """Describe the immutable pair, including an ordered digest of pending work."""
    with (
        closing(duckdb.connect(str(backup / "recall.duckdb"), read_only=True)) as source,
        closing(sqlite3.connect(backup / "recall.fts.sqlite")) as sidecar,
    ):
        pending = source.execute(
            "SELECT kind, id, op, queued_at FROM fts_sidecar_pending ORDER BY rowid"
        )
        digest = hashlib.sha256()
        count = 0
        while batch := pending.fetchmany(BATCH_SIZE):
            for row in batch:
                digest.update((json.dumps(row, default=str) + "\n").encode("utf-8"))
            count += len(batch)
        version = source.execute("SELECT index_migration_version FROM runtime_state").fetchone()
        migration = source.execute(
            "SELECT target_version, phase FROM index_migration_jobs"
        ).fetchone()
        assert version is not None and migration is not None
        return CaptureIdentity(
            python_version=platform.python_version(),
            sqlite_version=sqlite3.sqlite_version,
            duckdb_version=duckdb.__version__,
            database_bytes=(backup / "recall.duckdb").stat().st_size,
            sidecar_bytes=(backup / "recall.fts.sqlite").stat().st_size,
            schema_versions=[
                row[0]
                for row in source.execute(
                    "SELECT version FROM schema_version ORDER BY version"
                ).fetchall()
            ],
            schema_migrations=[
                row[0]
                for row in source.execute(
                    "SELECT migration_id FROM schema_migrations ORDER BY migration_id"
                ).fetchall()
            ],
            index_migration_version=int(version[0] or 0),
            index_migration_target=int(migration[0]),
            index_migration_phase=str(migration[1]),
            sidecar_metadata=dict(sidecar.execute("SELECT key, value FROM fts_sidecar_meta")),
            source_sessions={
                str(key): int(value)
                for key, value in source.execute(
                    "SELECT source, count(*) FROM sessions GROUP BY source ORDER BY source"
                ).fetchall()
            },
            pending_count=count,
            pending_sha256=digest.hexdigest(),
        )


class RebuildManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1, 2] = FORMAT_VERSION
    data_dir: str
    db_path: str
    lock_path: str
    build_id: str
    fields: str
    fingerprint_format: str = FINGERPRINT_FORMAT
    backup: str
    database_hash: str
    original_hash: str
    candidate_hash: str | None = None
    capture: CaptureIdentity | None = None
    replaces_operation: str | None = None
    state: Literal["building", "verified", "installed", "rolled_back", "released", "aborted"] = (
        "building"
    )

    @model_validator(mode="after")
    def require_capture(self) -> Self:
        if self.format_version == 2 and self.capture is None:
            raise ValueError("manifest format 2 requires capture identity")
        return self


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: str) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync(path.parent)


def _varints(blob: bytes) -> list[int]:
    result = []
    value = 0
    width = 0
    for byte in blob:
        width += 1
        if width == 9:
            result.append((value << 8) | byte)
            value = width = 0
        else:
            value = (value << 7) | (byte & 127)
            if byte < 128:
                result.append(value)
                value = width = 0
    if width:
        raise RuntimeError("incomplete FTS size record")
    return result


def verify_totals(conn: sqlite3.Connection, table: str, columns: int) -> tuple[int, ...]:
    """Compare read-only FTS averages with live per-document sizes."""
    totals = [0] * (columns + 1)
    for (blob,) in conn.execute(f"SELECT sz FROM {table}_docsize"):
        sizes = _varints(blob)
        if len(sizes) != columns:
            raise RuntimeError(f"invalid document sizes in {table}")
        totals[0] += 1
        for index, size in enumerate(sizes, 1):
            totals[index] += size
    row = conn.execute(f"SELECT block FROM {table}_data WHERE id=1").fetchone()
    averages = _varints(row[0]) if row is not None and row[0] else [0] * (columns + 1)
    if averages != totals:
        raise RuntimeError(f"{table} totals disagree: averages={averages}, live={totals}")
    return tuple(totals)


def _canonical_rows(
    source: duckdb.DuckDBPyConnection,
    kind: Literal["message", "tool_call"],
    fields: tuple[str, ...],
) -> Iterator[tuple[str, str]]:
    last_id = ""
    while True:
        if kind == "message":
            batch = source.execute(
                "SELECT message_id, COALESCE(fts_content,''), COALESCE(fts_thinking,'') "
                "FROM message_state WHERE message_id > ? ORDER BY message_id LIMIT ?",
                [last_id, BATCH_SIZE],
            ).fetchall()
            rows = [
                (str(key), scope_message_fts_columns(str(content), str(thinking), fields))
                for key, content, thinking in batch
            ]
        elif should_index_bash_fts(fields):
            batch = source.execute(
                "SELECT id, bash_command FROM tool_calls WHERE bash_command IS NOT NULL "
                "AND id > ? ORDER BY id LIMIT ?",
                [last_id, BATCH_SIZE],
            ).fetchall()
            rows = [(str(key), (str(command),)) for key, command in batch]
        else:
            return
        if not rows:
            return
        for key, columns in rows:
            yield key, content_fingerprint(columns)
        last_id = rows[-1][0]


def verify_candidate(
    source: duckdb.DuckDBPyConnection,
    sidecar: sqlite3.Connection,
    fields: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    if sidecar.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise RuntimeError("candidate SQLite integrity check failed")
    totals = {}
    for kind, table, column, width in (
        ("message", "message_fts", "message_id", 2),
        ("tool_call", "tool_calls_fts", "tool_call_id", 1),
    ):
        expected = _canonical_rows(source, kind, fields)
        actual = iter(
            sidecar.execute(
                f"SELECT {column}, content_fingerprint FROM {table}_rowid ORDER BY {column}"
            )
        )
        for expected_row in expected:
            if next(actual, None) != expected_row:
                raise RuntimeError(
                    f"candidate {kind} identity/content mismatch at {expected_row[0]}"
                )
        if next(actual, None) is not None:
            raise RuntimeError(f"candidate contains extra {kind} identities")
        if (
            sidecar.execute(
                f"SELECT rowid FROM {table} EXCEPT SELECT rowid FROM {table}_rowid"
            ).fetchone()
            is not None
            or sidecar.execute(
                f"SELECT rowid FROM {table}_rowid EXCEPT SELECT rowid FROM {table}"
            ).fetchone()
            is not None
        ):
            raise RuntimeError(f"candidate {kind} FTS membership differs from mapping")
        with sidecar:
            sidecar.execute(f"INSERT INTO {table}({table}) VALUES ('integrity-check')")
        totals[kind] = verify_totals(sidecar, table, width)
    return totals


@dataclass(frozen=True)
class KeywordRebuild:
    config: AppConfig
    operation_dir: Path
    build_id: str

    @property
    def manifest_path(self) -> Path:
        return self.operation_dir / "manifest.json"

    @property
    def candidate(self) -> Path:
        return self.operation_dir / "candidate.sqlite"

    @property
    def sidecar(self) -> Path:
        return self.config.data_dir / "recall.fts.sqlite"

    @property
    def marker(self) -> Path:
        return self.config.data_dir / MARKER_NAME

    def _identity(self) -> dict[str, str]:
        return dict(
            data_dir=str(self.config.data_dir.resolve()),
            db_path=str(self.config.db_path.resolve()),
            lock_path=str(self.config.lock_path.resolve()),
            build_id=self.build_id,
            fields=fts_fields_signature(self.config.fts.fields),
            fingerprint_format=FINGERPRINT_FORMAT,
        )

    def inspect(self) -> RebuildManifest | None:
        if not self.manifest_path.exists():
            return None
        manifest = RebuildManifest.model_validate_json(self.manifest_path.read_text())
        if any(getattr(manifest, key) != value for key, value in self._identity().items()):
            raise RuntimeError("operation target, fields or build differs from captured identity")
        return manifest

    def _save(self, manifest: RebuildManifest) -> None:
        _write_json(self.manifest_path, manifest.model_dump_json(indent=2))

    def _held(self) -> RebuildManifest:
        manifest = self.inspect()
        if manifest is None:
            raise RuntimeError("begin an explicit maintenance operation first")
        if manifest.state in {"released", "aborted"}:
            recovery = "choose a fresh operation directory"
            if self.marker.exists() and self.marker.read_text() == str(
                self.operation_dir.resolve()
            ):
                action = "abort" if manifest.state == "aborted" else "finish"
                recovery = (
                    f"retry {action} to release maintenance or "
                    "use replace with --previous-operation-dir"
                )
            raise RuntimeError(f"operation is {manifest.state}; {recovery}")
        if not self.marker.exists() or self.marker.read_text() != str(self.operation_dir.resolve()):
            raise RuntimeError("maintenance marker is missing or belongs to another operation")
        return manifest

    def _captured(self) -> RebuildManifest:
        manifest = self._held()
        if file_hash(self.config.db_path) != manifest.database_hash:
            raise RuntimeError(
                "canonical database changed since capture; "
                "abort the original or replace this operation"
            )
        if Path(str(self.config.db_path) + ".wal").exists():
            raise RuntimeError(
                "canonical database has a WAL after capture; "
                "abort the original or replace this operation"
            )
        backup = Path(manifest.backup)
        if (
            file_hash(backup / "recall.duckdb") != manifest.database_hash
            or file_hash(backup / "recall.fts.sqlite") != manifest.original_hash
        ):
            raise RuntimeError("matched rollback backup changed")
        return manifest

    def _run(self, action: Callable[[], RebuildManifest]) -> RebuildManifest:
        if not self.config.db_path.is_file():
            raise RuntimeError("canonical database does not exist")
        with (
            advisory_lock(self.config.lock_path),
            closing(connect_existing(self.config, read_only=False)),
        ):
            return action()

    def begin(self) -> RebuildManifest:
        return self._run(self._begin)

    def replace(self, previous_operation_dir: Path) -> RebuildManifest:
        return self._run(lambda: self._begin(previous_operation_dir.resolve()))

    def abort(self) -> RebuildManifest:
        return self._run(self._abort)

    def build(self) -> RebuildManifest:
        return self._run(self._build)

    def install(self) -> RebuildManifest:
        return self._run(self._install)

    def rollback(self) -> RebuildManifest:
        return self._run(self._rollback)

    def finish(self) -> RebuildManifest:
        return self._run(self._finish)

    def _begin(self, previous_operation_dir: Path | None = None) -> RebuildManifest:
        """Capture a consistent matched backup under exclusive offline ownership."""
        if self.inspect() is not None:
            manifest = self._captured()
            if previous_operation_dir is not None and manifest.replaces_operation != str(
                previous_operation_dir
            ):
                raise RuntimeError("named previous operation differs from captured replacement")
            return manifest
        if not self.config.db_path.is_file() or not self.sidecar.is_file():
            raise RuntimeError("initialization requires an existing database and sidecar")
        if self.operation_dir.resolve().parent.stat().st_dev != self.sidecar.stat().st_dev:
            raise RuntimeError("candidate and sidecar must reside on the same filesystem")
        required = 3 * self.config.db_path.stat().st_size + 4 * self.sidecar.stat().st_size
        if shutil.disk_usage(self.config.data_dir).free < required:
            raise RuntimeError(f"insufficient disk space: need at least {required} bytes")
        self.operation_dir.mkdir(exist_ok=True)
        if self.candidate.exists():
            raise RuntimeError("unidentified candidate exists; choose a fresh operation directory")
        owner = str(self.operation_dir.resolve())
        if self.marker.exists() and self.marker.read_text() != owner:
            if previous_operation_dir is None:
                raise RuntimeError(
                    "another keyword maintenance operation owns this target; "
                    "use replace with --previous-operation-dir"
                )
            if self.marker.read_text() != str(previous_operation_dir):
                raise RuntimeError("named previous operation does not own the maintenance marker")
            previous = RebuildManifest.model_validate_json(
                (previous_operation_dir / "manifest.json").read_text()
            )
            if any(
                getattr(previous, key) != value
                for key, value in self._identity().items()
                if key in {"data_dir", "db_path", "lock_path"}
            ):
                raise RuntimeError("named previous operation belongs to another target")
        elif previous_operation_dir is not None and not self.marker.exists():
            raise RuntimeError("named previous operation does not own the maintenance marker")
        # No ordinary writer may begin after the marker is durable. DuckDB's
        # exclusive file lock rejects a still-running daemon before capture.
        _write_json(self.marker, owner)
        with closing(connect_existing(self.config, read_only=False)) as source:
            self._standalone_hash()
            backup = consistent_backup(source, self.config, kind="keyword-rebuild")
        # Retain byte-exact rollback after the backup API has captured the pair.
        shutil.copyfile(self.sidecar, backup / "recall.fts.sqlite")
        _sync(backup / "recall.fts.sqlite")
        manifest = RebuildManifest.model_validate(
            dict(
                **self._identity(),
                backup=str(backup),
                database_hash=file_hash(self.config.db_path),
                original_hash=file_hash(backup / "recall.fts.sqlite"),
                capture=_capture_identity(backup),
                replaces_operation=str(previous_operation_dir) if previous_operation_dir else None,
            )
        )
        self._save(manifest)
        return manifest

    def _build(self) -> RebuildManifest:
        manifest = self._captured()
        if manifest.state != "building":
            return manifest
        # Always export the immutable matched copy, never a changing live handle.
        with (
            closing(
                duckdb.connect(str(Path(manifest.backup) / "recall.duckdb"), read_only=True)
            ) as source,
            closing(open_sidecar(self.candidate)) as target,
        ):
            bootstrap_sidecar(
                source,
                target,
                fts_fields=self.config.fts.fields,
                batch_size=BATCH_SIZE,
                progress_callback=lambda progress: logger.info(
                    "keyword-build operation=%s messages=%d tools=%d",
                    self.operation_dir,
                    progress.messages_processed,
                    progress.tool_calls_processed,
                ),
            )
            set_fts_fields_signature(target, self.config.fts.fields)
            verify_candidate(source, target, self.config.fts.fields)
            target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        _sync(self.candidate)
        manifest = manifest.model_copy(
            update={"candidate_hash": file_hash(self.candidate), "state": "verified"}
        )
        self._save(manifest)
        return manifest

    def _install(self) -> RebuildManifest:
        manifest = self._captured()
        current_hash = self._standalone_hash()
        if manifest.state == "installed":
            if current_hash != manifest.candidate_hash:
                raise RuntimeError("installed sidecar changed before release")
            return manifest
        if manifest.state != "verified" or manifest.candidate_hash is None:
            raise RuntimeError("only a verified candidate can be installed")
        # Replacement may have committed before the process saved its state.
        if current_hash != manifest.candidate_hash:
            if current_hash != manifest.original_hash:
                raise RuntimeError("current sidecar differs from both original and candidate")
            if file_hash(self.candidate) != manifest.candidate_hash:
                raise RuntimeError("candidate changed after verification")
            os.replace(self.candidate, self.sidecar)
            _sync(self.sidecar.parent)
            _sync(self.operation_dir)
        manifest = manifest.model_copy(update={"state": "installed"})
        self._save(manifest)
        return manifest

    def _standalone_hash(self) -> str:
        # Finish the OLD file's journal before hashing or replacing it. The
        # exclusive transaction also detects readers/writers outside maintenance.
        with closing(sqlite3.connect(self.sidecar, timeout=0)) as conn:
            result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result is not None and result[0] != 0:
                raise RuntimeError("old sidecar WAL is busy")
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("BEGIN EXCLUSIVE")
            conn.rollback()
        for suffix in ("-wal", "-shm", "-journal"):
            if Path(str(self.sidecar) + suffix).exists():
                raise RuntimeError(f"old sidecar still has {suffix}; installation refused")
        return file_hash(self.sidecar)

    def _rollback(self) -> RebuildManifest:
        manifest = self._captured()
        current = self._standalone_hash()
        if current not in {manifest.original_hash, manifest.candidate_hash}:
            raise RuntimeError("current sidecar changed; sidecar-only rollback is unsafe")
        if current != manifest.original_hash:
            restored = self.operation_dir / "rollback.sqlite"
            shutil.copyfile(Path(manifest.backup) / "recall.fts.sqlite", restored)
            _sync(restored)
            os.replace(restored, self.sidecar)
            _sync(self.sidecar.parent)
            _sync(self.operation_dir)
        manifest = manifest.model_copy(update={"state": "rolled_back"})
        self._save(manifest)
        return manifest

    def _release_marker(self) -> None:
        if self.marker.exists():
            if self.marker.read_text() != str(self.operation_dir.resolve()):
                raise RuntimeError("another operation owns the maintenance marker")
            self.marker.unlink()
            _sync(self.marker.parent)

    def _abort(self) -> RebuildManifest:
        manifest = self.inspect()
        if manifest is None or manifest.state != "aborted":
            manifest = self._held()
        # No restoration: the current canonical generation and backup validity
        # are irrelevant when leaving the byte-exact original in place.
        if self._standalone_hash() != manifest.original_hash:
            raise RuntimeError(
                "abort requires the original sidecar; use replace with --previous-operation-dir"
            )
        manifest = manifest.model_copy(update={"state": "aborted"})
        self._save(manifest)
        self._release_marker()
        return manifest

    def _finish(self) -> RebuildManifest:
        manifest = self.inspect()
        if manifest is not None and manifest.state == "released":
            self._release_marker()
            return manifest
        manifest = self._captured()
        if manifest.state not in {"installed", "rolled_back"}:
            raise RuntimeError("install or roll back before releasing maintenance")
        expected = (
            manifest.candidate_hash if manifest.state == "installed" else manifest.original_hash
        )
        if self._standalone_hash() != expected:
            raise RuntimeError("installed sidecar hash differs before maintenance release")
        # Save release first. A crash here leaves writers blocked, never permits
        # a later sidecar-only rollback after writers could have resumed.
        manifest = manifest.model_copy(update={"state": "released"})
        self._save(manifest)
        self._release_marker()
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("inspect", "begin", "replace", "abort", "build", "install", "rollback", "finish"),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--operation-dir", type=Path, required=True)
    parser.add_argument("--previous-operation-dir", type=Path)
    parser.add_argument("--build-id", required=True)
    parser.add_argument(
        "--fields", required=True, help="comma-separated content,thinking,bash; empty allowed"
    )
    args = parser.parse_args()
    if (args.action == "replace") != (args.previous_operation_dir is not None):
        parser.error("replace requires --previous-operation-dir; other actions do not accept it")
    logging.basicConfig(level=logging.INFO)
    fields = tuple(field for field in args.fields.split(",") if field)
    if set(fields) - {"content", "thinking", "bash"}:
        parser.error("unsupported keyword field")
    config = AppConfig(
        data_dir=args.data_dir.resolve(),
        db_path=args.db_path.resolve(),
        lock_path=args.lock_path.resolve(),
        config_path=args.operation_dir / "unused.toml",
        fts=FtsConfig(fields=fields),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )
    operation = KeywordRebuild(config, args.operation_dir.resolve(), args.build_id)
    try:
        if args.action == "inspect":
            result = operation.inspect()
        elif args.action == "replace":
            result = operation.replace(args.previous_operation_dir)
        else:
            result = getattr(operation, args.action)()
    except (OSError, RuntimeError, ValueError, sqlite3.Error, duckdb.Error) as err:
        print(
            json.dumps(
                {
                    "outcome": "failed",
                    "operation_dir": str(operation.operation_dir),
                    "action": args.action,
                    "error": str(err),
                }
            )
        )
        raise SystemExit(2) from err
    print(
        json.dumps(
            {
                "outcome": "absent" if result is None else result.state,
                "operation_dir": str(operation.operation_dir),
                "manifest": result.model_dump() if result is not None else None,
            }
        )
    )


if __name__ == "__main__":
    main()
