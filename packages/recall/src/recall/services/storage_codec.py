"""Codec-profile maintenance artifacts owned by the existing storage lifecycle.

The caller holds the sole database owner and drains/closes shared handles before
publication. No recovery path restores an older snapshot over durable progress.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from dataclasses import replace
from itertools import zip_longest
from pathlib import Path
from typing import Literal

import duckdb
from pydantic import BaseModel, ConfigDict

from recall.core.config import AppConfig
from recall.db.connection import (
    STORAGE_VERSION,
    _apply_runtime_pragmas,
    _sql_string_literal,
    storage_version,
    wal_size_bytes,
)
from recall.services.recreation import consistent_backup

CODEC_PROFILE = "automatic-without-roaring-v1"
# Maintenance is bounded separately from normal writer turns. Qualification
# records the measured headroom; these ceilings do not certify a host for rollout.
COPY_TIMEOUT_SECONDS = 600
VALIDATION_TIMEOUT_SECONDS = 3600


class CodecJournal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_version: Literal[1] = 1
    codec_profile: Literal["automatic-without-roaring-v1"] = CODEC_PROFILE
    operation_id: str
    plan_id: str
    db_path: str
    backup_path: str
    candidate_path: str
    source_sha256: str
    sidecar_sha256: str | None
    backup_hashes: dict[str, str]
    candidate_sha256: str | None = None
    phase: Literal["prepared", "copying", "validating", "publishing", "complete"]


def quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def codec_required(conn: duckdb.DuckDBPyConnection) -> bool:
    """Inspect actual persisted segments, including writes from an older binary."""
    for schema, table in conn.execute(
        "SELECT schema_name, table_name FROM duckdb_tables() "
        "WHERE database_name=current_database() AND NOT internal"
    ).fetchall():
        if conn.execute(
            "SELECT 1 FROM pragma_storage_info(?) WHERE lower(compression)='roaring' LIMIT 1",
            [f"{quote(schema)}.{quote(table)}"],
        ).fetchone():
            return True
    return False


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sidecar_digest(config: AppConfig) -> str | None:
    path = config.data_dir / "recall.fts.sqlite"
    # A codec operation never replaces SQLite. Refuse unsettled journals rather
    # than claiming a main-file hash covers committed WAL content.
    for suffix in ("-wal", "-journal"):
        journal = Path(f"{path}{suffix}")
        if journal.exists() and journal.stat().st_size:
            raise RuntimeError("SQLite journals must be settled before codec maintenance")
    return digest(path) if path.exists() else None


def settle_sidecar(config: AppConfig) -> None:
    """Recover/checkpoint committed SQLite WAL before capturing its preimage.

    Only explicit maintenance calls this, under the existing writer/owner lock.
    SQLite owns hot-journal recovery; no journal is deleted or ignored.
    """
    path = config.data_dir / "recall.fts.sqlite"
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as conn:
        state = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if state not in {(0, 0, 0), (0, -1, -1)}:
            raise RuntimeError("SQLite checkpoint is busy; drain sidecar readers/writers and retry")
    sidecar_digest(config)


def journal_path(config: AppConfig) -> Path:
    return config.data_dir / "snapshots" / "storage-codec-state" / "journal.json"


def load_journal(config: AppConfig) -> CodecJournal | None:
    path = journal_path(config)
    if not path.exists():
        return None
    journal = CodecJournal.model_validate_json(path.read_text())
    if journal.db_path != str(config.db_path.resolve()):
        raise RuntimeError("codec recovery journal belongs to a different database")
    return journal


def fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_journal(config: AppConfig, journal: CodecJournal) -> None:
    path = journal_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(journal.model_dump_json(indent=2))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_path(path.parent)
    fsync_path(path.parent.parent)


def verify_backup(journal: CodecJournal) -> None:
    backup = Path(journal.backup_path)
    manifest = json.loads((backup / "manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("kind") != "storage-codec":
        raise RuntimeError("verified codec backup required")
    if set(manifest.get("files", [])) != set(journal.backup_hashes):
        raise RuntimeError("codec backup inventory changed")
    for name, expected in journal.backup_hashes.items():
        if name not in {"recall.duckdb", "recall.fts.sqlite"}:
            raise RuntimeError("invalid codec backup member")
        if digest(backup / name) != expected:
            raise RuntimeError(f"codec backup hash mismatch: {name}")


def validate_sidecar_backup(config: AppConfig, backup: Path) -> None:
    """Compare SQLite's complete logical image, including FTS shadow/repair state."""
    path = config.data_dir / "recall.fts.sqlite"
    if not path.exists():
        if (backup / path.name).exists():
            raise RuntimeError("codec backup has an unexpected sidecar")
        return
    deadline = time.monotonic() + VALIDATION_TIMEOUT_SECONDS
    with (
        closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source,
        closing(sqlite3.connect((backup / path.name).as_uri() + "?mode=ro", uri=True)) as target,
    ):
        for conn in (source, target):
            conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10000)
        for before, after in zip_longest(source.iterdump(), target.iterdump()):
            if time.monotonic() >= deadline:
                raise TimeoutError("codec sidecar validation deadline exceeded")
            if before != after:
                raise RuntimeError("codec backup sidecar logical mismatch")


def prepare_codec(
    conn: duckdb.DuckDBPyConnection, config: AppConfig, *, operation_id: str, plan_id: str
) -> CodecJournal:
    existing = load_journal(config)
    if existing is not None and existing.phase != "complete":
        verify_backup(existing)
        return existing
    if storage_version(conn) != f"{STORAGE_VERSION}+":
        raise RuntimeError("complete fixed-format maintenance before inspecting a codec plan")
    row = conn.execute("SELECT phase FROM index_migration_jobs WHERE singleton").fetchone()
    if row != ("idle",):
        raise RuntimeError("complete original format/index job before codec maintenance")
    # Need backup pair + candidate + spill/WAL margin simultaneously. This is a
    # conservative admission bound, not a steady-state compression prediction.
    source_bytes = config.db_path.stat().st_size
    sidecar = config.data_dir / "recall.fts.sqlite"
    sidecar_bytes = sidecar.stat().st_size if sidecar.exists() else 0
    required_bytes = 3 * source_bytes + sidecar_bytes + 2 * 1024**3
    if shutil.disk_usage(config.db_path.parent).free < required_bytes:
        raise RuntimeError(
            f"insufficient codec maintenance space: need {required_bytes} free bytes"
        )
    settle_sidecar(config)
    backup = consistent_backup(conn, config, kind="storage-codec")
    hashes = {p.name: digest(p) for p in backup.iterdir() if p.name != "manifest.json"}
    source_hash = digest(config.db_path)
    if hashes.get("recall.duckdb") != source_hash:
        raise RuntimeError("codec backup does not match checkpointed source")
    side_hash = sidecar_digest(config)
    validate_sidecar_backup(config, backup)
    # SQLite backup can have a different physical layout. Compare its logical
    # image below, while the authoritative file hash binds publication.
    operation_dir = config.db_path.parent / f".storage-codec-{plan_id}"
    operation_dir.mkdir(exist_ok=False)
    journal = CodecJournal(
        operation_id=operation_id,
        plan_id=plan_id,
        db_path=str(config.db_path.resolve()),
        backup_path=str(backup.resolve()),
        candidate_path=str((operation_dir / config.db_path.name).resolve()),
        source_sha256=source_hash,
        sidecar_sha256=side_hash,
        backup_hashes=hashes,
        phase="prepared",
    )
    verify_backup(journal)
    save_journal(config, journal)
    return journal


@contextmanager
def query_deadline(conn: duckdb.DuckDBPyConnection, seconds: int):
    expired = threading.Event()

    def interrupt() -> None:
        expired.set()
        conn.interrupt()

    timer = threading.Timer(seconds, interrupt)
    timer.daemon = True
    timer.start()
    try:
        yield
        if expired.is_set():
            raise TimeoutError("codec maintenance query deadline exceeded")
    finally:
        timer.cancel()
        timer.join()


def _array_equal(left, right) -> bool:
    """Exact Arrow values; float equality compares bits, including NaN payloads."""
    import pyarrow as pa

    if left.type != right.type or len(left) != len(right):
        return False
    if not left.is_null().equals(right.is_null()):
        return False
    if pa.types.is_floating(left.type):
        integer = pa.uint32() if left.type.bit_width == 32 else pa.uint64()
        return left.view(integer).equals(right.view(integer))
    if pa.types.is_fixed_size_list(left.type) or pa.types.is_list(left.type):
        # flatten omits children of NULL parents, preserving NULL vs all-NULL.
        return (
            _array_equal(left.value_lengths(), right.value_lengths())
            and _array_equal(left.flatten(), right.flatten())
            if pa.types.is_list(left.type)
            else _array_equal(left.flatten(), right.flatten())
        )
    if pa.types.is_struct(left.type):
        return all(_array_equal(a, b) for a, b in zip(left.flatten(), right.flatten(), strict=True))
    return left.equals(right)


def _catalog(conn: duckdb.DuckDBPyConnection) -> list[tuple]:
    # SQL definitions cover constraints/defaults/indexes/macros/views/types and
    # sequences, rather than just the tables copied by the engine.
    results = []
    for function, name in (
        ("duckdb_tables", "table_name"),
        ("duckdb_indexes", "index_name"),
        ("duckdb_views", "view_name"),
        ("duckdb_sequences", "sequence_name"),
    ):
        internal = " AND NOT internal" if function in {"duckdb_tables", "duckdb_views"} else ""
        results.extend(
            conn.execute(
                f"SELECT '{function}', schema_name, {name}, sql FROM {function}() "
                f"WHERE database_name=current_database(){internal} ORDER BY 2,3"
            ).fetchall()
        )
    results.extend(
        conn.execute(
            "SELECT 'macro', schema_name, function_name, macro_definition, "
            "parameters, parameter_types "
            "FROM duckdb_functions() WHERE database_name=current_database() "
            "AND function_type IN ('macro','table_macro') ORDER BY 2,3,4"
        ).fetchall()
    )
    results.extend(
        conn.execute(
            "SELECT 'type', schema_name, type_name, logical_type, labels FROM duckdb_types() "
            "WHERE database_name=current_database() AND NOT internal ORDER BY 2,3"
        ).fetchall()
    )
    return results


def _rows_equal(source, target, sql: str, parameters: list) -> bool:
    a = source.execute(sql, parameters).to_arrow_reader(256)
    b = target.execute(sql, parameters).to_arrow_reader(256)
    for before, after in zip_longest(a, b):
        if (
            before is None
            or after is None
            or before.schema != after.schema
            or any(
                not _array_equal(x, y) for x, y in zip(before.columns, after.columns, strict=True)
            )
        ):
            return False
    return True


def validate_candidate(config: AppConfig, candidate: Path) -> None:
    import pyarrow as pa

    with (
        duckdb.connect(str(config.db_path), read_only=True) as source,
        duckdb.connect(str(candidate), read_only=True) as target,
    ):
        for conn in (source, target):
            _apply_runtime_pragmas(conn, config)
            conn.execute("SET threads=1")
            conn.execute("SET memory_limit='768MiB'")
        for conn, role in ((source, "source"), (target, "candidate")):
            spill = candidate.parent / f"validate-{role}-spill"
            conn.execute(f"SET temp_directory='{_sql_string_literal(spill)}'")
        with (
            query_deadline(source, VALIDATION_TIMEOUT_SECONDS),
            query_deadline(target, VALIDATION_TIMEOUT_SECONDS),
        ):
            if storage_version(source) != storage_version(target) or _catalog(source) != _catalog(
                target
            ):
                raise RuntimeError("codec candidate catalog or header mismatch")
            if codec_required(target):
                raise RuntimeError("codec candidate still contains Roaring segments")
            tables = source.execute(
                "SELECT schema_name, table_name FROM duckdb_tables() "
                "WHERE database_name=current_database() AND NOT internal ORDER BY 1,2"
            ).fetchall()
            for schema, table in tables:
                name = f"{quote(schema)}.{quote(table)}"
                # A one-worker physical copy commonly retains row order. An
                # exact comparison of all values (including keys) proves logical
                # equality directly. Changed order falls back to keyed/ordered
                # comparison; no physical-order assumption grants admission.
                if _rows_equal(source, target, f"SELECT * FROM {name}", []):
                    continue
                key = source.execute(
                    "SELECT constraint_column_names FROM duckdb_constraints() "
                    "WHERE database_name=current_database() AND schema_name=? AND table_name=? "
                    "AND constraint_type='PRIMARY KEY'",
                    [schema, table],
                ).fetchone()
                if key:
                    # Scan the historical file once in physical order. Sorting
                    # only this page avoids a wide global sort and repeated
                    # random decoding of the source's old validity masks.
                    columns = list(key[0])
                    names = ",".join(quote(column) for column in columns)
                    if (
                        source.execute(f"SELECT count(*) FROM {name}").fetchone()
                        != target.execute(f"SELECT count(*) FROM {name}").fetchone()
                    ):
                        raise RuntimeError(f"codec candidate exact row mismatch: {name}")
                    pages = source.execute(f"SELECT * FROM {name}").to_arrow_reader(256)
                    for page in pages:
                        before = pa.Table.from_batches([page]).sort_by(
                            [(column, "ascending") for column in columns]
                        )
                        key_values = [before.column(column).to_pylist() for column in columns]
                        values = [value for row in zip(*key_values, strict=True) for value in row]
                        row_placeholder = "(" + ",".join("?" for _ in columns) + ")"
                        placeholders = ",".join(row_placeholder for _ in range(len(before)))
                        sql = (
                            f"SELECT * FROM {name} WHERE ({names}) IN ({placeholders}) "
                            f"ORDER BY {names}"
                        )
                        after = target.execute(sql, values).to_arrow_table()
                        if (
                            before.schema != after.schema
                            or len(before) != len(after)
                            or any(
                                not _array_equal(a.combine_chunks(), b.combine_chunks())
                                for a, b in zip(before.columns, after.columns, strict=True)
                            )
                        ):
                            raise RuntimeError(f"codec candidate exact row mismatch: {name}")
                    continue
                order = ",".join(quote(k) for k in key[0]) if key else "ALL"
                sql = f"SELECT * FROM {name} ORDER BY {order}"
                if not _rows_equal(source, target, sql, []):
                    raise RuntimeError(f"codec candidate exact row mismatch: {name}")


def recover_codec(config: AppConfig) -> None:
    """Identify interrupted publication before any ordinary open can write.

    Unknown/later progress is never overwritten. Complete journals do not grant
    admission: the owner also inspects actual segments after opening the file.
    """
    journal = load_journal(config)
    if journal is None:
        return
    if not config.db_path.is_file():
        raise RuntimeError("codec recovery holds: database missing; restore explicitly from backup")
    if journal.phase == "complete":
        return
    verify_backup(journal)
    if not config.db_path.is_file() or wal_size_bytes(config.db_path):
        raise RuntimeError(
            "codec recovery holds: missing database or later durable WAL; inspect recovery"
        )
    active = digest(config.db_path)
    if active == journal.source_sha256:
        return  # Explicit maintenance resumes; never copy/replace on startup.
    if journal.phase == "publishing" and active == journal.candidate_sha256:
        if sidecar_digest(config) != journal.sidecar_sha256:
            raise RuntimeError("codec recovery holds: sidecar changed")
        with duckdb.connect(str(config.db_path), read_only=True) as reader:
            if codec_required(reader):
                raise RuntimeError("codec recovery holds: published profile mismatch")
        fsync_path(config.db_path.parent)
        fsync_path(Path(journal.candidate_path).parent)
        save_journal(config, journal.model_copy(update={"phase": "complete"}))
        return
    raise RuntimeError(
        "codec recovery holds: authoritative database changed; "
        "preserve current files and inspect recovery"
    )


def perform_codec_transition(config: AppConfig) -> None:
    journal = load_journal(config)
    if journal is None:
        raise RuntimeError("verified codec backup and intent required")
    recover_codec(config)
    journal = load_journal(config)
    assert journal is not None
    if journal.phase == "complete":
        return
    verify_backup(journal)
    if sidecar_digest(config) != journal.sidecar_sha256:
        raise RuntimeError("codec sidecar changed since backup")
    candidate = Path(journal.candidate_path)
    if candidate.exists():
        # Retain interrupted artifacts for diagnosis; a fresh attempt uses a
        # distinct sibling, never mutating or trusting a partially copied file.
        attempt = 1
        while candidate.with_name(f"attempt-{attempt}").exists():
            attempt += 1
        archive = candidate.with_name(f"attempt-{attempt}")
        archive.mkdir()
        os.replace(candidate, archive / candidate.name)
        wal = Path(f"{candidate}.wal")
        if wal.exists():
            os.replace(wal, archive / wal.name)
    journal = journal.model_copy(update={"phase": "copying", "candidate_sha256": None})
    save_journal(config, journal)
    with duckdb.connect(
        str(candidate), config={"storage_compatibility_version": STORAGE_VERSION}
    ) as conn:
        _apply_runtime_pragmas(conn, replace(config, data_dir=candidate.parent))
        conn.execute("SET threads=1")
        conn.execute("SET memory_limit='2GB'")
        with query_deadline(conn, COPY_TIMEOUT_SECONDS):
            conn.execute(f"ATTACH '{_sql_string_literal(config.db_path)}' AS original (READ_ONLY)")
            target = conn.execute("SELECT current_database()").fetchone()
            assert target is not None
            conn.execute(f"COPY FROM DATABASE original TO {quote(target[0])}")
            conn.execute("CHECKPOINT")
    journal = journal.model_copy(update={"phase": "validating"})
    save_journal(config, journal)
    # Both stores are reopened through the release engine; comparison streams
    # every stored row and float bit, without regeneration or normalization.
    validate_candidate(config, candidate)
    if digest(config.db_path) != journal.source_sha256 or wal_size_bytes(config.db_path):
        raise RuntimeError("codec source changed during conversion")
    if sidecar_digest(config) != journal.sidecar_sha256:
        raise RuntimeError("codec sidecar changed during conversion")
    if wal_size_bytes(candidate):
        raise RuntimeError("codec candidate WAL not settled")
    verify_backup(journal)
    fsync_path(candidate)
    fsync_path(candidate.parent)
    journal = journal.model_copy(
        update={"phase": "publishing", "candidate_sha256": digest(candidate)}
    )
    save_journal(config, journal)
    os.replace(candidate, config.db_path)
    fsync_path(config.db_path.parent)
    fsync_path(candidate.parent)
    recover_codec(config)
