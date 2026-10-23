"""Fenced, backed-up initialization and explicit metadata-only repair."""

from __future__ import annotations

import errno
import hashlib
import json
import shutil
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

import duckdb

from recall.core.config import AppConfig
from recall.core.daemon_status import SemanticWorkStatus
from recall.core.keyword_maintenance import require_keyword_writes_enabled
from recall.core.semantic_maintenance import (
    MARKER_NAME,
    RETAINED_PREIMAGES_MAX,
    PairIdentity,
    RepairPlan,
    RepairResult,
    SemanticAdmission,
    SemanticIntent,
    SemanticMaintenance,
    file_hash,
    read_intent,
    save_admission,
    write_durable,
)
from recall.db.connection import _apply_runtime_pragmas, resolve_temp_directory
from recall.db.migrations import run_pending_migrations
from recall.db.schema import get_schema_version
from recall.db.semantic_work import (
    COUNT_COLUMNS,
    DEFINITION_VERSION,
    INITIALIZATION_CHUNK,
    SESSION_CAPACITY,
    TARGET_CAPACITY,
    SemanticSchemaUpgrade,
    SemanticWorkUnavailable,
    _grant_write_owner,
    audited_transaction,
    capture_identity,
    expected_work_sql,
    grouped_counts_sql,
    install_build_state,
    install_schema,
    invalidate_readers,
)
from recall.services.recreation import consistent_backup

AUDIT_DEADLINE_SECONDS = 300


@contextmanager
def _audit_deadline(conn: duckdb.DuckDBPyConnection):
    timer = threading.Timer(AUDIT_DEADLINE_SECONDS, conn.interrupt)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        timer.join()


_PREFIX = "recreate-semantic-work-"
_CANONICAL = (
    ("sessions", "id"),
    ("session_state", "session_id"),
    ("messages", "id"),
    ("message_state", "message_id"),
    ("tool_calls", "id"),
    ("message_embeddings", "message_id"),
    ("tool_call_embeddings", "tool_call_id"),
)


def _pair_identity(directory: Path, db_path: Path | None = None) -> PairIdentity:
    sidecar = directory / "recall.fts.sqlite"
    return PairIdentity(
        database=file_hash(db_path or directory / "recall.duckdb"),
        sidecar=file_hash(sidecar) if sidecar.exists() else None,
    )


def _canonical_digest(conn: duckdb.DuckDBPyConnection, *, empty_vectors: bool = False) -> str:
    digest = hashlib.sha256()
    deadline = time.monotonic() + AUDIT_DEADLINE_SECONDS
    for table, key in _CANONICAL:
        digest.update(table.encode())
        if empty_vectors and table in {"message_embeddings", "tool_call_embeddings"}:
            continue
        relation = conn.sql(f"SELECT sha256(CAST(t AS VARCHAR)) FROM {table} t ORDER BY {key}")
        try:
            while rows := relation.fetchmany(INITIALIZATION_CHUNK):
                if time.monotonic() > deadline:
                    raise SemanticWorkUnavailable("canonical scope verification deadline exceeded")
                for (value,) in rows:
                    digest.update(bytes.fromhex(value))
        finally:
            relation.close()
    return digest.hexdigest()


def _sidecar_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        for statement in conn.iterdump():
            digest.update(statement.encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _settle_sidecar(config: AppConfig) -> None:
    path = config.data_dir / "recall.fts.sqlite"
    if path.exists():
        with closing(sqlite3.connect(path)) as sidecar:
            result = sidecar.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result is None or result[0] != 0:
                raise SemanticWorkUnavailable(
                    "keyword sidecar checkpoint busy; finish its owning operation"
                )


class SemanticLifecycle:
    """The advisory/lifecycle owner calls this while every ordinary writer is fenced."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.epoch = uuid4().hex
        self.intent: SemanticIntent | None = None
        self.phase = "auditing"
        self.error: str | None = None
        self.admitted = False
        self._plan: RepairPlan | None = None
        self.completed_targets: int | None = None

    def _admission_state(self, phase: Literal["auditing", "held", "open"]) -> None:
        self.admission = SemanticAdmission(
            epoch=self.epoch,
            db_path=str(self.config.db_path.resolve()),
            phase=phase,
            error=self.error,
        )
        save_admission(self.config.data_dir, self.admission)

    def hold(self) -> SemanticMaintenance:
        assert self.admitted
        self.admitted = False
        self.phase = "auditing"
        self._admission_state("held")
        return SemanticMaintenance(str(self.config.db_path.resolve()), self.epoch)

    def _admit(self) -> None:
        self._admission_state("open")
        self.admitted = True

    def diagnostics(self) -> SemanticWorkStatus:
        intent = self.intent
        return {
            "definition_version": DEFINITION_VERSION,
            "phase": self.phase,
            "build_id": intent.build_id if intent else None,
            "completed_targets": self.completed_targets,
            "captured_messages": intent.captured_messages if intent else None,
            "captured_tools": intent.captured_tools if intent else None,
            "started_at": intent.started_at if intent else None,
            "verified_at": intent.verified_at if intent else None,
            "error": self.error,
        }

    def _save(self, intent: SemanticIntent) -> None:
        write_durable(self.config.data_dir / MARKER_NAME, intent.model_dump_json(indent=2))
        self.intent = intent
        self.phase = intent.phase
        self.error = intent.error
        if intent.backup_path is not None:
            write_durable(
                Path(intent.backup_path) / "semantic-intent.json", intent.model_dump_json(indent=2)
            )

    def _retained(self) -> tuple[tuple[str, int], ...]:
        paths = sorted((self.config.data_dir / "snapshots").glob(_PREFIX + "*"))
        # Directory admission is bounded before creating another capture; failed copies count too.
        if len(paths) > RETAINED_PREIMAGES_MAX:
            raise SemanticWorkUnavailable(
                f"{len(paths)}"
                " semantic preimages exceeds capacity "
                f"{RETAINED_PREIMAGES_MAX}"
                "; explicit human archive/retirement required"
            )
        return tuple(
            (str(p), sum(f.stat().st_size for f in p.iterdir() if f.is_file())) for p in paths
        )

    def _admission(
        self, conn: duckdb.DuckDBPyConnection
    ) -> tuple[int, tuple[tuple[str, int], ...]]:
        require_keyword_writes_enabled(self.config.data_dir)
        version = get_schema_version(conn)
        if version not in (32, 33):
            raise SemanticWorkUnavailable(
                f"unsupported schema {version}; use the matching approved binary or human recovery"
            )
        job = conn.execute("SELECT phase FROM index_migration_jobs WHERE singleton").fetchone()
        if job is not None and job[0] != "idle":
            raise SemanticWorkUnavailable(
                "source/index/storage maintenance owns the database; finish that "
                "operation before semantic initialization"
            )
        counts = conn.execute(
            "WITH pending AS (" + expected_work_sql() + ") "
            "SELECT count(*),count(DISTINCT session_id) FROM pending"
        ).fetchone()
        assert counts is not None
        if counts[0] > TARGET_CAPACITY or counts[1] > SESSION_CAPACITY:
            raise SemanticWorkUnavailable(
                "semantic capacity requires <= "
                f"{TARGET_CAPACITY}"
                " targets/"
                f"{SESSION_CAPACITY}"
                " sessions; observed "
                f"{counts}"
            )
        retained = self._retained()
        if len(retained) >= RETAINED_PREIMAGES_MAX:
            raise SemanticWorkUnavailable(
                "semantic preimage capacity "
                f"{RETAINED_PREIMAGES_MAX}"
                ": "
                f"{retained}"
                "; explicit human retirement required"
            )
        pair_bytes = self.config.db_path.stat().st_size
        sidecar = self.config.data_dir / "recall.fts.sqlite"
        if sidecar.exists():
            pair_bytes += sidecar.stat().st_size
        required = 2 * pair_bytes + int(counts[0]) * 512 + int(counts[1]) * 256
        # Account for each filesystem separately, including spill and snapshot reserve.
        for location in (
            self.config.db_path.parent,
            self.config.data_dir / "snapshots",
            resolve_temp_directory(self.config),
        ):
            existing = location
            while not existing.exists():
                existing = existing.parent
            available = shutil.disk_usage(existing).free
            if available < required:
                raise SemanticWorkUnavailable(
                    "insufficient reserve at "
                    f"{location}"
                    ": require "
                    f"{required}"
                    ", available "
                    f"{available}"
                    "; retained="
                    f"{retained}"
                )
        return required, retained

    def _verify_backup(self, intent: SemanticIntent, *, compare_live: bool) -> None:
        if intent.backup_path is None or intent.backup_files is None:
            raise SemanticWorkUnavailable(
                "semantic preimage not completed; inspect repair-semantic-work --dry-run"
            )
        path = Path(intent.backup_path)
        if _pair_identity(path) != intent.backup_files:
            raise SemanticWorkUnavailable(
                "semantic preimage integrity mismatch; inspect repair-semantic-work --dry-run"
            )
        manifest = json.loads((path / "manifest.json").read_text())
        if not manifest.get("complete") or manifest.get("kind") != "recreate-semantic-work":
            raise SemanticWorkUnavailable("semantic backup manifest incomplete or incompatible")
        if (
            compare_live
            and _pair_identity(self.config.data_dir, self.config.db_path) != intent.live_files
        ):
            raise SemanticWorkUnavailable("unpublished preimage no longer matches live source")

    def _begin_capture(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        build_id: str,
        accepted: str | None = None,
        target_dimensions: int | None = None,
    ) -> SemanticIntent:
        self._admission(conn)
        previous = self.intent
        version = get_schema_version(conn)
        intent = SemanticIntent(
            db_path=str(self.config.db_path.resolve()),
            build_id=build_id,
            epoch=self.epoch,
            started_at=time.time(),
            phase="backup",
            source_schema_version=version,
            operation="dimensions"
            if target_dimensions is not None
            else "repair"
            if accepted is not None
            else "initialize",
            target_dimensions=target_dimensions,
            predecessor=previous.build_id if previous else None,
            accepted_plan_id=accepted,
        )
        # Retain the predecessor before exposing a successor or creating any new backup.
        if previous is not None:
            history = self.config.data_dir / "snapshots"
            history.mkdir(exist_ok=True)
            write_durable(
                history / f"semantic-intent-{previous.build_id}.json",
                previous.model_dump_json(indent=2),
            )
        marker = self.config.data_dir / MARKER_NAME
        if previous is None and marker.exists():
            try:
                read_intent(self.config.data_dir)
            except ValueError:
                archive = self.config.data_dir / "snapshots"
                archive.mkdir(exist_ok=True)
                shutil.copyfile(marker, archive / f"semantic-invalid-intent-{uuid4().hex}.json")
        self._save(intent)
        return intent

    def _capture(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        build_id: str,
        accepted: str | None = None,
        target_dimensions: int | None = None,
    ) -> SemanticIntent:
        self._admission(conn)
        if self.intent is not None and self.intent.build_id == build_id:
            intent = self.intent
        else:
            intent = self._begin_capture(
                conn, build_id=build_id, accepted=accepted, target_dimensions=target_dimensions
            )
        version = get_schema_version(conn)
        backup = consistent_backup(conn, self.config, kind="recreate-semantic-work")
        _settle_sidecar(self.config)
        conn.execute("FORCE CHECKPOINT")
        live = _pair_identity(self.config.data_dir, self.config.db_path)
        files = _pair_identity(backup)
        canonical = _canonical_digest(conn)
        with closing(duckdb.connect(str(backup / "recall.duckdb"), read_only=True)) as copied:
            _apply_runtime_pragmas(copied, self.config)
            if _canonical_digest(copied) != canonical:
                raise SemanticWorkUnavailable("backup authoritative scope differs from source")
        if _sidecar_digest(backup / "recall.fts.sqlite") != _sidecar_digest(
            self.config.data_dir / "recall.fts.sqlite"
        ):
            raise SemanticWorkUnavailable("backup keyword pair differs from source")
        counts = conn.execute(
            "SELECT (SELECT count(*) FROM messages),(SELECT count(*) FROM tool_calls)"
        ).fetchone()
        assert counts is not None
        intent = intent.model_copy(
            update={
                "backup_path": str(backup),
                "backup_files": files,
                "live_files": live,
                "canonical_digest": canonical,
                "reset_scope_digest": _canonical_digest(conn, empty_vectors=True)
                if target_dimensions is not None
                else None,
                "captured_messages": int(counts[0]),
                "captured_tools": int(counts[1]),
            }
        )
        self._save(intent)
        self._verify_backup(intent, compare_live=version == 32)
        return intent

    def _install(
        self, conn: duckdb.DuckDBPyConnection, intent: SemanticIntent, *, repair: bool
    ) -> None:
        self._verify_backup(intent, compare_live=intent.source_schema_version == 32)
        if intent.source_schema_version == 32 and not repair:
            run_pending_migrations(
                conn, 32, semantic_upgrade=SemanticSchemaUpgrade(id(conn), intent)
            )
        else:
            conn.execute("BEGIN")
            try:
                conn.execute("DROP VIEW IF EXISTS semantic_pending")
                for table in ("semantic_work_counts", "semantic_work", "semantic_work_state"):
                    conn.execute(f"DROP TABLE IF EXISTS {table}")
                install_schema(conn)
                install_build_state(conn, intent)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        self._save(intent.model_copy(update={"phase": "building"}))

    def _build(self, conn: duckdb.DuckDBPyConnection) -> None:
        intent = self.intent
        assert intent is not None
        self._verify_backup(intent, compare_live=False)
        if _canonical_digest(conn) != (intent.reset_scope_digest or intent.canonical_digest):
            raise SemanticWorkUnavailable(
                "published semantic scope changed; explicit metadata repair required"
            )
        conn.execute("CREATE OR REPLACE TEMP TABLE _semantic_capture AS " + expected_work_sql())
        state = conn.execute(
            "SELECT build_id,cursor_kind,cursor_id,completed_targets FROM "
            "semantic_work_state WHERE singleton"
        ).fetchone()
        if state is None or state[0] != intent.build_id:
            raise SemanticWorkUnavailable(
                "semantic intent/state build identity mismatch; explicit repair required"
            )
        stored = conn.execute(
            "SELECT backup_path, source_schema_version, "
            "source_manifest_sha256, captured_messages, captured_tools FROM "
            "semantic_work_state WHERE singleton"
        ).fetchone()
        if stored != (
            intent.backup_path,
            intent.source_schema_version,
            capture_identity(intent),
            intent.captured_messages,
            intent.captured_tools,
        ):
            raise SemanticWorkUnavailable(
                "semantic captured scope identity mismatch; explicit metadata repair required"
            )
        self.completed_targets = int(state[3])
        cursor = (state[1] or "", state[2] or "")
        while True:
            conn.execute(
                (
                    "CREATE OR REPLACE TEMP TABLE _semantic_chunk AS SELECT * FROM "
                    "_semantic_capture WHERE (kind,id)>(?,?) ORDER BY kind,id LIMIT ?"
                ),
                [*cursor, INITIALIZATION_CHUNK],
            )
            last = conn.execute(
                "SELECT kind,id FROM _semantic_chunk ORDER BY kind DESC,id DESC LIMIT 1"
            ).fetchone()
            if last is None:
                break
            conn.execute("BEGIN")
            try:
                conn.execute("INSERT INTO semantic_work SELECT * FROM _semantic_chunk")
                conn.execute(
                    "CREATE OR REPLACE TEMP TABLE _semantic_grouped AS "
                    + grouped_counts_sql("_semantic_chunk")
                )
                conn.execute(
                    "INSERT INTO semantic_work_counts SELECT "
                    f"{COUNT_COLUMNS}"
                    " FROM _semantic_grouped\n                    ON "
                    "CONFLICT(session_id) DO UPDATE SET\n                    "
                    "pending_messages=semantic_work_counts.pending_messages+excluded.pending_messages,"
                    "\n                    "
                    "pending_tools=semantic_work_counts.pending_tools+excluded.pending_tools,"
                    "\n                    "
                    "missing_content=semantic_work_counts.missing_content+excluded.missing_content,"
                    "\n                    "
                    "missing_thinking=semantic_work_counts.missing_thinking+excluded.missing_thinking,"
                    "\n                    "
                    "missing_bash=semantic_work_counts.missing_bash+excluded.missing_bash,"
                    "\n                    "
                    "raw_blocked_messages=semantic_work_counts.raw_blocked_messages+excluded.raw_blocked_messages"
                )
                conn.execute(
                    (
                        "UPDATE semantic_work_state SET "
                        "phase='building',cursor_kind=?,cursor_id=?,completed_targets=completed_targets+(SELECT"
                        " count(*) FROM _semantic_chunk) WHERE singleton"
                    ),
                    list(last),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            observed = conn.execute(
                "SELECT completed_targets FROM semantic_work_state WHERE singleton"
            ).fetchone()
            assert observed is not None
            self.completed_targets = int(observed[0])
            cursor = last
        self.phase = "verifying"
        conn.execute("UPDATE semantic_work_state SET phase='verifying' WHERE singleton")
        self._save(intent.model_copy(update={"phase": "verifying"}))
        self._audit(conn)
        conn.execute("CHECKPOINT")
        assert self.intent is not None
        self._save(
            self.intent.model_copy(
                update={"phase": "complete", "error": None, "verified_at": time.time()}
            )
        )
        self._admit()

    def _reset_dimensions(self, conn: duckdb.DuckDBPyConnection, intent: SemanticIntent) -> None:
        from recall.db.schema import _recreate_embedding_tables

        assert intent.operation == "dimensions" and intent.target_dimensions is not None
        self._verify_backup(intent, compare_live=False)
        if _canonical_digest(conn) != intent.canonical_digest:
            raise SemanticWorkUnavailable(
                "dimension reset source changed; explicit recovery required"
            )
        conn.execute("BEGIN")
        try:
            _recreate_embedding_tables(conn, intent.target_dimensions)
            conn.execute("DELETE FROM semantic_work")
            conn.execute("DELETE FROM semantic_work_counts")
            conn.execute(
                "UPDATE semantic_work_state SET phase='building', cursor_kind=NULL,"
                "cursor_id=NULL,completed_targets=0,verified_at=NULL,audited_at=NULL"
            )
            install_build_state(conn, intent)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        self._save(intent.model_copy(update={"phase": "building"}))

    def _finish_dimensions(self, conn: duckdb.DuckDBPyConnection) -> None:
        row = conn.execute(
            "SELECT embedding_dimensions FROM runtime_state WHERE singleton"
        ).fetchone()
        if row is not None and row[0] != self.config.embedding.dimensions:
            self.admitted = False
            intent = self._capture(
                conn, build_id=uuid4().hex, target_dimensions=self.config.embedding.dimensions
            )
            self._reset_dimensions(conn, intent)
            self._build(conn)

    def _audit(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.phase = "auditing"
        with _audit_deadline(conn), audited_transaction(conn):
            conn.execute(
                "UPDATE semantic_work_state SET "
                "phase='complete',verified_at=CURRENT_TIMESTAMP,audited_at=CURRENT_TIMESTAMP,error=NULL"
                " WHERE singleton"
            )
        _grant_write_owner(conn)

    def run(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.admitted = False
        self.error = None
        try:
            require_keyword_writes_enabled(self.config.data_dir)
            self._admission_state("auditing")
            self.intent = read_intent(self.config.data_dir)
            if self.intent is not None and self.intent.db_path != str(
                self.config.db_path.resolve()
            ):
                raise SemanticWorkUnavailable(
                    "semantic intent target identity differs; explicit metadata repair required"
                )
            version = get_schema_version(conn)
            if version < 32:
                run_pending_migrations(conn, version, through_version=32)
                version = get_schema_version(conn)
            if version == 32:
                reuse = False
                if self.intent is not None and self.intent.source_schema_version == 32:
                    try:
                        _settle_sidecar(self.config)
                        conn.execute("FORCE CHECKPOINT")
                        self._verify_backup(self.intent, compare_live=True)
                        reuse = True
                    except (OSError, ValueError, SemanticWorkUnavailable):
                        # Before DDL a fresh linked capture remains schema-32 fallback compatible.
                        reuse = False
                intent = self.intent if reuse else self._capture(conn, build_id=uuid4().hex)
                assert intent is not None
                self._install(conn, intent, repair=False)
                self._build(conn)
            elif version == 33:
                if (
                    self.intent is not None
                    and self.intent.phase == "failed"
                    and self.intent.recovery == "repair"
                ):
                    raise SemanticWorkUnavailable(
                        self.intent.error or "semantic work failed; explicit repair required"
                    )
                if (
                    self.intent is not None
                    and self.intent.operation == "dimensions"
                    and self.intent.phase != "complete"
                ):
                    dimensions = conn.execute(
                        "SELECT embedding_dimensions FROM runtime_state WHERE singleton"
                    ).fetchone()
                    if dimensions != (self.intent.target_dimensions,):
                        self._reset_dimensions(conn, self.intent)
                    self._build(conn)
                    return
                if (
                    self.intent is not None
                    and self.intent.accepted_plan_id is not None
                    and self.intent.phase != "complete"
                ):
                    if self.intent.backup_path is None:
                        accepted = self.intent.accepted_plan_id
                        build_id = self.intent.build_id
                        self._capture(conn, build_id=build_id, accepted=accepted)
                    assert self.intent is not None
                    self._verify_backup(self.intent, compare_live=False)
                    try:
                        installed = conn.execute(
                            "SELECT build_id FROM semantic_work_state WHERE singleton"
                        ).fetchone()
                    except (duckdb.CatalogException, duckdb.BinderException):
                        installed = None
                    if installed != (self.intent.build_id,):
                        self._install(conn, self.intent, repair=True)
                    self._build(conn)
                    return
                row = conn.execute(
                    "SELECT phase,definition_version FROM semantic_work_state WHERE singleton"
                ).fetchone()
                if row is None or row[1] != DEFINITION_VERSION:
                    raise SemanticWorkUnavailable(
                        "missing/unsupported semantic state; explicit metadata repair required"
                    )
                if row[0] == "complete" or (
                    self.intent is not None and self.intent.recovery == "audit"
                ):
                    self.phase = "auditing"
                    self._audit(conn)
                    if self.intent is not None:
                        self._save(
                            self.intent.model_copy(
                                update={"phase": "complete", "epoch": self.epoch, "error": None}
                            )
                        )
                    self.phase = "complete"
                    self._admit()
                elif self.intent is not None:
                    self._build(conn)
                else:
                    raise SemanticWorkUnavailable(
                        "semantic build intent missing; explicit metadata repair required"
                    )
            else:
                raise SemanticWorkUnavailable(
                    f"unsupported schema {version}; use matching approved binary"
                )
            if self.admitted:
                self._finish_dimensions(conn)
        except duckdb.FatalException:
            raise
        except Exception as err:
            self.fail(conn, err)

    def fail(self, conn: duckdb.DuckDBPyConnection, err: Exception) -> None:
        invalidate_readers(conn)
        previous_phase = self.phase
        self.admitted = False
        self.phase = "failed"
        self.error = f"{type(err).__name__}: {err}"[:2048]

        retryable = isinstance(err, (duckdb.OutOfMemoryException, duckdb.InterruptException)) or (
            isinstance(err, OSError) and err.errno == errno.ENOSPC
        )
        recovery = (
            "audit"
            if retryable
            and (
                self.intent is None
                or self.intent.phase == "complete"
                or previous_phase == "auditing"
            )
            else "resume"
            if retryable
            else "repair"
        )
        persistence_errors = []
        try:
            self._admission_state("held")
        except OSError as failure:
            persistence_errors.append(f"admission diagnostic: {failure}")
        if self.intent is not None:
            try:
                self._save(
                    self.intent.model_copy(
                        update={"phase": "failed", "error": self.error, "recovery": recovery}
                    )
                )
            except OSError as failure:
                persistence_errors.append(f"intent diagnostic: {failure}")
        try:
            conn.execute(
                "UPDATE semantic_work_state SET phase='failed',error=? WHERE singleton",
                [self.error],
            )
        except (duckdb.CatalogException, duckdb.BinderException):
            # Derived objects can be missing; the owner and admission diagnostic remain held.
            pass
        except (duckdb.IOException, duckdb.OutOfMemoryException) as failure:
            persistence_errors.append(f"database diagnostic: {failure}")
        if persistence_errors:
            self.error = (self.error + "; " + "; ".join(persistence_errors))[:2048]

    def preview(self, conn: duckdb.DuckDBPyConnection) -> RepairPlan:
        if self.admitted:
            raise SemanticWorkUnavailable(
                "semantic work is verified; metadata repair requires a held invalid state"
            )
        required, retained = self._admission(conn)
        version = get_schema_version(conn)
        if version != 33:
            raise SemanticWorkUnavailable(
                "schema "
                f"{version}"
                ": restart initialization; schema32 fallback remains available; no"
                " schema33 repair apply"
            )
        stat = self.config.db_path.stat()
        try:
            row = conn.execute(
                "SELECT mutation_seq FROM semantic_work_state WHERE singleton"
            ).fetchone()
        except (duckdb.CatalogException, duckdb.BinderException):
            row = None
        rollback = "schema33_current"
        for path, _ in retained:
            saved = Path(path) / "semantic-intent.json"
            if saved.exists():
                try:
                    previous = SemanticIntent.model_validate_json(saved.read_text())
                except ValueError:
                    continue
                if previous.source_schema_version == 32:
                    try:
                        self._verify_backup(previous, compare_live=False)
                    except (OSError, ValueError, SemanticWorkUnavailable):
                        continue
                    rollback = "schema32_5fa253b"
                    break
        from recall.services.daemon import _package_version

        values = dict(
            plan_id="",
            db_path=str(self.config.db_path.resolve()),
            file_identity=(stat.st_dev, stat.st_ino),
            binary_version=_package_version() or "unknown",
            schema_version=version,
            failed_build_id=self.intent.build_id if self.intent else None,
            diagnostic=self.error,
            mutation_seq=int(row[0]) if row else None,
            authoritative_generation=self.epoch,
            proposed_build_id=hashlib.sha256((self.epoch + ":repair").encode()).hexdigest()[:32],
            backup_root=str(self.config.data_dir / "snapshots"),
            spill_path=str(resolve_temp_directory(self.config)),
            required_bytes=required,
            retained_preimages=retained,
            predecessor=self.intent.build_id if self.intent else None,
            rollback_scope=rollback,
        )
        plan = RepairPlan.model_validate(values)
        plan = plan.model_copy(
            update={"plan_id": hashlib.sha256(plan.model_dump_json().encode()).hexdigest()}
        )
        self._plan = plan
        return plan

    def accept(self, conn: duckdb.DuckDBPyConnection, plan_id: str) -> RepairResult:
        # Durable acceptance wins before epoch validation, including after restart/completion.
        candidates = [self.intent] if self.intent is not None else []
        for path in (self.config.data_dir / "snapshots").glob("semantic-intent-*.json"):
            try:
                candidates.append(SemanticIntent.model_validate_json(path.read_text()))
            except ValueError:
                continue
        for path, _ in self._retained():
            saved = Path(path) / "semantic-intent.json"
            if saved.exists():
                try:
                    candidates.append(SemanticIntent.model_validate_json(saved.read_text()))
                except ValueError:
                    continue
        for intent in candidates:
            if (
                intent is not None
                and intent.db_path == str(self.config.db_path.resolve())
                and intent.accepted_plan_id == plan_id
            ):
                return RepairResult(
                    outcome="unchanged",
                    plan_id=plan_id,
                    build_id=intent.build_id,
                    phase=intent.phase,
                    error=intent.error,
                )
        plan = self.preview(conn)
        if plan.plan_id != plan_id:
            raise SemanticWorkUnavailable(
                "semantic repair plan conflict: generation or target changed; "
                "inspect --dry-run again"
            )
        self._begin_capture(conn, build_id=plan.proposed_build_id, accepted=plan_id)
        assert self.intent is not None
        return RepairResult(
            outcome="accepted",
            plan_id=plan_id,
            build_id=self.intent.build_id,
            phase=self.phase,
            error=self.error,
        )

    def apply(self, conn: duckdb.DuckDBPyConnection, plan_id: str) -> RepairResult:
        result = self.accept(conn, plan_id)
        if result.outcome == "accepted":
            self.run(conn)
            return result.model_copy(update={"phase": self.phase, "error": self.error})
        return result
