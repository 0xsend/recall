"""Transactional semantic work, derived from stored payloads at publication."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING

from recall.core.models import Message, Session, ToolCall

if TYPE_CHECKING:
    from recall.db._payload import FtsSidecarTouchedIds, PersistedSessionRows
    from recall.db.enrichment_inputs import GeneratedEmbedSession
    from recall.db.moves import MovedDuplicate
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from weakref import WeakKeyDictionary, ref

import duckdb

from recall.core.daemon_status import SemanticWorkStatus
from recall.core.models import TailFacts
from recall.core.semantic_maintenance import SemanticIntent

DEFINITION_VERSION = 1
INITIALIZATION_CHUNK = 8192
PUBLICATION_CHUNK = 256
TARGET_CAPACITY = 100_000_000
SESSION_CAPACITY = 1_000_000


class SemanticWorkUnavailable(RuntimeError):
    """The owner must initialize, audit or explicitly repair the metadata."""


@dataclass(frozen=True)
class TargetKeys:
    messages: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionKeys:
    ids: tuple[str, ...]
    incoming: TargetKeys = TargetKeys()


@dataclass(frozen=True)
class RawInsert:
    session: Session


@dataclass(frozen=True)
class RawAppend:
    session: Session


@dataclass(frozen=True)
class RawRewrite:
    session: Session
    previous_rows: PersistedSessionRows


RawPublication = RawInsert | RawAppend | RawRewrite


WORK_COLUMNS = (
    "kind, id, session_id, idx, missing_content, missing_thinking, "
    "missing_bash, needs_marker, raw_available"
)
COUNT_COLUMNS = (
    "session_id, pending_messages, pending_tools, missing_content, "
    "missing_thinking, missing_bash, raw_blocked_messages"
)


def capture_identity(intent: SemanticIntent) -> str:
    values = intent.model_dump(exclude={"epoch", "phase", "error", "recovery", "verified_at"})
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class SemanticSchemaUpgrade:
    connection_identity: int
    preimage: SemanticIntent

    def install(self, conn: duckdb.DuckDBPyConnection) -> None:
        assert self.connection_identity == id(conn)
        assert self.preimage.source_schema_version == 32
        assert self.preimage.backup_path is not None and self.preimage.backup_files is not None
        assert self.preimage.live_files is not None and self.preimage.canonical_digest is not None
        conn.execute("BEGIN")
        try:
            install_schema(conn)
            conn.execute("INSERT INTO schema_version(version) VALUES (33)")
            conn.execute(
                "INSERT INTO schema_migrations(migration_id) VALUES ('0033_semantic_work')"
            )
            install_build_state(conn, self.preimage)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def install_build_state(conn: duckdb.DuckDBPyConnection, intent: SemanticIntent) -> None:
    conn.execute(
        """UPDATE semantic_work_state SET phase='building',build_id=?,backup_path=?,
        source_schema_version=?,source_manifest_sha256=?,captured_messages=?,captured_tools=?,
        started_at=CURRENT_TIMESTAMP WHERE singleton""",
        [
            intent.build_id,
            intent.backup_path,
            intent.source_schema_version,
            capture_identity(intent),
            intent.captured_messages,
            intent.captured_tools,
        ],
    )


def expected_work_sql(*, scoped: bool = False) -> str:
    """Project NULL/empty facts before joining; never materialize vector payloads."""

    def scope(kind: str, column: str) -> str:
        return (
            f" WHERE {column} IN (SELECT id FROM _semantic_keys WHERE kind='{kind}')"
            if scoped
            else ""
        )

    return f"""
    WITH mv AS MATERIALIZED (
        SELECT message_id, content_embedding IS NULL AS c, thinking_embedding IS NULL AS t
        FROM message_embeddings{scope("message", "message_id")}
    ), mf AS MATERIALIZED (
        SELECT message_id, COALESCE(content, '') <> '' AS c, COALESCE(thinking, '') <> '' AS t
        FROM message_state{scope("message", "message_id")}
    ), tv AS MATERIALIZED (
        SELECT tool_call_id, bash_embedding IS NULL AS b
        FROM tool_call_embeddings{scope("tool", "tool_call_id")}
    ), candidates AS (
        SELECT 'message' AS kind, m.id, m.session_id, m.idx,
               COALESCE(e.c, TRUE) AND COALESCE(f.c, FALSE) AS missing_content,
               COALESCE(e.t, TRUE) AND COALESCE(f.t, FALSE) AS missing_thinking,
               FALSE AS missing_bash, e.message_id IS NULL AS needs_marker,
               f.message_id IS NOT NULL AS raw_available
        FROM (SELECT id, session_id, idx FROM messages{scope("message", "id")}) m
        LEFT JOIN mv e ON e.message_id=m.id LEFT JOIN mf f ON f.message_id=m.id
        WHERE e.message_id IS NULL OR (e.c AND f.c) OR (e.t AND f.t)
           OR (f.message_id IS NULL AND (e.c OR e.t))
        UNION ALL
        SELECT 'tool', t.id, t.session_id, t.idx, FALSE, FALSE,
               COALESCE(e.b, TRUE) AND COALESCE(t.bash_command, '') <> '',
               e.tool_call_id IS NULL, TRUE
        FROM (SELECT id, session_id, idx, bash_command FROM tool_calls{scope("tool", "id")}) t
        LEFT JOIN tv e ON e.tool_call_id=t.id
        WHERE e.tool_call_id IS NULL OR (e.b AND COALESCE(t.bash_command, '') <> '')
    ) SELECT {WORK_COLUMNS} FROM candidates
    """


def grouped_counts_sql(relation: str) -> str:
    return f"""SELECT session_id,
        count(*) FILTER (WHERE kind='message') AS pending_messages,
        count(*) FILTER (WHERE kind='tool') AS pending_tools,
        count(*) FILTER (WHERE missing_content) AS missing_content,
        count(*) FILTER (WHERE missing_thinking) AS missing_thinking,
        count(*) FILTER (WHERE missing_bash) AS missing_bash,
        count(*) FILTER (WHERE NOT raw_available) AS raw_blocked_messages
        FROM {relation} GROUP BY session_id"""


def install_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(Path(__file__).with_name("semantic_schema.sql").read_text())


def _read_complete_state(conn: duckdb.DuckDBPyConnection) -> SemanticWorkStatus:
    try:
        row = conn.execute(
            "SELECT definition_version, phase, verified_at, audited_at, "
            "error, build_id, completed_targets, captured_messages, "
            "captured_tools, epoch(started_at), epoch(verified_at) FROM "
            "semantic_work_state WHERE singleton"
        ).fetchone()
    except (duckdb.CatalogException, duckdb.BinderException) as err:
        raise SemanticWorkUnavailable(
            "semantic work unavailable; start the approved owner to initialize"
            " or inspect repair-semantic-work"
        ) from err
    if (
        row is None
        or row[0] != DEFINITION_VERSION
        or row[1] != "complete"
        or row[2] is None
        or row[3] is None
        or row[4]
    ):
        raise SemanticWorkUnavailable(
            "semantic work invalid or initializing: "
            f"{row}"
            "; inspect daemon status, then repair-semantic-work --dry-run when"
            " failed"
        )

    return {
        "definition_version": int(row[0]),
        "phase": str(row[1]),
        "build_id": str(row[5]) if row[5] is not None else None,
        "completed_targets": int(row[6]),
        "captured_messages": int(row[7]) if row[7] is not None else None,
        "captured_tools": int(row[8]) if row[8] is not None else None,
        "started_at": float(row[9]) if row[9] is not None else None,
        "verified_at": float(row[10]) if row[10] is not None else None,
        "error": None,
    }


@dataclass
class _VerifiedScope:
    valid: bool = True


_verified: WeakKeyDictionary[duckdb.DuckDBPyConnection, _VerifiedScope] = WeakKeyDictionary()


def mark_verified(conn: duckdb.DuckDBPyConnection) -> None:
    _verified[conn] = _VerifiedScope()


_writers: WeakKeyDictionary[duckdb.DuckDBPyConnection, _VerifiedScope] = WeakKeyDictionary()


def _grant_write_owner(conn: duckdb.DuckDBPyConnection) -> None:
    invalidate_readers(conn)
    previous = _owners.pop(conn, None)
    if previous is not None:
        previous.close()
    scope = _VerifiedScope()
    _verified[conn] = scope
    _writers[conn] = scope


def invalidate_readers(conn: duckdb.DuckDBPyConnection) -> None:
    scope = _verified.get(conn)
    if scope is not None:
        scope.valid = False
    writer = _writers.get(conn)
    if writer is not None:
        writer.valid = False


def verified_cursor(parent: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    cursor = parent.cursor()
    scope = _verified.get(parent)
    if scope is not None and scope.valid:
        _verified[cursor] = scope
    return cursor


def require_complete(conn: duckdb.DuckDBPyConnection) -> SemanticWorkStatus:
    scope = _verified.get(conn)
    if scope is None or not scope.valid:
        raise SemanticWorkUnavailable(
            "semantic coverage unverified in this owner; start the approved "
            "owner to audit; offline inspection cannot infer zero"
        )
    return _read_complete_state(conn)


@contextmanager
def read_snapshot(conn: duckdb.DuckDBPyConnection) -> Iterator[duckdb.DuckDBPyConnection]:
    """Read completion and counts from one lifecycle-bound committed snapshot."""
    cursor = verified_cursor(conn)
    try:
        cursor.execute("BEGIN TRANSACTION READ ONLY")
        try:
            yield cursor
            cursor.execute("COMMIT")
        except BaseException:
            cursor.execute("ROLLBACK")
            raise
    finally:
        cursor.close()


def _definition(conn: duckdb.DuckDBPyConnection) -> tuple:
    names = "('semantic_work','semantic_work_counts','semantic_work_state','semantic_pending')"
    columns = conn.execute(
        "SELECT table_name,column_name,data_type,is_nullable,column_default "
        "FROM information_schema.columns WHERE table_catalog=current_database() "
        f"AND table_schema='main' AND table_name IN {names} ORDER BY table_name,ordinal_position"
    ).fetchall()
    constraints = conn.execute(
        "SELECT table_name,constraint_type,constraint_text FROM duckdb_constraints() "
        "WHERE database_name=current_database() AND schema_name='main' "
        f"AND table_name IN {names} ORDER BY table_name,constraint_type,constraint_text"
    ).fetchall()
    view = conn.execute(
        "SELECT sql FROM duckdb_views() WHERE "
        "database_name=current_database() AND schema_name='main' AND "
        "view_name='semantic_pending'"
    ).fetchall()
    indexes = conn.execute(
        "SELECT table_name,index_name,expressions,is_unique FROM duckdb_indexes() "
        "WHERE database_name=current_database() AND schema_name='main' "
        f"AND table_name IN {names} ORDER BY index_name"
    ).fetchall()
    return tuple(columns), tuple(constraints), tuple(view), tuple(indexes)


@cache
def _compiled_definition() -> tuple:
    with duckdb.connect(":memory:", config={"threads": 1, "memory_limit": "16MB"}) as reference:
        reference.execute(
            "CREATE TABLE sessions(id TEXT); CREATE TABLE "
            "session_state(session_id TEXT,file_mtime DOUBLE)"
        )
        install_schema(reference)
        return _definition(reference)


def audit(conn: duckdb.DuckDBPyConnection) -> None:
    """Compare exact identities and masks, then grouped counts, in one snapshot."""
    invalidate_readers(conn)
    if _definition(conn) != _compiled_definition():
        raise SemanticWorkUnavailable(
            "semantic metadata schema differs from definition 1; explicit repair required"
        )
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE _semantic_expected AS
        WITH facts AS (
            SELECT 'message' AS kind, m.id, m.session_id, m.idx,
                   e.content_embedding IS NULL AND length(coalesce(s.content,'')) > 0
                       AS missing_content,
                   e.thinking_embedding IS NULL AND length(coalesce(s.thinking,'')) > 0
                       AS missing_thinking,
                   false AS missing_bash, e.message_id IS NULL AS needs_marker,
                   s.message_id IS NOT NULL AS raw_available,
                   e.content_embedding IS NULL OR e.thinking_embedding IS NULL AS absent_vector
            FROM messages m LEFT JOIN message_state s ON s.message_id=m.id
            LEFT JOIN message_embeddings e ON e.message_id=m.id
            UNION ALL
            SELECT 'tool', t.id, t.session_id, t.idx, false, false,
                   e.bash_embedding IS NULL AND length(coalesce(t.bash_command,'')) > 0,
                   e.tool_call_id IS NULL, true, e.bash_embedding IS NULL
            FROM tool_calls t LEFT JOIN tool_call_embeddings e ON e.tool_call_id=t.id
        ) SELECT kind,id,session_id,idx,missing_content,missing_thinking,missing_bash,
                 needs_marker,raw_available FROM facts
          WHERE missing_content OR missing_thinking OR missing_bash OR needs_marker
             OR (NOT raw_available AND absent_vector)
    """)
    difference = conn.execute(
        "SELECT EXISTS(\n        (SELECT "
        f"{WORK_COLUMNS}"
        " FROM _semantic_expected EXCEPT SELECT "
        f"{WORK_COLUMNS}"
        " FROM semantic_work)\n        UNION ALL\n        (SELECT "
        f"{WORK_COLUMNS}"
        " FROM semantic_work EXCEPT SELECT "
        f"{WORK_COLUMNS}"
        " FROM _semantic_expected))"
    ).fetchone()
    counts = conn.execute(
        "SELECT EXISTS(\n        ("
        f"{grouped_counts_sql('_semantic_expected')}"
        " EXCEPT SELECT "
        f"{COUNT_COLUMNS}"
        " FROM semantic_work_counts)\n        UNION ALL\n        (SELECT "
        f"{COUNT_COLUMNS}"
        " FROM semantic_work_counts EXCEPT "
        f"{grouped_counts_sql('_semantic_expected')}"
        "))"
    ).fetchone()
    conn.execute("DROP TABLE _semantic_expected")
    if difference != (False,) or counts != (False,):
        raise SemanticWorkUnavailable(
            "semantic work drift: exact target/mask/ownership or session "
            "totals differ; inspect repair-semantic-work --dry-run"
        )
    mark_verified(conn)


PENDING_COUNTS_SQL = (
    "pending AS (\n    SELECT c.*, ss.file_mtime,\n           s.id IS "
    "NOT NULL AND ss.session_id IS NOT NULL AND ss.file_mtime IS NOT "
    "NULL AS selectable,\n           "
    "c.pending_messages-c.raw_blocked_messages AS messages, "
    "c.pending_tools AS tools,\n           CASE WHEN s.id IS NOT NULL "
    "AND ss.session_id IS NOT NULL AND ss.file_mtime IS NOT NULL"
    "\n                THEN c.raw_blocked_messages ELSE "
    "c.pending_messages+c.pending_tools END AS blocked\n    FROM "
    "semantic_work_counts c LEFT JOIN sessions s ON s.id=c.session_id"
    "\n    LEFT JOIN session_state ss ON ss.session_id=c.session_id\n)"
)


class Publication:
    """Own a lifecycle handle's payload/work/counter transaction; no SQL capability escapes."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        proof = _writers.get(conn)
        if proof is None or not proof.valid:
            raise SemanticWorkUnavailable(
                "read capability cannot publish; obtain the approved write owner"
            )
        self._proof = proof
        self._connection = ref(conn)
        self._active = False
        self._closed = False
        self._on_invalid: Callable[[str], None] | None = None

    @property
    def _conn(self) -> duckdb.DuckDBPyConnection:
        connection = self._connection()
        if connection is None or self._closed:
            raise SemanticWorkUnavailable("publication capability expired")
        return connection

    def on_invalid(self, callback: Callable[[str], None]) -> None:
        assert self._on_invalid is None
        self._on_invalid = callback

    def close(self) -> None:
        assert not self._active
        self._closed = True

    @contextmanager
    def _transaction(self, keys: TargetKeys | SessionKeys) -> Iterator[None]:
        if self._closed or not self._proof.valid:
            raise SemanticWorkUnavailable(
                "publication capability expired on connection close/reopen"
            )
        assert not self._active, "publication must have one transaction owner"
        require_complete(self._conn)
        self._conn.execute("BEGIN")
        self._active = True
        try:
            self._capture(keys)
            yield
            if isinstance(keys, SessionKeys):
                self._conn.execute(
                    "INSERT OR IGNORE INTO _semantic_scope SELECT 'message',id FROM messages "
                    "WHERE session_id IN (SELECT unnest(?::TEXT[])) UNION ALL "
                    "SELECT 'tool',id FROM tool_calls "
                    "WHERE session_id IN (SELECT unnest(?::TEXT[]))",
                    [list(keys.ids), list(keys.ids)],
                )
            self._refresh()
            self._conn.execute("COMMIT")
        except BaseException as err:
            self._conn.execute("ROLLBACK")
            if isinstance(err, AssertionError):
                invalidate_readers(self._conn)
                self._conn.execute(
                    "UPDATE semantic_work_state SET phase='failed',error=? WHERE singleton",
                    [str(err)[:2048]],
                )
                if self._on_invalid is not None:
                    self._on_invalid(str(err))
            raise
        finally:
            self._active = False

    def _capture(self, keys: TargetKeys | SessionKeys) -> None:
        c = self._conn
        c.execute(
            "CREATE OR REPLACE TEMP TABLE _semantic_scope(kind TEXT, id TEXT, PRIMARY KEY(kind,id))"
        )
        if isinstance(keys, SessionKeys):
            c.execute(
                (
                    "INSERT OR IGNORE INTO _semantic_scope SELECT 'message', id FROM "
                    "messages WHERE session_id IN (SELECT unnest(?::TEXT[])) UNION ALL"
                    " SELECT 'tool', id FROM tool_calls WHERE session_id IN (SELECT "
                    "unnest(?::TEXT[])) UNION ALL SELECT kind,id FROM semantic_work "
                    "WHERE session_id IN (SELECT unnest(?::TEXT[]))"
                ),
                [list(keys.ids)] * 3,
            )
            targets = keys.incoming
        else:
            targets = keys
        from itertools import batched

        import pyarrow as pa

        schema = pa.schema([("kind", pa.string()), ("id", pa.string())])

        def batches():
            for kind, identifiers in (("message", targets.messages), ("tool", targets.tools)):
                for chunk in batched(identifiers, PUBLICATION_CHUNK):
                    yield pa.RecordBatch.from_arrays(
                        [
                            pa.array([kind] * len(chunk), type=pa.string()),
                            pa.array(chunk, type=pa.string()),
                        ],
                        schema=schema,
                    )

        reader = pa.RecordBatchReader.from_batches(schema, batches())
        c.register("_semantic_incoming", reader)
        try:
            c.execute("INSERT OR IGNORE INTO _semantic_scope SELECT * FROM _semantic_incoming")
        finally:
            c.unregister("_semantic_incoming")

    def _refresh(self) -> None:
        c = self._conn
        cursor: tuple[str, str] = ("", "")
        changed = False
        from recall.db.connection import _sql_string_literal

        while True:
            kind, identifier = (_sql_string_literal(value) for value in cursor)
            c.execute(
                "CREATE OR REPLACE TEMP TABLE _semantic_keys AS SELECT * FROM _semantic_scope "
                f"WHERE (kind,id)>('{kind}','{identifier}') "
                f"ORDER BY kind,id LIMIT {PUBLICATION_CHUNK}"
            )
            last = c.execute(
                "SELECT kind,id FROM _semantic_keys ORDER BY kind DESC,id DESC LIMIT 1"
            ).fetchone()
            if last is None:
                break
            c.execute(
                "CREATE OR REPLACE TEMP TABLE _semantic_before AS SELECT w.* FROM "
                "semantic_work w JOIN _semantic_keys k USING(kind,id)"
            )
            c.execute(
                "CREATE OR REPLACE TEMP TABLE _semantic_after AS " + expected_work_sql(scoped=True)
            )
            differs = c.execute(
                "SELECT EXISTS((SELECT * FROM _semantic_before EXCEPT SELECT * "
                "FROM _semantic_after) UNION ALL (SELECT * FROM _semantic_after "
                "EXCEPT SELECT * FROM _semantic_before))"
            ).fetchone()
            if differs == (True,):
                changed = True
                c.execute(
                    "CREATE OR REPLACE TEMP TABLE _semantic_sessions AS SELECT "
                    "session_id FROM _semantic_before UNION SELECT session_id FROM "
                    "_semantic_after"
                )
                # Keep constraints true throughout publication: replace complete final rows.
                c.execute(
                    "CREATE OR REPLACE TEMP TABLE _semantic_totals AS"
                    "\n                    SELECT session_id, "
                    "sum(pending_messages)::BIGINT AS pending_messages,"
                    "\n                        sum(pending_tools)::BIGINT AS "
                    "pending_tools,\n                        "
                    "sum(missing_content)::BIGINT AS missing_content,"
                    "\n                        sum(missing_thinking)::BIGINT AS "
                    "missing_thinking,\n                        "
                    "sum(missing_bash)::BIGINT AS missing_bash,"
                    "\n                        sum(raw_blocked_messages)::BIGINT AS "
                    "raw_blocked_messages\n                    FROM ("
                    "\n                        SELECT "
                    f"{COUNT_COLUMNS}"
                    " FROM semantic_work_counts WHERE session_id IN (SELECT session_id"
                    " FROM _semantic_sessions)\n                        UNION ALL "
                    "SELECT * FROM ("
                    f"{grouped_counts_sql('_semantic_after')}"
                    ")\n                        UNION ALL SELECT "
                    "session_id,-pending_messages,-pending_tools,-missing_content,-missing_thinking,-missing_bash,-raw_blocked_messages"
                    " FROM ("
                    f"{grouped_counts_sql('_semantic_before')}"
                    ")\n                    ) GROUP BY session_id"
                )
                invalid = c.execute(
                    "SELECT count(*) FROM _semantic_totals WHERE pending_messages<0 OR"
                    " pending_tools<0 OR missing_content<0 OR missing_thinking<0 OR "
                    "missing_bash<0 OR raw_blocked_messages<0"
                ).fetchone()
                assert invalid == (0,), "semantic counter preimage is inconsistent"
                c.execute(
                    "DELETE FROM semantic_work_counts WHERE session_id IN (SELECT "
                    "session_id FROM _semantic_sessions)"
                )
                c.execute(
                    "INSERT INTO semantic_work_counts SELECT * FROM _semantic_totals "
                    "WHERE pending_messages+pending_tools>0"
                )
                c.execute(
                    "DELETE FROM semantic_work WHERE (kind,id) IN (SELECT kind,id FROM"
                    " _semantic_keys)"
                )
                c.execute("INSERT INTO semantic_work SELECT * FROM _semantic_after")
            cursor = str(last[0]), str(last[1])
        if changed:
            capacity = c.execute(
                "SELECT count(*), coalesce(sum(pending_messages+pending_tools),0) "
                "FROM semantic_work_counts"
            ).fetchone()
            assert capacity is not None
            if capacity[0] > SESSION_CAPACITY or capacity[1] > TARGET_CAPACITY:
                raise SemanticWorkUnavailable(
                    "semantic capacity exceeded: "
                    f"{capacity}"
                    "; bounds "
                    f"{SESSION_CAPACITY}"
                    " sessions/"
                    f"{TARGET_CAPACITY}"
                    " targets"
                )
            c.execute("UPDATE semantic_work_state SET mutation_seq=mutation_seq+1 WHERE singleton")

    def raw(
        self,
        change: RawPublication,
        *,
        last_byte_offset: int = 0,
        context_text: str = "",
        context_mode: str = "off",
        sidecar_conn: sqlite3.Connection | None = None,
        fts_fields: tuple[str, ...] | None = None,
        sidecar_touched: FtsSidecarTouchedIds | None = None,
        host: str | None = None,
        tail_facts: TailFacts,
        on_commit: Callable[[], None] | None = None,
    ) -> bool:
        from recall.db import _payload

        session = change.session
        tools = list(_payload._collect_tool_calls(session))
        incoming = TargetKeys(tuple(m.id for m in session.messages), tuple(t.id for t in tools))
        keys = incoming if isinstance(change, RawAppend) else SessionKeys((session.id,), incoming)
        touched, owns_touched = _payload._sidecar_touched_accumulator(sidecar_conn, sidecar_touched)
        try:
            with self._transaction(keys):
                if isinstance(change, RawAppend):
                    return _payload._incremental_write_session(
                        self._conn,
                        session,
                        last_byte_offset=last_byte_offset,
                        context_text=context_text,
                        context_mode=context_mode,
                        sidecar_conn=sidecar_conn,
                        fts_fields=fts_fields,
                        sidecar_touched=touched,
                        host=host,
                        tail_facts=tail_facts,
                        on_commit=on_commit,
                    )
                if isinstance(change, RawInsert):
                    _payload._insert_new_session(
                        self._conn,
                        session,
                        last_byte_offset=last_byte_offset,
                        context_text=context_text,
                        context_mode=context_mode,
                        sidecar_conn=sidecar_conn,
                        fts_fields=fts_fields,
                        sidecar_touched=touched,
                        tail_facts=tail_facts,
                        on_commit=on_commit,
                    )
                    return _payload._session_has_searchable_content(session)
                assert isinstance(change, RawRewrite)
                previous_rows = change.previous_rows
                return _payload._sync_existing_session(
                    self._conn,
                    previous_rows,
                    session,
                    last_byte_offset=last_byte_offset,
                    context_text=context_text,
                    context_mode=context_mode,
                    sidecar_conn=sidecar_conn,
                    fts_fields=fts_fields,
                    sidecar_touched=touched,
                    tail_facts=tail_facts,
                    on_commit=on_commit,
                )
        except Exception:
            _payload._enqueue_owned_sidecar_touched_after_rollback(
                self._conn, touched, owns_touched
            )
            raise

    def enrich(self, result: GeneratedEmbedSession, *, embed: bool) -> bool | None:
        from recall.db._payload import _publish_enrichment

        keys = TargetKeys(result.inputs.targets.message_ids, result.inputs.targets.tool_ids)
        with self._transaction(keys):
            return _publish_enrichment(self._conn, result, embed=embed)

    def session(self, session: Session, *, last_byte_offset: int = 0) -> None:
        from recall.db._predicate import insert_session

        with self._transaction(TargetKeys()):
            insert_session(self._conn, session, last_byte_offset=last_byte_offset)

    def messages(
        self,
        messages: tuple[Message, ...],
        *,
        context_text: str = "",
        context_mode: str = "off",
        context_by_message: Mapping[str, tuple[str, str]] | None = None,
    ) -> None:
        from recall.db._predicate import insert_messages

        with self._transaction(TargetKeys(tuple(m.id for m in messages))):
            insert_messages(
                self._conn,
                messages,
                context_text=context_text,
                context_mode=context_mode,
                context_by_message=context_by_message,
            )

    def tools(self, tools: tuple[ToolCall, ...]) -> None:
        from recall.db._predicate import insert_tool_calls

        with self._transaction(TargetKeys(tools=tuple(t.id for t in tools))):
            insert_tool_calls(self._conn, tools)

    def message_vectors(
        self, rows: tuple[tuple[str, list[float] | None, list[float] | None], ...]
    ) -> None:
        from recall.db._predicate import insert_message_embeddings

        with self._transaction(TargetKeys(tuple(r[0] for r in rows))):
            insert_message_embeddings(self._conn, list(rows))

    def tool_vectors(self, rows: tuple[tuple[str, list[float] | None], ...]) -> None:
        from recall.db._predicate import insert_tool_call_embeddings

        with self._transaction(TargetKeys(tools=tuple(r[0] for r in rows))):
            insert_tool_call_embeddings(self._conn, list(rows))

    def delete(self, session_id: str, *, sidecar_conn: sqlite3.Connection | None = None) -> None:
        from recall.db._predicate import delete_session

        with self._transaction(SessionKeys((session_id,))):
            delete_session(self._conn, session_id, sidecar_conn=sidecar_conn)

    def move(self, plan: MovedDuplicate, *, queue_sidecar_deletes: bool) -> None:
        from recall.db.moves import _supersede_moved_duplicate

        with self._transaction(SessionKeys((plan.predecessor_id, plan.successor_id))):
            _supersede_moved_duplicate(
                self._conn, plan, queue_sidecar_deletes=queue_sidecar_deletes
            )


_owners: WeakKeyDictionary[duckdb.DuckDBPyConnection, Publication] = WeakKeyDictionary()


def publication(conn: duckdb.DuckDBPyConnection) -> Publication:
    owner = _owners.get(conn)
    if owner is None:
        owner = Publication(conn)
        _owners[conn] = owner
    return owner


def expire(conn: duckdb.DuckDBPyConnection) -> None:
    invalidate_readers(conn)
    owner = _owners.pop(conn, None)
    if owner is not None:
        owner.close()
