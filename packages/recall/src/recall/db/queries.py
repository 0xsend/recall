from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from itertools import batched
from typing import TYPE_CHECKING, Literal

import duckdb

from recall.core.config import FtsConfig
from recall.core.models import Message, Session, StopMarker, ToolCall
from recall.db.fts_sidecar import delete_message_fts, delete_tool_call_fts
from recall.db.mutations import execute_count

if TYPE_CHECKING:
    import pyarrow as pa

logger = logging.getLogger(__name__)
_sidecar_rebuild_skip_logged = False

# Bound on one IN-list when resolving harness ids for a single parse chunk.
_ID_LOOKUP_CHUNK = 500

# Ids recall generates are sha256 hex, but historical and synthetic rows carry
# hand-written ones, so the token set is the widest that still cannot close a
# string literal, escape one, or end a statement.
_LITERAL_SAFE_ID = re.compile(r"\A[A-Za-z0-9_.:+@-]{1,128}\Z")


class FtsRebuildOutOfMemoryError(RuntimeError):
    """Raised when an FTS rebuild exhausts DuckDB memory and needs recovery."""


class FtsSettingsRestoreError(RuntimeError):
    """Raised when FTS session settings cannot be restored after a rebuild failure."""


def load_fts_extension(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("INSTALL fts")
    conn.execute("LOAD fts")


def _save_and_set_fts_session_settings(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, object]:
    """Save current DuckDB settings and apply bounded FTS rebuild settings.

    DuckDB SET values are shared by the database instance, so restore is required
    after rebuild to avoid changing subsequent write/read behavior.
    """
    prior = {
        "threads": _current_setting(conn, "threads"),
        "preserve_insertion_order": _current_setting(conn, "preserve_insertion_order"),
    }
    host_threads = max(2, os.cpu_count() or 4)
    fts_threads = max(2, min(8, host_threads // 2))
    conn.execute("SET preserve_insertion_order=false")
    conn.execute(f"SET threads = {fts_threads}")
    logger.info(
        "create_fts_indexes: applied session settings "
        "(preserve_insertion_order=false, threads=%d; prior: %s)",
        fts_threads,
        prior,
    )
    return prior


def _current_setting(conn: duckdb.DuckDBPyConnection, name: str) -> object:
    row = conn.execute(f"SELECT current_setting('{name}')").fetchone()
    if row is None:
        raise RuntimeError(f"DuckDB current_setting({name!r}) returned no row")
    return row[0]


def _restore_fts_session_settings(
    conn: duckdb.DuckDBPyConnection, prior: dict[str, object]
) -> None:
    """Restore database-wide settings changed for FTS rebuild.

    DuckDB rejects SET while a transaction is aborted. Roll back that poisoned
    transaction once, then retry so callers never reuse FTS-tuned settings.
    """
    rollback_needed = False
    try:
        _set_fts_session_settings(conn, prior)
    except duckdb.TransactionException as err:
        if not _is_aborted_transaction_error(err):
            _raise_fts_settings_restore_error(prior, err, err)
        rollback_needed = True
        try:
            conn.execute("ROLLBACK")
        except duckdb.TransactionException as rollback_err:
            if not _is_no_active_transaction_error(rollback_err):
                _raise_fts_settings_restore_error(prior, err, rollback_err)
        except Exception as rollback_err:
            _raise_fts_settings_restore_error(prior, err, rollback_err)
        try:
            _set_fts_session_settings(conn, prior)
        except Exception as retry_err:
            _raise_fts_settings_restore_error(prior, err, retry_err)
    except Exception as err:
        _raise_fts_settings_restore_error(prior, err, err)

    if rollback_needed:
        logger.info("create_fts_indexes: FTS settings restored after rollback (prior: %s)", prior)
    else:
        logger.info("create_fts_indexes: FTS settings restored without rollback (prior: %s)", prior)


def _set_fts_session_settings(conn: duckdb.DuckDBPyConnection, prior: dict[str, object]) -> None:
    conn.execute(f"SET preserve_insertion_order = {prior['preserve_insertion_order']}")
    conn.execute(f"SET threads = {prior['threads']}")


def _is_aborted_transaction_error(err: duckdb.TransactionException) -> bool:
    message = str(err)
    return "Current transaction is aborted" in message or "please ROLLBACK" in message


def _is_no_active_transaction_error(err: duckdb.TransactionException) -> bool:
    return "no transaction is active" in str(err)


def _raise_fts_settings_restore_error(
    prior: dict[str, object], first_err: Exception, retry_err: Exception
) -> None:
    first_err_class = first_err.__class__.__name__
    retry_err_class = retry_err.__class__.__name__
    message = (
        "failed to restore FTS settings after rollback "
        f"(prior={prior!r}, first_error={first_err_class}: {first_err}, "
        f"retry_error={retry_err_class}: {retry_err})"
    )
    error = FtsSettingsRestoreError(message)
    if retry_err is not first_err:
        logger.warning(
            "create_fts_indexes: failed to restore prior settings %s after rollback: %s",
            prior,
            retry_err,
        )
        raise error from retry_err
    logger.warning(
        "create_fts_indexes: failed to restore prior settings %s: %s",
        prior,
        first_err,
    )
    raise error from first_err


def create_fts_indexes(conn: duckdb.DuckDBPyConnection, fts: FtsConfig) -> None:
    if fts.backend == "sqlite_sidecar":
        _log_sidecar_rebuild_skip_once()
        return
    if not fts.fields:
        return
    prior = _save_and_set_fts_session_settings(conn)
    try:
        try:
            load_fts_extension(conn)
            refreshed = refresh_message_fts_columns(conn)
            if refreshed:
                logger.info(
                    "create_fts_indexes: backfilled %d out-of-sync fts_content/fts_thinking rows",
                    refreshed,
                )
            message_fields = [
                _MESSAGE_FTS_COLUMNS[field]
                for field in ("content", "thinking")
                if field in fts.fields
            ]
            if message_fields:
                columns = ", ".join(message_fields)
                conn.execute(
                    "PRAGMA create_fts_index(message_state, message_id, "
                    f"{columns}, stemmer='porter', stopwords='english', overwrite=1)"
                )
            if "bash" in fts.fields:
                conn.execute(
                    "PRAGMA create_fts_index(tool_calls, id, bash_command, "
                    "stemmer='porter', stopwords='english', overwrite=1)"
                )
            # why: PRAGMA create_fts_index with overwrite=1 writes a DROP SCHEMA op that
            # DuckDB 1.5.5 cannot replay after replacing a persisted FTS schema
            # (DependencyException on shadow tables inside the
            # FTS schema). Force a checkpoint here so the drop+recreate is materialized
            # into the main DB file and never has to replay on daemon restart.
            # Retire only when test_create_fts_indexes_checkpoints_unreplayable_
            # shadow_schema_drop passes without this guard on supported runtimes.
            conn.execute("CHECKPOINT")
        except duckdb.OutOfMemoryException as err:
            raise FtsRebuildOutOfMemoryError("FTS rebuild exhausted DuckDB memory_limit") from err
    finally:
        _restore_fts_session_settings(conn, prior)


_MESSAGE_FTS_COLUMNS = {
    "content": "fts_content",
    "thinking": "fts_thinking",
}


def _log_sidecar_rebuild_skip_once() -> None:
    global _sidecar_rebuild_skip_logged
    if _sidecar_rebuild_skip_logged:
        return
    logger.info("skipping DuckDB FTS rebuild because fts.backend=sqlite_sidecar is active")
    _sidecar_rebuild_skip_logged = True


def refresh_message_fts_columns(conn: duckdb.DuckDBPyConnection) -> int:
    """Backfill `message_state.fts_content` / `fts_thinking` only for rows
    whose derived columns are out of sync with their source columns
    (REQ-FTS-INC-002).

    Under REQ-FTS-INC-001 every write site (`insert_messages`,
    `_upsert_messages`) populates the derived columns eagerly, so on a
    converged DB the WHERE predicate matches zero rows and the call
    skips the DELETE+INSERT churn that previously dominated large-DB
    FTS rebuild memory pressure.

    Returns the number of rows updated.
    """
    return execute_count(
        conn,
        """
        UPDATE message_state
        SET
            fts_content = COALESCE(context_text, '') || COALESCE(content, ''),
            fts_thinking = COALESCE(context_text, '') || COALESCE(thinking, '')
        WHERE
            fts_content IS DISTINCT FROM (COALESCE(context_text, '') || COALESCE(content, ''))
            OR fts_thinking IS DISTINCT FROM (COALESCE(context_text, '') || COALESCE(thinking, ''))
        """,
    )


def fetch_session_state(
    conn: duckdb.DuckDBPyConnection, source_path: str
) -> tuple[str, float, int] | None:
    row = conn.execute(
        """
        SELECT s.id, ss.file_mtime, ss.file_size
        FROM sessions s
        JOIN session_state ss ON ss.session_id = s.id
        WHERE s.source_path = ?
        """,
        [source_path],
    ).fetchone()
    if row is None:
        return None
    session_id, file_mtime, file_size = row
    return str(session_id), float(file_mtime), int(file_size)


SidecarPendingKind = Literal["message", "tool_call"]
SidecarPendingOp = Literal["upsert", "delete"]

_VALID_SIDECAR_PENDING_KINDS = {"message", "tool_call"}
_VALID_SIDECAR_PENDING_OPS = {"upsert", "delete"}


def delete_session(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    *,
    sidecar_conn: sqlite3.Connection | None = None,
) -> None:
    from recall.db.semantic_work import publication

    publication(conn).delete(session_id, sidecar_conn=sidecar_conn)


def _delete_session_sidecar_rows(
    conn: duckdb.DuckDBPyConnection,
    sidecar_conn: sqlite3.Connection,
    message_ids: list[str],
    tool_call_ids: list[str],
) -> None:
    failed_messages: list[str] = []
    failed_tool_calls: list[str] = []
    try:
        delete_message_fts(sidecar_conn, message_ids)
    except sqlite3.Error as err:
        logger.warning(
            "sidecar delete failed for messages: %s; queueing for reconciliation",
            err,
        )
        failed_messages = list(message_ids)
    try:
        delete_tool_call_fts(sidecar_conn, tool_call_ids)
    except sqlite3.Error as err:
        logger.warning(
            "sidecar delete failed for tool_calls: %s; queueing for reconciliation",
            err,
        )
        failed_tool_calls = list(tool_call_ids)
    if failed_messages:
        _enqueue_sidecar_pending_best_effort(conn, "message", failed_messages, "delete")
    if failed_tool_calls:
        _enqueue_sidecar_pending_best_effort(conn, "tool_call", failed_tool_calls, "delete")


def _enqueue_sidecar_pending_best_effort(
    conn: duckdb.DuckDBPyConnection,
    kind: SidecarPendingKind,
    ids: Iterable[str],
    op: SidecarPendingOp,
) -> None:
    try:
        enqueue_sidecar_pending(conn, kind, ids, op)
    except Exception:
        logger.exception(
            "failed to enqueue SQLite FTS sidecar reconciliation work "
            "(kind=%s, op=%s); continuing because DuckDB is authoritative",
            kind,
            op,
        )


def enqueue_sidecar_pending(
    conn: duckdb.DuckDBPyConnection,
    kind: SidecarPendingKind,
    ids: Iterable[str],
    op: SidecarPendingOp,
) -> None:
    _validate_sidecar_pending(kind, op)
    rows = [(kind, pending_id, op) for pending_id in dict.fromkeys(ids)]
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO fts_sidecar_pending(kind, id, op)
        VALUES (?, ?, ?)
        """,
        rows,
    )


def enqueue_session_sidecar_deletes(conn: duckdb.DuckDBPyConnection, session_id: str) -> None:
    """Queue every keyword-search row of a session for deletion.

    Set-based: DuckDB 1.5.5 retries a failed `import pandas` for each bound
    parameter, so queueing a large session one bound id at a time costs
    seconds of CPU inside the writer's transaction.
    """
    for kind, table in (("message", "messages"), ("tool_call", "tool_calls")):
        conn.execute(
            f"INSERT INTO fts_sidecar_pending(kind, id, op)"
            f" SELECT '{kind}', id, 'delete' FROM {table} WHERE session_id = ?",
            [session_id],
        )


def drain_sidecar_pending(
    conn: duckdb.DuckDBPyConnection,
    kind: str,
    ids: Iterable[str],
) -> int:
    materialized = list(dict.fromkeys(ids))
    if not materialized:
        return 0
    placeholders = ", ".join("?" for _ in materialized)
    return execute_count(
        conn,
        f"""
        DELETE FROM fts_sidecar_pending
        WHERE kind = ? AND id IN ({placeholders})
        """,
        [kind, *materialized],
    )


def _validate_sidecar_pending(kind: str, op: str) -> None:
    if kind not in _VALID_SIDECAR_PENDING_KINDS:
        raise ValueError("sidecar pending kind must be one of: message, tool_call")
    if op not in _VALID_SIDECAR_PENDING_OPS:
        raise ValueError("sidecar pending op must be one of: upsert, delete")


def insert_session(
    conn: duckdb.DuckDBPyConnection, session: Session, *, last_byte_offset: int = 0
) -> None:
    from recall.db.semantic_work import publication

    publication(conn).session(session, last_byte_offset=last_byte_offset)


def id_set_predicate(column: str, identifiers: Sequence[str]) -> tuple[str, list[str]]:
    """Render ``column IN (...)`` for a writer-owned id set, plus its parameters.

    DuckDB 1.5.5 searches for an absent optional ``pandas`` twice for every
    bound non-NULL parameter, and Python caches nothing about a failed import,
    so a 256-id primary-key lookup pays 512 filesystem import searches. One
    commit of a 2.9k-message session paid ~25k of them (REQ-RECON-027).

    Ids drawn from the literal-safe token set go into the statement instead of
    being bound; every other value keeps the parameterized form, so this stays
    a representation change rather than a new escaping rule. The alternatives
    measured worse or unsafe: a ``sys.modules["pandas"]`` sentinel makes DuckDB
    read ``__version__`` off it and fails the write transaction, and binding the
    ids as a registered Arrow table turned these primary-key lookups into full
    scans. Retire this when a DuckDB release caches the failed import.
    """
    assert identifiers, "an id-set predicate needs at least one id"
    if all(_LITERAL_SAFE_ID.match(identifier) for identifier in identifiers):
        literals = ", ".join(f"'{identifier}'" for identifier in identifiers)
        return f"{column} IN ({literals})", []
    placeholders = ", ".join("?" for _ in identifiers)
    return f"{column} IN ({placeholders})", list(identifiers)


@contextmanager
def _bound_identifiers(
    conn: duckdb.DuckDBPyConnection, identifiers: Sequence[str]
) -> Iterator[None]:
    """Bind a bounded writer-owned ID set without per-scalar Python conversion."""
    import pyarrow as pa

    assert 0 < len(identifiers) <= _ID_LOOKUP_CHUNK
    table = pa.table({"id": pa.array(identifiers, type=pa.string())})
    conn.register("_recall_bound_ids", table)
    try:
        yield
    finally:
        conn.unregister("_recall_bound_ids")


def insert_messages(
    conn: duckdb.DuckDBPyConnection,
    messages: Iterable[Message],
    *,
    context_text: str = "",
    context_mode: str = "off",
    context_by_message: Mapping[str, tuple[str, str]] | None = None,
) -> None:
    from recall.db.semantic_work import publication

    publication(conn).messages(
        tuple(messages),
        context_text=context_text,
        context_mode=context_mode,
        context_by_message=context_by_message,
    )


def insert_tool_calls(conn: duckdb.DuckDBPyConnection, tool_calls: Iterable[ToolCall]) -> None:
    from recall.db.semantic_work import publication

    publication(conn).tools(tuple(tool_calls))


def _insert_arrow_rows(
    conn: duckdb.DuckDBPyConnection,
    rows: Iterable[Sequence[object]],
    schema: pa.Schema,
    statement: str,
) -> None:
    """Bind bounded typed row batches without per-row SQL parameter conversion."""
    import pyarrow as pa

    for batch in batched(rows, 256):
        table = pa.Table.from_arrays(
            [
                pa.array(column, type=field.type)
                for column, field in zip(zip(*batch, strict=True), schema, strict=True)
            ],
            schema=schema,
        )
        conn.register("_recall_insert_rows", table)
        try:
            conn.execute(statement)
        finally:
            conn.unregister("_recall_insert_rows")


def insert_tool_use_ids(
    conn: duckdb.DuckDBPyConnection, rows: Iterable[tuple[str, str, str]]
) -> None:
    """Map recall tool_call ids to their harness tool_use ids (REQ-LIVE-006).

    Insert-only, so an unchanged re-index writes no new row version and keeps
    the churn `tool_use_ids` exists to keep off `tool_calls` off this table
    too. A mapping that genuinely changed — a rewritten transcript reusing a
    positional tool_call id for a different harness call — is retired by
    `delete_tool_call_tail_facts` before this runs, never updated in place.
    """
    import pyarrow as pa

    _insert_arrow_rows(
        conn,
        rows,
        pa.schema(
            [
                ("tool_call_id", pa.string()),
                ("session_id", pa.string()),
                ("tool_use_id", pa.string()),
            ]
        ),
        "INSERT OR IGNORE INTO tool_use_ids (tool_call_id, session_id, tool_use_id) "
        "SELECT tool_call_id, session_id, tool_use_id FROM _recall_insert_rows",
    )


def insert_tool_results(
    conn: duckdb.DuckDBPyConnection,
    rows: Iterable[tuple[str, str, bool, datetime | None]],
) -> None:
    """Record tool results against their recall tool_call id (REQ-LIVE-006).

    Insert-only for the same reason as the mapping above: a result that is
    already stored is the same result, and re-presenting it must be a no-op.
    """
    import pyarrow as pa

    def split_timestamps() -> Iterable[Sequence[object]]:
        for tool_id, summary, is_error, stamp in rows:
            aware = stamp is not None and stamp.utcoffset() is not None
            yield (tool_id, summary, is_error, None if aware else stamp, stamp if aware else None)

    _insert_arrow_rows(
        conn,
        split_timestamps(),
        pa.schema(
            [
                ("tool_call_id", pa.string()),
                ("result_summary", pa.string()),
                ("is_error", pa.bool_()),
                ("completed_at", pa.timestamp("us")),
                ("completed_at_utc", pa.timestamp("us", tz="UTC")),
            ]
        ),
        "INSERT OR IGNORE INTO tool_results "
        "(tool_call_id, result_summary, is_error, completed_at) "
        "SELECT tool_call_id, result_summary, is_error, "
        "COALESCE(CAST(completed_at_utc AS TIMESTAMP), completed_at) FROM _recall_insert_rows",
    )


def retire_stop_markers(
    conn: duckdb.DuckDBPyConnection, session_id: str, markers: Iterable[StopMarker]
) -> None:
    """Drop markers a full re-parse of the session no longer produces.

    A rewritten or truncated transcript keeps its session id, so every position
    its old tail occupied survives the upsert that writes the new ones. The row
    set must equal what reconciling the same bytes from scratch commits, which
    for a shortened transcript means no marker on a message index it no longer
    has (REQ-INDEX-026).

    Append-only writes never call this: a suffix knows only its own positions.

    The kept indices travel as one JSON array rather than one bound parameter
    each, because DuckDB pays a per-parameter cost that a long-running session's
    marker list would make visible.
    """
    kept = sorted({marker.idx for marker in markers})
    conn.execute(
        """DELETE FROM session_stop_markers WHERE session_id = ?
           AND message_idx NOT IN (
               SELECT unnest(json_transform_strict(?, '["BIGINT"]')))""",
        [session_id, json.dumps(kept, separators=(",", ":"))],
    )


def upsert_stop_markers(
    conn: duckdb.DuckDBPyConnection, session_id: str, markers: Iterable[StopMarker]
) -> None:
    """Record the latest harness lifecycle event at each message (REQ-LIVE-005).

    A task can complete without another message, replacing its open marker at
    the same position. Update only changed facts; presenting an unchanged marker
    remains a no-op. The reader excludes markers beyond a rewritten message tail.
    """
    import pyarrow as pa

    schema = pa.schema(
        [
            ("session_id", pa.string()),
            ("message_idx", pa.int64()),
            ("reason", pa.string()),
            ("ends_turn", pa.bool_()),
            ("event_order", pa.int64()),
        ]
    )

    def batches() -> Iterator[pa.RecordBatch]:
        rows = (
            (session_id, marker.idx, marker.reason, marker.ends_turn, event_order)
            for event_order, marker in enumerate(markers)
        )
        for batch in batched(rows, 256):
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array(column, type=field.type)
                    for column, field in zip(zip(*batch, strict=True), schema, strict=True)
                ],
                schema=schema,
            )

    # One SQL stream selects final facts across all input batches before any
    # update. Per-batch upserts transiently rewrite already-current positions.
    with pa.RecordBatchReader.from_batches(schema, batches()) as reader:
        conn.register("_recall_stop_markers", reader)
        try:
            conn.execute(
                "INSERT INTO session_stop_markers "
                "(session_id, message_idx, reason, ends_turn) "
                "SELECT session_id, message_idx, reason, ends_turn FROM _recall_stop_markers "
                "QUALIFY row_number() OVER ("
                "PARTITION BY session_id, message_idx ORDER BY event_order DESC) = 1 "
                "ON CONFLICT (session_id, message_idx) DO UPDATE SET "
                "reason = excluded.reason, ends_turn = excluded.ends_turn "
                "WHERE session_stop_markers.reason IS DISTINCT FROM excluded.reason "
                "OR session_stop_markers.ends_turn IS DISTINCT FROM excluded.ends_turn"
            )
        finally:
            conn.unregister("_recall_stop_markers")


def resolve_tool_call_ids(
    conn: duckdb.DuckDBPyConnection, session_id: str, tool_use_ids: Sequence[str]
) -> dict[str, str]:
    """Look up recall tool_call ids for harness tool_use ids within one session.

    The ID binding is bounded even for a very active session.
    """
    resolved: dict[str, str] = {}
    for start in range(0, len(tool_use_ids), _ID_LOOKUP_CHUNK):
        chunk = tool_use_ids[start : start + _ID_LOOKUP_CHUNK]
        with _bound_identifiers(conn, chunk):
            rows = conn.execute(
                "SELECT tool_use_id, tool_call_id FROM tool_use_ids "
                "WHERE session_id = ? AND tool_use_id IN (SELECT id FROM _recall_bound_ids)",
                [session_id],
            ).fetchall()
        resolved.update({str(row[0]): str(row[1]) for row in rows})
    return resolved


def fetch_tool_use_ids(
    conn: duckdb.DuckDBPyConnection, tool_call_ids: Sequence[str]
) -> dict[str, str]:
    """Return the stored harness tool_use id for each recall tool_call id.

    The inverse of `resolve_tool_call_ids`, and chunked for the same reason.
    """
    stored: dict[str, str] = {}
    for start in range(0, len(tool_call_ids), _ID_LOOKUP_CHUNK):
        chunk = tool_call_ids[start : start + _ID_LOOKUP_CHUNK]
        with _bound_identifiers(conn, chunk):
            rows = conn.execute(
                "SELECT tool_call_id, tool_use_id FROM tool_use_ids "
                "WHERE tool_call_id IN (SELECT id FROM _recall_bound_ids)"
            ).fetchall()
        stored.update({str(row[0]): str(row[1]) for row in rows})
    return stored


def delete_tool_call_tail_facts(
    conn: duckdb.DuckDBPyConnection, tool_call_ids: Sequence[str]
) -> None:
    """Retire the mapping and any paired result for these recall tool_call ids.

    The two rows always go together: recall tool_call ids are positional, so a
    result left behind after its call is removed or repointed is adopted by
    whatever call next lands on that id. Deleting results first keeps the
    invariant every other query relies on — a `tool_results` row is reachable
    through `tool_use_ids`.
    """
    if not tool_call_ids:
        return
    for start in range(0, len(tool_call_ids), _ID_LOOKUP_CHUNK):
        chunk = tool_call_ids[start : start + _ID_LOOKUP_CHUNK]
        with _bound_identifiers(conn, chunk):
            conn.execute(
                "DELETE FROM tool_results WHERE tool_call_id IN (SELECT id FROM _recall_bound_ids)"
            )
            conn.execute(
                "DELETE FROM tool_use_ids WHERE tool_call_id IN (SELECT id FROM _recall_bound_ids)"
            )


def insert_message_embeddings(
    conn: duckdb.DuckDBPyConnection, rows: list[tuple[str, list[float] | None, list[float] | None]]
) -> None:
    from recall.db.semantic_work import publication

    publication(conn).message_vectors(tuple(rows))


def insert_tool_call_embeddings(
    conn: duckdb.DuckDBPyConnection, rows: list[tuple[str, list[float] | None]]
) -> None:
    from recall.db.semantic_work import publication

    publication(conn).tool_vectors(tuple(rows))
