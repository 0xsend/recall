from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterable
from itertools import batched, groupby
from pathlib import Path
from typing import Literal

MIN_SQLITE_VERSION: tuple[int, int, int] = (3, 43, 0)
FTS_SIDECAR_BATCH_SIZE = 10_000
_FTS_ROWID_LOOKUP_CHUNK = 500

type _FtsKind = Literal["message", "tool_call"]
_FTS_TABLES: dict[_FtsKind, tuple[str, str]] = {
    "message": ("message_fts", "message_id"),
    "tool_call": ("tool_calls_fts", "tool_call_id"),
}

_TOKENIZER = "porter unicode61"
_ALL_FTS_FIELDS = ("bash", "content", "thinking")
_FTS_FIELDS_SIGNATURE_KEY = "fts_fields_signature"
logger = logging.getLogger(__name__)


class FtsSidecarUnavailableError(RuntimeError):
    """Raised when stdlib sqlite3 cannot create the required sidecar schema."""


def probe_sqlite_fts5_support(conn: sqlite3.Connection | None = None) -> None:
    if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
        raise FtsSidecarUnavailableError(
            "SQLite FTS5 sidecar requires SQLite >= 3.43.0 with contentless-delete "
            f'support; runtime is {sqlite3.sqlite_version}. Set backend = "duckdb" '
            "or use a Python runtime bundled with supported SQLite."
        )

    owns_conn = conn is None
    probe_conn = sqlite3.connect(":memory:") if conn is None else conn
    try:
        probe_conn.execute("DROP TABLE IF EXISTS _probe")
        probe_conn.execute(
            "CREATE VIRTUAL TABLE _probe USING fts5("
            "x, content='', contentless_delete=1, tokenize='porter unicode61'"
            ")"
        )
    except sqlite3.Error as err:
        raise FtsSidecarUnavailableError(
            "SQLite FTS5 sidecar requires FTS5 contentless-delete support "
            f"with tokenize='{_TOKENIZER}'. Set backend = \"duckdb\" or use a "
            "Python runtime bundled with supported SQLite."
        ) from err
    finally:
        if owns_conn:
            probe_conn.close()


def sidecar_path(data_dir: Path) -> Path:
    return data_dir / "recall.fts.sqlite"


def open_sidecar(path: Path) -> sqlite3.Connection:
    probe_sqlite_fts5_support()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
                    fts_content,
                    fts_thinking,
                    content='',
                    contentless_delete=1,
                    tokenize='porter unicode61'
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS tool_calls_fts USING fts5(
                    bash_command,
                    content='',
                    contentless_delete=1,
                    tokenize='porter unicode61'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS message_fts_rowid (
                    rowid INTEGER PRIMARY KEY,
                    message_id TEXT NOT NULL UNIQUE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_calls_fts_rowid (
                    rowid INTEGER PRIMARY KEY,
                    tool_call_id TEXT NOT NULL UNIQUE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bootstrap_progress (
                    kind TEXT PRIMARY KEY,
                    last_id TEXT,
                    completed_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fts_sidecar_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
    except sqlite3.Error:
        conn.close()
        raise
    return conn


def optimize_sidecar(sidecar_conn: sqlite3.Connection) -> None:
    """Merge SQLite FTS5 b-trees in-place to reclaim space and improve query locality.

    Each table gets an independent transaction so one broken FTS5 table cannot
    starve the other table's maintenance pass.
    """
    first_error: sqlite3.Error | None = None
    for table in ("message_fts", "tool_calls_fts"):
        try:
            with sidecar_conn:
                sidecar_conn.execute(f"INSERT INTO {table}({table}) VALUES('optimize')")
        except sqlite3.Error as err:
            logger.warning("sidecar optimize failed table=%s: %s", table, err)
            if first_error is None:
                first_error = err
    if first_error is not None:
        raise first_error


def upsert_message_fts(
    sidecar_conn: sqlite3.Connection,
    message_id: str,
    fts_content: str,
    fts_thinking: str,
    *,
    fields: Iterable[str] | None = None,
) -> None:
    upsert_message_fts_batch(sidecar_conn, [(message_id, fts_content, fts_thinking)], fields=fields)


def upsert_message_fts_batch(
    sidecar_conn: sqlite3.Connection,
    rows: Iterable[tuple[str, str, str]],
    *,
    fields: Iterable[str] | None = None,
    batch_size: int | None = None,
) -> None:
    """Write message FTS rows in bounded transactions.

    Large indexes can contain millions of FTS rows. Keeping the transaction
    bound explicit prevents unbounded SQLite rollback journals while avoiding a
    commit per row on the hot indexing path.
    """
    field_tuple = tuple(fields) if fields is not None else None
    for batch in batched(rows, _resolve_batch_size(batch_size)):
        with sidecar_conn:
            sidecar_conn.executemany(
                "INSERT OR IGNORE INTO message_fts_rowid(message_id) VALUES (?)",
                [(message_id,) for message_id, _fts_content, _fts_thinking in batch],
            )
            rowids = _fts_rowids(
                sidecar_conn, [message_id for message_id, _, _ in batch], "message"
            )
            sidecar_conn.executemany(
                """
                INSERT OR REPLACE INTO message_fts(rowid, fts_content, fts_thinking)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        rowids[message_id],
                        *scope_message_fts_columns(
                            fts_content,
                            fts_thinking,
                            field_tuple,
                        ),
                    )
                    for message_id, fts_content, fts_thinking in batch
                ],
            )


def upsert_tool_call_fts(
    sidecar_conn: sqlite3.Connection,
    tool_call_id: str,
    bash_command: str | None,
    *,
    fields: Iterable[str] | None = None,
) -> None:
    upsert_tool_call_fts_batch(sidecar_conn, [(tool_call_id, bash_command)], fields=fields)


def upsert_tool_call_fts_batch(
    sidecar_conn: sqlite3.Connection,
    rows: Iterable[tuple[str, str | None]],
    *,
    fields: Iterable[str] | None = None,
    batch_size: int | None = None,
) -> None:
    """Write tool-call FTS rows in bounded transactions."""
    field_tuple = tuple(fields) if fields is not None else None
    index_bash = should_index_bash_fts(field_tuple)
    for batch in batched(rows, _resolve_batch_size(batch_size)):
        with sidecar_conn:
            # Keep deletions between insertion runs: a later reinsert must
            # allocate its rowid after the preceding deletion has taken effect.
            for is_insert, run in groupby(
                batch,
                key=lambda row: row[1] is not None and index_bash,
            ):
                run_rows = list(run)
                if is_insert:
                    _upsert_tool_call_fts_insert_run(sidecar_conn, run_rows)
                    continue
                _delete_fts_rows(
                    sidecar_conn, (tool_call_id for tool_call_id, _ in run_rows), "tool_call"
                )


def fts_fields_signature(fields: Iterable[str] | None) -> str:
    """Return a stable signature for the field set whose content the sidecar stores."""
    if fields is None:
        normalized = _ALL_FTS_FIELDS
    else:
        normalized = tuple(dict.fromkeys(field for field in fields if field))
    return ",".join(sorted(normalized))


def get_fts_fields_signature(sidecar_conn: sqlite3.Connection) -> str | None:
    row = sidecar_conn.execute(
        "SELECT value FROM fts_sidecar_meta WHERE key = ?",
        [_FTS_FIELDS_SIGNATURE_KEY],
    ).fetchone()
    return str(row[0]) if row is not None else None


def set_fts_fields_signature(
    sidecar_conn: sqlite3.Connection,
    fields: Iterable[str] | None,
) -> str:
    signature = fts_fields_signature(fields)
    with sidecar_conn:
        sidecar_conn.execute(
            """
            INSERT INTO fts_sidecar_meta(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            [_FTS_FIELDS_SIGNATURE_KEY, signature],
        )
    return signature


def delete_message_fts(
    sidecar_conn: sqlite3.Connection,
    message_ids: Iterable[str],
) -> None:
    materialized = list(dict.fromkeys(message_ids))
    if not materialized:
        return
    with sidecar_conn:
        _delete_fts_rows(sidecar_conn, materialized, "message")


def delete_tool_call_fts(
    sidecar_conn: sqlite3.Connection,
    tool_call_ids: Iterable[str],
) -> None:
    materialized = list(dict.fromkeys(tool_call_ids))
    if not materialized:
        return
    with sidecar_conn:
        _delete_fts_rows(sidecar_conn, materialized, "tool_call")


def search_messages_fts(
    sidecar_conn: sqlite3.Connection,
    query: str,
    fields: list[str],
    limit: int,
    *,
    offset: int = 0,
) -> list[tuple[str, float]]:
    """Return message FTS hits with scores normalized to higher-is-better."""
    if not query.strip() or limit <= 0:
        return []
    if offset < 0:
        raise ValueError("message FTS offset must be non-negative")
    match_query = _message_match_query(query, fields)
    if not match_query:
        return []
    rows = sidecar_conn.execute(
        """
        SELECT mr.message_id, -bm25(message_fts) AS score
        FROM message_fts
        JOIN message_fts_rowid mr ON mr.rowid = message_fts.rowid
        WHERE message_fts MATCH ?
        ORDER BY bm25(message_fts) ASC
        LIMIT ?
        OFFSET ?
        """,
        [match_query, limit, offset],
    ).fetchall()
    return [(str(row[0]), float(row[1])) for row in rows]


def search_tool_calls_fts(
    sidecar_conn: sqlite3.Connection,
    query: str,
    limit: int,
    *,
    offset: int = 0,
) -> list[tuple[str, float]]:
    """Return tool call FTS hits with scores normalized to higher-is-better."""
    if not query.strip() or limit <= 0:
        return []
    if offset < 0:
        raise ValueError("tool call FTS offset must be non-negative")
    # Escape like the message path so identifiers with -, :, (, ) are searched as
    # terms instead of leaking raw FTS5 errors via `recall search --tool`.
    match_query = _escape_fts5_query(query)
    if not match_query:
        return []
    rows = sidecar_conn.execute(
        """
        SELECT tr.tool_call_id, -bm25(tool_calls_fts) AS score
        FROM tool_calls_fts
        JOIN tool_calls_fts_rowid tr ON tr.rowid = tool_calls_fts.rowid
        WHERE tool_calls_fts MATCH ?
        ORDER BY bm25(tool_calls_fts) ASC
        LIMIT ?
        OFFSET ?
        """,
        [match_query, limit, offset],
    ).fetchall()
    return [(str(row[0]), float(row[1])) for row in rows]


def _message_match_query(query: str, fields: list[str]) -> str:
    normalized = list(dict.fromkeys(fields))
    invalid = [field for field in normalized if field not in {"content", "thinking"}]
    if invalid:
        raise ValueError(f"unsupported message search field: {invalid[0]}")
    # Escape on every path. The default (no --field) and content+thinking paths
    # previously returned the raw query, so FTS5 interpreted -, :, (, ) as query
    # syntax and leaked `no such column` / `fts5: syntax error` for ordinary
    # identifiers like REQ-BRIDGE. _escape_fts5_query reduces
    # those to terms while preserving prefix (foo*) and "quoted phrase" operators.
    scoped_query = _escape_fts5_query(query)
    if not scoped_query:
        return ""
    if normalized == ["content"]:
        return f"{{fts_content}}: ({scoped_query})"
    if normalized == ["thinking"]:
        return f"{{fts_thinking}}: ({scoped_query})"
    return scoped_query


def scope_message_fts_columns(
    fts_content: str,
    fts_thinking: str,
    fields: Iterable[str] | None,
) -> tuple[str, str]:
    if fields is None:
        return fts_content, fts_thinking
    field_set = set(fields)
    return (
        fts_content if "content" in field_set else "",
        fts_thinking if "thinking" in field_set else "",
    )


def should_index_bash_fts(fields: Iterable[str] | None) -> bool:
    return fields is None or "bash" in set(fields)


def _escape_fts5_query(query: str) -> str:
    """Treat restricted-field queries as terms/phrases, not executable FTS5 syntax."""
    terms: list[str] = []
    position = 0
    for match in re.finditer(r'"([^"]*)"', query):
        terms.extend(
            _quote_fts5_term(term) for term in _bare_fts5_terms(query[position : match.start()])
        )
        phrase = match.group(1).strip()
        if phrase:
            terms.append(_quote_fts5_term(phrase))
        position = match.end()
    terms.extend(_quote_fts5_term(term) for term in _bare_fts5_terms(query[position:]))
    return " ".join(terms)


def _bare_fts5_terms(query: str) -> list[str]:
    return re.findall(r"\w+\*?", query, flags=re.UNICODE)


def _quote_fts5_term(term: str) -> str:
    prefix = term.endswith("*") and len(term) > 1
    term_value = term[:-1] if prefix else term
    escaped = term_value.replace('"', '""')
    suffix = "*" if prefix else ""
    return f'"{escaped}"{suffix}'


def _fts_rowids(
    conn: sqlite3.Connection, identifiers: Iterable[str], kind: _FtsKind
) -> dict[str, int]:
    """Resolve stable mapping rowids already inserted by this ordered run."""
    table, id_column = _FTS_TABLES[kind]
    unique_ids = list(dict.fromkeys(identifiers))
    mapping: dict[str, int] = {}
    for chunk in batched(unique_ids, _FTS_ROWID_LOOKUP_CHUNK):
        placeholders = ", ".join("?" for _ in chunk)
        for identifier, rowid in conn.execute(
            f"SELECT {id_column}, rowid FROM {table}_rowid WHERE {id_column} IN ({placeholders})",
            chunk,
        ):
            mapping[str(identifier)] = int(rowid)
    unresolved = next((identifier for identifier in unique_ids if identifier not in mapping), None)
    if unresolved is not None:
        label = kind.replace("_", " ")
        raise sqlite3.IntegrityError(f"failed to resolve sidecar rowid for {label} {unresolved}")
    return mapping


def _upsert_tool_call_fts_insert_run(
    conn: sqlite3.Connection,
    run_rows: list[tuple[str, str | None]],
) -> None:
    assert run_rows and all(bash_command is not None for _, bash_command in run_rows)
    conn.executemany(
        "INSERT OR IGNORE INTO tool_calls_fts_rowid(tool_call_id) VALUES (?)",
        [(tool_call_id,) for tool_call_id, _bash_command in run_rows],
    )
    rowids = _fts_rowids(
        conn, [tool_call_id for tool_call_id, _bash_command in run_rows], "tool_call"
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO tool_calls_fts(rowid, bash_command)
        VALUES (?, ?)
        """,
        [(rowids[tool_call_id], bash_command) for tool_call_id, bash_command in run_rows],
    )


def _delete_fts_rows(conn: sqlite3.Connection, identifiers: Iterable[str], kind: _FtsKind) -> None:
    """Retire a deletion run inside its caller's transaction, before any reinsert."""
    table, id_column = _FTS_TABLES[kind]
    for batch in batched(identifiers, _FTS_ROWID_LOOKUP_CHUNK):
        chunk = tuple(dict.fromkeys(batch))
        placeholders = ", ".join("?" for _ in chunk)
        conn.execute(
            f"DELETE FROM {table} WHERE rowid IN "
            f"(SELECT rowid FROM {table}_rowid WHERE {id_column} IN ({placeholders}))",
            chunk,
        )
        conn.execute(f"DELETE FROM {table}_rowid WHERE {id_column} IN ({placeholders})", chunk)


def _resolve_batch_size(batch_size: int | None) -> int:
    resolved = FTS_SIDECAR_BATCH_SIZE if batch_size is None else batch_size
    if resolved <= 0:
        raise ValueError("sidecar FTS batch size must be positive")
    return resolved
