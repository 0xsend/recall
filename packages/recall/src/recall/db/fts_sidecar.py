from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from itertools import batched
from pathlib import Path
from typing import Literal

from recall.core.read_cancellation import check_read_cancelled, read_cancelled
from recall.core.telemetry import Stage, measured

MIN_SQLITE_VERSION: tuple[int, int, int] = (3, 43, 0)
FTS_SIDECAR_BATCH_SIZE = 10_000
_FTS_ROWID_LOOKUP_CHUNK = 500

type _FtsKind = Literal["message", "tool_call"]
_FTS_TABLES: dict[_FtsKind, tuple[str, str]] = {
    "message": ("message_fts", "message_id"),
    "tool_call": ("tool_calls_fts", "tool_call_id"),
}

SIDECAR_SCHEMA_VERSION = 2
FINGERPRINT_FORMAT = "sha256-length-prefixed-utf8-v1"

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

    check_read_cancelled()
    conn = sqlite3.connect(path)
    conn.set_progress_handler(read_cancelled, 1_000)
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if existing:
            validate_sidecar_schema(conn)
            conn.commit()
            return conn
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
                    message_id TEXT NOT NULL UNIQUE,
                    content_fingerprint TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_calls_fts_rowid (
                    rowid INTEGER PRIMARY KEY,
                    tool_call_id TEXT NOT NULL UNIQUE,
                    content_fingerprint TEXT NOT NULL
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
            conn.executemany(
                "INSERT INTO fts_sidecar_meta(key, value) VALUES (?, ?)",
                [
                    ("schema_version", str(SIDECAR_SCHEMA_VERSION)),
                    ("fingerprint_format", FINGERPRINT_FORMAT),
                ],
            )
            # Older binaries insert mappings without a fingerprint, even for
            # existing ids. ABORT also defeats their INSERT OR IGNORE policy.
            for table, _ in _FTS_TABLES.values():
                conn.execute(f"""
                    CREATE TRIGGER {table}_require_fingerprint
                    BEFORE INSERT ON {table}_rowid
                    WHEN NEW.content_fingerprint IS NULL
                    BEGIN
                        SELECT RAISE(ABORT, 'keyword index requires fingerprint-aware writer');
                    END
                """)
    except (sqlite3.Error, FtsSidecarUnavailableError):
        conn.close()
        raise
    return conn


def open_sidecar_readonly(path: Path) -> sqlite3.Connection:
    """Open an existing keyword index with request-local cancellation.

    Temporary candidate membership is connection-local and writable; the main
    index remains read-only. SQLite's existing five-second busy wait bounds
    lock waits that cannot run its VM progress handler.
    """
    check_read_cancelled()
    if not path.exists():
        # Empty indexes have always supported searches before their first write.
        conn = open_sidecar(path)
        conn.close()
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)
    conn.set_progress_handler(read_cancelled, 1_000)
    return conn


def _candidate_join(
    conn: sqlite3.Connection, candidate_ids: Iterable[str] | None, id_column: str
) -> str:
    if candidate_ids is None:
        return ""
    conn.execute(
        "CREATE TEMP TABLE IF NOT EXISTS fts_candidates(id TEXT PRIMARY KEY) WITHOUT ROWID"
    )
    conn.execute("DELETE FROM temp.fts_candidates")
    for batch in batched(candidate_ids, _FTS_ROWID_LOOKUP_CHUNK):
        check_read_cancelled()
        # Cancellation can leave at most one batch to roll back at close.
        with conn:
            conn.executemany(
                "INSERT INTO temp.fts_candidates VALUES (?)", [(key,) for key in batch]
            )
    # Eligibility changes membership only: bm25 still sees the full FTS corpus.
    return f"JOIN temp.fts_candidates eligible ON eligible.id = {id_column}"


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


@measured(Stage.FTS)
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
        with publish_sidecar(sidecar_conn) as publication:
            publication.messages(batch, fields=field_tuple)


def upsert_tool_call_fts(
    sidecar_conn: sqlite3.Connection,
    tool_call_id: str,
    bash_command: str | None,
    *,
    fields: Iterable[str] | None = None,
) -> None:
    upsert_tool_call_fts_batch(sidecar_conn, [(tool_call_id, bash_command)], fields=fields)


@measured(Stage.FTS)
def upsert_tool_call_fts_batch(
    sidecar_conn: sqlite3.Connection,
    rows: Iterable[tuple[str, str | None]],
    *,
    fields: Iterable[str] | None = None,
    batch_size: int | None = None,
) -> None:
    """Write tool-call FTS rows in bounded transactions."""
    field_tuple = tuple(fields) if fields is not None else None
    for batch in batched(rows, _resolve_batch_size(batch_size)):
        with publish_sidecar(sidecar_conn) as publication:
            publication.tools(batch, fields=field_tuple)


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
    with publish_sidecar(sidecar_conn) as publication:
        publication.delete(materialized, "message")


def delete_tool_call_fts(
    sidecar_conn: sqlite3.Connection,
    tool_call_ids: Iterable[str],
) -> None:
    materialized = list(dict.fromkeys(tool_call_ids))
    if not materialized:
        return
    with publish_sidecar(sidecar_conn) as publication:
        publication.delete(materialized, "tool_call")


def search_messages_fts(
    sidecar_conn: sqlite3.Connection,
    query: str,
    fields: list[str],
    limit: int,
    *,
    offset: int = 0,
    candidate_ids: Iterable[str] | None = None,
) -> list[tuple[str, float]]:
    """Return message FTS hits with scores normalized to higher-is-better.

    Equal scores order by id. Candidate membership is applied before top-k,
    without changing corpus-wide BM25 statistics.
    """
    if not query.strip() or limit <= 0:
        return []
    if offset < 0:
        raise ValueError("message FTS offset must be non-negative")
    match_query = _message_match_query(query, fields)
    if not match_query:
        return []
    candidate_join = _candidate_join(sidecar_conn, candidate_ids, "mr.message_id")
    rows = sidecar_conn.execute(
        f"""
        SELECT mr.message_id, -bm25(message_fts) AS score
        FROM message_fts
        JOIN message_fts_rowid mr ON mr.rowid = message_fts.rowid
        {candidate_join}
        WHERE message_fts MATCH ?
        ORDER BY bm25(message_fts) ASC, mr.message_id ASC
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
    candidate_ids: Iterable[str] | None = None,
) -> list[tuple[str, float]]:
    """Return tool call FTS hits with scores normalized to higher-is-better.

    Equal scores order by id. Candidate membership is applied before top-k,
    without changing corpus-wide BM25 statistics.
    """
    if not query.strip() or limit <= 0:
        return []
    if offset < 0:
        raise ValueError("tool call FTS offset must be non-negative")
    # Escape like the message path so identifiers with -, :, (, ) are searched as
    # terms instead of leaking raw FTS5 errors via `recall search --tool`.
    match_query = _escape_fts5_query(query)
    if not match_query:
        return []
    candidate_join = _candidate_join(sidecar_conn, candidate_ids, "tr.tool_call_id")
    rows = sidecar_conn.execute(
        f"""
        SELECT tr.tool_call_id, -bm25(tool_calls_fts) AS score
        FROM tool_calls_fts
        JOIN tool_calls_fts_rowid tr ON tr.rowid = tool_calls_fts.rowid
        {candidate_join}
        WHERE tool_calls_fts MATCH ?
        ORDER BY bm25(tool_calls_fts) ASC, tr.tool_call_id ASC
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


def content_fingerprint(columns: tuple[str, ...]) -> str:
    """Hash ordered effective columns with byte lengths, including empty columns."""
    digest = hashlib.sha256(FINGERPRINT_FORMAT.encode("ascii") + b"\0")
    for column in columns:
        encoded = column.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def validate_sidecar_schema(conn: sqlite3.Connection) -> None:
    try:
        meta = dict(conn.execute("SELECT key, value FROM fts_sidecar_meta"))
        columns_valid = all(
            any(
                row[1] == "content_fingerprint" and row[3] == 1
                for row in conn.execute(f"PRAGMA table_info({table}_rowid)")
            )
            for table, _ in _FTS_TABLES.values()
        )
    except sqlite3.Error as err:
        raise FtsSidecarUnavailableError(
            "Keyword index requires explicit canonical initialization; "
            "see docs/keyword-initialization.md."
        ) from err
    if (
        meta.get("schema_version") != str(SIDECAR_SCHEMA_VERSION)
        or meta.get("fingerprint_format") != FINGERPRINT_FORMAT
        or not columns_valid
    ):
        raise FtsSidecarUnavailableError(
            "Keyword index requires explicit canonical initialization; "
            "see docs/keyword-initialization.md."
        )


class _Publication:
    """Publication capability valid only inside the owning SQLite transaction."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.active = True

    def messages(
        self,
        rows: Iterable[tuple[str, str, str]],
        *,
        fields: Iterable[str] | None = None,
    ) -> None:
        field_tuple = tuple(fields) if fields is not None else None
        scoped = [
            (key, scope_message_fts_columns(content, thinking, field_tuple))
            for key, content, thinking in rows
        ]
        self._publish(scoped, "message")

    def tools(
        self,
        rows: Iterable[tuple[str, str | None]],
        *,
        fields: Iterable[str] | None = None,
    ) -> None:
        index_bash = should_index_bash_fts(fields)
        scoped = [
            (key, (command,) if command is not None and index_bash else None)
            for key, command in rows
        ]
        self._publish(scoped, "tool_call")

    def delete(self, identifiers: Iterable[str], kind: _FtsKind) -> None:
        assert self.active and self._conn.in_transaction
        _delete_fts_rows(self._conn, identifiers, kind)

    def _publish(
        self,
        rows: Sequence[tuple[str, tuple[str, ...] | None]],
        kind: _FtsKind,
    ) -> None:
        assert self.active and self._conn.in_transaction
        table, id_column = _FTS_TABLES[kind]
        known: dict[str, tuple[int, str]] = {}
        for chunk in batched(dict.fromkeys(key for key, _ in rows), _FTS_ROWID_LOOKUP_CHUNK):
            placeholders = ",".join("?" for _ in chunk)
            known.update(
                {
                    str(key): (int(rowid), str(fingerprint))
                    for key, rowid, fingerprint in self._conn.execute(
                        f"SELECT {id_column}, rowid, content_fingerprint FROM {table}_rowid "
                        f"WHERE {id_column} IN ({placeholders})",
                        chunk,
                    )
                }
            )
        columns = "fts_content, fts_thinking" if kind == "message" else "bash_command"
        for key, values in rows:
            if values is None:
                self.delete([key], kind)
                known.pop(key, None)
                continue
            fingerprint = content_fingerprint(values)
            prior = known.get(key)
            if prior is not None and prior[1] == fingerprint:
                continue
            if prior is None:
                cursor = self._conn.execute(
                    f"INSERT INTO {table}_rowid({id_column}, content_fingerprint) VALUES (?, ?)",
                    [key, fingerprint],
                )
                assert cursor.lastrowid is not None
                rowid = cursor.lastrowid
            else:
                rowid = prior[0]
                self._conn.execute(
                    f"UPDATE {table}_rowid SET content_fingerprint=? WHERE rowid=?",
                    [fingerprint, rowid],
                )
            placeholders = ",".join("?" for _ in range(len(values) + 1))
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table}(rowid, {columns}) VALUES ({placeholders})",
                (rowid, *values),
            )
            known[key] = (rowid, fingerprint)


@contextmanager
def publish_sidecar(conn: sqlite3.Connection) -> Iterator[_Publication]:
    """Own a bounded publication transaction, including the caller's progress marker."""
    assert not conn.in_transaction, "publication cannot nest or commit another owner's work"
    conn.execute("BEGIN IMMEDIATE")
    publication = _Publication(conn)
    try:
        validate_sidecar_schema(conn)
        yield publication
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        publication.active = False


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
