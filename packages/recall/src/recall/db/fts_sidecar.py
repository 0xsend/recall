from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from recall.core.read_cancellation import check_read_cancelled, read_cancelled
from recall.core.telemetry import Span, Stage, measured

if TYPE_CHECKING:
    from recall.core.config import AppConfig

MIN_SQLITE_VERSION: tuple[int, int, int] = (3, 43, 0)
FTS_SIDECAR_BATCH_SIZE = 10_000
# Total time one writer-turn publication may wait for sidecar locks
# (REQ-FTS-SIDECAR-018). Uncontended statements never draw on it.
KEYWORD_PUBLISH_DEADLINE_S = 1.0
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


class FtsSidecarCorruptError(sqlite3.DatabaseError):
    """The file cannot serve as a keyword index in its declared format."""


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


def initialize_sidecar[T](path: Path, initialize: Callable[[sqlite3.Connection, bool], T]) -> T:
    """Initialize under the startup owner's lock, rebuilding an unusable file once.

    Keep format/feature refusals and operating failures visible. SQLite reports
    corruption from connect, schema access or later VM work, using both primary
    and extended result codes; all first-use work belongs inside this boundary.
    The callback receives whether the index existed before this open.
    """
    existed = path.exists()
    try:
        with closing(open_sidecar(path, check_fts_reads=True)) as conn:
            return initialize(conn, existed)
    except sqlite3.Error as err:
        code = getattr(err, "sqlite_errorcode", 0) & 0xFF
        if not isinstance(err, FtsSidecarCorruptError) and code not in {
            sqlite3.SQLITE_CORRUPT,
            sqlite3.SQLITE_NOTADB,
        }:
            raise
        retained = Path(tempfile.mkdtemp(prefix=f"{path.name}.corrupt-", dir=path.parent))
        logger.warning("Unusable keyword sidecar: %s; retaining files in %s", err, retained)
        # A corrupt database can have committed content in a WAL or a pending
        # rollback journal. Move the main file last, so a crash cannot expose a
        # missing database beside the old journals to a fresh open.
        for suffix in ("-wal", "-shm", "-journal", ""):
            source = path.with_name(path.name + suffix)
            if source.exists():
                source.rename(retained / source.name)
    with closing(open_sidecar(path)) as conn:
        return initialize(conn, False)


def _probe_sidecar_reads(conn: sqlite3.Connection) -> None:
    # Schema introspection does not read FTS's statistics header. Exercise the
    # ranking VM on at most one document before declaring the index usable.
    for table, _ in _FTS_TABLES.values():
        conn.execute(f"SELECT bm25({table}) FROM {table} LIMIT 1").fetchone()


def open_sidecar(path: Path, *, check_fts_reads: bool = False) -> sqlite3.Connection:
    probe_sqlite_fts5_support()
    path.parent.mkdir(parents=True, exist_ok=True)

    check_read_cancelled()
    conn = sqlite3.connect(path)
    conn.set_progress_handler(read_cancelled, 1_000)
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if existing:
            validate_sidecar_schema(conn, check_fts_reads=check_fts_reads)
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


def ensure_sidecar(config: AppConfig) -> None:
    """Create or validate the keyword index before a context's first write.

    Publication opens the file with ``mode=rw`` and never creates it, so every
    context that writes without a daemon start calls this once during setup.
    """
    if config.fts.backend != "sqlite_sidecar":
        return
    open_sidecar(sidecar_path(config.data_dir)).close()


def open_sidecar_for_publication(path: Path) -> sqlite3.Connection:
    """Open an existing keyword index for one bounded publication.

    The connection never creates the file, takes no transaction of its own and
    waits for no lock: `publish_sidecar` validates the schema inside its own
    transaction and charges any lock wait to its `PublicationDeadline`.
    A missing file raises `sqlite3.OperationalError`.
    """
    probe_sqlite_fts5_support()
    return sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=0)


def open_sidecar_readonly(path: Path) -> sqlite3.Connection:
    """Open an existing keyword index with request-local cancellation.

    Never creates the file: an index a running daemon lost must stay missing
    until startup rebuilds it, not reappear empty and pass for repaired.
    Temporary candidate membership is connection-local and writable; the main
    index remains read-only. SQLite's existing five-second busy wait bounds
    lock waits that cannot run its VM progress handler.
    """
    check_read_cancelled()
    if not path.exists():
        raise sqlite3.OperationalError(f"keyword index {path} does not exist")
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
) -> int:
    """Write message FTS rows in bounded transactions.

    Large indexes can contain millions of FTS rows. Keeping the transaction
    bound explicit prevents unbounded SQLite rollback journals while avoiding a
    commit per row on the hot indexing path.
    """
    field_tuple = tuple(fields) if fields is not None else None
    rewritten = 0
    for batch in batched(rows, _resolve_batch_size(batch_size)):
        with publish_sidecar(sidecar_conn) as publication:
            rewritten += publication.messages(batch, fields=field_tuple)
    return rewritten


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
) -> int:
    """Write tool-call FTS rows in bounded transactions."""
    field_tuple = tuple(fields) if fields is not None else None
    rewritten = 0
    for batch in batched(rows, _resolve_batch_size(batch_size)):
        with publish_sidecar(sidecar_conn) as publication:
            rewritten += publication.tools(batch, fields=field_tuple)
    return rewritten


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


def validate_sidecar_schema(conn: sqlite3.Connection, *, check_fts_reads: bool = False) -> None:
    try:
        _validate_sidecar_schema(conn)
        if check_fts_reads:
            _probe_sidecar_reads(conn)
    except sqlite3.Error as err:
        # With the runtime probe passed and fixed validation SQL, SQLITE_ERROR
        # means an unusable schema (e.g. a broken FTS virtual-table constructor).
        # Do not extend this translation to the caller's initialization SQL.
        if getattr(err, "sqlite_errorcode", 0) & 0xFF == sqlite3.SQLITE_ERROR:
            raise FtsSidecarCorruptError(f"Invalid keyword index schema: {err}") from err
        raise


def _validate_sidecar_schema(conn: sqlite3.Connection) -> None:
    required = {
        "message_fts": {"fts_content", "fts_thinking"},
        "tool_calls_fts": {"bash_command"},
        "message_fts_rowid": {"rowid", "message_id", "content_fingerprint"},
        "tool_calls_fts_rowid": {"rowid", "tool_call_id", "content_fingerprint"},
        "bootstrap_progress": {"kind", "last_id", "completed_at"},
        "fts_sidecar_meta": {"key", "value"},
    }
    meta_columns = {row[1] for row in conn.execute("PRAGMA table_info(fts_sidecar_meta)")}
    meta = (
        dict(conn.execute("SELECT key, value FROM fts_sidecar_meta"))
        if {"key", "value"} <= meta_columns
        else {}
    )
    # Identify incompatible formats before inspecting their virtual tables:
    # their constructors and layout need not be understood by this runtime.
    if ("schema_version" in meta and meta["schema_version"] != str(SIDECAR_SCHEMA_VERSION)) or (
        "fingerprint_format" in meta and meta["fingerprint_format"] != FINGERPRINT_FORMAT
    ):
        raise FtsSidecarUnavailableError(
            "Keyword index requires explicit canonical initialization; "
            "see docs/keyword-initialization.md."
        )
    columns = {
        f"{table}_rowid": {row[1]: row for row in conn.execute(f"PRAGMA table_info({table}_rowid)")}
        for table, _ in _FTS_TABLES.values()
    }
    legacy = "schema_version" not in meta and any(
        id_column in columns[f"{table}_rowid"]
        and "content_fingerprint" not in columns[f"{table}_rowid"]
        for table, id_column in _FTS_TABLES.values()
    )
    if legacy:
        raise FtsSidecarUnavailableError(
            "Keyword index requires explicit canonical initialization; "
            "see docs/keyword-initialization.md."
        )
    columns.update(
        {
            table: {row[1]: row for row in conn.execute(f"PRAGMA table_info({table})")}
            for table in required
            if table not in columns
        }
    )
    if any(not names <= columns[table].keys() for table, names in required.items()):
        raise FtsSidecarCorruptError("Keyword index is missing required tables or columns")
    table_types = {row[1]: row[2] for row in conn.execute("PRAGMA main.table_list")}
    if any(table_types[table] != "virtual" for table, _ in _FTS_TABLES.values()):
        raise FtsSidecarCorruptError("Keyword index requires virtual FTS tables")
    if not {"schema_version", "fingerprint_format"} <= meta.keys():
        raise FtsSidecarCorruptError("Keyword index format metadata is missing")
    if any(
        columns[f"{table}_rowid"]["content_fingerprint"][3] != 1
        for table, _ in _FTS_TABLES.values()
    ):
        raise FtsSidecarCorruptError("Keyword index fingerprints must be non-null")


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
    ) -> int:
        field_tuple = tuple(fields) if fields is not None else None
        scoped = [
            (key, scope_message_fts_columns(content, thinking, field_tuple))
            for key, content, thinking in rows
        ]
        return self._publish(scoped, "message")

    def tools(
        self,
        rows: Iterable[tuple[str, str | None]],
        *,
        fields: Iterable[str] | None = None,
    ) -> int:
        index_bash = should_index_bash_fts(fields)
        scoped = [
            (key, (command,) if command is not None and index_bash else None)
            for key, command in rows
        ]
        return self._publish(scoped, "tool_call")

    def delete(self, identifiers: Iterable[str], kind: _FtsKind) -> None:
        assert self.active and self._conn.in_transaction
        _delete_fts_rows(self._conn, identifiers, kind)

    def _publish(
        self,
        rows: Sequence[tuple[str, tuple[str, ...] | None]],
        kind: _FtsKind,
    ) -> int:
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
        rewritten = 0
        for key, values in rows:
            if values is None:
                rewritten += int(key in known)
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
            rewritten += 1
            known[key] = (rowid, fingerprint)
        return rewritten


class PublicationDeadlineExceededError(sqlite3.OperationalError):
    """A publication's lock wait used up its `PublicationDeadline`."""


@dataclass
class PublicationDeadline:
    """The lock-wait budget shared by every transaction of one publication.

    Only lock-taking statements whose zero-wait first attempt found the lock
    held are charged, with the elapsed time of their retry. Body work and an
    uncontended commit's fsyncs never are, however large the publication.
    """

    remaining_seconds: float = KEYWORD_PUBLISH_DEADLINE_S
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)

    @property
    def exhausted(self) -> bool:
        return self.remaining_seconds <= 0

    def run_locking(self, conn: sqlite3.Connection, statement: str) -> None:
        """Run `BEGIN IMMEDIATE` or `COMMIT`, waiting only within the budget.

        A busy `COMMIT` leaves its transaction open, so the retry is valid for
        both statements.
        """
        conn.execute("PRAGMA busy_timeout = 0")
        try:
            conn.execute(statement)
            return
        except sqlite3.OperationalError as err:
            if not is_lock_contention(err):
                raise
            if self.exhausted:
                raise PublicationDeadlineExceededError(
                    f"keyword publication lock wait exceeded {KEYWORD_PUBLISH_DEADLINE_S}s"
                ) from err
        started = self.clock()
        conn.execute(f"PRAGMA busy_timeout = {max(1, int(self.remaining_seconds * 1000))}")
        try:
            with Span(Stage.KEYWORD_PUBLISH_LOCK):
                conn.execute(statement)
        except sqlite3.OperationalError as err:
            if not is_lock_contention(err):
                raise
            raise PublicationDeadlineExceededError(
                f"keyword publication lock wait exceeded {KEYWORD_PUBLISH_DEADLINE_S}s: {err}"
            ) from err
        finally:
            self.remaining_seconds -= self.clock() - started
            # Body statements never wait: under a reader, a cache spill would
            # otherwise spend the whole timeout uncharged and then succeed.
            conn.execute("PRAGMA busy_timeout = 0")


def is_lock_contention(err: sqlite3.Error) -> bool:
    """Whether SQLite refused the statement because another connection holds a lock."""
    code = getattr(err, "sqlite_errorcode", 0) & 0xFF
    return code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}


@contextmanager
def publish_sidecar(
    conn: sqlite3.Connection, *, deadline: PublicationDeadline | None = None
) -> Iterator[_Publication]:
    """Own a bounded publication transaction, including the caller's progress marker.

    Without a deadline the connection's own busy timeout applies (bootstrap,
    rescope, startup drain). With one, lock waits are charged to it and body
    statements run without waiting.
    """
    assert not conn.in_transaction, "publication cannot nest or commit another owner's work"
    # Callers retire durable DuckDB intents once this returns, so the commit
    # must survive power loss first. In DELETE journal mode only EXTRA syncs
    # the directory after unlinking the journal; under FULL a lost unlink
    # leaves a hot journal that rolls the publication back after the intents
    # are gone (REQ-FTS-SIDECAR-018). SQLite refuses the change inside a
    # transaction, hence before BEGIN.
    conn.execute("PRAGMA synchronous = EXTRA")
    if deadline is None:
        conn.execute("BEGIN IMMEDIATE")
    else:
        deadline.run_locking(conn, "BEGIN IMMEDIATE")
    publication = _Publication(conn)
    try:
        validate_sidecar_schema(conn)
        yield publication
        if deadline is None:
            conn.commit()
        else:
            deadline.run_locking(conn, "COMMIT")
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


def open_existing_sidecar_readonly(path: Path, *, timeout: float = 5.0) -> sqlite3.Connection:
    """Open existing persistent data read-only; temporary candidates remain local."""
    check_read_cancelled()
    conn = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro", uri=True, timeout=min(5.0, timeout)
    )
    try:
        conn.set_progress_handler(read_cancelled, 1_000)
        validate_sidecar_schema(conn)
    except BaseException:
        conn.close()
        raise
    return conn
