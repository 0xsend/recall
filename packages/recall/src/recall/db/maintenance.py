"""Index maintenance: rebuild every ART index and probe index/table divergence.

A failed checkpoint (disk full during fsync) can leave rows in a table that are
absent from its ART indexes. DuckDB then answers every UPDATE of such a row with
``Failed to delete all rows from index`` at COMMIT and invalidates the whole
database instance, so the daemon dies identically on every restart. All of
recall's explicit indexes are non-unique performance indexes, so dropping and
recreating them from the catalog is safe and takes about a second on a 2.2M-row
database (REQ-RESIL-017). The probe (REQ-RESIL-018) detects the condition
read-only by comparing an index-eligible point lookup with a forced sequential
scan of the same key.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import duckdb

from recall.core.time import utcnow_naive

logger = logging.getLogger("recall.db.maintenance")

PROBE_SAMPLE_DEFAULT = 2
PROBE_SAMPLE_MAX = 10
PROBE_BUDGET_SECONDS_DEFAULT = 10.0

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_CREATE_INDEX_RE = re.compile(r"^CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", re.IGNORECASE)


@dataclass(frozen=True)
class IndexRebuildResult:
    dropped: int
    created: int
    healed: tuple[str, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class IndexProbeTarget:
    index_name: str
    table: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class DivergedKey:
    table: str
    column: str
    key: str
    index_count: int
    full_count: int


@dataclass(frozen=True)
class IndexDivergenceReport:
    checked_at: datetime
    indexes_probed: int
    samples_checked: int
    samples_unverifiable: int
    diverged: tuple[DivergedKey, ...]
    complete: bool
    elapsed_seconds: float

    @property
    def diverged_count(self) -> int:
        return len(self.diverged)

    def to_payload(self) -> dict[str, Any]:
        """JSON-ready shape shared by the RPC, `daemon status`, and the CLI."""
        payload = asdict(self)
        payload["checked_at"] = self.checked_at.isoformat()
        payload["diverged"] = [asdict(key) for key in self.diverged]
        payload["diverged_count"] = self.diverged_count
        return payload


def schema_index_statements() -> tuple[str, ...]:
    """Every ``CREATE INDEX IF NOT EXISTS`` statement in schema.sql, whitespace-normalized."""
    lines = [
        line
        for line in _SCHEMA_PATH.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("--")
    ]
    statements: list[str] = []
    for chunk in "\n".join(lines).split(";"):
        statement = " ".join(chunk.split())
        if _CREATE_INDEX_RE.match(statement):
            statements.append(statement)
    assert statements, "schema.sql defines no indexes; the parser or the schema is broken"
    return tuple(statements)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _index_name(statement: str) -> str:
    match = _CREATE_INDEX_RE.match(statement)
    assert match is not None, statement
    return match.group(1)


def rebuild_indexes(conn: duckdb.DuckDBPyConnection) -> IndexRebuildResult:
    """Drop and recreate every catalog index, heal schema drift, then CHECKPOINT.

    Runs in autocommit so an interrupted rebuild converges on the next run: the
    healing pass recreates any schema.sql index the interruption left missing.
    PRIMARY KEY / UNIQUE constraint indexes are not listed by ``duckdb_indexes()``
    and are never touched.
    """
    start = time.perf_counter()
    listed = conn.execute(
        """
        SELECT index_name, sql
        FROM duckdb_indexes()
        WHERE schema_name = 'main'
        ORDER BY index_name
        """
    ).fetchall()
    dropped = 0
    created = 0
    listed_names: set[str] = set()
    for name_value, sql_value in listed:
        name = str(name_value)
        listed_names.add(name)
        if sql_value is None:
            logger.warning("index %s has no stored SQL; leaving it in place", name)
            continue
        conn.execute(f"DROP INDEX {_quote(name)}")
        dropped += 1
        conn.execute(str(sql_value))
        created += 1

    healed: list[str] = []
    for statement in schema_index_statements():
        name = _index_name(statement)
        if name in listed_names:
            continue
        try:
            conn.execute(statement)
        except duckdb.CatalogException as err:
            # A stale-schema database may lack the table; the migration that adds
            # it recreates the index. Nothing to heal here.
            logger.warning("skipping schema index %s: %s", name, err)
            continue
        created += 1
        healed.append(name)

    conn.execute("CHECKPOINT")
    elapsed = time.perf_counter() - start
    logger.info(
        "rebuilt indexes dropped=%d created=%d healed=%s elapsed=%.3fs",
        dropped,
        created,
        ",".join(healed) or "-",
        elapsed,
    )
    return IndexRebuildResult(
        dropped=dropped,
        created=created,
        healed=tuple(healed),
        elapsed_seconds=elapsed,
    )


def _parse_index_expressions(expressions: Any) -> tuple[str, ...]:
    """Turn duckdb_indexes().expressions (``[cwd]`` / ``[a, b]`` or a list) into names."""
    if isinstance(expressions, (list, tuple)):
        parts = [str(part) for part in expressions]
    else:
        text = str(expressions).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = text.split(",")
    # A quoted column renders as '"source"' inside the list: both quote kinds go.
    return tuple(part.strip().strip("'\"") for part in parts if part.strip())


class _Queryable(Protocol):
    """The slice of a DuckDB connection the probe uses; lets the deadline guard wrap it."""

    def execute(self, query: Any, parameters: Any = None) -> Any: ...


class _BudgetExhausted(Exception):
    """Raised by `_DeadlineGuardedConnection` instead of starting a query past the deadline."""


class _DeadlineGuardedConnection:
    """Every probe query passes through here, so none can start after the deadline.

    REQ-RESIL-018 bounds the daemon's exclusive startup gate by the probe
    budget. Checking once per key was not enough: discovery, the row count, the
    two settings reads, and the key sample all ran unchecked.
    """

    def __init__(self, inner: _Queryable, deadline: float) -> None:
        self._inner = inner
        self._deadline = deadline

    def execute(self, query: Any, parameters: Any = None) -> Any:
        if time.perf_counter() >= self._deadline:
            raise _BudgetExhausted
        return self._inner.execute(query, parameters)


def index_probe_targets(conn: _Queryable) -> tuple[IndexProbeTarget, ...]:
    """Every non-constraint ``main`` index whose columns are all VARCHAR.

    Derived from the catalog so a new index is probed without a code change.
    Non-VARCHAR columns are skipped because the forced-sequential arm relies on
    string concatenation to defeat index selection.
    """
    column_types = {
        (str(table), str(column)): str(data_type).upper()
        for table, column, data_type in conn.execute(
            """
            SELECT table_name, column_name, data_type
            FROM duckdb_columns()
            WHERE schema_name = 'main'
            """
        ).fetchall()
    }
    rows = conn.execute(
        """
        SELECT index_name, table_name, expressions
        FROM duckdb_indexes()
        WHERE schema_name = 'main' AND NOT is_primary
        ORDER BY index_name
        """
    ).fetchall()
    targets: list[IndexProbeTarget] = []
    for index_name, table, expressions in rows:
        columns = _parse_index_expressions(expressions)
        if not columns:
            continue
        if any(column_types.get((str(table), column)) != "VARCHAR" for column in columns):
            continue
        targets.append(IndexProbeTarget(str(index_name), str(table), columns))
    return tuple(targets)


def _count(conn: _Queryable, sql: str, params: Sequence[Any]) -> int:
    row = conn.execute(sql, list(params)).fetchone()
    assert row is not None, sql
    return int(row[0])


def _recent_keys(conn: _Queryable, target: IndexProbeTarget, sample: int) -> list[tuple[Any, ...]]:
    columns = ", ".join(_quote(column) for column in target.columns)
    not_null = " AND ".join(f"{_quote(column)} IS NOT NULL" for column in target.columns)
    # rowid grows with appends, so the highest rowid per key is the most recently
    # written occurrence -- the rows a failed checkpoint is most likely to have
    # left out of the index.
    sql = (
        f"SELECT {columns} FROM ("
        f"SELECT {columns}, max(rowid) AS r FROM {_quote(target.table)} "
        f"WHERE {not_null} GROUP BY {columns}"
        f") ORDER BY r DESC LIMIT ?"
    )
    return [tuple(row) for row in conn.execute(sql, [sample]).fetchall()]


def _index_scan_bound(conn: _Queryable, table_rows: int) -> int:
    max_count = _count(conn, "SELECT current_setting('index_scan_max_count')", [])
    row = conn.execute("SELECT current_setting('index_scan_percentage')").fetchone()
    assert row is not None
    percentage = float(row[0])
    return max(max_count, int(percentage * table_rows))


def probe_index_divergence(
    conn: duckdb.DuckDBPyConnection,
    *,
    sample: int = PROBE_SAMPLE_DEFAULT,
    budget_seconds: float = PROBE_BUDGET_SECONDS_DEFAULT,
    now: Callable[[], datetime] = utcnow_naive,
) -> IndexDivergenceReport:
    """Compare an index-eligible lookup with a forced sequential scan per sampled key.

    Read-only. A sample whose full-scan count exceeds DuckDB's index-scan bound
    is reported as unverifiable rather than healthy: above the bound the planner
    scans both arms sequentially and cannot see divergence. No query -- target
    discovery included -- starts after ``budget_seconds`` has elapsed; whatever
    was not reached stays unverified and the report is marked incomplete.
    """
    if sample < 1:
        raise ValueError("sample must be at least 1")
    if budget_seconds < 0:
        raise ValueError("budget_seconds must not be negative")
    checked_at = now()
    start = time.perf_counter()
    guarded = _DeadlineGuardedConnection(conn, start + budget_seconds)

    indexes_probed = 0
    samples_checked = 0
    samples_unverifiable = 0
    diverged: list[DivergedKey] = []
    complete = True
    try:
        for target in index_probe_targets(guarded):
            table = _quote(target.table)
            table_rows = _count(guarded, f"SELECT COUNT(*) FROM {table}", [])
            bound = _index_scan_bound(guarded, table_rows)
            keys = _recent_keys(guarded, target, sample)
            indexes_probed += 1
            index_arm = " AND ".join(f"{_quote(column)} = ?" for column in target.columns)
            sequential_arm = " AND ".join(
                f"({_quote(column)} || '') = ?" for column in target.columns
            )
            for key in keys:
                index_count = _count(
                    guarded, f"SELECT COUNT(*) FROM {table} WHERE {index_arm}", key
                )
                full_count = _count(
                    guarded, f"SELECT COUNT(*) FROM {table} WHERE {sequential_arm}", key
                )
                if index_count != full_count:
                    diverged.append(
                        DivergedKey(
                            table=target.table,
                            column=", ".join(target.columns),
                            key=", ".join(str(part) for part in key),
                            index_count=index_count,
                            full_count=full_count,
                        )
                    )
                elif full_count > bound:
                    samples_unverifiable += 1
                else:
                    samples_checked += 1
    except _BudgetExhausted:
        # Budget-truncated: the unreached remainder is unverified, reported
        # through `complete`, never as a healthy count.
        complete = False

    elapsed = time.perf_counter() - start
    report = IndexDivergenceReport(
        checked_at=checked_at,
        indexes_probed=indexes_probed,
        samples_checked=samples_checked,
        samples_unverifiable=samples_unverifiable,
        diverged=tuple(diverged),
        complete=complete,
        elapsed_seconds=elapsed,
    )
    if report.diverged:
        logger.warning(
            "index/table divergence detected keys=%d indexes_probed=%d elapsed=%.3fs: %s",
            report.diverged_count,
            indexes_probed,
            elapsed,
            "; ".join(
                f"{k.table}.{k.column}={k.key!r} index={k.index_count} full={k.full_count}"
                for k in report.diverged[:5]
            ),
        )
    else:
        logger.info(
            "index probe clean indexes_probed=%d checked=%d unverifiable=%d complete=%s "
            "elapsed=%.3fs",
            indexes_probed,
            samples_checked,
            samples_unverifiable,
            complete,
            elapsed,
        )
    return report
