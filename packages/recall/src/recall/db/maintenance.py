"""Index maintenance: rebuild every ART index and probe index/table divergence.

A failed checkpoint (disk full during fsync) can leave rows in a table that are
absent from its ART indexes. DuckDB then answers every UPDATE of such a row with
``Failed to delete all rows from index`` at COMMIT and invalidates the whole
database instance, so the daemon dies identically on every restart. All of
recall's explicit indexes are non-unique performance indexes, so dropping and
recreating them from the catalog is safe and takes about a second on a 2.2M-row
database (REQ-RESIL-017). The probe (REQ-RESIL-018) detects the condition
read-only by comparing a point lookup with a forced sequential scan of the
same key. Healthy sampled coverage additionally requires attributed analyzed
index access; equal counts alone do not establish index use.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Self

import duckdb
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from recall.core.time import utcnow_naive

logger = logging.getLogger("recall.db.maintenance")

PROBE_PLAN_BYTES_MAX = 2 * 1024 * 1024
PROBE_PLAN_NODES_MAX = 4096
PROBE_PLAN_DEPTH_MAX = 32
PROBE_CATALOG_ROWS_MAX = 65536
PROBE_KEY_BYTES_MAX = 64 * 1024
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

    @property
    def coverage_verified(self) -> bool:
        return self.complete and self.samples_unverifiable == 0 and not self.diverged

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
        WHERE database_name=current_database() AND schema_name = 'main'
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
        if len(expressions) > 64 or not all(isinstance(part, str) for part in expressions):
            return ()
        parts = expressions
    elif isinstance(expressions, str) and len(expressions) <= 65536:
        text = expressions.strip()
        if not (text.startswith("[") and text.endswith("]")):
            return ()
        parts = text[1:-1].split(",")
    else:
        return ()
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
            WHERE database_name=current_database() AND schema_name = 'main'
            """
        ).fetchmany(PROBE_CATALOG_ROWS_MAX + 1)
    }
    if len(column_types) > PROBE_CATALOG_ROWS_MAX:
        raise _ProbeCatalogUnavailable
    rows = conn.execute(
        """
        SELECT index_name, table_name, expressions
        FROM duckdb_indexes()
        WHERE database_name=current_database() AND schema_name = 'main' AND NOT is_primary
        ORDER BY index_name
        """
    ).fetchmany(PROBE_CATALOG_ROWS_MAX + 1)
    if len(rows) > PROBE_CATALOG_ROWS_MAX:
        raise _ProbeCatalogUnavailable
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
    field_limit = PROBE_KEY_BYTES_MAX // max(1, len(target.columns))
    projected = ", ".join(
        f"CASE WHEN octet_length(encode({_quote(column)})) <= {field_limit} "
        f"THEN {_quote(column)} END"
        for column in target.columns
    )
    not_null = " AND ".join(f"{_quote(column)} IS NOT NULL" for column in target.columns)
    # rowid grows with appends, so the highest rowid per key is the most recently
    # written occurrence -- the rows a failed checkpoint is most likely to have
    # left out of the index.
    sql = (
        f"SELECT {projected} FROM ("
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


class _AnalyzedAccess(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    table: str | None = Field(default=None, alias="Table")
    access_type: str | None = Field(default=None, alias="Type")
    filters: str | tuple[str, ...] | None = Field(default=None, alias="Filters")


class _AnalyzedOperator(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    operator_type: str | None = None
    operator_cardinality: int | None = None
    extra_info: _AnalyzedAccess = Field(default_factory=_AnalyzedAccess)
    children: tuple[Self, ...] = Field(default=(), max_length=PROBE_PLAN_NODES_MAX)


class _ProbeCatalogUnavailable(Exception):
    pass


def _catalog_rows(conn: _Queryable, sql: str) -> list[tuple[Any, ...]]:
    rows = conn.execute(sql).fetchmany(PROBE_CATALOG_ROWS_MAX + 1)
    if len(rows) > PROBE_CATALOG_ROWS_MAX:
        raise _ProbeCatalogUnavailable
    return rows


def _attributable_targets(conn: _Queryable) -> frozenset[str]:
    paths: dict[tuple[str, str], list[str | None]] = {}
    unsupported: set[str] = set()
    for name, table, expressions in _catalog_rows(
        conn,
        "SELECT index_name,table_name,expressions FROM duckdb_indexes() "
        "WHERE database_name=current_database() AND schema_name='main'",
    ):
        columns = _parse_index_expressions(expressions)
        if not columns or any(
            re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", column) is None for column in columns
        ):
            unsupported.add(table)
        elif len(columns) == 1:
            paths.setdefault((table, columns[0]), []).append(name)
    for table, columns in _catalog_rows(
        conn,
        "SELECT table_name,constraint_column_names FROM duckdb_constraints() "
        "WHERE database_name=current_database() AND schema_name='main' "
        "AND constraint_type IN ('PRIMARY KEY','UNIQUE','FOREIGN KEY')",
    ):
        if (
            not isinstance(columns, (list, tuple))
            or not columns
            or not all(isinstance(column, str) for column in columns)
        ):
            unsupported.add(table)
        elif len(columns) == 1:
            paths.setdefault((table, columns[0]), []).append(None)
    return frozenset(
        names[0]
        for (table, _column), names in paths.items()
        if table not in unsupported and len(names) == 1 and names[0] is not None
    )


def _qualified_table(value: str | None) -> tuple[str, ...] | None:
    if value is None or len(value) > 4096:
        return None
    token = r'(?:"(?:[^"]|"")*"|[A-Za-z_][A-Za-z_0-9]*)'
    if re.fullmatch(token + r"(?:\." + token + r")*", value) is None:
        return None
    return tuple(
        part[1:-1].replace('""', '"') if part.startswith('"') else part
        for part in re.findall(token, value)
    )


def _analyzed_index_access(
    conn: _Queryable,
    sql: str,
    key: tuple[Any, ...],
    target: IndexProbeTarget,
    database: str,
    count: int,
) -> bool:
    row = conn.execute("EXPLAIN (ANALYZE,FORMAT JSON) " + sql, list(key)).fetchone()
    if row is None or len(row) != 2 or row[0] != "analyzed_plan" or not isinstance(row[1], str):
        return False
    if len(row[1].encode()) > PROBE_PLAN_BYTES_MAX:
        return False
    try:
        root = _AnalyzedOperator.model_validate_json(row[1])
    except (ValidationError, ValueError, RecursionError):
        return False
    stack = [(root, 0)]
    visited = 0
    explained = False
    scans: list[_AnalyzedOperator] = []
    while stack:
        node, depth = stack.pop()
        visited += 1
        if depth > PROBE_PLAN_DEPTH_MAX or visited > PROBE_PLAN_NODES_MAX:
            return False
        if node.operator_type == "EXPLAIN_ANALYZE":
            explained = True
        if node.operator_type == "TABLE_SCAN":
            scans.append(node)
        stack.extend((child, depth + 1) for child in node.children)
    if not explained or len(scans) != 1:
        return False
    scan = scans[0]
    if scan.extra_info.access_type != "Index Scan" or scan.operator_cardinality != count:
        return False
    if _qualified_table(scan.extra_info.table) != (database, "main", target.table):
        return False
    assert len(target.columns) == 1 and len(key) == 1 and isinstance(key[0], str)
    value = "'" + key[0].replace("'", "''") + "'"
    column = target.columns[0]
    expected = {column + "=" + value, _quote(column) + "=" + value}
    filters = scan.extra_info.filters
    if isinstance(filters, tuple):
        return len(filters) == 1 and filters[0] in expected
    return filters in expected


def probe_index_divergence(
    conn: duckdb.DuckDBPyConnection,
    *,
    sample: int = PROBE_SAMPLE_DEFAULT,
    budget_seconds: float = PROBE_BUDGET_SECONDS_DEFAULT,
    now: Callable[[], datetime] = utcnow_naive,
) -> IndexDivergenceReport:
    """Compare counts and establish attributed analyzed index access per sampled key.

    Read-only. A sample whose full-scan count exceeds DuckDB's index-scan bound
    is reported as unverifiable rather than healthy. Compound or competing
    access paths and unrecognized analyzed metadata also remain unverifiable.
    No query -- target
    discovery included -- starts after ``budget_seconds`` has elapsed; whatever
    was not reached stays unverified and the report is marked incomplete.
    """
    if not 1 <= sample <= PROBE_SAMPLE_MAX:
        raise ValueError(f"sample must be between 1 and {PROBE_SAMPLE_MAX}")
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
        targets = index_probe_targets(guarded)
        attributable = _attributable_targets(guarded)
        database_row = guarded.execute("SELECT current_database()").fetchone()
        assert database_row is not None and isinstance(database_row[0], str)
        database = database_row[0]
        for target in targets:
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
                if any(value is None for value in key):
                    samples_unverifiable += 1
                    continue
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
                elif (
                    len(target.columns) != 1
                    or target.index_name not in attributable
                    or full_count > bound
                    or not _analyzed_index_access(
                        guarded,
                        f"SELECT COUNT(*) FROM {table} WHERE {index_arm}",
                        key,
                        target,
                        database,
                        index_count,
                    )
                ):
                    samples_unverifiable += 1
                else:
                    samples_checked += 1
    except (_BudgetExhausted, _ProbeCatalogUnavailable):
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
