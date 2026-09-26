"""Index rebuild and index/table divergence probe (REQ-RESIL-017, REQ-RESIL-018).

Background: after a disk-full checkpoint failure, rows landed in `session_state`
but not in its ART indexes; every later UPDATE+COMMIT on such a row invalidated
the database. The repair that worked by hand was DROP/CREATE of every entry in
`duckdb_indexes()` plus CHECKPOINT (about one second for 2.2M messages). True
divergence cannot be manufactured through DuckDB's public API -- both an
RLIMIT_FSIZE-forced checkpoint failure and a real ENOSPC on an 8 MiB APFS image
left the file either consistent (WAL replay restored it) or unopenable (reopen
aborts on a duplicate primary key). The probe's detection arm is therefore
exercised through a connection wrapper that answers the two query shapes
differently; everything else runs against real DuckDB.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from recall.db import maintenance
from recall.db.maintenance import (
    IndexDivergenceReport,
    IndexRebuildResult,
    index_probe_targets,
    probe_index_divergence,
    rebuild_indexes,
    schema_index_statements,
)
from recall.db.schema import ensure_schema

_SCHEMA_SQL = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "recall"
    / "src"
    / "recall"
    / "db"
    / "schema.sql"
)


def _index_names(conn: duckdb.DuckDBPyConnection) -> set[str]:
    rows = conn.execute(
        "SELECT index_name FROM duckdb_indexes() WHERE schema_name = 'main'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _schema_index_names() -> set[str]:
    text = _SCHEMA_SQL.read_text(encoding="utf-8")
    return set(re.findall(r"CREATE INDEX IF NOT EXISTS\s+(\w+)", text))


def _fresh_db(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "maint.duckdb"))
    ensure_schema(conn)
    return conn


def _seed_sessions(conn: duckdb.DuckDBPyConnection, count: int) -> None:
    conn.execute(
        """
        INSERT INTO sessions (id, source, source_path)
        SELECT 's' || i, 'codex', '/tmp/s' || i || '.jsonl' FROM range(?) t(i)
        """,
        [count],
    )
    conn.execute(
        """
        INSERT INTO session_state (session_id, cwd, git_repo, file_mtime, file_size)
        SELECT 's' || i, '/cwd' || (i % 3), '/repo' || (i % 2), 1.0, 10 FROM range(?) t(i)
        """,
        [count],
    )


def _row_counts(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    tables = [
        str(row[0])
        for row in conn.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
        ).fetchall()
    ]
    return {table: int(_scalar(conn, f'SELECT COUNT(*) FROM "{table}"')) for table in tables}


def _scalar(conn: duckdb.DuckDBPyConnection, sql: str) -> Any:
    row = conn.execute(sql).fetchone()
    assert row is not None, sql
    return row[0]


class TestSchemaIndexStatements:
    def test_every_schema_index_has_a_statement_including_multiline_ones(self) -> None:
        statements = schema_index_statements()

        names: set[str] = set()
        for statement in statements:
            match = re.search(r"CREATE INDEX IF NOT EXISTS\s+(\w+)", statement)
            assert match is not None, statement
            names.add(match.group(1))
        assert names == _schema_index_names()
        # The usage_events composite index is split across two lines in schema.sql.
        assert any(
            "idx_usage_events_source_sid" in s and "source_session_id" in s for s in statements
        )


class TestRebuildIndexes:
    def test_recreates_listed_indexes_heals_drift_and_preserves_rows(self, tmp_path: Path) -> None:
        conn = _fresh_db(tmp_path)
        try:
            _seed_sessions(conn, 12)
            conn.execute("DROP INDEX idx_sessions_source")  # drift observed on a reference host
            listed_before = _index_names(conn)
            assert "idx_sessions_source" not in listed_before
            counts_before = _row_counts(conn)

            result = rebuild_indexes(conn)

            assert isinstance(result, IndexRebuildResult)
            assert result.dropped == len(listed_before)
            assert result.healed == ("idx_sessions_source",)
            assert result.created == result.dropped + 1
            assert result.elapsed_seconds >= 0.0
            assert _index_names(conn) == _schema_index_names()
            assert _row_counts(conn) == counts_before
            # The rebuilt indexes answer point lookups.
            assert conn.execute(
                "SELECT COUNT(*) FROM session_state WHERE cwd = '/cwd1'"
            ).fetchone() == (4,)
        finally:
            conn.close()

    def test_second_run_heals_nothing_and_keeps_index_set(self, tmp_path: Path) -> None:
        conn = _fresh_db(tmp_path)
        try:
            rebuild_indexes(conn)
            names_after_first = _index_names(conn)

            second = rebuild_indexes(conn)

            assert second.healed == ()
            assert second.dropped == len(names_after_first)
            assert second.created == second.dropped
            assert _index_names(conn) == names_after_first
        finally:
            conn.close()

    def test_index_absent_from_schema_is_recreated_from_its_own_sql(self, tmp_path: Path) -> None:
        conn = _fresh_db(tmp_path)
        try:
            conn.execute("CREATE INDEX idx_local_extra ON session_state(git_branch)")

            result = rebuild_indexes(conn)

            assert "idx_local_extra" in _index_names(conn)
            assert "idx_local_extra" not in result.healed
        finally:
            conn.close()

    def test_rebuild_survives_a_reopen(self, tmp_path: Path) -> None:
        """CHECKPOINT after the rebuild persists the new index storage."""
        conn = _fresh_db(tmp_path)
        _seed_sessions(conn, 5)
        conn.execute("DROP INDEX idx_session_state_cwd")
        rebuild_indexes(conn)
        conn.close()

        reopened = duckdb.connect(str(tmp_path / "maint.duckdb"), read_only=True)
        try:
            assert "idx_session_state_cwd" in _index_names(reopened)
        finally:
            reopened.close()


class TestProbeTargets:
    def test_targets_come_from_catalog_and_skip_non_varchar_indexes(self, tmp_path: Path) -> None:
        conn = _fresh_db(tmp_path)
        try:
            conn.execute("CREATE TABLE probe_t (k VARCHAR, ts TIMESTAMP, n INTEGER)")
            conn.execute("CREATE INDEX idx_probe_k ON probe_t(k)")
            conn.execute("CREATE INDEX idx_probe_ts ON probe_t(ts)")
            conn.execute("CREATE INDEX idx_probe_n ON probe_t(n)")

            targets = {t.index_name: t for t in index_probe_targets(conn)}

            assert targets["idx_probe_k"].table == "probe_t"
            assert targets["idx_probe_k"].columns == ("k",)
            assert "idx_probe_ts" not in targets
            assert "idx_probe_n" not in targets
            # schema.sql indexes: VARCHAR ones in, TIMESTAMP/BOOLEAN ones out.
            assert "idx_session_state_git_repo" in targets
            assert "idx_session_state_started" not in targets
            assert "idx_message_state_has_thinking" not in targets
            assert targets["idx_usage_events_source_sid"].columns == ("source", "source_session_id")
        finally:
            conn.close()


class _FakeClock:
    """Stands in for the `time` module inside `recall.db.maintenance`."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


class _SlowQueryConnection:
    """Advance the fake clock per query and count queries started past the deadline."""

    def __init__(
        self,
        inner: duckdb.DuckDBPyConnection,
        clock: _FakeClock,
        *,
        seconds_per_query: float,
        deadline: float,
    ) -> None:
        self._inner = inner
        self._clock = clock
        self._seconds_per_query = seconds_per_query
        self._deadline = deadline
        self.queries = 0
        self.queries_started_after_deadline = 0

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        if self._clock.now >= self._deadline:
            self.queries_started_after_deadline += 1
        self.queries += 1
        self._clock.now += self._seconds_per_query
        return self._inner.execute(sql, params) if params is not None else self._inner.execute(sql)


class _DivergingConnection:
    """Wrap a real connection; make the forced-sequential arm disagree for one key.

    The probe's two arms are `WHERE col = ?` (index-eligible) and
    `WHERE (col || '') = ?` (sequential). Real DuckDB keeps them equal; this
    wrapper adds `extra` rows to the sequential answer for `diverged_key` on
    `diverged_column`, which is exactly what the incident's file reported
    (1790 via index vs 1797 via full scan for one git_repo).
    """

    def __init__(
        self,
        inner: duckdb.DuckDBPyConnection,
        *,
        diverged_column: str,
        diverged_key: str,
        extra: int,
    ) -> None:
        self._inner = inner
        self._column = diverged_column
        self._key = diverged_key
        self._extra = extra

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        cursor = (
            self._inner.execute(sql, params) if params is not None else self._inner.execute(sql)
        )
        sequential_arm = f"(\"{self._column}\" || '') = ?" in sql
        if sequential_arm and params is not None and list(params) == [self._key]:
            row = cursor.fetchone()
            assert row is not None
            return _FixedResult((int(row[0]) + self._extra,))
        return cursor


class _FixedResult:
    def __init__(self, row: tuple[Any, ...]) -> None:
        self._row = row

    def fetchone(self) -> tuple[Any, ...]:
        return self._row

    def fetchall(self) -> list[tuple[Any, ...]]:
        return [self._row]


class TestProbeIndexDivergence:
    def test_healthy_database_reports_no_divergence(self, tmp_path: Path) -> None:
        conn = _fresh_db(tmp_path)
        try:
            _seed_sessions(conn, 9)

            report = probe_index_divergence(conn, sample=2, budget_seconds=30.0)

            assert isinstance(report, IndexDivergenceReport)
            assert report.diverged == ()
            assert report.diverged_count == 0
            assert report.complete is True
            assert report.indexes_probed == len(index_probe_targets(conn))
            # session_state.cwd has 3 distinct keys, 3 rows each: both samples checked.
            assert report.samples_checked >= 2
            assert isinstance(report.checked_at, datetime)
        finally:
            conn.close()

    def test_reports_key_whose_arms_disagree(self, tmp_path: Path) -> None:
        conn = _fresh_db(tmp_path)
        try:
            _seed_sessions(conn, 9)
            diverging = _DivergingConnection(
                conn, diverged_column="git_repo", diverged_key="/repo1", extra=7
            )

            report = probe_index_divergence(cast(Any, diverging), sample=2, budget_seconds=30.0)

            assert report.diverged_count == 1
            (key,) = report.diverged
            assert key.table == "session_state"
            assert key.column == "git_repo"
            assert key.key == "/repo1"
            assert key.index_count == 4
            assert key.full_count == 11
        finally:
            conn.close()

    def test_sample_above_index_scan_bound_is_unverifiable_not_healthy(
        self, tmp_path: Path
    ) -> None:
        conn = _fresh_db(tmp_path)
        try:
            bound = int(_scalar(conn, "SELECT current_setting('index_scan_max_count')"))
            conn.execute(
                """
                INSERT INTO session_state (session_id, cwd, git_repo, file_mtime, file_size)
                SELECT 'h' || i, '/hot', '/hot-repo', 1.0, 10 FROM range(?) t(i)
                """,
                [bound + 50],
            )

            report = probe_index_divergence(conn, sample=1, budget_seconds=30.0)

            # Both session_state VARCHAR indexes sample the same hot key; neither can be
            # index-scanned, so neither counts as a checked-and-healthy sample.
            assert report.samples_unverifiable >= 2
            assert report.diverged == ()
        finally:
            conn.close()

    def test_no_query_is_issued_after_the_budget_elapses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-018: the deadline gates every query, discovery included, so the
        daemon's exclusive startup gate cannot run past the budget."""
        conn = _fresh_db(tmp_path)
        try:
            _seed_sessions(conn, 9)
            clock = _FakeClock()
            monkeypatch.setattr(maintenance, "time", clock)
            budget = 1.0
            slow = _SlowQueryConnection(conn, clock, seconds_per_query=0.4, deadline=budget)

            report = probe_index_divergence(cast(Any, slow), sample=2, budget_seconds=budget)

            assert slow.queries > 0, "the probe ran until the budget elapsed"
            assert slow.queries_started_after_deadline == 0
            assert report.complete is False
        finally:
            conn.close()

    def test_budget_exhausted_stops_before_first_query(self, tmp_path: Path) -> None:
        conn = _fresh_db(tmp_path)
        try:
            _seed_sessions(conn, 3)

            report = probe_index_divergence(conn, sample=2, budget_seconds=0.0)

            assert report.complete is False
            assert report.indexes_probed == 0
            assert report.samples_checked == 0
        finally:
            conn.close()

    def test_probe_runs_on_a_read_only_connection(self, tmp_path: Path) -> None:
        conn = _fresh_db(tmp_path)
        _seed_sessions(conn, 4)
        conn.close()

        readonly = duckdb.connect(str(tmp_path / "maint.duckdb"), read_only=True)
        try:
            report = probe_index_divergence(readonly, sample=1, budget_seconds=30.0)
            assert report.complete is True
        finally:
            readonly.close()

    @pytest.mark.parametrize("sample", [0, -1])
    def test_rejects_non_positive_sample(self, tmp_path: Path, sample: int) -> None:
        conn = _fresh_db(tmp_path)
        try:
            with pytest.raises(ValueError, match="sample"):
                probe_index_divergence(conn, sample=sample)
        finally:
            conn.close()
