"""A migration transaction whose commit and rollback both fail never ends as a normal return.

The runner reports a soft failure for an ordinary migration error and keeps the
migration pending. When the rollback also fails, the transaction may still be
open on the connection, so the run has to stop with both errors instead of
reporting a soft failure that lets later code commit the half-applied change.
The injected failures stand in for a storage error on COMMIT followed by one on
ROLLBACK; the engine cannot be made to fail that way deterministically.
"""

from __future__ import annotations

from typing import Any, cast

import duckdb
import pytest
from recall import _native
from recall.db import migrations
from recall.db.migrations import get_migrations, run_pending_migrations
from recall.db.schema import get_schema_version

COMMIT_ERROR = "injected commit failure"
ROLLBACK_ERROR = "injected rollback failure"


class _FailingTransactionEnd:
    """Forwarding proxy that fails the named transaction-ending statements."""

    def __init__(self, inner: duckdb.DuckDBPyConnection, failing: dict[str, str]) -> None:
        self._inner = inner
        self._failing = failing

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        statement = sql.strip().upper()
        if statement in self._failing:
            raise duckdb.IOException(self._failing[statement])
        return self._inner.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _connection_at(version: int, *tables: str) -> duckdb.DuckDBPyConnection:
    conn = _native.duckdb_connect(":memory:")
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version VALUES (?)", [version])
    for table in tables:
        conn.execute(table)
    return conn


SESSIONS = (
    "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, "
    "source_path TEXT UNIQUE NOT NULL, source_session_id TEXT)"
)
SOURCE_FILES = (
    "CREATE TABLE source_files (source TEXT, source_path TEXT, committed_generation BIGINT, "
    "desired_generation BIGINT, next_retry_at DOUBLE)"
)
USAGE_CURSORS = "CREATE TABLE usage_log_cursors (path TEXT PRIMARY KEY)"

# One migration per body shape that opens its own transaction: re-raising after
# the rollback (0026, 0035, 0036) and reporting False after it (0030, 0031, 0032).
TRANSACTION_MIGRATIONS = [
    pytest.param("0026_pending_admission", 25, SOURCE_FILES, id="0026-reraises"),
    pytest.param("0030_drop_catalog_pending_index", 29, SOURCE_FILES, id="0030-reports-false"),
    pytest.param("0031_normalization_checkpoint", 30, SOURCE_FILES, id="0031-reports-false"),
    pytest.param("0032_usage_cursor_identity", 31, USAGE_CURSORS, id="0032-reports-false"),
    pytest.param("0035_source_path_identity", 32, SESSIONS, id="0035-reraises"),
    pytest.param("0036_usage_file_identity", 35, USAGE_CURSORS, id="0036-reraises"),
]


def _only(migration_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    chosen = [m for m in get_migrations() if m.id == migration_id]
    assert len(chosen) == 1
    monkeypatch.setattr(migrations, "get_migrations", lambda: chosen)


@pytest.mark.parametrize(("migration_id", "version", "table"), TRANSACTION_MIGRATIONS)
def test_failed_commit_and_failed_rollback_stop_the_run_with_both_errors(
    migration_id: str, version: int, table: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _only(migration_id, monkeypatch)
    inner = _connection_at(version, table)
    conn = _FailingTransactionEnd(inner, {"COMMIT": COMMIT_ERROR, "ROLLBACK": ROLLBACK_ERROR})

    with pytest.raises(BaseExceptionGroup) as raised:
        run_pending_migrations(cast("duckdb.DuckDBPyConnection", conn), version)

    assert sorted(str(error) for error in raised.value.exceptions) == [
        COMMIT_ERROR,
        ROLLBACK_ERROR,
    ]


@pytest.mark.parametrize(("migration_id", "version", "table"), TRANSACTION_MIGRATIONS)
def test_failed_commit_with_a_working_rollback_keeps_the_migration_pending(
    migration_id: str, version: int, table: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _only(migration_id, monkeypatch)
    inner = _connection_at(version, table)
    conn = _FailingTransactionEnd(inner, {"COMMIT": COMMIT_ERROR})

    run_pending_migrations(cast("duckdb.DuckDBPyConnection", conn), version)

    assert get_schema_version(inner) == version
    assert inner.execute("SELECT count(*) FROM schema_migrations").fetchone() == (0,)
    inner.execute("BEGIN")
    inner.execute("COMMIT")
