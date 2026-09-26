from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from recall.core.config import FtsConfig
from recall.db import queries


def _create_fts_test_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE message_state ("
        "message_id VARCHAR, content VARCHAR, thinking VARCHAR, "
        "context_text VARCHAR DEFAULT '', fts_content VARCHAR DEFAULT '', "
        "fts_thinking VARCHAR DEFAULT ''"
        ")"
    )
    conn.execute("CREATE TABLE tool_calls (id VARCHAR, bash_command VARCHAR)")
    conn.execute(
        "INSERT INTO message_state "
        "(message_id, content, thinking) VALUES "
        "('msg-1', 'alpha document', 'alpha thought'), "
        "('msg-2', 'beta document', 'beta thought')"
    )
    conn.execute("INSERT INTO tool_calls VALUES ('tool-1', 'echo alpha'), ('tool-2', 'echo beta')")
    return conn


def _read_fts_settings(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    threads_row = conn.execute("SELECT current_setting('threads')").fetchone()
    insertion_row = conn.execute("SELECT current_setting('preserve_insertion_order')").fetchone()
    assert threads_row is not None
    assert insertion_row is not None
    return {
        "threads": threads_row[0],
        "preserve_insertion_order": insertion_row[0],
    }


def _fts_schema_count(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.schemata
        WHERE schema_name LIKE 'fts\\_main\\_%' ESCAPE '\\'
        """
    ).fetchone()
    assert row is not None
    return int(row[0])


def _capture_settings_during_restore(
    monkeypatch: pytest.MonkeyPatch, conn: duckdb.DuckDBPyConnection
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    original_restore = getattr(queries, "_restore_fts_session_settings", None)

    def capture_restore(proxy_conn: duckdb.DuckDBPyConnection, prior: dict[str, Any]) -> None:
        captured.update(_read_fts_settings(conn))
        if original_restore is not None:
            original_restore(proxy_conn, prior)

    monkeypatch.setattr(queries, "_restore_fts_session_settings", capture_restore, raising=False)
    return captured


class _ExecuteProxy:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        fail_on: str | None = None,
        error: duckdb.OutOfMemoryException | None = None,
    ) -> None:
        self._conn = conn
        self._fail_on = fail_on
        self._error = error

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        if self._fail_on is not None and self._fail_on in sql:
            if self._error is None:
                raise duckdb.OutOfMemoryException("synthetic fts oom")
            raise self._error
        return self._conn.execute(sql, *args, **kwargs)


class _AbortRestoreOnceProxy:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self.rollback_calls = 0
        self._abort_next_set = True

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        normalized = sql.strip()
        if normalized.startswith("SET ") and self._abort_next_set:
            self._abort_next_set = False
            raise duckdb.TransactionException("Current transaction is aborted (please ROLLBACK)")
        if normalized == "ROLLBACK":
            self.rollback_calls += 1
        return self._conn.execute(sql, *args, **kwargs)


class _PersistentRestoreFailureProxy:
    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self.rollback_calls = 0
        self.set_attempts = 0

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        normalized = sql.strip()
        if normalized.startswith("SET "):
            self.set_attempts += 1
            if self.set_attempts == 1:
                raise duckdb.TransactionException(
                    "Current transaction is aborted (please ROLLBACK)"
                )
            raise duckdb.TransactionException("synthetic persistent restore failure")
        if normalized == "ROLLBACK":
            self.rollback_calls += 1
        return self._conn.execute(sql, *args, **kwargs)


class _CreateFtsAbortProxy:
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        persistent_restore_failure: bool = False,
    ) -> None:
        self._conn = conn
        self._persistent_restore_failure = persistent_restore_failure
        self._aborted_after_oom = False
        self._pragma_failed = False
        self.rollback_calls = 0

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> duckdb.DuckDBPyConnection:
        normalized = sql.strip()
        if "PRAGMA create_fts_index" in normalized and not self._pragma_failed:
            self._pragma_failed = True
            self._aborted_after_oom = True
            raise duckdb.OutOfMemoryException("synthetic fts oom")
        if normalized.startswith("SET ") and self._aborted_after_oom:
            raise duckdb.TransactionException("Current transaction is aborted (please ROLLBACK)")
        if (
            normalized.startswith("SET ")
            and self._persistent_restore_failure
            and self.rollback_calls > 0
        ):
            raise duckdb.TransactionException("synthetic persistent restore failure")
        if normalized == "ROLLBACK":
            self.rollback_calls += 1
            self._aborted_after_oom = False
        return self._conn.execute(sql, *args, **kwargs)


def _exception_chain_contains(err: BaseException, expected: type[BaseException]) -> bool:
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        if isinstance(current, expected):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def test_create_fts_indexes_skips_duckdb_rebuild_for_sidecar_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _create_fts_test_conn()
    calls = 0
    real_load_fts_extension = queries.load_fts_extension

    def spy_load_fts_extension(spy_conn: duckdb.DuckDBPyConnection) -> None:
        nonlocal calls
        calls += 1
        real_load_fts_extension(spy_conn)

    monkeypatch.setattr(queries, "load_fts_extension", spy_load_fts_extension)

    queries.create_fts_indexes(conn, FtsConfig(backend="sqlite_sidecar"))

    assert calls == 0
    assert _fts_schema_count(conn) == 0


@pytest.mark.parametrize(
    ("cpu_count", "expected_threads"),
    [(2, 2), (4, 2), (8, 4), (16, 8), (64, 8), (None, 2)],
)
def test_applies_session_settings_cpu_param(
    monkeypatch: pytest.MonkeyPatch, cpu_count: int | None, expected_threads: int
) -> None:
    conn = _create_fts_test_conn()
    monkeypatch.setattr(queries.os, "cpu_count", lambda: cpu_count)
    captured = _capture_settings_during_restore(monkeypatch, conn)

    queries.create_fts_indexes(conn, FtsConfig(backend="duckdb"))

    assert captured["preserve_insertion_order"] is False
    assert captured["threads"] == expected_threads


@pytest.mark.parametrize("fail_on", ["PRAGMA create_fts_index", "UPDATE message_state"])
def test_oom_wrapped(fail_on: str) -> None:
    conn = _create_fts_test_conn()
    original = duckdb.OutOfMemoryException("synthetic fts oom")
    proxy = _ExecuteProxy(conn, fail_on=fail_on, error=original)

    with pytest.raises(queries.FtsRebuildOutOfMemoryError) as exc_info:
        queries.create_fts_indexes(
            cast(duckdb.DuckDBPyConnection, proxy), FtsConfig(backend="duckdb")
        )

    assert exc_info.value.__cause__ is original


def test_restores_settings_on_success() -> None:
    conn = _create_fts_test_conn()
    conn.execute("SET threads = 12")
    conn.execute("SET preserve_insertion_order = true")

    queries.create_fts_indexes(conn, FtsConfig(backend="duckdb"))

    assert _read_fts_settings(conn) == {
        "threads": 12,
        "preserve_insertion_order": True,
    }


def test_restores_settings_on_exception() -> None:
    conn = _create_fts_test_conn()
    conn.execute("SET threads = 12")
    conn.execute("SET preserve_insertion_order = true")
    proxy = _ExecuteProxy(conn, fail_on="PRAGMA create_fts_index")

    with pytest.raises(queries.FtsRebuildOutOfMemoryError):
        queries.create_fts_indexes(
            cast(duckdb.DuckDBPyConnection, proxy), FtsConfig(backend="duckdb")
        )

    assert _read_fts_settings(conn) == {
        "threads": 12,
        "preserve_insertion_order": True,
    }


def test_restore_after_aborted_txn_via_rollback() -> None:
    conn = _create_fts_test_conn()
    conn.execute("SET threads = 12")
    conn.execute("SET preserve_insertion_order = true")
    prior = _read_fts_settings(conn)
    conn.execute("SET threads = 2")
    conn.execute("SET preserve_insertion_order = false")
    proxy = _AbortRestoreOnceProxy(conn)

    queries._restore_fts_session_settings(cast(duckdb.DuckDBPyConnection, proxy), prior)

    assert _read_fts_settings(conn) == prior
    assert proxy.rollback_calls == 1
    assert conn.execute("SELECT 1").fetchone() == (1,)


def test_restore_raises_on_persistent_failure() -> None:
    conn = _create_fts_test_conn()
    prior = _read_fts_settings(conn)
    proxy = _PersistentRestoreFailureProxy(conn)

    with pytest.raises(queries.FtsSettingsRestoreError) as exc_info:
        queries._restore_fts_session_settings(cast(duckdb.DuckDBPyConnection, proxy), prior)

    assert proxy.rollback_calls == 1
    assert isinstance(exc_info.value.__cause__, duckdb.TransactionException)
    assert "synthetic persistent restore failure" in str(exc_info.value)
    assert "prior=" in str(exc_info.value)


def test_create_fts_indexes_propagates_settings_restore_error_when_unrecoverable() -> None:
    conn = _create_fts_test_conn()
    proxy = _CreateFtsAbortProxy(conn, persistent_restore_failure=True)

    with pytest.raises(queries.FtsSettingsRestoreError) as exc_info:
        queries.create_fts_indexes(
            cast(duckdb.DuckDBPyConnection, proxy), FtsConfig(backend="duckdb")
        )

    assert _exception_chain_contains(exc_info.value, queries.FtsRebuildOutOfMemoryError)
    assert isinstance(exc_info.value.__cause__, duckdb.TransactionException)


def test_inv_resil_003_holds_after_abort() -> None:
    conn = _create_fts_test_conn()
    conn.execute("SET threads = 12")
    conn.execute("SET preserve_insertion_order = true")
    prior = _read_fts_settings(conn)
    proxy = _CreateFtsAbortProxy(conn)

    with pytest.raises(queries.FtsRebuildOutOfMemoryError):
        queries.create_fts_indexes(
            cast(duckdb.DuckDBPyConnection, proxy), FtsConfig(backend="duckdb")
        )

    assert _read_fts_settings(conn) == prior
    assert proxy.rollback_calls == 1
    assert conn.execute("SELECT 1").fetchone() == (1,)


def test_existing_happy_path_still_passes() -> None:
    conn = _create_fts_test_conn()

    queries.create_fts_indexes(conn, FtsConfig(backend="duckdb"))
    queries.create_fts_indexes(conn, FtsConfig(backend="duckdb"))

    message_terms = conn.execute("SELECT COUNT(*) FROM fts_main_message_state.terms").fetchone()
    tool_terms = conn.execute("SELECT COUNT(*) FROM fts_main_tool_calls.terms").fetchone()
    assert message_terms is not None
    assert tool_terms is not None
    assert message_terms[0] > 0
    assert tool_terms[0] > 0


def test_create_fts_indexes_checkpoints_unreplayable_shadow_schema_drop(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "recall.duckdb"
    src_path = Path(__file__).resolve().parents[2] / "packages" / "recall" / "src"
    script = textwrap.dedent(
        f"""
        import os
        import sys

        sys.path.insert(0, {str(src_path)!r})

        import duckdb

        from recall.core.config import FtsConfig
        from recall.db.queries import create_fts_indexes

        conn = duckdb.connect({str(db_path)!r})
        conn.execute(
            "CREATE TABLE message_state ("
            "message_id VARCHAR, content VARCHAR, thinking VARCHAR, "
            "context_text VARCHAR DEFAULT '', fts_content VARCHAR DEFAULT '', "
            "fts_thinking VARCHAR DEFAULT ''"
            ")"
        )
        conn.execute("CREATE TABLE tool_calls (id VARCHAR, bash_command VARCHAR)")
        conn.execute(
            "INSERT INTO message_state "
            "(message_id, content, thinking) "
            "SELECT format('msg-{{}}', i), 'document ' || i, 'thought ' || i "
            "FROM range(1000) t(i)"
        )
        conn.execute(
            "INSERT INTO tool_calls "
            "SELECT format('tool-{{}}', i), 'echo ' || i "
            "FROM range(10) t(i)"
        )

        create_fts_indexes(conn, FtsConfig(backend="duckdb"))
        # Persist the prior schema independently of the production guard. If
        # both builds stay in WAL, replay succeeds and misses the 1.5.5 bug.
        conn.execute("CHECKPOINT")
        create_fts_indexes(conn, FtsConfig(backend="duckdb"))
        # Keep a benign write in WAL so the parent actually exercises replay.
        conn.execute("CREATE TABLE wal_replay_marker (id INTEGER)")
        conn.execute("INSERT INTO wal_replay_marker VALUES (1)")
        os._exit(0)
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)

    wal_path = Path(f"{db_path}.wal")
    assert wal_path.exists()
    assert wal_path.stat().st_size > 0

    conn = duckdb.connect(str(db_path))
    try:
        row = conn.execute("SELECT COUNT(*) FROM fts_main_message_state.terms").fetchone()
    finally:
        conn.close()

    assert row is not None
    terms_count = row[0]
    assert terms_count > 0
