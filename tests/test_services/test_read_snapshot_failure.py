"""A DuckDB execution failure inside a daemon read fails that read once (#231).

DuckDB aborts an open transaction when a statement fails during execution. A
read that swallowed such a failure and kept querying used to surface it as
"Current transaction is aborted" or as silently defaulted fields. These tests
inject real DuckDB execution failures, not Python-raised stand-ins, because
only a real failure aborts the transaction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.core.types import UNATTRIBUTED_HOST
from recall.db.read_snapshot import AbortedReadSnapshotError, read_snapshot
from recall.db.schema import ensure_schema
from recall.services.rpc_server import INTERNAL_ERROR, ClientConnection, RpcServer
from recall.services.search import SearchResult, _attach_hosts

# Large enough to exceed a few MiB of DuckDB memory, small enough to stay fast.
_MEMORY_HUNGRY_QUERY = "SELECT list(range(i)) AS l FROM range(20000) r(i) ORDER BY l"


class _Writer:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        return None


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RpcServer:
    for key in tuple(os.environ):
        if key.startswith("RECALL_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "config.toml"))
    return RpcServer(config=AppConfig.load())


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    ensure_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


async def _status(server: RpcServer) -> dict[str, Any]:
    writer = _Writer()
    request = json.dumps({"jsonrpc": "2.0", "method": "recall.daemon_status", "id": 1})
    await server._process_request(
        request.encode(), cast(asyncio.StreamWriter, writer), cast(ClientConnection, None)
    )
    return json.loads(b"".join(writer.chunks))


def _out_of_memory_once(real: Callable[..., Any]) -> Callable[..., Any]:
    """Exhaust DuckDB memory on the caller's cursor the first time, then read normally."""
    calls = 0

    def count(conn: duckdb.DuckDBPyConnection, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls > 1:
            return real(conn, *args, **kwargs)
        row = conn.execute("SELECT current_setting('memory_limit')").fetchone()
        assert row is not None
        conn.execute("SET memory_limit = '4MB'")
        try:
            conn.execute(_MEMORY_HUNGRY_QUERY).fetchall()
        finally:
            # A sibling cursor sits outside the aborted transaction.
            sibling = conn.cursor()
            try:
                sibling.execute(f"SET memory_limit = '{row[0]}'")
            finally:
                sibling.close()
        raise AssertionError("the memory-hungry query must exceed a 4MB limit")

    return count


def test_out_of_memory_in_status_pending_counts_fails_once_then_status_reads(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import recall.services.daemon as daemon

    async def scenario() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        healthy = await _status(server)
        monkeypatch.setattr(
            daemon, "count_pending_split", _out_of_memory_once(daemon.count_pending_split)
        )
        failed = await _status(server)
        recovered = await _status(server)
        await server.stop()
        return healthy, failed, recovered

    with caplog.at_level(logging.ERROR, logger="recall.rpc_server"):
        healthy, failed, recovered = asyncio.run(scenario())

    assert healthy["result"]["embed_pending_eligible"] == 0
    assert failed["error"]["code"] == INTERNAL_ERROR
    assert failed["error"]["message"].startswith("Out of Memory Error")
    assert "transaction is aborted" not in failed["error"]["message"].lower()
    assert "error" not in recovered
    assert recovered["result"]["embed_pending_eligible"] == 0
    assert [record.getMessage().split(":")[0] for record in caplog.records] == [
        "out of memory in recall.daemon_status"
    ]


def _fail_column_on_execution(
    conn: duckdb.DuckDBPyConnection, table: str, column: str, sql_type: str
) -> None:
    """Make every read of `table.column` bind and then fail while executing."""
    conn.execute(f"ALTER TABLE {table} RENAME TO {table}_rows")
    conn.execute(
        f"CREATE VIEW {table} AS SELECT * REPLACE "
        f"(CAST(error('injected {column} failure') AS {sql_type}) AS {column}) "
        f"FROM {table}_rows"
    )


def _restore_table(conn: duckdb.DuckDBPyConnection, table: str) -> None:
    conn.execute(f"DROP VIEW {table}")
    conn.execute(f"ALTER TABLE {table}_rows RENAME TO {table}")


def _hit(session_id: str) -> SearchResult:
    return SearchResult(
        kind="message",
        session_id=session_id,
        source="codex",
        source_path=None,
        score=1.0,
        message_id="msg-1",
        tool_call_id=None,
        role="assistant",
        content="hello",
        thinking=None,
        timestamp=None,
        tool_name=None,
        bash_command=None,
    )


@pytest.mark.parametrize(
    "column",
    [
        # The main runtime row read, which reported "unread" defaults.
        "last_attempted_at",
        # The fatal-memory read, which retried an older shape on the same
        # transaction and reported default fatal memory.
        "last_fatal_at",
    ],
)
def test_execution_failure_in_status_runtime_reads_fails_once_then_status_reads(
    server: RpcServer, column: str
) -> None:
    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        server._run_with_read_cursor_sync(
            lambda conn: _fail_column_on_execution(conn, "runtime_state", column, "TIMESTAMP")
        )
        failed = await _status(server)
        server._run_with_read_cursor_sync(lambda conn: _restore_table(conn, "runtime_state"))
        recovered = await _status(server)
        await server.stop()
        return failed, recovered

    failed, recovered = asyncio.run(scenario())

    assert failed["error"]["code"] == INTERNAL_ERROR
    assert f"injected {column} failure" in failed["error"]["message"]
    assert "error" not in recovered
    assert recovered["result"]["runtime_status"]["fatal_repeat_count"] == 0


def test_search_host_lookup_execution_failure_fails_the_search() -> None:
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE session_state (session_id TEXT, host TEXT)")
        conn.execute("INSERT INTO session_state VALUES ('session-1', 'ae-dev')")
        _fail_column_on_execution(conn, "session_state", "host", "TEXT")

        with pytest.raises(duckdb.InvalidInputException, match="injected host failure"):
            _attach_hosts(conn, [_hit("session-1")])


def test_search_host_lookup_on_a_schema_without_hosts_reports_unattributed() -> None:
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE session_state (session_id TEXT)")
        conn.execute("INSERT INTO session_state VALUES ('session-1')")

        hosts = [hit.host for hit in _attach_hosts(conn, [_hit("session-1")])]

    assert hosts == [UNATTRIBUTED_HOST]


def test_read_snapshot_names_a_read_that_swallowed_an_execution_failure(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    def swallowing_read(snapshot: duckdb.DuckDBPyConnection) -> str:
        with suppress(duckdb.InvalidInputException):
            snapshot.execute("SELECT error('injected')").fetchall()
        return "partial"

    with pytest.raises(AbortedReadSnapshotError):
        read_snapshot(conn, swallowing_read)

    assert conn.execute("SELECT 42").fetchone() == (42,)


def test_read_snapshot_commits_a_read_that_tolerated_a_schema_mismatch(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    def tolerant_read(snapshot: duckdb.DuckDBPyConnection) -> int:
        with suppress(duckdb.BinderException):
            snapshot.execute("SELECT missing_column FROM runtime_state").fetchall()
        row = snapshot.execute("SELECT count(*) FROM runtime_state").fetchone()
        assert row is not None
        return int(row[0])

    assert read_snapshot(conn, tolerant_read) == 1
    assert conn.execute("SELECT 42").fetchone() == (42,)
