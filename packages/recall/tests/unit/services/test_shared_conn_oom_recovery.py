from __future__ import annotations

import logging
from typing import Any

import duckdb
import pytest
from recall.db import FtsRebuildOutOfMemoryError
from recall.services.rpc_server import (
    RpcServer,
    _ConnectionLifecycleGate,
    _is_oom_error,
    _is_shared_conn_failure,
)


class _OomRecoveryConn:
    def __init__(self) -> None:
        self.closed = False
        self.execute_calls: list[str] = []

    def execute(self, sql: str) -> Any:
        self.execute_calls.append(sql)
        raise AssertionError("OOM recovery must not probe a poisoned DuckDB handle")

    def close(self) -> None:
        self.closed = True


def _server_with_conn(conn: object) -> RpcServer:
    server = object.__new__(RpcServer)
    server._conn = conn
    server._conn_lifecycle_gate = _ConnectionLifecycleGate()
    return server


def test_is_shared_conn_failure_treats_duckdb_oom_as_recoverable() -> None:
    err = duckdb.OutOfMemoryException("Out of Memory Error: synthetic")

    assert _is_shared_conn_failure(err) is True


def test_oom_recovery_closes_and_clears_shared_connection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conn = _OomRecoveryConn()
    server = _server_with_conn(conn)
    err = duckdb.OutOfMemoryException("Out of Memory Error: synthetic")

    caplog.set_level(logging.WARNING, logger="recall.rpc_server")

    server._recover_shared_conn(err, "embed_loop")

    assert server._conn is None
    assert conn.closed is True
    assert conn.execute_calls == []
    assert any(
        'shared connection recovery label="embed_loop"' in record.getMessage()
        and 'origin="OutOfMemoryException: Out of Memory Error: synthetic"' in record.getMessage()
        and 'path="close_clear"' in record.getMessage()
        for record in caplog.records
    )


def test_oom_shaped_generic_exception_closes_and_clears_shared_connection() -> None:
    conn = _OomRecoveryConn()
    server = _server_with_conn(conn)

    server._recover_shared_conn(Exception("Out of Memory Error: synthetic wrapper"), "index")

    assert server._conn is None
    assert conn.closed is True
    assert conn.execute_calls == []


def test_wrapped_fts_oom_closes_and_clears_shared_connection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conn = _OomRecoveryConn()
    server = _server_with_conn(conn)
    err = FtsRebuildOutOfMemoryError("FTS rebuild exhausted DuckDB memory_limit")

    caplog.set_level(logging.WARNING, logger="recall.rpc_server")

    assert _is_oom_error(err) is True
    assert _is_shared_conn_failure(err) is True

    server._recover_shared_conn(err, "fts_rebuild")

    assert server._conn is None
    assert conn.closed is True
    assert conn.execute_calls == []
    assert any(
        'shared connection recovery label="fts_rebuild"' in record.getMessage()
        and 'origin="FtsRebuildOutOfMemoryError: FTS rebuild exhausted DuckDB memory_limit"'
        in record.getMessage()
        and 'path="close_clear"' in record.getMessage()
        for record in caplog.records
    )
