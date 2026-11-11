"""Worker observation is independent of the caller's error/cancellation policy."""

import asyncio
import threading
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.services.rpc_server import RpcServer
from recall.services.self_repair import read_failure_marker


def server_at(path: Path) -> RpcServer:
    return RpcServer(
        config=replace(
            AppConfig.load(),
            data_dir=path,
            db_path=path / "recall.duckdb",
            lock_path=path / "recall.lock",
        )
    )


def test_wrapped_writer_origin_is_persisted_before_policy(tmp_path: Path) -> None:
    server = server_at(tmp_path)
    original = duckdb.FatalException("Failed to delete all rows from index")

    def operation() -> None:
        raise ValueError("ordinary request wrapper") from original

    async def scenario() -> None:
        try:
            with pytest.raises(ValueError, match="ordinary request wrapper"):
                await server._writer_call(operation, "source_commit")
            marker = read_failure_marker(tmp_path)
            assert marker is not None and marker.failure is not None
            assert marker.failure.exception_type == "FatalException"
            assert marker.failure.message == "Failed to delete all rows from index"
            assert marker.failure.site == "source_commit"
            assert server._shutdown_event.is_set()
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_leased_writer_retains_the_admitted_operation_site(tmp_path: Path) -> None:
    server = server_at(tmp_path)

    def leaf(_conn: duckdb.DuckDBPyConnection) -> None:
        raise duckdb.FatalException("Failed to delete all rows from index")

    async def scenario() -> None:
        try:
            with pytest.raises(duckdb.FatalException):
                await server._writer_call(
                    lambda: server._database.call("database_work", leaf), "source_commit"
                )
            marker = read_failure_marker(tmp_path)
            assert marker is not None and marker.failure is not None
            assert marker.failure.site == "source_commit"
            assert marker.failure.signature.startswith("source_commit:FatalException:")
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_cancelled_reader_still_persists_worker_fatal(tmp_path: Path) -> None:
    server = server_at(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def operation(conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("SELECT 1").fetchone()
        entered.set()
        assert release.wait(5), "test did not release read worker"
        raise duckdb.FatalException("Failed to delete all rows from index")

    async def scenario() -> None:
        task = asyncio.create_task(server._run_readonly(operation))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            # Shutdown must not disable observation of already admitted work.
            server._shutdown_event.set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            marker = read_failure_marker(tmp_path)
            assert marker is not None and marker.failure is not None
            assert marker.failure.exception_type == "FatalException"
            assert marker.failure.message == "Failed to delete all rows from index"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())


def test_incomplete_exception_graph_holds_without_fabricating_repair(tmp_path: Path) -> None:
    from recall.services.self_repair import marker_path, refusal_marker_path

    server = server_at(tmp_path)
    graph = ExceptionGroup(
        "too many independent failures", [ValueError(str(i)) for i in range(65_536)]
    )
    assert server._stop_on_fatal_db(graph, "incomplete_graph")
    assert server._shutdown_event.is_set()
    assert refusal_marker_path(tmp_path).exists()
    assert not marker_path(tmp_path).exists()


def test_known_rpc_error_cannot_hide_its_fatal_cause(tmp_path: Path) -> None:
    from typing import cast

    from recall.core.rpc_types import RpcError
    from recall.services.rpc_server import ClientConnection

    server = server_at(tmp_path)

    async def handler(params: dict, client: object) -> None:
        raise RpcError(code=-32603, message="wrapped reply") from duckdb.FatalException(
            "Failed to delete all rows from index"
        )

    class Writer:
        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

    server._methods["recall.failure"] = handler

    async def scenario() -> None:
        try:
            await server._process_request(
                b'{"jsonrpc":"2.0","method":"recall.failure","id":1}',
                cast(asyncio.StreamWriter, Writer()),
                cast(ClientConnection, None),
            )
            marker = read_failure_marker(tmp_path)
            assert marker is not None and marker.failure is not None
            assert marker.failure.exception_type == "FatalException"
            assert marker.failure.message == "Failed to delete all rows from index"
            assert server._shutdown_event.is_set()
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_failed_first_publication_does_not_promote_a_later_error_to_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    from recall.services import repair_events

    server = server_at(tmp_path)
    publish = repair_events.publish_exclusive
    primary_attempts = 0

    def fail_first_primary(path: Path, payload: bytes, **kwargs) -> bool:
        nonlocal primary_attempts
        if path.name == "daemon-failure.json":
            primary_attempts += 1
            if primary_attempts == 1:
                raise OSError(errno.EIO, "injected primary write failure")
        return publish(path, payload, **kwargs)

    monkeypatch.setattr(repair_events, "publish_exclusive", fail_first_primary)
    assert server._stop_on_fatal_db(
        duckdb.FatalException("Failed to delete all rows from index"), "first"
    )
    assert server._stop_on_fatal_db(
        duckdb.FatalException("database has been invalidated"), "cleanup"
    )
    assert primary_attempts == 1
    assert not (tmp_path / "daemon-failure.json").exists()
    events = tuple((tmp_path / "fatal-events").glob("*.json"))
    assert len(events) == 1
    assert b"cleanup" in events[0].read_bytes()
