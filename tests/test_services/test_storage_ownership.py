"""Real handles expose maintenance races that call-sequence mocks cannot."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from lane_harness import lane_config
from recall.services import index_migration
from recall.services.coordinator import set_paused
from recall.services.rpc_server import RpcServer


@pytest.fixture
def server(tmp_path: Path) -> Iterator[RpcServer]:
    cfg = lane_config(tmp_path)
    cfg.data_dir.mkdir(parents=True)
    with duckdb.connect(str(cfg.db_path)) as conn:
        conn.execute("CREATE TABLE owned_marker AS SELECT 17 AS value")
    instance = RpcServer(config=cfg)
    instance._get_conn()
    yield instance
    if instance._conn is not None:
        instance._conn.close()


def test_storage_cancellation_keeps_handle_and_writer_ownership(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    real_transition = index_migration.perform_storage_transition
    retired = server._conn

    def held_transition(config):
        assert server._conn is None
        with pytest.raises(duckdb.ConnectionException):
            retired.execute("SELECT 1")
        entered.set()
        assert release.wait(timeout=5)
        real_transition(config)

    monkeypatch.setattr(index_migration, "perform_storage_transition", held_transition)

    async def scenario() -> None:
        maintenance = asyncio.create_task(server._run_storage_maintenance())
        assert await asyncio.to_thread(entered.wait, 3)
        maintenance.cancel()
        writer_requested = asyncio.Event()

        async def write() -> None:
            writer_requested.set()
            await server._writer_call(
                lambda: server._get_conn().execute("UPDATE owned_marker SET value=23"),
                "competing writer",
            )

        writer = asyncio.create_task(write())
        await writer_requested.wait()
        assert not writer.done()
        assert not maintenance.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await maintenance
        await writer
        status = await server._run_readonly(index_migration.migration_status)
        assert status.storage_version == "v1.2.0+"
        assert status.phase == "idle"
        assert await server._run_readonly(
            lambda conn: conn.execute("SELECT value FROM owned_marker").fetchone()
        ) == (23,)

    asyncio.run(scenario())


def test_storage_failure_reopens_and_paused_restart_resumes_original_backup(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(_config):
        raise RuntimeError("owned interruption before format commit")

    async def scenario() -> None:
        server._conn.execute("UPDATE runtime_state SET index_migration_version=0 WHERE singleton")
        set_paused(server._config, True)
        with monkeypatch.context() as patch:
            patch.setattr(index_migration, "perform_storage_transition", interrupted)
            with pytest.raises(RuntimeError, match="owned interruption"):
                await server._run_storage_maintenance()
        failed = await server._run_readonly(index_migration.migration_status)
        assert failed.phase == "failed"
        assert failed.backup_path is not None
        assert failed.storage_version == "v1.0.0+"
        assert failed.applied_version == 0
        await server._maybe_begin_index_migration()
        resumed = await server._run_readonly(index_migration.migration_status)
        assert resumed.phase == "idle"
        assert resumed.storage_version == "v1.2.0+"
        assert resumed.backup_path == failed.backup_path
        assert resumed.started_at == failed.started_at
        assert resumed.applied_version == 0
        assert resumed.captured == 0

    asyncio.run(scenario())


def test_storage_waits_for_active_read_cursor_before_replacing_handle(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader_entered = threading.Event()
    release_reader = threading.Event()
    finish_started = threading.Event()
    real_finish = server._finish_storage_maintenance

    def observed_finish(config) -> None:
        finish_started.set()
        real_finish(config)

    def held_reader(conn: duckdb.DuckDBPyConnection):
        before = conn.execute("SELECT value FROM owned_marker").fetchone()
        reader_entered.set()
        assert release_reader.wait(timeout=5)
        after = conn.execute("SELECT value FROM owned_marker").fetchone()
        return before, after

    monkeypatch.setattr(server, "_finish_storage_maintenance", observed_finish)

    async def scenario() -> None:
        reader = asyncio.create_task(server._run_readonly(held_reader))
        assert await asyncio.to_thread(reader_entered.wait, 3)
        maintenance = asyncio.create_task(server._run_storage_maintenance())
        assert await asyncio.to_thread(finish_started.wait, 3)
        assert not maintenance.done()
        release_reader.set()
        assert await reader == ((17,), (17,))
        await maintenance
        assert await server._run_readonly(
            lambda conn: conn.execute("SELECT value FROM owned_marker").fetchone()
        ) == (17,)

    asyncio.run(scenario())
