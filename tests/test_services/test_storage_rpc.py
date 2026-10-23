"""The maintenance operation exposes preview, admission and durable completion."""

from __future__ import annotations

import asyncio
from pathlib import Path

import duckdb
import pytest
from lane_harness import lane_config
from recall.core.rpc_types import RpcError
from recall.services.rpc_server import RpcServer


def test_storage_rpc_preview_conflict_apply_retry_and_completion(tmp_path: Path) -> None:
    cfg = lane_config(tmp_path)
    cfg.data_dir.mkdir(parents=True)
    with duckdb.connect(str(cfg.db_path)) as conn:
        conn.execute("CREATE TABLE owned_marker AS SELECT 3 AS value")
    server = RpcServer(config=cfg)

    async def scenario() -> None:
        try:
            preview = await server._handle_migrate_storage({"dry_run": True}, None)
            assert preview["status"] == "planned"
            assert preview["accepted"] is False
            assert preview["migration"]["storage_version"] == "v1.0.0+"
            assert not (cfg.data_dir / "snapshots").exists()
            await server._writer_call(
                lambda: server._get_conn().execute(
                    "UPDATE runtime_state SET index_migration_version=0 WHERE singleton"
                ),
                "invalidate preview",
            )
            with pytest.raises(RpcError) as conflict:
                await server._handle_migrate_storage({"plan_id": preview["plan_id"]}, None)
            assert "plan changed" in conflict.value.message
            fresh = await server._handle_migrate_storage({"dry_run": True}, None)
            assert fresh["plan_id"] != preview["plan_id"]
            assert fresh["operation_id"] == preview["operation_id"]
            accepted = await server._handle_migrate_storage({"plan_id": fresh["plan_id"]}, None)
            assert accepted["accepted"] is True
            assert accepted["status"] == "accepted"
            duplicate = await server._handle_migrate_storage({}, None)
            assert duplicate["operation_id"] == accepted["operation_id"]
            assert server._storage_maintenance_task is not None
            await asyncio.wait_for(asyncio.shield(server._storage_maintenance_task), 5)
            done = await server._handle_migrate_storage({"dry_run": True}, None)
            assert done["status"] == "succeeded"
            status = await server._handle_daemon_status({}, None)
            assert status["storage_maintenance"]["phase"] == "succeeded"
            assert status["storage_maintenance"]["snapshot"] is False
            assert status["storage_maintenance"]["error"] is None
            assert status["write_admission"]["allowed"] is True
            assert done["migration"]["storage_version"] == "v1.2.0+"
            assert done["migration"]["applied_version"] == 0
            again = await server._handle_migrate_storage({}, None)
            assert again["accepted"] is False
            assert again["operation_id"] == accepted["operation_id"]
            assert len(list((cfg.data_dir / "snapshots").iterdir())) == 1
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_storage_rpc_reports_conflict_while_queued_without_mutating_job(tmp_path: Path) -> None:
    cfg = lane_config(tmp_path)
    cfg.data_dir.mkdir(parents=True)
    with duckdb.connect(str(cfg.db_path)) as conn:
        conn.execute("CREATE TABLE owned_marker AS SELECT 3 AS value")
    server = RpcServer(config=cfg)

    async def scenario() -> None:
        try:
            async with server._write_lock:
                accepted = await server._handle_migrate_storage({}, None)
                assert accepted["accepted"] is True
                server._get_conn().execute(
                    "UPDATE runtime_state SET index_migration_version=0 WHERE singleton"
                )
            assert server._storage_maintenance_task is not None
            await asyncio.wait_for(asyncio.shield(server._storage_maintenance_task), 5)
            failed = await server._handle_migrate_storage({"dry_run": True}, None)
            assert failed["status"] == "failed"
            assert "plan changed before execution" in failed["error"]
            assert failed["migration"]["phase"] == "idle"
            assert not (cfg.data_dir / "snapshots").exists()
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_failed_maintenance_status_is_current_and_includes_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services import index_migration

    cfg = lane_config(tmp_path)
    cfg.data_dir.mkdir(parents=True)
    with duckdb.connect(str(cfg.db_path)) as conn:
        conn.execute("CREATE TABLE owned_marker AS SELECT 3 AS value")
    server = RpcServer(config=cfg)

    def fail(_config) -> None:
        raise OSError("scratch capacity exhausted")

    monkeypatch.setattr(index_migration, "perform_storage_transition", fail)

    async def scenario() -> None:
        try:
            await server._handle_migrate_storage({}, None)
            assert server._storage_maintenance_task is not None
            await asyncio.wait_for(server._storage_maintenance_task, 5)
            status = await server._handle_daemon_status({}, None)
            assert status["storage_maintenance"]["phase"] == "failed"
            assert status["storage_maintenance"]["snapshot"] is False
            assert "scratch capacity exhausted" in status["storage_maintenance"]["error"]
            assert status["write_admission"]["allowed"] is True
        finally:
            await server.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "method,params",
    [
        ("_handle_set_installed_scheduler", {"scheduler": "systemd"}),
        ("_handle_live_mark", {"session": "blocked-session", "pid": 1}),
    ],
)
def test_rpc_metadata_writers_obey_storage_admission(
    tmp_path: Path, method: str, params: dict
) -> None:
    cfg = lane_config(tmp_path)
    server = RpcServer(config=cfg)

    async def scenario() -> None:
        release = asyncio.Event()

        async def maintenance() -> None:
            await release.wait()

        task = asyncio.create_task(maintenance())
        server._storage_maintenance_task = task
        try:
            with pytest.raises(RpcError) as denied:
                await asyncio.wait_for(getattr(server, method)(params, None), 2)
            assert denied.value.data["reason"] == "storage_maintenance"
            release.set()
            await task
            await getattr(server, method)(params, None)
            if method == "_handle_live_mark":
                assert server._get_conn().execute("SELECT count(*) FROM live_marks").fetchone() == (
                    1,
                )
            else:
                assert server._get_conn().execute(
                    "SELECT installed_scheduler FROM runtime_state"
                ).fetchone() == ("systemd",)
        finally:
            release.set()
            await task
            await server.stop()

    asyncio.run(scenario())
