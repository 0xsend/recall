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
