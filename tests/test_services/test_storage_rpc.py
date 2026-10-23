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


@pytest.mark.parametrize("outcome", ["success", "failure", "shutdown"])
def test_overlapping_storage_applies_have_one_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from recall.services import index_migration

    cfg = lane_config(tmp_path)
    cfg.data_dir.mkdir(parents=True)
    with duckdb.connect(str(cfg.db_path)) as conn:
        conn.execute("CREATE TABLE owned_marker AS SELECT 3 AS value")
    server = RpcServer(config=cfg)

    if outcome == "failure":

        def unavailable(_config, *, semantic_capability=None) -> None:
            raise OSError("snapshot unavailable")

        monkeypatch.setattr(index_migration, "perform_storage_transition", unavailable)

    async def scenario() -> None:
        snapshot_started = asyncio.Event()
        duplicate_snapshot = asyncio.Event()
        release_snapshot = asyncio.Event()
        read_status = server._handle_daemon_status

        async def snapshot(params, client):
            if snapshot_started.is_set():
                duplicate_snapshot.set()
            snapshot_started.set()
            await release_snapshot.wait()
            return await read_status(params, client)

        monkeypatch.setattr(server, "_handle_daemon_status", snapshot)
        requests = []
        observer = None
        try:
            preview = await server._handle_migrate_storage({"dry_run": True}, None)
            async with server._write_lock:
                requests.append(asyncio.create_task(server._handle_migrate_storage({}, None)))
                await asyncio.wait_for(snapshot_started.wait(), 2)
                requests.append(asyncio.create_task(server._handle_migrate_storage({}, None)))
                observer = asyncio.create_task(duplicate_snapshot.wait())
                await asyncio.wait_for(
                    asyncio.wait({requests[1], observer}, return_when=asyncio.FIRST_COMPLETED),
                    2,
                )
                if outcome != "shutdown":
                    release_snapshot.set()
                # Accepted requests must not depend on the snapshot read
                # completing; shutdown owns and cancels that work too.
                accepted, duplicate = await asyncio.wait_for(asyncio.gather(*requests), 2)
                assert accepted["accepted"] is True
                assert duplicate["accepted"] is False
                assert duplicate["status"] == "running"
                assert duplicate["error"] is None
                assert duplicate["operation_id"] == preview["operation_id"]
            task = server._storage_maintenance_task
            assert task is not None
            if outcome == "shutdown":
                await asyncio.wait_for(server.stop(), 2)
                assert task.cancelled()
                assert not (cfg.data_dir / "snapshots").exists()
            else:
                await asyncio.wait_for(task, 5)
                status = await server._handle_daemon_status({}, None)
                progress = status["storage_maintenance"]
                assert progress["phase"] == ("succeeded" if outcome == "success" else "failed")
                assert progress["snapshot"] is False
                assert progress["operation_id"] == preview["operation_id"]
                assert status["write_admission"]["allowed"] is True
                if outcome == "success":
                    assert progress["error"] is None
                    done = await server._handle_migrate_storage({"dry_run": True}, None)
                    assert done["migration"]["storage_version"] == "v1.2.0+"
                else:
                    assert "snapshot unavailable" in progress["error"]
        finally:
            release_snapshot.set()
            if observer is not None:
                observer.cancel()
                await asyncio.gather(observer, return_exceptions=True)
            await asyncio.gather(*requests, return_exceptions=True)
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

    def fail(_config, *, semantic_capability=None) -> None:
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


@pytest.mark.parametrize("fail", [False, True])
def test_terminal_maintenance_refreshes_previous_inventory_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    from recall.services import index_migration

    cfg = lane_config(tmp_path)
    cfg.data_dir.mkdir(parents=True)
    with duckdb.connect(str(cfg.db_path)) as conn:
        conn.execute("CREATE TABLE owned_marker AS SELECT 3 AS value")
    server = RpcServer(config=cfg)

    if fail:

        def unavailable(_config, *, semantic_capability=None) -> None:
            raise OSError("snapshot unavailable")

        monkeypatch.setattr(index_migration, "perform_storage_transition", unavailable)

    async def until(predicate) -> None:
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0.01)

    async def scenario() -> None:
        inventory = asyncio.create_task(server._run_inventory_loop())
        try:
            await until(lambda: server._inventory_complete)
            # An inventory turn refused during maintenance is incomplete until
            # its next successful walk, even after the maintenance owner exits.
            server._inventory_complete = False
            server._inventory_error = "RpcError: storage maintenance is running"
            await server._handle_migrate_storage({}, None)
            assert server._storage_maintenance_task is not None
            await asyncio.wait_for(server._storage_maintenance_task, 5)
            await until(lambda: server._inventory_complete and server._inventory_error is None)
            status = await server._handle_daemon_status({}, None)
            assert status["reconciliation"]["error"] is None
        finally:
            server._shutdown_event.set()
            server._inventory_wakeup.set()
            await inventory
            await server.stop()

    asyncio.run(scenario())
