from __future__ import annotations

import asyncio
import multiprocessing
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, replace
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
import recall.services.daemon as daemon_module
from recall.core.config import AppConfig, CompactionConfig, DaemonConfig, FtsConfig
from recall.core.rpc_client import RpcClient
from recall.core.types import DaemonMode, SchedulerKind, Source
from recall.db import connect
from recall.services.daemon import install_scheduler, uninstall_scheduler
from recall.services.runtime_state import load_runtime_status_from_conn


@pytest.fixture
def scheduler_home() -> Iterator[Path]:
    # macOS pytest paths exceed the Unix socket path limit.
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="recall-install-") as directory:
        yield Path(directory)


def _serve_database(
    config: AppConfig, control: Connection, startup_delay: float = 0, break_metadata: bool = False
) -> None:
    """Own the real database and RPC transport without host scheduler side effects."""
    from recall.services.rpc_server import RpcServer

    async def run() -> None:
        server = RpcServer(config=config)
        conn = server._get_conn()
        if break_metadata:
            conn.execute("DROP TABLE runtime_state")
        control.send("locked")
        # Delay is an injected startup interleaving; the RPC response, not time,
        # must prove the install finished successfully.
        await asyncio.sleep(startup_delay)
        server._server = await asyncio.start_unix_server(
            server._handle_connection, path=str(config.data_dir / "recall.sock")
        )
        try:
            await asyncio.wait_for(asyncio.to_thread(control.recv), timeout=30)
        finally:
            await server.stop()

    asyncio.run(run())


@pytest.mark.parametrize("startup_delay", [0, 0.2])
def test_install_and_uninstall_persist_scheduler_while_daemon_owns_database(
    scheduler_home: Path, monkeypatch: pytest.MonkeyPatch, startup_delay: float
) -> None:
    """REQ-RPC-006/012: management must respect the daemon's database ownership."""
    tmp_path = scheduler_home
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "config.toml"))
    config = AppConfig.load()
    binary = tmp_path / "recall"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(binary))
    context = multiprocessing.get_context("spawn")
    control, child_control = context.Pipe()
    child = context.Process(target=_serve_database, args=(config, child_control, startup_delay))
    loaded = False

    def run_command(args: list[str], *, check: bool):
        nonlocal loaded
        if args[:2] == ["launchctl", "bootstrap"]:
            child.start()
            assert control.poll(10), "child daemon did not become ready"
            assert control.recv() == "locked"
            loaded = True
        return daemon_module.subprocess.CompletedProcess(
            args, 0 if loaded else 1, stdout="", stderr=""
        )

    monkeypatch.setattr(daemon_module, "_run_command", run_command)
    monkeypatch.setattr(
        daemon_module, "_run_lifecycle_query", lambda args: run_command(args, check=False)
    )
    monkeypatch.setattr(daemon_module, "_post_install_smoke_test", lambda _args: None)
    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    try:
        status = install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)
        assert child.is_alive()
        assert status.installed is True
        assert status.runtime_status.installed_scheduler == SchedulerKind.LAUNCHD
        assert status.scheduler == SchedulerKind.LAUNCHD
        status = uninstall_scheduler(config=config)
        assert child.is_alive(), "exercise asynchronous daemon teardown after bootout"
        assert status.installed is False
        assert status.runtime_status.installed_scheduler is None
    finally:
        if child.pid is not None:
            control.send("stop")
            child.join(timeout=10)
            if child.is_alive():
                child.kill()
                child.join(timeout=5)
        control.close()
        child_control.close()

    assert child.exitcode == 0
    conn = connect(config)
    try:
        assert load_runtime_status_from_conn(conn).installed_scheduler is None
    finally:
        conn.close()


def test_scheduler_management_rejects_invalid_status_from_daemon(
    scheduler_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(scheduler_home))
    monkeypatch.setenv("RECALL_DATA_DIR", str(scheduler_home / "data"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(scheduler_home / "config.toml"))
    config = AppConfig.load()
    conn = connect(config)
    try:
        payload = asdict(daemon_module.daemon_status(config=config, conn=conn))
    finally:
        conn.close()
    payload["scheduler"] = "not-a-scheduler"
    monkeypatch.setattr(RpcClient, "call", lambda _self, _method: payload)
    with pytest.raises(RuntimeError, match="daemon returned invalid scheduler status"):
        daemon_module._scheduler_status_from_rpc(RpcClient(config))


def test_scheduler_metadata_write_failure_is_reported_from_owning_daemon(
    scheduler_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(scheduler_home))
    monkeypatch.setenv("RECALL_DATA_DIR", str(scheduler_home / "data"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(scheduler_home / "config.toml"))
    config = AppConfig.load()
    context = multiprocessing.get_context("spawn")
    control, child_control = context.Pipe()
    child = context.Process(target=_serve_database, args=(config, child_control, 0, True))
    child.start()
    try:
        assert control.poll(10)
        assert control.recv() == "locked"
        with pytest.raises(
            RuntimeError, match=r"could not persist installed scheduler:.*runtime_state"
        ):
            daemon_module._update_installed_scheduler(config, SchedulerKind.LAUNCHD)
    finally:
        control.send("stop")
        child.join(timeout=10)
        if child.is_alive():
            child.kill()
            child.join(timeout=5)
        control.close()
        child_control.close()
    assert child.exitcode == 0


def _start_full_watch_daemon(config: AppConfig, control: Connection) -> None:
    from recall.services.rpc_server import RpcServer

    async def run() -> None:
        server = RpcServer(config=config)
        control.send("spawned")
        assert await asyncio.to_thread(control.recv) == "start"
        # Exercise bootstrap returning before the daemon opens the database.
        await asyncio.sleep(0.1)
        task = asyncio.create_task(server.start(watch=True))
        try:
            async with asyncio.timeout(10):
                while not server._inventory_complete:
                    if task.done():
                        await task
                        raise RuntimeError("daemon exited before watch readiness")
                    await asyncio.sleep(0.01)
            status = await server._handle_daemon_status({}, None)
            assert status["reconciliation"]["rpc_ready"] is True
            assert status["reconciliation"]["live_observation_ready"] is True
            assert server._get_conn().execute("SELECT 1").fetchone() == (1,)
            control.send("ready")
            assert await asyncio.to_thread(control.recv) == "stop"
            server.request_shutdown()
            await asyncio.wait_for(task, timeout=5)
        except Exception as err:
            control.send(f"startup failed: {err}")
            raise
        finally:
            await server.stop()

    asyncio.run(run())


def test_watch_install_does_not_lock_out_daemon_starting_after_bootstrap(
    scheduler_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(scheduler_home))
    monkeypatch.setenv("RECALL_DATA_DIR", str(scheduler_home / "data"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(scheduler_home / "config.toml"))
    config = replace(
        AppConfig.load(),
        daemon=DaemonConfig(mode=DaemonMode.WATCH, embed=False, source=Source.CODEX),
        fts=FtsConfig(backend="sqlite_sidecar"),
        compaction=CompactionConfig(auto_trigger=False),
    )
    binary = scheduler_home / "recall"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(binary))
    context = multiprocessing.get_context("spawn")
    control, child_control = context.Pipe()
    child = context.Process(target=_start_full_watch_daemon, args=(config, child_control))
    loaded = False
    startup_result: str | None = None
    original_status = daemon_module.daemon_status

    def run_command(args: list[str], *, check: bool):
        nonlocal loaded
        if args[:2] == ["launchctl", "bootstrap"]:
            child.start()
            assert control.poll(10)
            assert control.recv() == "spawned"
            control.send("start")
            loaded = True
        return daemon_module.subprocess.CompletedProcess(
            args, 0 if loaded else 1, stdout="", stderr=""
        )

    def status_during_startup(*args, **kwargs):
        nonlocal startup_result
        if kwargs.get("conn") is not None:
            # Hold any client-owned connection across actual child startup,
            # as a slow status query can do on a large corpus.
            assert control.poll(10)
            startup_result = control.recv()
        return original_status(*args, **kwargs)

    monkeypatch.setattr(daemon_module, "_run_command", run_command)
    monkeypatch.setattr(
        daemon_module, "_run_lifecycle_query", lambda args: run_command(args, check=False)
    )
    monkeypatch.setattr(daemon_module, "_post_install_smoke_test", lambda _args: None)
    monkeypatch.setattr(daemon_module, "daemon_status", status_during_startup)
    monkeypatch.setattr(daemon_module.sys, "platform", "darwin")
    try:
        status = install_scheduler(scheduler=SchedulerKind.LAUNCHD, config=config)
        assert status.installed is True
        assert status.runtime_status.installed_scheduler == SchedulerKind.LAUNCHD
        if startup_result is None:
            assert control.poll(10)
            startup_result = control.recv()
        assert startup_result == "ready"
        assert child.is_alive()
    finally:
        if child.pid is not None:
            control.send("stop")
            child.join(timeout=10)
            if child.is_alive():
                child.kill()
                child.join(timeout=5)
        control.close()
        child_control.close()
    assert child.exitcode == 0


@pytest.mark.parametrize("scheduler", [SchedulerKind.LAUNCHD, SchedulerKind.SYSTEMD])
def test_watch_install_errors_when_activated_daemon_never_becomes_ready(
    scheduler_home: Path, monkeypatch: pytest.MonkeyPatch, scheduler: SchedulerKind
) -> None:
    monkeypatch.setenv("HOME", str(scheduler_home))
    monkeypatch.setenv("RECALL_DATA_DIR", str(scheduler_home / "data"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(scheduler_home / "config.toml"))
    config = replace(AppConfig.load(), daemon=DaemonConfig(mode=DaemonMode.WATCH, embed=False))
    binary = scheduler_home / "recall"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(daemon_module, "_resolve_recall_binary", lambda: str(binary))
    monkeypatch.setattr(daemon_module, "_install_launchd", lambda *_args: None)
    monkeypatch.setattr(daemon_module, "_install_systemd", lambda *_args: None)
    monkeypatch.setattr(
        daemon_module.sys, "platform", "darwin" if scheduler == SchedulerKind.LAUNCHD else "linux"
    )
    monkeypatch.setattr(daemon_module, "_systemd_user_available", lambda: True)
    monkeypatch.setattr(daemon_module, "_SCHEDULER_DATABASE_WAIT_SECONDS_MAX", 0.01)
    with pytest.raises(RuntimeError, match="daemon did not become ready"):
        install_scheduler(scheduler=scheduler, config=config)
    assert not config.db_path.exists(), "waiting for a daemon must not open its database"
