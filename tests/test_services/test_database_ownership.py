"""Real process ownership at the daemon/status boundary (REQ-CONC-004)."""

from __future__ import annotations

import fcntl
import json
import os
import select
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
from recall.core.config import AppConfig
from recall.db import connect
from recall.services.daemon import daemon_status
from recall.services.runtime_state import load_runtime_status


@contextmanager
def child(code: str, config: AppConfig) -> Iterator[subprocess.Popen[str]]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("RECALL_")}
    env.update(RECALL_DATA_DIR=str(config.data_dir), RECALL_CONFIG_PATH=str(config.config_path))
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=15)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def event(process: subprocess.Popen[str]) -> str:
    assert process.stdout is not None
    deadline = time.monotonic() + 15
    data = bytearray()
    while not data.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        assert remaining > 0 and select.select([process.stdout], [], [], remaining)[0], (
            "child event deadline expired"
        )
        piece = os.read(process.stdout.fileno(), 1)
        if not piece:
            assert process.stderr is not None
            pytest.fail(process.stderr.read())
        data.extend(piece)
    return data.decode().strip()


def release(process: subprocess.Popen[str]) -> None:
    assert process.stdin is not None
    process.stdin.write("release\n")
    process.stdin.flush()


def config_at(path: Path) -> AppConfig:
    config_file = path / "config.toml"
    config_file.write_text(
        "[daemon]\nmode='poll'\nembed=false\n[compaction]\nauto_trigger=false\n"
        + "".join(
            f"[sources.{source}]\nroots=[]\n"
            for source in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        )
    )
    with pytest.MonkeyPatch.context() as patch:
        for key in tuple(os.environ):
            if key.startswith("RECALL_"):
                patch.delenv(key)
        patch.setenv("RECALL_DATA_DIR", str(path))
        patch.setenv("RECALL_CONFIG_PATH", str(config_file))
        return AppConfig.load()


@pytest.mark.parametrize("clients", [1, 4])
@pytest.mark.parametrize("populated", [False, True], ids=["missing", "populated"])
def test_startup_status_cannot_create_or_displace_owner(
    socket_dir: Path, clients: int, populated: bool
) -> None:
    config = config_at(socket_dir)
    if populated:
        with connect(config) as conn:
            conn.execute(
                "INSERT INTO sessions VALUES ('persisted', 'codex', '/fixture/session', NULL)"
            )
            conn.execute("UPDATE runtime_state SET last_failure_message = 'persisted failure'")
        offline = daemon_status(config)
        assert offline.runtime_unavailable_reason is None
        assert offline.runtime_status.last_failure_message == "persisted failure"
        assert offline.embed_pending_measured_at is None
        assert offline.reconciliation["semantic_coverage"]["complete"] is None
    with child(
        """
import asyncio, sys
from recall.services.rpc_server import RpcServer
server = RpcServer()
original = server._cleanup_stale_socket
def pause():
    print('before-bind', flush=True)
    assert sys.stdin.readline().strip() == 'release'
    original()
server._cleanup_stale_socket = pause
async def run():
    task = asyncio.create_task(server.start())
    while server._fts_sidecar_startup is None and not task.done():
        await asyncio.sleep(.01)
    if task.done():
        await task
    print('ready', flush=True)
    assert (await asyncio.to_thread(sys.stdin.readline)).strip() == 'release'
    server.request_shutdown()
    await task
asyncio.run(run())
""",
        config,
    ) as daemon:
        assert event(daemon) == "before-bind"

        def read_status(_: int) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [str(Path(sys.executable).parent / "recall"), "daemon", "status", "--json"],
                env={
                    **{k: v for k, v in os.environ.items() if not k.startswith("RECALL_")},
                    "RECALL_DATA_DIR": str(config.data_dir),
                    "RECALL_CONFIG_PATH": str(config.config_path),
                    "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
                },
                capture_output=True,
                text=True,
                timeout=15,
            )

        with ThreadPoolExecutor(max_workers=clients) as pool:
            results = list(pool.map(read_status, range(clients)))
        for result in results:
            assert result.returncode == 0, result.stderr
            status = json.loads(result.stdout)
            reason = status["runtime_unavailable_reason"]
            assert reason is not None
            assert reason.startswith("database ownership busy at ")
            assert str(config.db_path.resolve()) + ".ownership" in reason
            assert status["embed_pending_measured_at"] is None
        assert config.db_path.exists() == populated
        release(daemon)
        assert event(daemon) == "ready"
        served = read_status(0)
        assert served.returncode == 0, served.stderr
        assert json.loads(served.stdout)["daemon_pid"] == daemon.pid
        release(daemon)
        assert daemon.wait(timeout=15) == 0
    if populated:
        with connect(config) as conn:
            assert conn.execute("SELECT id FROM sessions").fetchall() == [("persisted",)]


def test_missing_offline_status_never_creates_database(socket_dir: Path) -> None:
    config = config_at(socket_dir)
    assert load_runtime_status(config).last_attempted_at is None
    status = daemon_status(config)
    assert status.embed_pending_measured_at is None
    assert status.runtime_unavailable_reason is not None
    assert not config.db_path.exists()


def test_offline_status_preserves_persisted_runtime(socket_dir: Path) -> None:
    config = config_at(socket_dir)
    with connect(config) as conn:
        conn.execute("UPDATE runtime_state SET last_failure_message = 'persisted failure'")
    assert load_runtime_status(config).last_failure_message == "persisted failure"
    assert daemon_status(config).runtime_status.last_failure_message == "persisted failure"


_DAEMON = """
import asyncio, logging, sys
from recall.services.rpc_server import RpcServer
class Contention(logging.Handler):
    def emit(self, record):
        if 'waiting for database ownership' in record.getMessage():
            print('waiting', flush=True)
logging.getLogger('recall.ownership').addHandler(Contention())
logging.getLogger('recall.ownership').setLevel(logging.INFO)
server = RpcServer()
async def run():
    task = asyncio.create_task(server.start())
    while server._fts_sidecar_startup is None and not task.done():
        await asyncio.sleep(.01)
    if task.done():
        await task
    print('ready', flush=True)
    server.request_shutdown()
    await task
asyncio.run(run())
"""


def test_daemon_waits_for_the_complete_offline_snapshot(socket_dir: Path) -> None:
    config = config_at(socket_dir)
    connect(config).close()
    with child(
        """
import sys
from recall.services import runtime_state
from recall.services.daemon import offline_daemon_status
from recall.core.config import AppConfig
original = runtime_state.load_runtime_status_from_conn
def pause(conn):
    print('snapshot-open', flush=True)
    assert sys.stdin.readline().strip() == 'release'
    return original(conn)
runtime_state.load_runtime_status_from_conn = pause
status, page = offline_daemon_status(AppConfig.load(), limit=10, cursor=None)
assert status.embed_pending_measured_at is None
assert status.reconciliation["semantic_coverage"]["complete"] is None
assert page is not None
print('snapshot-closed', flush=True)
""",
        config,
    ) as reader:
        assert event(reader) == "snapshot-open"
        with child(_DAEMON, config) as daemon:
            assert event(daemon) == "waiting"
            assert not (socket_dir / "recall.sock").exists()
            assert not (socket_dir / "recall.pid").exists()
            release(reader)
            assert event(reader) == "snapshot-closed"
            assert reader.wait(timeout=15) == 0
            assert event(daemon) == "ready"
            assert daemon.wait(timeout=15) == 0


def test_daemon_timeout_preserves_another_owners_artifacts(socket_dir: Path) -> None:
    config = config_at(socket_dir)
    with child(
        """
import fcntl, sys
from pathlib import Path
from recall.core.config import AppConfig
path = Path(str(AppConfig.load().db_path.resolve()) + '.ownership')
with path.open('a') as gate:
    fcntl.flock(gate, fcntl.LOCK_EX)
    print('owned', flush=True)
    sys.stdin.readline()
""",
        config,
    ) as owner:
        assert event(owner) == "owned"
        pid_file = socket_dir / "recall.pid"
        socket_file = socket_dir / "recall.sock"
        pid_file.write_text("123456789")
        socket_file.write_text("owner socket marker")
        with child(_DAEMON, config) as daemon:
            assert event(daemon) == "waiting"
            assert daemon.wait(timeout=10) != 0
            assert daemon.stderr is not None
            error = daemon.stderr.read()
            assert "database ownership busy" in error
            assert str(config.db_path) in error
            assert "retry" in error
        assert pid_file.read_text() == "123456789"
        assert socket_file.read_text() == "owner socket marker"


def test_owner_death_and_symlink_alias_recover_without_cleanup(socket_dir: Path) -> None:
    from dataclasses import replace

    from recall.db.ownership import ownership_path

    config = config_at(socket_dir)
    connect(config).close()
    alias = socket_dir / "alias.duckdb"
    alias.symlink_to(config.db_path)
    alias_config = replace(config, db_path=alias)
    assert ownership_path(alias) == ownership_path(config.db_path)
    with child(
        """
import fcntl, sys
from recall.core.config import AppConfig
from recall.db.ownership import ownership_path
with ownership_path(AppConfig.load().db_path).open('a') as gate:
    fcntl.flock(gate, fcntl.LOCK_EX)
    print('owned', flush=True)
    sys.stdin.readline()
""",
        config,
    ) as owner:
        assert event(owner) == "owned"
        assert daemon_status(alias_config).runtime_unavailable_reason is not None
        owner.kill()
        owner.wait(timeout=15)
    assert ownership_path(alias).exists()
    assert daemon_status(alias_config).embed_pending_measured_at is None
    assert daemon_status(alias_config).runtime_unavailable_reason is None


def test_failed_startup_closes_database_and_removes_owned_artifacts(socket_dir: Path) -> None:
    config = config_at(socket_dir)
    with child(
        """
import asyncio
from recall.services import daemon
from recall.services.rpc_server import RpcServer
def fail(config):
    raise RuntimeError('startup snapshot failure')
daemon.run_startup_snapshot_gc = fail
async def run():
    try:
        await RpcServer().start()
    except RuntimeError as err:
        assert str(err) == 'startup snapshot failure'
    else:
        raise AssertionError('startup should fail')
    print('failed-cleanly', flush=True)
asyncio.run(run())
""",
        config,
    ) as daemon:
        assert event(daemon) == "failed-cleanly"
        assert daemon.wait(timeout=15) == 0
    assert not (socket_dir / "recall.sock").exists()
    assert not (socket_dir / "recall.pid").exists()
    assert daemon_status(config).embed_pending_measured_at is None
    assert daemon_status(config).runtime_unavailable_reason is None


def test_gate_permission_error_is_explicit_and_does_not_open_database(socket_dir: Path) -> None:
    from recall.db.ownership import ownership_path

    config = config_at(socket_dir)
    gate = ownership_path(config.db_path)
    gate.mkdir()
    status = daemon_status(config)
    assert status.runtime_unavailable_reason is not None
    assert str(gate) in status.runtime_unavailable_reason
    assert not config.db_path.exists()
    with child(_DAEMON, config) as daemon:
        assert daemon.wait(timeout=10) != 0
        assert daemon.stderr is not None
        assert "cannot open database ownership gate" in daemon.stderr.read()
    assert not (socket_dir / "recall.sock").exists()


def test_cancelled_startup_keeps_ownership_until_worker_exits(socket_dir: Path) -> None:
    config = config_at(socket_dir)
    with child(
        """
import asyncio, sys, threading
from recall.services import daemon
from recall.services.rpc_server import RpcServer
original = daemon.run_startup_snapshot_gc
entered = threading.Event()
def pause(config):
    print('worker-open', flush=True)
    entered.set()
    assert sys.stdin.readline().strip() == 'release'
    return original(config)
daemon.run_startup_snapshot_gc = pause
async def run():
    server = RpcServer()
    task = asyncio.create_task(server.start())
    while not entered.is_set():
        await asyncio.sleep(.01)
    # Cancellation is delivered while startup's executor is still working.
    task.cancel()
    print('cancelled', flush=True)
    try:
        await task
    except asyncio.CancelledError:
        pass
    print('drained', flush=True)
asyncio.run(run())
""",
        config,
    ) as daemon:
        markers = {event(daemon), event(daemon)}
        assert markers == {"worker-open", "cancelled"}
        assert daemon_status(config).runtime_unavailable_reason is not None
        assert (socket_dir / "recall.pid").exists()
        release(daemon)
        assert event(daemon) == "drained"
        assert daemon.wait(timeout=15) == 0
    assert not (socket_dir / "recall.sock").exists()
    assert not (socket_dir / "recall.pid").exists()
    assert daemon_status(config).embed_pending_measured_at is None
    assert daemon_status(config).runtime_unavailable_reason is None


def test_stale_live_pid_does_not_suppress_an_unowned_snapshot(socket_dir: Path) -> None:
    config = config_at(socket_dir)
    with connect(config) as conn:
        conn.execute("UPDATE runtime_state SET last_failure_message = 'survives PID reuse'")
    (socket_dir / "recall.pid").write_text(str(os.getpid()))
    status = daemon_status(config)
    assert status.runtime_unavailable_reason is None
    assert status.runtime_status.last_failure_message == "survives PID reuse"
    assert status.embed_pending_measured_at is None
    assert status.reconciliation["semantic_coverage"]["complete"] is None


@pytest.mark.parametrize("busy", [True, False], ids=["busy", "missing"])
def test_unavailable_snapshot_does_not_attribute_a_stale_live_pid(
    socket_dir: Path, busy: bool
) -> None:
    config = config_at(socket_dir)
    (socket_dir / "recall.pid").write_text(str(os.getpid()))
    gate = Path(f"{config.db_path.resolve()}.ownership")
    with gate.open("a") as owner:
        if busy:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = daemon_status(config)
        assert status.daemon_pid is None
        reason = status.runtime_unavailable_reason
        assert reason is not None
        assert "daemon pid" not in reason
        expected_reason = "ownership busy" if busy else "database does not exist"
        assert expected_reason in reason
        for output in (["--format", "text"], ["--json", "--cta"]):
            env = {k: v for k, v in os.environ.items() if not k.startswith("RECALL_")}
            env.update(
                RECALL_DATA_DIR=str(config.data_dir),
                RECALL_CONFIG_PATH=str(config.config_path),
                PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            )
            result = subprocess.run(
                [str(Path(sys.executable).parent / "recall"), "daemon", "status", *output],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert result.returncode == 0, result.stderr
            if "--json" in output:
                payload = json.loads(result.stdout)
                assert payload["data"]["runtime_unavailable_reason"] == reason
                assert payload["data"]["daemon_pid"] is None
                assert payload["cta"] == []
            else:
                assert f"Database: {reason}" in result.stdout
                assert f"pid={os.getpid()}" not in result.stdout
                assert "RPC not answering" not in result.stdout
            assert "Next:" not in result.stderr
    assert not config.db_path.exists()
