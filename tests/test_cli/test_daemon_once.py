"""`recall daemon --once` runs its cycle as the data dir's daemon (REQ-DAEMON-089).

A poll-mode service runs `--once` on every tick with no resident daemon, so the
cycle must not depend on reaching or forking one. When a daemon does own the
database, the cycle is its to serve, and `--once` never opens the database beside it.
"""

from __future__ import annotations

import fcntl
import json
import logging
import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
from recall.cli.app import app
from recall.cli.daemon import _DaemonLogHandler
from recall.core.config import AppConfig
from recall.db.ownership import ownership_path
from typer.testing import CliRunner, Result

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "claude_code"
SOURCES = ("claude_code", "codex", "pi_agent", "grok", "kimi_code")


@pytest.fixture
def poll_host(socket_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data dir whose installed service runs `--once`, with one transcript and no daemon."""
    from recall.cli import daemon as daemon_cli
    from recall.cli import rpc as rpc_cli
    from recall.core import rpc_client

    data = socket_dir / "data"
    data.mkdir()
    project = socket_dir / "cc" / "proj-a"
    project.mkdir(parents=True)
    shutil.copy(FIXTURES / "session1.jsonl", project / "session1.jsonl")
    config = socket_dir / "config.toml"
    roots = {source: [] for source in SOURCES} | {"claude_code": [str(socket_dir / "cc")]}
    config.write_text(
        '[daemon]\nmode = "poll"\nembed = false\n\n[compaction]\nauto_trigger = false\n'
        + "".join(f"\n[sources.{name}]\nroots = {json.dumps(r)}\n" for name, r in roots.items())
    )
    for key, path in {
        "RECALL_DATA_DIR": data,
        "RECALL_DB_PATH": data / "recall.duckdb",
        "RECALL_LOCK_PATH": data / "recall.lock",
        "RECALL_CONFIG_PATH": config,
    }.items():
        monkeypatch.setenv(key, str(path))
    monkeypatch.setattr(rpc_cli, "_in_process_server", None)
    monkeypatch.setattr(
        daemon_cli,
        "resolve_service_scope",
        lambda _config: SimpleNamespace(
            service=SimpleNamespace(scheduler=SimpleNamespace(value="systemd")),
            other_service=None,
        ),
    )

    def no_fork(*_args: object, **_kwargs: object) -> None:
        pytest.fail("daemon --once must never fork a daemon")

    monkeypatch.setattr(rpc_client, "spawn_detached_recall", no_fork)
    return data


@pytest.fixture(autouse=True)
def _restore_process_logging() -> Iterator[None]:
    """A cycle run here configures the daemon's root logger; hand it back as found."""
    root = logging.getLogger()
    recall_logger = logging.getLogger("recall")
    handlers = root.handlers[:]
    recall_level = recall_logger.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        recall_logger.setLevel(recall_level)


def _invoke() -> Result:
    result = CliRunner().invoke(app, ["daemon", "--once", "--json"])
    # One process runs one cycle. The handler keeps this invocation's captured
    # stderr, which the runner closes, so a later invocation installs its own.
    root = logging.getLogger()
    root.handlers[:] = [h for h in root.handlers if not isinstance(h, _DaemonLogHandler)]
    return result


def _once() -> tuple[int, dict]:
    result = _invoke()
    return result.exit_code, json.loads(result.stdout)


def _runtime(data: Path) -> tuple[int, str | None]:
    with duckdb.connect(str(data / "recall.duckdb"), read_only=True) as conn:
        sessions = conn.execute("SELECT count(*) FROM sessions").fetchone()
        kind = conn.execute("SELECT last_run_kind FROM runtime_state").fetchone()
    assert sessions is not None
    return sessions[0], kind[0] if kind else None


def test_once_under_an_installed_service_runs_the_cycle_without_a_daemon(poll_host: Path) -> None:
    exit_code, payload = _once()

    assert exit_code == 0, payload
    assert payload["index_summary"]["indexed"] == 1
    assert payload["index_summary"]["failed"] == 0
    assert payload["record_status_persisted"] is True
    # Nothing outlives the command: no socket, no pid file, ownership free.
    assert not (poll_host / "recall.sock").exists()
    assert not (poll_host / "recall.pid").exists()
    assert _runtime(poll_host) == (1, "daemon-once")

    exit_code, payload = _once()

    assert exit_code == 0, payload
    assert payload["index_summary"]["indexed"] == 0
    assert payload["index_summary"]["skipped"] == 1
    assert _runtime(poll_host) == (1, "daemon-once")


@pytest.fixture
def resident_daemon(poll_host: Path) -> Iterator[threading.Thread]:
    from recall.services.rpc_server import start_server_blocking

    shutdown = threading.Event()
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            start_server_blocking(config=AppConfig.load(), shutdown_event=shutdown)
        except BaseException as err:  # surfaced by the readiness wait
            errors.append(err)

    thread = threading.Thread(target=serve, name="resident-daemon", daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not (poll_host / "recall.pid").exists():
        assert not errors, errors
        assert time.monotonic() < deadline, "resident daemon never listened"
        time.sleep(0.05)
    try:
        yield thread
    finally:
        shutdown.set()
        thread.join(timeout=30)
        assert not thread.is_alive(), "resident daemon did not stop"
        assert not errors, errors


def test_once_beside_a_resident_daemon_is_served_by_it(
    poll_host: Path, resident_daemon: threading.Thread, monkeypatch: pytest.MonkeyPatch
) -> None:
    import recall.db.ownership as ownership_module

    monkeypatch.setattr(ownership_module, "STARTUP_OWNERSHIP_TIMEOUT_SECONDS", 0.2)

    exit_code, payload = _once()

    assert exit_code == 0, payload
    assert payload["index_summary"]["failed"] == 0
    assert resident_daemon.is_alive()
    # The resident still owns the database; `--once` never took it.
    owner = ownership_path(poll_host / "recall.duckdb").open("a")
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        owner.close()


def test_once_refuses_when_the_owner_serves_no_requests(
    poll_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import recall.db.ownership as ownership_module

    monkeypatch.setattr(ownership_module, "STARTUP_OWNERSHIP_TIMEOUT_SECONDS", 0.2)
    owner = ownership_path(poll_host / "recall.duckdb").open("a")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        exit_code, payload = _once()
    finally:
        owner.close()

    assert exit_code == 1
    error = payload["error"]
    assert error["code"] == "RUNTIME"
    assert error["message"].startswith("no daemon cycle ran: database ownership busy")
    assert not (poll_host / "recall.duckdb").exists()
    assert not (poll_host / "recall.sock").exists()


def test_once_runs_the_cycle_itself_when_the_owner_exits_while_serving_it(
    poll_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two `--once` at a time: the first finishes and exits while the second waits on it."""
    import recall.db.ownership as ownership_module
    from recall.cli import daemon as daemon_cli
    from recall.cli.contract import ErrorCode
    from recall.cli.rpc import DaemonUnreachableError

    monkeypatch.setattr(ownership_module, "STARTUP_OWNERSHIP_TIMEOUT_SECONDS", 0.2)
    owner = ownership_path(poll_host / "recall.duckdb").open("a")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def owner_exits(*_args: object, **_kwargs: object) -> None:
        owner.close()
        raise DaemonUnreachableError(code=ErrorCode.RUNTIME, message="connection closed by daemon")

    monkeypatch.setattr(daemon_cli, "rpc_call_or_error", owner_exits)
    try:
        exit_code, payload = _once()
    finally:
        owner.close()

    assert exit_code == 0, payload
    assert payload["index_summary"]["indexed"] == 1
    assert _runtime(poll_host) == (1, "daemon-once")


@pytest.mark.parametrize(
    ("failure", "code", "exit_code"),
    [
        (duckdb.IOException("disk read failed"), "RUNTIME", 1),
        (ValueError("bad cycle input"), "VALIDATION", 2),
    ],
)
def test_a_failed_cycle_answers_like_the_served_request(
    poll_host: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    code: str,
    exit_code: int,
) -> None:
    from recall.services.rpc_server import RpcServer

    async def fail(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(RpcServer, "_reconcile_index_request", fail)

    result = _invoke()

    assert result.exit_code == exit_code, result.output
    assert json.loads(result.stdout)["error"] == {"code": code, "message": str(failure)}
    assert not (poll_host / "recall.sock").exists()
    assert not (poll_host / "recall.pid").exists()
