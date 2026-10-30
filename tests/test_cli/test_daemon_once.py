"""`recall daemon --once` runs its cycle as the data dir's daemon (REQ-DAEMON-089).

A poll-mode service runs `--once` on every tick with no resident daemon, so the
cycle must not depend on reaching or forking one. When a daemon does own the
database, the cycle is its to serve, and `--once` never opens the database beside it.
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
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


def _ownership_is_free(data: Path) -> bool:
    gate = ownership_path(data / "recall.duckdb").open("a")
    try:
        fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        gate.close()
    return True


def _fail_bind(*_args: object, **_kwargs: object) -> None:
    raise OSError(errno.EADDRINUSE, "Address already in use")


def _fail_sidecar_sync(*_args: object, **_kwargs: object) -> None:
    from recall.db import RecallLockError

    raise RecallLockError("sidecar held by another writer")


def _fail_snapshot_gc(*_args: object, **_kwargs: object) -> None:
    raise duckdb.IOException("snapshot directory unreadable")


@pytest.mark.parametrize(
    ("target", "replacement", "code", "exit_code"),
    [
        ("recall.services.rpc_server._bind_owner_only_socket", _fail_bind, "RUNTIME", 1),
        (
            "recall.services.daemon.run_startup_fts_sidecar_sync",
            _fail_sidecar_sync,
            "LOCKED",
            1,
        ),
        ("recall.services.daemon.run_startup_snapshot_gc", _fail_snapshot_gc, "RUNTIME", 1),
    ],
    ids=["bind", "sidecar-lock", "duckdb-startup-step"],
)
def test_a_failed_startup_answers_with_the_error_envelope_and_releases_ownership(
    poll_host: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    replacement: object,
    code: str,
    exit_code: int,
) -> None:
    monkeypatch.setattr(target, replacement)

    result = _invoke()

    assert result.exit_code == exit_code, result.output
    error = json.loads(result.stdout)["error"]
    assert error["code"] == code
    assert error["message"].startswith("no daemon cycle ran: ")
    assert _ownership_is_free(poll_host)
    assert not (poll_host / "recall.sock").exists()
    assert not (poll_host / "recall.pid").exists()


def test_a_refusal_before_ownership_answers_with_the_error_envelope(poll_host: Path) -> None:
    from recall.core.keyword_maintenance import MARKER_NAME

    (poll_host / MARKER_NAME).write_text("{}")

    exit_code, payload = _once()

    assert exit_code == 1
    assert payload["error"]["code"] == "RUNTIME"
    message = payload["error"]["message"]
    assert message.startswith("no daemon cycle ran: Keyword maintenance is held")
    assert _ownership_is_free(poll_host)


@pytest.fixture
def owner_that_drops_answers(poll_host: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list]:
    """An owner that accepts each request and closes the connection before answering."""
    import recall.db.ownership as ownership_module

    monkeypatch.setattr(ownership_module, "STARTUP_OWNERSHIP_TIMEOUT_SECONDS", 0.2)
    gate = ownership_path(poll_host / "recall.duckdb").open("a")
    fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(poll_host / "recall.sock"))
    listener.listen()
    listener.settimeout(0.1)
    requests: list[dict] = []
    done = threading.Event()

    def serve() -> None:
        while not done.is_set():
            try:
                peer, _ = listener.accept()
            except TimeoutError:
                continue
            with peer, peer.makefile("rb") as frames:
                requests.append(json.loads(frames.readline()))

    thread = threading.Thread(target=serve, name="answer-dropping-owner", daemon=True)
    thread.start()
    try:
        yield requests
    finally:
        done.set()
        thread.join(timeout=5)
        listener.close()
        gate.close()


def test_a_lost_answer_after_the_owner_accepted_the_cycle_is_reported_unconfirmed(
    poll_host: Path, owner_that_drops_answers: list[dict]
) -> None:
    exit_code, payload = _once()

    assert exit_code == 1
    assert [r["method"] for r in owner_that_drops_answers] == ["recall.daemon_run"] * 3
    error = payload["error"]
    assert error["code"] == "RUNTIME"
    assert error["message"].startswith("daemon cycle outcome unconfirmed: ")
    assert "may have run in part" in error["message"]
    assert "no daemon cycle ran" not in error["message"]


@pytest.mark.parametrize("next_attempt", ["owner-accepts-no-connection", "startup-fails"])
def test_a_lost_answer_stays_unconfirmed_when_later_attempts_fail(
    poll_host: Path, monkeypatch: pytest.MonkeyPatch, next_attempt: str
) -> None:
    """The owner accepted the cycle and died; the takeover attempts that follow fail too."""
    import recall.db.ownership as ownership_module
    from recall.cli import daemon as daemon_cli
    from recall.cli.contract import ErrorCode
    from recall.cli.rpc import DaemonAnswerLostError, DaemonUnreachableError

    monkeypatch.setattr(ownership_module, "STARTUP_OWNERSHIP_TIMEOUT_SECONDS", 0.2)
    gate = ownership_path(poll_host / "recall.duckdb").open("a")
    fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
    answers = iter(
        [
            DaemonAnswerLostError(code=ErrorCode.RUNTIME, message="connection closed by daemon"),
            DaemonUnreachableError(code=ErrorCode.RUNTIME, message="connection refused"),
            DaemonUnreachableError(code=ErrorCode.RUNTIME, message="connection refused"),
        ]
    )

    def owner_answers(*_args: object, **_kwargs: object) -> None:
        if next_attempt == "startup-fails":
            gate.close()
        raise next(answers)

    monkeypatch.setattr(daemon_cli, "rpc_call_or_error", owner_answers)
    monkeypatch.setattr("recall.services.rpc_server._bind_owner_only_socket", _fail_bind)
    try:
        exit_code, payload = _once()
    finally:
        gate.close()

    assert exit_code == 1
    message = payload["error"]["message"]
    assert message.startswith("daemon cycle outcome unconfirmed: ")
    assert "may have run in part" in message
    assert "no daemon cycle ran" not in message


def test_a_shutdown_signal_during_the_cycle_reports_an_interrupted_cycle(
    poll_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services.rpc_server import RpcServer

    async def signalled_mid_cycle(server: RpcServer, *_args: object, **_kwargs: object) -> None:
        server._signal_shutdown()
        await asyncio.sleep(60)

    monkeypatch.setattr(RpcServer, "_reconcile_index_request", signalled_mid_cycle)

    exit_code, payload = _once()

    assert exit_code == 1
    error = payload["error"]
    assert error["code"] == "RUNTIME"
    assert error["message"].startswith("daemon cycle interrupted: ")
    assert "no daemon cycle ran" not in error["message"]
    assert _ownership_is_free(poll_host)
    assert not (poll_host / "recall.sock").exists()
    with duckdb.connect(str(poll_host / "recall.duckdb"), read_only=True) as conn:
        recorded = conn.execute("SELECT last_failure_message FROM runtime_state").fetchone()
    assert recorded == ("cancelled: the daemon received a shutdown signal",)


def _write_sessions(root: Path, count: int) -> None:
    project = root / "bulk"
    project.mkdir(parents=True)
    for index in range(count):
        lines = (
            json.dumps(
                {
                    "type": "message",
                    "timestamp": f"2024-01-15T10:{turn:02d}:00Z",
                    "message": {
                        "role": "user" if turn % 2 == 0 else "assistant",
                        "content": [{"type": "text", "text": f"bulk {index} {turn}"}],
                    },
                }
            )
            for turn in range(20)
        )
        (project / f"bulk-{index}.jsonl").write_text("\n".join(lines) + "\n")


def _start_once(data: Path, name: str) -> tuple[subprocess.Popen[str], Path]:
    stderr = data.parent / f"{name}.err"
    process = subprocess.Popen(
        [str(Path(sys.executable).parent / "recall"), "daemon", "--once", "--json"],
        stdout=subprocess.PIPE,
        stderr=stderr.open("w"),
        env=os.environ.copy(),
        text=True,
    )
    return process, stderr


def _wait_for(condition: Callable[[], bool], what: str, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.05)


def test_two_overlapping_once_processes_both_complete_the_cycle(poll_host: Path) -> None:
    sessions = 120
    _write_sessions(poll_host.parent / "cc", sessions)

    first, _ = _start_once(poll_host, "first")
    try:
        _wait_for(lambda: (poll_host / "recall.pid").exists(), "the first cycle's ownership")
        second, second_err = _start_once(poll_host, "second")
        try:
            outputs = [process.communicate(timeout=120)[0] for process in (first, second)]
        finally:
            second.kill()
    finally:
        first.kill()

    assert [first.returncode, second.returncode] == [0, 0], outputs
    # The second started while the first owned the database.
    assert "waiting for database ownership" in second_err.read_text()
    summaries = [json.loads(output)["index_summary"] for output in outputs]
    assert all(summary["failed"] == 0 for summary in summaries)
    assert _runtime(poll_host) == (sessions + 1, "daemon-once")
    assert not (poll_host / "recall.sock").exists()
    assert not (poll_host / "recall.pid").exists()
    # Ownership is taken again by the next cycle.
    exit_code, payload = _once()
    assert exit_code == 0, payload
    assert payload["index_summary"]["indexed"] == 0


def test_an_interrupt_before_the_cycle_starts_exits_130_with_nothing_on_stdout(
    poll_host: Path,
) -> None:
    gate = ownership_path(poll_host / "recall.duckdb").open("a")
    fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        process, stderr = _start_once(poll_host, "interrupted")
        try:
            _wait_for(
                lambda: "waiting for database ownership" in stderr.read_text(),
                "the ownership wait",
            )
            process.send_signal(signal.SIGINT)
            stdout, _ = process.communicate(timeout=30)
        finally:
            process.kill()
    finally:
        gate.close()

    assert process.returncode == 130, stderr.read_text()
    assert stdout == ""
    assert not (poll_host / "recall.duckdb").exists()
    assert not (poll_host / "recall.sock").exists()
