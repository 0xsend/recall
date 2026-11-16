"""Integration test: CLI → real daemon → DB round-trip over Unix socket.

Starts a real RPC daemon in a subprocess, sends requests over the socket,
and verifies correct responses. This exercises the full RPC path that
in-process testing skips.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from conftest import _can_acquire_duckdb_lock

pytestmark = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)


@pytest.fixture
def daemon_env(tmp_path: Path, socket_dir: Path):
    """Set up an isolated environment and start a real daemon."""
    tmp = tmp_path
    data_dir = socket_dir / "data"
    data_dir.mkdir(parents=True)

    # Install a fixture session
    claude_dir = tmp / ".claude" / "projects" / "p1"
    claude_dir.mkdir(parents=True)
    fixture_src = (
        Path(__file__).resolve().parents[2] / "fixtures" / "claude_code" / "session1.jsonl"
    )
    shutil.copy(fixture_src, claude_dir / "session1.jsonl")

    # Transport/storage scenarios do not depend on model downloads or inference.
    env = {key: value for key, value in os.environ.items() if not key.startswith("RECALL_")}
    env["HF_HOME"] = str(tmp / "hf")
    env["HF_HUB_CACHE"] = str(tmp / "hf" / "hub")
    env["HF_HUB_OFFLINE"] = "1"
    env["HOME"] = str(tmp)
    env["RECALL_DATA_DIR"] = str(data_dir)
    config_path = tmp / "config.toml"
    config_path.write_text(
        "[daemon]\nembed=false\n[cli]\nstatus_notices=false\n"
        + "".join(
            f"[sources.{source}]\nroots=[{json.dumps(str(source_root))}]\n"
            for source, source_root in (
                ("claude_code", claude_dir),
                ("codex", tmp / "empty-codex"),
                ("pi_agent", tmp / "empty-pi"),
                ("grok", tmp / "empty-grok"),
                ("kimi_code", tmp / "empty-kimi"),
            )
        )
    )
    env["RECALL_CONFIG_PATH"] = str(config_path)
    env["RECALL_DAEMON_EMBED"] = "false"
    env["RECALL_CONTEXT_MODE"] = "off"

    recall_bin = shutil.which("recall") or sys.argv[0]
    sock_path = data_dir / "recall.sock"
    pid_path = data_dir / "recall.pid"

    # Start daemon in foreground via subprocess (background would double-fork)
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_f = open(log_dir / "daemon.log", "w")  # noqa: SIM115
    stderr_f = open(log_dir / "daemon.err.log", "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [recall_bin, "daemon", "--verbose"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout_f,
        stderr=stderr_f,
        start_new_session=True,
    )

    # Wait for socket to appear
    waited = 0.0
    while waited < 10.0:
        if sock_path.exists():
            break
        time.sleep(0.2)
        waited += 0.2
    else:
        proc.kill()
        proc.wait()
        pytest.fail(f"daemon did not create socket within 10s at {sock_path}")

    # Verify daemon is accepting connections (socket file can exist before
    # the server is ready to accept)
    ready = False
    for _ in range(25):
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(str(sock_path))
            s.close()
            ready = True
            break
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            time.sleep(0.2)
    if not ready:
        proc.kill()
        proc.wait()
        pytest.fail("daemon socket exists but not accepting connections")

    yield {
        "tmp": tmp,
        "data_dir": data_dir,
        "sock_path": sock_path,
        "pid_path": pid_path,
        "proc": proc,
        "env": env,
        "recall_bin": recall_bin,
    }

    # Cleanup: stop daemon
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _rpc_call(
    sock_path: Path, method: str, params: dict | None = None, *, request_id: object = None
) -> dict:
    """Send a JSON-RPC request over the Unix socket and return the response."""
    if request_id is None:
        request_id = int(time.monotonic() * 1000)
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": request_id,
    }
    line = json.dumps(request, separators=(",", ":")) + "\n"

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(30.0)
    sock.connect(str(sock_path))
    sock.sendall(line.encode("utf-8"))

    buf = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line_bytes, buf = buf.split(b"\n", 1)
            response = json.loads(line_bytes)
            # Skip progress notifications
            if response.get("id") is None:
                continue
            sock.close()
            return response

    sock.close()
    raise RuntimeError("connection closed without response")


class TestRealDaemonRoundTrip:
    def test_method_not_found(self, daemon_env: dict) -> None:
        sock_path = daemon_env["sock_path"]
        resp = _rpc_call(sock_path, "recall.nonexistent", {})
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_search_missing_query(self, daemon_env: dict) -> None:
        sock_path = daemon_env["sock_path"]
        resp = _rpc_call(sock_path, "recall.search", {})
        assert "error" in resp
        assert resp["error"]["code"] == -32602
        assert "query" in resp["error"]["message"]


class TestCliToDaemonRoundTrip:
    """Run actual `recall` CLI commands as subprocesses against a real daemon.

    This exercises the full CLI→RPC client→Unix socket→RPC server→DB path.
    """

    def _run_recall(self, daemon_env: dict, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [daemon_env["recall_bin"], *args],
            env=daemon_env["env"],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_cli_index_and_search_json(self, daemon_env: dict) -> None:
        """recall index --json + recall search --json via real daemon."""
        # Index
        idx = self._run_recall(daemon_env, ["index", "--full", "--yes", "--no-embed", "--json"])
        assert idx.returncode == 0, f"index failed: {idx.stderr}"
        idx_data = json.loads(idx.stdout)
        assert "total" in idx_data
        assert "indexed" in idx_data

        # Search
        srch = self._run_recall(daemon_env, ["search", "Hello", "--mode", "keyword", "--json"])
        assert srch.returncode == 0, f"search failed: {srch.stderr}"
        srch_data = json.loads(srch.stdout)
        assert isinstance(srch_data, list)
        assert any(hit["content"] == "Hello" for hit in srch_data)

    def test_cli_list_json(self, daemon_env: dict) -> None:
        self._run_recall(daemon_env, ["index", "--full", "--yes", "--no-embed"])
        result = self._run_recall(daemon_env, ["list", "--json"])
        assert result.returncode == 0, f"list failed: {result.stderr}"
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        assert payload

    def test_cli_stats_json(self, daemon_env: dict) -> None:
        self._run_recall(daemon_env, ["index", "--full", "--yes", "--no-embed"])
        result = self._run_recall(daemon_env, ["stats", "--json"])
        assert result.returncode == 0, f"stats failed: {result.stderr}"
        payload = json.loads(result.stdout)
        assert "sessions" in payload

    def test_cli_daemon_status(self, daemon_env: dict) -> None:
        result = self._run_recall(daemon_env, ["daemon", "status", "--json"])
        assert result.returncode == 0, f"daemon status failed: {result.stderr}"
        payload = json.loads(result.stdout)
        assert "runtime_status" in payload

    def test_cli_validation_error(self, daemon_env: dict) -> None:
        result = self._run_recall(daemon_env, ["search", "--json"])
        assert result.returncode == 2
        payload = json.loads(result.stdout)
        assert payload["error"]["code"] == "VALIDATION"


class TestConcurrentReads:
    """Verify REQ-RPC-012: concurrent read requests are served in parallel.

    The unit tests in test_rpc_server.py inject delays to prove true
    parallelism. This integration test verifies the real daemon handles
    two concurrent socket requests without errors.
    """

    def test_parallel_stats_requests_succeed(self, daemon_env: dict) -> None:
        """Two concurrent recall.stats requests via socket both succeed."""
        sock_path = daemon_env["sock_path"]
        _rpc_call(sock_path, "recall.index", {"full": True, "embed": False})

        import concurrent.futures

        def do_stats():
            return _rpc_call(sock_path, "recall.stats", {})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(do_stats)
            f2 = pool.submit(do_stats)
            r1 = f1.result(timeout=10)
            r2 = f2.result(timeout=10)

        assert "error" not in r1, f"stats 1 failed: {r1}"
        assert "error" not in r2, f"stats 2 failed: {r2}"


@pytest.mark.parametrize("buffered_request", [False, True])
def test_quiet_follow_releases_subscription_when_peer_leaves(
    daemon_env: dict, buffered_request: bool
) -> None:
    """REQ-LIVE-011: EOF ends an idle follow, even with queued input behind it."""
    sock_path = daemon_env["sock_path"]
    indexed = _rpc_call(sock_path, "recall.index", {"embed": False})
    assert "error" not in indexed, indexed
    sessions = _rpc_call(sock_path, "recall.list")["result"]
    session_id = sessions[0]["id"]

    def wait_for_subscriptions(expected: int) -> None:
        deadline = time.monotonic() + 2
        while True:
            response = _rpc_call(sock_path, "recall.daemon_status")
            assert "error" not in response, response
            count = response["result"]["follow_subscriptions"]
            assert time.monotonic() < deadline, f"expected {expected} subscriptions, got {count}"
            if count == expected:
                return
            # There is no status subscription API; poll the condition with a bound.
            time.sleep(0.01)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(2)
        peer.connect(str(sock_path))
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "recall.show_follow",
            "params": {"session_id": session_id, "timeout": 30},
        }
        peer.sendall((json.dumps(request) + "\n").encode())
        wait_for_subscriptions(1)
        if buffered_request:
            peer.sendall(b'{"jsonrpc":"2.0","id":2,"method":"recall.stats"}\n')
            peer.shutdown(socket.SHUT_WR)
        else:
            peer.close()
        wait_for_subscriptions(0)
        if buffered_request:
            with peer.makefile("rb") as replies:
                response = json.loads(replies.readline())
            assert response["id"] == 2
            assert response["result"]["sessions"] >= 1


def test_pipelined_reads_receive_results_after_half_close(daemon_env: dict) -> None:
    sock_path = daemon_env["sock_path"]
    indexed = _rpc_call(sock_path, "recall.index", {"embed": False})
    assert "error" not in indexed, indexed
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(2)
        peer.connect(str(sock_path))
        peer.sendall(
            b'{"jsonrpc":"2.0","id":1,"method":"recall.stats"}\n'
            b'{"jsonrpc":"2.0","id":2,"method":"recall.stats"}\n'
        )
        peer.shutdown(socket.SHUT_WR)
        with peer.makefile("rb") as replies:
            responses = [json.loads(replies.readline()), json.loads(replies.readline())]
        assert [response["id"] for response in responses] == [1, 2]
        assert all(response["result"]["sessions"] >= 1 for response in responses)


@pytest.mark.parametrize("format_name", ["text", "json", "jsonl", "toon"])
def test_req_cli_005_keyword_index_lock_reaches_cli_in_every_format(
    daemon_env: dict, format_name: str
) -> None:
    from toon_format import decode

    sock = daemon_env["sock_path"]
    indexed = _rpc_call(sock, "recall.index", {"full": True, "embed": False})
    assert "error" not in indexed, indexed
    status = _rpc_call(sock, "recall.daemon_status")["result"]
    assert status["daemon_pid"] == daemon_env["proc"].pid
    assert _rpc_call(sock, "recall.stats")["result"]["sessions"] > 0
    sidecar = (daemon_env["data_dir"] / "recall.fts.sqlite").resolve()
    message = (
        "search failed: keyword index read failed: OperationalError: database is locked; "
        f"index: {sidecar}; retry after the lock is released"
    )
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sqlite3, sys\n"
                "conn = sqlite3.connect(sys.argv[1], isolation_level=None)\n"
                "conn.execute('BEGIN EXCLUSIVE')\n"
                "print('acquired', flush=True)\n"
                "sys.stdin.readline()\n"
                "conn.execute('ROLLBACK')\n"
                "conn.close()\n"
            ),
            str(sidecar),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdin is not None
    args = [
        daemon_env["recall_bin"],
        "search",
        "Hello",
        "--mode",
        "keyword",
        "--format",
        format_name,
    ]
    try:
        assert select.select([holder.stdout], [], [], 10)[0], "lock acquisition timed out"
        assert holder.stdout.readline() == "acquired\n"
        response = _rpc_call(
            sock,
            "recall.search",
            {"query": "Hello", "mode": "keyword"},
            request_id="keyword-lock-contract",
        )
        assert response == {
            "jsonrpc": "2.0",
            "id": "keyword-lock-contract",
            "error": {
                "code": -32000,
                "message": message,
                "data": {"kind": "keyword_index_locked", "retryable": True},
            },
        }
        result = subprocess.run(
            args, env=daemon_env["env"], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 1, (result.stdout, result.stderr)
        if format_name == "text":
            assert result.stdout == ""
            assert result.stderr == f"error: {message}\n"
        else:
            payload: Any = (
                decode(result.stdout) if format_name == "toon" else json.loads(result.stdout)
            )
            assert payload == {
                "error": {
                    "code": "LOCKED",
                    "message": message,
                    "details": {"kind": "keyword_index_locked", "retryable": True},
                }
            }
            assert payload["error"]["details"]["retryable"] is True
            assert result.stderr == ""
        held = _rpc_call(sock, "recall.daemon_status")["result"]["reconciliation"]
        assert held["keyword_search_ready"] is False
        assert held["keyword_publish_error"] is None
        assert "keyword index read failed" in held["keyword_repair_error"]
    finally:
        holder.communicate("release\n", timeout=10)
        assert holder.returncode == 0
    result = subprocess.run(args, env=daemon_env["env"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Hello" in result.stdout
    assert _rpc_call(sock, "recall.daemon_status")["result"]["daemon_pid"] == daemon_env["proc"].pid
