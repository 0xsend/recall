"""Integration test: CLI → real daemon → DB round-trip over Unix socket.

Starts a real RPC daemon in a subprocess, sends requests over the socket,
and verifies correct responses. This exercises the full RPC path that
in-process testing skips.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from conftest import _can_acquire_duckdb_lock

pytestmark = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)


def _short_tmp() -> Path:
    """Return a short temp dir that won't exceed Unix socket path limits."""
    return Path(tempfile.mkdtemp(prefix="rpc"))


@pytest.fixture
def daemon_env():
    """Set up an isolated environment and start a real daemon."""
    tmp = _short_tmp()
    data_dir = tmp / "data"
    data_dir.mkdir(parents=True)

    # Install a fixture session
    claude_dir = tmp / ".claude" / "projects" / "p1"
    claude_dir.mkdir(parents=True)
    fixture_src = (
        Path(__file__).resolve().parents[2] / "fixtures" / "claude_code" / "session1.jsonl"
    )
    if fixture_src.exists():
        shutil.copy(fixture_src, claude_dir / "session1.jsonl")

    env = os.environ.copy()
    # Forward HF cache so the daemon does not redownload the embedding model
    # after HOME is redirected to the isolated temp directory.
    real_hf_cache = os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub")
    real_hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    env["HF_HUB_CACHE"] = real_hf_cache
    env["HF_HOME"] = real_hf_home
    env["HOME"] = str(tmp)
    env["RECALL_DATA_DIR"] = str(data_dir)

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
    shutil.rmtree(tmp, ignore_errors=True)


def _rpc_call(sock_path: Path, method: str, params: dict | None = None) -> dict:
    """Send a JSON-RPC request over the Unix socket and return the response."""
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
        idx = self._run_recall(daemon_env, ["index", "--full", "--yes", "--json"])
        assert idx.returncode == 0, f"index failed: {idx.stderr}"
        idx_data = json.loads(idx.stdout)
        assert "total" in idx_data
        assert "indexed" in idx_data

        # Search
        srch = self._run_recall(daemon_env, ["search", "git", "--json"])
        assert srch.returncode == 0, f"search failed: {srch.stderr}"
        srch_data = json.loads(srch.stdout)
        assert isinstance(srch_data, list)

    def test_cli_list_json(self, daemon_env: dict) -> None:
        self._run_recall(daemon_env, ["index", "--full", "--yes"])
        result = self._run_recall(daemon_env, ["list", "--json"])
        assert result.returncode == 0, f"list failed: {result.stderr}"
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        assert payload

    def test_cli_stats_json(self, daemon_env: dict) -> None:
        self._run_recall(daemon_env, ["index", "--full", "--yes"])
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
        _rpc_call(sock_path, "recall.index", {"full": True})

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
