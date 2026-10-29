from __future__ import annotations

import errno
import json
import os
import pwd
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SocketDaemon:
    env: dict[str, str] = field(repr=False)
    caller_cwd: Path
    root: Path
    socket_path: Path

    def cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(Path(sys.executable).with_name("recall")), *args],
            cwd=self.caller_cwd,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def rpc(self, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(10)
            connection.connect(str(self.socket_path))
            connection.sendall(
                (
                    json.dumps(
                        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
                    )
                    + "\n"
                ).encode()
            )
            with connection.makefile() as reader:
                for line in reader:
                    reply = json.loads(line)
                    if reply.get("id") == 1:
                        return reply
        raise AssertionError("daemon closed without a response")


@pytest.fixture
def socket_daemon(socket_dir: Path) -> Iterator[SocketDaemon]:
    caller_cwd = socket_dir / "caller"
    daemon_cwd = socket_dir / "daemon"
    root = caller_cwd / "relative-root"
    for directory in (caller_cwd, daemon_cwd, root):
        directory.mkdir()
    sources = ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
    config = socket_dir / "config.toml"
    config.write_text(
        "[daemon]\nembed=false\n[embedding.context]\nmode='off'\n"
        "[compaction]\nauto_trigger=false\n"
        + "".join(f"[sources.{source}]\nroots=['{root}/{source}']\n" for source in sources)
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("RECALL_")}
    env.update(
        RECALL_DATA_DIR=str(socket_dir / "data"),
        RECALL_CONFIG_PATH=str(config),
        PATH=os.pathsep.join((str(Path(sys.executable).parent), env["PATH"])),
    )
    daemon = SocketDaemon(env, caller_cwd, root, socket_dir / "data/recall.sock")
    # Keep the usage harvester inside this fixture as well as all five sources.
    launcher = (
        "import asyncio, os\n"
        "from pathlib import Path\n"
        "from recall.services import usage_harvest\n"
        "from recall.services.rpc_server import run_rpc_server\n"
        "empty = Path(os.environ['RECALL_DATA_DIR']).parent / 'empty-usage.jsonl'\n"
        "empty.write_text('')\n"
        "usage_harvest.default_grok_unified_log = lambda **kwargs: empty\n"
        "asyncio.run(run_rpc_server(watch=False, idle_timeout=120))\n"
    )
    with (socket_dir / "daemon.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", launcher],
            cwd=daemon_cwd,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                assert process.poll() is None, (socket_dir / "daemon.log").read_text()
                try:
                    reply = daemon.rpc("recall.daemon_status")
                    assert "result" in reply, (reply, (socket_dir / "daemon.log").read_text())
                    status = reply["result"]
                    assert status["daemon_pid"] == process.pid, (
                        reply,
                        (socket_dir / "daemon.log").read_text(),
                    )
                    break
                except (OSError, ValueError):
                    assert time.monotonic() < deadline, "daemon readiness deadline expired"
                    time.sleep(0.05)
            yield daemon
        finally:
            if process.poll() is None:
                assert os.getpgid(process.pid) == process.pid
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
            assert process.poll() is not None


@pytest.mark.parametrize("via_params", [False, True], ids=["flags", "params"])
def test_index_relative_root_across_working_directories(
    socket_daemon: SocketDaemon, via_params: bool
) -> None:
    daemon = socket_daemon
    options = (
        ["--params", '{"root":"relative-root"}'] if via_params else ["--root", "relative-root"]
    )
    args = ["index", *options, "--no-embed", "--no-progress", "--json"]
    preview = daemon.cli(*args, "--dry-run")
    # Exercise execution even when preview selected an incorrect path.
    source = daemon.root / "claude_code/project/session.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text((REPO / "tests/fixtures/claude_code/session1.jsonl").read_text())
    execution = daemon.cli(*args)

    assert preview.returncode == 0, preview.stderr
    assert execution.returncode == 0, (execution.stdout, execution.stderr)
    assert json.loads(preview.stdout)["request"]["root"] == str(daemon.root)
    assert json.loads(preview.stdout)["request"]["host"] == "relative-root"
    assert json.loads(execution.stdout)["failed"] == 0
    sessions = daemon.rpc("recall.list", {"limit": 10})["result"]
    assert len(sessions) == 1
    detail = daemon.rpc("recall.show", {"session_id": sessions[0]["id"]})["result"]
    assert detail["source_path"] == str(source)
    assert daemon.rpc("recall.stats")["result"]["sessions"] == 1


@pytest.mark.parametrize("explicit_host", [False, True])
@pytest.mark.parametrize(
    "kind", ["inaccessible-parent", "symlink-loop", "long-name", "unknown-user"]
)
def test_index_root_filesystem_errors_are_structured(
    socket_daemon: SocketDaemon, kind: str, explicit_host: bool
) -> None:
    daemon = socket_daemon
    parent = daemon.caller_cwd / "blocked-parent"
    if kind == "unknown-user":
        for suffix in range(32):
            username = f"recall-test-missing-user-{suffix}"
            try:
                pwd.getpwnam(username)
            except KeyError:
                root = f"~{username}/existing-root"
                break
        else:
            pytest.fail("no absent username found in 32 account lookup candidates")
        message = f"cannot expand root: {root}: Could not determine home directory."
    elif kind == "inaccessible-parent":
        parent.mkdir()
        root = parent / "root"
        root.mkdir()
        parent.chmod(0)
        reason = os.strerror(errno.EACCES)
        message = f"cannot access root: {root}: {reason}"
    elif kind == "symlink-loop":
        root = daemon.caller_cwd / "loop"
        root.symlink_to(root)
        reason = os.strerror(errno.ELOOP)
        message = f"cannot access root: {root}: {reason}"
    else:
        root = daemon.caller_cwd / ("x" * 256)
        reason = os.strerror(errno.ENAMETOOLONG)
        message = f"cannot access root: {root}: {reason}"

    params: dict[str, object] = {"root": str(root)}
    if explicit_host:
        params["host"] = "explicit-host"
    try:
        before = daemon.rpc("recall.stats")["result"]
        results = [
            (
                format_name,
                daemon.cli(
                    "index",
                    "--params",
                    json.dumps(params),
                    "--format",
                    format_name,
                    "--no-embed",
                    "--no-progress",
                    *(["--dry-run"] if dry_run else []),
                ),
            )
            for dry_run in (True, False)
            for format_name in ("json", "text")
        ]
        reply = daemon.rpc("recall.index", params)
        for format_name, result in results:
            assert result.returncode == 2, (result.stdout, result.stderr)
            if format_name == "json":
                assert json.loads(result.stdout) == {
                    "error": {"code": "VALIDATION", "message": message}
                }
                assert result.stderr == ""
            else:
                assert result.stdout == ""
                assert result.stderr == f"error: {message}\n"
        assert reply == {"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": message}}
        assert daemon.rpc("recall.stats")["result"] == before
    finally:
        if kind == "inaccessible-parent":
            parent.chmod(0o700)
