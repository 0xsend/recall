"""Unavailable context backends refuse startup and selected model work."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import pytest

SOURCES = ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
BACKENDS = [("llm-local", "recall[mlx]"), ("llm-remote", "recall[anthropic]"), ("llm-codex", "CLI")]
# Missing optional packages are controlled at the import boundary, not by
# replacing the product's backend factory or availability probe. This also
# prevents an installed MLX extra from downloading a model on a developer host.
CLI = [
    sys.executable,
    "-c",
    "import sys; sys.modules['mlx_lm'] = None; sys.modules['anthropic'] = None; "
    "from recall.cli import main; main()",
]


def _config(mode: str, *, embed: bool = False, fallback: str = "template") -> str:
    return (
        f"[daemon]\nembed = {str(embed).lower()}\nidle_timeout = 3600\n"
        f'[embedding.context]\nmode = "{mode}"\nfallback = "{fallback}"\n'
        'executable = "missing-recall-context-cli"\n'
        + "".join(f"[sources.{source}]\nroots = []\n" for source in SOURCES)
    )


def _rpc(root: Path, method: str) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(5)
        connection.connect(str(root / "recall.sock"))
        request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": {}}
        connection.sendall((json.dumps(request) + "\n").encode())
        with connection.makefile() as stream:
            return json.loads(stream.readline())


@contextmanager
def _daemon(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: str,
    args: list[str],
) -> Iterator[subprocess.Popen[str]]:
    (root / "config.toml").write_text(config)
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(root / "config.toml"))
    monkeypatch.setenv("RECALL_DATA_DIR", str(root))
    monkeypatch.setenv("RECALL_DB_PATH", str(root / "recall.duckdb"))
    monkeypatch.setenv("RECALL_LOCK_PATH", str(root / "recall.lock"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    (root / "logs").mkdir()
    with (
        (root / "logs/daemon.log").open("w") as stdout,
        (root / "logs/daemon.err.log").open("w") as stderr,
    ):
        process = subprocess.Popen([*CLI, "daemon", *args], stdout=stdout, stderr=stderr, text=True)
        try:
            deadline = time.monotonic() + 15
            while process.poll() is None:
                try:
                    status = _rpc(root, "recall.daemon_status").get("result", {})
                    if status.get("daemon_pid") == process.pid and status.get(
                        "reconciliation", {}
                    ).get("raw_indexing_ready"):
                        break
                except (OSError, ValueError):
                    pass
                assert time.monotonic() < deadline, (root / "logs/daemon.err.log").read_text()
                time.sleep(0.05)
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _assert_actionable(message: str, mode: str, remediation: str) -> None:
    assert mode in message
    assert "unavailable" in message
    assert remediation in message
    assert "template" in message and "off" in message


@pytest.mark.parametrize(("mode", "remediation"), BACKENDS)
@pytest.mark.parametrize(
    ("args", "embed", "fallback"),
    [
        (["--mode", "watch"], False, "template"),
        (["--mode", "watch"], True, "off"),
        (["--mode", "poll"], False, "error"),
        (["--once"], False, "template"),
        (["--once"], True, "error"),
    ],
)
def test_daemon_refuses_unavailable_context_before_ready(
    socket_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    remediation: str,
    args: list[str],
    embed: bool,
    fallback: str,
) -> None:
    with _daemon(socket_dir, monkeypatch, _config(mode, embed=embed, fallback=fallback), args) as p:
        message = (socket_dir / "logs/daemon.err.log").read_text()
        assert p.poll() not in (None, 0), message
        _assert_actionable(message, mode, remediation)
        assert "ERROR recall." in message
        assert not (socket_dir / "recall.sock").exists()
        assert not (socket_dir / "recall.duckdb").exists()


@pytest.mark.parametrize(("mode", "remediation"), BACKENDS)
@pytest.mark.parametrize("selection", ["config", "override"])
def test_index_refuses_unavailable_context_before_generation(
    socket_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    remediation: str,
    selection: str,
) -> None:
    alternate = socket_dir / "import-home"
    source = alternate / ".claude/projects/example"
    source.mkdir(parents=True)
    message = {"type": "user", "message": {"role": "user", "content": "Context " * 20}}
    (source / "incoming.jsonl").write_text(json.dumps(message) + "\n")
    with _daemon(socket_dir, monkeypatch, _config("off"), ["--mode", "watch"]) as p:
        assert p.poll() is None, (socket_dir / "logs/daemon.err.log").read_text()
        args = ["index", "--no-embed", "--root", str(alternate), "--source", "claude-code"]
        if selection == "config":
            (socket_dir / "config.toml").write_text(_config(mode))
        else:
            args.extend(["--context", mode])
        result = subprocess.run([*CLI, *args], capture_output=True, text=True, timeout=15)
        assert result.returncode != 0, result.stdout
        _assert_actionable(result.stdout + result.stderr, mode, remediation)
        stats = _rpc(socket_dir, "recall.stats")
        assert "error" not in stats
    with duckdb.connect(str(socket_dir / "recall.duckdb"), read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (0,)
        assert conn.execute(
            "SELECT COUNT(*) FROM message_state WHERE context_mode = ?", [mode]
        ).fetchone() == (0,)


@pytest.mark.parametrize("mode", ["off", "template", "llm-codex"])
def test_available_context_allows_startup_and_empty_index(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config = _config(mode).replace("missing-recall-context-cli", sys.executable)
    with _daemon(socket_dir, monkeypatch, config, ["--mode", "watch"]) as p:
        assert p.poll() is None, (socket_dir / "logs/daemon.err.log").read_text()
        result = subprocess.run(
            [*CLI, "index", "--no-embed", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["total"] == 0


@pytest.mark.parametrize(("mode", "remediation"), BACKENDS)
def test_empty_runs_need_no_context_backend(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, mode: str, remediation: str
) -> None:
    with _daemon(socket_dir, monkeypatch, _config("off"), ["--mode", "watch"]) as p:
        assert p.poll() is None
        (socket_dir / "config.toml").write_text(_config(mode))
        for args in (
            ["index", "--no-embed"],
            ["index", "--recompute-context", "--no-embed"],
            ["daemon", "--once", "--no-embed"],
        ):
            result = subprocess.run([*CLI, *args], capture_output=True, text=True, timeout=15)
            assert result.returncode == 0, result.stdout + result.stderr
        assert not list(socket_dir.glob("snapshots/*"))
        stats = _rpc(socket_dir, "recall.stats")
        assert "error" not in stats
