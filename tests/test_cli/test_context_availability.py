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


def _incoming(root: Path) -> Path:
    alternate = root / "import-home"
    source = alternate / ".claude/projects/example"
    source.mkdir(parents=True)
    (source / "incoming.jsonl").write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "Context " * 20}}) + "\n"
    )
    return alternate


def _repair_codex(executable: Path) -> None:
    # An executable boundary double; this exercises subprocess generation and
    # publication, not provider authentication or actual Codex inference.
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "sys.stdin.read()\n"
        "pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_text('Recovered context')\n"
    )
    executable.chmod(0o755)


@pytest.mark.parametrize("restart", [False, True])
def test_refused_index_retries_pending_context_after_raw_commit(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, restart: bool
) -> None:
    root = socket_dir
    alternate = _incoming(root)
    executable = root / "context-codex"
    config = _config("off").replace("missing-recall-context-cli", str(executable))
    args = [
        "index",
        "--no-embed",
        "--root",
        str(alternate),
        "--source",
        "claude-code",
        "--context",
        "llm-codex",
        "--json",
    ]

    def request() -> subprocess.CompletedProcess[str]:
        return subprocess.run([*CLI, *args], capture_output=True, text=True, timeout=15)

    def retry_and_repair() -> None:
        # An exempt deferred LLM cycle does not consume the outstanding obligation.
        (root / "config.toml").write_text(config.replace('mode = "off"', 'mode = "llm-codex"'))
        exempt = subprocess.run(
            [*CLI, "daemon", "--once", "--no-embed", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert exempt.returncode == 0, exempt.stdout + exempt.stderr
        second = request()
        assert second.returncode == 1, second.stdout + second.stderr
        _assert_actionable(second.stdout + second.stderr, "llm-codex", "CLI")
        _repair_codex(executable)
        repaired = request()
        assert repaired.returncode == 0, repaired.stdout + repaired.stderr
        assert json.loads(repaired.stdout)["context_messages"] == 1
        replay = request()
        assert replay.returncode == 0, replay.stdout + replay.stderr
        assert json.loads(replay.stdout)["context_input_tokens"] == 0

    with _daemon(root, monkeypatch, config, ["--mode", "watch"]) as process:
        assert process.poll() is None
        first = request()
        assert first.returncode == 1, first.stdout + first.stderr
        if not restart:
            retry_and_repair()
    if restart:
        (root / "logs").rename(root / "first-logs")
        with _daemon(root, monkeypatch, config, ["--mode", "watch"]) as process:
            assert process.poll() is None
            retry_and_repair()
    with duckdb.connect(str(root / "recall.duckdb"), read_only=True) as conn:
        assert conn.execute("SELECT context_mode, context_text FROM message_state").fetchall() == [
            ("llm-codex", "[Recovered context] ")
        ]
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (0,)


@pytest.mark.parametrize("candidate", ["directory", "non-executable"])
def test_codex_unexecutable_absolute_candidate_refuses_startup(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, candidate: str
) -> None:
    executable = socket_dir / "context-codex"
    if candidate == "directory":
        executable.mkdir()
    else:
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o644)
    config = _config("llm-codex").replace("missing-recall-context-cli", str(executable))
    with _daemon(socket_dir, monkeypatch, config, ["--mode", "watch"]) as process:
        assert process.poll() not in (None, 0)
        message = (socket_dir / "logs/daemon.err.log").read_text()
        _assert_actionable(message, "llm-codex", "executable")
        assert not (socket_dir / "recall.duckdb").exists()


@pytest.mark.parametrize("scheduler", ["systemd", "launchd"])
def test_context_refusal_stops_scheduler_and_recovers(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, scheduler: str
) -> None:
    import configparser
    import plistlib

    from recall.cli.status_notices import _startup_refusal_notice
    from recall.core.config import AppConfig
    from recall.services.daemon import _launchd_plist_contents, _write_systemd_units
    from recall.services.self_repair import read_refusal_marker

    root = socket_dir
    config_text = _config("llm-codex")
    with _daemon(root, monkeypatch, config_text, ["--mode", "watch"]) as process:
        config = AppConfig.load()
        if scheduler == "systemd":
            service = root / "recall.service"
            _write_systemd_units(
                config, "scratch-recall daemon --mode watch", service, root / "recall.timer"
            )
            unit = configparser.ConfigParser(interpolation=None, strict=False)
            unit.read(service)
            assert unit["Service"]["Restart"] == "always"
            assert str(process.poll()) in unit["Service"]["RestartPreventExitStatus"].split()
        else:
            plist = plistlib.loads(
                _launchd_plist_contents(config, "scratch-recall daemon --mode watch").encode()
            )
            conditions = plist["KeepAlive"]["PathState"]
            # The generated definition stops keeping this failed job alive.
            assert all(Path(path).exists() != expected for path, expected in conditions.items())
        refusal = read_refusal_marker(root)
        assert refusal is not None
        _assert_actionable(refusal, "llm-codex", "CLI")
        notice = _startup_refusal_notice({"startup_refusal": refusal})
        assert notice is not None and "llm-codex" in notice
        assert process.returncode == 3
    (root / "logs").rename(root / "refused-logs")
    # A manual start after repairing config clears the existing refusal marker.
    with _daemon(root, monkeypatch, _config("off"), ["--mode", "watch"]) as process:
        assert process.poll() is None
        assert read_refusal_marker(root) is None
        status = _rpc(root, "recall.daemon_status")["result"]
        assert status["startup_refusal"] is None
        assert _startup_refusal_notice(status) is None
        if scheduler == "launchd":
            assert all(Path(path).exists() == expected for path, expected in conditions.items())


@pytest.mark.parametrize("structured", [False, True])
def test_once_context_refusal_reports_committed_raw_progress(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, structured: bool
) -> None:
    root = socket_dir
    config = (
        _config("off")
        .replace(
            "[sources.claude_code]\nroots = []",
            f'[sources.claude_code]\nroots = ["{root / "import-home/.claude/projects"}"]',
        )
        .replace("embed = false", "embed = false\nembed_idle_session = 1")
    )
    with _daemon(root, monkeypatch, config, ["--mode", "watch"]) as process:
        assert process.poll() is None
        alternate = _incoming(root)
        incoming = alternate / ".claude/projects/example/incoming.jsonl"
        os.utime(incoming, (time.time() - 60, time.time() - 60))
        (root / "config.toml").write_text(config.replace('mode = "off"', 'mode = "llm-codex"'))
        args = ["daemon", "--once", "--embed"] + (
            ["--json"] if structured else ["--format", "text"]
        )
        result = subprocess.run([*CLI, *args], capture_output=True, text=True, timeout=15)
        assert result.returncode == 1, result.stdout + result.stderr
        message = result.stderr
        if structured:
            error = json.loads(result.stdout)["error"]
            assert error["code"] == "RUNTIME"
            assert error["details"]["raw_history_committed"] is True
            assert error["details"]["enrichment"] == "refused"
            assert error["details"]["retryable"] is True
            assert error["details"]["index_summary"]["indexed"] == 1
            message = error["message"]
        assert "raw history" in message.lower() and "committed" in message
        assert "enrichment" in message and "refused" in message and "retry" in message
        stats = _rpc(root, "recall.stats")["result"]
        assert stats["sessions"] == 1
    with duckdb.connect(str(root / "recall.duckdb"), read_only=True) as conn:
        from recall.services.runtime_state import load_runtime_status_from_conn

        status = load_runtime_status_from_conn(conn)
        assert status.last_run_kind is not None and status.last_run_kind.value == "daemon-once"
        assert status.last_failure_at is not None
        assert (
            status.last_successful_at is None or status.last_successful_at < status.last_failure_at
        )
        assert (
            status.last_failure_message is not None and "enrichment" in status.last_failure_message
        )
