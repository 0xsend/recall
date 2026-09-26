"""Reconciliation contracts exercised through an isolated daemon's Unix socket."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True)
class DaemonLane:
    root: Path
    config: Path
    socket: Path

    def transcript(self, name: str, text: str) -> Path:
        path = self.root / f"rollout-{name}.jsonl"
        records = [
            {"type": "session_meta", "payload": {"id": name, "cwd": "/owned/project"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": text}},
        ]
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return path

    def rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        with socket.socket(socket.AF_UNIX) as client:
            client.settimeout(2.0)
            client.connect(str(self.socket))
            client.sendall(
                (
                    json.dumps(
                        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
                    )
                    + "\n"
                ).encode()
            )
            with client.makefile("rb") as stream:
                for _ in range(1024):
                    line = stream.readline(4 * 1024 * 1024)
                    assert line, "daemon closed the response stream"
                    response = json.loads(line)
                    if response.get("id") == 1:
                        assert "error" not in response, response.get("error")
                        return response["result"]
            raise AssertionError("daemon exceeded the bounded response notification count")

    def wait_for(self, predicate: Callable[[], bool], *, description: str) -> None:
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        raise AssertionError(f"daemon did not converge: {description}")

    def session_id(self, name: str) -> str | None:
        rows = self.rpc("recall.list", {"limit": 100})
        return next((row["id"] for row in rows if row["source_session_id"] == name), None)

    @contextmanager
    def running(self) -> Iterator[None]:
        env = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "SYSTEMROOT", "TMPDIR", "LANG", "LC_ALL", "PYTHONUTF8"}
        }
        env.update(
            HOME=str(self.config.parent),
            RECALL_CONFIG_PATH=str(self.config),
            RECALL_DATA_DIR=str(self.socket.parent),
        )
        with (self.config.parent / "daemon.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import faulthandler, signal; "
                    "signal.signal(signal.SIGUSR1, "
                    "lambda *_: faulthandler.cancel_dump_traceback_later()); "
                    "faulthandler.dump_traceback_later(10, repeat=True); "
                    "from recall.cli.app import app; app()",
                    "daemon",
                    "--mode",
                    "poll",
                ],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )
            try:

                def ready() -> bool:
                    assert process.poll() is None, "owned daemon exited during startup"
                    try:
                        self.rpc("recall.daemon_status")
                    except (OSError, AssertionError):
                        return False
                    return True

                self.wait_for(ready, description="RPC/status readiness")
                # Capture blocked startup stacks before the unchanged 12s deadline.
                process.send_signal(signal.SIGUSR1)
                yield
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


@pytest.fixture
def daemon_lane(tmp_path: Path) -> Iterator[DaemonLane]:
    # Use a short socket path even when the pytest root lives in macOS's long TMPDIR.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="rec", dir="/tmp") as short:
        root = tmp_path / "transcripts"
        root.mkdir()
        config = tmp_path / "config.toml"
        config.write_text(
            "[daemon]\nmode = 'poll'\nembed = false\ninterval = 1\n"
            "live_discovery_interval = 1\ndebounce = 1\n"
            "[embedding.context]\nmode = 'off'\n"
            "[compaction]\nauto_trigger = false\n"
            "[fts]\nbackend = 'sqlite_sidecar'\n"
            f"[sources.codex]\nroots = [{json.dumps(str(root))}]\n"
            "[sources.claude_code]\nroots = []\n"
            "[sources.pi_agent]\nroots = []\n"
            "[sources.grok]\nroots = []\n"
            "[sources.kimi_code]\nroots = []\n"
        )
        yield DaemonLane(root, config, Path(short) / "recall.sock")


def test_poll_reconciles_startup_and_later_old_imports_without_manual_index(
    daemon_lane: DaemonLane,
) -> None:
    """REQ-RECON-001/002: root membership, not recency, determines coverage."""
    first = daemon_lane.transcript("before-start", "historical one")
    os.utime(first, (1, 1))
    with daemon_lane.running():
        daemon_lane.wait_for(
            lambda: daemon_lane.session_id("before-start") is not None,
            description="startup historical import",
        )
        second = daemon_lane.transcript("after-start", "historical two")
        os.utime(second, (1, 1))
        daemon_lane.wait_for(
            lambda: daemon_lane.session_id("after-start") is not None,
            description="periodic historical import",
        )
        result = daemon_lane.rpc("recall.stats")
        assert result["sessions"] == 2
        assert result["messages"] == 2


def test_maintenance_pause_survives_restart_and_read_clients(
    daemon_lane: DaemonLane,
) -> None:
    """REQ-RECON-009: restart and read-client activity preserve maintenance intent."""
    with daemon_lane.running():
        paused = daemon_lane.rpc("recall.daemon_pause")
        assert paused["paused"] is True
        daemon_lane.transcript("paused-import", "held until resumed")
    with daemon_lane.running():
        status = daemon_lane.rpc("recall.daemon_status")
        assert status["reconciliation"]["paused"] is True
        assert daemon_lane.session_id("paused-import") is None
        resumed = daemon_lane.rpc("recall.daemon_resume")
        assert resumed["paused"] is False
        daemon_lane.wait_for(
            lambda: daemon_lane.session_id("paused-import") is not None,
            description="resume drains held work",
        )


def test_same_size_rewrite_is_stale_before_reconciliation_and_resets_cursor(
    daemon_lane: DaemonLane,
) -> None:
    """REQ-RECON-006/007: equal sizes cannot prove freshness or a compatible tail."""
    path = daemon_lane.transcript("rewrite", "old question")
    with daemon_lane.running():
        daemon_lane.wait_for(
            lambda: daemon_lane.session_id("rewrite") is not None,
            description="initial session",
        )
        session_id = daemon_lane.session_id("rewrite")
        original = daemon_lane.rpc("recall.show", {"session_id": session_id, "tail": 5})
        daemon_lane.rpc("recall.daemon_pause")
        stat = path.stat()
        path.write_bytes(path.read_bytes().replace(b"old question", b"new question"))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        stale = daemon_lane.rpc("recall.show", {"session_id": session_id, "tail": 5})
        assert stale["freshness"]["current"] is False
        daemon_lane.rpc("recall.daemon_resume")
        daemon_lane.wait_for(
            lambda: (
                daemon_lane.rpc("recall.show", {"session_id": session_id})["messages"][0]["content"]
                == "new question"
            ),
            description="same-size source rewrite",
        )
        delta = daemon_lane.rpc(
            "recall.show", {"session_id": session_id, "after": original["cursor"]}
        )
        assert delta["cursor_reset"] is True
        recovered = daemon_lane.rpc("recall.show", {"session_id": session_id, "tail": 5})
        assert recovered["messages"][0]["content"] == "new question"


def test_poll_refreshes_known_active_paths_between_full_inventory_scans(
    daemon_lane: DaemonLane,
) -> None:
    """A roster beyond the watch budget stays fresh without notifications or --fresh."""
    daemon_lane.config.write_text(
        daemon_lane.config.read_text().replace("interval = 1\n", "interval = 30\n")
    )
    paths = [daemon_lane.transcript(f"poll-{index:03}", "initial") for index in range(80)]
    with daemon_lane.running():
        daemon_lane.wait_for(
            lambda: daemon_lane.rpc("recall.stats")["sessions"] == len(paths),
            description="initial active roster",
        )
        assert daemon_lane.rpc("recall.daemon_status")["reconciliation"]["catalog_scan_complete"]
        for index in (0, 31, 64, 79):
            with paths[index].open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "type": "event_msg",
                            "payload": {
                                "type": "user_message",
                                "message": f"between-scans-{index}",
                            },
                        }
                    )
                    + "\n"
                )
        daemon_lane.wait_for(
            lambda: all(
                any(
                    row["content"] == f"between-scans-{index}"
                    for row in daemon_lane.rpc("recall.show", {"session_id": f"poll-{index:03}"})[
                        "messages"
                    ]
                )
                for index in (0, 31, 64, 79)
            ),
            description="all writes before the next full inventory scan",
        )
        assert daemon_lane.rpc("recall.stats")["messages"] == 84
