"""A scratch lane: two indexed transcripts and an `RpcServer` reading them.

The live-session handlers are only interesting against a real index — the SQL,
the derivations, and the freshness `stat` all have to agree — and every such
test needs the same three things: a data dir nothing else writes to, at least
two sessions so "only the one I asked for" is provable, and a way to append to
a transcript the way a running agent would.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.types import DaemonMode
from recall.services.rpc_server import RpcServer
from recall.services.watcher import LiveWatchRuntime, build_live_watch_runtime

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# The newest timestamp inside the fixtures is 2026-09-07T18:00Z; every installed
# transcript is stamped after that and inside live.idle_window (24h). A calendar
# date here aged out of the idle roster on hosted CI the next day.
LANE_NOW = datetime.now().replace(microsecond=0)

# Session ids the two fixture transcripts declare.
WATCHED_SESSION = "live-mid-tool"
UNWATCHED_SESSION = "live-end-turn"


@dataclass(frozen=True)
class Lane:
    """One scratch data dir, indexed, with a server over it."""

    server: RpcServer
    watched: Path
    unwatched: Path

    def append(self, path: Path, text: str, *, uuid: str) -> int:
        """Append one user message the way a live harness would, returning bytes added."""
        session_id = json.loads(path.read_text(encoding="utf-8").splitlines()[0])["sessionId"]
        line = (
            json.dumps(
                {
                    "parentUuid": None,
                    "isSidechain": False,
                    "type": "user",
                    "message": {"role": "user", "content": [{"type": "text", "text": text}]},
                    "uuid": uuid,
                    "timestamp": "2026-09-07T18:05:00Z",
                    "cwd": "/home/dev/project",
                    "sessionId": session_id,
                    "version": "2.1.80",
                    "gitBranch": "feat/live-agent-sessions",
                }
            )
            + "\n"
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        return len(line.encode("utf-8"))

    def watch_runtime(self, paths: list[Path]) -> LiveWatchRuntime:
        """The real watch runtime, its live set seeded with `paths`, never started.

        `index_session_now` reads `parsers` and settles `queue`; `_handle_live`
        reads `live_set`. No observer is started, so nothing but the test marks
        or promotes.
        """
        runtime = build_live_watch_runtime(
            config=self.server._config, observer_factory=_UnstartedObserver
        )
        runtime.live_set.seed(
            ((path, path.stat().st_mtime) for path in paths),
            monotonic_now=time.monotonic(),
        )
        return runtime


class _UnstartedObserver:
    """An observer the lane never starts or subscribes; any call is a test bug."""

    def schedule(
        self, handler: object, path: str, recursive: bool = False, event_filter: Any = None
    ) -> object:
        raise AssertionError("the lane watch runtime subscribes nothing")

    def unschedule(self, watch: object) -> None:
        raise AssertionError("the lane watch runtime subscribes nothing")

    def start(self) -> None:
        raise AssertionError("the lane watch runtime is never started")

    def stop(self) -> None:
        raise AssertionError("the lane watch runtime is never started")

    def join(self, timeout: float) -> None:
        raise AssertionError("the lane watch runtime is never started")


def lane_config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(mode=DaemonMode.POLL),
        cli=CliConfig(),
    )


def build_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lane:
    """Install both fixture transcripts under a scratch HOME, index, and serve them.

    A plain function rather than a fixture in a `conftest.py`: a second conftest
    under `tests/` shadows the root one for every sibling that does
    `import conftest`, which is how nine unrelated suites stopped collecting.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    projects = tmp_path / ".claude" / "projects" / "proj"
    projects.mkdir(parents=True)
    installed: list[Path] = []
    for fixture, name in (
        ("live_mid_tool.jsonl", "live.jsonl"),
        ("live_end_turn.jsonl", "other.jsonl"),
    ):
        path = projects / name
        shutil.copy(FIXTURES / "claude_code" / fixture, path)
        os.utime(path, (LANE_NOW.timestamp(), LANE_NOW.timestamp()))
        installed.append(path)
    server = RpcServer(lane_config(tmp_path))
    server._executor = ThreadPoolExecutor(max_workers=2)

    async def reconcile() -> None:
        for path in installed:
            await server.index_session_now(path)

    asyncio.run(reconcile())
    return Lane(server=server, watched=installed[0], unwatched=installed[1])
