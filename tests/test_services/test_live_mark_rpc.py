"""`recall.live_mark` — the daemon's one write path for a writer-pid mark (REQ-LIVE-008).

The hook that calls this fires at `SessionStart`, before anything has indexed
the transcript it names, so these tests cover the write standing alone as well
as what it changes on the next `recall.live` read.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from lane_harness import build_lane
from recall.core.rpc_types import RpcError
from recall.core.types import default_session_host


def _run(coro):
    return asyncio.run(coro)


def _reaped_pid() -> int:
    """A pid that has exited and been waited on, so `kill(pid, 0)` finds nothing."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _mark(server: Any, **params: Any) -> dict[str, Any]:
    return _run(server._handle_live_mark(params, None))


def _live_rows(server: Any, **params: Any) -> list[Any]:
    """Rows a `recall live --all` read returns, with no live set to promote any."""
    server._watch_runtime = None
    result = _run(server._handle_live({"all": True, "limit": 50, **params}, None))
    return list(result["sessions"])


def test_a_mark_reports_the_row_it_wrote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lane = build_lane(tmp_path, monkeypatch)
    try:
        result = _mark(lane.server, session="live-mid-tool", pid=os.getpid())
    finally:
        _run(lane.server.stop())

    assert result["marked"] is True
    assert result["source"] == "claude_code"
    assert result["source_session_id"] == "live-mid-tool"
    assert result["host"] == default_session_host()
    assert result["pid"] == os.getpid()


def test_a_mark_for_a_session_no_index_has_seen_is_still_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook fires at SessionStart; the first index pass may be minutes away."""
    lane = build_lane(tmp_path, monkeypatch)
    try:
        result = _mark(lane.server, session="not-indexed-yet", pid=os.getpid())
    finally:
        _run(lane.server.stop())

    assert result["marked"] is True
    assert result["source_session_id"] == "not-indexed-yet"


def test_re_marking_a_session_replaces_the_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed harness re-runs its hook with a new pid; the old one is not a second row."""
    lane = build_lane(tmp_path, monkeypatch)
    try:
        # A live process is scheduler evidence, but not proof of a new transcript write.
        quiet_at = time.time() - 600
        os.utime(lane.watched, (quiet_at, quiet_at))
        _run(lane.server.index_session_now(lane.watched))
        _mark(lane.server, session="live-mid-tool", pid=_reaped_pid())
        _mark(lane.server, session="live-mid-tool", pid=os.getpid())
        active_paths = lane.server._active_source_paths(lane.server._config)
        rows = _live_rows(lane.server)
    finally:
        _run(lane.server.stop())

    marked = [row for row in rows if row.source_session_id == "live-mid-tool"]
    assert len(marked) == 1
    assert marked[0].writer_pid == os.getpid()
    assert marked[0].liveness.value == "idle"
    assert str(lane.watched) in active_paths


def test_a_dead_marked_pid_ends_the_session_on_the_next_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = build_lane(tmp_path, monkeypatch)
    try:
        _mark(lane.server, session="live-mid-tool", pid=_reaped_pid())
        rows = _live_rows(lane.server)
    finally:
        _run(lane.server.stop())

    marked = [row for row in rows if row.source_session_id == "live-mid-tool"]
    assert len(marked) == 1
    assert marked[0].liveness.value == "ended"


def test_a_missing_session_id_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lane = build_lane(tmp_path, monkeypatch)
    try:
        with pytest.raises(RpcError) as refusal:
            _mark(lane.server, pid=os.getpid())
        assert "session is required" in refusal.value.message
    finally:
        _run(lane.server.stop())


def test_a_non_positive_pid_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mark naming no process would probe the caller's own process group."""
    lane = build_lane(tmp_path, monkeypatch)
    try:
        with pytest.raises(RpcError) as refusal:
            _mark(lane.server, session="live-mid-tool", pid=0)
        assert "pid must be positive" in refusal.value.message
    finally:
        _run(lane.server.stop())


def test_an_unknown_source_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lane = build_lane(tmp_path, monkeypatch)
    try:
        with pytest.raises(RpcError) as refusal:
            _mark(lane.server, session="live-mid-tool", pid=os.getpid(), source="nope")
        assert "unsupported source" in refusal.value.message
    finally:
        _run(lane.server.stop())
