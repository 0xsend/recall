"""The watch path persists tail facts, not just the full-index path (REQ-LIVE-005/006).

`recall live` reads turn state from `tool_results` and `session_stop_markers`.
Every live session reaches the database through the *daemon*, so a write path
that drops those rows makes the turn state wrong in exactly the case the
feature exists for: a finished turn keeps reporting `working` on a tool that
returned minutes ago, and `awaiting_input` is unreachable. Found by the U18 bug
bash, which observed a completed session reporting `running_tool.name = Bash`
indefinitely while `recall index --full` on the same transcript reported
`awaiting_input`.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
    SourceConfig,
)
from recall.core.types import DaemonMode, default_session_host
from recall.parsers import all_parsers
from recall.services.live import Liveness, TurnPhase, live_view
from recall.services.watcher import _resolve_parser_for_path, index_single_session

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NOW = datetime(2026, 9, 10, 12, 0, 0)
IDLE_WINDOW = 86400.0


def _config(tmp_path: Path, lane: Path) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(mode=DaemonMode.WATCH),
        cli=CliConfig(),
        sources={
            "claude_code": SourceConfig(roots=(lane,)),
            "codex": SourceConfig(roots=()),
            "pi_agent": SourceConfig(roots=()),
            "grok": SourceConfig(roots=()),
            "kimi_code": SourceConfig(roots=()),
        },
    )


def _lane(tmp_path: Path, fixture_name: str) -> tuple[AppConfig, Path]:
    lane = tmp_path / "lane"
    (lane / "proj").mkdir(parents=True)
    transcript = lane / "proj" / "live.jsonl"
    shutil.copy(FIXTURES / "claude_code" / fixture_name, transcript)
    # Stamp the transcript at the tests' fixed `now`; `last_activity_at` is the
    # max of `ended_at` and this mtime, and a copy stamped at real wall clock
    # falls outside the idle window.
    os.utime(transcript, (NOW.timestamp(), NOW.timestamp()))
    config = _config(tmp_path, lane)
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    return config, transcript


def _index_as_the_daemon_does(config: AppConfig, transcript: Path) -> None:
    """Drive the same entry point `index_session_now` and the drain loop use."""
    from recall.db import connect

    parsers = all_parsers(config.sources)
    parser = _resolve_parser_for_path(str(transcript), parsers)
    assert parser is not None
    conn = connect(config)
    try:
        index_single_session(transcript, parser, config, conn=conn)
    finally:
        conn.close()


def _counts(config: AppConfig) -> dict[str, int]:
    conn = duckdb.connect(str(config.db_path), read_only=True)
    try:
        counts: dict[str, int] = {}
        for table in ("tool_calls", "tool_results", "session_stop_markers"):
            row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
            assert row is not None
            counts[table] = int(row[0])
        return counts
    finally:
        conn.close()


def _rows(config: AppConfig) -> list:
    conn = duckdb.connect(str(config.db_path), read_only=True)
    try:
        return live_view(
            conn=conn,
            watched_paths=[],
            include_idle=True,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=default_session_host(),
        )
    finally:
        conn.close()


@pytest.mark.parametrize("seed_history", [False, True])
@pytest.mark.parametrize("split_after", [1, 2])
def test_codex_lifecycle_survives_split_start_and_abort_only_appends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed_history: bool, split_after: int
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    lane = tmp_path / "codex"
    lane.mkdir()
    config = replace(
        _config(tmp_path, lane),
        sources={
            "claude_code": SourceConfig(roots=()),
            "codex": SourceConfig(roots=(lane,)),
            "pi_agent": SourceConfig(roots=()),
            "grok": SourceConfig(roots=()),
            "kimi_code": SourceConfig(roots=()),
        },
    )
    config.db_path.parent.mkdir(parents=True)
    transcript = lane / "rollout.jsonl"

    def append(records: list[dict]) -> None:
        with transcript.open("a") as handle:
            handle.write("".join(json.dumps(record) + "\n" for record in records))
        os.utime(transcript, (NOW.timestamp(), NOW.timestamp()))
        _index_as_the_daemon_does(config, transcript)

    header = {"type": "session_meta", "payload": {"id": "split-task"}}
    prior = [
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "Prior answer"}},
        {"type": "event_msg", "payload": {"type": "task_complete"}},
    ]
    append([header, *prior] if seed_history else [header])
    turn = [
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "next"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "Inspect"}},
        {"type": "event_msg", "payload": {"type": "agent_reasoning", "text": "Inspecting"}},
    ]
    append(turn[:split_after])
    append(turn[split_after:])
    assert _rows(config)[0].turn.state is TurnPhase.WORKING
    assert _rows(config)[0].turn.stop_reason == "task_started"

    append([{"type": "event_msg", "payload": {"type": "turn_aborted", "reason": "interrupted"}}])
    assert _rows(config)[0].turn.state is TurnPhase.AWAITING_INPUT
    assert _rows(config)[0].turn.stop_reason == "interrupted"
    before = _counts(config)
    _index_as_the_daemon_does(config, transcript)
    assert _counts(config) == before
    assert _rows(config)[0].turn.state is TurnPhase.AWAITING_INPUT


@pytest.mark.parametrize("tool_shape", ["canonical", "ImageView", "WebSearch"])
def test_codex_open_turn_survives_paired_tools_and_completion_without_another_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_shape: str
) -> None:
    """Native task lifecycle remains truthful between tools and at the final stop."""
    monkeypatch.setenv("HOME", str(tmp_path))
    lane = tmp_path / "codex"
    lane.mkdir()
    config = replace(
        _config(tmp_path, lane),
        sources={
            "claude_code": SourceConfig(roots=()),
            "codex": SourceConfig(roots=(lane,)),
            "pi_agent": SourceConfig(roots=()),
            "grok": SourceConfig(roots=()),
            "kimi_code": SourceConfig(roots=()),
        },
    )
    config.db_path.parent.mkdir(parents=True)
    transcript = lane / "rollout.jsonl"
    records = [
        {"type": "session_meta", "payload": {"id": "native-turn"}},
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-one"}},
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": "Inspect"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Inspecting"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "call-one",
                "name": "read_file",
                "arguments": {"path": "a.py"},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-one",
                "output": "Contents",
            },
        },
    ]

    if tool_shape != "canonical":
        item = (
            {"type": "ImageView", "id": "call-one", "path": "/owned/image.png"}
            if tool_shape == "ImageView"
            else {"type": "WebSearch", "id": "call-one", "query": "reference"}
        )
        records[4:] = [{"type": "event_msg", "payload": {"type": "item_completed", "item": item}}]

    def write_and_index() -> None:
        transcript.write_text("".join(json.dumps(record) + "\n" for record in records))
        os.utime(transcript, (NOW.timestamp(), NOW.timestamp()))
        _index_as_the_daemon_does(config, transcript)

    write_and_index()
    turn = _rows(config)[0].turn
    assert turn.state is TurnPhase.WORKING
    assert turn.running_tool is None
    assert turn.stop_reason == "task_started"

    # Completion can follow the last tool result without adding a message.
    records.append({"type": "event_msg", "payload": {"type": "task_complete"}})
    write_and_index()
    turn = _rows(config)[0].turn
    assert turn.state is TurnPhase.AWAITING_INPUT
    assert turn.stop_reason == "task_complete"

    # A rewrite can replace that stop with an open turn at the same message idx.
    records.pop()
    write_and_index()
    assert _rows(config)[0].turn.state is TurnPhase.WORKING
    _index_as_the_daemon_does(config, transcript)
    assert _rows(config)[0].turn.state is TurnPhase.WORKING
    assert _counts(config)["session_stop_markers"] == 1


def test_the_watch_path_persists_tool_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool call with no result row is an open call forever, so the turn never ends."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config, transcript = _lane(tmp_path, "live_end_turn.jsonl")

    _index_as_the_daemon_does(config, transcript)

    counts = _counts(config)
    assert counts["tool_calls"] > 0
    assert counts["tool_results"] == counts["tool_calls"]


def test_the_watch_path_persists_stop_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config, transcript = _lane(tmp_path, "live_end_turn.jsonl")

    _index_as_the_daemon_does(config, transcript)

    assert _counts(config)["session_stop_markers"] > 0


def test_a_finished_turn_indexed_by_the_watch_path_reads_awaiting_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The observable the bug bash caught: `working` on a tool that already returned."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config, transcript = _lane(tmp_path, "live_end_turn.jsonl")

    _index_as_the_daemon_does(config, transcript)

    rows = _rows(config)
    assert len(rows) == 1
    assert rows[0].turn.state is TurnPhase.AWAITING_INPUT
    assert rows[0].turn.running_tool is None
    assert rows[0].turn.stop_reason == "end_turn"


def test_a_mid_tool_session_indexed_by_the_watch_path_still_reads_working(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other side of the boundary: an unpaired tool_use is genuinely still running."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config, transcript = _lane(tmp_path, "live_mid_tool.jsonl")

    _index_as_the_daemon_does(config, transcript)

    rows = _rows(config)
    assert len(rows) == 1
    assert rows[0].turn.state is TurnPhase.WORKING
    assert rows[0].turn.running_tool is not None
    assert rows[0].turn.running_tool.name == "Bash"


def test_an_incremental_watch_write_persists_the_tail_facts_it_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon's steady state is the incremental path, not the first full parse."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config, transcript = _lane(tmp_path, "live_mid_tool.jsonl")
    _index_as_the_daemon_does(config, transcript)
    before = _counts(config)
    assert before["tool_results"] == 0, "the mid-tool fixture ends on an unpaired tool_use"

    session_id = json.loads(transcript.read_text(encoding="utf-8").splitlines()[0])["sessionId"]
    tool_use_id = _last_tool_use_id(transcript)
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "parentUuid": None,
                    "isSidechain": False,
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": "done",
                            }
                        ],
                    },
                    "uuid": "tail-facts-result",
                    "timestamp": "2026-09-07T18:05:00Z",
                    "cwd": "/home/dev/project",
                    "sessionId": session_id,
                    "version": "2.1.80",
                }
            )
            + "\n"
        )
    os.utime(transcript, (NOW.timestamp(), NOW.timestamp()))

    _index_as_the_daemon_does(config, transcript)

    assert _counts(config)["tool_results"] == 1
    rows = _rows(config)
    assert rows[0].turn.running_tool is None
    assert rows[0].liveness is Liveness.IDLE


def _last_tool_use_id(transcript: Path) -> str:
    for line in reversed(transcript.read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        content = record.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                return str(block["id"])
    raise AssertionError("fixture carries no tool_use block")
