from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from recall.parsers.claude_code import ClaudeCodeParser
from recall.parsers.codex import CodexParser
from recall.parsers.pi_agent import PiAgentParser
from recall.parsers.protocol import SessionParser


@pytest.fixture()
def now() -> datetime:
    return datetime(2026, 4, 14, 12, 0, tzinfo=UTC)


@pytest.fixture(
    params=[
        pytest.param(
            (ClaudeCodeParser, "session-live.jsonl", "session-stale.jsonl"),
            id="claude-code",
        ),
        pytest.param((CodexParser, "rollout-live.jsonl", "rollout-stale.jsonl"), id="codex"),
        pytest.param(
            (PiAgentParser, "session-live.jsonl", "session-stale.jsonl"),
            id="pi-agent",
        ),
    ]
)
def parser_case(request: pytest.FixtureRequest) -> tuple[SessionParser, str, str]:
    parser_cls, recent_name, stale_name = request.param
    return cast(SessionParser, parser_cls()), recent_name, stale_name


def _write_file(path: Path, *, mtime: datetime) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))
    return path


def test_live_candidates_returns_only_matching_recent_files(
    parser_case: tuple[SessionParser, str, str],
    now: datetime,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parser, recent_name, stale_name = parser_case
    watch_root = tmp_path / "sessions"

    recent = _write_file(
        watch_root / recent_name,
        mtime=now - timedelta(seconds=30),
    )
    stale = _write_file(
        recent.parent / stale_name,
        mtime=now - timedelta(seconds=90),
    )
    ignored = _write_file(
        recent.parent / "ignored.txt",
        mtime=now - timedelta(seconds=15),
    )

    watch_root = recent.parent
    monkeypatch.setattr(parser, "watch_roots", lambda: [watch_root])

    live_paths = parser.live_candidates(now=now, idle_threshold=60.0)

    assert recent in live_paths
    assert stale not in live_paths
    assert ignored not in live_paths


def test_live_candidates_returns_empty_for_missing_watch_roots(
    parser_case: tuple[SessionParser, str, str],
    now: datetime,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parser, _, _ = parser_case
    watch_root = tmp_path / "missing"
    monkeypatch.setattr(parser, "watch_roots", lambda: [watch_root])

    assert parser.live_candidates(now=now, idle_threshold=60.0) == []


def test_live_candidates_skips_stat_errors(
    parser_case: tuple[SessionParser, str, str],
    now: datetime,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    parser, matching_name, _ = parser_case
    watch_root = tmp_path / "sessions"
    recent = _write_file(watch_root / matching_name, mtime=now - timedelta(seconds=20))
    broken_name = (
        "rollout-broken.jsonl"
        if parser.file_pattern == "rollout*.jsonl"
        else "session-broken.jsonl"
    )
    broken = _write_file(
        watch_root / broken_name,
        mtime=now - timedelta(seconds=10),
    )
    original_stat = Path.stat

    def stat_with_error(path: Path, *, follow_symlinks: bool = True):
        if path == broken:
            msg = "simulated stat failure"
            raise OSError(msg)
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(parser, "watch_roots", lambda: [watch_root])
    monkeypatch.setattr(Path, "stat", stat_with_error)
    caplog.set_level("DEBUG")

    live_paths = parser.live_candidates(now=now, idle_threshold=60.0)

    assert live_paths == [recent]
    assert "stat failed for" in caplog.text
