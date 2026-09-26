"""Assembling `LiveSession` rows from the live set and the index (REQ-LIVE-002).

`live_view` is the join the `recall live` command and its RPC both go through.
These tests exercise it against a real indexed database so the SQL, the
derivations, and the `--all` widening are proved together — the derivations
themselves are pinned as pure functions in `test_live_derivations.py`.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.types import DaemonMode, default_session_host
from recall.services.indexer import index_sessions
from recall.services.live import Liveness, TurnPhase, live_view, live_view_page

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
IDLE_WINDOW = 86400.0

# Fixed clock, and every installed transcript is stamped after the newest
# timestamp inside the fixtures (2026-09-07T18:00Z) so `last_activity_at`
# — GREATEST(ended_at, file_mtime) — is the mtime these tests actually set.
NOW = datetime(2026, 9, 10, 12, 0, 0)

# The label the indexer stamps on every row these tests write.
LOCAL_HOST = default_session_host()


def _app_config(tmp_path: Path) -> AppConfig:
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


def _install(tmp_path: Path, fixture_name: str, dest_name: str, *, mtime: datetime = NOW) -> Path:
    projects = tmp_path / ".claude" / "projects" / "proj"
    projects.mkdir(parents=True, exist_ok=True)
    dest = projects / dest_name
    shutil.copy(FIXTURES / "claude_code" / fixture_name, dest)
    stamp = mtime.timestamp()
    os.utime(dest, (stamp, stamp))
    return dest


def _connect(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(_app_config(tmp_path).db_path), read_only=True)


def test_a_watched_session_is_active_and_carries_its_indexed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_mid_tool.jsonl", "live.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        rows = live_view(
            conn=conn,
            watched_paths=[str(path)],
            include_idle=False,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
        )
    finally:
        conn.close()

    assert len(rows) == 1
    row = rows[0]
    assert row.liveness is Liveness.ACTIVE
    assert row.id is not None
    assert row.source == "claude_code"
    assert row.path == str(path)
    assert row.freshness.current is False
    assert row.freshness.limitations == ("catalog_progress_unavailable",)
    assert row.turn.state is TurnPhase.WORKING
    assert row.turn.running_tool is not None
    assert row.turn.running_tool.name == "Bash"
    assert row.cursor is not None


def test_a_watched_path_the_index_has_never_seen_is_still_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon promotes on first write — routinely before any pass indexes it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_mid_tool.jsonl", "indexed.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)
    fresh = _install(tmp_path, "live_end_turn.jsonl", "brand-new.jsonl")

    conn = _connect(tmp_path)
    try:
        rows = live_view(
            conn=conn,
            watched_paths=[str(fresh)],
            include_idle=False,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
        )
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0].liveness is Liveness.ACTIVE
    assert rows[0].id is None
    assert rows[0].path == str(fresh)
    assert rows[0].freshness.current is False
    assert rows[0].freshness.indexed_size is None
    assert rows[0].turn.state is TurnPhase.UNKNOWN
    assert rows[0].cursor is None


def test_without_all_an_unwatched_session_is_not_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl", "quiet.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        rows = live_view(
            conn=conn,
            watched_paths=[],
            include_idle=False,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
        )
    finally:
        conn.close()

    assert rows == []


def test_all_adds_recently_active_sessions_as_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl", "quiet.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        rows = live_view(
            conn=conn,
            watched_paths=[],
            include_idle=True,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
        )
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0].liveness is Liveness.IDLE
    assert rows[0].turn.state is TurnPhase.AWAITING_INPUT
    assert rows[0].turn.stop_reason == "end_turn"


def test_all_leaves_out_sessions_older_than_the_idle_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl", "stale.jsonl", mtime=NOW - timedelta(days=30))
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        rows = live_view(
            conn=conn,
            watched_paths=[],
            include_idle=True,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
        )
    finally:
        conn.close()

    assert rows == []


def test_rows_are_sorted_by_last_activity_descending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    older = _install(
        tmp_path, "live_mid_tool.jsonl", "older.jsonl", mtime=NOW - timedelta(minutes=10)
    )
    newer = _install(
        tmp_path, "live_end_turn.jsonl", "newer.jsonl", mtime=NOW - timedelta(minutes=1)
    )
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        rows = live_view(
            conn=conn,
            watched_paths=[],
            include_idle=True,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
        )
    finally:
        conn.close()

    assert [row.path for row in rows] == [str(newer), str(older)]


def test_a_watched_session_outranks_the_idle_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transcript being written to right now is active however old its rows are."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_end_turn.jsonl", "live.jsonl", mtime=NOW - timedelta(days=30))
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        rows = live_view(
            conn=conn,
            watched_paths=[str(path)],
            include_idle=True,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
        )
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0].liveness is Liveness.ACTIVE


def test_the_source_filter_narrows_the_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl", "quiet.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        kept = live_view(
            conn=conn,
            watched_paths=[],
            include_idle=True,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
            source="claude_code",
        )
        dropped = live_view(
            conn=conn,
            watched_paths=[],
            include_idle=True,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
            source="codex",
        )
    finally:
        conn.close()

    assert len(kept) == 1
    assert dropped == []


@pytest.mark.parametrize(
    "source,project,host",
    [
        pytest.param("codex", None, None, id="source"),
        pytest.param(None, "/nonmatching-project", None, id="project"),
        pytest.param(None, None, f"{LOCAL_HOST}-other", id="host"),
    ],
)
@pytest.mark.parametrize(
    "watched,include_idle",
    [
        pytest.param(True, False, id="active"),
        pytest.param(True, True, id="active-with-all"),
        pytest.param(False, True, id="idle"),
    ],
)
def test_nonmatching_filters_exclude_indexed_sessions_without_anonymous_reentry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str | None,
    project: str | None,
    host: str | None,
    watched: bool,
    include_idle: bool,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_end_turn.jsonl", "indexed.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    with _connect(tmp_path) as conn:
        baseline = live_view(
            conn=conn,
            watched_paths=[str(path)] if watched else [],
            include_idle=include_idle,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
        )
        rows = live_view(
            conn=conn,
            watched_paths=[str(path)] if watched else [],
            include_idle=include_idle,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
            source=source,
            project=project,
            host=host,
        )

    assert len(baseline) == 1
    assert baseline[0].id is not None
    assert baseline[0].source == "claude_code"
    assert baseline[0].git_repo == "/home/dev/project"
    assert baseline[0].host == LOCAL_HOST
    assert baseline[0].liveness is (Liveness.ACTIVE if watched else Liveness.IDLE)
    assert rows == []


@pytest.mark.parametrize("source", ["claude_code", "codex"])
def test_source_filter_keeps_only_matching_indexed_identity_among_watched_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    claude = _install(tmp_path, "live_end_turn.jsonl", "claude.jsonl")
    codex = tmp_path / ".codex" / "sessions" / "rollout.jsonl"
    codex.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "codex" / "live_tool_pair" / "rollout.jsonl", codex)
    stamp = NOW.timestamp()
    os.utime(codex, (stamp, stamp))
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    with _connect(tmp_path) as conn:
        rows = live_view(
            conn=conn,
            watched_paths=[str(claude), str(codex)],
            include_idle=True,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
            source=source,
        )

    assert len(rows) == 1
    assert rows[0].path == str(claude if source == "claude_code" else codex)
    assert rows[0].id is not None
    assert rows[0].source == source
    assert rows[0].liveness is Liveness.ACTIVE
    assert rows[0].freshness.current is False
    assert rows[0].freshness.limitations == ("catalog_progress_unavailable",)
    assert rows[0].cursor is not None


@pytest.mark.parametrize(
    "project,host",
    [
        pytest.param("/HOME/DEV/PROJECT", None, id="project"),
        pytest.param(None, LOCAL_HOST, id="host"),
        pytest.param("/HOME/DEV/PROJECT", LOCAL_HOST, id="project-and-host"),
    ],
)
def test_matching_metadata_filters_compose_without_losing_indexed_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    project: str | None,
    host: str | None,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_end_turn.jsonl", "indexed.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    with _connect(tmp_path) as conn:
        rows = live_view(
            conn=conn,
            watched_paths=[str(path)],
            include_idle=False,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=1,
            local_host=LOCAL_HOST,
            source="claude_code",
            project=project,
            host=host,
        )

    assert len(rows) == 1
    assert rows[0].path == str(path)
    assert rows[0].id is not None
    assert rows[0].liveness is Liveness.ACTIVE
    assert rows[0].git_repo == "/home/dev/project"
    assert rows[0].host == LOCAL_HOST


@pytest.mark.parametrize(
    "source,project,host",
    [
        pytest.param("claude_code", None, None, id="source"),
        pytest.param(None, "/home/dev/project", None, id="project"),
        pytest.param(None, None, LOCAL_HOST, id="host"),
    ],
)
def test_metadata_filters_exclude_watched_paths_with_no_indexed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str | None,
    project: str | None,
    host: str | None,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_mid_tool.jsonl", "indexed.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)
    fresh = _install(tmp_path, "live_end_turn.jsonl", "brand-new.jsonl")

    with _connect(tmp_path) as conn:
        rows = live_view(
            conn=conn,
            watched_paths=[str(fresh)],
            include_idle=False,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            local_host=LOCAL_HOST,
            source=source,
            project=project,
            host=host,
        )

    assert rows == []


def test_limit_does_not_restore_an_indexed_path_as_anonymous_when_its_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    older = _install(
        tmp_path, "live_mid_tool.jsonl", "older.jsonl", mtime=NOW - timedelta(minutes=10)
    )
    newer = _install(
        tmp_path, "live_end_turn.jsonl", "newer.jsonl", mtime=NOW - timedelta(minutes=1)
    )
    index_sessions(source=None, full=False, recreate=True, verbose=False)
    # The watched file advances before the next index pass. Its indexed
    # activity still sorts outside this page, but its identity still exists.
    stamp = NOW.timestamp()
    os.utime(older, (stamp, stamp))

    with _connect(tmp_path) as conn:
        rows = live_view(
            conn=conn,
            watched_paths=[str(older), str(newer)],
            include_idle=False,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=1,
            local_host=LOCAL_HOST,
        )

    assert len(rows) == 1
    assert rows[0].path == str(newer)
    assert rows[0].id is not None
    assert rows[0].source == "claude_code"
    assert rows[0].freshness.current is False
    assert rows[0].freshness.limitations == ("catalog_progress_unavailable",)


def test_live_view_rejects_a_non_positive_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl", "quiet.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    conn = _connect(tmp_path)
    try:
        with pytest.raises(ValueError, match="limit"):
            live_view(
                conn=conn,
                watched_paths=[],
                include_idle=True,
                now=NOW,
                idle_window_seconds=IDLE_WINDOW,
                limit=0,
                local_host=LOCAL_HOST,
            )
    finally:
        conn.close()


def test_roster_page_does_not_share_open_tools_between_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    working = _install(tmp_path, "live_mid_tool.jsonl", "working.jsonl")
    finished = _install(tmp_path, "live_end_turn.jsonl", "finished.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)
    with _connect(tmp_path) as conn:
        page = live_view_page(
            conn=conn,
            watched_paths=[str(working), str(finished)],
            include_idle=True,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            cursor=None,
            local_host=LOCAL_HOST,
        )
    by_path = {row.path: row.turn for row in page.sessions}
    working_turn = by_path[str(working)]
    assert working_turn.state is TurnPhase.WORKING
    assert working_turn.running_tool is not None
    assert working_turn.running_tool.name == "Bash"
    assert by_path[str(finished)].state is TurnPhase.AWAITING_INPUT
    assert by_path[str(finished)].running_tool is None


def test_roster_page_keeps_each_sessions_own_eight_message_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = []
    for name in ("first", "second"):
        path = _install(tmp_path, "live_end_turn.jsonl", f"{name}.jsonl")
        records = []
        for index in range(12):
            role = "user" if index == 0 or (name == "second" and index == 10) else "assistant"
            records.append(
                {
                    "type": role,
                    "sessionId": name,
                    "message": {
                        "role": role,
                        "content": [{"type": "text", "text": f"{name}-{index}"}],
                        "stop_reason": "end_turn",
                    },
                }
            )
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        os.utime(path, (NOW.timestamp(), NOW.timestamp()))
        paths.append(str(path))
    index_sessions(source=None, full=False, recreate=True, verbose=False)
    with _connect(tmp_path) as conn:
        page = live_view_page(
            conn=conn,
            watched_paths=paths,
            include_idle=True,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=50,
            cursor=None,
            local_host=LOCAL_HOST,
        )
    assert len(page.sessions) == 2
    assert page.next_cursor is None
    by_path = {row.path: row for row in page.sessions}
    assert by_path[paths[0]].turn.last_user_text is None
    assert by_path[paths[0]].turn.last_assistant_text == "first-11"
    assert by_path[paths[1]].turn.last_user_text == "second-10"
    assert by_path[paths[1]].turn.last_assistant_text == "second-11"
    assert all(row.turn.state is TurnPhase.AWAITING_INPUT for row in page.sessions)


def test_live_view_page_uses_a_bounded_sql_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first = _install(tmp_path, "live_mid_tool.jsonl", "first.jsonl")
    second = _install(tmp_path, "live_end_turn.jsonl", "second.jsonl")
    index_sessions(source=None, full=False, recreate=True, verbose=False)

    with _connect(tmp_path) as conn:
        page_one = live_view_page(
            conn=conn,
            watched_paths=[str(first), str(second)],
            include_idle=False,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=1,
            cursor=None,
            local_host=LOCAL_HOST,
        )
        page_two = live_view_page(
            conn=conn,
            watched_paths=[str(first), str(second)],
            include_idle=False,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=1,
            cursor=page_one.next_cursor,
            local_host=LOCAL_HOST,
        )

    assert len(page_one.sessions) == 1
    assert page_one.next_cursor is not None
    assert len(page_two.sessions) == 1
    assert page_two.next_cursor is None
    assert {page_one.sessions[0].path, page_two.sessions[0].path} == {str(first), str(second)}
