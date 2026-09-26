"""The writer-pid mark a harness hook writes, and what it changes (REQ-LIVE-008).

A mark is enrichment: it never makes a session live, and its absence changes
nothing. What it adds is the one fact recall cannot derive from a transcript —
that the process writing it is gone.
These tests drive `live_view` against a real indexed database so the store, the
pid probe, and the join are proved together.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.types import DaemonMode
from recall.services.indexer import index_sessions
from recall.services.live import Liveness, live_view
from recall.services.live_marks import pid_alive, upsert_live_mark

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
IDLE_WINDOW = 86400.0
NOW = datetime(2026, 9, 10, 12, 0, 0)
MARKED_AT = datetime(2026, 9, 10, 11, 30, 0)
HOST = "laptop"


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


def _install(tmp_path: Path, fixture_name: str, dest_name: str) -> Path:
    projects = tmp_path / ".claude" / "projects" / "proj"
    projects.mkdir(parents=True, exist_ok=True)
    dest = projects / dest_name
    shutil.copy(FIXTURES / "claude_code" / fixture_name, dest)
    stamp = NOW.timestamp()
    os.utime(dest, (stamp, stamp))
    return dest


def _index(tmp_path: Path) -> None:
    index_sessions(source=None, full=False, recreate=True, verbose=False)


def _writable(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(_app_config(tmp_path).db_path))


def _readonly(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(_app_config(tmp_path).db_path), read_only=True)


def _session_identity(tmp_path: Path, path: Path) -> tuple[str, str]:
    """The (source, source_session_id) the index recorded for a transcript."""
    conn = _readonly(tmp_path)
    try:
        row = conn.execute(
            "SELECT source, source_session_id FROM sessions WHERE source_path = ?",
            [str(path)],
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, f"{path} was not indexed"
    return str(row[0]), str(row[1])


def _indexed_host(tmp_path: Path) -> str:
    conn = _readonly(tmp_path)
    try:
        row = conn.execute("SELECT host FROM session_state LIMIT 1").fetchone()
    finally:
        conn.close()
    assert row is not None
    return str(row[0])


def _mark(
    tmp_path: Path,
    path: Path,
    *,
    pid: int,
    host: str | None = None,
) -> None:
    source, source_session_id = _session_identity(tmp_path, path)
    stamped_host = host if host is not None else _indexed_host(tmp_path)
    conn = _writable(tmp_path)
    try:
        upsert_live_mark(
            conn,
            source=source,
            source_session_id=source_session_id,
            host=stamped_host,
            pid=pid,
            marked_at=MARKED_AT,
        )
    finally:
        conn.close()


def _reaped_pid() -> int:
    """A pid that has exited and been waited on, so `kill(pid, 0)` finds nothing."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _view(
    tmp_path: Path,
    *,
    watched_paths: list[str] | None = None,
    include_idle: bool = True,
    limit: int = 50,
    local_host: str | None = None,
) -> list:
    conn = _readonly(tmp_path)
    try:
        return live_view(
            conn=conn,
            watched_paths=watched_paths or [],
            include_idle=include_idle,
            now=NOW,
            idle_window_seconds=IDLE_WINDOW,
            limit=limit,
            local_host=local_host if local_host is not None else _indexed_host(tmp_path),
        )
    finally:
        conn.close()


def test_marking_the_same_session_twice_leaves_one_row_carrying_the_latest_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SessionStart hook re-runs on resume; the mark is a fact, not a log."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_mid_tool.jsonl", "live.jsonl")
    _index(tmp_path)
    source, source_session_id = _session_identity(tmp_path, path)

    conn = _writable(tmp_path)
    try:
        upsert_live_mark(
            conn,
            source=source,
            source_session_id=source_session_id,
            host=HOST,
            pid=111,
            marked_at=MARKED_AT,
        )
        upsert_live_mark(
            conn,
            source=source,
            source_session_id=source_session_id,
            host=HOST,
            pid=222,
            marked_at=datetime(2026, 9, 10, 11, 45, 0),
        )
        rows = conn.execute("SELECT pid, marked_at FROM live_marks").fetchall()
    finally:
        conn.close()

    assert rows == [(222, datetime(2026, 9, 10, 11, 45, 0))]


def test_pid_alive_reports_this_process_and_not_a_reaped_one() -> None:
    assert pid_alive(os.getpid()) is True
    assert pid_alive(_reaped_pid()) is False


def test_a_non_positive_pid_is_never_probed() -> None:
    """`kill(0, 0)` signals the whole process group and `kill(-1, 0)` every process."""
    assert pid_alive(0) is False
    assert pid_alive(-1) is False


def test_a_session_whose_marked_pid_is_gone_reads_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one fact a transcript cannot carry: the agent that was writing it exited."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_end_turn.jsonl", "quiet.jsonl")
    _index(tmp_path)
    _mark(tmp_path, path, pid=_reaped_pid())

    rows = _view(tmp_path)

    assert len(rows) == 1
    assert rows[0].liveness is Liveness.ENDED


def test_a_session_whose_marked_pid_is_running_stays_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live process is not evidence the transcript is being written to."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_end_turn.jsonl", "quiet.jsonl")
    _index(tmp_path)
    _mark(tmp_path, path, pid=os.getpid())

    rows = _view(tmp_path)

    assert len(rows) == 1
    assert rows[0].liveness is Liveness.IDLE


def test_a_dead_pid_ends_a_session_the_daemon_is_still_watching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The U18 case: a SIGKILLed agent must not read `active` until the live set demotes it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_mid_tool.jsonl", "live.jsonl")
    _index(tmp_path)
    _mark(tmp_path, path, pid=_reaped_pid())

    rows = _view(tmp_path, watched_paths=[str(path)], include_idle=True)

    assert len(rows) == 1
    assert rows[0].liveness is Liveness.ENDED


def test_a_watched_session_leaves_the_default_roster_after_its_marked_process_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_mid_tool.jsonl", "live.jsonl")
    _index(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE,
    )
    try:
        _mark(tmp_path, path, pid=process.pid)
        active = _view(tmp_path, watched_paths=[str(path)], include_idle=False)
    finally:
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            raise

    default = _view(tmp_path, watched_paths=[str(path)], include_idle=False)
    diagnostics = _view(tmp_path, watched_paths=[str(path)], include_idle=True)

    assert len(active) == 1
    assert active[0].id is not None
    assert active[0].liveness is Liveness.ACTIVE
    assert process.returncode == 0
    assert default == []
    assert len(diagnostics) == 1
    assert diagnostics[0].id == active[0].id
    assert diagnostics[0].liveness is Liveness.ENDED


def test_ended_watched_sessions_do_not_consume_the_default_result_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    active = _install(tmp_path, "live_mid_tool.jsonl", "active.jsonl")
    ended = _install(tmp_path, "live_end_turn.jsonl", "ended.jsonl")
    stamp = NOW.timestamp() + 60
    os.utime(ended, (stamp, stamp))
    _index(tmp_path)
    _mark(tmp_path, active, pid=os.getpid())
    _mark(tmp_path, ended, pid=_reaped_pid())

    default = _view(tmp_path, watched_paths=[str(active), str(ended)], include_idle=False, limit=1)
    diagnostics = _view(
        tmp_path, watched_paths=[str(active), str(ended)], include_idle=True, limit=1
    )

    assert len(default) == 1
    assert default[0].path == str(active)
    assert default[0].id is not None
    assert default[0].liveness is Liveness.ACTIVE
    assert len(diagnostics) == 1
    assert diagnostics[0].path == str(ended)
    assert diagnostics[0].id is not None
    assert diagnostics[0].liveness is Liveness.ENDED


def test_a_live_pid_leaves_a_watched_session_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_mid_tool.jsonl", "live.jsonl")
    _index(tmp_path)
    _mark(tmp_path, path, pid=os.getpid())

    rows = _view(tmp_path, watched_paths=[str(path)], include_idle=False)

    assert len(rows) == 1
    assert rows[0].liveness is Liveness.ACTIVE


def test_a_mark_from_another_host_is_never_probed_against_local_pids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pid number means nothing off the machine that stamped it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_end_turn.jsonl", "quiet.jsonl")
    _index(tmp_path)
    conn = _writable(tmp_path)
    try:
        conn.execute("UPDATE session_state SET host = ?", ["devbox"])
    finally:
        conn.close()
    _mark(tmp_path, path, pid=_reaped_pid(), host="devbox")

    rows = _view(tmp_path, local_host="laptop")

    assert len(rows) == 1
    assert rows[0].liveness is Liveness.IDLE


def test_a_row_carries_the_writer_pid_its_mark_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _install(tmp_path, "live_end_turn.jsonl", "quiet.jsonl")
    _index(tmp_path)
    _mark(tmp_path, path, pid=os.getpid())

    rows = _view(tmp_path)

    assert rows[0].writer_pid == os.getpid()


def test_an_unmarked_row_carries_no_writer_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _install(tmp_path, "live_end_turn.jsonl", "quiet.jsonl")
    _index(tmp_path)

    rows = _view(tmp_path)

    assert rows[0].writer_pid is None
