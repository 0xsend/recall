"""Sidecar files must participate in the index change signal.

Grok enriches session metadata from sibling sidecars (REQ-PARSE-014), but the
change signal was the primary session file's mtime/size alone. A session whose
sidecar changed -- or whose row predates sidecar support -- was therefore
skipped as unchanged forever, leaving `started_at` NULL with no error anywhere.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import duckdb
from recall.services.indexer import index_sessions


def _grok_fixture_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "grok" / "with_sidecars"


def _install_grok_session(home: Path) -> Path:
    """Copy the sidecar-bearing grok fixture into a fake ~/.grok/sessions."""
    target = home / ".grok" / "sessions"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_grok_fixture_dir(), target, dirs_exist_ok=True)
    session_dirs = [p.parent for p in target.rglob("**/chat_history.jsonl")]
    assert len(session_dirs) == 1, session_dirs
    return session_dirs[0]


def _db_path(home: Path) -> Path:
    return home / ".local/share/recall" / "recall.duckdb"


def _started_at(home: Path) -> object:
    conn = duckdb.connect(str(_db_path(home)))
    try:
        row = conn.execute(
            "SELECT ss.started_at FROM session_state ss "
            "JOIN sessions s ON s.id = ss.session_id WHERE s.source = 'grok'"
        ).fetchone()
        return None if row is None else row[0]
    finally:
        conn.close()


def test_grok_parser_declares_its_sidecars() -> None:
    """The parser must name the files whose content it folds into metadata."""
    from recall.parsers.grok import GrokParser

    session_dir = _grok_fixture_dir() / "%2Fwork%2Fproject" / "sess-uuid-1"
    chat_history = session_dir / "chat_history.jsonl"

    sidecars = GrokParser().sidecar_paths(chat_history)

    assert set(sidecars) == {
        session_dir / "summary.json",
        session_dir / "signals.json",
    }


def test_parsers_without_sidecars_declare_none() -> None:
    """Sources with no sidecars keep an empty fingerprint so they never churn."""
    from recall.parsers.claude_code import ClaudeCodeParser
    from recall.parsers.codex import CodexParser

    assert ClaudeCodeParser().sidecar_paths(Path("/tmp/session1.jsonl")) == []
    assert CodexParser().sidecar_paths(Path("/tmp/rollout.jsonl")) == []


def test_sidecar_edit_reindexes_an_otherwise_unchanged_session(tmp_path, monkeypatch) -> None:
    """Rewriting summary.json alone must invalidate the cached row."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    session_dir = _install_grok_session(tmp_path)

    first = index_sessions(source=None, full=True, recreate=True, verbose=False)
    assert first.indexed == 1
    assert first.failed == 0

    # Unchanged tree: still skipped.
    second = index_sessions(source=None, full=False, recreate=False, verbose=False)
    assert second.skipped == 1
    assert second.indexed == 0

    # Touch ONLY the sidecar; chat_history.jsonl keeps its mtime and size.
    summary_path = session_dir / "summary.json"
    chat_history = session_dir / "chat_history.jsonl"
    before = chat_history.stat()
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    raw["created_at"] = "2026-07-04T01:02:03.000000Z"
    summary_path.write_text(json.dumps(raw), encoding="utf-8")
    after = chat_history.stat()
    assert (before.st_mtime, before.st_size) == (after.st_mtime, after.st_size)

    third = index_sessions(source=None, full=False, recreate=False, verbose=False)
    assert third.indexed == 1, "sidecar edit must re-index the session"
    assert third.skipped == 0

    # Compare against the parser's own conversion rather than a literal: the
    # stored value is localized, so a hard-coded UTC date is timezone-dependent.
    from recall.parsers.common import parse_timestamp

    parsed = parse_timestamp("2026-07-04T01:02:03.000000Z")
    assert parsed is not None
    expected = parsed.astimezone().replace(tzinfo=None)

    refreshed = _started_at(tmp_path)
    assert refreshed is not None
    assert refreshed == expected


def test_row_predating_sidecar_support_is_backfilled(tmp_path, monkeypatch) -> None:
    """A NULL sidecar_mtime is the pre-migration state and must re-index once."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    _install_grok_session(tmp_path)
    assert index_sessions(source=None, full=True, recreate=True, verbose=False).indexed == 1

    # Simulate a row written before sidecars fed the fingerprint: drop the
    # sidecar stamp and the metadata it produced.
    conn = duckdb.connect(str(_db_path(tmp_path)))
    try:
        conn.execute("UPDATE session_state SET sidecar_mtime = NULL, started_at = NULL")
    finally:
        conn.close()

    assert _started_at(tmp_path) is None

    summary = index_sessions(source=None, full=False, recreate=False, verbose=False)
    assert summary.indexed == 1, "legacy row must be treated as changed"
    assert _started_at(tmp_path) is not None, "re-index must backfill started_at"


def test_missing_start_timestamp_warns_once_per_source(tmp_path, monkeypatch, caplog) -> None:
    """Format drift is loud but aggregated: one warning naming the source and count."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    session_dir = _install_grok_session(tmp_path)

    # Simulate upstream renaming the timestamp key: sidecar present and valid
    # JSON, but the field the parser reads is gone.
    summary_path = session_dir / "summary.json"
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    raw.pop("created_at", None)
    raw.pop("last_active_at", None)
    summary_path.write_text(json.dumps(raw), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="recall.indexer"):
        index_sessions(source=None, full=True, recreate=True, verbose=False)

    drift = [r for r in caplog.records if "without a start timestamp" in r.getMessage()]
    assert len(drift) == 1, f"expected exactly one aggregated warning, got {len(drift)}"
    message = drift[0].getMessage()
    assert "grok" in message
    assert "1" in message


def test_no_warning_when_sidecars_are_absent(tmp_path, monkeypatch, caplog) -> None:
    """A source with no sidecars at all is not drift -- never warn about it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

    claude_target = tmp_path / ".claude" / "projects" / "proj1"
    claude_target.mkdir(parents=True)
    shutil.copy(
        Path(__file__).resolve().parents[2] / "fixtures" / "claude_code" / "session1.jsonl",
        claude_target / "session1.jsonl",
    )

    with caplog.at_level(logging.WARNING, logger="recall.indexer"):
        index_sessions(source=None, full=True, recreate=True, verbose=False)

    assert not [r for r in caplog.records if "without a start timestamp" in r.getMessage()]
