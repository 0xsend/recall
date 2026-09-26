"""Tests for scripts/backfill_codex_tokens.py — the REQ-PARSE-011 token backfill.

The script repairs Codex sessions that were indexed before the token_count
parser fix and therefore carry NULL token totals. These tests index a
Codex fixture with the *current* parser (which populates tokens), NULL the
tokens to recreate the pre-fix state, then assert the backfill restores them
without touching anything else.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import duckdb
import pytest
from conftest import _can_acquire_duckdb_lock
from recall.services.indexer import index_sessions

pytestmark = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="sandbox cannot open a writable DuckDB database",
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures" / "codex"
# session4 carries two token_count events (100/10 then 150/15); the parser keeps
# the cumulative max, so a correct backfill restores (150, 15).
TOKEN_FIXTURE = FIXTURES / "session4" / "rollout-2026-05-30T20-43-26-token-count.jsonl"
# session1 has no token_count events at all → tokens are legitimately NULL.
NO_TOKEN_FIXTURE = FIXTURES / "session1" / "rollout.jsonl"


def _load_backfill_module():
    """Load the standalone script as a module (scripts/ is not a package)."""
    path = ROOT / "scripts" / "backfill_codex_tokens.py"
    spec = importlib.util.spec_from_file_location("backfill_codex_tokens", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module: @dataclass(frozen=True) resolves its own module
    # via sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backfill_module = _load_backfill_module()


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / ".local" / "share" / "recall" / "recall.duckdb"


def _index_codex_fixture(tmp_path: Path, fixture: Path) -> None:
    """Place a Codex fixture under the (monkeypatched) HOME and index it."""
    codex_dir = tmp_path / ".codex" / "sessions" / "s1"
    codex_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture, codex_dir / fixture.name)
    result = index_sessions(source=None, full=False, recreate=True, verbose=False)
    assert result.indexed >= 1


def _fetch_tokens(db_path: Path, session_id: str | None = None) -> tuple[int | None, int | None]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        if session_id is None:
            row = conn.execute(
                "SELECT input_tokens, output_tokens FROM session_state LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT input_tokens, output_tokens FROM session_state WHERE session_id = ?",
                [session_id],
            ).fetchone()
        assert row is not None
        return row[0], row[1]
    finally:
        conn.close()


def _null_codex_tokens(db_path: Path) -> None:
    """Recreate the pre-fix backlog: wipe tokens the current parser would set."""
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE session_state SET input_tokens = NULL, output_tokens = NULL "
            "WHERE session_id IN (SELECT id FROM sessions WHERE source = 'codex')"
        )
    finally:
        conn.close()


def _insert_raw_session(db_path: Path, *, session_id: str, source: str, source_path: str) -> None:
    """Insert a minimal session row (NULL tokens) for sources we don't index here."""
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO sessions(id, source, source_path) VALUES (?, ?, ?)",
            [session_id, source, source_path],
        )
        conn.execute(
            "INSERT INTO session_state(session_id, file_mtime, file_size, "
            "input_tokens, output_tokens) VALUES (?, ?, ?, NULL, NULL)",
            [session_id, 0.0, 0],
        )
    finally:
        conn.close()


def test_backfill_repopulates_nulled_codex_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _index_codex_fixture(tmp_path, TOKEN_FIXTURE)
    db = _db_path(tmp_path)
    assert _fetch_tokens(db) == (150, 15)  # current parser populates it

    # A non-Codex session with NULL tokens must be left untouched by the repair.
    _insert_raw_session(db, session_id="cc1", source="claude_code", source_path="/x/cc.jsonl")

    _null_codex_tokens(db)  # simulate the pre-fix backlog

    stats = backfill_module.backfill()
    assert stats.scanned == 1  # only the Codex session is a candidate
    assert stats.updated == 1
    assert stats.no_tokens == 0
    assert _fetch_tokens(db) == (150, 15)  # restored
    assert _fetch_tokens(db, "cc1") == (None, None)  # non-Codex untouched


def test_backfill_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _index_codex_fixture(tmp_path, TOKEN_FIXTURE)
    _null_codex_tokens(_db_path(tmp_path))

    first = backfill_module.backfill()
    assert first.updated == 1
    second = backfill_module.backfill()
    assert second.scanned == 0  # no NULL-token Codex rows remain
    assert second.updated == 0


def test_backfill_skips_sessions_without_token_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _index_codex_fixture(tmp_path, NO_TOKEN_FIXTURE)
    db = _db_path(tmp_path)
    assert _fetch_tokens(db) == (None, None)  # legitimately NULL — no token data

    stats = backfill_module.backfill()
    assert stats.scanned == 1
    assert stats.updated == 0
    assert stats.no_tokens == 1
    assert _fetch_tokens(db) == (None, None)  # stays NULL — nothing invented


def test_backfill_dry_run_reports_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _index_codex_fixture(tmp_path, TOKEN_FIXTURE)
    db = _db_path(tmp_path)
    _null_codex_tokens(db)

    stats = backfill_module.backfill(dry_run=True)
    assert stats.updated == 1  # would update one session
    assert _fetch_tokens(db) == (None, None)  # but wrote nothing
