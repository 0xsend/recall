"""Persisted response accounting across complete source boundaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from recall.core.config import AppConfig
from recall.core.models import NormalizationCheckpoint, Session, TailFacts
from recall.core.types import Source
from recall.db.fts_sidecar import ensure_sidecar
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog
from recall.services.coordinator import (
    capture_path,
    commit_prepared_raw_sources,
    observe_path,
    prepare_raw_sources,
)
from recall.services.indexer import _write_session, index_sessions

from tests.keyword_turns import without_keyword_index
from tests.test_parsers.test_claude_usage import encoded, records
from tests.test_services.test_checkpoint_resume_fallbacks import _first_wins_harness


def _scope(harness):
    row = harness.conn.execute(
        "SELECT scope_token, format_revision, committed_offset, prefix_sha256 "
        "FROM claude_usage_scopes"
    ).fetchone()
    assert row is not None
    assert row[1:] == (1, harness.item.committed_offset, harness.item.committed_prefix_sha256)
    return row


def _totals(harness):
    return harness.conn.execute(
        "SELECT input_tokens, output_tokens, message_count FROM session_state"
    ).fetchone()


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"input_tokens": 4}, (4, None, 1)),
        ({}, (None, None, 1)),
        ({"input_tokens": 0, "output_tokens": 0}, (0, 0, 1)),
        ({"output_tokens": 0}, (None, 0, 1)),
    ],
)
def test_full_rewrite_replaces_nullable_usage(tmp_path: Path, usage: dict, expected: tuple):
    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    try:
        items = records("main")[:1]
        harness.path.write_bytes(encoded(items))
        harness.reconcile()
        items[0]["message"]["usage"] = usage
        harness.path.write_bytes(encoded(items))
        harness.reconcile(full=True)
        assert _totals(harness) == expected
        scope = _scope(harness)
        cells = harness.conn.execute(
            "SELECT input_tokens, cache_creation_input_tokens, cache_read_input_tokens, "
            "output_tokens FROM claude_response_usage ORDER BY response_key"
        ).fetchall()
        harness.catalog.force_reconcile(harness.source, harness.key_path)
        harness.reconcile(full=True)
        assert _totals(harness) == expected
        assert (
            harness.conn.execute(
                "SELECT input_tokens, cache_creation_input_tokens, cache_read_input_tokens, "
                "output_tokens FROM claude_response_usage ORDER BY response_key"
            ).fetchall()
            == cells
        )
        assert _scope(harness)[0] != scope[0]
        reference = harness.parser.parse(harness.path).session
        assert expected == (
            reference.input_tokens,
            reference.output_tokens,
            reference.message_count,
        )
    finally:
        harness.conn.close()


@pytest.mark.parametrize("reverse", [False, True])
def test_old_corrections_fold_only_supplied_fields_across_restart(tmp_path: Path, reverse: bool):
    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    try:
        items = records("main")[:1]
        harness.path.write_bytes(encoded(items))
        harness.reconcile()
        token = _scope(harness)[0]
        other = json.loads(json.dumps(items[0]))
        other["message"]["id"] = "other"
        other["message"]["usage"] = {"input_tokens": 7, "output_tokens": 3}
        correction = records("main")[2]
        correction["message"]["usage"] = {"input_tokens": 90, "output_tokens": 0}
        append = [other, correction]
        if reverse:
            append.reverse()
        checkpoint = NormalizationCheckpoint.decode(harness.stored_checkpoint())
        assert checkpoint is not None
        harness.set_checkpoint(checkpoint.encode())
        items += append
        harness.path.write_bytes(encoded(items))
        assert harness.reconcile() is True
        assert _totals(harness) == (797, 3, 3)
        assert _scope(harness)[0] == token
        row = _totals(harness)
        harness.reconcile()
        assert _totals(harness) == row
        harness.reconcile(full=True)
        assert _totals(harness) == row
    finally:
        harness.conn.close()


@pytest.mark.parametrize("damage", ["missing", "boundary", "format"])
def test_unproved_scope_retries_full_without_advancing_totals(tmp_path: Path, damage: str):
    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    try:
        items = records("main")
        harness.path.write_bytes(encoded(items[:1]))
        harness.reconcile()
        prior = _totals(harness)
        offset = harness.item.committed_offset
        if damage == "missing":
            harness.conn.execute("DELETE FROM claude_usage_scopes")
        elif damage == "boundary":
            harness.conn.execute("UPDATE claude_usage_scopes SET committed_offset = 1")
        else:
            harness.conn.execute("UPDATE claude_usage_scopes SET format_revision = 2")
        harness.path.write_bytes(encoded(items))
        harness.reconcile()
        assert _totals(harness) == prior
        assert harness.item.committed_offset == offset
        assert harness.stored_checkpoint() is None
        assert harness.reconcile() is False
        assert _totals(harness) == (800, 40, 3)
        _scope(harness)
    finally:
        harness.conn.close()


def test_failed_acknowledgement_rolls_back_usage_and_retry(tmp_path: Path, monkeypatch):
    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    try:
        items = records("main")
        harness.path.write_bytes(encoded(items[:1]))
        harness.reconcile()
        prior = (_totals(harness), _scope(harness), harness.stored_checkpoint())
        harness.path.write_bytes(encoded(items))
        with monkeypatch.context() as patch:
            patch.setattr(SourceCatalog, "acknowledge", lambda *args, **kwargs: False)
            harness.reconcile()
        assert (_totals(harness), _scope(harness), harness.stored_checkpoint()) == prior
        assert harness.reconcile() is True
        assert _totals(harness) == (800, 40, 3)
        _scope(harness)
    finally:
        harness.conn.close()


def test_competing_partial_capture_cannot_publish_twice(tmp_path: Path):
    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    try:
        items = records("main")
        harness.path.write_bytes(encoded(items[:1]))
        harness.reconcile()
        harness.path.write_bytes(encoded(items))
        item = observe_path(
            harness.parser, capture_path(harness.parser, harness.path), conn=harness.conn
        )
        prepared = prepare_raw_sources((item,), {harness.source: harness.parser})
        assert prepared[0].result is not None
        # A partial commit does not advance committed_generation.
        partial = (replace(prepared[0], result=replace(prepared[0].result, reached_end=False)),)
        commit_prepared_raw_sources(partial, harness.config, conn=harness.conn)
        prior = (_totals(harness), _scope(harness))
        commit_prepared_raw_sources(partial, harness.config, conn=harness.conn)
        assert (_totals(harness), _scope(harness)) == prior
    finally:
        harness.conn.close()


def test_metadata_envelope_refusal_keeps_reference_totals(tmp_path: Path):
    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    try:
        items = records("main")
        # High-entropy native metadata exceeds the compressed envelope, independently of IDs.
        items[0]["cwd"] = "/" + "".join(
            hashlib.sha256(str(i).encode()).hexdigest() for i in range(4000)
        )
        harness.path.write_bytes(encoded(items[:1]))
        harness.reconcile()
        assert harness.stored_checkpoint() is None
        harness.path.write_bytes(encoded(items))
        assert harness.reconcile() is False
        assert harness.stored_checkpoint() is None
        assert _totals(harness) == (800, 40, 3)
        _scope(harness)
        harness.reconcile(full=True)
        assert _totals(harness) == (800, 40, 3)
    finally:
        harness.conn.close()


def test_write_session_refuses_claude_publication_without_usage_capture() -> None:
    import duckdb

    conn = duckdb.connect(":memory:")
    try:
        ensure_schema(conn)
        session = Session(
            id="claude-missing-capture",
            source=Source.CLAUDE_CODE,
            source_path="/tmp/missing-capture.jsonl",
            input_tokens=10,
            output_tokens=5,
            file_mtime=0.0,
            file_size=0,
        )
        with pytest.raises(ValueError, match="ClaudeUsageFull"):
            _write_session(conn, session, turn=without_keyword_index(), tail_facts=TailFacts())
        assert conn.execute("SELECT count(*) FROM sessions").fetchone() == (0,)
    finally:
        conn.close()


def test_batch_index_sessions_publishes_claude_ledger_atomically(tmp_path: Path) -> None:
    import duckdb

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    home = tmp_path / "home"
    transcript = home / ".claude" / "projects" / "proj" / "batch.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_bytes(encoded(records("main")))
    config = replace(
        AppConfig.load(),
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
    )
    ensure_sidecar(config)
    with duckdb.connect(str(config.db_path)) as conn:
        ensure_schema(conn, embed_dim=config.embedding.dimensions)
        summary = index_sessions(
            source=Source.CLAUDE_CODE,
            full=True,
            recreate=False,
            verbose=False,
            config=config,
            conn=conn,
            home_root=home,
        )
        assert summary.indexed == 1
        assert conn.execute("SELECT input_tokens, output_tokens FROM session_state").fetchone() == (
            800,
            40,
        )
        scope = conn.execute(
            "SELECT session_id, scope_token, format_revision, committed_offset "
            "FROM claude_usage_scopes"
        ).fetchone()
        assert scope is not None
        assert scope[2] == 1
        assert scope[3] == transcript.stat().st_size
        assert conn.execute(
            "SELECT input_tokens, cache_creation_input_tokens, cache_read_input_tokens, "
            "output_tokens FROM claude_response_usage WHERE scope_token = ?",
            [scope[1]],
        ).fetchall() == [(100, 500, 200, 40)]


def test_batch_index_sessions_preserves_sidecar_mtime_and_skips_unchanged(tmp_path: Path) -> None:
    import duckdb

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    home = tmp_path / "home"
    grok_session = home / ".grok" / "sessions" / "s1"
    grok_session.mkdir(parents=True)
    (grok_session / "chat_history.jsonl").write_text(
        json.dumps({"type": "user", "message": "hello", "timestamp": "2026-01-01T00:00:00Z"})
        + "\n",
        encoding="utf-8",
    )
    (grok_session / "summary.json").write_text(
        json.dumps(
            {
                "created_at": "2026-01-01T00:00:00Z",
                "last_active_at": "2026-01-01T00:01:00Z",
                "current_model_id": "grok-4",
                "info": {"cwd": "/tmp/work"},
            }
        ),
        encoding="utf-8",
    )
    config = replace(
        AppConfig.load(),
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
    )
    ensure_sidecar(config)
    with duckdb.connect(str(config.db_path)) as conn:
        ensure_schema(conn, embed_dim=config.embedding.dimensions)
        first = index_sessions(
            source=Source.GROK,
            full=False,
            recreate=False,
            verbose=False,
            config=config,
            conn=conn,
            home_root=home,
        )
        assert first.indexed == 1
        sidecar_row = conn.execute("SELECT sidecar_mtime FROM session_state").fetchone()
        assert sidecar_row is not None and float(sidecar_row[0]) > 0.0

        second = index_sessions(
            source=Source.GROK,
            full=False,
            recreate=False,
            verbose=False,
            config=config,
            conn=conn,
            home_root=home,
        )
        assert second.indexed == 0
        assert second.skipped == 1
