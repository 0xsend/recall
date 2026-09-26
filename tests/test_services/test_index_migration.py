"""REQ-RECON-010..016: versioned historical index migration is not normal reconciliation."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from lane_harness import lane_config
from recall.core.config import AppConfig
from recall.db import connect
from recall.db.source_files import SourceCatalog, SourceFile, SourceSignature, source_key
from recall.parsers.claude_code import ClaudeCodeParser
from recall.parsers.codex import CodexParser
from recall.parsers.grok import GrokParser
from recall.parsers.kimi_code import KimiCodeParser
from recall.parsers.pi_agent import PiAgentParser
from recall.parsers.protocol import SessionParser
from recall.parsers.registry import ParserType
from recall.services.coordinator import (
    capture_path,
    commit_prepared_raw_sources,
    observe_path,
    prepare_raw_sources,
    set_paused,
)
from recall.services.index_migration import (
    INDEX_MIGRATION_VERSION,
    applied_version,
    begin_migration,
    captured_keys,
    is_eligible,
    mark_captured_complete,
    maybe_complete_migration,
    migration_status,
    rollback_migration,
    verify_and_complete,
)


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    cfg = lane_config(tmp_path)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
def conn(config: AppConfig) -> Iterator[duckdb.DuckDBPyConnection]:
    opened = connect(config)
    try:
        yield opened
    finally:
        opened.close()


def _write_transcript(path: Path, body: str = "hello") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _seed_current(conn: duckdb.DuckDBPyConnection, path: Path) -> None:
    catalog = SourceCatalog(conn, clock=lambda: 1.0)
    stat = path.stat()
    catalog.observe(
        "claude-code",
        str(path.parent),
        str(path),
        SourceSignature(
            dev=1,
            inode=1,
            ctime_ns=stat.st_ctime_ns,
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            parser_revision="rev-a",
        ),
    )
    conn.execute(
        """UPDATE source_files
           SET committed_generation = desired_generation, committed_offset = size
           WHERE source_path = ?""",
        [str(path)],
    )


def _desired(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
    row = conn.execute(
        "SELECT desired_generation, committed_generation FROM source_files WHERE source_path = ?",
        [str(path)],
    ).fetchone()
    assert row is not None
    return int(row[0])


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_database_records_current_version_and_is_not_eligible(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    assert applied_version(conn) == INDEX_MIGRATION_VERSION
    assert is_eligible(conn) is False
    status = migration_status(conn)
    assert status.phase == "idle"
    assert status.applied_version == INDEX_MIGRATION_VERSION
    assert status.eligible is False


def test_applied_behind_package_version_is_eligible(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    assert applied_version(conn) == 0
    assert is_eligible(conn) is True
    assert migration_status(conn).eligible is True


def test_pending_work_with_current_version_is_not_migration(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    path = _write_transcript(tmp_path / "sessions" / "pending.jsonl")
    _seed_current(conn, path)
    catalog = SourceCatalog(conn, clock=lambda: 2.0)
    catalog.force_reconcile("claude-code", str(path))
    pending = conn.execute(
        "SELECT COUNT(*) FROM source_files WHERE desired_generation > committed_generation"
    ).fetchone()
    assert pending == (1,)
    assert is_eligible(conn) is False
    assert migration_status(conn).phase == "idle"


def test_parser_revision_change_does_not_start_migration(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    path = _write_transcript(tmp_path / "sessions" / "rev.jsonl")
    _seed_current(conn, path)
    catalog = SourceCatalog(conn, clock=lambda: 2.0)
    stat = path.stat()
    catalog.observe(
        "claude-code",
        str(path.parent),
        str(path),
        SourceSignature(
            dev=1,
            inode=1,
            ctime_ns=stat.st_ctime_ns,
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            parser_revision="rev-b",
        ),
    )
    assert is_eligible(conn) is False
    assert migration_status(conn).phase == "idle"


def test_reopen_with_current_version_is_not_eligible(
    config: AppConfig, conn: duckdb.DuckDBPyConnection
) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    assert is_eligible(conn) is True
    conn.execute(
        "UPDATE runtime_state SET index_migration_version = ? WHERE singleton",
        [INDEX_MIGRATION_VERSION],
    )
    conn.close()
    reopened = connect(config)
    try:
        assert is_eligible(reopened) is False
        assert migration_status(reopened).phase == "idle"
    finally:
        reopened.close()


def test_begin_refuses_when_operator_pause_is_set(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    path = _write_transcript(tmp_path / "sessions" / "paused.jsonl")
    _seed_current(conn, path)
    set_paused(config, True)
    with pytest.raises(RuntimeError, match="paused"):
        begin_migration(conn, config)
    assert _desired(conn, path) == 1
    assert applied_version(conn) == 0


def test_begin_backs_up_before_marking_captured_sources_pending(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    path = _write_transcript(tmp_path / "sessions" / "hist.jsonl", "original")
    digest = _file_digest(path)
    _seed_current(conn, path)
    status = begin_migration(conn, config)
    assert status.phase == "running"
    assert status.backup_path is not None
    backup = Path(status.backup_path)
    assert (backup / "manifest.json").is_file()
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "index-migration"
    assert manifest["complete"] is True
    assert (backup / "recall.duckdb").is_file()
    assert _desired(conn, path) == 2
    assert _file_digest(path) == digest
    captured = captured_keys(conn)
    assert source_key("claude-code", str(path)) in captured
    assert status.captured == 1
    assert status.remaining == 1


def test_post_capture_files_are_not_in_captured_scope(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    first = _write_transcript(tmp_path / "sessions" / "one.jsonl")
    _seed_current(conn, first)
    begin_migration(conn, config)
    later = _write_transcript(tmp_path / "sessions" / "two.jsonl")
    _seed_current(conn, later)
    keys = captured_keys(conn)
    assert source_key("claude-code", str(first)) in keys
    assert source_key("claude-code", str(later)) not in keys
    later_row = conn.execute(
        "SELECT desired_generation, committed_generation FROM source_files WHERE source_path = ?",
        [str(later)],
    ).fetchone()
    assert later_row == (1, 1)


@pytest.mark.parametrize("phase", ["running", "failed"])
def test_interrupted_begin_resumes_same_backup_and_scope(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path, phase: str
) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    path = _write_transcript(tmp_path / "sessions" / "resume.jsonl")
    _seed_current(conn, path)
    first = begin_migration(conn, config)
    keys = captured_keys(conn)
    mark_captured_complete(conn, source_key("claude-code", str(path)))
    conn.execute("UPDATE index_migration_jobs SET phase = ? WHERE singleton", [phase])
    later = _write_transcript(tmp_path / "sessions" / "later.jsonl")
    _seed_current(conn, later)
    second = begin_migration(conn, config)
    assert second.phase == "running"
    assert second.backup_path == first.backup_path
    assert captured_keys(conn) == keys
    assert second.completed == 1
    assert second.remaining == 0
    assert _desired(conn, path) == 2


def test_complete_requires_verified_captured_scope(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    path = _write_transcript(tmp_path / "sessions" / "verify.jsonl")
    _seed_current(conn, path)
    begin_migration(conn, config)
    with pytest.raises(RuntimeError, match="captured"):
        verify_and_complete(conn, verified=True)
    assert applied_version(conn) == 0
    failed = verify_and_complete(conn, verified=False)
    assert failed.phase == "failed"
    assert applied_version(conn) == 0
    mark_captured_complete(conn, source_key("claude-code", str(path)))
    conn.execute(
        """UPDATE source_files
           SET committed_generation = desired_generation, committed_offset = size
           WHERE source_path = ?""",
        [str(path)],
    )
    done = verify_and_complete(conn, verified=True)
    assert done.phase == "idle"
    assert applied_version(conn) == INDEX_MIGRATION_VERSION
    assert is_eligible(conn) is False


def test_empty_captured_scope_cannot_complete_or_begin(
    config: AppConfig, conn: duckdb.DuckDBPyConnection
) -> None:
    """A catalog with no sources is not a verified historical migration."""
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    with pytest.raises(RuntimeError, match="captured"):
        begin_migration(conn, config)
    assert applied_version(conn) == 0
    conn.execute(
        """UPDATE index_migration_jobs
           SET phase = 'running', captured_count = 0, completed_count = 0
           WHERE singleton"""
    )
    status = maybe_complete_migration(conn)
    assert status.phase == "running"
    assert applied_version(conn) == 0
    with pytest.raises(RuntimeError, match="captured"):
        verify_and_complete(conn, verified=True)
    assert applied_version(conn) == 0


def test_rollback_restores_backup_and_leaves_version_unapplied(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    path = _write_transcript(tmp_path / "sessions" / "roll.jsonl")
    _seed_current(conn, path)
    status = begin_migration(conn, config)
    assert _desired(conn, path) == 2
    conn.close()
    restored = rollback_migration(config)
    assert status.backup_path is not None
    assert restored == Path(status.backup_path)
    reopened = connect(config)
    try:
        assert applied_version(reopened) == 0
        assert _desired(reopened, path) == 1
        assert migration_status(reopened).phase == "idle"
    finally:
        reopened.close()


def test_subsequent_transcript_changes_remain_for_normal_operation(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    first = _write_transcript(tmp_path / "sessions" / "old.jsonl")
    _seed_current(conn, first)
    begin_migration(conn, config)
    mark_captured_complete(conn, source_key("claude-code", str(first)))
    conn.execute(
        """UPDATE source_files
           SET committed_generation = desired_generation, committed_offset = size
           WHERE source_path = ?""",
        [str(first)],
    )
    later = _write_transcript(tmp_path / "sessions" / "new.jsonl")
    catalog = SourceCatalog(conn, clock=lambda: 9.0)
    stat = later.stat()
    catalog.observe(
        "claude-code",
        str(later.parent),
        str(later),
        SourceSignature(
            dev=2,
            inode=2,
            ctime_ns=stat.st_ctime_ns,
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            parser_revision="rev-a",
        ),
    )
    verify_and_complete(conn, verified=True)
    assert is_eligible(conn) is False
    later_row = conn.execute(
        "SELECT desired_generation, committed_generation FROM source_files WHERE source_path = ?",
        [str(later)],
    ).fetchone()
    assert later_row is not None
    assert later_row[0] > later_row[1] or later_row[0] == 1
    assert source_key("claude-code", str(later)) not in captured_keys(conn)
    status = migration_status(conn)
    assert status.phase == "idle"
    pending = conn.execute(
        """SELECT COUNT(*) FROM source_files
           WHERE source_path = ? AND desired_generation > committed_generation""",
        [str(later)],
    ).fetchone()
    # A newly observed file starts pending (desired 1, committed 0) until acknowledged.
    assert pending == (1,)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_UNSUPPORTED_RECORD = '{"type":"future_conversation","content":"unrecognized"}\n'
_SOURCE_CASES = (
    (ClaudeCodeParser, "claude_code/session1.jsonl", "session.jsonl"),
    (CodexParser, "codex/session1/rollout.jsonl", "rollout.jsonl"),
    (GrokParser, "grok/session1.jsonl", "chat_history.jsonl"),
    (PiAgentParser, "pi_agent/session1.jsonl", "session.jsonl"),
    (KimiCodeParser, "kimi_code/session1/agents/main/wire.jsonl", "wire.jsonl"),
)


def _install_fixture(tmp_path: Path, fixture: str, filename: str) -> Path:
    dest_dir = tmp_path / "transcripts" / fixture.replace("/", "-")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.copy(FIXTURES / fixture, dest)
    return dest.resolve()


def _parser_for(parser_type: ParserType, path: Path) -> SessionParser:
    return parser_type(roots=(path.parent,))


def _reconcile(
    parser: SessionParser, path: Path, config: AppConfig, conn: duckdb.DuckDBPyConnection
) -> int:
    item = observe_path(parser, capture_path(parser, path), conn=conn)
    prepared = prepare_raw_sources((item,), {parser.source.value: parser})
    return commit_prepared_raw_sources(prepared, config, conn=conn)


def _message_snapshot(conn: duckdb.DuckDBPyConnection) -> tuple[tuple[object, ...], ...]:
    return tuple(
        conn.execute(
            """SELECT messages.id, messages.session_id, messages.idx,
                      message_state.role, message_state.content
               FROM messages
               JOIN message_state ON message_state.message_id = messages.id
               ORDER BY messages.session_id, messages.idx"""
        ).fetchall()
    )


def _tool_snapshot(conn: duckdb.DuckDBPyConnection) -> tuple[tuple[object, ...], ...]:
    return tuple(
        conn.execute(
            "SELECT id, session_id, message_id, idx, tool_name FROM tool_calls "
            "ORDER BY session_id, idx, id"
        ).fetchall()
    )


def _catalog_row(conn: duckdb.DuckDBPyConnection, parser: SessionParser, path: Path) -> SourceFile:
    item = SourceCatalog(conn, clock=lambda: 1.0).get(parser.source.value, str(path))
    assert item is not None
    return item


def _backup_digest(backup: Path) -> str:
    return hashlib.sha256((backup / "recall.duckdb").read_bytes()).hexdigest()


def _index_supported_then_append_unsupported(
    parser_type: ParserType,
    fixture: str,
    filename: str,
    tmp_path: Path,
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[SessionParser, Path]:
    path = _install_fixture(tmp_path, fixture, filename)
    parser = _parser_for(parser_type, path)
    _reconcile(parser, path, config, conn)
    assert _message_snapshot(conn)
    path.write_text(path.read_text(encoding="utf-8") + _UNSUPPORTED_RECORD, encoding="utf-8")
    _reconcile(parser, path, config, conn)
    return parser, path


@pytest.mark.parametrize(("parser_type", "fixture", "filename"), _SOURCE_CASES)
def test_unsupported_history_does_not_settle_captured_key_outside_migration(
    parser_type: ParserType,
    fixture: str,
    filename: str,
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    tmp_path: Path,
) -> None:
    parser, path = _index_supported_then_append_unsupported(
        parser_type, fixture, filename, tmp_path, config, conn
    )
    item = _catalog_row(conn, parser, path)
    assert item.last_error == "unsupported"
    assert item.current is False
    assert captured_keys(conn) == ()
    assert migration_status(conn).phase == "idle"
    assert applied_version(conn) == INDEX_MIGRATION_VERSION


@pytest.mark.parametrize(("parser_type", "fixture", "filename"), _SOURCE_CASES)
def test_migration_settles_history_preserving_unsupported_sources(
    parser_type: ParserType,
    fixture: str,
    filename: str,
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    tmp_path: Path,
) -> None:
    parser, path = _index_supported_then_append_unsupported(
        parser_type, fixture, filename, tmp_path, config, conn
    )
    messages = _message_snapshot(conn)
    tools = _tool_snapshot(conn)
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    status = begin_migration(conn, config)
    assert status.phase == "running"
    assert status.remaining == 1
    assert status.backup_path is not None
    backup = Path(status.backup_path)
    original_backup = _backup_digest(backup)
    key = source_key(parser.source.value, str(path))
    assert key in captured_keys(conn)
    _reconcile(parser, path, config, conn)
    item = _catalog_row(conn, parser, path)
    assert item.last_error == "unsupported"
    assert item.current is False
    records = json.loads(item.diagnostics or "{}").get("records")
    assert isinstance(records, list) and records
    assert all(
        isinstance(record, dict)
        and record.get("kind") == "unsupported_record"
        and "future_conversation" in str(record.get("detail"))
        and isinstance(record.get("byte_offset"), int)
        and record["byte_offset"] >= 0
        for record in records
    )
    assert _message_snapshot(conn) == messages
    assert _tool_snapshot(conn) == tools
    assert _backup_digest(backup) == original_backup
    done = migration_status(conn)
    assert done.phase == "idle"
    assert done.remaining == 0
    assert done.completed == 1
    assert applied_version(conn) == INDEX_MIGRATION_VERSION
    assert captured_keys(conn) == ()


def test_malformed_history_does_not_settle_a_captured_key(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    path = _install_fixture(tmp_path, "claude_code/session1.jsonl", "session.jsonl")
    parser = _parser_for(ClaudeCodeParser, path)
    _reconcile(parser, path, config, conn)
    messages = _message_snapshot(conn)
    path.write_text(path.read_text(encoding="utf-8") + "{bad}\n", encoding="utf-8")
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    begin_migration(conn, config)
    _reconcile(parser, path, config, conn)
    item = _catalog_row(conn, parser, path)
    assert item.last_error != "unsupported"
    assert item.retry_count >= 1
    assert _message_snapshot(conn) == messages
    status = migration_status(conn)
    assert status.phase == "running"
    assert status.remaining == 1
    assert applied_version(conn) == 0
    assert source_key(parser.source.value, str(path)) in captured_keys(conn)


def test_unsupported_settlement_survives_interrupt_and_resume(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    supported = _install_fixture(tmp_path, "claude_code/session1.jsonl", "supported.jsonl")
    blocked = _install_fixture(tmp_path, "claude_code/session1.jsonl", "blocked.jsonl")
    supported_parser = _parser_for(ClaudeCodeParser, supported)
    blocked_parser = _parser_for(ClaudeCodeParser, blocked)
    _reconcile(supported_parser, supported, config, conn)
    _reconcile(blocked_parser, blocked, config, conn)
    blocked.write_text(blocked.read_text(encoding="utf-8") + _UNSUPPORTED_RECORD, encoding="utf-8")
    _reconcile(blocked_parser, blocked, config, conn)
    messages = _message_snapshot(conn)
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    first = begin_migration(conn, config)
    assert first.remaining == 2
    assert first.backup_path is not None
    original_backup = _backup_digest(Path(first.backup_path))
    _reconcile(blocked_parser, blocked, config, conn)
    interrupted = migration_status(conn)
    assert interrupted.phase == "running"
    assert interrupted.completed == 1
    assert interrupted.remaining == 1
    assert _catalog_row(conn, blocked_parser, blocked).last_error == "unsupported"
    resumed = begin_migration(conn, config)
    assert resumed.backup_path == first.backup_path
    assert resumed.completed == 1
    assert resumed.remaining == 1
    assert captured_keys(conn) == tuple(
        sorted(
            (
                source_key(supported_parser.source.value, str(supported)),
                source_key(blocked_parser.source.value, str(blocked)),
            )
        )
    )
    _reconcile(supported_parser, supported, config, conn)
    done = migration_status(conn)
    assert done.phase == "idle"
    assert done.remaining == 0
    assert applied_version(conn) == INDEX_MIGRATION_VERSION
    assert _message_snapshot(conn) == messages
    assert _backup_digest(Path(first.backup_path)) == original_backup
    assert _catalog_row(conn, blocked_parser, blocked).last_error == "unsupported"
    assert _catalog_row(conn, blocked_parser, blocked).current is False
    assert _catalog_row(conn, supported_parser, supported).current is True


# --- rollback backup presence (REQ-RECON-010) ----------------------------


def _reported_migration(config: AppConfig, conn: duckdb.DuckDBPyConnection) -> dict:
    from recall.services.coordinator import reconciliation_status

    migration = reconciliation_status(config, conn=conn)["index_migration"]
    assert isinstance(migration, dict)
    return migration


def test_status_reports_a_backup_that_is_still_on_disk(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    _seed_current(conn, _write_transcript(tmp_path / "sessions" / "hist.jsonl", "original"))
    begin_migration(conn, config)

    migration = _reported_migration(config, conn)

    assert migration["backup_path"] is not None
    assert migration["backup_path_present"] is True


def test_status_reports_a_deleted_backup_as_absent(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    """The recorded path is the operator's rollback; a stale pointer reads as one
    that still exists, which is the opposite of what a rollback needs to know."""
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    _seed_current(conn, _write_transcript(tmp_path / "sessions" / "hist.jsonl", "original"))
    status = begin_migration(conn, config)
    assert status.backup_path is not None
    shutil.rmtree(status.backup_path)

    migration = _reported_migration(config, conn)

    assert migration["backup_path"] == status.backup_path
    assert migration["backup_path_present"] is False


def test_status_reports_no_presence_when_no_backup_was_taken(
    config: AppConfig, conn: duckdb.DuckDBPyConnection
) -> None:
    """A job that never took a backup has nothing to be present or missing."""
    migration = _reported_migration(config, conn)

    assert migration["backup_path"] is None
    assert migration["backup_path_present"] is None
