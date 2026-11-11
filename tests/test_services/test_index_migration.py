"""REQ-RECON-010..016: versioned historical index migration is not normal reconciliation."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from lane_harness import lane_config
from recall.core.config import AppConfig
from recall.db import connect
from recall.db.lease import StandaloneLifetime
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
        begin_migration(conn, config, observer=StandaloneLifetime())
    assert _desired(conn, path) == 1
    assert applied_version(conn) == 0


def test_begin_backs_up_before_marking_captured_sources_pending(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    path = _write_transcript(tmp_path / "sessions" / "hist.jsonl", "original")
    digest = _file_digest(path)
    _seed_current(conn, path)
    status = begin_migration(conn, config, observer=StandaloneLifetime())
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
    begin_migration(conn, config, observer=StandaloneLifetime())
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
    first = begin_migration(conn, config, observer=StandaloneLifetime())
    keys = captured_keys(conn)
    mark_captured_complete(conn, source_key("claude-code", str(path)))
    conn.execute("UPDATE index_migration_jobs SET phase = ? WHERE singleton", [phase])
    later = _write_transcript(tmp_path / "sessions" / "later.jsonl")
    _seed_current(conn, later)
    second = begin_migration(conn, config, observer=StandaloneLifetime())
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
    begin_migration(conn, config, observer=StandaloneLifetime())
    with pytest.raises(RuntimeError, match="captured"):
        verify_and_complete(conn, verified=True, observer=StandaloneLifetime())
    assert applied_version(conn) == 0
    failed = verify_and_complete(conn, verified=False, observer=StandaloneLifetime())
    assert failed.phase == "failed"
    assert applied_version(conn) == 0
    mark_captured_complete(conn, source_key("claude-code", str(path)))
    conn.execute(
        """UPDATE source_files
           SET committed_generation = desired_generation, committed_offset = size
           WHERE source_path = ?""",
        [str(path)],
    )
    done = verify_and_complete(conn, verified=True, observer=StandaloneLifetime())
    assert done.phase == "idle"
    assert applied_version(conn) == INDEX_MIGRATION_VERSION
    assert is_eligible(conn) is False


def test_empty_captured_scope_cannot_complete_or_begin(
    config: AppConfig, conn: duckdb.DuckDBPyConnection
) -> None:
    """A catalog with no sources is not a verified historical migration."""
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    with pytest.raises(RuntimeError, match="captured"):
        begin_migration(conn, config, observer=StandaloneLifetime())
    assert applied_version(conn) == 0
    conn.execute(
        """UPDATE index_migration_jobs
           SET phase = 'running', captured_count = 0, completed_count = 0
           WHERE singleton"""
    )
    status = maybe_complete_migration(conn, observer=StandaloneLifetime())
    assert status.phase == "running"
    assert applied_version(conn) == 0
    with pytest.raises(RuntimeError, match="captured"):
        verify_and_complete(conn, verified=True, observer=StandaloneLifetime())
    assert applied_version(conn) == 0


def test_rollback_restores_backup_and_leaves_version_unapplied(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    conn.execute("UPDATE runtime_state SET index_migration_version = 0 WHERE singleton")
    path = _write_transcript(tmp_path / "sessions" / "roll.jsonl")
    _seed_current(conn, path)
    status = begin_migration(conn, config, observer=StandaloneLifetime())
    assert _desired(conn, path) == 2
    conn.close()
    restored = rollback_migration(config, preserve=config.data_dir.resolve().parent / "failed-set")
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
    begin_migration(conn, config, observer=StandaloneLifetime())
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
        ),
    )
    verify_and_complete(conn, verified=True, observer=StandaloneLifetime())
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
    return commit_prepared_raw_sources(prepared, config, conn=conn).changed


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
    assert item.current is True
    assert item.committed_generation == item.desired_generation
    assert item.committed_offset == path.stat().st_size
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
    status = begin_migration(conn, config, observer=StandaloneLifetime())
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
    assert item.current is True
    assert item.committed_generation == item.desired_generation
    assert item.committed_offset == path.stat().st_size
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
    begin_migration(conn, config, observer=StandaloneLifetime())
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
    first = begin_migration(conn, config, observer=StandaloneLifetime())
    assert first.remaining == 2
    assert first.backup_path is not None
    original_backup = _backup_digest(Path(first.backup_path))
    _reconcile(blocked_parser, blocked, config, conn)
    interrupted = migration_status(conn)
    assert interrupted.phase == "running"
    assert interrupted.completed == 1
    assert interrupted.remaining == 1
    assert _catalog_row(conn, blocked_parser, blocked).last_error == "unsupported"
    resumed = begin_migration(conn, config, observer=StandaloneLifetime())
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
    blocked_item = _catalog_row(conn, blocked_parser, blocked)
    assert blocked_item.current is True
    assert blocked_item.committed_generation == blocked_item.desired_generation
    assert blocked_item.committed_offset == blocked.stat().st_size
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
    begin_migration(conn, config, observer=StandaloneLifetime())

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
    status = begin_migration(conn, config, observer=StandaloneLifetime())
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


def test_copied_database_status_does_not_probe_source_backup(
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.backup_access_guard import ForeignAccessGuard

    source = tmp_path / "source-installation" / "snapshots" / "original"
    source.mkdir(parents=True)
    conn.execute("UPDATE index_migration_jobs SET backup_path=?", [str(source)])
    guard = ForeignAccessGuard(monkeypatch, source.parent.parent)
    migration = _reported_migration(config, conn)
    assert (migration["backup_path"], migration["backup_path_present"]) == (str(source), None)
    assert "backup_path_scope" not in migration
    assert guard.attempts == []


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (None, None),
        ("", None),
        ("snapshots/b", True),
        ("snapshots/missing", False),
        ("snapshots/./b", True),
        ("snapshots/x/../b", True),
        ("../source/b", None),
        ("../current/snapshots/b", None),
        (".", True),
        ("snapshots/..", True),
        ("file", True),
        ("file/b", False),
        ("internal/b", None),
        ("external/b", None),
        ("dangling/b", None),
        ("cycle/b", None),
        ("final-internal", None),
        ("final-external", None),
        ("dangling", None),
        ("cycle", None),
        ("snapshots/link/../b", True),
        ("bad\0name", None),
        ("bad\ud800name", None),
        ("x" * 4097, None),
        ("/".join(["x"] * 257), None),
        ("~/backup", False),
    ],
)
def test_backup_status_path_matrix(
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: str | None,
    expected: bool | None,
) -> None:
    from tests.backup_access_guard import ForeignAccessGuard

    root = config.data_dir
    (root / "snapshots/b").mkdir(parents=True)
    (root / "file").write_text("ordinary file")
    source = tmp_path / "source"
    source.mkdir()
    (root / "internal").symlink_to("snapshots", target_is_directory=True)
    (root / "external").symlink_to(source, target_is_directory=True)
    (root / "final-internal").symlink_to("snapshots/b", target_is_directory=True)
    (root / "final-external").symlink_to(source)
    (root / "dangling").symlink_to(source / "absent")
    (root / "cycle").symlink_to("cycle")
    (root / "snapshots/link").symlink_to(source)
    # DuckDB's UTF-8 boundary cannot store an unpaired surrogate. Exercise that
    # input at the access boundary instead; all serializable cases use real DB.
    if record is not None and "\ud800" in record:
        from recall.services.migration_backup import backup_presence

        guard = ForeignAccessGuard(monkeypatch, source)
        assert backup_presence(config, record) is None
    else:
        conn.execute("UPDATE index_migration_jobs SET backup_path=?", [record])
        guard = ForeignAccessGuard(monkeypatch, source)
        migration = _reported_migration(config, conn)
        assert (migration["backup_path"], migration["backup_path_present"]) == (
            record or None,
            expected,
        )
        assert "backup_path_scope" not in migration
    assert guard.attempts == []


@pytest.mark.parametrize(
    "spelling", ["absolute", "dotted", "outside", "prefix", "return", "slashes", "alias"]
)
def test_backup_absolute_containment(
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spelling: str,
) -> None:
    from tests.backup_access_guard import ForeignAccessGuard

    root = config.data_dir
    (root / "snapshots/b").mkdir(parents=True)
    foreign = root.parent / (root.name + "-other")
    (foreign / "snapshots/b").mkdir(parents=True)
    alias = tmp_path / "source-alias"
    alias.symlink_to(foreign, target_is_directory=True)
    values = {
        "alias": (str(alias / "snapshots/b"), None),
        "absolute": (str(root / "snapshots/b"), True),
        "dotted": (str(root) + "/snapshots/./b", True),
        "outside": (str(foreign / "snapshots/b"), None),
        "prefix": (str(root) + "-other/snapshots/b", None),
        "return": (str(root) + "/../" + root.name + "/snapshots/b", True),
        "slashes": ("//" + str(root).lstrip("/") + "//snapshots//b", True),
    }
    record, expected = values[spelling]
    conn.execute("UPDATE index_migration_jobs SET backup_path=?", [record])
    monkeypatch.chdir(tmp_path)
    guard = ForeignAccessGuard(monkeypatch, foreign, alias)
    migration = _reported_migration(config, conn)
    assert (migration["backup_path"], migration["backup_path_present"]) == (record, expected)
    assert guard.attempts == []


@pytest.mark.parametrize(
    "operation",
    [
        "stat",
        "lstat",
        "open",
        "builtin",
        "io",
        "readlink",
        "resolve",
        "realpath",
        "access",
        "listdir",
        "scandir",
        "dirfd",
        "read",
        "dup",
    ],
)
def test_foreign_effect_recorder_rejects_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    import builtins
    import io
    import os

    from tests.backup_access_guard import ForeignAccessGuard

    source = tmp_path / "source"
    source.mkdir()
    (source / "input").write_bytes(b"source")
    descriptor = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    payload = os.open(source / "input", os.O_RDONLY)
    try:
        guard = ForeignAccessGuard(monkeypatch, source)
        guard.descriptors.update({descriptor: str(source), payload: str(source / "input")})

        def open_control(module):
            with module.open(source / "input", "rb"):
                pytest.fail("forbidden open was not rejected")

        calls = {
            "stat": lambda: os.stat(source),
            "lstat": lambda: os.lstat(source),
            "open": lambda: os.open(source / "input", os.O_RDONLY),
            "builtin": lambda: open_control(builtins),
            "io": lambda: open_control(io),
            "readlink": lambda: os.readlink(source / "link"),
            "resolve": lambda: source.resolve(),
            "realpath": lambda: os.path.realpath(source),
            "access": lambda: os.access(source, os.R_OK),
            "listdir": lambda: os.listdir(source),
            "scandir": lambda: os.scandir(source),
            "dirfd": lambda: os.open("input", os.O_RDONLY, dir_fd=descriptor),
            "read": lambda: os.read(payload, 1),
            "dup": lambda: os.dup(payload),
        }
        with pytest.raises(AssertionError, match="forbidden source effect"):
            calls[operation]()
        assert len(guard.attempts) == 1
    finally:
        os.close(payload)
        os.close(descriptor)


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("move", [False, True])
def test_closed_database_copy_and_move_backup_identity(
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: bool,
    move: bool,
) -> None:
    from dataclasses import replace

    from tests.backup_access_guard import ForeignAccessGuard

    source = config.data_dir
    (source / "snapshots/b").mkdir(parents=True)
    recorded = "snapshots/b" if relative else str(source / "snapshots/b")
    conn.execute("UPDATE index_migration_jobs SET backup_path=?", [recorded])
    conn.execute("CHECKPOINT")
    conn.close()
    destination = tmp_path / "copy"
    if move:
        shutil.move(str(source), destination)
    else:
        shutil.copytree(source, destination)
    copied_config = replace(config, data_dir=destination, db_path=destination / "recall.duckdb")
    copied = connect(copied_config)
    try:
        guard = ForeignAccessGuard(monkeypatch, source)
        status = _reported_migration(copied_config, copied)
        assert (status["backup_path"], status["backup_path_present"]) == (
            recorded,
            True if relative else None,
        )
        shutil.rmtree(destination / "snapshots/b")
        status = _reported_migration(copied_config, copied)
        assert (status["backup_path"], status["backup_path_present"]) == (
            recorded,
            False if relative else None,
        )
        assert guard.attempts == []
    finally:
        copied.close()


@pytest.mark.parametrize("statement", ["DELETE FROM index_migration_scope", "COMMIT"])
def test_migration_primary_is_observed_before_independent_rollback_failure(
    config: AppConfig, conn: duckdb.DuckDBPyConnection, tmp_path: Path, statement: str
):
    conn.execute("UPDATE runtime_state SET index_migration_version=0 WHERE singleton")
    path = _write_transcript(tmp_path / "sessions" / "kept.jsonl")
    _seed_current(conn, path)
    observed = []
    at_rollback = []
    primary = duckdb.FatalException("Failed to delete all rows from index in migration")
    cleanup = duckdb.FatalException("database has been invalidated during rollback")

    class Observer(StandaloneLifetime):
        def observe(self, error, operation):
            observed.append(error)
            return super().observe(error, operation)

    class Connection:
        def register(self, *args):
            return conn.register(*args)

        def unregister(self, *args):
            return conn.unregister(*args)

        def execute(self, sql, params=None):
            if sql == statement:
                raise primary
            if sql == "ROLLBACK":
                at_rollback.extend(observed)
                conn.execute(sql)
                raise cleanup
            return conn.execute(sql) if params is None else conn.execute(sql, params)

    observer = Observer()
    with pytest.raises(BaseExceptionGroup):
        observer.call_db(
            "begin_migration",
            lambda: begin_migration(cast(Any, Connection()), config, observer=observer),
        )
    assert at_rollback and at_rollback[0] is primary
    assert observed[0] is primary
    assert migration_status(conn).phase == "idle"
    assert _desired(conn, path) == 1
