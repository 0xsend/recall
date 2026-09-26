from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import pytest
import recall.services.daemon as daemon_module
import recall.services.fts_sidecar_reconcile as reconcile_module
import recall.services.indexer as indexer_module
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.models import Message, ParseResult, Session, ToolCall
from recall.core.types import DaemonMode, Role, Source
from recall.db import connect, insert_messages, insert_session
from recall.db.fts_sidecar import (
    get_fts_fields_signature,
    open_sidecar,
    search_messages_fts,
    search_tool_calls_fts,
    sidecar_path,
    upsert_message_fts,
    upsert_tool_call_fts,
)
from recall.services.daemon import run_startup_fts_sidecar_sync
from recall.services.fts_sidecar_bootstrap import bootstrap_sidecar
from recall.services.fts_sidecar_reconcile import reconcile_sidecar
from recall.services.watcher import index_single_session


def _app_config(tmp_path: Path, *, fts: FtsConfig) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=fts,
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(mode=DaemonMode.POLL),
        cli=CliConfig(),
    )


def _create_sidecar_source_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE message_state (
            message_id TEXT PRIMARY KEY,
            fts_content TEXT DEFAULT '',
            fts_thinking TEXT DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE tool_calls (
            id TEXT PRIMARY KEY,
            bash_command TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE fts_sidecar_pending (
            kind TEXT NOT NULL,
            id TEXT NOT NULL,
            op TEXT NOT NULL,
            queued_at TIMESTAMP NOT NULL DEFAULT now()
        )
        """
    )


def _message_ids(conn, query: str, fields: list[str]) -> list[str]:
    return [message_id for message_id, _score in search_messages_fts(conn, query, fields, 10)]


def _tool_call_ids(conn, query: str) -> list[str]:
    return [tool_call_id for tool_call_id, _score in search_tool_calls_fts(conn, query, 10)]


def test_bootstrap_sidecar_respects_configured_fields(tmp_path: Path) -> None:
    duckdb_conn = duckdb.connect(":memory:")
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _create_sidecar_source_tables(duckdb_conn)
        duckdb_conn.execute(
            "INSERT INTO message_state VALUES (?, ?, ?)",
            ["msg-bootstrap", "bootstrapcontentterm", "bootstrapthinkingterm"],
        )
        duckdb_conn.execute(
            "INSERT INTO tool_calls VALUES (?, ?)",
            ["tool-bootstrap", "echo bootstrapbashterm"],
        )

        bootstrap_sidecar(
            duckdb_conn,
            sidecar_conn,
            fts_fields=("content",),
            batch_size=10,
        )

        assert _message_ids(sidecar_conn, "bootstrapcontentterm", ["content"]) == ["msg-bootstrap"]
        assert _message_ids(sidecar_conn, "bootstrapthinkingterm", ["content", "thinking"]) == []
        assert _tool_call_ids(sidecar_conn, "bootstrapbashterm") == []
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_reconcile_sidecar_respects_configured_fields(tmp_path: Path) -> None:
    duckdb_conn = duckdb.connect(":memory:")
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        _create_sidecar_source_tables(duckdb_conn)
        duckdb_conn.executemany(
            "INSERT INTO message_state VALUES (?, ?, ?)",
            [
                ("msg-pending", "pendingcontentterm", "pendingthinkingterm"),
                ("msg-orphan", "orphancontentterm", "orphanthinkingterm"),
            ],
        )
        duckdb_conn.executemany(
            "INSERT INTO tool_calls VALUES (?, ?)",
            [
                ("tool-pending", "echo pendingbashterm"),
                ("tool-orphan", "echo orphanbashterm"),
            ],
        )
        duckdb_conn.executemany(
            "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES (?, ?, ?)",
            [
                ("message", "msg-pending", "upsert"),
                ("tool_call", "tool-pending", "upsert"),
            ],
        )

        stats = reconcile_sidecar(
            duckdb_conn,
            sidecar_conn,
            fts_fields=("content",),
            batch_size=10,
        )

        assert stats.pending_drained == {"message": 1, "tool_call": 1}
        assert _message_ids(sidecar_conn, "pendingcontentterm", ["content"]) == ["msg-pending"]
        assert _message_ids(sidecar_conn, "orphancontentterm", ["content"]) == ["msg-orphan"]
        assert _message_ids(sidecar_conn, "pendingthinkingterm", ["content", "thinking"]) == []
        assert _message_ids(sidecar_conn, "orphanthinkingterm", ["content", "thinking"]) == []
        assert _tool_call_ids(sidecar_conn, "pendingbashterm") == []
        assert _tool_call_ids(sidecar_conn, "orphanbashterm") == []
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_reconcile_sidecar_streams_differences_without_full_id_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duckdb_conn = duckdb.connect(":memory:")
    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")

    def fail_full_id_materialization(*_args: object, **_kwargs: object) -> set[str]:
        raise AssertionError("reconcile must not materialize the full sidecar or DuckDB id set")

    monkeypatch.setattr(
        reconcile_module,
        "_sidecar_ids",
        fail_full_id_materialization,
        raising=False,
    )
    monkeypatch.setattr(
        reconcile_module,
        "_duckdb_ids",
        fail_full_id_materialization,
        raising=False,
    )

    try:
        _create_sidecar_source_tables(duckdb_conn)
        duckdb_conn.executemany(
            "INSERT INTO message_state VALUES (?, ?, ?)",
            [
                ("msg-consistent", "consistentcontentterm", "consistentthinkingterm"),
                ("msg-orphan", "orphancontentterm", "orphanthinkingterm"),
                ("msg-pending", "pendingcontentterm", "pendingthinkingterm"),
            ],
        )
        duckdb_conn.executemany(
            "INSERT INTO tool_calls VALUES (?, ?)",
            [
                ("tool-consistent", "echo consistentbashterm"),
                ("tool-orphan", "echo orphanbashterm"),
                ("tool-pending", "echo pendingbashterm"),
                ("tool-null", None),
            ],
        )
        duckdb_conn.executemany(
            "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES (?, ?, ?)",
            [
                ("message", "msg-pending", "upsert"),
                ("tool_call", "tool-pending", "upsert"),
            ],
        )
        upsert_message_fts(
            sidecar_conn,
            "msg-consistent",
            "consistentcontentterm",
            "consistentthinkingterm",
            fields=("content", "bash"),
        )
        upsert_message_fts(
            sidecar_conn,
            "msg-ghost",
            "ghostcontentterm",
            "ghostthinkingterm",
            fields=("content", "bash"),
        )
        upsert_tool_call_fts(
            sidecar_conn,
            "tool-consistent",
            "echo consistentbashterm",
            fields=("content", "bash"),
        )
        upsert_tool_call_fts(
            sidecar_conn,
            "tool-ghost",
            "echo ghostbashterm",
            fields=("content", "bash"),
        )

        stats = reconcile_sidecar(
            duckdb_conn,
            sidecar_conn,
            fts_fields=("content", "bash"),
            batch_size=2,
        )

        assert stats.pending_drained == {"message": 1, "tool_call": 1}
        assert stats.orphans_backfilled == {"message": 1, "tool_call": 1}
        assert stats.ghosts_deleted == {"message": 1, "tool_call": 1}
        assert stats.pending_remaining == {"message": 0, "tool_call": 0}
        assert sidecar_conn.execute(
            "SELECT message_id FROM message_fts_rowid ORDER BY message_id"
        ).fetchall() == [("msg-consistent",), ("msg-orphan",), ("msg-pending",)]
        assert sidecar_conn.execute(
            "SELECT tool_call_id FROM tool_calls_fts_rowid ORDER BY tool_call_id"
        ).fetchall() == [("tool-consistent",), ("tool-orphan",), ("tool-pending",)]
        assert _message_ids(sidecar_conn, "ghostcontentterm", ["content"]) == []
        assert _message_ids(sidecar_conn, "orphancontentterm", ["content"]) == ["msg-orphan"]
        assert _message_ids(sidecar_conn, "pendingcontentterm", ["content"]) == ["msg-pending"]
        assert _message_ids(sidecar_conn, "orphanthinkingterm", ["content", "thinking"]) == []
        assert _tool_call_ids(sidecar_conn, "ghostbashterm") == []
        assert _tool_call_ids(sidecar_conn, "orphanbashterm") == ["tool-orphan"]
        assert _tool_call_ids(sidecar_conn, "pendingbashterm") == ["tool-pending"]
    finally:
        sidecar_conn.close()
        duckdb_conn.close()


def test_startup_sidecar_sync_respects_configured_fields(tmp_path: Path) -> None:
    config = _app_config(
        tmp_path,
        fts=FtsConfig(fields=("content",), backend="sqlite_sidecar"),
    )
    conn = connect(config)
    try:
        conn.execute(
            """
            INSERT INTO message_state (
                message_id,
                role,
                content,
                thinking,
                has_thinking,
                fts_content,
                fts_thinking
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "msg-startup",
                "assistant",
                "startupcontentterm",
                "startupthinkingterm",
                True,
                "startupcontentterm",
                "startupthinkingterm",
            ],
        )
    finally:
        conn.close()

    result = run_startup_fts_sidecar_sync(config)

    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        assert result.enabled is True
        assert _message_ids(sidecar_conn, "startupcontentterm", ["content"]) == ["msg-startup"]
        assert _message_ids(sidecar_conn, "startupthinkingterm", ["content", "thinking"]) == []
    finally:
        sidecar_conn.close()


def test_startup_sidecar_sync_rescopes_messages_when_thinking_enabled(
    tmp_path: Path,
) -> None:
    config = _app_config(
        tmp_path,
        fts=FtsConfig(fields=("content",), backend="sqlite_sidecar"),
    )
    conn = connect(config)
    try:
        conn.execute(
            """
            INSERT INTO message_state (
                message_id,
                role,
                content,
                thinking,
                has_thinking,
                fts_content,
                fts_thinking
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "msg-enable-thinking",
                "assistant",
                "enablecontentterm",
                "enablethinkingterm",
                True,
                "enablecontentterm",
                "enablethinkingterm",
            ],
        )
    finally:
        conn.close()

    first_result = run_startup_fts_sidecar_sync(config)
    enabled_config = replace(
        config,
        fts=FtsConfig(fields=("content", "thinking"), backend="sqlite_sidecar"),
    )
    second_result = run_startup_fts_sidecar_sync(enabled_config)

    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        assert first_result.enabled is True
        assert second_result.enabled is True
        assert _message_ids(sidecar_conn, "enablethinkingterm", ["thinking"]) == [
            "msg-enable-thinking"
        ]
    finally:
        sidecar_conn.close()


def test_startup_sidecar_sync_rescopes_messages_when_thinking_disabled(
    tmp_path: Path,
) -> None:
    config = _app_config(
        tmp_path,
        fts=FtsConfig(fields=("content", "thinking"), backend="sqlite_sidecar"),
    )
    conn = connect(config)
    try:
        conn.execute(
            """
            INSERT INTO message_state (
                message_id,
                role,
                content,
                thinking,
                has_thinking,
                fts_content,
                fts_thinking
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "msg-disable-thinking",
                "assistant",
                "disablecontentterm",
                "disablethinkingterm",
                True,
                "disablecontentterm",
                "disablethinkingterm",
            ],
        )
    finally:
        conn.close()

    first_result = run_startup_fts_sidecar_sync(config)
    disabled_config = replace(
        config,
        fts=FtsConfig(fields=("content",), backend="sqlite_sidecar"),
    )
    second_result = run_startup_fts_sidecar_sync(disabled_config)

    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        assert first_result.enabled is True
        assert second_result.enabled is True
        assert _message_ids(sidecar_conn, "disablethinkingterm", ["content", "thinking"]) == []
    finally:
        sidecar_conn.close()


def test_startup_sidecar_sync_rescopes_tool_calls_when_bash_field_toggles(
    tmp_path: Path,
) -> None:
    config = _app_config(
        tmp_path,
        fts=FtsConfig(fields=("content", "bash"), backend="sqlite_sidecar"),
    )
    conn = connect(config)
    try:
        conn.execute(
            """
            INSERT INTO tool_calls (
                id,
                session_id,
                message_id,
                idx,
                tool_name,
                tool_input,
                bash_command
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "tool-toggle-bash",
                "session-toggle-bash",
                None,
                0,
                "Bash",
                "{}",
                "echo togglebashterm",
            ],
        )
    finally:
        conn.close()

    run_startup_fts_sidecar_sync(config)
    disabled_config = replace(
        config,
        fts=FtsConfig(fields=("content",), backend="sqlite_sidecar"),
    )
    run_startup_fts_sidecar_sync(disabled_config)
    enabled_config = replace(
        config,
        fts=FtsConfig(fields=("content", "bash"), backend="sqlite_sidecar"),
    )
    run_startup_fts_sidecar_sync(enabled_config)

    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        assert _tool_call_ids(sidecar_conn, "togglebashterm") == ["tool-toggle-bash"]

        run_startup_fts_sidecar_sync(disabled_config)

        assert _tool_call_ids(sidecar_conn, "togglebashterm") == []
    finally:
        sidecar_conn.close()


def test_startup_sidecar_sync_skips_rescope_when_fields_signature_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _app_config(
        tmp_path,
        fts=FtsConfig(fields=("content",), backend="sqlite_sidecar"),
    )
    conn = connect(config)
    try:
        conn.execute(
            """
            INSERT INTO message_state (
                message_id,
                role,
                content,
                thinking,
                has_thinking,
                fts_content,
                fts_thinking
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "msg-noop-signature",
                "assistant",
                "noopcontentterm",
                "noopthinkingterm",
                True,
                "noopcontentterm",
                "noopthinkingterm",
            ],
        )
    finally:
        conn.close()

    run_startup_fts_sidecar_sync(config)
    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        assert get_fts_fields_signature(sidecar_conn) == "content"
    finally:
        sidecar_conn.close()

    def fail_rescope(*args: object, **kwargs: object) -> None:
        raise AssertionError("unchanged FTS field signature must not trigger re-scope")

    monkeypatch.setattr(daemon_module, "rescope_sidecar_for_fields", fail_rescope)

    result = daemon_module.run_startup_fts_sidecar_sync(config)

    assert result.enabled is True


class _StaticParser:
    source = Source.CODEX

    def __init__(self, session: Session) -> None:
        self._session = session

    def discover(self) -> list[Path]:
        return []

    def sidecar_paths(self, path: Path) -> list[Path]:
        _ = path
        return []

    def parse(
        self,
        path: Path,
        *,
        offset: int = 0,
        message_idx_base: int = 0,
        orphan_tool_call_idx_base: int = 0,
        resume_state: Mapping[str, Any] | None = None,
    ) -> ParseResult:
        _ = (path, offset, message_idx_base, orphan_tool_call_idx_base, resume_state)
        return ParseResult(session=self._session, next_byte_offset=42, is_full_parse=True)

    roots: tuple[Path, ...] | None = None

    def default_roots(self) -> list[Path]:
        return []

    def watch_roots(self) -> list[Path]:
        return []

    @property
    def file_pattern(self) -> str:
        return "*.jsonl"

    def live_candidates(self, *, now, idle_threshold: float) -> list[Path]:
        _ = (now, idle_threshold)
        return []


def test_index_single_session_dual_writes_sidecar_with_field_scope(tmp_path: Path) -> None:
    config = _app_config(
        tmp_path,
        fts=FtsConfig(fields=("content",), backend="sqlite_sidecar"),
    )
    session_path = tmp_path / "session.jsonl"
    session_path.write_text("{}\n", encoding="utf-8")
    session = Session(
        id="session-watch",
        source=Source.CODEX,
        source_path=str(session_path.resolve()),
        file_mtime=session_path.stat().st_mtime,
        file_size=session_path.stat().st_size,
        message_count=1,
        messages=[
            Message(
                id="msg-watch",
                session_id="session-watch",
                idx=0,
                role=Role.ASSISTANT,
                content="watchcontentterm",
                thinking="watchthinkingterm",
                has_thinking=True,
                tool_calls=[
                    ToolCall(
                        id="tool-watch",
                        session_id="session-watch",
                        message_id="msg-watch",
                        idx=0,
                        tool_name="Bash",
                        bash_command="echo watchbashterm",
                    )
                ],
            )
        ],
    )
    config.data_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(config)
    conn.close()

    changed = index_single_session(session_path, _StaticParser(session), config)

    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        assert changed is True
        assert _message_ids(sidecar_conn, "watchcontentterm", ["content"]) == ["msg-watch"]
        assert _message_ids(sidecar_conn, "watchthinkingterm", ["content", "thinking"]) == []
        assert _tool_call_ids(sidecar_conn, "watchbashterm") == []
    finally:
        sidecar_conn.close()


def test_index_single_session_queues_sidecar_reconcile_after_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _app_config(
        tmp_path,
        fts=FtsConfig(fields=("content",), backend="sqlite_sidecar"),
    )
    session_path = tmp_path / "session.jsonl"
    session_path.write_text("{}\n", encoding="utf-8")
    original_session = Session(
        id="session-watch-rollback",
        source=Source.CODEX,
        source_path=str(session_path.resolve()),
        file_mtime=0.0,
        file_size=0,
        message_count=1,
        messages=[
            Message(
                id="msg-watch-rollback",
                session_id="session-watch-rollback",
                idx=0,
                role=Role.ASSISTANT,
                content="watchrollbackterm",
            )
        ],
    )
    parsed_session = Session(
        id=original_session.id,
        source=original_session.source,
        source_path=original_session.source_path,
        file_mtime=session_path.stat().st_mtime,
        file_size=session_path.stat().st_size,
        message_count=0,
        messages=[],
    )
    conn = connect(config)
    try:
        insert_session(conn, original_session)
        insert_messages(conn, original_session.messages)
    finally:
        conn.close()

    sidecar_conn = open_sidecar(sidecar_path(config.data_dir))
    try:
        upsert_message_fts(
            sidecar_conn,
            "msg-watch-rollback",
            "watchrollbackterm",
            "",
            fields=config.fts.fields,
        )
    finally:
        sidecar_conn.close()

    def fail_after_sidecar_delete(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated watch rollback after sidecar mutation")

    monkeypatch.setattr(indexer_module, "_update_session_row", fail_after_sidecar_delete)

    with pytest.raises(RuntimeError, match="simulated watch rollback"):
        index_single_session(session_path, _StaticParser(parsed_session), config)

    conn = connect(config)
    try:
        pending = conn.execute(
            """
            SELECT kind, id, op
            FROM fts_sidecar_pending
            WHERE kind = 'message' AND id = 'msg-watch-rollback'
            """
        ).fetchall()
    finally:
        conn.close()

    assert pending == [("message", "msg-watch-rollback", "upsert")]


def test_index_single_session_duckdb_backend_does_not_create_sidecar(tmp_path: Path) -> None:
    config = _app_config(tmp_path, fts=FtsConfig(fields=("content",), backend="duckdb"))
    session_path = tmp_path / "session.jsonl"
    session_path.write_text("{}\n", encoding="utf-8")
    session = Session(
        id="session-duckdb",
        source=Source.CODEX,
        source_path=str(session_path.resolve()),
        file_mtime=session_path.stat().st_mtime,
        file_size=session_path.stat().st_size,
        message_count=1,
        messages=[
            Message(
                id="msg-duckdb",
                session_id="session-duckdb",
                idx=0,
                role=Role.ASSISTANT,
                content="duckdbcontentterm",
            )
        ],
    )
    config.data_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(config)
    conn.close()

    changed = index_single_session(session_path, _StaticParser(session), config)

    assert changed is True
    assert not sidecar_path(config.data_dir).exists()
