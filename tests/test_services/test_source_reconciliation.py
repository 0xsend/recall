"""Regression floors for source reconciliation (REQ-RECON-004 through -006)."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import duckdb
import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    ContextConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
    SourceConfig,
)
from recall.core.types import DaemonMode, Source
from recall.db.connection import connect
from recall.parsers.codex import CodexParser
from recall.services.indexer import _write_session, index_sessions
from recall.services.watcher import index_single_session


def _rollout(path: Path, lines: list[Mapping[str, object]]) -> None:
    path.write_bytes(b"".join(json.dumps(line).encode() + b"\n" for line in lines))


def _runtime(tmp_path: Path) -> tuple[Path, AppConfig, duckdb.DuckDBPyConnection, CodexParser]:
    source_root = tmp_path / "codex"
    source_root.mkdir()
    path = source_root / "rollout.jsonl"
    fixture = Path(__file__).resolve().parents[2] / "fixtures/codex/session1/rollout.jsonl"
    path.write_bytes(fixture.read_bytes())
    data_dir = tmp_path / "data"
    config = AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(fields=(), backend="sqlite_sidecar"),
        embedding=EmbeddingConfig(context=ContextConfig(mode="off")),
        daemon=DaemonConfig(mode=DaemonMode.POLL, embed=False),
        cli=CliConfig(),
        sources={
            source.value: SourceConfig(roots=(source_root,) if source is Source.CODEX else ())
            for source in Source
        },
    )
    parser = CodexParser()
    conn = connect(config)
    index_single_session(path, parser, config, conn=conn)
    return path, config, conn, parser


def _message_rows(conn: duckdb.DuckDBPyConnection) -> list[tuple[object, ...]]:
    return conn.execute(
        "SELECT m.id, m.idx, ms.content FROM messages m "
        "JOIN message_state ms ON ms.message_id = m.id ORDER BY m.idx"
    ).fetchall()


def test_partial_tail_is_not_acknowledged_before_its_record_is_complete(tmp_path: Path) -> None:
    """A checkpoint may name only records the parser committed to the result."""
    path = tmp_path / "rollout.jsonl"
    first = {
        "type": "event_msg",
        "payload": {"type": "user_message", "message": "first"},
    }
    second = {
        "type": "event_msg",
        "payload": {"type": "agent_message", "message": "later"},
    }
    _rollout(path, [first])
    complete_offset = path.stat().st_size
    encoded_second = json.dumps(second).encode() + b"\n"
    split = len(encoded_second) // 2
    with path.open("ab") as handle:
        handle.write(encoded_second[:split])

    partial = CodexParser().parse(path)

    assert partial.session.is_complete is False
    assert partial.next_byte_offset == complete_offset
    assert [message.content for message in partial.session.messages] == ["first"]

    with path.open("ab") as handle:
        handle.write(encoded_second[split:])
    completed = CodexParser().parse(path, offset=partial.next_byte_offset, message_idx_base=1)

    assert completed.session.is_complete is True
    assert completed.next_byte_offset == path.stat().st_size
    assert [message.content for message in completed.session.messages] == ["later"]


def test_growing_metadata_rewrite_full_normalizes_instead_of_appending(tmp_path: Path) -> None:
    """A changed source has no trusted resume checkpoint during U1."""
    path, config, conn, parser = _runtime(tmp_path)
    try:
        before = _message_rows(conn)
        original_stat = path.stat()
        lines = path.read_text().splitlines()
        metadata = json.loads(lines[0])
        metadata["payload"]["rewrite_padding"] = "x" * 2048
        lines[0] = json.dumps(metadata)
        path.write_text("\n".join(lines) + "\n")
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

        index_single_session(path, parser, config, conn=conn)

        assert _message_rows(conn) == before
    finally:
        conn.close()


def test_malformed_rewrite_does_not_replace_valid_indexed_history(tmp_path: Path) -> None:
    """A partial normalization result cannot delete an already valid suffix."""
    path, config, conn, parser = _runtime(tmp_path)
    try:
        before = _message_rows(conn)
        lines = path.read_text().splitlines()
        # Preserve the metadata and first complete record, then make the
        # remainder unreadable.  A destructive full sync would delete every
        # previously indexed message after that first record.
        path.write_text("\n".join([*lines[:2], "{malformed", *lines[2:]]) + "\n")

        assert index_single_session(path, parser, config, conn=conn) is False
        assert _message_rows(conn) == before
    finally:
        conn.close()


def test_commit_callback_failure_rolls_back_raw_session_write(tmp_path: Path) -> None:
    """A catalog acknowledgment failure cannot publish its raw session alone."""
    path, _config, conn, parser = _runtime(tmp_path)
    try:
        before = _message_rows(conn)
        replacement = parser.parse(path).session.model_copy(deep=True)
        replacement.messages[0].content = "uncommitted replacement"

        def fail_acknowledgement() -> None:
            raise RuntimeError("catalog acknowledgment failed")

        with pytest.raises(RuntimeError, match="catalog acknowledgment failed"):
            _write_session(
                conn,
                replacement,
                tail_facts=parser.parse(path).tail_facts,
                on_commit=fail_acknowledgement,
            )
        assert _message_rows(conn) == before
    finally:
        conn.close()


def test_full_rewrite_discards_llm_context_for_changed_conversation_input(tmp_path: Path) -> None:
    """A positional ID is not provenance for an LLM-derived prefix."""
    path, config, conn, _parser = _runtime(tmp_path)
    try:
        message_id = str(_message_rows(conn)[0][0])
        conn.execute(
            "UPDATE message_state SET context_text = ?, context_mode = ? WHERE message_id = ?",
            ["[summary of Hi] ", "llm-local", message_id],
        )
        path.write_bytes(path.read_bytes().replace(b'"message":"Hi"', b'"message":"Yo"'))

        index_sessions(
            source=None,
            full=True,
            recreate=False,
            verbose=False,
            embed=False,
            config=config,
            conn=conn,
            workers=1,
        )

        assert conn.execute(
            "SELECT content, context_text, context_mode FROM message_state WHERE message_id = ?",
            [message_id],
        ).fetchone() == ("Yo", "", "off")
    finally:
        conn.close()
