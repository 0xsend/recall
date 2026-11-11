"""Session ownership is metadata, never conversation or account identity."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall.parsers.claude_code import ClaudeCodeParser
from recall.services.indexer import index_sessions

ACCOUNT = "628452cb-a113-4a8e-b6cb-5a482fbcb0af"
ORGANIZATION = "d6e61ee6-9e35-4d85-8bd2-117811c401b8"
SESSION = "ac518357-a795-4d23-b276-adbc3131dc23"
OWNER = {
    "type": "session-owner",
    "accountUuid": ACCOUNT,
    "organizationUuid": ORGANIZATION,
    "sessionId": SESSION,
}
CONVERSATION = [
    {
        "type": role,
        "sessionId": SESSION,
        "cwd": "/synthetic/orchard",
        "gitBranch": "orchard",
        "timestamp": f"2026-01-01T00:00:0{idx}Z",
        "message": {"role": role, "content": content},
    }
    for idx, (role, content) in enumerate(
        [("user", "Orchard question"), ("assistant", "Orchard answer")]
    )
]
RECORDS = [OWNER, CONVERSATION[0], OWNER, CONVERSATION[1], OWNER]


def _write(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_owner_records_preserve_conversation_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    parser = ClaudeCodeParser()
    _write(path, CONVERSATION)
    reference = parser.parse(path)
    _write(path, RECORDS)
    result = parser.parse(path)

    assert result.diagnostics == ()
    assert result.session.is_complete
    assert result.next_byte_offset == path.stat().st_size
    assert result.normalization_checkpoint is not None
    assert [m.content for m in result.session.messages] == ["Orchard question", "Orchard answer"]
    assert result.session.message_count == 2
    assert result.session.tool_count == 0
    assert result.session.model_dump(exclude={"file_size", "file_mtime"}) == (
        reference.session.model_dump(exclude={"file_size", "file_mtime"})
    )
    assert result.tail_facts == reference.tail_facts
    for identifier in (ACCOUNT, ORGANIZATION):
        assert identifier not in repr(result)


@pytest.mark.parametrize("split", range(1, len(RECORDS)))
def test_owner_records_resume_like_full_parse(tmp_path: Path, split: int) -> None:
    path = tmp_path / "session.jsonl"
    parser = ClaudeCodeParser()
    _write(path, RECORDS[:split])
    prefix = parser.parse(path)
    checkpoint = prefix.normalization_checkpoint
    assert prefix.diagnostics == ()
    assert checkpoint is not None
    _write(path, RECORDS)
    full = parser.parse(path)
    suffix = parser.parse(
        path,
        offset=checkpoint.offset,
        message_idx_base=checkpoint.message_idx_base,
        resume_state=checkpoint.adapter_state,
    )
    assert suffix.diagnostics == ()
    assert prefix.session.messages + suffix.session.messages == full.session.messages
    assert suffix.normalization_checkpoint == full.normalization_checkpoint


def test_owner_only_uses_filename_identity(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    # Even fields shaped like conversation must never turn account metadata
    # into messages, token usage or unrelated session facts.
    _write(path, [{**OWNER, "cwd": ACCOUNT, "message": {"role": "user", "content": ORGANIZATION}}])
    result = ClaudeCodeParser().parse(path)
    assert result.diagnostics == ()
    assert result.session.source_session_id == "session"
    assert result.session.cwd is None
    assert result.session.messages == []
    assert result.session.input_tokens is None
    for identifier in (ACCOUNT, ORGANIZATION):
        assert identifier not in repr(result)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_conflicting_owner_preserves_identity_at_every_boundary(
    tmp_path: Path, position: int
) -> None:
    path = tmp_path / f"{SESSION}.jsonl"
    parser = ClaudeCodeParser()
    _write(path, CONVERSATION)
    reference = parser.parse(path)
    records = list(CONVERSATION)
    records.insert(position, {**OWNER, "sessionId": "f82096c8-ff3a-4767-b74f-397a242c9082"})
    _write(path, records)
    result = parser.parse(path)
    assert result.diagnostics == ()
    assert result.session.model_dump(exclude={"file_size", "file_mtime"}) == (
        reference.session.model_dump(exclude={"file_size", "file_mtime"})
    )
    assert result.tail_facts == reference.tail_facts
    for split in range(1, len(records)):
        _write(path, records[:split])
        prefix = parser.parse(path)
        checkpoint = prefix.normalization_checkpoint
        assert checkpoint is not None
        _write(path, records)
        suffix = parser.parse(
            path,
            offset=checkpoint.offset,
            message_idx_base=checkpoint.message_idx_base,
            resume_state=checkpoint.adapter_state,
        )
        assert suffix.session.source_session_id == SESSION
        assert suffix.diagnostics == ()
        assert prefix.session.messages + suffix.session.messages == reference.session.messages
        assert suffix.normalization_checkpoint == result.normalization_checkpoint


def test_unknown_owner_type_remains_diagnostic(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _write(path, [*RECORDS, {"type": "session-owner-future"}])
    result = ClaudeCodeParser().parse(path)
    assert [(d.kind, d.detail) for d in result.diagnostics] == [
        ("unsupported_record", "record: 'session-owner-future'")
    ]
    assert [m.content for m in result.session.messages] == ["Orchard question", "Orchard answer"]
    assert result.normalization_checkpoint is None


def test_indexed_owner_identifiers_are_absent_from_all_tables_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    _write(sources / "session.jsonl", RECORDS)
    config = tmp_path / "config.toml"
    config.write_text(
        '[context]\nmode = "off"\n'
        + "".join(
            f"[sources.{name}]\n"
            f"roots = {json.dumps([str(sources)] if name == 'claude_code' else [])}\n"
            for name in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        )
    )
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config))
    summary = index_sessions(source=None, full=True, recreate=False, verbose=False)
    assert summary.indexed == 1
    assert summary.failed == 0
    from recall.core.config import AppConfig
    from recall.db import sidecar_path
    from recall.services.sessions import load_session

    with duckdb.connect(str(AppConfig.load().db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM messages").fetchone() == (2,)
        row = conn.execute("SELECT id FROM sessions").fetchone()
        assert row is not None
        session_id = row[0]
        loaded = load_session(session_id, include_tools=True, conn=conn)
        assert loaded is not None
        assert loaded.is_complete
        outputs = [loaded.model_dump_json(), str(summary), caplog.text, str(capsys.readouterr())]
        tables = conn.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE'"
        ).fetchall()
        assert tables
        for schema, table in tables:
            quoted = ".".join('"' + name.replace('"', '""') + '"' for name in (schema, table))
            outputs.append(repr(conn.execute(f"SELECT * FROM {quoted}").fetchall()))
        with sqlite3.connect(sidecar_path(AppConfig.load().data_dir)) as sidecar:
            sidecar_tables = sidecar.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
            assert sidecar_tables
            for (table,) in sidecar_tables:
                quoted = '"' + table.replace('"', '""') + '"'
                outputs.append(repr(sidecar.execute(f"SELECT * FROM {quoted}").fetchall()))
        for identifier in (ACCOUNT, ORGANIZATION):
            assert all(identifier not in output for output in outputs)
