"""Session identifiers resolve through indexed metadata, never transcript discovery."""

from __future__ import annotations

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services.sessions import load_session


def _make_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    return conn


def _insert_session(conn: duckdb.DuckDBPyConnection, session_id: str) -> None:
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id)"
        " VALUES (?, 'claude_code', ?, NULL)",
        [session_id, f"/tmp/{session_id}.jsonl"],
    )
    conn.execute(
        "INSERT INTO session_state (session_id, file_mtime, file_size,"
        " message_count, tool_count) VALUES (?, 0, 0, 0, 0)",
        [session_id],
    )


def test_prefix_resolves_unique_session() -> None:
    conn = _make_conn()
    full_id = "aabbccddeeff00112233445566778899"
    _insert_session(conn, full_id)

    loaded = load_session(full_id[:12], include_tools=False, conn=conn)

    assert loaded.id == full_id


def test_load_session_includes_host() -> None:
    """REQ-HOST-API-002: show/load_session exposes session_state.host."""
    conn = _make_conn()
    sid = "aabbccddeeff001122334455667788aa"
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id)"
        " VALUES (?, 'claude_code', ?, NULL)",
        [sid, f"/tmp/{sid}.jsonl"],
    )
    conn.execute(
        "INSERT INTO session_state (session_id, file_mtime, file_size,"
        " message_count, tool_count, host) VALUES (?, 0, 0, 0, 0, ?)",
        [sid, "build-box"],
    )

    loaded = load_session(sid, include_tools=False, conn=conn)

    assert loaded.host == "build-box"


def test_prefix_ambiguous_raises_with_candidates() -> None:
    conn = _make_conn()
    first = "aabbccddeeff00112233445566778899"
    second = "aabbccddeeff99887766554433221100"
    _insert_session(conn, first)
    _insert_session(conn, second)

    with pytest.raises(ValueError, match="ambiguous session id prefix"):
        load_session("aabbccddeeff", include_tools=False, conn=conn)


@pytest.mark.parametrize("identifier", ["deadbeef1234", "0199aaaa-bbbb-cccc-dddd-eeeeffff0000"])
def test_unknown_uuid_is_answered_from_index_without_transcript_discovery(
    tmp_path, monkeypatch: pytest.MonkeyPatch, identifier: str
) -> None:
    """An absent identifier never turns a read into a transcript-tree traversal."""
    from pathlib import Path

    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".claude" / "projects").mkdir(parents=True)
    conn = _make_conn()

    def forbidden_tree_scan(*args, **kwargs):
        raise AssertionError("ordinary session lookup traversed the transcript tree")

    monkeypatch.setattr(Path, "rglob", forbidden_tree_scan)
    monkeypatch.setattr(Path, "glob", forbidden_tree_scan)
    try:
        with pytest.raises(ValueError, match="unindexed sources were not searched"):
            load_session(identifier, include_tools=False, conn=conn)
    finally:
        conn.close()
