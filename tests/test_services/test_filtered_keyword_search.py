"""Filtered top-k membership across the real authoritative/keyword stores."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.types import SearchMode, Source
from recall.db.fts_sidecar import (
    delete_message_fts,
    delete_tool_call_fts,
    open_sidecar,
    sidecar_path,
    upsert_message_fts_batch,
    upsert_tool_call_fts_batch,
)
from recall.db.schema import ensure_schema
from recall.services.search import search


@pytest.mark.parametrize("kind", ["message", "tool_call"])
def test_filtered_git_top_k_includes_current_hits_beyond_global_prefix(
    tmp_path: Path, kind: str
) -> None:
    config = AppConfig(
        data_dir=tmp_path,
        db_path=tmp_path / "recall.duckdb",
        fts=FtsConfig(backend="sqlite_sidecar"),
        lock_path=tmp_path / "recall.lock",
        config_path=tmp_path / "config.toml",
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    sidecar = open_sidecar(sidecar_path(tmp_path))
    session = "001e5dabce5012f1eaa198ed037cd495"
    try:
        conn.execute(
            "INSERT INTO sessions(id, source, source_path) VALUES (?, 'claude_code', '/claude')",
            [session],
        )
        conn.execute(
            "INSERT INTO sessions(id, source, source_path) VALUES ('other', 'codex', '/codex')"
        )
        # All scores tie. The desired IDs sort after the former 10,000-hit cap.
        ids = [f"a-{i:05}" for i in range(10_001)] + ["z-0", "z-1", "z-2"]
        conn.execute("CREATE TEMP TABLE seed AS SELECT unnest(?) AS id", [ids])
        conn.execute(
            "INSERT INTO messages(id, session_id, idx) "
            "SELECT id, CASE WHEN starts_with(id, 'z') THEN ? ELSE 'other' END, "
            "row_number() OVER () FROM seed",
            [session],
        )
        conn.execute(
            "INSERT INTO message_state(message_id, role, content, fts_content) "
            "SELECT id, 'assistant', 'git', 'git' FROM seed"
        )
        if kind == "message":
            upsert_message_fts_batch(sidecar, [(key, "git", "") for key in ids])
        else:
            conn.execute(
                "INSERT INTO tool_calls(id, session_id, message_id, idx, tool_name, bash_command) "
                "SELECT id, session_id, id, 0, CASE WHEN session_id = ? THEN 'Bash' ELSE 'Write' "
                "END, 'git' FROM messages",
                [session],
            )
            upsert_tool_call_fts_batch(sidecar, [(key, "git") for key in ids])

        def request(**kwargs):
            return search(
                query="git",
                source=None,
                tool=None,
                limit=10,
                mode=SearchMode.KEYWORD,
                config=config,
                conn=conn,
                **kwargs,
            )

        def keys(rows):
            return [row.tool_call_id if kind == "tool_call" else row.message_id for row in rows]

        results = request(session=session)
        assert keys(results) == ["z-0", "z-1", "z-2"]
        assert len({row.score for row in results}) == 1
        assert all(row.lexical_match for row in results)
        assert keys(request(session="absent")) == []
        assert (
            search(
                query="unfindable",
                source=Source.CLAUDE_CODE,
                tool=None,
                session=session,
                mode=SearchMode.KEYWORD,
                config=config,
                conn=conn,
            )
            == []
        )
        filtered = search(
            query="git",
            source=Source.CLAUDE_CODE,
            tool="bAsH" if kind == "tool_call" else None,
            limit=2,
            mode=SearchMode.KEYWORD,
            config=config,
            conn=conn,
        )
        assert keys(filtered) == ["z-0", "z-1"]

        # Authoritative removals must exclude stale sidecar entries before top-k.
        if kind == "tool_call":
            conn.execute("DELETE FROM tool_calls WHERE id = 'z-0'")
            upsert_tool_call_fts_batch(sidecar, [("z-1", "replaced")])
            delete_tool_call_fts(sidecar, ["z-2"])
        else:
            conn.execute("DELETE FROM message_state WHERE message_id = 'z-0'")
            upsert_message_fts_batch(sidecar, [("z-1", "replaced", "")])
            delete_message_fts(sidecar, ["z-2"])
        assert request(session=session) == []
    finally:
        sidecar.close()
        conn.close()


@pytest.mark.parametrize("kind", ["message", "tool_call"])
def test_filtered_bm25_uses_global_statistics(tmp_path: Path, kind: str) -> None:
    from recall.db.fts_sidecar import search_messages_fts, search_tool_calls_fts

    sidecar = open_sidecar(tmp_path / "scores.sqlite")
    documents = [
        ("outside", "git git git git"),
        ("eligible-a", "git git"),
        ("eligible-b", "git"),
        ("unmatched", "different"),
    ]
    try:
        if kind == "message":
            upsert_message_fts_batch(sidecar, [(key, text, "") for key, text in documents])
            global_hits = search_messages_fts(sidecar, "git", ["content"], 4)
            filtered_hits = search_messages_fts(
                sidecar, "git", ["content"], 2, candidate_ids=iter(["eligible-b", "eligible-a"])
            )
        else:
            upsert_tool_call_fts_batch(sidecar, documents)
            global_hits = search_tool_calls_fts(sidecar, "git", 4)
            filtered_hits = search_tool_calls_fts(
                sidecar, "git", 2, candidate_ids=iter(["eligible-b", "eligible-a"])
            )
        assert [key for key, _ in global_hits] == ["outside", "eligible-a", "eligible-b"]
        assert filtered_hits == global_hits[1:]
    finally:
        sidecar.close()
