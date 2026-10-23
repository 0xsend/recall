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
@pytest.mark.parametrize("broad_estimate", [False, True])
def test_filtered_git_top_k_includes_current_hits_beyond_global_prefix(
    tmp_path: Path, kind: str, broad_estimate: bool
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
        if broad_estimate:
            # Stale counters can choose global paging even for sparse term hits.
            # Strategy estimates must never reintroduce a global-prefix cap.
            conn.execute(
                "INSERT INTO session_state(session_id, file_mtime, file_size, "
                "message_count, tool_count) VALUES (?, 0, 0, 10004, 10004)",
                [session],
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
        with pytest.raises(ValueError, match="session not found in indexed metadata: absent"):
            request(session="absent")
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


@pytest.mark.parametrize("kind", ["message", "tool_call"])
def test_filtered_paging_and_candidate_ranking_agree(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module = importlib.import_module("recall.services.search")
    config = AppConfig(
        data_dir=tmp_path,
        db_path=tmp_path / "recall.duckdb",
        lock_path=tmp_path / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(backend="sqlite_sidecar"),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    sidecar = open_sidecar(sidecar_path(tmp_path))
    try:
        conn.execute(
            "INSERT INTO sessions VALUES ('eligible', 'claude_code', '/eligible', NULL), "
            "('outside', 'codex', '/outside', NULL)"
        )
        # Identical documents tie. Stale and outside hits precede the eligible
        # cutoff, and insertion order disagrees with the required id ordering.
        documents = ["z-2", "z-1", "z-0", "a-outside", "a-stale"]
        conn.execute(
            "INSERT INTO messages SELECT id, CASE WHEN id='a-outside' THEN 'outside' "
            "ELSE 'eligible' END, row_number() OVER (), NULL FROM unnest(?) AS t(id)",
            [documents],
        )
        conn.execute(
            "INSERT INTO message_state(message_id, role, content, fts_content) "
            "SELECT id, 'assistant', 'git', 'git' FROM messages WHERE id != 'a-stale'"
        )
        if kind == "message":
            upsert_message_fts_batch(sidecar, [(key, "git", "") for key in documents])
            table, mapping, column = "message_fts", "message_fts_rowid", "message_id"
        else:
            conn.execute(
                "INSERT INTO tool_calls(id, session_id, message_id, idx, tool_name, bash_command) "
                "SELECT id, session_id, id, 0, 'Bash', 'git' FROM messages WHERE id != 'a-stale'"
            )
            upsert_tool_call_fts_batch(sidecar, [(key, "git") for key in documents])
            table, mapping, column = "tool_calls_fts", "tool_calls_fts_rowid", "tool_call_id"
        # Direct SQLite scores use the whole corpus, independently of the
        # product's membership/ranking orchestration.
        global_hits = sidecar.execute(
            f"SELECT r.{column}, -bm25({table}) FROM {table} "
            f"JOIN {mapping} r ON r.rowid={table}.rowid "
            f"WHERE {table} MATCH 'git' ORDER BY bm25({table}), r.{column}"
        ).fetchall()
        expected = [(key, score) for key, score in global_hits if key in {"z-0", "z-1"}]
        assert [key for key, _ in expected] == ["z-0", "z-1"]
        conn.execute(
            "INSERT INTO session_state(session_id, file_mtime, file_size, "
            "message_count, tool_count) "
            "VALUES ('eligible', 0, 0, 3, 3), ('outside', 0, 0, 1, 1)"
        )
        for fraction in (0.0, 1.0):
            monkeypatch.setattr(module, "_SIDECAR_SELECTIVE_FRACTION", fraction)
            monkeypatch.setattr(module, "_SIDECAR_FILTER_BATCH_SIZE", 2)
            results = search(
                query="git",
                source=Source.CLAUDE_CODE,
                session="eligible",
                tool="bAsH" if kind == "tool_call" else None,
                limit=2,
                mode=SearchMode.KEYWORD,
                config=config,
                conn=conn,
            )
            assert [(getattr(row, column), row.score) for row in results] == expected
            assert all(row.lexical_match for row in results)
    finally:
        sidecar.close()
        conn.close()
