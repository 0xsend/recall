from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from recall.db.fts_sidecar import (
    open_sidecar,
    upsert_message_fts_batch,
    upsert_tool_call_fts_batch,
)


def statistics(conn: sqlite3.Connection, table: str) -> tuple:
    return (
        conn.execute(f"SELECT block FROM {table}_data WHERE id=1").fetchone(),
        conn.execute(
            f"SELECT rowid, bm25({table}) FROM {table} WHERE {table} MATCH 'needle'"
        ).fetchall(),
        conn.execute(f"SELECT count(*) FROM {table}").fetchone(),
    )


@pytest.mark.parametrize("kind", ["message", "tool_call"])
def test_identical_keyword_retries_preserve_totals_and_scores(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "keywords.sqlite"
    conn = open_sidecar(path)
    try:
        if kind == "message":
            table = "message_fts"
            upsert_message_fts_batch(
                conn,
                [(str(i), "needle alpha" if i == 0 else "ordinary text", "") for i in range(20)],
            )
        else:
            table = "tool_calls_fts"
            upsert_tool_call_fts_batch(
                conn, [(str(i), "needle alpha" if i == 0 else "ordinary text") for i in range(20)]
            )
        before = statistics(conn, table)
        conn.close()
        conn = open_sidecar(path)
        if kind == "message":
            upsert_message_fts_batch(conn, [("0", "needle alpha", "")] * 20, batch_size=7)
        else:
            upsert_tool_call_fts_batch(conn, [("0", "needle alpha")] * 20, batch_size=7)
        assert statistics(conn, table) == before
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["message", "tool_call"])
def test_changed_publication_then_retries_match_single_change_history(
    tmp_path: Path, kind: str
) -> None:
    from recall.db.fts_sidecar import upsert_message_fts, upsert_tool_call_fts

    connections = [open_sidecar(tmp_path / name) for name in ("retried", "reference")]
    table = "message_fts" if kind == "message" else "tool_calls_fts"
    try:
        for conn in connections:
            if kind == "message":
                upsert_message_fts_batch(
                    conn, [(str(i), "needle old" if i == 0 else "other", "") for i in range(20)]
                )
                upsert_message_fts(conn, "0", "needle changed", "")
            else:
                upsert_tool_call_fts_batch(
                    conn, [(str(i), "needle old" if i == 0 else "other") for i in range(20)]
                )
                upsert_tool_call_fts(conn, "0", "needle changed")
        conn = connections[0]
        mapping = f"{table}_rowid"
        before = conn.execute(
            f"SELECT rowid, content_fingerprint FROM {mapping} ORDER BY rowid"
        ).fetchall()
        if kind == "message":
            upsert_message_fts_batch(conn, [("0", "needle changed", "")] * 20, batch_size=7)
        else:
            upsert_tool_call_fts_batch(conn, [("0", "needle changed")] * 20, batch_size=7)
        assert (
            conn.execute(
                f"SELECT rowid, content_fingerprint FROM {mapping} ORDER BY rowid"
            ).fetchall()
            == before
        )
        assert statistics(conn, table) == statistics(connections[1], table)
        assert conn.execute(
            f"SELECT count(*) FROM {table} WHERE {table} MATCH 'old'"
        ).fetchone() == (0,)
        assert conn.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH 'changed'"
        ).fetchall() == [(1,)]
    finally:
        for conn in connections:
            conn.close()


def test_effective_columns_and_ordered_tool_deletion(tmp_path: Path) -> None:
    from recall.db.fts_sidecar import upsert_message_fts, upsert_tool_call_fts

    conn = open_sidecar(tmp_path / "keywords.sqlite")
    try:
        upsert_message_fts(conn, "m", "needle café", "excluded", fields=("content",))
        before = statistics(conn, "message_fts")
        changes = conn.total_changes
        upsert_message_fts(conn, "m", "needle café", "different", fields=("bash", "content"))
        assert statistics(conn, "message_fts") == before
        assert conn.total_changes == changes
        # Column boundaries and empty columns are part of the document identity.
        upsert_message_fts(conn, "m", "", "needle café", fields=("thinking", "content"))
        assert conn.execute(
            "SELECT count(*) FROM message_fts WHERE message_fts MATCH 'fts_content:needle'"
        ).fetchone() == (0,)
        assert conn.execute(
            "SELECT count(*) FROM message_fts WHERE message_fts MATCH 'fts_thinking:needle'"
        ).fetchone() == (1,)
        upsert_tool_call_fts_batch(
            conn, [("t", "old"), ("t", None), ("t", "needle"), ("t", "needle")]
        )
        assert conn.execute(
            "SELECT count(*) FROM tool_calls_fts WHERE tool_calls_fts MATCH 'old'"
        ).fetchone() == (0,)
        assert conn.execute(
            "SELECT count(*) FROM tool_calls_fts WHERE tool_calls_fts MATCH 'needle'"
        ).fetchone() == (1,)
        upsert_tool_call_fts(conn, "t", "needle", fields=("thinking",))
        assert conn.execute("SELECT count(*) FROM tool_calls_fts_rowid").fetchone() == (0,)
    finally:
        conn.close()


def test_publication_rolls_back_fingerprint_and_batch_progress(tmp_path: Path) -> None:
    from recall.db.fts_sidecar import publish_sidecar, upsert_message_fts

    conn = open_sidecar(tmp_path / "keywords.sqlite")
    try:
        upsert_message_fts(conn, "m", "needle old", "")
        before = statistics(conn, "message_fts")
        fingerprint = conn.execute("SELECT content_fingerprint FROM message_fts_rowid").fetchall()
        with pytest.raises(RuntimeError, match="interrupted"), publish_sidecar(conn) as publication:
            publication.messages([("m", "changed", ""), ("new", "new", "")])
            conn.execute("INSERT INTO bootstrap_progress VALUES ('message', 'new', NULL)")
            raise RuntimeError("interrupted")
        conn.close()
        conn = open_sidecar(tmp_path / "keywords.sqlite")
        assert statistics(conn, "message_fts") == before
        assert (
            conn.execute("SELECT content_fingerprint FROM message_fts_rowid").fetchall()
            == fingerprint
        )
        assert conn.execute("SELECT * FROM bootstrap_progress").fetchall() == []
    finally:
        conn.close()
