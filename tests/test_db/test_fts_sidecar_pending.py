from __future__ import annotations

import duckdb
from recall.db.queries import drain_sidecar_pending, enqueue_sidecar_pending
from recall.db.schema import ensure_schema


def test_enqueue_sidecar_pending_inserts_rows(tmp_path) -> None:
    conn = _duckdb_with_schema(tmp_path)
    try:
        enqueue_sidecar_pending(conn, "message", ["a", "b", "c"], "upsert")

        rows = conn.execute(
            """
            SELECT kind, id, op
            FROM fts_sidecar_pending
            ORDER BY id
            """
        ).fetchall()
        assert rows == [
            ("message", "a", "upsert"),
            ("message", "b", "upsert"),
            ("message", "c", "upsert"),
        ]
    finally:
        conn.close()


def test_drain_sidecar_pending_removes_matching_rows(tmp_path) -> None:
    conn = _duckdb_with_schema(tmp_path)
    try:
        enqueue_sidecar_pending(conn, "message", ["a", "b", "c"], "upsert")
        enqueue_sidecar_pending(conn, "message", ["d", "e"], "delete")

        removed = drain_sidecar_pending(conn, "message", ["a", "b", "c"])

        assert removed == 3
        assert conn.execute(
            """
            SELECT id, op
            FROM fts_sidecar_pending
            ORDER BY id
            """
        ).fetchall() == [("d", "delete"), ("e", "delete")]
    finally:
        conn.close()


def test_enqueue_empty_is_noop(tmp_path) -> None:
    conn = _duckdb_with_schema(tmp_path)
    try:
        enqueue_sidecar_pending(conn, "message", [], "upsert")

        row = conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone()
        assert row is not None
        assert row[0] == 0
    finally:
        conn.close()


def _duckdb_with_schema(tmp_path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall.duckdb"))
    ensure_schema(conn)
    return conn
