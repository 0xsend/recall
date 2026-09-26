"""Regression tests for tool_call_embeddings write churn.

`insert_tool_call_embeddings` used a plain `INSERT OR REPLACE`, which rewrote
every row on every call. Repeated watch-mode re-indexing therefore accumulated
dead row-versions until the table held ~115 physical versions per live row
(140 GiB of dead space, read as only 1.01x bloat). Re-inserting an unchanged
vector must now be a no-op, while a genuinely changed vector must still update.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from recall.db.queries import insert_tool_call_embeddings
from recall.db.schema import ensure_schema


def _physical_row_versions(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    """Physical row-versions (live + dead) via the estimator's own metric."""
    row = conn.execute(
        f"""
        SELECT MIN(column_rows) FROM (
            SELECT SUM(count) AS column_rows
            FROM pragma_storage_info('{table}')
            WHERE segment_type != 'VALIDITY'
            GROUP BY column_name
        )
        """
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def test_identical_reinserts_do_not_churn_row_versions(tmp_path: Path) -> None:
    db = tmp_path / "recall.duckdb"
    conn = duckdb.connect(str(db))
    ensure_schema(conn, embed_dim=8)
    rows: list[tuple[str, list[float] | None]] = [(f"tc{i}", [float(i)] * 8) for i in range(300)]

    insert_tool_call_embeddings(conn, rows)
    conn.execute("CHECKPOINT")
    baseline = _physical_row_versions(conn, "tool_call_embeddings")

    # Re-present the same embeddings the way repeated re-index cycles do.
    for _ in range(20):
        insert_tool_call_embeddings(conn, rows)
        conn.execute("CHECKPOINT")
    after = _physical_row_versions(conn, "tool_call_embeddings")

    count_row = conn.execute("SELECT count(*) FROM tool_call_embeddings").fetchone()
    assert count_row is not None
    assert count_row[0] == 300
    # OR REPLACE would leave ~20*300 dead versions (~21x baseline). The anti-join
    # keeps physical row-versions flat regardless of re-insert count.
    assert after <= baseline * 2
    conn.close()


def test_changed_vector_is_written(tmp_path: Path) -> None:
    db = tmp_path / "recall.duckdb"
    conn = duckdb.connect(str(db))
    ensure_schema(conn, embed_dim=8)

    insert_tool_call_embeddings(conn, [("tc1", [1.0] * 8)])
    insert_tool_call_embeddings(conn, [("tc1", [2.0] * 8)])  # changed -> must update

    stored_row = conn.execute(
        "SELECT bash_embedding FROM tool_call_embeddings WHERE tool_call_id = 'tc1'"
    ).fetchone()
    assert stored_row is not None
    assert list(stored_row[0]) == [2.0] * 8
    conn.close()


def test_new_rows_are_inserted(tmp_path: Path) -> None:
    db = tmp_path / "recall.duckdb"
    conn = duckdb.connect(str(db))
    ensure_schema(conn, embed_dim=8)

    insert_tool_call_embeddings(conn, [("tc1", [1.0] * 8)])
    insert_tool_call_embeddings(conn, [("tc1", [1.0] * 8), ("tc2", [3.0] * 8)])

    stored = {
        cid: list(vec)
        for cid, vec in conn.execute(
            "SELECT tool_call_id, bash_embedding FROM tool_call_embeddings ORDER BY tool_call_id"
        ).fetchall()
    }
    assert stored == {"tc1": [1.0] * 8, "tc2": [3.0] * 8}
    conn.close()
