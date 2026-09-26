"""Tail-fact retirement preserves unrelated pairs across bounded ID sets."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.db.queries import (
    delete_tool_call_tail_facts,
    insert_tool_results,
    insert_tool_use_ids,
)
from recall.db.schema import ensure_schema


@pytest.mark.parametrize("count", [0, 1, 501])
def test_tail_fact_deletion_preserves_other_pairs_and_rolls_back(
    tmp_path: Path, count: int
) -> None:
    with duckdb.connect(str(tmp_path / "tail.duckdb")) as conn:
        ensure_schema(conn)
        ids = [f"call-'{i}" for i in range(count)]
        all_ids = [*ids, "outside"]
        insert_tool_use_ids(conn, [(item, "session", f"harness-{item}") for item in all_ids])
        insert_tool_results(conn, [(item, f"result-{item}", False, None) for item in all_ids])
        tables = ("tool_use_ids", "tool_results")
        before = {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY tool_call_id").fetchall()
            for table in tables
        }
        conn.execute("BEGIN")
        # Repeated and absent IDs do not multiply effects or cross the requested scope.
        delete_tool_call_tail_facts(conn, [*ids, *ids[:1], "absent"] if ids else [])
        for table in tables:
            expected = [row for row in before[table] if row[0] not in ids]
            assert (
                conn.execute(f"SELECT * FROM {table} ORDER BY tool_call_id").fetchall() == expected
            )
        conn.execute("ROLLBACK")
        for table in tables:
            assert (
                conn.execute(f"SELECT * FROM {table} ORDER BY tool_call_id").fetchall()
                == before[table]
            )
