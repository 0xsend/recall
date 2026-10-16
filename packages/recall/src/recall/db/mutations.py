"""Consume DuckDB mutation counts without materializing returned rows."""

from collections.abc import Sequence

import duckdb


def execute_count(
    conn: duckdb.DuckDBPyConnection, statement: str, parameters: Sequence[object] = ()
) -> int:
    """Execute a mutation without RETURNING and return its affected-row count.

    DuckDB 1.5.5 replaces updated rows when RETURNING is used, even for
    unindexed fields. Consume the count here so callers cannot mistake the
    always-present result row `(0,)` for a successful conditional mutation.
    The connection's transaction remains owned by the caller.
    """
    result = conn.execute(statement, parameters)
    assert result.description is not None and result.description[0][0] == "Count"
    row = result.fetchone()
    assert row is not None and len(row) == 1
    count = row[0]
    assert isinstance(count, int) and count >= 0
    return count
