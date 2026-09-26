"""Explicit reparse batches preserve pending age, generations and unrelated rows."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog, SourceSignature


def _rows(conn: duckdb.DuckDBPyConnection) -> list[tuple[object, ...]]:
    return conn.execute("SELECT * FROM source_files ORDER BY source_key").fetchall()


@pytest.mark.parametrize("count", [1, 256])
def test_reparse_batch_preserves_metadata_and_caller_rollback(tmp_path: Path, count: int) -> None:
    with duckdb.connect(str(tmp_path / "catalog.duckdb")) as conn:
        ensure_schema(conn)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        for i in range(count + 1):
            path = f"/owned/{i}'source"
            catalog.observe("codex", "/owned", path, SourceSignature(1, i, 3, 4, 5))
            if i % 2 == 0:
                assert catalog.acknowledge("codex", path, 1, 5, "a" * 64)
        conn.execute("""UPDATE source_files SET retry_count=7, next_retry_at=300,
            last_error='retained history', last_serviced_seq=91""")
        before = _rows(conn)
        columns = [column[0] for column in conn.description]
        expected = []
        targets = {f"/owned/{i}'source" for i in range(count)}
        for values in before:
            row = dict(zip(columns, values, strict=True))
            if row["source_path"] in targets:
                generation = row["desired_generation"]
                assert isinstance(generation, int)
                row["desired_generation"] = generation + 1
                if row["first_pending_at"] is None:
                    row["first_pending_at"] = 200.0
                    row["first_pending_seq"] = 91
                row["next_retry_at"] = 0.0
            expected.append(tuple(row[column] for column in columns))
        conn.execute("BEGIN")
        SourceCatalog(conn, clock=lambda: 200.0).force_reconcile_batch(
            [("codex", path) for path in sorted(targets)]
        )
        assert _rows(conn) == expected
        conn.execute("ROLLBACK")
        assert _rows(conn) == before


@pytest.mark.parametrize("invalid", ["empty", "too_many", "duplicate", "missing"])
def test_invalid_reparse_batch_leaves_all_rows_unchanged(tmp_path: Path, invalid: str) -> None:
    with duckdb.connect(str(tmp_path / "invalid.duckdb")) as conn:
        ensure_schema(conn)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        catalog.observe("codex", "/owned", "/owned/source", SourceSignature(1, 2, 3, 4, 5))
        before = _rows(conn)
        requests = {
            "empty": [],
            "too_many": [("codex", f"/owned/{i}") for i in range(257)],
            "duplicate": [("codex", "/owned/source"), ("codex", "/owned/source")],
            "missing": [("codex", "/owned/source"), ("codex", "/owned/missing")],
        }[invalid]
        with pytest.raises(KeyError if invalid == "missing" else ValueError):
            catalog.force_reconcile_batch(requests)
        assert _rows(conn) == before
