"""Catalog bookkeeping must not accumulate replaced rows."""

from pathlib import Path

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog, SourceSignature


def test_catalog_bookkeeping_preserves_rows_and_conditional_outcomes(tmp_path: Path) -> None:
    with duckdb.connect(str(tmp_path / "catalog.duckdb")) as conn:
        ensure_schema(conn)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        catalog.observe("codex", "/owned", "/owned/session", SourceSignature(1, 2, 3, 4, 5))
        before = conn.execute("SELECT rowid FROM source_files").fetchone()
        assert catalog.acknowledge("codex", "/owned/session", 1, 5, "a" * 64)
        assert not catalog.acknowledge("codex", "/owned/session", 1, 5, "a" * 64)
        assert not catalog.acknowledge("codex", "/owned/absent", 1, 5, "a" * 64)
        catalog.defer_until_changed("codex", "/owned/session", "unsupported", {})
        catalog.retry_now("codex", "/owned/session")
        catalog.force_reconcile("codex", "/owned/session")
        assert catalog.acknowledge("codex", "/owned/session", 2, 5, "a" * 64)
        assert not catalog.acknowledge("codex", "/owned/session", 1, 5, "a" * 64)
        assert conn.execute("SELECT rowid FROM source_files").fetchone() == before
        row = catalog.get("codex", "/owned/session")
        assert row is not None and row.current and row.committed_generation == 2
        with pytest.raises(KeyError):
            catalog.retry_now("codex", "/owned/absent")
        with pytest.raises(KeyError):
            catalog.defer_until_changed("codex", "/owned/absent", "unsupported", {})


def test_scan_generation_advances_without_replacing_root(tmp_path: Path) -> None:
    with duckdb.connect(str(tmp_path / "roots.duckdb")) as conn:
        ensure_schema(conn)
        catalog = SourceCatalog(conn, clock=lambda: 100.0)
        assert catalog.start_scan("codex", "/owned", 100.0) == 1
        before = conn.execute("SELECT rowid FROM reconciliation_roots").fetchone()
        for generation in range(2, 12):
            assert catalog.start_scan("codex", "/owned", float(generation)) == generation
        assert conn.execute("SELECT rowid FROM reconciliation_roots").fetchone() == before
        assert conn.execute(
            "SELECT scan_generation, scan_complete, scan_started_at FROM reconciliation_roots"
        ).fetchone() == (11, False, 11.0)
