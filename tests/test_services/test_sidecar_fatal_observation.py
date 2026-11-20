"""A pending row's fail-soft publication must not consume a terminal graph."""

from contextlib import closing
from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall.db.fts_sidecar import open_sidecar
from recall.db.schema import ensure_schema
from recall.services import fts_sidecar_reconcile as reconcile


def test_pending_row_fallback_propagates_origin_and_retains_its_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = duckdb.FatalException("Failed to delete all rows from index in pending row")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("cannot publish pending row") from original

    monkeypatch.setattr(reconcile, "_apply_pending_row", fail)
    db_path = tmp_path / "sidecar-source.duckdb"
    with (
        duckdb.connect(str(db_path)) as conn,
        closing(open_sidecar(tmp_path / "sidecar.sqlite")) as sidecar,
    ):
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES ('message', 'm1', 'delete')"
        )
        with pytest.raises(ValueError, match="cannot publish pending row") as caught:
            reconcile.repair_pending_batch(conn, sidecar)
        assert caught.value.__cause__ is original
    with duckdb.connect(str(db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT kind, id, op FROM fts_sidecar_pending").fetchall() == [
            ("message", "m1", "delete")
        ]
