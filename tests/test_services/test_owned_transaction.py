"""An owned outer transaction exposes either the old pair or the complete pair."""

from pathlib import Path

import duckdb
import pytest
from recall.services.owned_jobs import DaemonJobs


def test_failed_outer_transaction_rolls_back_all_visible_effects(tmp_path: Path) -> None:
    db = tmp_path / "atomic.duckdb"
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE pair (history INTEGER, budget INTEGER)")
        conn.execute("INSERT INTO pair VALUES (0, 0)")
        with (
            jobs.lease(conn) as lease,
            pytest.raises(RuntimeError, match="second write failed"),
            lease.transaction("completion") as transaction,
        ):
            transaction.call(
                "history", lambda c: c.execute("UPDATE pair SET history = 1").fetchone()
            )
            raise RuntimeError("second write failed")
    with duckdb.connect(str(db), read_only=True) as conn:
        assert conn.execute("SELECT * FROM pair").fetchall() == [(0, 0)]
