"""An owned outer transaction exposes either the old pair or the complete pair."""

from contextlib import suppress
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


def test_a_consumed_fatal_cannot_admit_another_write_but_rollback_finishes(tmp_path: Path) -> None:
    db = tmp_path / "terminal.duckdb"
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    original = duckdb.FatalException("Failed to delete all rows from index")

    def fail(_conn: duckdb.DuckDBPyConnection) -> None:
        raise original

    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE witness(value INTEGER)")
        conn.execute("INSERT INTO witness VALUES (0)")
        with (
            jobs.lease(conn) as lease,
            pytest.raises(duckdb.FatalException) as caught,
            lease.transaction("write_scope") as transaction,
        ):
            transaction.call(
                "first_write", lambda conn: conn.execute("UPDATE witness SET value=1").fetchone()
            )
            with suppress(duckdb.FatalException):
                transaction.call("failure_before_policy", fail)
            transaction.call(
                "late_write", lambda conn: conn.execute("INSERT INTO witness VALUES (2)").fetchone()
            )
        assert caught.value is original
    with duckdb.connect(str(db), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == [(0,)]


def test_discarded_transaction_rolls_back_and_rejects_further_effects(tmp_path) -> None:
    from recall.db.lease import StandaloneLifetime

    path = tmp_path / "discard.duckdb"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE witness(value INT)")
        with StandaloneLifetime().lease(conn) as lease, lease.transaction("stale") as transaction:
            transaction.call(
                "write", lambda active: active.execute("INSERT INTO witness VALUES (9)").fetchone()
            )
            transaction.discard()
            with pytest.raises(AssertionError, match="discarded"):
                transaction.call(
                    "late",
                    lambda active: active.execute("INSERT INTO witness VALUES (10)").fetchone(),
                )
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == []
