"""A migration that turns a native error into a soft failure cannot hide a recognized fatal.

The injected statement failure stands in for the engine's index-delete fatal;
its real trigger (a diverged ART index) cannot be provoked deterministically.
"""

from __future__ import annotations

from collections.abc import Callable

import duckdb
import pytest
from recall.db import migrations
from recall.db.lease import DbLease
from recall.db.migrations import Migration, run_pending_migrations

INDEX_DELETE = "Failed to delete all rows from index"


def _swallowing(error: Exception) -> Callable[[DbLease], bool]:
    """The shape of the historical migrations: log the database error, report False."""

    def upgrade(lease: DbLease) -> bool:
        def fail(_conn: object) -> None:
            raise error

        try:
            lease.call("injected_statement", fail)
        except Exception:
            return False
        return True

    return upgrade


def test_runner_surfaces_a_fatal_the_migration_body_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = Migration("0099_swallowed_fatal", 99, _swallowing(duckdb.FatalException(INDEX_DELETE)))
    monkeypatch.setattr(migrations, "get_migrations", lambda: [fake])
    with duckdb.connect() as conn, pytest.raises(duckdb.FatalException, match=INDEX_DELETE):
        run_pending_migrations(conn, 1)


def test_runner_keeps_an_ordinary_failed_migration_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = Migration("0099_ordinary_failure", 99, _swallowing(duckdb.Error("ordinary failure")))
    monkeypatch.setattr(migrations, "get_migrations", lambda: [fake])
    with duckdb.connect() as conn:
        run_pending_migrations(conn, 1)
        assert conn.execute("SELECT count(*) FROM schema_migrations").fetchone() == (0,)
