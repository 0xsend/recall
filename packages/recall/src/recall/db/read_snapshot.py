"""Read snapshots that never outlive a failed statement.

DuckDB aborts an open transaction when a statement fails during execution (out
of memory, a conversion error, `error()`): every later statement then fails
with "Current transaction is aborted" until ROLLBACK, and COMMIT silently
discards it. A read that catches such an error and keeps querying therefore
either fails later with that opaque message or commits a partial snapshot.
"""

from __future__ import annotations

from collections.abc import Callable

import duckdb

# DuckDB raises these while binding, before execution, so an open transaction
# stays usable after them. They are what an older schema raises for a missing
# table or column, which a read-only open cannot migrate. Reads that degrade on
# an older schema catch exactly these; every execution failure propagates.
SCHEMA_MISMATCH_ERRORS = (duckdb.CatalogException, duckdb.BinderException)


class AbortedReadSnapshotError(RuntimeError):
    """A read returned normally from inside a transaction a failed statement aborted."""


def read_snapshot[T](
    conn: duckdb.DuckDBPyConnection, read: Callable[[duckdb.DuckDBPyConnection], T]
) -> T:
    """Run `read` in one transaction; roll back on any exception.

    A read that swallowed an execution failure is a defect in that read. It
    fails here, by name, instead of committing whatever the read had gathered.
    """
    conn.execute("BEGIN")
    try:
        result = read(conn)
        try:
            conn.execute("SELECT 1")
        except duckdb.TransactionException as err:
            raise AbortedReadSnapshotError(
                "read snapshot ended inside an aborted transaction: a read swallowed a "
                "DuckDB execution error"
            ) from err
        conn.execute("COMMIT")
        return result
    except BaseException:
        conn.execute("ROLLBACK")
        raise
