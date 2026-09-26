"""Classification of the one DuckDB failure that no caller can recover from.

Certain index faults make DuckDB mark the whole *database instance* invalid;
every later statement on it fails with "database has been invalidated". Within
one process `duckdb.connect()` to the same path returns that same cached
instance, so closing and reopening the handle reattaches to the dead database.
Only a process restart clears it.

This lives in `db/` rather than beside its first caller because both the RPC
dispatcher and the watch-loop callers need it, and `services/watcher.py` must
not import `services/rpc_server.py`.
"""

from __future__ import annotations

import errno

import duckdb

# Matched against the message so a fatal wrapped in a non-DuckDB exception --
# the shape background loops produce when they re-raise with context -- is still
# recognised.
_FATAL_DB_INVALIDATION_MARKERS = (
    "database has been invalidated",
    "previous fatal error",
)
# DuckDB reports a full disk in the message of whichever exception type the
# failing operation raised (FatalException on checkpoint fsync, TransactionException
# on a WAL write), so the marker is matched on text, not type.
_DISK_FULL_MARKER = "No space left on device"
# The exact text DuckDB raises when an UPDATE's index delete finds fewer rows in the
# ART than the table holds -- the signature of a table/index divergence.
_INDEX_DIVERGENCE_MARKER = "Failed to delete all rows from index"


def is_fatal_db_invalidation(err: BaseException) -> bool:
    """Return True when the shared DuckDB instance is fatally invalidated.

    Callers must treat this as terminal: stop the process so the scheduler
    restarts it with a fresh instance. Continuing serves errors forever.
    """
    if isinstance(err, duckdb.FatalException):
        return True
    message = str(err)
    return any(marker in message for marker in _FATAL_DB_INVALIDATION_MARKERS)


def is_disk_full_error(err: BaseException) -> bool:
    """Return True when `err` reports a full disk (ENOSPC), fatal or not.

    A checkpoint that fails on fsync for lack of space can leave a table's rows
    absent from its ART indexes, so callers flag the database for index
    verification on the next open (REQ-RESIL-019).
    """
    if isinstance(err, OSError) and err.errno == errno.ENOSPC:
        return True
    return _DISK_FULL_MARKER in str(err)


def is_index_divergence_error(err: BaseException) -> bool:
    """Return True when `err` is DuckDB's table/index divergence fault.

    Rebuilding the indexes heals it; retrying the write never does.
    """
    return _INDEX_DIVERGENCE_MARKER in str(err)
