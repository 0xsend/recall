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
from collections.abc import Iterator
from dataclasses import dataclass

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


EXCEPTION_GRAPH_NODE_MAX = 65_536


@dataclass(frozen=True)
class FatalGraph:
    failure: BaseException | None
    complete: bool
    nodes: int
    candidates: tuple[BaseException, ...]


def _exception_children(node: BaseException) -> Iterator[BaseException]:
    if node.__cause__ is not None:
        yield node.__cause__
    if node.__context__ is not None and node.__context__ is not node.__cause__:
        yield node.__context__
    if isinstance(node, BaseExceptionGroup):
        yield from node.exceptions


def inspect_fatal_graph(error: BaseException) -> FatalGraph:
    """Inspect cause, distinct context and group children without recursion.

    Concrete index faults outrank generic invalidation symptoms in one graph.
    Iterators keep a wide group's pending edges out of the worklist. The owner,
    rather than this traversal, arbitrates separately observed jobs.
    """
    pending: list[tuple[BaseException | None, Iterator[BaseException]]] = [(None, iter((error,)))]
    visited: set[int] = set()
    selected: BaseException | None = None
    selected_rank = 3
    candidates: tuple[list[BaseException], list[BaseException], list[BaseException]] = ([], [], [])
    while pending:
        parent, children = pending[-1]
        try:
            node = next(children)
        except StopIteration:
            pending.pop()
            if parent is None:
                continue
            message = str(parent)
            if isinstance(parent, duckdb.FatalException):
                rank = 0 if _INDEX_DIVERGENCE_MARKER in message else 1
            elif any(marker in message for marker in _FATAL_DB_INVALIDATION_MARKERS):
                rank = 2
            else:
                continue
            candidates[rank].append(parent)
            if rank < selected_rank:
                selected, selected_rank = parent, rank
            continue
        identity = id(node)
        if identity in visited:
            continue
        if len(visited) == EXCEPTION_GRAPH_NODE_MAX:
            return FatalGraph(
                selected,
                False,
                len(visited),
                tuple(node for bucket in candidates for node in bucket),
            )
        visited.add(identity)
        pending.append((node, _exception_children(node)))
    return FatalGraph(
        selected, True, len(visited), tuple(node for bucket in candidates for node in bucket)
    )


def is_fatal_db_invalidation(err: BaseException) -> bool:
    """True for a fatal anywhere in the owned graph or incomplete classification.

    An incomplete search is terminal diagnostic uncertainty, never permission
    to serve or evidence that a repairable fault was completely classified.
    """
    graph = inspect_fatal_graph(err)
    return not graph.complete or graph.failure is not None


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
