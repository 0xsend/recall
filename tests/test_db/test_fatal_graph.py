"""Fatal causes remain terminal through wrappers, groups and cancellation."""

import asyncio

import duckdb
import pytest
from recall.db.fatal import is_fatal_db_invalidation


@pytest.mark.parametrize("wrapper", [RuntimeError, asyncio.CancelledError, SystemExit])
def test_fatal_cause_is_terminal_through_outer_translation(wrapper: type[BaseException]) -> None:
    outer = wrapper("translated")
    outer.__cause__ = duckdb.FatalException("Failed to delete all rows from index")
    assert is_fatal_db_invalidation(outer)


def test_suppressed_context_still_carries_owned_failure() -> None:
    outer = RuntimeError("translated")
    outer.__context__ = duckdb.FatalException("Failed to delete all rows from index")
    outer.__suppress_context__ = True
    assert is_fatal_db_invalidation(outer)


def test_exception_group_preserves_fatal_in_later_branch() -> None:
    error = ExceptionGroup("workers", [ValueError("ordinary"), duckdb.FatalException("origin")])
    assert is_fatal_db_invalidation(error)


def test_deep_cyclic_graph_is_terminal_without_recursion() -> None:
    root = RuntimeError("root")
    current: BaseException = root
    for _ in range(10_000):
        child = RuntimeError("wrapper")
        current.__cause__ = child
        current = child
    current.__cause__ = duckdb.FatalException("origin")
    current.__cause__.__context__ = root
    assert is_fatal_db_invalidation(root)
