"""Preparation workers remain owned when a consumer leaves the queue early."""

import asyncio
import threading

import duckdb
import pytest
from recall.services.owned_jobs import DaemonJobs
from recall.services.preparation import StandalonePreparation


def test_unconsumed_parser_fatal_is_observed_and_propagated_after_drain() -> None:
    published: list[tuple[BaseException, str]] = []
    stopped = threading.Event()
    entered = threading.Event()
    release = threading.Event()
    original = duckdb.FatalException("Failed to delete all rows from index in parser")
    owner = DaemonJobs(lambda error, site, role: published.append((error, site)), stopped.set)

    def parser() -> int:
        entered.set()
        assert release.wait(5)
        raise original

    queue = owner.prepare(((1, lambda: 7), (2, parser)), workers=2)
    try:
        with pytest.raises(duckdb.FatalException, match="in parser"), queue:
            assert queue.take(1) == 7
            assert entered.wait(5)
            release.set()
            # No consumer ever requests the failed second capture.
        assert published == [(original, "parse_source")]
        assert stopped.is_set()
    finally:
        release.set()
        asyncio.run(owner.close())


@pytest.mark.parametrize("workers", [1, 2])
def test_standalone_preparation_keeps_ordinary_parse_errors_recoverable(workers: int) -> None:
    def malformed() -> int:
        raise ValueError("bad source")

    with StandalonePreparation().prepare(
        ((1, malformed), (2, lambda: 19)), workers=workers
    ) as queue:
        with pytest.raises(ValueError, match="bad source"):
            queue.take(1)
        assert queue.take(2) == 19


def test_primary_control_survives_a_different_unconsumed_parser_fatal() -> None:
    completed = threading.Event()
    fatal = duckdb.FatalException("Failed to delete all rows from index")
    primary = KeyboardInterrupt("consumer stopped")

    def parser() -> int:
        try:
            raise fatal
        finally:
            completed.set()

    with (
        pytest.raises(BaseExceptionGroup) as caught,
        StandalonePreparation().prepare(((1, parser),), workers=2),
    ):
        assert completed.wait(5)
        raise primary
    assert caught.value.exceptions == (primary, fatal)
