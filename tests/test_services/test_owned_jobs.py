"""Fatal selection precedes fail-soft policy and survives later cleanup."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import duckdb
import pytest
from recall.services.owned_jobs import Completed, DaemonJobs, ObservedFailure


def fail(error: BaseException) -> None:
    raise error


def test_origin_is_observed_before_recoverable_wrapper_reaches_policy() -> None:
    published: list[tuple[BaseException, str]] = []
    stopped = threading.Event()
    owner = DaemonJobs(lambda error, site, role: published.append((error, site)), stopped.set)
    original = duckdb.FatalException("Failed to delete all rows from index")
    wrapper = ValueError("request failed")
    wrapper.__cause__ = original
    result = owner.inline("source_commit", lambda: fail(wrapper))
    assert isinstance(result, ObservedFailure)
    assert stopped.is_set()
    assert published == [(original, "source_commit")]


def test_first_observation_remains_primary_when_cleanup_also_fails() -> None:
    published: list[tuple[BaseException, str]] = []
    owner = DaemonJobs(lambda error, site, role: published.append((error, site)), lambda: None)
    original = duckdb.FatalException("Failed to delete all rows from index")
    later = duckdb.FatalException("database has been invalidated")
    owner.inline("work", lambda: fail(original))
    owner.inline("rollback", lambda: fail(later))
    assert published == [(original, "work"), (later, "rollback")]


def test_repropagated_graph_is_not_republished_but_new_cleanup_context_is() -> None:
    published: list[tuple[BaseException, str]] = []
    owner = DaemonJobs(lambda error, site, role: published.append((error, site)), lambda: None)
    first = duckdb.FatalException("Failed to delete all rows from index first")
    same_job = duckdb.FatalException("Failed to delete all rows from index second")
    graph = ExceptionGroup("one observed job", [first, same_job])
    owner.observe(graph, "work")
    owner.observe(graph, "outer handler")
    later = duckdb.FatalException("database has been invalidated during close")
    later.__context__ = graph
    owner.observe(later, "close")
    owner.observe(later, "outer cleanup")
    assert published == [(first, "work"), (later, "close")]
    assert owner.first_fatal is not None and owner.first_fatal.graph.failure is first


def test_marker_writer_failure_cannot_suppress_terminal_signal() -> None:
    stopped = threading.Event()
    owner = DaemonJobs(lambda error, site, role: fail(OSError("full")), stopped.set)
    result = owner.inline("work", lambda: fail(duckdb.FatalException("index fatal")))
    assert isinstance(result, ObservedFailure)
    assert stopped.is_set()


@pytest.mark.parametrize(
    "control", [asyncio.CancelledError, SystemExit, KeyboardInterrupt, GeneratorExit]
)
def test_control_flow_is_inspected_then_propagated(control: type[BaseException]) -> None:
    published: list[tuple[BaseException, str]] = []
    owner = DaemonJobs(lambda error, site, role: published.append((error, site)), lambda: None)
    original = duckdb.FatalException("Failed to delete all rows from index")
    outer = control()
    outer.__cause__ = original
    with pytest.raises(control):
        owner.inline("cancelled_work", lambda: fail(outer))
    assert published == [(original, "cancelled_work")]


def test_simultaneous_failures_both_observed_with_one_primary() -> None:
    published: list[tuple[BaseException, str]] = []
    stopped = threading.Event()
    owner = DaemonJobs(lambda error, site, role: published.append((error, site)), stopped.set)
    barrier = threading.Barrier(2)

    def work() -> None:
        barrier.wait(timeout=5)
        raise duckdb.FatalException("Failed to delete all rows from index")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(owner.inline, site, work) for site in ("first", "second")]
        assert all(isinstance(result.result(timeout=5), ObservedFailure) for result in results)
    assert stopped.is_set()
    assert len(published) == 2


def test_ordinary_parser_error_remains_recoverable() -> None:
    published: list[tuple[BaseException, str]] = []
    stopped = threading.Event()
    owner = DaemonJobs(lambda error, site, role: published.append((error, site)), stopped.set)
    result = owner.inline("parser", lambda: fail(ValueError("malformed input")))
    assert isinstance(result, ObservedFailure)
    assert not stopped.is_set()
    assert published == []
    assert owner.inline("value", lambda: 42) == Completed(42)


@pytest.mark.parametrize("control", [SystemExit, KeyboardInterrupt, GeneratorExit])
def test_departed_worker_control_propagates_after_owned_drain(
    control: type[BaseException],
) -> None:
    from recall.services.owned_jobs import JobLane, WorkNotAdmitted

    stopped = threading.Event()
    completed = threading.Event()
    owner = DaemonJobs(lambda error, site, role: None, stopped.set)
    terminal = control("worker terminated")

    def work() -> None:
        try:
            raise terminal
        finally:
            completed.set()

    owner.submit(JobLane.PREPARATION, "parser", work)
    assert completed.wait(timeout=5)
    with pytest.raises(control, match="worker terminated"):
        asyncio.run(owner.close())
    assert stopped.is_set()
    with pytest.raises(WorkNotAdmitted):
        owner.submit(JobLane.PREPARATION, "late", lambda: 1)
