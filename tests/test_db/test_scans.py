"""Corpus capacity is held through result consumption and released on cancellation."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import duckdb
import pytest
from recall.core.read_cancellation import ReadCancelled, read_cancellation
from recall.db.scans import ScanCapacity, execute_scan


def test_cancelled_scan_releases_its_place_for_the_next_reader() -> None:
    capacity = ScanCapacity(3)
    cancelled = threading.Event()
    waiting = threading.Event()

    def cancelled_reader() -> None:
        with read_cancellation(cancelled):
            waiting.set()
            with capacity.lease("ordinary"):
                pytest.fail("cancelled reader acquired capacity")

    with ThreadPoolExecutor(max_workers=2) as pool:
        with capacity.lease("ordinary"):
            abandoned = pool.submit(cancelled_reader)
            assert waiting.wait(2)
            cancelled.set()
            capacity.wake()
            with pytest.raises(ReadCancelled):
                abandoned.result(timeout=2)
            next_reader = pool.submit(lambda: _read(capacity))
        assert next_reader.result(timeout=2) == 42


def _read(capacity: ScanCapacity) -> int:
    with capacity.lease("ordinary"):
        return 42


def test_scan_retains_capacity_until_results_are_consumed() -> None:
    capacity = ScanCapacity(2)
    consuming = threading.Event()
    finish = threading.Event()
    follower_started = threading.Event()

    class Result:
        def fetchall(self) -> list[tuple[int]]:
            consuming.set()
            assert finish.wait(2)
            return [(7,)]

    class Connection:
        def execute(self, statement, parameters):
            return Result()

    def first_reader():
        with capacity.bind():
            return execute_scan(cast(duckdb.DuckDBPyConnection, Connection()), "SELECT 7")

    def follower():
        follower_started.set()
        return _read(capacity)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_reader)
        assert consuming.wait(2)
        second = pool.submit(follower)
        assert follower_started.wait(2)
        assert not second.done()
        finish.set()
        assert first.result(timeout=2) == [(7,)]
        assert second.result(timeout=2) == 42


def test_status_remains_available_during_publication() -> None:
    capacity = ScanCapacity(3)
    ordinary_started = threading.Event()

    def ordinary() -> int:
        ordinary_started.set()
        return _read(capacity)

    def status() -> int:
        with capacity.lease("status"):
            return 17

    with ThreadPoolExecutor(max_workers=2) as pool:
        with capacity.publication():
            deferred = pool.submit(ordinary)
            assert ordinary_started.wait(2)
            assert pool.submit(status).result(timeout=2) == 17
            with pytest.raises(TimeoutError):
                deferred.result(timeout=0.05)
        assert deferred.result(timeout=2) == 42


def test_publication_reads_its_own_history_without_admitting_another_scan() -> None:
    capacity = ScanCapacity(2)
    entered = threading.Event()

    def follower() -> int:
        entered.set()
        return _read(capacity)

    with duckdb.connect() as conn, ThreadPoolExecutor(max_workers=1) as pool:
        with capacity.bind(), capacity.publication():
            deferred = pool.submit(follower)
            assert entered.wait(2)
            assert execute_scan(conn, "SELECT 42") == [(42,)]
            with pytest.raises(TimeoutError):
                deferred.result(timeout=0.05)
        assert deferred.result(timeout=2) == 42
