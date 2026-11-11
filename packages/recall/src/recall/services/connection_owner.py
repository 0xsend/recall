"""Scoped connection and cursor lifetimes for supported daemon DB work."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Concatenate

import duckdb

from recall.core.read_cancellation import read_cancellation
from recall.services.owned_jobs import DaemonJobs, DbLease, ObservedFailure
from recall.services.search_request import search_request


class ReadAbandoned(Exception):
    """A cancelled request withdrew before its worker opened a cursor."""


class _LifecycleGate:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    def acquire_read(self) -> None:
        with self._condition:
            while self._writer_active or self._writers_waiting > 0:
                self._condition.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._condition:
            assert self._readers > 0
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        with self._condition:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._condition.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1

    def release_write(self) -> None:
        with self._condition:
            assert self._writer_active
            self._writer_active = False
            self._condition.notify_all()


class ReadToken:
    """Cancellation and timing only; the read owner keeps its cursor private."""

    def __init__(self, cancel: Callable[[], None]) -> None:
        self.started_at = time.monotonic()
        self.cancel = cancel


class _ReadExecution:
    def __init__(self) -> None:
        self._timing = search_request.get()
        self._lock = threading.Lock()
        self._cursor: duckdb.DuckDBPyConnection | None = None
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled.set()
            if self._cursor is not None:
                self._cursor.interrupt()

    def run[T](self, owner: ConnectionOwner, operation: str, work: Callable[[DbLease], T]) -> T:
        with owner._shared():
            cursor = owner._observed(f"{operation}:cursor_open", lambda: owner._open().cursor())
            try:
                with self._lock:
                    if self._cancelled.is_set():
                        raise ReadAbandoned
                    self._cursor = cursor
                if self._timing is not None:
                    self._timing.cursor_opened()
                with read_cancellation(self._cancelled), owner._jobs.lease(cursor) as lease:
                    result = owner._observed(operation, lambda: work(lease))
            except BaseException as primary:
                owner._jobs.observe(primary, operation)
                try:
                    self._close(owner, operation, cursor)
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "read and cursor close failed", [primary, cleanup]
                    ) from None
                raise
            else:
                self._close(owner, operation, cursor)
                return result
            finally:
                with self._lock:
                    self._cursor = None

    def _close(
        self, owner: ConnectionOwner, operation: str, cursor: duckdb.DuckDBPyConnection
    ) -> None:
        with self._lock:
            self._cursor = None
        owner._observed(f"{operation}:cursor_close", cursor.close)


class ExclusiveDatabase:
    """A scoped exclusive owner; reopened connections cannot escape as values."""

    def __init__(self, owner: ConnectionOwner) -> None:
        self._owner = owner
        self._active = True

    def run[T](self, operation: str, work: Callable[[DbLease], T]) -> T:
        assert self._active, "exclusive database scope expired"
        return self._owner._run(operation, work)

    def reopen(self) -> None:
        assert self._active, "exclusive database scope expired"
        self._owner._close()
        self._owner._open()


class ConnectionOwner:
    """Own every supported parent/cursor and observe before cleanup or policy.

    Raw handles exist only in reviewed leaf callbacks passed to DbLease.call.
    The caller holds the writer turn; this owner holds connection lifecycle
    access through the entire callback, including materialization.
    """

    def __init__(self, jobs: DaemonJobs, opener: Callable[[], duckdb.DuckDBPyConnection]) -> None:
        self._jobs = jobs
        self._opener = opener
        self._gate = _LifecycleGate()
        self._open_lock = threading.Lock()
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._closed = False
        self._reads: dict[ReadToken, _ReadExecution] = {}
        self._reads_lock = threading.Lock()

    def _observed[T](self, operation: str, work: Callable[[], T]) -> T:
        outcome = self._jobs.inline(operation, work)
        if isinstance(outcome, ObservedFailure):
            outcome.raise_error()
            raise AssertionError("failure did not propagate")
        return outcome.value

    def _open(self) -> duckdb.DuckDBPyConnection:
        with self._open_lock:
            assert not self._closed, "connection lifetime is closed"
            if self._connection is None:
                self._connection = self._observed("database_open", self._opener)
            return self._connection

    def _close(self) -> None:
        if self._connection is not None:
            connection = self._connection
            self._connection = None
            self._observed("database_close", connection.close)

    @contextmanager
    def _shared(self) -> Iterator[None]:
        self._gate.acquire_read()
        try:
            assert not self._closed, "connection lifetime is closed"
            yield
        finally:
            self._gate.release_read()

    def _run[T](self, operation: str, work: Callable[[DbLease], T]) -> T:
        with self._jobs.lease(self._open()) as lease:
            return self._observed(operation, lambda: work(lease))

    def run[T](self, operation: str, work: Callable[[DbLease], T]) -> T:
        with self._shared():
            return self._run(operation, work)

    def call[T](self, operation: str, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        operation = self._jobs.operation or operation
        return self.run(operation, lambda lease: lease.call(operation, work))

    def unit[**P, T](
        self, operation: str
    ) -> Callable[[Callable[Concatenate[duckdb.DuckDBPyConnection, P], T]], Callable[P, T]]:
        """Bind a raw leaf callback to a scoped, observed connection lifetime."""

        def decorate(
            work: Callable[Concatenate[duckdb.DuckDBPyConnection, P], T],
        ) -> Callable[P, T]:
            @wraps(work)
            def run(*args: P.args, **kwargs: P.kwargs) -> T:
                return self.call(operation, lambda conn: work(conn, *args, **kwargs))

            return run

        return decorate

    def if_open[T](
        self, operation: str, work: Callable[[duckdb.DuckDBPyConnection], T]
    ) -> T | None:
        with self._shared():
            if self._connection is None:
                return None
            with self._jobs.lease(self._connection) as lease:
                return lease.call(operation, work)

    def recover(self, *, discard: bool) -> str:
        with self.exclusive():
            connection = self._connection
            if connection is None:
                return "close_clear"
            if not discard:
                path = "rollback_verified"
                try:
                    self._observed(
                        "database_rollback", lambda: connection.execute("ROLLBACK").fetchall()
                    )
                except duckdb.TransactionException as error:
                    if "no transaction is active" in str(error):
                        path = "no_active_txn"
                    else:
                        discard = True
                except Exception:
                    discard = True
                if not discard:
                    try:
                        row = self._observed(
                            "database_healthcheck",
                            lambda: connection.execute("SELECT 1").fetchone(),
                        )
                        if row == (1,):
                            return path
                    except Exception:
                        pass
            self._close()
            return "close_clear"

    @contextmanager
    def exclusive(self) -> Iterator[ExclusiveDatabase]:
        self._gate.acquire_write()
        database = ExclusiveDatabase(self)
        try:
            assert not self._closed, "connection lifetime is closed"
            yield database
        finally:
            database._active = False
            self._gate.release_write()

    def prepare_read(self) -> ReadToken:
        execution = _ReadExecution()
        token = ReadToken(lambda: self._observed("read_interrupt", execution.cancel))
        with self._reads_lock:
            assert not self._closed, "connection lifetime is closed"
            assert len(self._reads) < 72, "read admission capacity exceeded"
            self._reads[token] = execution
        return token

    def read[T](self, token: ReadToken, operation: str, work: Callable[[DbLease], T]) -> T:
        with self._reads_lock:
            execution = self._reads[token]
        try:
            return execution.run(self, operation, work)
        finally:
            self.discard_read(token)

    def discard_read(self, token: ReadToken) -> None:
        with self._reads_lock:
            self._reads.pop(token, None)

    def read_sync[T](self, operation: str, work: Callable[[DbLease], T]) -> T:
        return self.read(self.prepare_read(), operation, work)

    def interrupt_reads(self) -> None:
        with self._reads_lock:
            tokens = tuple(self._reads)
        for token in tokens:
            self._observed("read_interrupt", token.cancel)

    @contextmanager
    def maintenance(self) -> Iterator[None]:
        """Keep the file exclusively owned across close, replacement and reopen."""
        with self.exclusive():
            self._close()
            try:
                yield
            except BaseException as primary:
                self._jobs.observe(primary, "database_maintenance")
                try:
                    self._open()
                except BaseException as cleanup:
                    raise BaseExceptionGroup(
                        "maintenance and reopen failed", [primary, cleanup]
                    ) from None
                raise
            else:
                self._open()

    def close(self) -> None:
        """Caller drains jobs first; lifecycle access also joins active cursor scopes."""
        self._gate.acquire_write()
        try:
            if not self._closed:
                self._closed = True
                self._close()
        finally:
            self._gate.release_write()
