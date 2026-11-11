"""Active DB unit and outer-transaction capabilities, independent of service policy."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol
from weakref import WeakKeyDictionary

import duckdb

from recall.db.fatal import inspect_fatal_graph


class LeaseOwner(Protocol):
    @property
    def generation(self) -> int: ...

    def call_db[T](self, operation: str, work: Callable[[], T]) -> T: ...

    def cleanup_db[T](self, operation: str, work: Callable[[], T]) -> T: ...

    def observe(self, error: BaseException, operation: str) -> bool: ...

    def connection_epoch(self, connection: duckdb.DuckDBPyConnection) -> object: ...


class DbLifetime(LeaseOwner, Protocol):
    def lease(self, connection: duckdb.DuckDBPyConnection) -> AbstractContextManager[DbLease]: ...


class StandaloneLifetime:
    """Scoped DB capabilities without daemon marker or shutdown policy."""

    generation = 1

    def __init__(self) -> None:
        self._epochs: WeakKeyDictionary[duckdb.DuckDBPyConnection, object] = WeakKeyDictionary()
        self._terminal_error: BaseException | None = None

    def call_db[T](self, operation: str, work: Callable[[], T]) -> T:
        if self._terminal_error is not None:
            raise self._terminal_error
        return self.cleanup_db(operation, work)

    def cleanup_db[T](self, operation: str, work: Callable[[], T]) -> T:
        try:
            return work()
        except BaseException as error:
            self.observe(error, operation)
            raise

    def observe(self, error: BaseException, operation: str) -> bool:
        graph = inspect_fatal_graph(error)
        terminal = not graph.complete or graph.failure is not None
        if terminal and self._terminal_error is None:
            self._terminal_error = graph.failure if graph.failure is not None else error
        return terminal

    def connection_epoch(self, connection: duckdb.DuckDBPyConnection) -> object:
        return self._epochs.setdefault(connection, object())

    @contextmanager
    def lease(self, connection: duckdb.DuckDBPyConnection) -> Iterator[DbLease]:
        lease = DbLease(self, connection)
        try:
            yield lease
        finally:
            lease._active = False


class DbLease:
    """A required live capability for a supported DB leaf callback.

    Raw handles stay a reviewed boundary: a callback must materialize its value
    result inside call(), never retain or return a handle for later use.
    """

    def __init__(self, owner: LeaseOwner, connection: duckdb.DuckDBPyConnection) -> None:
        self._owner = owner
        self._connection = connection
        self._active = True
        self._generation = owner.generation
        self._connection_epoch = owner.connection_epoch(connection)

    def call[T](self, operation: str, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        value = self._owner.call_db(operation, lambda: work(self._connection))
        assert not isinstance(value, duckdb.DuckDBPyConnection), "DB handle escaped its lease"
        return value

    def cleanup_call[T](self, operation: str, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        """Compensation/rollback remains observed after ordinary DB admission closes."""
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        value = self._owner.cleanup_db(operation, lambda: work(self._connection))
        assert not isinstance(value, duckdb.DuckDBPyConnection), "DB handle escaped its lease"
        return value

    @property
    def connection_epoch(self) -> object:
        assert self._active, "expired DB lease"
        return self._connection_epoch

    @property
    def lineage(self) -> tuple[object, int]:
        assert self._active, "expired DB lease"
        return self._owner, self._generation

    @contextmanager
    def transaction(self, operation: str) -> Iterator[TransactionCapability]:
        self.call(f"{operation}:begin", lambda conn: conn.execute("BEGIN").fetchone())
        transaction = TransactionCapability(self)
        try:
            yield transaction
            self.call(f"{operation}:commit", lambda conn: conn.execute("COMMIT").fetchone())
        except BaseException as primary:
            self._owner.observe(primary, operation)
            try:
                self.cleanup_call(
                    f"{operation}:rollback", lambda conn: conn.execute("ROLLBACK").fetchone()
                )
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "owned transaction and rollback failed", [primary, cleanup]
                ) from None
            raise
        finally:
            transaction._active = False


class TransactionCapability:
    """Only the outer lease scope begins/commits; apply functions receive this."""

    def __init__(self, lease: DbLease) -> None:
        self._lease = lease
        self._active = True

    def call[T](self, operation: str, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        assert self._active, "expired transaction capability"
        return self._lease.call(operation, work)
