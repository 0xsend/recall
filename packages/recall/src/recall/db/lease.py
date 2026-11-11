"""Active DB unit and outer-transaction capabilities, independent of service policy."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Never, Protocol
from weakref import WeakKeyDictionary

import duckdb

from recall.core.failure_observation import FailureObserver
from recall.db.fatal import inspect_fatal_graph


@dataclass(frozen=True)
class CompletionFailure:
    error: BaseException


class CompletionReadRequired(RuntimeError):
    """The failed native epoch cannot serve another write or completion."""


@dataclass(eq=False)
class _ConnectionEpoch:
    failed_completion: CompletionFailure | None = None
    closed: bool = False


_epochs: WeakKeyDictionary[duckdb.DuckDBPyConnection, _ConnectionEpoch] = WeakKeyDictionary()
_epoch_lock = Lock()


@dataclass(slots=True)
class _CompletionUse:
    consumed: bool = False


@dataclass(frozen=True)
class CompletionPermit:
    _identity: object
    generation: int
    _use: _CompletionUse = field(default_factory=_CompletionUse, repr=False)

    def consume(self) -> None:
        """The issuing arbiter admits at most one execution of this completion."""
        self.assert_unused()
        self._use.consumed = True

    def assert_unused(self) -> None:
        assert not self._use.consumed, "completion permit was already consumed"


class CompletionRelease:
    """Live success authority within one already admitted completion scope."""

    def __init__(self, guard: Callable[[], None]) -> None:
        self._guard = guard
        self._active = True

    def check(self) -> None:
        assert self._active, "completion release authority expired"
        self._guard()

    def expire(self) -> None:
        assert self._active, "completion release authority expired"
        self._active = False


def connection_epoch(connection: duckdb.DuckDBPyConnection) -> object:
    """Borrowers share identity without retaining or exposing the native handle."""
    with _epoch_lock:
        return _epochs.setdefault(connection, _ConnectionEpoch())


def _completion_state(connection: duckdb.DuckDBPyConnection) -> _ConnectionEpoch:
    with _epoch_lock:
        return _epochs.setdefault(connection, _ConnectionEpoch())


class LeaseOwner(Protocol):
    @property
    def generation(self) -> int: ...

    def call_db[T](self, operation: str, work: Callable[[], T]) -> T: ...

    def cleanup_db[T](self, operation: str, work: Callable[[], T]) -> T: ...

    def observe(self, error: BaseException, operation: str) -> bool: ...

    def connection_epoch(self, connection: duckdb.DuckDBPyConnection) -> object: ...

    def completion_permit(self) -> CompletionPermit: ...

    def completing(self, permit: CompletionPermit) -> AbstractContextManager[CompletionRelease]: ...

    def admitting(self) -> AbstractContextManager[None]: ...

    def reject_admission(self, error: BaseException, operation: str) -> None: ...

    def completion_failed(self, error: BaseException) -> CompletionFailure: ...


class DbLifetime(LeaseOwner, Protocol):
    def lease(self, connection: duckdb.DuckDBPyConnection) -> AbstractContextManager[DbLease]: ...


class Closable(Protocol):
    def close(self) -> None: ...


@contextmanager
def owned_resource[T: Closable](
    lifetime: DbLifetime,
    operation: str,
    opener: Callable[[], T],
) -> Iterator[T]:
    """Observe a scoped resource before close and preserve independent failures."""
    resource = lifetime.call_db(f"{operation}:open", opener)
    try:
        yield resource
    except BaseException as primary:
        lifetime.observe(primary, operation)
        try:
            lifetime.cleanup_db(f"{operation}:close", resource.close)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                f"database unit and close failed: {primary}; cleanup: {cleanup}", [primary, cleanup]
            ) from None
        raise
    else:
        lifetime.cleanup_db(f"{operation}:close", resource.close)


@contextmanager
def owned_connection(
    lifetime: DbLifetime,
    operation: str,
    opener: Callable[[], duckdb.DuckDBPyConnection],
) -> Iterator[DbLease]:
    """Keep a private auxiliary handle observed through its entire scope."""
    with (
        owned_resource(lifetime, operation, lambda: _OwnedConnection(opener())) as owned,
        lifetime.lease(owned.connection) as lease,
    ):
        yield lease


@dataclass
class _OwnedConnection:
    connection: duckdb.DuckDBPyConnection

    def close(self) -> None:
        self.connection.close()
        _completion_state(self.connection).closed = True


class StandaloneLifetime:
    """Scoped DB capabilities without daemon marker or shutdown policy."""

    generation = 1

    def __init__(self) -> None:
        self._terminal_error: BaseException | None = None
        self._arbiter = RLock()
        self._identity = object()
        self._pending_completion: CompletionFailure | None = None

    @property
    def pending_completion(self) -> CompletionFailure | None:
        with self._arbiter:
            return self._pending_completion

    def completion_failed(self, error: BaseException) -> CompletionFailure:
        with self._arbiter:
            assert self._pending_completion is None, "completion failure is already unresolved"
            self._pending_completion = CompletionFailure(error)
            return self._pending_completion

    def resolve_completion(self, failure: CompletionFailure) -> None:
        with self._arbiter:
            assert self._pending_completion is failure, "completion resolution is stale"
            self._pending_completion = None

    def call_db[T](self, operation: str, work: Callable[[], T]) -> T:
        if self._terminal_error is not None:
            raise self._terminal_error
        if self._pending_completion is not None:
            raise CompletionReadRequired(
                "fresh completion diagnosis is required"
            ) from self._pending_completion.error
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
        with self._arbiter:
            if terminal and self._terminal_error is None:
                self._terminal_error = graph.failure if graph.failure is not None else error
        return terminal

    def completion_permit(self) -> CompletionPermit:
        with self._arbiter:
            self._require_completion_admission()
            return CompletionPermit(self._identity, self.generation)

    def _require_completion_admission(self) -> None:
        if self._terminal_error is not None:
            raise self._terminal_error
        if self._pending_completion is not None:
            raise CompletionReadRequired(
                "fresh completion diagnosis is required"
            ) from self._pending_completion.error

    def reject_admission(self, error: BaseException, operation: str) -> None:
        with self._arbiter:
            if self._terminal_error is None:
                self._terminal_error = error

    @contextmanager
    def admitting(self) -> Iterator[None]:
        with self._arbiter:
            self.call_db("admission", lambda: None)
            yield

    @contextmanager
    def completing(self, permit: CompletionPermit) -> Iterator[CompletionRelease]:
        with self._arbiter:
            permit.assert_unused()

            def check() -> None:
                self._require_completion_admission()
                assert (
                    permit._identity is self._identity and permit.generation == self.generation
                ), "completion authority belongs to another lifetime"

            check()
            permit.consume()
            release = CompletionRelease(check)
            try:
                yield release
            finally:
                release.expire()

    def connection_epoch(self, connection: duckdb.DuckDBPyConnection) -> object:
        return connection_epoch(connection)

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
        self._completion_state = _completion_state(connection)

    def call[T](self, operation: str, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        self._require_native_admission()
        value = self._owner.call_db(operation, lambda: work(self._connection))
        assert not isinstance(value, duckdb.DuckDBPyConnection), "DB handle escaped its lease"
        return value

    def _require_native_admission(self) -> None:
        if self._completion_state.failed_completion is not None:
            raise CompletionReadRequired(
                "failed completion requires closing this native connection and a fresh diagnosis"
            ) from self._completion_state.failed_completion.error

    @contextmanager
    def admitting(self) -> Iterator[None]:
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        with self._owner.admitting():
            self._require_native_admission()
            yield

    def cleanup_call[T](self, operation: str, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        """Compensation/rollback remains observed after ordinary DB admission closes."""
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        value = self._owner.cleanup_db(operation, lambda: work(self._connection))
        assert not isinstance(value, duckdb.DuckDBPyConnection), "DB handle escaped its lease"
        return value

    def observe(self, error: BaseException, operation: str) -> bool:
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        return self._owner.observe(error, operation)

    def completion_permit(self) -> CompletionPermit:
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        return self._owner.completion_permit()

    def reject_admission(self, error: BaseException, operation: str) -> Never:
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        self._owner.reject_admission(error, operation)
        raise error

    def completion_failed(self, error: BaseException) -> None:
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        failure = self._owner.completion_failed(error)
        assert self._completion_state.failed_completion is None, "native epoch already failed"
        self._completion_state.failed_completion = failure

    @property
    def completion_unresolved(self) -> bool:
        return self._completion_state.failed_completion is not None

    @property
    def connection_closed(self) -> bool:
        """Acknowledged close receipt remains readable after this lease expires."""
        return self._completion_state.closed

    @contextmanager
    def completing(self, permit: CompletionPermit) -> Iterator[CompletionRelease]:
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        with self._owner.completing(permit) as release:
            yield release

    @property
    def observer(self) -> FailureObserver:
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        return self._owner

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
            if not transaction._discarded:
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
        else:
            if transaction._discarded:
                self.cleanup_call(
                    f"{operation}:discard", lambda conn: conn.execute("ROLLBACK").fetchone()
                )
        finally:
            transaction._active = False


class TransactionCapability:
    """Only the outer lease scope begins/commits; apply functions receive this."""

    def __init__(self, lease: DbLease) -> None:
        self._lease = lease
        self._active = True
        self._discarded = False

    def call[T](self, operation: str, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        assert self._active, "expired transaction capability"
        assert not self._discarded, "transaction was discarded"
        return self._lease.call(operation, work)

    def discard(self) -> None:
        """Request outer rollback; this capability can admit no further effects."""
        assert self._active and not self._discarded, "transaction is not active"
        self._discarded = True

    def cleanup_call[T](self, operation: str, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        assert self._active, "expired transaction capability"
        return self._lease.cleanup_call(operation, work)

    def observe(self, error: BaseException, operation: str) -> bool:
        assert self._active, "expired transaction capability"
        return self._lease.observe(error, operation)
