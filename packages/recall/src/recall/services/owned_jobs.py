"""Observation and lifetime ownership for daemon work.

Owned outcomes cross the policy boundary only after exception graph inspection.
The arbiter orders the first fatal against completion commits; shutdown and a
failed diagnostic cannot detach observation from an admitted worker.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from weakref import WeakKeyDictionary

import duckdb

from recall.core.repair_state import PublicationRole
from recall.core.telemetry import DiagnosticExecutor
from recall.db.fatal import FatalGraph, inspect_fatal_graph
from recall.services.scheduling import demote_current_thread

logger = logging.getLogger("recall.owned_jobs")


@dataclass(frozen=True)
class Completed[T]:
    value: T


@dataclass(frozen=True)
class ObservedFailure:
    error: BaseException
    operation: str

    def raise_error(self) -> None:
        raise self.error


@dataclass(frozen=True)
class FatalObservation:
    operation: str
    graph: FatalGraph


class JobLane(StrEnum):
    WRITER = "writer"
    READ = "read"
    QUERY_MODEL = "query-model"
    ENRICHMENT = "enrichment"
    PREPARATION = "preparation"
    TRANSPORT = "transport"
    CLEANUP = "cleanup"


_CAPACITIES = {
    JobLane.WRITER: 4,
    JobLane.READ: 8,
    JobLane.QUERY_MODEL: 1,
    JobLane.ENRICHMENT: 1,
    JobLane.PREPARATION: 2,
    JobLane.TRANSPORT: 2,
    JobLane.CLEANUP: 2,
}


class WorkNotAdmitted(RuntimeError):
    """The lifetime is closed, terminal, or its bounded work slots are full."""


@dataclass(frozen=True)
class CompletionPermit:
    _identity: object
    generation: int


class DbLease:
    """A required live capability for a supported DB leaf callback.

    Raw handles stay a reviewed boundary: a callback must materialize its value
    result inside call(), never retain or return a handle for later use.
    """

    def __init__(self, owner: DaemonJobs, connection: duckdb.DuckDBPyConnection) -> None:
        self._owner = owner
        self._connection = connection
        self._active = True
        self._generation = owner.generation
        self._connection_epoch = owner._connection_epochs.setdefault(connection, object())

    def call[T](self, operation: str, work: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        assert self._active and self._generation == self._owner.generation, "expired DB lease"
        outcome = self._owner.inline(operation, lambda: work(self._connection))
        if isinstance(outcome, ObservedFailure):
            outcome.raise_error()
            raise AssertionError("failure did not propagate")
        assert not isinstance(outcome.value, duckdb.DuckDBPyConnection), (
            "DB handle escaped its lease"
        )
        return outcome.value

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
                self.call(f"{operation}:rollback", lambda conn: conn.execute("ROLLBACK").fetchone())
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


@dataclass(frozen=True)
class OwnedJob[T]:
    _future: Future[Completed[T] | ObservedFailure]

    async def wait(self) -> Completed[T] | ObservedFailure:
        # Cancelling a client never cancels the worker's observation or lifetime.
        return await asyncio.shield(asyncio.wrap_future(self._future))


class DaemonJobs:
    def __init__(
        self,
        publish: Callable[[BaseException, str, PublicationRole], None],
        shutdown: Callable[[], None],
        hold: Callable[[str], None] | None = None,
    ) -> None:
        self._publish = publish
        self._hold = hold
        self._shutdown = shutdown
        self._arbiter = threading.RLock()
        self._identity = object()
        self._connection_epochs: WeakKeyDictionary[duckdb.DuckDBPyConnection, object] = (
            WeakKeyDictionary()
        )
        self._generation = 1
        self._first_fatal: FatalObservation | None = None
        self._observed: dict[int, BaseException] = {}
        self._pools: dict[JobLane, ThreadPoolExecutor] = {}
        self._pending: set[Future[Any]] = set()
        self._slots = {
            lane: threading.BoundedSemaphore(size + 64) for lane, size in _CAPACITIES.items()
        }
        self._closed = False
        self._closing = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._operation: ContextVar[str | None] = ContextVar("owned_db_operation", default=None)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def first_fatal(self) -> FatalObservation | None:
        with self._arbiter:
            return self._first_fatal

    @property
    def operation(self) -> str | None:
        return self._operation.get()

    def observe(self, error: BaseException, operation: str) -> bool:
        graph = inspect_fatal_graph(error)
        if graph.complete and graph.failure is None:
            return False
        with self._arbiter:
            role = PublicationRole.SECONDARY
            if self._first_fatal is None:
                self._first_fatal = FatalObservation(operation, graph)
                role = PublicationRole.PRIMARY
            try:
                selected = graph.failure
                if (
                    graph.complete
                    and selected is not None
                    and id(selected) not in self._observed
                    and len(self._observed) < 65_536
                ):
                    self._observed[id(selected)] = selected
                    with suppress(Exception):
                        self._publish(selected, operation, role)
                if not graph.complete:
                    if self._hold is not None:
                        with suppress(Exception):
                            self._hold(operation)
                    with suppress(Exception):
                        logger.error("incomplete exception classification operation=%s", operation)
            finally:
                if self._loop is None or self._loop_thread == threading.get_ident():
                    self._shutdown()
                else:
                    self._loop.call_soon_threadsafe(self._shutdown)
        return True

    def inline[T](self, operation: str, work: Callable[[], T]) -> Completed[T] | ObservedFailure:
        token = self._operation.set(self._operation.get() or operation)
        try:
            return Completed(work())
        except BaseException as error:
            self.observe(error, operation)
            if not isinstance(error, Exception):
                raise
            return ObservedFailure(error, operation)
        finally:
            self._operation.reset(token)

    @contextmanager
    def lease(self, connection: duckdb.DuckDBPyConnection) -> Iterator[DbLease]:
        with self._arbiter:
            if self._closed:
                raise WorkNotAdmitted("DB lifetime is closed")
            lease = DbLease(self, connection)
        try:
            yield lease
        finally:
            lease._active = False

    def completion_permit(self) -> CompletionPermit:
        with self._arbiter:
            self._require_admission()
            return CompletionPermit(self._identity, self._generation)

    @contextmanager
    def completing(self, permit: CompletionPermit) -> Iterator[None]:
        with self._arbiter:
            self._require_admission()
            if permit._identity is not self._identity or permit.generation != self._generation:
                raise WorkNotAdmitted("completion permit belongs to a different owner generation")
            yield

    def _require_admission(self) -> None:
        if self._closed or self._closing or self._first_fatal is not None:
            raise WorkNotAdmitted("daemon work admission is closed")

    def submit[T](self, lane: JobLane, operation: str, work: Callable[[], T]) -> OwnedJob[T]:
        return self._submit(lane, operation, work, cleanup=False)

    def _submit[T](
        self, lane: JobLane, operation: str, work: Callable[[], T], *, cleanup: bool
    ) -> OwnedJob[T]:
        with self._arbiter:
            if cleanup:
                if self._closed:
                    raise WorkNotAdmitted("cleanup lifetime is closed")
            else:
                self._require_admission()
            slot = self._slots[lane]
            if not slot.acquire(blocking=False):
                raise WorkNotAdmitted(f"{lane} work capacity is full")
            try:
                pool = self._pools.get(lane)
                if pool is None:
                    pool = DiagnosticExecutor(
                        max_workers=_CAPACITIES[lane],
                        thread_name_prefix=f"recall-{lane}",
                        initializer=demote_current_thread if lane == JobLane.ENRICHMENT else None,
                    )
                    self._pools[lane] = pool

                def run() -> Completed[T] | ObservedFailure:
                    return self.inline(operation, work)

                future = pool.submit(run)
            except BaseException as error:
                slot.release()
                self.observe(error, f"{operation}:submit")
                raise
            self._pending.add(future)

            def finished(completed: Future[Completed[T] | ObservedFailure]) -> None:
                with self._arbiter:
                    self._pending.discard(completed)
                    slot.release()

            future.add_done_callback(finished)
            return OwnedJob(future)

    async def drain(self) -> None:
        with self._arbiter:
            pending = tuple(self._pending)
        if pending:
            await asyncio.gather(
                *(asyncio.shield(asyncio.wrap_future(future)) for future in pending),
                return_exceptions=True,
            )

    async def close(self) -> None:
        with self._arbiter:
            if self._closed:
                return
            self._closing = True
        current = asyncio.current_task()
        tasks = tuple(task for task in self._tasks if task is not current)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.drain()
        with self._arbiter:
            self._closed = True
        # Work is already drained. Shutdown joins only idle owned threads.
        for lane, pool in self._pools.items():
            outcome = self.inline(
                f"{lane}:pool_shutdown", lambda pool=pool: pool.shutdown(wait=True)
            )
            if isinstance(outcome, ObservedFailure):
                outcome.raise_error()
        self._pools.clear()
        self._generation += 1

    def finish[T](self, operation: str, work: Callable[[], T]) -> asyncio.Future[T]:
        """Explicit cleanup still runs after terminal ordinary admission."""
        job = self._submit(JobLane.CLEANUP, operation, work, cleanup=True)
        return self._value_future(job)

    def _value_future[T](self, job: OwnedJob[T]) -> asyncio.Future[T]:
        async def value() -> T:
            result = await job.wait()
            if isinstance(result, ObservedFailure):
                result.raise_error()
                raise AssertionError("failure did not propagate")
            return result.value

        return asyncio.ensure_future(value())

    def start[T](self, lane: JobLane, operation: str, work: Callable[[], T]) -> asyncio.Future[T]:
        """Expose a value future only after the worker's observer is installed."""
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        job = self.submit(lane, operation, work)

        return self._value_future(job)

    async def observe_async[T](self, operation: str, work: Callable[[], Awaitable[T]]) -> T:
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        try:
            return await work()
        except BaseException as error:
            self.observe(error, operation)
            raise

    def spawn[T](self, operation: str, work: Callable[[], Awaitable[T]]) -> asyncio.Task[T]:
        if len(self._tasks) >= 1024:
            raise WorkNotAdmitted("owned async work capacity is full")
        task = asyncio.create_task(self.observe_async(operation, work))
        self._tasks.add(task)

        def finished(completed: asyncio.Task[T]) -> None:
            self._tasks.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(finished)
        return task
