"""Observation and lifetime ownership for daemon work.

Owned outcomes cross the policy boundary only after exception graph inspection.
The arbiter orders the first fatal against completion commits; shutdown and a
failed diagnostic cannot detach observation from an admitted worker.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import duckdb

from recall.core.repair_state import PublicationRole
from recall.core.telemetry import DiagnosticExecutor
from recall.db.fatal import FatalGraph, inspect_fatal_graph
from recall.db.lease import CompletionFailure, CompletionPermit, DbLease, connection_epoch
from recall.services.preparation import PreparationQueue
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
        *,
        verification: Callable[[BaseException, str], None] | None = None,
    ) -> None:
        self._publish = publish
        self._hold = hold
        self._verification = verification
        self._shutdown = shutdown
        self._arbiter = threading.RLock()
        self._identity = object()
        self._generation = 1
        self._pending_completion: CompletionFailure | None = None
        self._first_fatal: FatalObservation | None = None
        self._terminal_control: BaseException | None = None
        self._recovery_hold = False
        self._observed: dict[int, BaseException] = {}
        self._pools: dict[JobLane, ThreadPoolExecutor] = {}
        self._pending: set[Future[Any]] = set()
        self._preparations: set[PreparationQueue[Any, Any]] = set()
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
        if not isinstance(error, (Exception, asyncio.CancelledError)):
            with self._arbiter:
                if self._terminal_control is None:
                    self._terminal_control = error
                self._request_shutdown()
        if graph.complete and graph.failure is None and not graph.verification_candidates:
            return False
        with self._arbiter:
            terminal = not graph.complete or graph.failure is not None
            role = PublicationRole.SECONDARY
            if terminal and self._first_fatal is None:
                self._first_fatal = FatalObservation(operation, graph)
                role = PublicationRole.PRIMARY
            try:
                unseen = tuple(node for node in graph.candidates if id(node) not in self._observed)
                disk = tuple(
                    node for node in graph.verification_candidates if id(node) not in self._observed
                )
                if graph.complete and (unseen or disk):
                    if len(self._observed) + len(unseen) + len(disk) <= 65_536:
                        # Re-propagating one graph is not a new event. A later
                        # cleanup can share its context while introducing its
                        # own independently observed diagnostic.
                        self._observed.update((id(node), node) for node in (*unseen, *disk))
                        if unseen:
                            with suppress(Exception):
                                self._publish(unseen[0], operation, role)
                        if disk and self._verification is not None:
                            try:
                                self._verification(disk[0], operation)
                            except Exception as error:
                                self._diagnostic_hold(
                                    f"{operation}:verification_publication:{error}"
                                )
                    else:
                        self._diagnostic_hold(f"{operation}:observation_identity_capacity")
                if not graph.complete:
                    if self._hold is not None:
                        with suppress(Exception):
                            self._hold(operation)
                    with suppress(Exception):
                        logger.error("incomplete exception classification operation=%s", operation)
            finally:
                if terminal or self._recovery_hold:
                    self._request_shutdown()
        return terminal or self._recovery_hold

    def _diagnostic_hold(self, operation: str) -> None:
        self._recovery_hold = True
        if self._hold is not None:
            with suppress(Exception):
                self._hold(operation)

    def reject_admission(self, error: BaseException, operation: str) -> None:
        with self._arbiter:
            try:
                self._diagnostic_hold(f"{operation}: {error}")
            finally:
                self._request_shutdown()

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
            self._generation += 1

    def _request_shutdown(self) -> None:
        if self._loop is None or self._loop_thread == threading.get_ident():
            self._shutdown()
        else:
            self._loop.call_soon_threadsafe(self._shutdown)

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

    def call_db[T](self, operation: str, work: Callable[[], T]) -> T:
        with self._arbiter:
            if self._closed:
                raise WorkNotAdmitted("DB lifetime is closed")
            if self._terminal_control is not None:
                raise self._terminal_control
            if self._recovery_hold:
                raise WorkNotAdmitted("recovery diagnostics closed DB admission")
            if self._pending_completion is not None:
                raise WorkNotAdmitted(
                    "fresh completion diagnosis is required"
                ) from self._pending_completion.error
            if self._first_fatal is not None:
                failure = self._first_fatal.graph.failure
                if failure is not None:
                    raise failure
                raise WorkNotAdmitted("incomplete exception classification closed DB admission")
        return self.cleanup_db(operation, work)

    def cleanup_db[T](self, operation: str, work: Callable[[], T]) -> T:
        outcome = self.inline(operation, work)
        if isinstance(outcome, ObservedFailure):
            outcome.raise_error()
            raise AssertionError("failure did not propagate")
        return outcome.value

    def connection_epoch(self, connection: duckdb.DuckDBPyConnection) -> object:
        return connection_epoch(connection)

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
    def admitting(self) -> Iterator[None]:
        with self._arbiter:
            self._require_admission()
            yield

    @contextmanager
    def completing(self, permit: CompletionPermit) -> Iterator[None]:
        with self._arbiter:
            permit.assert_unused()
            self._require_admission()
            if permit._identity is not self._identity or permit.generation != self._generation:
                raise WorkNotAdmitted("completion permit belongs to a different owner generation")
            permit.consume()
            yield

    def _require_admission(self) -> None:
        if (
            self._closed
            or self._closing
            or self._first_fatal is not None
            or self._terminal_control is not None
            or self._recovery_hold
            or self._pending_completion is not None
        ):
            origin = self._first_fatal.graph.failure if self._first_fatal is not None else None
            raise WorkNotAdmitted("daemon work admission is closed") from origin

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

    def prepare[K, V](
        self, work: Iterable[tuple[K, Callable[[], V]]], *, workers: int
    ) -> PreparationQueue[K, V]:
        with self._arbiter:
            self._require_admission()
            if len(self._preparations) >= 16:
                raise WorkNotAdmitted("preparation queue capacity is full")
            queue = PreparationQueue(
                work, workers=workers, observe=self.observe, retired=self._retire_preparation
            )
            self._preparations.add(queue)
            return queue

    def _retire_preparation(self, queue: PreparationQueue[Any, Any]) -> None:
        with self._arbiter:
            self._preparations.discard(queue)

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
        # Work is already drained. Every pool joins before a terminal outcome
        # propagates; one failed join must not abandon the remaining pools.
        errors: list[BaseException] = []
        for queue in tuple(self._preparations):
            try:
                queue.close()
            except BaseException as error:
                errors.append(error)
        for lane, pool in self._pools.items():
            try:
                self.cleanup_db(f"{lane}:pool_shutdown", lambda pool=pool: pool.shutdown(wait=True))
            except BaseException as error:
                errors.append(error)
        self._pools.clear()
        self._generation += 1
        terminal = self._terminal_control
        if terminal is not None and all(error is not terminal for error in errors):
            errors.insert(0, terminal)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("owned work and pool cleanup failed", errors)

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
