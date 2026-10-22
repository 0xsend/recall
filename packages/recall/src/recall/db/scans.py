"""Admission for full-corpus statements inside daemon database workers."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Literal

import duckdb

from recall.core.read_cancellation import check_read_cancelled

ScanPriority = Literal["ordinary", "status"]
_ClaimKind = Literal["ordinary", "status", "publication"]
_SCAN_WAIT_SECONDS = 30.0
# One coherent pending count and one live snapshot can precede ordinary work.
_STATUS_BURST_MAX = 2


@dataclass(frozen=True, eq=False)
class _Claim:
    priority: _ClaimKind


class ScanCapacity:
    """Serialize actual SQL scans, without holding capacity during model work.

    A fresh status scan may pass waiting ordinary scans. One ordinary scan
    follows each bounded status burst, so continuous status polling cannot starve
    enrichment or search. Capacity remains owned until all result rows are read.
    A worker may group related statements inside one reentrant lease.
    Bounded publication may overlap a status snapshot, but never an ordinary
    corpus scan. This leaves fresh status available during a writer checkpoint.
    """

    def __init__(self, waiters_max: int) -> None:
        assert waiters_max > 0
        self._waiters_max = waiters_max
        self._condition = threading.Condition()
        self._waiting: deque[_Claim] = deque()
        self._owner: _Claim | None = None
        self._owner_thread: int | None = None
        self._publisher: _Claim | None = None
        self._publisher_thread: int | None = None
        self._status_streak = 0

    def bind(self) -> _Scope:
        return _Scope(self)

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def lease(self, priority: ScanPriority) -> _Lease:
        return _Lease(self, _Claim(priority))

    def publication(self) -> _Lease:
        return _Lease(self, _Claim("publication"))

    def _owned_by_current_thread(self) -> bool:
        with self._condition:
            return self._owner_thread == threading.get_ident()

    def _runnable(self, claim: _Claim) -> bool:
        if claim.priority == "publication":
            return self._publisher is None and (
                self._owner is None or self._owner.priority == "status"
            )
        return self._owner is None and (self._publisher is None or claim.priority == "status")

    def _next(self) -> _Claim:
        assert self._waiting
        runnable = [claim for claim in self._waiting if self._runnable(claim)]
        assert runnable
        prefer_status = self._status_streak < _STATUS_BURST_MAX
        return next(
            (claim for claim in runnable if (claim.priority == "status") == prefer_status),
            runnable[0],
        )

    def _acquire(self, claim: _Claim) -> None:
        deadline = time.monotonic() + _SCAN_WAIT_SECONDS
        with self._condition:
            assert self._owner_thread != threading.get_ident(), "scan leases cannot nest"
            assert self._publisher_thread != threading.get_ident(), (
                "publication must not contain corpus scans"
            )
            assert len(self._waiting) < self._waiters_max
            self._waiting.append(claim)
            try:
                while not self._runnable(claim) or self._next() is not claim:
                    check_read_cancelled()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("full-corpus scan admission timed out")
                    self._condition.wait(timeout=remaining)
                check_read_cancelled()
                self._waiting.remove(claim)
                if claim.priority == "publication":
                    self._publisher = claim
                    self._publisher_thread = threading.get_ident()
                else:
                    self._owner = claim
                    self._owner_thread = threading.get_ident()
                self._status_streak = (
                    min(_STATUS_BURST_MAX, self._status_streak + 1)
                    if claim.priority == "status"
                    else 0
                )
                self._condition.notify_all()
            except BaseException:
                self._waiting.remove(claim)
                self._condition.notify_all()
                raise

    def _release(self, claim: _Claim) -> None:
        with self._condition:
            if claim.priority == "publication":
                assert self._publisher is claim and self._publisher_thread == threading.get_ident()
                self._publisher = None
                self._publisher_thread = None
            else:
                assert self._owner is claim and self._owner_thread == threading.get_ident()
                self._owner = None
                self._owner_thread = None
            self._condition.notify_all()


class _Scope:
    def __init__(self, capacity: ScanCapacity) -> None:
        self._capacity = capacity
        self._token: Token[ScanCapacity | None] | None = None

    def __enter__(self) -> None:
        assert self._token is None
        self._token = _capacity.set(self._capacity)

    def __exit__(self, *_exc_info: object) -> Literal[False]:
        assert self._token is not None
        _capacity.reset(self._token)
        self._token = None
        return False


class _Lease:
    def __init__(self, capacity: ScanCapacity, claim: _Claim) -> None:
        self._capacity = capacity
        self._claim = claim
        self._state: Literal["idle", "owner", "borrower"] = "idle"

    def __enter__(self) -> None:
        assert self._state == "idle"
        check_read_cancelled()
        if self._capacity._owned_by_current_thread():
            assert self._claim.priority != "publication", "scans must not publish"
            self._state = "borrower"
        else:
            self._capacity._acquire(self._claim)
            self._state = "owner"

    def __exit__(self, *_exc_info: object) -> Literal[False]:
        assert self._state != "idle"
        if self._state == "owner":
            self._capacity._release(self._claim)
        self._state = "idle"
        return False


_capacity: ContextVar[ScanCapacity | None] = ContextVar("corpus_scan_capacity", default=None)


def execute_scan(
    conn: duckdb.DuckDBPyConnection,
    statement: str,
    parameters: Sequence[object] | Mapping[str, object] | None = None,
    *,
    priority: ScanPriority = "ordinary",
) -> list[tuple[Any, ...]]:
    """Execute and exhaust a full-corpus query under its worker's capacity.

    Standalone connections have no daemon worker scope. The daemon binds one
    capacity to its writer and read workers. Callers consume the result here
    rather than retaining a native cursor after releasing admission.
    """
    capacity = _capacity.get()
    lease = capacity.lease(priority) if capacity is not None else nullcontext()
    with lease:
        check_read_cancelled()
        return conn.execute(statement, parameters).fetchall()


def execute_scan_one(
    conn: duckdb.DuckDBPyConnection,
    statement: str,
    parameters: Sequence[object] | Mapping[str, object] | None = None,
) -> tuple[Any, ...] | None:
    """Return the first row after consuming the result under admission."""
    rows = execute_scan(conn, statement, parameters)
    return rows[0] if rows else None
