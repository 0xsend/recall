"""Cancellation shared by database engines within one synchronous read worker."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event

_cancelled: ContextVar[Event | None] = ContextVar("read_cancelled", default=None)


class ReadCancelled(RuntimeError):
    """The owner cancelled this read before its next database operation."""


def read_cancelled() -> bool:
    event = _cancelled.get()
    return event is not None and event.is_set()


def check_read_cancelled() -> None:
    if read_cancelled():
        raise ReadCancelled("read cancelled")


@contextmanager
def read_cancellation(event: Event) -> Iterator[None]:
    token = _cancelled.set(event)
    try:
        check_read_cancelled()
        yield
    finally:
        _cancelled.reset(token)
