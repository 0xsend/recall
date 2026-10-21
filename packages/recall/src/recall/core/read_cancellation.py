"""Cancellation shared by database engines within one synchronous read worker."""

from __future__ import annotations

from contextvars import ContextVar, Token
from threading import Event
from typing import Literal

_cancelled: ContextVar[Event | None] = ContextVar("read_cancelled", default=None)


class ReadCancelled(RuntimeError):
    """The owner cancelled this read before its next database operation."""


def read_cancelled() -> bool:
    event = _cancelled.get()
    return event is not None and event.is_set()


def check_read_cancelled() -> None:
    if read_cancelled():
        raise ReadCancelled("read cancelled")


class _ReadCancellation:
    """Restore cancellation state without modifying an in-flight exception."""

    def __init__(self, event: Event) -> None:
        self._event = event
        self._token: Token[Event | None] | None = None

    def __enter__(self) -> None:
        assert self._token is None
        token = _cancelled.set(self._event)
        try:
            check_read_cancelled()
        except BaseException:
            _cancelled.reset(token)
            raise
        self._token = token

    def __exit__(self, *_exc_info: object) -> Literal[False]:
        assert self._token is not None
        _cancelled.reset(self._token)
        self._token = None
        return False


def read_cancellation(event: Event) -> _ReadCancellation:
    return _ReadCancellation(event)
