"""Observation authority passed to work that cannot hold a database lease."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol


class FailureObserver(Protocol):
    def observe(self, error: BaseException, operation: str) -> bool:
        """Inspect an error before policy; return whether its graph is terminal."""
        ...


@contextmanager
def observe_failure(observer: FailureObserver, operation: str) -> Iterator[None]:
    """Observe the native exception before an inner handler translates it."""
    try:
        yield
    except BaseException as error:
        observer.observe(error, operation)
        raise


def _observed_cleanup(
    observer: FailureObserver,
    operation: str,
    cleanup: Callable[[], object],
    primary: BaseException | None,
) -> None:
    if primary is not None:
        observer.observe(primary, operation)
    try:
        cleanup()
    except BaseException as error:
        observer.observe(error, operation + ":cleanup")
        if primary is not None:
            raise BaseExceptionGroup("operation and cleanup failed", [primary, error]) from None
        raise


@contextmanager
def observe_before_cleanup(
    observer: FailureObserver, operation: str, close: Callable[[], None]
) -> Iterator[None]:
    """Keep an originating failure ahead of independent native cleanup failures."""
    try:
        yield
    except BaseException as primary:
        _observed_cleanup(observer, operation, close, primary)
        raise
    else:
        _observed_cleanup(observer, operation, close, None)


@contextmanager
def rollback_on_failure(
    observer: FailureObserver, operation: str, rollback: Callable[[], object]
) -> Iterator[None]:
    """Observe a transaction failure before rollback; retain both if rollback fails."""
    try:
        yield
    except BaseException as primary:
        _observed_cleanup(observer, operation, rollback, primary)
        raise
