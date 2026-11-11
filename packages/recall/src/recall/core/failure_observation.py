"""Observation authority passed to work that cannot hold a database lease."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol


class FailureObserver(Protocol):
    def observe(self, error: BaseException, operation: str) -> bool:
        """Inspect an error before policy; return whether its graph is terminal."""
        ...


@contextmanager
def observe_before_cleanup(
    observer: FailureObserver, operation: str, close: Callable[[], None]
) -> Iterator[None]:
    """Keep an originating failure ahead of independent native cleanup failures."""
    try:
        yield
    except BaseException as primary:
        observer.observe(primary, operation)
        try:
            close()
        except BaseException as cleanup:
            observer.observe(cleanup, operation + ":close")
            raise BaseExceptionGroup("database scan and close failed", [primary, cleanup]) from None
        raise
    else:
        try:
            close()
        except BaseException as cleanup:
            observer.observe(cleanup, operation + ":close")
            raise
