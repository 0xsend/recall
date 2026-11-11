"""Observation authority passed to work that cannot hold a database lease."""

from typing import Protocol


class FailureObserver(Protocol):
    def observe(self, error: BaseException, operation: str) -> bool:
        """Inspect an error before policy; return whether its graph is terminal."""
        ...
