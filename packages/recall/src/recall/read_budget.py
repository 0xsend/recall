"""Process-local responsive read deadlines; only durations cross the RPC wire.

This module stays dependency-free so the CLI can start its clock before
importing the command tree.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

RESPONSIVE_READ_SECONDS = 2.0
RESPONSIVE_READ_METHODS = frozenset({"recall.live", "recall.daemon_status"})
# Leave room for receiving, decoding, projecting and writing the response.
READ_OUTPUT_RESERVE_SECONDS = 0.1
CLI_READ_BUDGET: ContextVar[ReadBudget | None] = ContextVar("cli_read_budget", default=None)
SERVER_READ_BUDGET: ContextVar[ReadBudget | None] = ContextVar("server_read_budget", default=None)


@dataclass(frozen=True)
class ReadBudget:
    deadline: float

    @classmethod
    def start(cls, seconds: float = RESPONSIVE_READ_SECONDS) -> ReadBudget:
        return cls(time.monotonic() + seconds)

    def remaining(self, *, reserve: float = 0.0) -> float:
        return max(0.0, self.deadline - time.monotonic() - reserve)


@contextmanager
def cli_read_budget() -> Iterator[None]:
    token = CLI_READ_BUDGET.set(ReadBudget.start())
    try:
        yield
    finally:
        CLI_READ_BUDGET.reset(token)
