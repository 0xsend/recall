"""Invocation-owned local read capabilities, independent of CLI rendering."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from recall.core.config import AppConfig
from recall.core.fleet import FleetScope, LocalEndpoint
from recall.core.rpc_client import RpcClient
from recall.read_budget import ReadBudget


@dataclass(frozen=True)
class OrdinaryRead:
    """Ordinary reads retain the caller's lifecycle callbacks and deadline."""

    def call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        auto_fork: bool,
        before_auto_fork: Callable[[AppConfig], None],
        budget: ReadBudget | None,
        idle_timeout: float | None,
        on_progress: Callable[[dict[str, Any]], None] | None,
        on_notification: Callable[[str, dict[str, Any]], None] | None,
        connected: Callable[[], None],
    ) -> Any:
        client = RpcClient(before_auto_fork=before_auto_fork)
        try:
            client.connect(auto_fork=auto_fork, read_budget=budget)
            connected()
            return client.call(
                method,
                params,
                idle_timeout=idle_timeout,
                read_budget=budget,
                on_progress=on_progress,
                on_notification=on_notification,
            )
        finally:
            client.close()


@dataclass(frozen=True)
class AggregateRead:
    """A live socket read with no lifecycle or snapshot authority."""

    scope: FleetScope
    config: AppConfig

    def call(self, method: str, params: dict[str, Any], *, seconds: float) -> Any:
        if not any(isinstance(endpoint, LocalEndpoint) for endpoint in self.scope.endpoints):
            raise ValueError("local endpoint is not selected")
        budget = ReadBudget.start(seconds)
        client = RpcClient(config=self.config)
        try:
            client.connect(auto_fork=False, read_budget=budget)
            return client.call(method, params, idle_timeout=seconds, read_budget=budget)
        finally:
            client.close()


@dataclass
class ReadInvocation:
    capability: OrdinaryRead | AggregateRead = field(default_factory=OrdinaryRead)
    local_succeeded: bool = False
    status_attempted: bool = False
    status: dict[str, Any] | None = None

    def bind(self, capability: AggregateRead) -> None:
        assert isinstance(self.capability, OrdinaryRead), "read scope cannot be rebound"
        self.capability = capability

    def aggregate_status(self) -> dict[str, Any] | None:
        assert isinstance(self.capability, AggregateRead)
        if not self.local_succeeded or self.status_attempted:
            return self.status
        self.status_attempted = True
        # Advisory failure never changes the primary result.
        from recall.core.rpc_client import RpcConnectionError
        from recall.core.rpc_types import RpcAdmissionBusyError, RpcCallError
        from recall.core.schema_version import SchemaTooNewError

        try:
            result = self.capability.call("recall.daemon_status", {"limit": 1}, seconds=2.0)
        except (
            RpcConnectionError,
            RpcCallError,
            RpcAdmissionBusyError,
            SchemaTooNewError,
            ValueError,
        ):
            return None
        self.status = result if isinstance(result, dict) else None
        return self.status


_active: ContextVar[ReadInvocation | None] = ContextVar("read_invocation", default=None)


def begin_invocation() -> Token[ReadInvocation | None]:
    return _active.set(ReadInvocation())


def end_invocation(token: Token[ReadInvocation | None]) -> None:
    _active.reset(token)


def current_invocation() -> ReadInvocation:
    invocation = _active.get()
    if invocation is None:
        invocation = ReadInvocation()
    return invocation


def bind_aggregate(capability: AggregateRead) -> None:
    invocation = current_invocation()
    invocation.bind(capability)
    _active.set(invocation)
