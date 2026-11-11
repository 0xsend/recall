"""Ordered, bounded execution of a resolved fleet query."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from recall.core.config import AppConfig
from recall.core.fleet import FleetConfig, FleetEndpoint, FleetScope, LocalEndpoint, SshEndpoint
from recall.core.host_text import HostDiagnostic, HostMessageError, HostValue
from recall.core.read_scope import AggregateRead, current_invocation
from recall.core.rpc_client import RpcCallError, RpcConnectionError
from recall.core.rpc_params import Params
from recall.core.rpc_types import APP_NOT_FOUND, RpcAdmissionBusyError
from recall.core.schema_version import SchemaTooNewError
from recall.core.types import default_session_host
from recall.services import fleet_transport
from recall.services.fleet_transport import DEFAULT_CONCURRENCY, extract_remote_error


@dataclass(frozen=True)
class EndpointSuccess:
    name: HostValue
    value: Any


@dataclass(frozen=True)
class EndpointFailure:
    name: HostValue
    error: str


@dataclass(frozen=True)
class EndpointNotFound:
    name: HostValue
    error: str


EndpointOutcome = EndpointSuccess | EndpointFailure | EndpointNotFound


def _error_excerpt(value: str) -> str:
    return extract_remote_error("", value, 1)


def _endpoint_error(error: Exception) -> str:
    if isinstance(error, RpcCallError) and error.host_message is not None:
        return HostDiagnostic(error.host_message, _error_excerpt)
    if isinstance(error, HostMessageError):
        return HostDiagnostic(error.host_message, _error_excerpt)
    return _error_excerpt(str(error))


@dataclass(frozen=True)
class FleetOperation:
    method: str
    params: Params
    remote_argv: Sequence[str]
    decode: Callable[[Any], Any]
    local_seconds: float = 45.0
    remote_seconds: float = 45.0
    not_found: bool = False


class FleetExecutor:
    """Resolve once; every operation uses that immutable scope and capability."""

    def __init__(
        self,
        inventory: FleetConfig,
        *,
        inventory_only: bool = False,
        selector: str | None = None,
        census: bool = False,
        config: AppConfig | None = None,
    ) -> None:
        self.scope = FleetScope(
            inventory,
            caller=default_session_host(),
            inventory_only=inventory_only,
            selector=selector,
            census=census,
        )
        self.local_read = AggregateRead(self.scope, config or AppConfig.load())

    def execute(
        self,
        operation: FleetOperation,
        *,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> list[EndpointOutcome]:
        endpoints = self.scope.endpoints
        if not endpoints:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(endpoints)))) as pool:
            futures = [pool.submit(self._read, endpoint, operation) for endpoint in endpoints]
            outcomes = [future.result() for future in futures]
        invocation = current_invocation()
        if invocation.capability is self.local_read:
            invocation.local_succeeded = any(
                isinstance(endpoint, LocalEndpoint) and isinstance(outcome, EndpointSuccess)
                for endpoint, outcome in zip(endpoints, outcomes, strict=True)
            )
        return outcomes

    def _read(self, endpoint: FleetEndpoint, operation: FleetOperation) -> EndpointOutcome:
        try:
            if isinstance(endpoint, LocalEndpoint):
                value = self.local_read.call(
                    operation.method,
                    operation.params.model_dump(exclude_none=True),
                    seconds=operation.local_seconds,
                )
            else:
                assert isinstance(endpoint, SshEndpoint)
                result = fleet_transport.run_remote(
                    endpoint.host,
                    operation.remote_argv,
                    timeout=operation.remote_seconds,
                )
                if not result.ok:
                    error = result.error or "remote failed"
                    if operation.not_found and "not found" in error.lower():
                        return EndpointNotFound(endpoint.name, error)
                    return EndpointFailure(endpoint.name, extract_remote_error("", error, 1))
                value = json.loads(result.stdout) if result.stdout.strip() else []
                if isinstance(value, dict) and "data" in value:
                    value = value["data"]
            return EndpointSuccess(endpoint.name, operation.decode(value))
        except RpcCallError as error:
            if operation.not_found and error.code == APP_NOT_FOUND:
                return EndpointNotFound(endpoint.name, error.message)
            return EndpointFailure(endpoint.name, _endpoint_error(error))
        except (
            RpcConnectionError,
            RpcAdmissionBusyError,
            SchemaTooNewError,
            ValueError,
            OSError,
        ) as error:
            return EndpointFailure(endpoint.name, _endpoint_error(error))
