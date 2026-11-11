"""Inventory-only setup for transport/merge tests; public scope tests cover local RPC."""

from collections.abc import Iterable
from pathlib import Path

from recall.core.fleet import FleetConfig, FleetHost
from recall.services.fleet_endpoints import FleetExecutor


def inventory_executor(hosts: Iterable[FleetHost], selector: str | None = None) -> FleetExecutor:
    return FleetExecutor(
        FleetConfig(tuple(hosts), Path("fixture-fleet.toml")),
        inventory_only=True,
        selector=selector,
    )
