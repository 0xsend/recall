"""Fleet inventory config (REQ-FLEET-CFG-*).

Address book only — SSH auth is the operator's (keys / agent). Missing file
means zero hosts, not an error (fleet status can report empty).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


class FleetConfigError(ValueError):
    """Invalid fleet inventory (fail closed)."""


@dataclass(frozen=True)
class FleetHost:
    name: str
    ssh: str
    hostname: str | None = None

    @property
    def identity(self) -> str:
        return self.hostname if self.hostname is not None else self.name


@dataclass(frozen=True)
class FleetConfig:
    hosts: tuple[FleetHost, ...]
    path: Path

    def __post_init__(self) -> None:
        names: set[str] = set()
        identities: set[str] = set()
        for host in self.hosts:
            if host.name in names:
                raise FleetConfigError(f"duplicate fleet host name {host.name!r} in {self.path}")
            if host.identity in identities:
                raise FleetConfigError(
                    f"duplicate fleet host identity {host.identity!r} in {self.path}"
                )
            names.add(host.name)
            identities.add(host.identity)


def default_fleet_path() -> Path:
    override = os.environ.get("RECALL_FLEET_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "recall" / "fleet.toml"


def load_fleet_config(path: Path | None = None) -> FleetConfig:
    """Load fleet.toml. Missing path → empty hosts; invalid content → FleetConfigError."""
    config_path = path if path is not None else default_fleet_path()
    if not config_path.is_file():
        return FleetConfig(hosts=(), path=config_path)

    try:
        raw = config_path.read_text(encoding="utf-8")
        data = tomllib.loads(raw) if raw.strip() else {}
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise FleetConfigError(f"cannot read fleet config {config_path}: {err}") from err

    if not isinstance(data, dict):
        raise FleetConfigError(f"fleet config root must be a table: {config_path}")

    entries = data.get("host", [])
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise FleetConfigError(f"fleet config [[host]] must be an array of tables: {config_path}")

    hosts: list[FleetHost] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise FleetConfigError(f"host[{index}] must be a table in {config_path}")
        name = entry.get("name")
        ssh = entry.get("ssh")
        if not isinstance(name, str) or not name.strip():
            raise FleetConfigError(f"host[{index}] missing non-empty name in {config_path}")
        if not isinstance(ssh, str) or not ssh.strip():
            raise FleetConfigError(f"host[{index}] missing non-empty ssh in {config_path}")
        name = name.strip()
        ssh = ssh.strip()
        if ssh.startswith("-"):
            # ssh would parse the target as an option (e.g. -oProxyCommand=...).
            raise FleetConfigError(
                f"host {name!r} ssh target must not start with '-': {ssh!r} in {config_path}"
            )
        hostname = entry.get("hostname")
        if "hostname" in entry:
            if not isinstance(hostname, str):
                raise FleetConfigError(f"host {name!r} hostname must be a short hostname")
            hostname = hostname.strip()
            if (
                not hostname
                or hostname == "local"
                or "." in hostname
                or any(char.isspace() for char in hostname)
            ):
                raise FleetConfigError(
                    f"host {name!r} hostname must be nonempty, short, and not local"
                )
        host = FleetHost(name=name, ssh=ssh, hostname=hostname)
        hosts.append(host)

    return FleetConfig(hosts=tuple(hosts), path=config_path)


@dataclass(frozen=True)
class LocalEndpoint:
    name: str
    selectors: tuple[str, ...]


@dataclass(frozen=True)
class SshEndpoint:
    host: FleetHost

    @property
    def name(self) -> str:
        return self.host.name


FleetEndpoint = LocalEndpoint | SshEndpoint


@dataclass(frozen=True, init=False)
class FleetScope:
    """Validated membership; selectors cannot add endpoints after construction."""

    endpoints: tuple[FleetEndpoint, ...]
    selector: str | None

    def __init__(
        self,
        inventory: FleetConfig,
        *,
        caller: str,
        inventory_only: bool = False,
        selector: str | None = None,
        census: bool = False,
    ) -> None:
        if not inventory.hosts and (inventory_only or census):
            raise FleetConfigError(
                f"no fleet hosts configured ({inventory.path}); "
                "add [[host]] name/ssh entries to fleet.toml"
            )
        usable_identity = bool(caller and caller != "local")
        if not usable_identity and not inventory_only:
            raise FleetConfigError("calling hostname is unavailable; cannot resolve fleet identity")
        self_hosts = [
            host for host in inventory.hosts if usable_identity and host.identity == caller
        ]
        endpoints: list[FleetEndpoint] = []
        if not inventory_only or self_hosts:
            if any(host.name == caller and host.identity != caller for host in inventory.hosts):
                raise FleetConfigError(
                    f"fleet display name {caller!r} collides with calling endpoint"
                )
            endpoints.append(LocalEndpoint(caller, (caller, *(host.name for host in self_hosts))))
        endpoints.extend(
            SshEndpoint(host)
            for host in inventory.hosts
            if not usable_identity or host.identity != caller
        )
        if selector is not None:
            endpoints = [
                endpoint
                for endpoint in endpoints
                if (
                    selector in endpoint.selectors
                    if isinstance(endpoint, LocalEndpoint)
                    else selector == endpoint.name
                )
            ]
        object.__setattr__(self, "endpoints", tuple(endpoints))
        object.__setattr__(self, "selector", selector)
