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


@dataclass(frozen=True)
class FleetConfig:
    hosts: tuple[FleetHost, ...]
    path: Path


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
    seen: set[str] = set()
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
        if name in seen:
            raise FleetConfigError(f"duplicate fleet host name {name!r} in {config_path}")
        seen.add(name)
        hosts.append(FleetHost(name=name, ssh=ssh))

    return FleetConfig(hosts=tuple(hosts), path=config_path)
