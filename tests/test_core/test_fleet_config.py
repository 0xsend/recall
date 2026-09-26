"""Fleet inventory loader (REQ-FLEET-CFG-*)."""

from __future__ import annotations

from pathlib import Path

import pytest
from recall.core.fleet import FleetConfigError, load_fleet_config


def test_missing_fleet_file_returns_empty_hosts(tmp_path: Path) -> None:
    path = tmp_path / "missing.toml"
    config = load_fleet_config(path)
    assert config.hosts == ()
    assert config.path == path


def test_loads_hosts_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "fleet.toml"
    path.write_text(
        """
[[host]]
name = "devbox"
ssh = "devbox.example.ts.net"

[[host]]
name = "buildbox"
ssh = "buildbox"
""",
        encoding="utf-8",
    )
    config = load_fleet_config(path)
    assert [h.name for h in config.hosts] == ["devbox", "buildbox"]
    assert config.hosts[0].ssh == "devbox.example.ts.net"


def test_duplicate_names_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "fleet.toml"
    path.write_text(
        """
[[host]]
name = "dup"
ssh = "a.example"

[[host]]
name = "dup"
ssh = "b.example"
""",
        encoding="utf-8",
    )
    with pytest.raises(FleetConfigError, match="duplicate"):
        load_fleet_config(path)


def test_missing_name_or_ssh_fails(tmp_path: Path) -> None:
    path = tmp_path / "fleet.toml"
    path.write_text(
        """
[[host]]
name = "only-name"
""",
        encoding="utf-8",
    )
    with pytest.raises(FleetConfigError, match="ssh"):
        load_fleet_config(path)


def test_ssh_target_starting_with_dash_fails(tmp_path: Path) -> None:
    # ssh would parse such a target as an option (e.g. -oProxyCommand=...).
    path = tmp_path / "fleet.toml"
    path.write_text(
        """
[[host]]
name = "evil"
ssh = "-oProxyCommand=touch /tmp/pwned"
""",
        encoding="utf-8",
    )
    with pytest.raises(FleetConfigError, match="must not start with '-'"):
        load_fleet_config(path)
