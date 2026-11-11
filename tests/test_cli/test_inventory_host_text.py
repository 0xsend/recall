"""Inventory host values keep raw data and one-line terminal diagnostics."""

from __future__ import annotations

import json

import pytest
from recall.cli.app import app
from typer.testing import CliRunner

CASES = [
    ("edge", "edge"),
    ("edge\nforged", '"edge\\nforged"'),
    ("edge\rforged", '"edge\\rforged"'),
    ("edge\x1b[31m", '"edge\\u001b[31m"'),
    ("edge host=trusted", '"edge host=trusted"'),
    ("équipe", '"\\u00e9quipe"'),
]


@pytest.mark.parametrize("origin,encoded", CASES)
@pytest.mark.parametrize("healthy", [False, True])
def test_inventory_status_host_values(tmp_path, monkeypatch, origin, encoded, healthy):
    inventory = tmp_path / "fleet.toml"
    inventory.write_text(
        "[[host]]\nname=" + json.dumps(origin) + "\nssh=" + json.dumps(origin) + "\n"
    )
    targets = []

    def remote(argv, **kwargs):
        targets.append(argv[argv.index("ControlPath=none") + 1])
        from subprocess import CompletedProcess

        if not healthy:
            return CompletedProcess(argv, 255, "", "refused")
        output = (
            "0.46.1\n"
            if "--version" in argv
            else "daemon_version: 0.46.1\nbinary_version: 0.46.1\nversion_drift: false\n"
        )
        return CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr("recall.services.fleet_transport.subprocess.run", remote)
    args = ["fleet", "status", "--config", str(inventory)]
    text = CliRunner().invoke(app, [*args, "--format", "text"])
    structured = CliRunner().invoke(app, [*args, "--json"])
    assert text.exit_code == structured.exit_code == 0
    assert text.stderr_bytes == structured.stderr_bytes == b""
    suffix = "\tok\tbinary=0.46.1\tdaemon=0.46.1\tok\n" if healthy else "\tFAIL\trefused\n"
    assert text.stdout_bytes == (encoded + "\t" + encoded + suffix).encode()
    assert all(target == origin for target in targets)
    row = json.loads(structured.stdout)[0]
    assert row["name"] == row["ssh"] == origin


@pytest.mark.parametrize(
    "kind,origin,encoded",
    [
        (kind, origin, encoded)
        for kind in ("duplicate", "identity", "ssh", "hostname", "collision")
        for origin, encoded in CASES
        if kind != "identity" or origin in ("edge", "équipe")
    ],
)
def test_inventory_host_diagnostics(tmp_path, monkeypatch, origin, encoded, kind):
    inventory = tmp_path / "fleet.toml"
    name = json.dumps(origin)
    text_host = repr(origin) if origin == "edge" else encoded
    if kind == "duplicate":
        inventory.write_text(
            f'[[host]]\nname={name}\nssh="one"\n[[host]]\nname={name}\nssh="two"\n'
        )
        raw = f"duplicate fleet host name {origin!r} in {inventory}"
        terminal = f"duplicate fleet host name {text_host} in {inventory}"
    elif kind == "identity":
        # Non-ASCII is a valid short identity; control characters are not.
        inventory.write_text(
            f'[[host]]\nname="one"\nssh="one"\nhostname={name}\n[[host]]\nname="two"\nssh="two"\nhostname={name}\n'
        )
        raw = f"duplicate fleet host identity {origin!r} in {inventory}"
        terminal = f"duplicate fleet host identity {text_host} in {inventory}"
    elif kind == "ssh":
        inventory.write_text(f'[[host]]\nname={name}\nssh="-option"\n')
        raw = f"host {origin!r} ssh target must not start with '-': '-option' in {inventory}"
        terminal = f"host {text_host} ssh target must not start with '-': '-option' in {inventory}"
    elif kind == "hostname":
        inventory.write_text(f'[[host]]\nname={name}\nssh="one"\nhostname=42\n')
        raw = f"host {origin!r} hostname must be a short hostname"
        terminal = f"host {text_host} hostname must be a short hostname"
    else:
        monkeypatch.setattr("recall.services.fleet_endpoints.default_session_host", lambda: origin)
        inventory.write_text(f'[[host]]\nname={name}\nssh="one"\nhostname="different"\n')
        raw = f"fleet display name {origin!r} collides with calling endpoint"
        terminal = f"fleet display name {text_host} collides with calling endpoint"
    args = ["list", "--fleet", "--fleet-config", str(inventory)]
    text = CliRunner().invoke(app, [*args, "--format", "text"])
    structured = CliRunner().invoke(app, [*args, "--json"])
    assert text.exit_code == structured.exit_code == 2
    assert text.stdout_bytes == b""
    assert text.stderr_bytes == ("error: " + terminal + "\n").encode()
    assert structured.stderr_bytes == b""
    assert json.loads(structured.stdout) == {"error": {"code": "VALIDATION", "message": raw}}
