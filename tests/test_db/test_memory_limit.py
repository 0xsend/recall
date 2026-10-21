from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from recall.core.config import AppConfig
from recall.db.connection import connect, connect_readonly, resolve_memory_limit


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in tuple(os.environ):
        if name.startswith("RECALL_"):
            monkeypatch.delenv(name)
    path = tmp_path / "config.toml"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path))
    return path


@pytest.mark.parametrize(
    ("requested", "opt_out", "resolved", "setting_expected"),
    [
        (None, False, "2GB", "1.8 GiB"),
        ("512MB", False, "512MiB", "512.0 MiB"),
        ("2000000000B", False, "2000000000B", "1.8 GiB"),
        ("2000000001B", False, "2GB", "1.8 GiB"),
        ("2GB", False, "2GB", "1.8 GiB"),
        ("4GB", False, "2GB", "1.8 GiB"),
        ("4GB", True, "4GiB", "4.0 GiB"),
        ("512MB", True, "512MiB", "512.0 MiB"),
        (None, True, "2GB", "1.8 GiB"),
    ],
)
def test_memory_limit_at_managed_connection_boundary(
    config_path: Path, requested: str | None, opt_out: bool, resolved: str, setting_expected: str
) -> None:
    contents = "[duckdb]\n"
    if requested is not None:
        contents += f'memory_limit = "{requested}"\n'
    contents += f"memory_limit_exceeds_floor = {str(opt_out).lower()}\n"
    config_path.write_text(contents)
    config = AppConfig.load()
    assert resolve_memory_limit(config) == resolved
    with connect(config) as conn:
        actual = conn.execute(
            "SELECT value FROM duckdb_settings() WHERE name='memory_limit'"
        ).fetchone()
        assert actual is not None
        assert actual == (setting_expected,)
    with connect_readonly(config) as conn:
        assert conn.execute("SELECT current_setting('memory_limit')").fetchone() == actual


@pytest.mark.parametrize("opt_out,expected", [(False, "2GB"), (True, "4GiB")])
def test_environment_memory_limit_uses_same_clamp(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, opt_out: bool, expected: str
) -> None:
    config_path.write_text(
        f'[duckdb]\nmemory_limit = "512MB"\nmemory_limit_exceeds_floor = {str(opt_out).lower()}\n'
    )
    monkeypatch.setenv("RECALL_DUCKDB_MEMORY_LIMIT", "4GB")
    assert resolve_memory_limit(AppConfig.load()) == expected


@pytest.mark.parametrize("value", ["", "nonsense", "-1GB", "0B", "1e3MB", "1ib", "1IB"])
@pytest.mark.parametrize("opt_out", [False, True])
def test_invalid_memory_limit_remains_error(config_path: Path, value: str, opt_out: bool) -> None:
    config_path.write_text(
        f'[duckdb]\nmemory_limit = "{value}"\nmemory_limit_exceeds_floor = {str(opt_out).lower()}\n'
    )
    with pytest.raises(ValueError, match="memory_limit"):
        resolve_memory_limit(AppConfig.load())


@pytest.mark.parametrize("value", ['"false"', "1", "[]"])
def test_memory_opt_out_requires_boolean(config_path: Path, value: str) -> None:
    config_path.write_text(f"[duckdb]\nmemory_limit_exceeds_floor = {value}\n")
    with pytest.raises(ValueError, match="memory_limit_exceeds_floor must be a boolean"):
        AppConfig.load()


@pytest.mark.parametrize("opt_out", [False, True])
def test_daemon_status_reports_memory_resolution(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, opt_out: bool
) -> None:
    from recall.cli.app import app
    from typer.testing import CliRunner

    config_path.write_text(
        f'[duckdb]\nmemory_limit = "4GB"\nmemory_limit_exceeds_floor = {str(opt_out).lower()}\n'
    )
    monkeypatch.setattr("recall.services.daemon._read_crontab", lambda: "")
    runner = CliRunner()
    selected = (
        "duckdb_memory_limit_requested,duckdb_memory_limit_applied,duckdb_memory_limit_clamped"
    )
    result = runner.invoke(app, ["daemon", "status", "--json", "--fields", selected])
    assert result.exit_code == 0, result.output
    status = json.loads(result.stdout)
    assert status == {
        "duckdb_memory_limit_requested": "4GiB",
        "duckdb_memory_limit_applied": "4GiB" if opt_out else "2GB",
        "duckdb_memory_limit_clamped": not opt_out,
    }
    text = runner.invoke(app, ["daemon", "status", "--format", "text"])
    assert text.exit_code == 0, text.output
    assert "requested=4GiB" in text.stdout
    assert f"applied={status['duckdb_memory_limit_applied']}" in text.stdout
    assert f"clamped={str(not opt_out).lower()}" in text.stdout
