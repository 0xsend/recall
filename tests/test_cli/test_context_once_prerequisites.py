"""Standalone prerequisites win before any schema connection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

from tests.test_cli.test_context_availability import _config
from tests.test_cli.test_context_startup import _environment


@pytest.mark.parametrize("database", ["absent", "ordinary", "newer"])
def test_once_unavailable_prerequisite_precedes_every_database_connection(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, database: str
) -> None:
    from recall.db.schema import ensure_schema

    root = socket_dir
    if database != "absent":
        with duckdb.connect(str(root / "recall.duckdb")) as conn:
            ensure_schema(conn)
            if database == "newer":
                conn.execute("INSERT INTO schema_version(version) VALUES (99999)")
    (root / "config.toml").write_text(_config("llm-codex"))
    env = _environment(root, monkeypatch)
    spy = (
        "import duckdb\nfrom pathlib import Path\n"
        "original = duckdb.connect\n"
        "def connect(*args, **kwargs):\n"
        f"    Path({str(root / 'connected')!r}).touch()\n"
        "    return original(*args, **kwargs)\n"
        "duckdb.connect = connect\nfrom recall.cli import main\nmain()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", spy, "daemon", "--once", "--json"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    error = json.loads(result.stdout)["error"]
    assert error["code"] == "RUNTIME"
    assert "llm-codex" in error["message"] and "no daemon cycle ran" in error["message"]
    assert not (root / "connected").exists()
    assert not (root / "recall.sock").exists()


@pytest.mark.parametrize("structured", [False, True])
def test_once_available_prerequisite_preserves_newer_schema_error(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, structured: bool
) -> None:
    from recall.db.schema import ensure_schema

    root = socket_dir
    with duckdb.connect(str(root / "recall.duckdb")) as conn:
        ensure_schema(conn)
        conn.execute("INSERT INTO schema_version(version) VALUES (99999)")
    (root / "config.toml").write_text(
        _config("llm-codex").replace("missing-recall-context-cli", sys.executable)
    )
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "recall"),
            "daemon",
            "--once",
            *(["--json"] if structured else ["--format", "text"]),
        ],
        env=_environment(root, monkeypatch),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    message = result.stderr
    if structured:
        assert result.stderr == ""
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "RUNTIME"
        message = error["message"]
    assert "Database schema 99999 is newer than this build supports" in message
    assert "Automatic downgrade is not supported." in message
    assert "no daemon cycle ran" not in message
    assert not (root / "recall.sock").exists()
