"""A process recall starts of itself runs the invoked build, not the first `recall` on PATH."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import recall.core.rpc_client as rpc_client
from recall.cli.app import app
from recall.core.config import AppConfig
from recall.core.rpc_client import RpcClient, RpcConnectionError
from typer.testing import CliRunner

RUN_DEADLINE_SECONDS = 10.0


def _recorder(directory: Path, extra: str = "") -> Path:
    """A `recall` executable that records the arguments it ran with in `<directory>/ran`."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "recall"
    script.write_text(f'#!/bin/sh\n{extra}\nprintf \'%s\\n\' "$*" >> "{directory}/ran"\n')
    script.chmod(0o755)
    return script


def _wait_for_run(recorder: Path) -> str:
    ran = recorder.parent / "ran"
    deadline = time.monotonic() + RUN_DEADLINE_SECONDS
    while not ran.exists():
        assert time.monotonic() < deadline, f"{recorder} never ran"
        time.sleep(0.05)
    return ran.read_text()


@pytest.fixture
def decoy_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Another install's `recall`, first and only on PATH."""
    decoy = _recorder(tmp_path / "other-install" / "bin")
    monkeypatch.setenv("PATH", str(decoy.parent))
    return decoy


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "config.toml"))
    return AppConfig.load()


def _auto_fork(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    # The recorders never open a socket, so the wait for one ends in an error.
    monkeypatch.setattr(rpc_client, "AUTO_FORK_MAX_WAIT", 0.2)
    with pytest.raises(RpcConnectionError, match="did not start"):
        RpcClient(config=config).connect()


def test_auto_fork_starts_the_interpreters_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: AppConfig, decoy_on_path: Path
) -> None:
    entrypoint = _recorder(tmp_path / "invoked-env" / "bin")
    monkeypatch.setattr(sys, "executable", str(entrypoint.parent / "python"))

    _auto_fork(config, monkeypatch)

    assert _wait_for_run(entrypoint) == "daemon start --background\n"
    assert not (decoy_on_path.parent / "ran").exists()
    log = (config.data_dir / "logs" / "daemon.log").read_text()
    assert f"starting {entrypoint} daemon start --background" in log
    assert "entrypoint from beside sys.executable" in log


def test_auto_fork_prefers_the_invoked_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: AppConfig, decoy_on_path: Path
) -> None:
    invoked = _recorder(tmp_path / "wrapper")
    interpreter_entrypoint = _recorder(tmp_path / "invoked-env" / "bin")
    monkeypatch.setattr(sys, "argv", [str(invoked), "stats"])
    monkeypatch.setattr(sys, "executable", str(interpreter_entrypoint.parent / "python"))

    _auto_fork(config, monkeypatch)

    assert _wait_for_run(invoked) == "daemon start --background\n"
    assert not (interpreter_entrypoint.parent / "ran").exists()
    assert not (decoy_on_path.parent / "ran").exists()


def test_auto_fork_falls_back_to_path_without_a_running_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: AppConfig, decoy_on_path: Path
) -> None:
    monkeypatch.setattr(sys, "executable", str(tmp_path / "no-entrypoint" / "python"))

    _auto_fork(config, monkeypatch)

    assert _wait_for_run(decoy_on_path) == "daemon start --background\n"
    assert "entrypoint from PATH" in (config.data_dir / "logs" / "daemon.log").read_text()


def test_background_start_runs_the_interpreters_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: AppConfig, decoy_on_path: Path
) -> None:
    # Stands in for a daemon: it writes the PID file `daemon start` waits for.
    entrypoint = _recorder(
        tmp_path / "invoked-env" / "bin", extra=f'echo $$ > "{config.data_dir}/recall.pid"'
    )
    monkeypatch.setattr(sys, "executable", str(entrypoint.parent / "python"))

    result = CliRunner().invoke(app, ["daemon", "start", "--background", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["started"] is True
    assert _wait_for_run(entrypoint) == "daemon\n"
    assert not (decoy_on_path.parent / "ran").exists()


def test_auto_fork_skips_an_invoked_script_that_cannot_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: AppConfig, decoy_on_path: Path
) -> None:
    unrunnable = _recorder(tmp_path / "copied")
    unrunnable.chmod(0o644)
    entrypoint = _recorder(tmp_path / "invoked-env" / "bin")
    monkeypatch.setattr(sys, "argv", [str(unrunnable), "stats"])
    monkeypatch.setattr(sys, "executable", str(entrypoint.parent / "python"))

    _auto_fork(config, monkeypatch)

    assert _wait_for_run(entrypoint) == "daemon start --background\n"
    assert not (decoy_on_path.parent / "ran").exists()


def test_failed_auto_fork_names_the_command_and_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: AppConfig
) -> None:
    failing = _recorder(tmp_path / "invoked-env" / "bin", extra="exit 3")
    monkeypatch.setattr(sys, "executable", str(failing.parent / "python"))
    monkeypatch.setattr(rpc_client, "AUTO_FORK_MAX_WAIT", RUN_DEADLINE_SECONDS)

    started_at = time.monotonic()
    with pytest.raises(RpcConnectionError) as raised:
        RpcClient(config=config).connect()

    assert time.monotonic() - started_at < RUN_DEADLINE_SECONDS
    message = raised.value.message
    assert f"`{failing} daemon start --background` exited with status 3" in message
    assert str(config.data_dir / "logs" / "daemon.err.log") in message


def test_auto_fork_runs_a_symlink_named_recall_as_invoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: AppConfig, decoy_on_path: Path
) -> None:
    target = _recorder(tmp_path / "build").rename(tmp_path / "build" / "recall-build")
    invoked = tmp_path / "bin" / "recall"
    invoked.parent.mkdir()
    invoked.symlink_to(target)
    monkeypatch.setattr(sys, "argv", [str(invoked), "stats"])

    _auto_fork(config, monkeypatch)

    assert _wait_for_run(target) == "daemon start --background\n"
    log = (config.data_dir / "logs" / "daemon.log").read_text()
    assert f"starting {invoked} daemon start --background" in log


def test_auto_fork_names_an_entrypoint_whose_interpreter_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: AppConfig
) -> None:
    broken = tmp_path / "bin" / "recall"
    broken.parent.mkdir()
    broken.write_text("#!/nonexistent/python\n")
    broken.chmod(0o755)
    monkeypatch.setattr(sys, "argv", [str(broken), "stats"])

    with pytest.raises(RpcConnectionError) as raised:
        RpcClient(config=config).connect()

    assert f"could not run `{broken} daemon start --background`" in raised.value.message
