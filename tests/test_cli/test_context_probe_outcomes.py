"""Prerequisite proof, bounded diagnostics, and caller-visible startup refusal."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

from tests.test_cli.test_context_availability import _config
from tests.test_cli.test_context_startup import _environment, _process


def _import_module(root: Path, body: str) -> None:
    modules = root / "modules"
    modules.mkdir()
    (modules / "mlx_lm.py").write_text(body)


@pytest.mark.parametrize("termination", ["exit-zero", "signal"])
def test_incomplete_prerequisite_probe_refuses_startup(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, termination: str
) -> None:
    root = socket_dir
    body = (
        "import sys; sys.exit(0)\n"
        if termination == "exit-zero"
        else "import os, signal; os.kill(os.getpid(), signal.SIGTERM)\n"
    )
    _import_module(root, body)
    (root / "config.toml").write_text(_config("llm-local"))
    with _process(root, _environment(root, monkeypatch)) as child:
        child.wait(timeout=8)
        assert child.returncode == 3
    reason = (root / "daemon-refused").read_text()
    if termination == "exit-zero":
        assert "without completing" in reason
    else:
        assert "SIGTERM" in reason and "-15" in reason
    assert not (root / "recall.duckdb").exists()


def test_prerequisite_stderr_capture_is_bounded(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.core.config import ContextConfig
    from recall.services.context_backends import ContextBackendUnavailableError
    from recall.services.context_backends.startup import require_context_prerequisites

    _import_module(
        socket_dir,
        "import os\n"
        "chunk = b'x' * 65536\n"
        "for _ in range(512): os.write(2, chunk)\n"
        "os.write(2, b'final import cause')\n"
        "raise RuntimeError('failed after large diagnostics')\n",
    )
    _environment(socket_dir, monkeypatch)
    tracemalloc.start()
    try:
        with pytest.raises(ContextBackendUnavailableError, match="final import cause"):
            asyncio.run(require_context_prerequisites(ContextConfig(mode="llm-local")))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # 32 MiB of child output must never accumulate in the owner's Python heap.
    assert peak < 2 * 1024 * 1024


@pytest.mark.parametrize(
    "arguments", [["--mode", "poll"], ["--once", "--format", "text"], ["--once", "--json"]]
)
def test_startup_refusal_has_one_remediation_paragraph(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    root = socket_dir
    (root / "config.toml").write_text(_config("llm-codex"))
    result = subprocess.run(
        [str(Path(sys.executable).parent / "recall"), "daemon", *arguments],
        env=_environment(root, monkeypatch),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == (1 if "--once" in arguments else 3)
    if "--json" in arguments:
        assert json.loads(result.stdout)["error"]["message"].count("To fix,") == 1
        assert result.stderr.count("To fix,") == 1
    else:
        assert result.stderr.count("To fix,") == 1
    assert not (root / "recall.duckdb").exists()


def test_prerequisite_timeout_is_inconclusive_in_status(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = socket_dir
    _import_module(root, "import time\nwhile True: time.sleep(1)\n")
    (root / "config.toml").write_text(_config("llm-local"))
    env = _environment(root, monkeypatch)
    with _process(root, env) as child:
        child.wait(timeout=8)
        assert child.returncode == 3
    result = subprocess.run(
        [str(Path(sys.executable).parent / "recall"), "daemon", "status", "--json"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    reason = json.loads(result.stdout)["startup_refusal"]
    assert "timed out after 5s" in reason
    assert "availability is unconfirmed" in reason
    assert "does not prove a missing dependency" in reason
    assert "Check host load" in reason and "explicitly start" in reason
    assert "install" not in reason
    assert reason == (root / "daemon-refused").read_text().strip()
