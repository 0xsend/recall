"""Shutdown reaps codex subprocesses even after an upgrade swaps the package.

The documented upgrade flow is `uv tool install --force …`
followed by `recall daemon restart`, which replaces site-packages under the
still-running daemon. `_signal_shutdown` used to lazy-import the codex
terminator, so on that path the import raised `ModuleNotFoundError`, the failure
was swallowed by the shutdown guard, and the `codex exec` children were left
running with no parent.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import pytest
from recall.services.context_backends.codex_cli import CodexCliBackend
from recall.services.rpc_server import RpcServer

CODEX_CLI_MODULE = "recall.services.context_backends.codex_cli"


class _UninstalledModuleFinder(MetaPathFinder):
    """Refuse one module the way a replaced site-packages tree does."""

    def __init__(self, blocked: str) -> None:
        self._blocked = blocked

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname == self._blocked:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@contextmanager
def codex_cli_uninstalled() -> Iterator[None]:
    """Make `import recall.…codex_cli` fail for the duration of the block."""
    finder = _UninstalledModuleFinder(CODEX_CLI_MODULE)
    evicted = sys.modules.pop(CODEX_CLI_MODULE, None)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        if evicted is not None:
            sys.modules[CODEX_CLI_MODULE] = evicted


@pytest.fixture
def registered_codex_child() -> Iterator[subprocess.Popen[str]]:
    """A live process in its own group, registered the way the backend registers one."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    with CodexCliBackend._active_processes_lock:
        CodexCliBackend._active_processes.add(child)
    try:
        yield child
    finally:
        with CodexCliBackend._active_processes_lock:
            CodexCliBackend._active_processes.discard(child)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in tuple(os.environ):
        if key.startswith("RECALL_"):
            monkeypatch.delenv(key)


def test_shutdown_terminates_codex_after_the_package_is_replaced(
    registered_codex_child: subprocess.Popen[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = RpcServer()
    caplog.set_level(logging.INFO, logger="recall.rpc_server")

    with codex_cli_uninstalled():
        server._signal_shutdown()

    assert registered_codex_child.poll() is not None
    assert server._shutdown_event.is_set()
    assert "terminated 1 active codex subprocess(es)" in caplog.text
    assert "failed to terminate codex subprocesses" not in caplog.text
