"""Reject forbidden fixture effects, including descriptor-relative access.

This guards synthetic source installations only. It does not replace local
filesystem behavior or claim coverage of native-library syscalls.
"""

from __future__ import annotations

import builtins
import io
import os
import posixpath
from pathlib import Path
from typing import Any

import pytest


class ForeignAccessGuard:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, *roots: Path) -> None:
        self.roots = tuple(str(root) for root in roots)
        self.attempts: list[tuple[str, str]] = []
        self.descriptors: dict[int, str] = {}
        # Preserve feature detection when wrapping supported system calls.
        original_open, original_stat = os.open, os.stat
        for namespace, names in (
            (os, ("stat", "lstat", "open", "readlink", "access", "listdir", "scandir")),
            (builtins, ("open",)),
            (io, ("open",)),
            (os.path, ("realpath",)),
        ):
            for name in names:
                self._wrap(monkeypatch, namespace, name)
        monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, os.open, os.stat})
        monkeypatch.setattr(os, "supports_follow_symlinks", {*os.supports_follow_symlinks, os.stat})
        assert original_open is not os.open and original_stat is not os.stat
        for name in ("read", "pread", "fstat", "fdopen", "dup"):
            if hasattr(os, name):
                self._wrap(monkeypatch, os, name)
        self._wrap(monkeypatch, Path, "resolve")

    def _operand(self, value: Any, directory: int | None = None) -> str:
        if isinstance(value, int):
            return self.descriptors.get(value, "")
        path = os.fsdecode(value)
        if not path.startswith("/"):
            base = self.descriptors.get(directory, "") if directory is not None else os.getcwd()
            path = posixpath.join(base, path)
        return posixpath.normpath(path)

    def _wrap(self, patch: pytest.MonkeyPatch, namespace: Any, name: str) -> None:
        original = getattr(namespace, name)

        def guarded(value, *args, **kwargs):
            path = self._operand(value, kwargs.get("dir_fd"))
            if any(path == root or path.startswith(root + "/") for root in self.roots):
                self.attempts.append((name, path))
                raise AssertionError(f"forbidden source effect: {name} {path}")
            result = original(value, *args, **kwargs)
            if namespace is os and name in {"open", "dup"}:
                self.descriptors[result] = path
            return result

        patch.setattr(namespace, name, guarded)
