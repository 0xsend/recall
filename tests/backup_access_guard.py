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
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *roots: Path,
        following_aliases: tuple[Path, ...] = (),
    ) -> None:
        self.roots = tuple(str(root) for root in roots)
        self.following_aliases = tuple(str(path) for path in following_aliases)
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
            follows_alias = (
                name in {"stat", "open", "access", "listdir", "scandir", "resolve", "realpath"}
                and kwargs.get("follow_symlinks", True)
                and not (namespace is os and name == "open" and args and args[0] & os.O_NOFOLLOW)
                and any(
                    path == alias or path.startswith(alias + "/")
                    for alias in self.following_aliases
                )
            )
            if follows_alias or any(
                path == root or path.startswith(root + "/") for root in self.roots
            ):
                self.attempts.append((name, path))
                raise AssertionError(f"forbidden source effect: {name} {path}")
            result = original(value, *args, **kwargs)
            if namespace is os and name in {"open", "dup"}:
                self.descriptors[result] = path
            return result

        if namespace is os:
            for feature in ("supports_dir_fd", "supports_fd", "supports_follow_symlinks"):
                supported = getattr(os, feature)
                if original in supported:
                    patch.setattr(os, feature, {*supported, guarded})
        patch.setattr(namespace, name, guarded)
