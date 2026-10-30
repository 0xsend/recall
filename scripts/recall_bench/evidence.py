"""Exclusive file creation, tool identity and the final outcome line."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TOOL_ROOT = Path(__file__).resolve().parents[2]
# Every file that decides what the tool does, so edited harness code cannot
# report the same identity as the commit it started from.
TOOL_PATHS = (
    "scripts/bench.py",
    "scripts/recall_bench",
    "scripts/with_bench_lock.py",
    "pyproject.toml",
    "uv.lock",
)
GIT_TIMEOUT_SECONDS = 30
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
NEW_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW

Outcome = Literal["passed", "failed", "incomplete"]
EXIT_STATUS: dict[Outcome, int] = {"passed": 0, "failed": 1, "incomplete": 2}


def check_name(name: str) -> str:
    """Accept one path component; anything else could leave the directory."""
    if name in {"", ".", ".."} or "/" in name or "\0" in name:
        raise ValueError(f"not a single path component: {name!r}")
    return name


class Directory:
    """A directory this run created, written only through exclusive creation.

    Files and subdirectories are created relative to the held descriptor with
    `O_EXCL | O_NOFOLLOW`, so nothing already present is overwritten or
    followed.
    """

    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self._descriptor = descriptor

    @classmethod
    def create(cls, path: Path) -> Directory:
        os.mkdir(path, 0o755)
        return cls(path, os.open(path, DIRECTORY_FLAGS))

    def subdirectory(self, name: str) -> Directory:
        os.mkdir(check_name(name), 0o755, dir_fd=self._descriptor)
        return Directory(self.path / name, os.open(name, DIRECTORY_FLAGS, dir_fd=self._descriptor))

    def write_new(self, name: str, data: bytes) -> Path:
        descriptor = os.open(check_name(name), NEW_FILE_FLAGS, 0o644, dir_fd=self._descriptor)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return self.path / name

    def publish(self, name: str, data: bytes) -> Path:
        """Write a file whole or not at all: readers never see a partial one."""
        partial = f".{check_name(name)}.partial"
        self.write_new(partial, data)
        os.rename(partial, name, src_dir_fd=self._descriptor, dst_dir_fd=self._descriptor)
        return self.path / name

    def is_regular(self, name: str) -> bool:
        mode = os.stat(check_name(name), dir_fd=self._descriptor, follow_symlinks=False).st_mode
        return stat.S_ISREG(mode)

    def close(self) -> None:
        os.close(self._descriptor)

    @contextmanager
    def held(self) -> Iterator[Directory]:
        try:
            yield self
        finally:
            self.close()


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=TOOL_ROOT,
        capture_output=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=False,
    )


def tool_identity() -> dict[str, object]:
    """The tool checkout's commit plus a hash of every uncommitted tool change."""
    head = _git("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"head": None, "error": head.stderr.decode(errors="replace").strip()}
    diff = _git("diff", "HEAD", "--", *TOOL_PATHS)
    untracked = _git("ls-files", "-z", "--others", "--exclude-standard", "--", *TOOL_PATHS)
    if diff.returncode != 0 or untracked.returncode != 0:
        return {"head": head.stdout.decode().strip(), "error": "git diff failed"}
    digest = hashlib.sha256(diff.stdout)
    names = sorted(name for name in untracked.stdout.decode().split("\0") if name)
    for name in names:
        digest.update(name.encode())
        digest.update((TOOL_ROOT / name).read_bytes())
    return {
        "head": head.stdout.decode().strip(),
        "dirty": bool(diff.stdout or names),
        "dirty_hash": digest.hexdigest(),
    }


@dataclass
class Result:
    """The one JSON object every job prints as its last stdout line."""

    job: str
    outcome: Outcome = "incomplete"
    scope: dict[str, object] = field(default_factory=dict)
    steps: list[dict[str, object]] = field(default_factory=list)
    exit_codes: dict[str, int] = field(default_factory=dict)
    error: dict[str, object] | None = None

    def step(self, name: str, **fields: object) -> None:
        self.steps.append({"step": name, **fields})

    def emit(self) -> None:
        line = {
            "outcome": self.outcome,
            "job": self.job,
            "scope": self.scope,
            "steps": self.steps,
            "exit_codes": self.exit_codes,
        }
        if self.error is not None:
            line["error"] = self.error
        sys.stdout.write(json.dumps(line, sort_keys=True) + "\n")
        sys.stdout.flush()
