"""Tool identity and the final outcome line."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .fs import Cancelled, DeadlineExpired, Refused

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
Outcome = Literal["passed", "failed", "incomplete"]
EXIT_STATUS: dict[Outcome, int] = {"passed": 0, "failed": 1, "incomplete": 2}


# What decides the behaviour of the build under test.
BUILD_PATHS = ("packages/recall/src", "packages/recall/pyproject.toml", "pyproject.toml", "uv.lock")


def _git(*args: str, root: Path = TOOL_ROOT) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=False,
    )


def tool_identity() -> dict[str, object]:
    """The tool checkout's commit plus a hash of every uncommitted tool change."""
    return checkout_identity(TOOL_ROOT, TOOL_PATHS)


def checkout_identity(root: Path, paths: tuple[str, ...]) -> dict[str, object]:
    """A checkout's commit plus a hash of every uncommitted change under `paths`."""
    head = _git("rev-parse", "HEAD", root=root)
    if head.returncode != 0:
        return {"head": None, "error": head.stderr.decode(errors="replace").strip()}
    diff = _git("diff", "HEAD", "--", *paths, root=root)
    untracked = _git("ls-files", "-z", "--others", "--exclude-standard", "--", *paths, root=root)
    if diff.returncode != 0 or untracked.returncode != 0:
        return {"head": head.stdout.decode().strip(), "error": "git diff failed"}
    digest = hashlib.sha256(diff.stdout)
    names = sorted(name for name in untracked.stdout.decode().split("\0") if name)
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return {
        "head": head.stdout.decode().strip(),
        "dirty": bool(diff.stdout or names),
        "dirty_hash": digest.hexdigest(),
    }


class JobEnded(Exception):
    """A job stopped short of `passed`; `kind` and `detail` go into the outcome line."""

    outcome: Outcome = "failed"

    def __init__(self, kind: str, message: str, **detail: object) -> None:
        super().__init__(message)
        self.kind = kind
        self.detail = detail

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "message": str(self), **self.detail}


class JobFailed(JobEnded):
    outcome: Outcome = "failed"


class JobIncomplete(JobEnded):
    """The work did not finish: a deadline, a hold timeout, or a missing seam."""

    outcome: Outcome = "incomplete"


@dataclass
class Result:
    """The one JSON object every job prints as its last stdout line."""

    job: str
    outcome: Outcome = "incomplete"
    scope: dict[str, object] = field(default_factory=dict)
    steps: list[dict[str, object]] = field(default_factory=list)
    exit_codes: dict[str, int] = field(default_factory=dict)
    error: dict[str, object] | None = None
    # Entries a failed job created but could not remove, with the reason.
    leftovers: list[dict[str, str]] = field(default_factory=list)

    def step(self, name: str, **fields: object) -> None:
        self.steps.append({"step": name, **fields})

    def end(self, error: BaseException | None) -> int:
        """Set the outcome from how the job ended; return the exit status."""
        if error is None:
            self.outcome = "passed"
            return EXIT_STATUS["passed"]
        if isinstance(error, Cancelled):
            self.outcome = "incomplete"
            self.error = {"kind": "cancelled", "signal": error.signum}
            return 128 + error.signum
        if isinstance(error, DeadlineExpired):
            self.outcome = "incomplete"
            self.error = {"kind": "deadline", "message": str(error)}
        elif isinstance(error, JobEnded):
            self.outcome = error.outcome
            self.error = error.describe()
        elif isinstance(error, Refused):
            self.outcome = "failed"
            self.error = error.describe()
        elif isinstance(error, OSError):
            self.outcome = "failed"
            self.error = {"kind": type(error).__name__, "message": str(error)}
        else:
            raise error
        return EXIT_STATUS[self.outcome]

    def line(self) -> dict[str, object]:
        line: dict[str, object] = {
            "outcome": self.outcome,
            "job": self.job,
            "scope": self.scope,
            "steps": self.steps,
            "exit_codes": self.exit_codes,
            "leftovers": self.leftovers,
        }
        if self.error is not None:
            line["error"] = self.error
        return line

    def to_json(self) -> bytes:
        return (json.dumps(self.line(), sort_keys=True, default=str) + "\n").encode()

    def emit(self) -> None:
        sys.stdout.write(self.to_json().decode())
        sys.stdout.flush()
