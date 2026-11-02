"""Tool identity and the final outcome line."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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
Outcome = Literal["passed", "failed", "incomplete"]
EXIT_STATUS: dict[Outcome, int] = {"passed": 0, "failed": 1, "incomplete": 2}


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
    # Entries a failed job created but could not remove, with the reason.
    leftovers: list[dict[str, str]] = field(default_factory=list)

    def step(self, name: str, **fields: object) -> None:
        self.steps.append({"step": name, **fields})

    def emit(self) -> None:
        line = {
            "outcome": self.outcome,
            "job": self.job,
            "scope": self.scope,
            "steps": self.steps,
            "exit_codes": self.exit_codes,
            "leftovers": self.leftovers,
        }
        if self.error is not None:
            line["error"] = self.error
        sys.stdout.write(json.dumps(line, sort_keys=True) + "\n")
        sys.stdout.flush()
