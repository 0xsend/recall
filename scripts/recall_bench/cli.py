"""The only way a bench job runs the build's recall CLI.

A `Command` is built by one constructor per allowed invocation, and each
constructor writes the whole argument vector, so no caller can add an option
or shape a value into one. `daemon stop|restart|install|uninstall` act on the
account's real service manager whatever the data directory is; the only stop
is `--soft`, which signals the daemon recorded in the run directory. Daemon
start belongs to the supervisor, never to a CLI call.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from .fs import Refused, identity
from .isolation import Build, RunDir, child_environment

CLI_TIMEOUT_SECONDS_MAX = 3600
LIMIT_MAX = 10_000
VALUE_BYTES_MAX = 4096


def _value(name: str, value: str) -> str:
    # A leading dash would parse as an option; NUL cannot cross exec.
    if not value or value.startswith("-") or "\0" in value:
        raise Refused("recall", value, f"{name} must be non-empty and must not start with '-'")
    if len(value.encode()) > VALUE_BYTES_MAX:
        raise Refused("recall", value[:64], f"{name} is longer than {VALUE_BYTES_MAX} bytes")
    return value


def _limit(limit: int) -> str:
    if not 0 < limit <= LIMIT_MAX:
        raise Refused("recall", limit, f"limit must be within 1..{LIMIT_MAX}")
    return str(limit)


@dataclass(frozen=True)
class Command:
    """One allowed recall invocation; construct it only through the class methods."""

    argv: tuple[str, ...]

    @classmethod
    def stats(cls) -> Command:
        return cls(("stats", "--json"))

    @classmethod
    def daemon_status(cls) -> Command:
        return cls(("daemon", "status", "--json"))

    @classmethod
    def daemon_stop_soft(cls) -> Command:
        return cls(("daemon", "stop", "--soft", "--json"))

    @classmethod
    def live(cls, limit: int) -> Command:
        return cls(("live", "--json", "--limit", _limit(limit)))

    @classmethod
    def show(cls, session: str) -> Command:
        return cls(("show", "--json", _value("session", session)))

    @classmethod
    def search(cls, query: str, limit: int) -> Command:
        return cls(("search", "--json", "--limit", _limit(limit), _value("query", query)))


def run_cli(
    build: Build, run: RunDir, command: Command, *, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    """Run one allowed command with the build's recall against `run` only."""
    if not 0 < timeout_seconds <= CLI_TIMEOUT_SECONDS_MAX:
        raise ValueError(f"timeout must be within (0, {CLI_TIMEOUT_SECONDS_MAX}] seconds")
    # WAIVER(TOCTOU): exec takes a path, and macOS has no descriptor-relative exec,
    # so the entry point is re-checked by identity just before spawn rather than
    # held. Permanent: the window is the spawn itself.
    if identity(os.stat(build.recall, follow_symlinks=False)) != build.recall_identity:
        raise Refused("--build", build.recall, "the recall entry point changed since resolution")
    return subprocess.run(
        [str(build.recall), *command.argv],
        env=child_environment(build, run),
        cwd=run.path,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
