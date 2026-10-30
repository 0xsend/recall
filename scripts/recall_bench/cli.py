"""The only way a bench job runs the build's recall CLI."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

from .isolation import Build, Refused, RunDir, child_environment

READ_COMMANDS = frozenset({"status", "stats", "live", "show", "search", "index"})
CLI_TIMEOUT_SECONDS_MAX = 3600


def check_command(args: Sequence[str]) -> None:
    """Refuse every recall command outside the allow-list.

    `daemon stop|restart|install|uninstall` act on the account's real service
    manager whatever the data directory is, so the only stop is `--soft`, which
    signals the daemon recorded in the run directory. Daemon start belongs to
    the supervisor, never to a CLI call.
    """
    command = list(args)
    if command and command[0] in READ_COMMANDS:
        return
    if command[:2] == ["daemon", "status"]:
        return
    if command[:2] == ["daemon", "stop"] and "--soft" in command[2:]:
        return
    raise Refused("recall", " ".join(command), "command is not on the bench allow-list")


def run_cli(
    build: Build, run: RunDir, args: Sequence[str], *, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    """Run one allowed command with the build's recall against `run` only."""
    check_command(args)
    if not 0 < timeout_seconds <= CLI_TIMEOUT_SECONDS_MAX:
        raise ValueError(f"timeout must be within (0, {CLI_TIMEOUT_SECONDS_MAX}] seconds")
    return subprocess.run(
        [str(build.recall), *args],
        env=child_environment(build, run),
        cwd=run.path,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
