"""The only way a bench job runs the build's recall CLI.

A `Command` can only hold an argument vector of one of the allowed shapes:
construction parses it against that grammar, so no caller can add an option
or shape a value into one, whether through a class method or directly.
`daemon stop|restart|install|uninstall` act on the account's real service
manager whatever the data directory is; the only stop is `--soft`, which
signals the daemon recorded in the run directory. Daemon start belongs to the
supervisor, never to a CLI call.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .fs import Refused
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


def _limit(text: str) -> str:
    # Canonical decimal only, so the product parser reads the same number.
    if not (text.isascii() and text.isdecimal() and str(int(text)) == text):
        raise Refused("recall", text, "limit must be a decimal integer")
    if not 0 < int(text) <= LIMIT_MAX:
        raise Refused("recall", text, f"limit must be within 1..{LIMIT_MAX}")
    return text


def _check_grammar(argv: tuple[str, ...]) -> None:
    """Accept exactly the invocations the class methods write."""
    if not isinstance(argv, tuple) or not all(isinstance(part, str) for part in argv):
        raise Refused("recall", argv, "argv must be a tuple of strings")
    match argv:
        case ("stats", "--json") | ("daemon", "status", "--json"):
            pass
        case ("daemon", "stop", "--soft", "--json"):
            pass
        case ("live", "--json", "--limit", limit):
            _limit(limit)
        case ("show", "--json", session):
            _value("session", session)
        case ("search", "--json", "--limit", limit, query):
            _limit(limit)
            _value("query", query)
        case _:
            raise Refused("recall", " ".join(argv), "not an allowed recall invocation")


@dataclass(frozen=True)
class Command:
    """One allowed recall invocation; any other argument vector cannot be constructed."""

    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_grammar(self.argv)

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
        return cls(("live", "--json", "--limit", str(limit)))

    @classmethod
    def show(cls, session: str) -> Command:
        return cls(("show", "--json", session))

    @classmethod
    def search(cls, query: str, limit: int) -> Command:
        return cls(("search", "--json", "--limit", str(limit), query))


def run_cli(
    build: Build, run: RunDir, command: Command, *, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    """Run one allowed command with the build's recall against `run` only.

    The run directory and the build are checked again immediately before the
    child starts, and nothing starts if either changed since it was created.
    """
    if not 0 < timeout_seconds <= CLI_TIMEOUT_SECONDS_MAX:
        raise ValueError(f"timeout must be within (0, {CLI_TIMEOUT_SECONDS_MAX}] seconds")
    # WAIVER(TOCTOU): the child starts in the held run directory, but execve takes
    # the entry point and its interpreter by name, and the child opens its RECALL_*,
    # TMPDIR and HF_HOME paths by name for its lifetime. Replacing or aliasing one of
    # those names after these checks redirects the child: a link or bind mount at
    # the name, or the run's path re-created as another directory. Operators must
    # not repurpose a live run or build; the timeout bounds how long a child waits,
    # not what its names reach. Permanent: closing it needs an OS sandbox (plan
    # section 8, option c).
    with run.held() as root:
        build.recheck()
        with root.as_working_directory():
            process = subprocess.Popen(
                [str(build.recall), *command.argv],
                env=child_environment(build, run),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
    with process:
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
