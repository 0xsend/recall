"""The only way a bench job runs the build's recall CLI.

A `Command` can only hold an argument vector of one of the allowed shapes:
construction parses it against that grammar, so no caller can add an option
or shape a value into one, whether through a class method or directly.
`daemon stop|restart|install|uninstall` act on the account's real service
manager whatever the data directory is; the only stop is `--soft`, which
signals the daemon recorded in the run directory. Daemon start belongs to the
supervisor, never to a CLI call.

Every call runs under the job's supervisor through the build's guarded
launcher (`launch.py`), which judges the call again with the build's own
parser and keeps the CLI from starting a daemon or reaching a service
manager. There is no other way to run the build's CLI.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fs import Refused
from .isolation import Build, RunDir
from .supervisor import Completed, Deadline, Supervisor

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

    @property
    def name(self) -> str:
        return "-".join(part for part in self.argv[:2] if not part.startswith("-"))

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
    supervisor: Supervisor,
    build: Build,
    run: RunDir,
    command: Command,
    *,
    timeout_seconds: float,
    deadline: Deadline,
) -> Completed:
    """Run one allowed command with the build's recall against `run` only.

    The supervisor proves the run directory and the build again immediately
    before the child starts, and nothing starts if either changed.
    """
    if not 0 < timeout_seconds <= CLI_TIMEOUT_SECONDS_MAX:
        raise ValueError(f"timeout must be within (0, {CLI_TIMEOUT_SECONDS_MAX}] seconds")
    return supervisor.run(
        command.name,
        [*run.launcher_argv(build, "cli"), *command.argv],
        timeout_seconds=timeout_seconds,
        deadline=deadline,
    )
