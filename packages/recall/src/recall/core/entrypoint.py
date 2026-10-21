"""Start recall again as a detached process, from the build that is running."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from recall.core.config import create_private_dir

logger = logging.getLogger("recall.entrypoint")


class EntrypointSource(StrEnum):
    ARGV0 = "sys.argv[0]"
    INTERPRETER = "beside sys.executable"
    PATH = "PATH"


class DetachedStartError(Exception):
    """The entrypoint could not be executed."""


@dataclass(frozen=True)
class RecallEntrypoint:
    path: Path
    source: EntrypointSource


@dataclass(frozen=True)
class DetachedProcess:
    process: subprocess.Popen[bytes]
    command: str
    log_dir: Path

    @property
    def stderr_log(self) -> Path:
        return self.log_dir / "daemon.err.log"


def running_recall_entrypoint() -> RecallEntrypoint:
    """The `recall` executable of the running build.

    A process recall starts of itself must run the same build: the first
    `recall` on PATH may be another install, which would then open this
    process's data dir with a different version. PATH is consulted only when
    the running build has no `recall` entrypoint on disk.
    """
    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.name == "recall" and _is_executable_file(argv0):
        # Absolute, not resolved: a symlink named `recall` is what was invoked.
        return RecallEntrypoint(path=argv0.absolute(), source=EntrypointSource.ARGV0)
    if sys.executable:
        sibling = Path(sys.executable).with_name("recall")
        if _is_executable_file(sibling):
            return RecallEntrypoint(path=sibling, source=EntrypointSource.INTERPRETER)
    on_path = shutil.which("recall")
    if on_path is not None:
        return RecallEntrypoint(path=Path(on_path), source=EntrypointSource.PATH)
    raise RuntimeError("could not resolve the running recall CLI entrypoint")


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def spawn_detached_recall(data_dir: Path, args: Sequence[str]) -> DetachedProcess:
    """Run `recall <args>` from the running build in its own session.

    Output goes to `<data_dir>/logs/daemon.log` and `daemon.err.log`
    (REQ-RPC-016). The command and where its executable came from are logged
    and written to `daemon.log` first, so a daemon that never appears can be
    traced to the binary that was started.
    """
    entrypoint = running_recall_entrypoint()
    command = [str(entrypoint.path), *args]
    create_private_dir(data_dir)
    log_dir = data_dir / "logs"
    create_private_dir(log_dir)
    started = f"starting {shlex.join(command)} (entrypoint from {entrypoint.source})"
    logger.info("%s", started)
    with (
        open(log_dir / "daemon.log", "a") as stdout_f,
        open(log_dir / "daemon.err.log", "a") as stderr_f,
    ):
        stdout_f.write(f"recall: {started}\n")
        stdout_f.flush()
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_f,
                stderr=stderr_f,
                start_new_session=True,
            )
        except OSError as err:
            # A missing shebang interpreter surfaces as ENOENT naming the script.
            raise DetachedStartError(f"could not run `{shlex.join(command)}`: {err}") from err
    return DetachedProcess(process=process, command=shlex.join(command), log_dir=log_dir)
