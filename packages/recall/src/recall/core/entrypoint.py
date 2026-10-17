"""Start recall again as a detached process, from the build that is running."""

from __future__ import annotations

import logging
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


@dataclass(frozen=True)
class RecallEntrypoint:
    path: Path
    source: EntrypointSource


@dataclass(frozen=True)
class DetachedProcess:
    pid: int
    log_path: Path


def running_recall_entrypoint() -> RecallEntrypoint:
    """The `recall` executable of the running build.

    A process recall starts of itself must run the same build: the first
    `recall` on PATH may be another install, which would then open this
    process's data dir with a different version. PATH is consulted only when
    the running build has no `recall` entrypoint on disk.
    """
    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.name == "recall" and argv0.is_file():
        return RecallEntrypoint(path=argv0.resolve(), source=EntrypointSource.ARGV0)
    if sys.executable:
        sibling = Path(sys.executable).with_name("recall")
        if sibling.is_file():
            return RecallEntrypoint(path=sibling, source=EntrypointSource.INTERPRETER)
    on_path = shutil.which("recall")
    if on_path is not None:
        return RecallEntrypoint(path=Path(on_path), source=EntrypointSource.PATH)
    raise RuntimeError("could not resolve the running recall CLI entrypoint")


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
    stdout_log = log_dir / "daemon.log"
    started = f"starting {shlex.join(command)} (entrypoint from {entrypoint.source})"
    logger.info("%s", started)
    with (
        open(stdout_log, "a") as stdout_f,
        open(log_dir / "daemon.err.log", "a") as stderr_f,
    ):
        stdout_f.write(f"recall: {started}\n")
        stdout_f.flush()
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout_f,
            stderr=stderr_f,
            start_new_session=True,
        )
    return DetachedProcess(pid=process.pid, log_path=stdout_log)
