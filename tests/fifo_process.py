"""Killable workers for scenarios that can regress to blocking filesystem IO."""

from __future__ import annotations

import multiprocessing
import os
import stat
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


def _run(sender: Connection, scenario: Callable[..., None], args: tuple[Any, ...]) -> None:
    try:
        scenario(*args)
    except BaseException:
        sender.send(traceback.format_exc())
    else:
        sender.send(None)
    finally:
        sender.close()


def run_fifo_scenario(scenario: Callable[..., None], root: Path, *args: Any) -> None:
    """Release owned FIFOs, then kill and reap any worker that exceeds its bound."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_run, args=(sender, scenario, (root, *args)))
    process.start()
    sender.close()
    try:
        completed = receiver.poll(10)
        failure = receiver.recv() if completed else "FIFO scenario exceeded ten seconds"
    finally:
        for path in root.rglob("*"):
            if stat.S_ISFIFO(path.lstat().st_mode):
                descriptor = os.open(path, os.O_RDWR | os.O_NONBLOCK)
                os.close(descriptor)
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        receiver.close()
        assert not process.is_alive(), "owned FIFO worker could not be reaped"
    assert completed, failure
    assert failure is None, failure
    assert process.exitcode == 0
