"""Disposable native child for recovery durability boundaries."""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import sys
from unittest.mock import patch

import duckdb
from recall.core.config import AppConfig
from recall.db.connection import connect_owned
from recall.db.lease import owned_connection
from recall.services.recovery_admission import file_jobs
from recall.services.rpc_server import RpcServer


def checkpoint(stage: str, selected: str, fault: str) -> None:
    if stage != selected:
        return
    print(f"checkpoint:{stage}", flush=True)
    if fault == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
    if fault == "error":
        raise OSError(errno.EIO, f"injected {stage}")
    assert fault == "fatal"
    raise duckdb.FatalException(f"Failed to delete all rows from index at {stage}")


def main() -> None:
    operation, selected, fault = sys.argv[1:]
    config = AppConfig.load()
    if operation == "recreate":
        server = RpcServer(config)
        jobs = server._jobs
    else:
        jobs, admission = file_jobs(config)
    call = jobs.call_db

    def intercepted(stage, work):
        result = call(stage, work)
        checkpoint(stage, selected, fault)
        return result

    with patch.object(jobs, "call_db", intercepted):
        if operation == "recreate":
            try:
                server._reset_database(config)
            finally:
                asyncio.run(server.stop())
        else:
            with owned_connection(
                jobs,
                "fence_child",
                lambda: connect_owned(config, lifetime=jobs, admission=admission.normal),
            ):
                pass
    raise AssertionError("selected durability boundary was not reached")


if __name__ == "__main__":
    main()
