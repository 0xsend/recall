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
from recall.db.connection import advisory_lock, connect_owned
from recall.db.lease import owned_connection
from recall.db.ownership import offline_ownership
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


def repair(config: AppConfig, selected: str, fault: str) -> None:
    from recall.services.repair_events import read_events
    from recall.services.repair_protocol import RepairProtocol

    jobs, admission = file_jobs(config)
    protocol = RepairProtocol(
        config.data_dir, barrier=lambda stage: checkpoint(stage, selected, fault)
    )
    with offline_ownership(config.db_path), advisory_lock(config.lock_path):
        with owned_connection(
            jobs,
            "protocol_source",
            lambda: connect_owned(config, lifetime=jobs, admission=admission.normal),
        ) as lease:
            protocol.prepare(lease, read_events(config.data_dir))
            arm = protocol.arm(lease)
            checkpoint("after_arm_commit", selected, fault)
            rebuilt = protocol.rebuild(lease, arm)
        checkpoint("after_owned_close", selected, fault)
        with owned_connection(
            jobs, "protocol_fresh", lambda: admission.reopen(protocol, rebuilt)
        ) as lease:
            protocol.verify(lease, rebuilt)
            protocol.retire(lease)


def main() -> None:
    operation, selected, fault = sys.argv[1:]
    config = AppConfig.load()
    if operation == "startup":
        from recall.services.self_repair import DaemonStartupRefused, run_startup_self_repair

        try:
            outcome = run_startup_self_repair(config)
        except DaemonStartupRefused as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(3) from error
        print(outcome)
        return
    if operation == "repair":
        repair(config, selected, fault)
        raise AssertionError("selected durability boundary was not reached")
    if operation in {"manual_rebuild", "manual_compact"}:
        from recall.services.manual_repair import ManualRepair
        from recall.services.self_repair import DaemonStartupRefused

        manual = ManualRepair(config, barrier=lambda stage: checkpoint(stage, selected, fault))
        try:
            result = manual.compact() if operation == "manual_compact" else manual.execute()
        except DaemonStartupRefused as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(3) from error
        print(result)
        if selected != "none":
            raise AssertionError("selected manual durability boundary was not reached")
        return
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
