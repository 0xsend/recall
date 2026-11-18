"""Disposable native child for recovery durability boundaries."""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import sys
from collections.abc import Iterator
from contextlib import contextmanager
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


def repair(config: AppConfig, selected: str, fault: str, *, publish_progress: bool) -> None:
    from recall.services.repair_events import read_events
    from recall.services.repair_progress import NO_PROGRESS, ProgressPublisher
    from recall.services.repair_protocol import RepairProtocol

    jobs, admission = file_jobs(config)
    # The daemon publishes while it owns the database; so does this child.
    protocol = RepairProtocol(
        config.data_dir,
        barrier=lambda stage: checkpoint(stage, selected, fault),
        progress=ProgressPublisher(config.data_dir) if publish_progress else NO_PROGRESS,
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


class _Released(Exception):
    """The parent closed stdin: the held owner may finish."""


_held = False


def hold_until_released() -> None:
    """Report that an owner holds the database once, then wait for the parent to close stdin."""
    global _held
    if _held:
        return
    _held = True
    print("held", flush=True)
    sys.stdin.read()


@contextmanager
def _first_open_fails_replaying_the_wal() -> Iterator[None]:
    """The manual repair's first open fails like a damaged WAL, not like a lock conflict."""
    from recall.services import manual_repair

    normal_open = manual_repair.connect_owned
    failed: list[bool] = []

    def connect(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise duckdb.IOException("IO Error: Failure while replaying WAL file")
        return normal_open(*args, **kwargs)

    with patch.object(manual_repair, "connect_owned", connect):
        yield
    assert failed, "the manual repair never opened the database"


@contextmanager
def _admission_refuses_open(number: int, reason: str) -> Iterator[None]:
    """Admission holds the manual repair's open `number` (1 is the copy, 2 the writer)."""
    from recall.services.manual_repair import ManualRepair

    admit = ManualRepair._admit
    calls: list[bool] = []

    def refusing(self, lease, version):
        calls.append(True)
        if len(calls) == number:
            self._hold(reason)
        return admit(self, lease, version)

    with patch.object(ManualRepair, "_admit", refusing):
        yield


@contextmanager
def _hold_while_copying_the_database() -> Iterator[None]:
    """Stop the manual repair inside the copy that precedes its diagnostic open."""
    from recall.services import manual_repair

    copy = manual_repair.shutil.copyfile
    held: list[bool] = []

    def copying(source, target, *args, **kwargs):
        if not held:
            held.append(True)
            hold_until_released()
        return copy(source, target, *args, **kwargs)

    with patch.object(manual_repair.shutil, "copyfile", copying):
        yield


def hold_owner(config: AppConfig, owner: str) -> None:
    """Sit inside one long-lived offline owner without publishing any repair progress."""
    from recall.services.fts_sidecar_rebuild import KeywordRebuild
    from recall.services.manual_repair import ManualRepair
    from recall.services.recovery_admission import offline_file_lifetime

    if owner == "advisory_lock":
        with advisory_lock(config.lock_path):
            hold_until_released()
    elif owner == "file_lifetime":
        with (
            offline_file_lifetime(config) as (jobs, admission),
            owned_connection(jobs, "held", lambda: admission.open_preserving(read_only=True)),
        ):
            hold_until_released()
    elif owner == "write_lease":
        from recall.services.write_access import standalone_write_lease

        with standalone_write_lease(config):
            hold_until_released()
    elif owner == "compaction":

        def hold_then_stop(*_args: object, **_kwargs: object) -> None:
            hold_until_released()
            raise _Released

        from recall.services.compaction import compact

        with patch("recall.services.compaction.estimate_bloat_ratio_on", hold_then_stop):
            try:
                compact(config)
            except _Released:
                return
    elif owner == "manual_rebuild":
        ManualRepair(config, barrier=lambda _stage: hold_until_released()).execute()
    elif owner == "manual_rebuild_from_wal":
        with _first_open_fails_replaying_the_wal():
            ManualRepair(config, barrier=lambda _stage: hold_until_released()).execute()
    elif owner == "manual_rebuild_copying_the_wal":
        with _first_open_fails_replaying_the_wal(), _hold_while_copying_the_database():
            ManualRepair(config).execute()
    elif owner == "keyword_rebuild":
        rebuild = KeywordRebuild(
            config=config, operation_dir=config.data_dir / "keyword-rebuild", build_id="held"
        )
        with patch.object(KeywordRebuild, "_abort", lambda _self, _lifetime: hold_until_released()):
            rebuild.abort()
    else:
        assert owner == "supersede", owner

        def hold_then_stop(*_args: object, **_kwargs: object) -> None:
            hold_until_released()
            raise _Released

        from recall.cli.db import _supersede_moved_locally

        with patch("recall.cli.db.census_moved_duplicates", hold_then_stop):
            try:
                _supersede_moved_locally(config, apply=False)
            except _Released:
                return
    assert _held, f"{owner} finished without being held"


def try_owner(config: AppConfig, owner: str) -> int:
    """Run one offline writer to its first admission; the exit status says how it ended.

    Zero is a completed run, 3 a refusal by recovery admission, 2 another process holding
    `recall.lock` or the database, 4 a check of the operation itself that failed.
    """
    from recall.cli.contract import CliError
    from recall.cli.db import _supersede_moved_locally
    from recall.db.connection import RecallLockError
    from recall.services.fts_sidecar_rebuild import KeywordRebuild
    from recall.services.index_migration import rollback_migration
    from recall.services.manual_repair import ManualRepair
    from recall.services.self_repair import DaemonStartupRefused
    from recall.services.write_access import standalone_write_lease

    try:
        if owner == "write_lease":
            with standalone_write_lease(config):
                pass
        elif owner == "rollback":
            rollback_migration(config, preserve=config.data_dir.parent / "failed-set")
        elif owner == "manual_rebuild":
            ManualRepair(config).execute()
        elif owner == "manual_rebuild_from_wal":
            with _first_open_fails_replaying_the_wal():
                ManualRepair(config).execute()
        elif owner == "manual_rebuild_from_wal_copy_refused":
            with (
                _first_open_fails_replaying_the_wal(),
                _admission_refuses_open(
                    1, "durable state does not admit an explicit manual selection"
                ),
            ):
                ManualRepair(config).execute()
        elif owner == "manual_rebuild_from_wal_writer_refused":
            with (
                _first_open_fails_replaying_the_wal(),
                _admission_refuses_open(
                    2, "manual writer differs from its diagnosed recovery state"
                ),
            ):
                ManualRepair(config).execute()
        elif owner == "keyword_rebuild":
            KeywordRebuild(
                config=config, operation_dir=config.data_dir / "keyword-rebuild", build_id="try"
            ).abort()
        else:
            assert owner == "supersede", owner
            _supersede_moved_locally(config, apply=False)
    except DaemonStartupRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return 3
    except RecallLockError as error:
        print(f"locked: {error}", file=sys.stderr)
        return 2
    except CliError as error:
        print(f"cli error {error.exit_code}: {error}", file=sys.stderr)
        return error.exit_code
    except RuntimeError as error:
        print(f"failed: {error}", file=sys.stderr)
        return 4
    return 0


def main() -> None:
    operation, selected, fault = sys.argv[1:]
    config = AppConfig.load()
    if operation == "hold_owner":
        hold_owner(config, selected)
        return
    if operation == "try_owner":
        raise SystemExit(try_owner(config, selected))
    if operation == "startup":
        from recall.services.self_repair import DaemonStartupRefused, run_startup_self_repair

        try:
            outcome = run_startup_self_repair(config)
        except DaemonStartupRefused as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(3) from error
        print(outcome)
        return
    if operation in {"repair", "repair_with_progress"}:
        repair(config, selected, fault, publish_progress=operation == "repair_with_progress")
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
