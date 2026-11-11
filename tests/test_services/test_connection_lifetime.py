"""A read's primary diagnostic survives failure while closing its cursor."""

import asyncio
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from recall.services.self_repair import read_failure_marker

from tests.test_services.test_rpc_owned_boundary import server_at


def test_cursor_cleanup_cannot_replace_an_actionable_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = server_at(tmp_path)
    primary = duckdb.FatalException("Failed to delete all rows from index")

    class Cursor:
        def close(self) -> None:
            raise duckdb.FatalException("database has been invalidated by cleanup")

    class Parent:
        def cursor(self) -> Cursor:
            return Cursor()

        def close(self) -> None:
            pass

    from recall.services.connection_owner import ConnectionOwner

    server._database = ConnectionOwner(
        server._jobs, lambda: cast(Any, Parent()), server._recovery_admission.resolve_completion
    )

    def read(_conn: object) -> None:
        raise primary

    with pytest.raises(BaseExceptionGroup, match="read and cursor close failed"):
        server._run_with_read_cursor_sync(read)
    marker = read_failure_marker(tmp_path)
    assert marker is not None and marker.failure is not None
    assert marker.failure.message == "Failed to delete all rows from index"
    asyncio.run(server.stop())


def test_exclusive_reopen_drains_read_cursor_and_expires_its_lease(tmp_path: Path) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from recall.db.connection import connect
    from recall.db.lease import DbLease
    from recall.services.connection_owner import ConnectionOwner
    from recall.services.owned_jobs import DaemonJobs

    server = server_at(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    owner = ConnectionOwner(
        jobs, lambda: connect(server._config), server._recovery_admission.resolve_completion
    )
    entered = threading.Event()
    release = threading.Event()
    escaped: list[DbLease] = []

    def read(lease: DbLease) -> tuple[object, list[tuple[int]]]:
        escaped.append(lease)
        entered.set()
        assert release.wait(5)
        return lease.connection_epoch, lease.call(
            "witness_read", lambda conn: conn.execute("SELECT value FROM witness").fetchall()
        )

    def reopen() -> tuple[object, list[tuple[int]]]:
        with owner.exclusive() as database:
            database.reopen()
            return database.run(
                "fresh_witness",
                lambda lease: (
                    lease.connection_epoch,
                    lease.call(
                        "fresh_witness_read",
                        lambda conn: conn.execute("SELECT value FROM witness").fetchall(),
                    ),
                ),
            )

    try:
        owner.call(
            "witness_create",
            lambda conn: conn.execute("CREATE TABLE witness(value INT)").fetchall(),
        )
        owner.call(
            "witness_insert",
            lambda conn: conn.execute("INSERT INTO witness VALUES (73)").fetchall(),
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            reading = pool.submit(lambda: owner.read_sync("read_witness", read))
            assert entered.wait(5)
            reopening = pool.submit(reopen)
            release.set()
            old_epoch, old_rows = reading.result(timeout=5)
            new_epoch, new_rows = reopening.result(timeout=5)
        assert old_rows == new_rows == [(73,)]
        assert old_epoch is not new_epoch
        with pytest.raises(AssertionError, match="expired DB lease"):
            escaped[0].call("late_read", lambda conn: conn.execute("SELECT 1").fetchall())
    finally:
        release.set()
        owner.close()


def test_maintenance_close_failure_is_observed_before_its_caller_can_translate(
    tmp_path: Path,
) -> None:
    from recall.services.connection_owner import ConnectionOwner

    server = server_at(tmp_path)
    error = duckdb.FatalException("Failed to delete all rows from index during close")

    class Connection:
        def close(self) -> None:
            raise error

    owner = ConnectionOwner(
        server._jobs, lambda: cast(Any, Connection()), server._recovery_admission.resolve_completion
    )
    owner.run("open", lambda lease: None)
    with pytest.raises(duckdb.FatalException, match="during close"), owner.maintenance():
        pytest.fail("a failed close must not enter maintenance")
    marker = read_failure_marker(tmp_path)
    assert marker is not None and marker.failure is not None
    assert marker.failure.message == "Failed to delete all rows from index during close"
    assert marker.failure.site == "database_close"
    assert server._shutdown_event.is_set()
    owner.close()
    asyncio.run(server.stop())


def test_owned_reads_preserve_domain_exception_identity_and_validation(tmp_path: Path) -> None:
    from recall.core.poll_reads import PollSnapshotSelected, poll_read_request
    from recall.core.rpc_types import (
        INVALID_PARAMS,
        RpcAdmissionBusyError,
        RpcCallError,
        RpcConnectionError,
        RpcError,
    )

    server = server_at(tmp_path)
    request = poll_read_request("recall.stats", {})
    assert request is not None
    errors = [
        RpcError(INVALID_PARAMS, "invalid cursor"),
        RpcCallError(INVALID_PARAMS, "invalid cursor"),
        RpcConnectionError("endpoint unavailable"),
        RpcAdmissionBusyError(),
        PollSnapshotSelected(server._config, request, "scratch"),
    ]
    try:
        for error in errors:

            def fail(conn: duckdb.DuckDBPyConnection, error=error) -> None:
                assert conn.execute("SELECT 1").fetchone() == (1,)
                raise error

            with pytest.raises(type(error)) as caught:
                server._run_with_read_cursor_sync(fail)
            assert caught.value is error
        assert not server._shutdown_event.is_set()
        assert read_failure_marker(tmp_path) is None
    finally:
        asyncio.run(server.stop())
