"""Inner disk failures retain verification even when a caller returns defaults."""

import asyncio
from pathlib import Path
from uuid import uuid4

import duckdb
import pytest
from recall.core.repair_state import PublicationRole
from recall.db.repair import SchemaFence
from recall.services.owned_jobs import WorkNotAdmitted
from recall.services.repair_events import RecoveryEvent, publish_event, read_events
from recall.services.self_repair import marker_path, remember_fatal_failure

from tests.test_services.test_rpc_owned_boundary import server_at


def test_display_flag_failure_cannot_replace_original_disk_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import recall.services.self_repair as self_repair

    server = server_at(tmp_path)
    server._database.call("fixture", lambda conn: conn.execute("SELECT 1").fetchone())
    original = duckdb.TransactionException("No space left on device")
    cleanup = duckdb.FatalException("Failed to delete all rows from index in display flag")

    def flag_fails(*args):
        raise cleanup

    def fails():
        raise original

    monkeypatch.setattr(self_repair, "set_needs_index_verification", flag_fails)

    async def scenario():
        try:
            with pytest.raises(duckdb.TransactionException) as result:
                await server._writer_call(fails, "source_commit")
            assert result.value is original
            batch = read_events(tmp_path)
            assert batch.primary is not None and batch.primary.failure is not None
            assert batch.primary.failure.message == str(cleanup)
            assert len(batch.secondary) == 1
            assert batch.secondary[0][1].failure is None
            assert batch.secondary[0][1].needs_index_verification is True
            assert server._shutdown_event.is_set()
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_inner_fail_soft_disk_error_publishes_one_secondary_and_keeps_primary(
    tmp_path: Path,
) -> None:
    server = server_at(tmp_path)
    server._database.call("fixture", lambda conn: conn.execute("SELECT 1").fetchone())
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("unrelated fatal from prior process"),
        site="old_origin",
        fence=SchemaFence(41),
    )
    primary = marker_path(tmp_path).read_bytes()
    error = duckdb.TransactionException("Could not write WAL: No space left on device")

    def fail(_conn):
        raise error

    def translated(lease):
        try:
            lease.call("row_probe", fail)
        except duckdb.TransactionException:
            return "default"

    try:
        assert server._database.run("helper", translated) == "default"
        wrapper = ValueError("translated")
        wrapper.__cause__ = error
        assert server._jobs.observe(wrapper, "outer") is False
        batch = read_events(tmp_path)
        assert marker_path(tmp_path).read_bytes() == primary
        assert len(batch.secondary) == 1
        event = batch.secondary[0][1]
        assert isinstance(event, RecoveryEvent)
        assert event.failure is None and event.needs_index_verification is True
        assert server._jobs.first_fatal is None
        assert not server._shutdown_event.is_set()
    finally:
        asyncio.run(server.stop())


def test_live_full_spool_stops_following_mutation_and_completion(tmp_path: Path) -> None:
    server = server_at(tmp_path)
    server._database.call(
        "fixture", lambda conn: conn.execute("CREATE TABLE witness(k INT)").fetchone()
    )
    permit = server._jobs.completion_permit()
    for _ in range(63):
        publish_event(
            tmp_path,
            RecoveryEvent(event_id=uuid4(), failure=None, needs_index_verification=True),
            role=PublicationRole.SECONDARY,
        )
    try:
        server._jobs.observe(duckdb.TransactionException("No space left on device"), "inner")
        assert read_events(tmp_path).full
        with pytest.raises(WorkNotAdmitted):
            server._database.call(
                "forbidden", lambda conn: conn.execute("INSERT INTO witness VALUES (1)").fetchone()
            )
        with pytest.raises(WorkNotAdmitted), server._jobs.completing(permit):
            pytest.fail("a full spool cannot release recovery state")
        assert server._jobs.first_fatal is None
        assert server._shutdown_event.is_set()
    finally:
        asyncio.run(server.stop())
    with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == []
