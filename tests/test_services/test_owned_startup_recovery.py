"""Actual startup consumes the durable protocol under the parent's exclusive lifetime."""

import asyncio
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.db.connection import connect
from recall.db.maintenance import DivergedKey, IndexDivergenceReport
from recall.db.repair import SchemaFence, load_repair_row
from recall.services import repair_protocol, self_repair
from recall.services.owned_jobs import DaemonJobs
from recall.services.recovery_admission import RecoveryAdmission
from recall.services.repair_events import EventFailure, RecoveryEvent, publish_event, read_events
from recall.services.repair_protocol import RepairProtocol
from recall.services.self_repair import DaemonStartupRefused, marker_path, remember_fatal_failure

from tests.test_services.test_rpc_owned_boundary import server_at


def negative_probe() -> IndexDivergenceReport:
    return IndexDivergenceReport(
        checked_at=datetime(2026, 11, 1),
        indexes_probed=1,
        samples_checked=1,
        samples_unverifiable=0,
        diverged=(),
        complete=True,
        elapsed_seconds=0.0,
    )


@pytest.mark.parametrize(
    ("barrier", "expected_phase"),
    [("before_refused_commit", "verified"), ("after_refused_commit", "refused")],
)
def test_refusal_acknowledgment_failure_reads_fresh_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, barrier: str, expected_phase: str
) -> None:
    server = server_at(tmp_path)
    remember_fatal_failure(
        tmp_path, duckdb.FatalException("Failed to delete all rows from index first"), site="first"
    )
    self_repair.run_startup_self_repair(server._config)
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index later"),
        site="second",
        fence=SchemaFence(41),
    )
    original = marker_path(tmp_path).read_bytes()
    resolutions: list[str] = []
    read = RecoveryAdmission.read_retained_state

    def read_fresh(admission: RecoveryAdmission):
        row = read(admission)
        resolutions.append(row.phase)
        return row

    def interrupt(stage: str) -> None:
        if stage == barrier:
            raise OSError("commit acknowledgment lost")

    class InterruptedProtocol(RepairProtocol):
        def __init__(self, data_dir, **kwargs):
            super().__init__(data_dir, barrier=interrupt, **kwargs)

    monkeypatch.setattr(repair_protocol, "RepairProtocol", InterruptedProtocol)
    monkeypatch.setattr(RecoveryAdmission, "read_retained_state", read_fresh)
    with pytest.raises(DaemonStartupRefused, match=f"phase={expected_phase}"):
        self_repair.run_startup_self_repair(server._config)
    assert resolutions == [expected_phase]
    assert marker_path(tmp_path).read_bytes() == original
    with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
        row = load_repair_row(fresh)
        assert row.phase == expected_phase
        assert row.attempt_no == 1
        assert row.success_since_attempt is False
        assert row.success_since_failure is False
    asyncio.run(server.stop())


@pytest.mark.parametrize("later", ["distinct_index", "positive_probe"])
def test_spent_start_refuses_new_instruction_and_preserves_prior_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, later: str
) -> None:
    server = server_at(tmp_path)
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index first"),
        site="first_commit",
    )
    monkeypatch.setattr(
        self_repair, "probe_index_divergence", lambda *args, **kwargs: negative_probe()
    )
    self_repair.run_startup_self_repair(server._config)
    with connect(server._config) as fresh:
        before = load_repair_row(fresh)

    if later == "distinct_index":
        event = RecoveryEvent(
            event_id=UUID("00000000-0000-4000-8000-000000000002"),
            failure=EventFailure(
                site="different_origin",
                exception_type="FatalException",
                message="Failed to delete all rows from index second",
                signature="different",
                failure_class="index-divergence",
                at=datetime(2026, 11, 2),
            ),
            needs_index_verification=False,
        )
        publish_event(tmp_path, event)
        monkeypatch.setattr(
            self_repair,
            "probe_index_divergence",
            lambda *args, **kwargs: pytest.fail(
                "spent actionable admission precedes optional probe"
            ),
        )
    else:
        mismatch = DivergedKey(table="witness", column="id", key="a", index_count=0, full_count=1)
        monkeypatch.setattr(
            self_repair,
            "probe_index_divergence",
            lambda *args, **kwargs: replace(negative_probe(), diverged=(mismatch,)),
        )

    with pytest.raises(DaemonStartupRefused):
        self_repair.run_startup_self_repair(server._config)
    batch = read_events(tmp_path)
    events = batch.identified()
    assert len(events) == 1
    if later == "positive_probe":
        assert events[0].probe is True and events[0].failure is None
    with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
        row = load_repair_row(fresh)
        assert (row.phase, row.attempt_no) == ("refused", 1)
        assert (row.success_since_attempt, row.success_since_failure) == (False, False)
        assert (row.trigger_json, row.target_manifest, row.oracle_ref, row.oracle_hash) == (
            before.trigger_json,
            before.target_manifest,
            before.oracle_ref,
            before.oracle_hash,
        )
        assert row.acknowledged_events[0].event_id == events[0].event_id
        assert row.acknowledged_events[0].disposition == "folded"
        if later == "positive_probe":
            assert fresh.execute("SELECT last_fatal_signature FROM runtime_state").fetchone() == (
                before.signature,
            )
    assert self_repair.refusal_marker_path(tmp_path).is_file()
    asyncio.run(server.stop())


def test_refused_start_neither_listens_nor_estimates_bloat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import recall.services.compaction as compaction

    server = server_at(tmp_path)
    observations: list[str] = []

    async def refuse() -> None:
        assert not server._socket_path.exists()
        raise DaemonStartupRefused("unfinished recovery")

    def estimate(*args, **kwargs):
        observations.append("bloat")
        raise AssertionError("bloat must follow recovery admission")

    monkeypatch.setattr(server, "_run_startup_self_repair", refuse)
    monkeypatch.setattr(compaction, "estimate_bloat_ratio", estimate)

    async def scenario() -> None:
        try:
            with pytest.raises(DaemonStartupRefused, match="unfinished recovery"):
                await server._prepare_owned(idle_timeout=None, watch=False)
            assert observations == []
            assert not server._socket_path.exists()
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_actual_startup_records_verified_spent_after_fresh_reopen(tmp_path: Path) -> None:
    server = server_at(tmp_path)
    with connect(server._config) as conn:
        conn.execute("CREATE TABLE repair_witness(k INT, value TEXT)")
        conn.execute("INSERT INTO repair_witness VALUES (1, 'same'), (1, 'same'), (2, NULL)")
        conn.execute("CREATE INDEX repair_witness_compound ON repair_witness(k, value)")
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES "
            "('message', 'm-witness', 'delete'), ('message', 'm-witness', 'delete')"
        )
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index in original run"),
        site="source_commit",
    )
    before = server._database.run("epoch", lambda lease: lease.connection_epoch)

    async def scenario() -> None:
        try:
            await server._run_startup_self_repair()
            after = server._database.run("epoch", lambda lease: lease.connection_epoch)
            assert after is not before
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
        row = load_repair_row(fresh)
        assert (row.phase, row.attempt_no) == ("verified", 1)
        assert (row.success_since_attempt, row.success_since_failure) == (False, False)
        assert fresh.execute("SELECT * FROM repair_witness ORDER BY k").fetchall() == [
            (1, "same"),
            (1, "same"),
            (2, None),
        ]
        assert fresh.execute("SELECT kind, id, op FROM fts_sidecar_pending").fetchall() == [
            ("message", "m-witness", "delete"),
            ("message", "m-witness", "delete"),
        ]
        assert fresh.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE index_name = 'repair_witness_compound'"
        ).fetchall() == [("repair_witness_compound",)]
    assert not marker_path(tmp_path).exists()
    assert read_events(tmp_path).secondary == ()


def test_new_owner_holds_running_before_a_canonical_write_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preparing = server_at(tmp_path)
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index before interruption"),
        site="source_commit",
    )
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with connect(preparing._config) as conn, jobs.lease(conn) as lease:
        protocol = RepairProtocol(tmp_path)
        protocol.prepare(lease, read_events(tmp_path))
        protocol.arm(lease)
    asyncio.run(jobs.close())
    asyncio.run(preparing.stop())
    original = marker_path(tmp_path).read_bytes()
    server = server_at(tmp_path)
    operations: list[str] = []
    call = server._jobs.call_db

    def capture(operation, work):
        operations.append(operation)
        return call(operation, work)

    monkeypatch.setattr(server._jobs, "call_db", capture)
    try:
        with pytest.raises(DaemonStartupRefused):
            asyncio.run(server._run_startup_self_repair())
        assert "database_writer_open" not in operations
        assert marker_path(tmp_path).read_bytes() == original
    finally:
        asyncio.run(server.stop())
    with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
        assert (load_repair_row(fresh).phase, load_repair_row(fresh).attempt_no) == ("running", 1)
