"""Independent durable instructions precede destructive index work."""

from datetime import datetime
from pathlib import Path
from uuid import UUID

import duckdb
from recall.db.schema import ensure_schema
from recall.services.owned_jobs import DaemonJobs
from recall.services.repair_events import EventFailure, RecoveryEvent, publish_event, read_events
from recall.services.repair_protocol import RepairProtocol

EVENT = UUID("00000000-0000-4000-8000-000000000001")


def actionable(data_dir: Path) -> None:
    publish_event(
        data_dir,
        RecoveryEvent(
            event_id=EVENT,
            failure=EventFailure(
                site="keyword_retirement",
                exception_type="FatalException",
                message="Failed to delete all rows from index",
                signature="first",
                failure_class="index-divergence",
                at=datetime(2026, 11, 1),
            ),
            needs_index_verification=False,
        ),
    )


def test_replacement_retires_only_the_previous_settled_oracle(tmp_path: Path) -> None:
    from recall.db.repair import load_repair_row, save_repair_row

    db = tmp_path / "recall.duckdb"
    actionable(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    protocol = RepairProtocol(tmp_path)
    with duckdb.connect(str(db)) as conn:
        ensure_schema(conn)
        with jobs.lease(conn) as lease:
            protocol.prepare(lease, read_events(tmp_path))
            rebuilt = protocol.rebuild(lease, protocol.arm(lease))
    with duckdb.connect(str(db)) as fresh:
        with jobs.lease(fresh) as lease:
            old = protocol.verify(lease, rebuilt)
            protocol.retire(lease)
        assert old.oracle_ref is not None
        previous = tmp_path / old.oracle_ref
        # This fixture models a later qualified run; completion has its own
        # acceptance family and is not established by setting these flags.
        save_repair_row(
            fresh,
            old.model_copy(update={"success_since_attempt": True, "success_since_failure": True}),
        )
        event = RecoveryEvent(
            event_id=UUID(int=2), failure=None, needs_index_verification=True, probe=True
        )
        publish_event(tmp_path, event)
        with jobs.lease(fresh) as lease:
            current = RepairProtocol(tmp_path).prepare(lease, read_events(tmp_path))
        assert current.phase == "required"
        assert not previous.exists()
        assert {path.name for path in (tmp_path / "repair-oracles").iterdir()} == {
            str(event.event_id)
        }
    with duckdb.connect(str(db), read_only=True) as fresh:
        saved = load_repair_row(fresh)
        assert saved.oracle_ref == f"repair-oracles/{event.event_id}"
        assert (tmp_path / saved.oracle_ref / "manifest.json").is_file()


def test_required_saves_complete_oracle_and_catalog_without_mutation(tmp_path: Path) -> None:
    db = tmp_path / "recall.duckdb"
    actionable(tmp_path)
    original = (tmp_path / "daemon-failure.json").read_bytes()
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(db)) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE witness (id VARCHAR, value VARCHAR)")
        conn.execute("INSERT INTO witness VALUES ('a', 'acknowledged')")
        conn.execute("CREATE INDEX witness_only ON witness(value)")
        with jobs.lease(conn) as lease:
            row = RepairProtocol(tmp_path).prepare(lease, read_events(tmp_path))
        assert row.phase == "required"
        assert row.trigger_id == EVENT
        assert row.attempt_no == 0
        assert row.success_since_attempt
        assert row.oracle_ref is not None and row.oracle_hash is not None
        assert (tmp_path / row.oracle_ref / "manifest.json").is_file()
        assert row.target_manifest is not None
        assert "witness_only" in {target.index_name for target in row.target_manifest.targets}
        assert conn.execute("SELECT * FROM witness").fetchall() == [("a", "acknowledged")]
        assert (tmp_path / "daemon-failure.json").read_bytes() == original
    with duckdb.connect(str(db)) as fresh:
        assert fresh.execute("SELECT phase, attempt_no FROM daemon_index_repair").fetchone() == (
            "required",
            0,
        )


def test_committed_arm_is_spent_and_a_fresh_process_cannot_resume(tmp_path: Path) -> None:
    import pytest
    from recall.db.repair import load_repair_row
    from recall.services.self_repair import DaemonStartupRefused

    db = tmp_path / "recall.duckdb"
    actionable(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(db)) as conn:
        ensure_schema(conn)
        protocol = RepairProtocol(tmp_path)
        with jobs.lease(conn) as lease:
            protocol.prepare(lease, read_events(tmp_path))
            capability = protocol.arm(lease)
            assert capability is not None
        assert conn.execute(
            "SELECT phase, attempt_no, success_since_attempt, success_since_failure "
            "FROM daemon_index_repair"
        ).fetchone() == ("running", 1, False, False)
    with duckdb.connect(str(db)) as fresh:
        before = load_repair_row(fresh)
        with jobs.lease(fresh) as lease, pytest.raises(DaemonStartupRefused):
            RepairProtocol(tmp_path).prepare(lease, read_events(tmp_path))
        assert load_repair_row(fresh) == before


def test_same_process_arm_can_verify_exact_rows_after_fresh_reopen(tmp_path: Path) -> None:
    db = tmp_path / "recall.duckdb"
    actionable(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    protocol = RepairProtocol(tmp_path)
    with duckdb.connect(str(db)) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE witness (id VARCHAR, value BLOB)")
        conn.execute("INSERT INTO witness VALUES ('a', '\\x00\\xFF'::BLOB)")
        conn.execute("CREATE INDEX witness_only ON witness(id)")
        with jobs.lease(conn) as lease:
            protocol.prepare(lease, read_events(tmp_path))
            capability = protocol.arm(lease)
            rebuilt = protocol.rebuild(lease, capability)
    with duckdb.connect(str(db)) as fresh:
        with jobs.lease(fresh) as lease:
            row = protocol.verify(lease, rebuilt)
        assert row.phase == "verified"
        assert row.attempt_no == 1
        assert not row.success_since_attempt
        assert not row.success_since_failure
        assert fresh.execute("SELECT * FROM witness").fetchall() == [("a", b"\x00\xff")]
        assert fresh.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE index_name = 'witness_only'"
        ).fetchone() == ("witness_only",)
        assert {entry.disposition for entry in row.acknowledged_events} == {"verified"}
        with jobs.lease(fresh) as lease:
            protocol.retire(lease)
        assert not (tmp_path / "daemon-failure.json").exists()
        assert fresh.execute(
            "SELECT success_since_attempt, success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


def test_a_rebuild_cannot_be_verified_on_its_original_connection(tmp_path: Path) -> None:
    import pytest
    from recall.services.self_repair import DaemonStartupRefused

    actionable(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(tmp_path / "recall.duckdb")) as conn:
        ensure_schema(conn)
        protocol = RepairProtocol(tmp_path)
        with jobs.lease(conn) as lease:
            protocol.prepare(lease, read_events(tmp_path))
            rebuilt = protocol.rebuild(lease, protocol.arm(lease))
            with pytest.raises(DaemonStartupRefused, match="fresh"):
                protocol.verify(lease, rebuilt)
        assert conn.execute("SELECT phase FROM daemon_index_repair").fetchone() == ("running",)


def test_duplicate_event_files_fold_one_observation_once(tmp_path: Path) -> None:
    actionable(tmp_path)
    batch = read_events(tmp_path)
    assert isinstance(batch.primary, RecoveryEvent)
    publish_event(tmp_path, batch.primary)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(tmp_path / "recall.duckdb")) as conn:
        ensure_schema(conn)
        with jobs.lease(conn) as lease:
            row = RepairProtocol(tmp_path).prepare(lease, read_events(tmp_path))
        assert len(row.acknowledged_events) == 1
        assert conn.execute("SELECT fatal_repeat_count FROM runtime_state").fetchone() == (1,)


def test_counter_exhaustion_holds_before_an_unusable_running_commit(tmp_path: Path) -> None:
    import pytest
    from recall.db.repair import ATTEMPT_MAX, load_repair_row
    from recall.services.self_repair import DaemonStartupRefused

    db = tmp_path / "recall.duckdb"
    actionable(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(db)) as conn:
        ensure_schema(conn)
        protocol = RepairProtocol(tmp_path)
        with jobs.lease(conn) as lease:
            protocol.prepare(lease, read_events(tmp_path))
            conn.execute("UPDATE daemon_index_repair SET attempt_no = ?", [ATTEMPT_MAX - 1])
            before = load_repair_row(conn)
            with pytest.raises(DaemonStartupRefused, match="counter"):
                protocol.arm(lease)
        assert load_repair_row(conn) == before
    with duckdb.connect(str(db), read_only=True) as fresh:
        row = load_repair_row(fresh)
        assert (row.phase, row.attempt_no) == ("required", ATTEMPT_MAX - 1)
        assert row.success_since_attempt is True
