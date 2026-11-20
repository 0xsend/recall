"""Independent durable instructions precede destructive index work."""

from datetime import datetime
from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services.owned_jobs import DaemonJobs
from recall.services.repair_events import EventFailure, RecoveryEvent, publish_event, read_events
from recall.services.repair_protocol import RepairProtocol

EVENT = UUID("00000000-0000-4000-8000-000000000001")


def test_fresh_verification_requires_the_actual_schema_indexes(tmp_path: Path) -> None:
    from recall.db.repair import load_repair_row
    from recall.services.self_repair import DaemonStartupRefused

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
        fresh.execute("DROP INDEX idx_messages_session")
        with jobs.lease(fresh) as lease, pytest.raises(DaemonStartupRefused, match="catalog"):
            protocol.verify(lease, rebuilt)
        assert load_repair_row(fresh).phase == "running"


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


@pytest.mark.parametrize("entry", ["startup", "library"])
def test_missing_unfinished_fold_is_not_an_empty_healthy_namespace(
    tmp_path: Path, entry: str
) -> None:
    import pytest
    from recall.db.repair import load_repair_row
    from recall.services.self_repair import DaemonStartupRefused

    actionable(tmp_path)
    database = tmp_path / "recall.duckdb"
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)

    def interrupt(stage: str) -> None:
        if stage == "before_required_commit":
            raise OSError("required commit unavailable")

    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        with jobs.lease(conn) as lease, pytest.raises(OSError):
            RepairProtocol(tmp_path, barrier=interrupt).prepare(lease, read_events(tmp_path))
        before = load_repair_row(conn)
        before_runtime = conn.execute("SELECT * FROM runtime_state").fetchall()
        assert before.phase == "idle" and before.acknowledged_events[0].disposition == "folded"
    (tmp_path / "daemon-failure.json").unlink()
    with duckdb.connect(str(database)) as fresh:
        if entry == "startup":
            with jobs.lease(fresh) as lease, pytest.raises(DaemonStartupRefused):
                RepairProtocol(tmp_path).begin_start(lease)
        else:
            from recall.services.indexer import index_sessions

            from tests.test_services.test_run_completion import _config

            with pytest.raises(DaemonStartupRefused):
                index_sessions(
                    source=None,
                    full=False,
                    recreate=False,
                    verbose=False,
                    config=_config(tmp_path, None),
                    conn=fresh,
                )
        assert load_repair_row(fresh) == before
        assert fresh.execute("SELECT * FROM runtime_state").fetchall() == before_runtime


def test_restart_reclaims_final_ledger_after_durable_unlink_interruption(tmp_path: Path) -> None:
    import pytest
    from recall.db.repair import load_repair_row

    actionable(tmp_path)
    database = tmp_path / "recall.duckdb"
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)

    def interrupt(stage: str) -> None:
        if stage == "after_event_unlink_sync":
            raise RuntimeError("owner departed after durable unlink")

    protocol = RepairProtocol(tmp_path, barrier=interrupt)
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        with jobs.lease(conn) as lease:
            protocol.prepare(lease, read_events(tmp_path))
            rebuilt = protocol.rebuild(lease, protocol.arm(lease))
    with duckdb.connect(str(database)) as fresh:
        with jobs.lease(fresh) as lease:
            protocol.verify(lease, rebuilt)
            with pytest.raises(RuntimeError, match="owner departed"):
                protocol.retire(lease)
        retained = load_repair_row(fresh)
        assert len(retained.acknowledged_events) == 1
    assert not (tmp_path / "daemon-failure.json").exists()
    with duckdb.connect(str(database)) as restarted:
        with jobs.lease(restarted) as lease:
            RepairProtocol(tmp_path).begin_start(lease)
        current = load_repair_row(restarted)
        assert current.acknowledged_events == ()
        assert current.trigger_json == retained.trigger_json
        assert current.oracle_hash == retained.oracle_hash
        assert not current.success_since_attempt and not current.success_since_failure


def test_new_probe_before_retirement_refuses_and_keeps_every_event(tmp_path: Path) -> None:
    import pytest
    from recall.db.repair import load_repair_row
    from recall.services.self_repair import DaemonStartupRefused

    actionable(tmp_path)
    database = tmp_path / "recall.duckdb"
    original = (tmp_path / "daemon-failure.json").read_bytes()
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    next_event = RecoveryEvent(
        event_id=UUID(int=2), failure=None, needs_index_verification=True, probe=True
    )

    def arrive(stage: str) -> None:
        if stage == "before_event_unlink":
            publish_event(tmp_path, next_event)

    protocol = RepairProtocol(tmp_path, barrier=arrive)
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        with jobs.lease(conn) as lease:
            protocol.prepare(lease, read_events(tmp_path))
            rebuilt = protocol.rebuild(lease, protocol.arm(lease))
    with duckdb.connect(str(database)) as fresh:
        with jobs.lease(fresh) as lease:
            verified = protocol.verify(lease, rebuilt)
            with pytest.raises(DaemonStartupRefused):
                protocol.retire(lease)
        current = load_repair_row(fresh)
        assert current.phase == "refused"
        assert current.oracle_hash == verified.oracle_hash
        assert current.trigger_json == verified.trigger_json
        assert not current.success_since_attempt and not current.success_since_failure
        assert next_event.event_id in {entry.event_id for entry in current.acknowledged_events}
    assert (tmp_path / "daemon-failure.json").read_bytes() == original
    assert (tmp_path / "fatal-events" / f"{next_event.event_id}.json").is_file()


@pytest.mark.parametrize("uncaptured", [False, True])
def test_only_the_saved_unfinished_trigger_can_substitute_for_a_missing_event(
    tmp_path: Path,
    uncaptured: bool,
) -> None:
    from recall.db.repair import load_repair_row
    from recall.services.self_repair import DaemonStartupRefused

    actionable(tmp_path)
    database = tmp_path / "recall.duckdb"
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        with jobs.lease(conn) as lease:
            protocol = RepairProtocol(tmp_path)
            protocol.prepare(lease, read_events(tmp_path))
            if uncaptured:
                extra = RecoveryEvent(
                    event_id=UUID(int=2), failure=None, needs_index_verification=True, probe=True
                )
                path = publish_event(tmp_path, extra)
                protocol._fold(lease, read_events(tmp_path))
            else:
                path = tmp_path / "daemon-failure.json"
        before = load_repair_row(conn)
    path.unlink()
    with duckdb.connect(str(database)) as fresh:
        with jobs.lease(fresh) as lease:
            if uncaptured:
                with pytest.raises(DaemonStartupRefused):
                    RepairProtocol(tmp_path).prepare(lease, read_events(tmp_path))
            else:
                resumed = RepairProtocol(tmp_path).prepare(lease, read_events(tmp_path))
                assert resumed == before
        assert load_repair_row(fresh) == before


def test_final_absent_entries_do_not_exhaust_a_new_instruction_fold(tmp_path: Path) -> None:
    from recall.db.repair import EventAcknowledgement, RepairRow, load_repair_row, save_repair_row

    database = tmp_path / "recall.duckdb"
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        row = load_repair_row(conn)
        # Legal interrupted-retirement state: every prior record was finally
        # diagnosed and durably removed before its ledger cleanup was committed.
        final = tuple(
            EventAcknowledgement(
                event_id=UUID(int=ordinal),
                content_hash=RecoveryEvent(
                    event_id=UUID(int=ordinal),
                    failure=None,
                    needs_index_verification=True,
                    probe=True,
                ).content_hash,
                disposition="diagnosed",
            )
            for ordinal in range(1, 66)
        )
        save_repair_row(
            conn,
            RepairRow.model_validate(
                row.model_dump()
                | {
                    "acknowledged_events": final,
                    "success_since_failure": False,
                }
            ),
        )
        event = RecoveryEvent(
            event_id=UUID(int=66), failure=None, needs_index_verification=True, probe=True
        )
        publish_event(tmp_path, event)
        with jobs.lease(conn) as lease:
            current = RepairProtocol(tmp_path).prepare(lease, read_events(tmp_path))
        assert current.phase == "required"
        assert current.trigger_id == event.event_id
        assert len(current.acknowledged_events) == 1
        assert current.acknowledged_events[0].event_id == event.event_id
