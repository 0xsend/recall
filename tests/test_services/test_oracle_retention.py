"""Exclusive crash recovery keeps pinned references and a bounded replacement."""

import errno
import signal
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.db.lease import StandaloneLifetime
from recall.db.repair import load_repair_row, save_repair_row
from recall.db.repair_oracle import OracleUnavailable, capture_oracle
from recall.db.schema import ensure_schema
from recall.services.owned_jobs import DaemonJobs
from recall.services.repair_events import RecoveryEvent, publish_event, read_events
from recall.services.repair_oracles import OracleStore
from recall.services.repair_protocol import RepairProtocol

FIRST = UUID(int=1)
SECOND = UUID(int=2)


def _event(root: Path, identity: UUID) -> None:
    publish_event(
        root,
        RecoveryEvent(event_id=identity, failure=None, needs_index_verification=True, probe=True),
    )


def _settled(root: Path) -> None:
    _event(root, FIRST)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    protocol = RepairProtocol(root)
    with duckdb.connect(str(root / "recall.duckdb")) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE witness (id INTEGER, payload VARCHAR)")
        conn.execute("INSERT INTO witness VALUES (1, 'acknowledged'), (1, 'acknowledged')")
        conn.execute("CREATE INDEX witness_idx ON witness(id)")
        with jobs.lease(conn) as lease:
            protocol.prepare(lease, read_events(root))
            rebuilt = protocol.rebuild(lease, protocol.arm(lease))
    with duckdb.connect(str(root / "recall.duckdb")) as fresh:
        with jobs.lease(fresh) as lease:
            row = protocol.verify(lease, rebuilt)
            protocol.retire(lease)
        # Fixture-only later qualified completion, not a completion rehearsal.
        save_repair_row(
            fresh,
            row.model_copy(update={"success_since_attempt": True, "success_since_failure": True}),
        )


@pytest.mark.parametrize("error_number", [errno.EIO, errno.ENOSPC])
@pytest.mark.parametrize(
    "stage", ["before_required_commit", "after_required_commit", "before_oracle_retirement"]
)
def test_replacement_fault_never_deletes_the_durable_pin(
    tmp_path: Path, stage: str, error_number: int
) -> None:
    _settled(tmp_path)
    _event(tmp_path, SECOND)
    old = tmp_path / "repair-oracles" / str(FIRST)
    original = (old / "manifest.json").read_bytes()
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)

    def fail(at: str) -> None:
        if at == stage:
            assert (old / "manifest.json").read_bytes() == original
            assert len(tuple((tmp_path / "repair-oracles").iterdir())) == 2
            raise OSError(error_number, "injected storage error")

    with (
        duckdb.connect(str(tmp_path / "recall.duckdb")) as conn,
        jobs.lease(conn) as lease,
        pytest.raises(OSError, match="injected storage"),
    ):
        RepairProtocol(tmp_path, barrier=fail).prepare(lease, read_events(tmp_path))
    with duckdb.connect(str(tmp_path / "recall.duckdb")) as fresh:
        row = load_repair_row(fresh)
        assert row.trigger_id == (FIRST if stage == "before_required_commit" else SECOND)
        assert row.oracle_ref is not None
        assert (tmp_path / row.oracle_ref / "manifest.json").is_file()
        with jobs.lease(fresh) as lease:
            recovered = RepairProtocol(tmp_path).prepare(lease, read_events(tmp_path))
        assert recovered.phase == "required"
        assert recovered.trigger_id == SECOND
        assert fresh.execute("SELECT * FROM witness").fetchall() == [
            (1, "acknowledged"),
            (1, "acknowledged"),
        ]
    assert not old.exists()
    assert {path.name for path in (tmp_path / "repair-oracles").iterdir()} == {str(SECOND)}


_KILL_CAPTURE = """
import os,signal,sys
from pathlib import Path
from uuid import UUID
import duckdb
from recall.db.lease import StandaloneLifetime
from recall.db.repair_oracle import capture_oracle
from recall.db.schema import ensure_schema
root=Path(sys.argv[1])
def barrier(stage):
    if stage==sys.argv[2]:
        os.kill(os.getpid(),signal.SIGKILL)
with duckdb.connect(str(root/'recall.duckdb')) as conn:
    ensure_schema(conn)
    conn.execute("CREATE TABLE witness(id INTEGER, payload VARCHAR)")
    conn.execute("INSERT INTO witness VALUES (1,'durable'),(1,'durable')")
    capture_oracle(conn,root/'repair-oracles'/str(UUID(int=1)),UUID(int=1),
        sidecar=root/'recall.fts.sqlite',barrier=barrier,observer=StandaloneLifetime())
"""


@pytest.mark.parametrize(
    "stage", ["capture_synced", "capture_published", "capture_directory_synced"]
)
def test_killed_precommit_capture_is_recovered_without_accumulating_oracles(
    tmp_path: Path, stage: str
) -> None:
    _event(tmp_path, FIRST)
    child = subprocess.run(
        [sys.executable, "-c", _KILL_CAPTURE, str(tmp_path), stage],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == -signal.SIGKILL, child.stderr
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(tmp_path / "recall.duckdb")) as fresh:
        with jobs.lease(fresh) as lease:
            row = RepairProtocol(tmp_path).prepare(lease, read_events(tmp_path))
        assert row.phase == "required"
        assert fresh.execute("SELECT * FROM witness").fetchall() == [(1, "durable"), (1, "durable")]
    assert {path.name for path in (tmp_path / "repair-oracles").iterdir()} == {str(FIRST)}


def test_unknown_workspace_content_holds_without_removing_the_pin(tmp_path: Path) -> None:
    _settled(tmp_path)
    workspace = tmp_path / "repair-oracles" / f".capture-{SECOND}"
    workspace.mkdir()
    unknown = workspace / "operator-notes"
    unknown.write_text("retain")
    with duckdb.connect(str(tmp_path / "recall.duckdb")) as conn:
        row = load_repair_row(conn)
        with pytest.raises(OracleUnavailable, match="unknown file"):
            OracleStore(tmp_path).reconcile(row, SECOND)
    assert unknown.read_text() == "retain"
    assert (tmp_path / "repair-oracles" / str(FIRST) / "manifest.json").is_file()


def test_a_pinned_running_oracle_is_not_replaced_by_damaged_current_data(tmp_path: Path) -> None:
    _event(tmp_path, FIRST)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(tmp_path / "recall.duckdb")) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE witness(id INTEGER)")
        conn.execute("INSERT INTO witness VALUES (1), (1)")
        protocol = RepairProtocol(tmp_path)
        with jobs.lease(conn) as lease:
            row = protocol.prepare(lease, read_events(tmp_path))
            protocol.arm(lease)
        assert row.oracle_ref is not None
        saved = tmp_path / row.oracle_ref / "table-0.rows"
        original = saved.read_bytes()
        conn.execute("DELETE FROM witness")
        OracleStore(tmp_path).reconcile(load_repair_row(conn))
        assert saved.read_bytes() == original
        assert conn.execute("SELECT phase FROM daemon_index_repair").fetchone() == ("running",)


def test_unreferenced_complete_capture_is_discarded_for_a_different_trigger(tmp_path: Path) -> None:
    root = tmp_path / "repair-oracles"
    with duckdb.connect(str(tmp_path / "recall.duckdb")) as conn:
        ensure_schema(conn)
        capture_oracle(conn, root / str(FIRST), FIRST, observer=StandaloneLifetime())
        OracleStore(tmp_path).reconcile(load_repair_row(conn), SECOND)
    assert not (root / str(FIRST)).exists()


def test_interrupted_retirement_resumes_after_missing_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _settled(tmp_path)
    _event(tmp_path, SECOND)
    old = tmp_path / "repair-oracles" / str(FIRST)
    original = Path.unlink
    removed: list[str] = []

    def fail(path: Path, *args, **kwargs):
        if path.parent == old and path.name.startswith("table-"):
            if removed:
                raise OSError(errno.EIO, "retirement interrupted")
            removed.append(path.name)
        return original(path, *args, **kwargs)

    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail)
        with (
            duckdb.connect(str(tmp_path / "recall.duckdb")) as conn,
            jobs.lease(conn) as lease,
            pytest.raises(OSError, match="retirement interrupted"),
        ):
            RepairProtocol(tmp_path).prepare(lease, read_events(tmp_path))
    assert len(removed) == 1
    assert (old / "manifest.json").is_file()
    with duckdb.connect(str(tmp_path / "recall.duckdb")) as fresh:
        saved = load_repair_row(fresh)
        assert saved.phase == "required" and saved.trigger_id == SECOND
        with jobs.lease(fresh) as lease:
            RepairProtocol(tmp_path).prepare(lease, read_events(tmp_path))
    assert not old.exists()
    assert (tmp_path / "repair-oracles" / str(SECOND) / "manifest.json").is_file()
