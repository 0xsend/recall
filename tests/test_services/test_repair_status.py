"""`daemon status` and the CLI say what an automatic index repair is doing, without a daemon."""

import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.core.repair_progress import PROGRESS_FILENAME, Phase, read_progress
from recall.core.rpc_client import RpcClient
from recall.core.rpc_types import RpcConnectionError
from recall.db.lease import StandaloneLifetime
from recall.db.ownership import offline_ownership, offline_writer_ownership, read_offline
from recall.services import daemon as daemon_service
from recall.services.daemon import offline_daemon_status
from recall.services.index_migration import _create_backup
from recall.services.repair_progress import ProgressPublisher, repair_in_progress_notice
from recall.services.self_repair import REFUSAL_FILENAME

from tests.test_services.test_repair_process import _child, _fixture

REASON_PREFIX = "database ownership busy at"


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, _, environment = _fixture(tmp_path, probe=False)
    for name in ("RECALL_CONFIG_PATH", "RECALL_DATA_DIR", "RECALL_DB_PATH", "RECALL_LOCK_PATH"):
        monkeypatch.setenv(name, environment[name])
    return AppConfig.load(), environment


@contextmanager
def _owner(config: AppConfig, *, phase: Phase = "capturing_reference", age: float = 0.0):
    """A live owner: it holds the gate and publishes a record updated `age` seconds ago."""
    clock = [time.time() - age]
    publisher = ProgressPublisher(config.data_dir, clock=lambda: clock[0], interval_seconds=3600)
    with offline_ownership(config.db_path):
        publisher.begin("index", 1)
        publisher.phase(phase)
        publisher.tables_planned(10)
        publisher.table_finished("main.messages", _stats())
        _wait_until_published(config, phase)
        try:
            yield publisher
        finally:
            publisher.end()


def _wait_until_published(config: AppConfig, phase: str) -> None:
    """The writer thread publishes asynchronously; wait for the record to show the update."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = read_progress(config.data_dir)
        if (
            record is not None
            and record.phase == phase
            and record.progress is not None
            and record.progress.done == 1
        ):
            return
        time.sleep(0.01)
    raise AssertionError("the progress record never showed the update")


def _write_record(config: AppConfig) -> None:
    (config.data_dir / PROGRESS_FILENAME).write_text(
        '{"schema":1,"owner_pid":1,"attempt_no":1,"trigger_kind":"index",'
        '"phase":"admitting","started_at":1.0,"phase_started_at":1.0,"updated_at":1.0}'
    )


def _stats():
    from recall.db.oracle_sort import TableSortStats

    return TableSortStats(runs=1, merge_levels=0, bytes_merged=0, stream_bytes=4096)


def _status(config: AppConfig):
    status, _ = offline_daemon_status(config, limit=1, cursor=None)
    return status


def test_a_fresh_record_under_a_held_gate_is_a_running_repair(scratch) -> None:
    config, _ = scratch
    with _owner(config):
        status = _status(config)
    assert status.daemon_pid is None
    assert status.recovery is not None
    assert status.recovery["state"] == "running"
    assert status.recovery["phase"] == "capturing_reference"
    assert status.recovery["progress"] == {
        "unit": "table",
        "done": 1,
        "total": 10,
        "current": None,
        "bytes_done": 4096,
    }
    assert status.runtime_unavailable_reason is not None
    assert status.runtime_unavailable_reason.startswith(REASON_PREFIX)
    assert "automatic index repair in progress" in status.runtime_unavailable_reason
    assert "data requests are unavailable until the repair finishes" in status.recovery["message"]


def test_a_record_that_stopped_updating_is_a_stalled_repair(scratch) -> None:
    config, _ = scratch
    with _owner(config, age=60):
        status = _status(config)
    assert status.recovery is not None and status.recovery["state"] == "stalled"
    assert "no update for" in status.recovery["message"]
    assert status.runtime_unavailable_reason is not None
    assert status.runtime_unavailable_reason.startswith(REASON_PREFIX)
    assert status.daemon_pid is None


@pytest.mark.parametrize("damage", ["missing", "invalid"])
def test_a_missing_or_invalid_record_under_a_held_gate_changes_nothing(
    scratch, damage: str
) -> None:
    config, _ = scratch
    with offline_ownership(config.db_path):
        if damage == "invalid":
            (config.data_dir / PROGRESS_FILENAME).write_text("{")
        status = _status(config)
    assert status.recovery is None
    assert status.runtime_unavailable_reason is not None
    assert status.runtime_unavailable_reason.startswith(REASON_PREFIX)
    assert "repair" not in status.runtime_unavailable_reason


@pytest.mark.parametrize("held", [True, False], ids=["held-gate", "free-gate"])
def test_a_refusal_marker_wins_over_the_record(scratch, held: bool) -> None:
    config, _ = scratch
    (config.data_dir / REFUSAL_FILENAME).write_text("refusing to start: test\n")
    if held:
        with _owner(config):
            status = _status(config)
    else:
        _write_record(config)
        status = _status(config)
    assert status.recovery is None
    assert status.startup_refusal == "refusing to start: test"


@pytest.mark.parametrize(
    ("stage", "durable", "consequence"),
    [
        ("before_required_commit", "idle", "the next start recaptures the reference"),
        ("after_required_commit", "required", "the next start resumes from the saved reference"),
        (
            "after_arm_commit",
            "running",
            "the next start refuses and names `recall db rebuild-indexes`",
        ),
    ],
)
def test_a_record_left_by_a_killed_owner_names_what_the_next_start_does(
    scratch, stage: str, durable: str, consequence: str
) -> None:
    config, environment = scratch
    killed = _child(environment, "repair_with_progress", stage, "kill")
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    assert read_progress(config.data_dir) is not None
    status = _status(config)
    assert status.daemon_pid is None
    assert status.runtime_unavailable_reason is None
    assert status.recovery is not None
    assert status.recovery["state"] == "interrupted"
    assert status.recovery["durable_phase"] == durable
    assert consequence in status.recovery["message"]
    assert "recorded by pid" in status.recovery["message"]


@contextmanager
def _holder(environment: dict[str, str], owner: str):
    """A process other than the repair, inside a long-lived offline owner, publishing nothing."""
    child = subprocess.Popen(
        [sys.executable, "-m", "tests.repair_process", "hold_owner", owner, "unused"],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdin is not None
        assert child.stdout.readline().strip() == "held", child.stderr and child.stderr.read()
        yield child
    finally:
        if child.stdin is not None:
            child.stdin.close()
        try:
            child.wait(timeout=30)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
        assert child.stdout is not None and child.stderr is not None
        child.stdout.close()
        child.stderr.close()
    assert child.returncode == 0


def _dead_owners_fresh_record(environment: dict[str, str], config: AppConfig, *, settled: bool):
    """SIGKILL an automatic repair after it armed, leaving a fresh record and no owner.

    `settled` first completes the repair by hand and then puts the record back, for owners
    that refuse to open a database with an unresolved repair instruction; the free-gate
    status then calls the record stale instead of interrupted.
    """
    killed = _child(environment, "repair_with_progress", "after_arm_commit", "kill")
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    record_path = config.data_dir / PROGRESS_FILENAME
    if settled:
        kept = record_path.read_bytes()
        finished = _child(environment, "manual_rebuild", "none", "none")
        assert finished.returncode == 0, finished.stderr
        record_path.write_bytes(kept)
    record = read_progress(config.data_dir)
    assert record is not None
    assert time.time() - record.updated_at < 10, "the dead owner's record is still fresh"
    shown = _status(config).recovery
    assert shown is not None
    assert shown["state"] == ("stale" if settled else "interrupted")


@pytest.mark.parametrize(
    ("owner", "settled"),
    [
        ("file_lifetime", True),
        ("write_lease", True),
        ("compaction", True),
        ("manual_rebuild", False),
        ("keyword_rebuild", True),
        ("supersede", True),
    ],
)
def test_an_offline_owner_beside_a_killed_repairs_record_is_busy_not_a_running_repair(
    scratch, owner: str, settled: bool
) -> None:
    config, environment = scratch
    _dead_owners_fresh_record(environment, config, settled=settled)

    with _holder(environment, owner):
        status = _status(config)
        notice = repair_in_progress_notice(config)
        shown = read_progress(config.data_dir)

    assert status.recovery is None
    assert status.runtime_unavailable_reason is not None
    assert status.runtime_unavailable_reason.startswith(REASON_PREFIX)
    assert "repair" not in status.runtime_unavailable_reason
    assert notice is None
    assert shown is None


def _try_owner(environment: dict[str, str], owner: str):
    return _child(environment, "try_owner", owner, "unused")


def _assert_interrupted_record_kept(config: AppConfig, kept: bytes) -> None:
    """The free-gate status still shows the dead repair and the record is untouched."""
    assert (config.data_dir / PROGRESS_FILENAME).read_bytes() == kept
    recovery = _status(config).recovery
    assert recovery is not None
    assert recovery["state"] == "interrupted"
    assert recovery["durable_phase"] == "running"
    assert "recall db rebuild-indexes" in recovery["message"]


@pytest.mark.parametrize("owner", ["write_lease", "keyword_rebuild", "supersede"])
def test_an_owner_refused_by_admission_leaves_the_interrupted_record(scratch, owner: str) -> None:
    config, environment = scratch
    _dead_owners_fresh_record(environment, config, settled=False)
    kept = (config.data_dir / PROGRESS_FILENAME).read_bytes()

    refused = _try_owner(environment, owner)

    assert refused.returncode == 3, refused.stderr
    assert "refused:" in refused.stderr or "cli error 3" in refused.stderr
    _assert_interrupted_record_kept(config, kept)


@pytest.mark.parametrize("owner", ["write_lease", "manual_rebuild", "keyword_rebuild", "supersede"])
def test_an_owner_that_finds_the_lock_held_leaves_the_interrupted_record(
    scratch, owner: str
) -> None:
    config, environment = scratch
    _dead_owners_fresh_record(environment, config, settled=False)
    kept = (config.data_dir / PROGRESS_FILENAME).read_bytes()

    with _holder(environment, "advisory_lock"):
        blocked = _try_owner(environment, owner)

    assert blocked.returncode == 2, blocked.stderr
    _assert_interrupted_record_kept(config, kept)


def _migration_backup(config: AppConfig, *, recorded: str | None) -> Path:
    """Record `recorded` as the index migration's backup and return a real backup's directory."""
    with duckdb.connect(str(config.db_path)) as conn:
        backup = _create_backup(
            conn, config, kind="storage-migration", observer=StandaloneLifetime()
        )
        conn.execute(
            "UPDATE index_migration_jobs SET backup_path = ? WHERE singleton",
            [recorded if recorded is not None else None],
        )
    return Path(backup)


def _lower_the_backup_schema(backup: Path) -> None:
    with duckdb.connect(str(backup / "recall.duckdb")) as old:
        old.execute("DROP TABLE daemon_index_repair")
        old.execute("DELETE FROM schema_version WHERE version>36")
        old.execute("INSERT INTO schema_version(version) VALUES (36)")
        old.execute("DELETE FROM schema_migrations WHERE migration_id='0041_daemon_index_repair'")


def _no_backup(config: AppConfig) -> str:
    return "no index-migration backup to restore"


def _missing_backup(config: AppConfig) -> str:
    _migration_backup(config, recorded=str(config.data_dir / "no-such-backup"))
    return "original migration backup is incomplete or missing"


def _backup_of_a_lower_schema(config: AppConfig) -> str:
    backup = _migration_backup(config, recorded=None)
    with duckdb.connect(str(config.db_path)) as conn:
        conn.execute(
            "UPDATE index_migration_jobs SET backup_path = ? WHERE singleton", [str(backup)]
        )
    _lower_the_backup_schema(backup)
    return "migration rollback cannot lower the logical schema"


@pytest.mark.parametrize(
    "arrange",
    [_no_backup, _missing_backup, _backup_of_a_lower_schema],
    ids=["no-backup", "backup-missing", "backup-of-a-lower-schema"],
)
def test_a_rollback_that_fails_its_checks_leaves_the_interrupted_record(scratch, arrange) -> None:
    config, environment = scratch
    _dead_owners_fresh_record(environment, config, settled=False)
    kept = (config.data_dir / PROGRESS_FILENAME).read_bytes()
    reason = arrange(config)

    failed = _try_owner(environment, "rollback")

    assert failed.returncode == 4, failed.stderr
    assert reason in failed.stderr
    _assert_interrupted_record_kept(config, kept)


def _wal_beside_the_database(config: AppConfig) -> Path:
    """The killed repair left a WAL, or one is put there so the manual repair has one to replay."""
    wal = Path(f"{config.db_path}.wal")
    if not wal.exists():
        wal.write_bytes(b"")
    return wal


@pytest.mark.parametrize(
    "owner",
    ["manual_rebuild_copying_the_wal", "manual_rebuild_from_wal"],
    ids=["copying", "opened"],
)
def test_a_manual_repair_that_opens_from_a_diagnosed_wal_shows_no_dead_repair_while_it_works(
    scratch, owner: str
) -> None:
    config, environment = scratch
    _dead_owners_fresh_record(environment, config, settled=False)
    _wal_beside_the_database(config)

    with _holder(environment, owner):
        status = _status(config)
        notice = repair_in_progress_notice(config)
        shown = read_progress(config.data_dir)

    assert status.recovery is None
    assert status.runtime_unavailable_reason is not None
    assert status.runtime_unavailable_reason.startswith(REASON_PREFIX)
    assert "index repair" not in status.runtime_unavailable_reason
    assert notice is None
    assert shown is None


@pytest.mark.parametrize(
    ("owner", "reason"),
    [
        (
            "manual_rebuild_from_wal_copy_refused",
            "durable state does not admit an explicit manual selection",
        ),
        (
            "manual_rebuild_from_wal_writer_refused",
            "manual writer differs from its diagnosed recovery state",
        ),
    ],
    ids=["copy", "writer"],
)
def test_a_manual_repair_refused_after_a_diagnosed_wal_leaves_the_record(
    scratch, owner: str, reason: str
) -> None:
    config, environment = scratch
    _dead_owners_fresh_record(environment, config, settled=False)
    kept = (config.data_dir / PROGRESS_FILENAME).read_bytes()
    _wal_beside_the_database(config)

    refused = _try_owner(environment, owner)

    assert refused.returncode == 3, refused.stderr
    assert reason in refused.stderr
    assert (config.data_dir / PROGRESS_FILENAME).read_bytes() == kept


def test_an_owner_that_completes_clears_the_dead_repairs_record(scratch) -> None:
    config, environment = scratch
    _dead_owners_fresh_record(environment, config, settled=True)

    finished = _try_owner(environment, "write_lease")

    assert finished.returncode == 0, finished.stderr
    assert read_progress(config.data_dir) is None
    assert _status(config).recovery is None
    assert sorted(path.name for path in config.data_dir.glob("repair-progress*")) == []


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_a_writer_scope_that_ends_in_any_exception_gives_the_record_back(
    scratch, failure: type[BaseException]
) -> None:
    config, environment = scratch
    _dead_owners_fresh_record(environment, config, settled=False)
    kept = (config.data_dir / PROGRESS_FILENAME).read_bytes()

    with pytest.raises(failure), offline_writer_ownership(config):
        assert read_progress(config.data_dir) is None
        raise failure

    _assert_interrupted_record_kept(config, kept)
    assert sorted(path.name for path in config.data_dir.glob("repair-progress*")) == [
        PROGRESS_FILENAME
    ]


def test_a_writer_scope_that_ends_normally_discards_the_record(scratch) -> None:
    config, environment = scratch
    _dead_owners_fresh_record(environment, config, settled=False)

    with offline_writer_ownership(config):
        assert read_progress(config.data_dir) is None

    assert sorted(path.name for path in config.data_dir.glob("repair-progress*")) == []


def test_a_killed_writer_leaves_the_record_to_a_free_gate_and_the_next_daemon_removes_it(
    scratch,
) -> None:
    config, environment = scratch
    _dead_owners_fresh_record(environment, config, settled=True)
    kept = (config.data_dir / PROGRESS_FILENAME).read_bytes()
    writer = subprocess.Popen(
        [sys.executable, "-m", "tests.repair_process", "hold_owner", "file_lifetime", "unused"],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with writer:
        assert writer.stdout is not None
        assert writer.stdout.readline().strip() == "held", writer.stderr and writer.stderr.read()
        assert _status(config).recovery is None
        writer.kill()
        writer.wait()

    recovery = _status(config).recovery
    assert recovery is not None and recovery["state"] == "stale"
    assert [path.name for path in config.data_dir.glob("repair-progress*")] == [
        "repair-progress.json.parked"
    ]
    assert (config.data_dir / "repair-progress.json.parked").read_bytes() == kept
    ProgressPublisher(config.data_dir)
    assert list(config.data_dir.glob("repair-progress*")) == []


def test_a_read_only_holder_leaves_the_interrupted_record_to_be_shown(scratch) -> None:
    config, environment = scratch
    killed = _child(environment, "repair_with_progress", "after_arm_commit", "kill")
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    assert read_offline(config, lambda _conn: None) is None
    recovery = _status(config).recovery
    assert recovery is not None and recovery["state"] == "interrupted"
    assert read_progress(config.data_dir) is not None


def test_a_record_that_outlived_a_finished_repair_is_stale(scratch) -> None:
    config, environment = scratch
    killed = _child(environment, "repair_with_progress", "before_required_commit", "kill")
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    kept = config.data_dir / PROGRESS_FILENAME
    shutil.copyfile(kept, config.data_dir.parent / "kept-progress.json")
    kept.unlink()
    restarted = _child(environment, "startup", "unused", "unused")
    assert restarted.returncode == 0, restarted.stderr
    shutil.copyfile(config.data_dir.parent / "kept-progress.json", kept)
    status = _status(config)
    assert status.recovery is not None and status.recovery["state"] == "stale"
    assert status.recovery["durable_phase"] == "verified"
    assert "start normally" in status.recovery["message"]


def test_without_a_record_a_free_gate_shows_nothing(scratch) -> None:
    config, _ = scratch
    status = _status(config)
    assert status.recovery is None


@pytest.fixture
def forks(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    spawned: list[list[str]] = []

    def spawn(data_dir, arguments):
        spawned.append(list(arguments))
        raise OSError("the test never starts a process")

    monkeypatch.setattr("recall.core.rpc_client.spawn_detached_recall", spawn)
    return spawned


def test_a_data_command_refuses_with_the_repair_instead_of_forking(scratch, forks) -> None:
    config, _ = scratch
    with _owner(config):
        client = RpcClient(config, repair_notice=repair_in_progress_notice)
        with pytest.raises(RpcConnectionError) as refused:
            client.connect()
    assert "automatic index repair in progress" in refused.value.message
    assert "data requests are unavailable until the repair finishes" in refused.value.message
    assert "recall daemon status" in refused.value.message
    assert forks == []


def test_live_mark_names_the_repair_instead_of_a_missing_socket(scratch, forks) -> None:
    config, _ = scratch
    with _owner(config):
        client = RpcClient(config, repair_notice=repair_in_progress_notice)
        with pytest.raises(RpcConnectionError) as refused:
            client.connect(auto_fork=False)
    assert "automatic index repair in progress" in refused.value.message
    assert "daemon not running" not in refused.value.message
    assert forks == []


def test_without_a_repair_live_mark_keeps_its_text(scratch, forks) -> None:
    config, _ = scratch
    client = RpcClient(config, repair_notice=repair_in_progress_notice)
    with pytest.raises(RpcConnectionError) as refused:
        client.connect(auto_fork=False)
    assert "daemon not running" in refused.value.message


def test_a_held_gate_without_a_record_still_forks_as_before(scratch, forks) -> None:
    config, _ = scratch
    with offline_ownership(config.db_path):
        client = RpcClient(config, repair_notice=repair_in_progress_notice)
        with pytest.raises(RpcConnectionError, match="failed to auto-fork"):
            client.connect()
    assert forks == [["daemon", "start", "--background"]]


def test_daemon_start_refuses_while_a_repair_holds_the_database(scratch, forks) -> None:
    from recall.cli.contract import CliError
    from recall.cli.daemon import refuse_start_while_repairing

    config, _ = scratch
    with _owner(config):
        with pytest.raises(CliError) as refused:
            refuse_start_while_repairing(config)
        assert refused.value.exit_code == 1
        assert "automatic index repair in progress" in refused.value.message
    assert forks == []


def test_a_service_start_reports_the_repair_instead_of_waiting_for_a_pid_file(scratch) -> None:
    from recall.core.types import SchedulerKind

    config, _ = scratch
    # The guard answers before the capability is used, so a stand-in is enough.
    capability = cast(
        Any, SimpleNamespace(record=SimpleNamespace(scheduler=SchedulerKind.SYSTEMD, service=None))
    )
    with _owner(config):
        result = daemon_service._start_service(
            config, capability, timeout=30.0, started_at=time.monotonic()
        )
    assert result.started is False
    assert result.message is not None
    assert "automatic index repair in progress" in result.message


def test_offline_rebuild_names_the_repair_beside_its_own_text(scratch) -> None:
    from recall.cli.db import DAEMON_HOLDS_DB_MESSAGE, _daemon_holds_message

    config, _ = scratch
    assert _daemon_holds_message(config) == DAEMON_HOLDS_DB_MESSAGE
    with _owner(config):
        message = _daemon_holds_message(config)
    assert message.startswith(DAEMON_HOLDS_DB_MESSAGE + "; ")
    assert "automatic index repair in progress" in message


def test_a_cli_data_call_refuses_with_the_repair_instead_of_forking(
    scratch, forks, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.cli.contract import CliError
    from recall.cli.rpc import rpc_call_or_error

    # The suite serves CLI calls in process; this test is about the socket client.
    monkeypatch.setattr("recall.cli.rpc._in_process_server", None)
    config, _ = scratch
    with _owner(config), pytest.raises(CliError) as refused:
        rpc_call_or_error("recall.daemon_status", {})
    assert "automatic index repair in progress" in refused.value.message
    assert forks == []


def test_a_cli_call_that_may_not_fork_names_the_repair(
    scratch, forks, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.cli.contract import CliError
    from recall.cli.rpc import rpc_call_or_error

    # The suite serves CLI calls in process; this test is about the socket client.
    monkeypatch.setattr("recall.cli.rpc._in_process_server", None)
    config, _ = scratch
    with _owner(config), pytest.raises(CliError) as refused:
        rpc_call_or_error("recall.daemon_status", {}, auto_fork=False)
    assert "automatic index repair in progress" in refused.value.message
    assert "daemon not running" not in refused.value.message
    assert forks == []


def test_a_fleet_query_names_the_repair_for_the_calling_daemon(scratch, forks, tmp_path) -> None:
    from recall.core.fleet import FleetConfig
    from recall.services.fleet_endpoints import FleetExecutor
    from recall.services.fleet_query import fleet_list

    config, _ = scratch
    with _owner(config):
        result = fleet_list(FleetExecutor(FleetConfig((), tmp_path / "empty.toml")))
    assert result.hosts_ok == 0
    assert result.hosts_failed == 1
    assert "automatic index repair in progress" in result.errors[0]["error"]
    assert "daemon not running" not in result.errors[0]["error"]
    assert forks == []
