"""Every owning configured file flow publishes its selected native failure."""

from pathlib import Path

import duckdb
import pytest
from recall.services.repair_events import RecoveryEvent, read_events

from tests.test_services.test_rpc_owned_boundary import server_at


@pytest.mark.parametrize("operation", ["compact", "startup"])
def test_public_offline_file_flow_retains_original_native_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from recall.db.connection import connect
    from recall.services import compaction
    from recall.services.repair_protocol import RepairProtocol
    from recall.services.self_repair import DaemonStartupRefused, run_startup_self_repair

    config = server_at(tmp_path)._config
    with connect(config) as conn:
        conn.execute("CREATE TABLE witness(k INT)")
        conn.execute("INSERT INTO witness VALUES (17)")
    origin = duckdb.FatalException("Failed to delete all rows from index in offline operation")

    def failed(*args, **kwargs):
        raise origin

    if operation == "compact":
        monkeypatch.setattr(compaction, "_copy_database", failed)
        with pytest.raises(compaction.CompactionError) as caught:
            compaction.compact(config)
    else:
        monkeypatch.setattr(RepairProtocol, "begin_start", failed)
        with pytest.raises(DaemonStartupRefused) as caught:
            run_startup_self_repair(config)
    assert caught.value.__cause__ is origin
    primary = read_events(tmp_path).primary
    assert isinstance(primary, RecoveryEvent) and primary.failure is not None
    assert primary.failure.message == str(origin)
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == [(17,)]
        assert fresh.execute("SELECT phase,attempt_no FROM daemon_index_repair").fetchone() == (
            "idle",
            0,
        )


def test_offline_sidecar_optimization_observes_disk_failure_before_fail_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from recall.db import fts_sidecar
    from recall.db.connection import connect
    from recall.services.compaction import compact

    config = server_at(tmp_path)._config
    with connect(config) as conn:
        conn.execute("CREATE TABLE witness(k INT)")
        conn.execute("INSERT INTO witness VALUES (17)")

    sidecar = fts_sidecar.open_sidecar(fts_sidecar.sidecar_path(config.data_dir))
    sidecar.close()
    attempts = 0

    def failed(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise sqlite3.OperationalError("No space left on device")

    monkeypatch.setattr(fts_sidecar, "optimize_sidecar", failed)
    result = compact(config)
    assert result.replaced
    assert attempts == 1
    batch = read_events(tmp_path)
    assert batch.primary is None
    assert len(batch.secondary) == 1
    event = batch.secondary[0][1]
    assert isinstance(event, RecoveryEvent)
    assert event.failure is None and event.needs_index_verification and not event.probe
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == [(17,)]
