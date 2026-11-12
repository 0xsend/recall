"""Ordinary compaction cannot grant a recovery release or reuse a failed backup."""

import asyncio
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.core.config import CompactionConfig
from recall.db.connection import connect
from recall.db.repair import SchemaFence, load_repair_row
from recall.services import compaction, daemon
from recall.services.repair_events import (
    RecoveryEvent,
    import_legacy_events,
    publish_event,
    read_events,
)
from recall.services.self_repair import marker_path, remember_fatal_failure, run_startup_self_repair

from tests.test_services.test_rpc_owned_boundary import server_at


@pytest.mark.parametrize("hold", ["refusal", "full_spool"])
def test_preserving_compact_checks_files_before_any_engine_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hold: str
) -> None:
    from recall.core.repair_state import PublicationRole
    from recall.services.self_repair import DaemonStartupRefused, refusal_marker_path

    server = server_at(tmp_path)
    server._config.db_path.write_bytes(b"preserved unopened database")
    if hold == "refusal":
        refusal_marker_path(tmp_path).write_text("explicit offline recovery required")
    else:
        for identity in range(1, 65):
            publish_event(
                tmp_path,
                RecoveryEvent(
                    event_id=UUID(int=identity), failure=None, needs_index_verification=True
                ),
                role=PublicationRole.SECONDARY,
            )

    def unopened(*args, **kwargs):
        pytest.fail("held ordinary maintenance cannot open the engine")

    monkeypatch.setattr(duckdb, "connect", unopened)
    try:
        with pytest.raises(DaemonStartupRefused):
            compaction.compact(server._config)
        assert server._config.db_path.read_bytes() == b"preserved unopened database"
    finally:
        asyncio.run(server.stop())


@pytest.mark.parametrize("site", ["estimate", "copy", "fresh_verification"])
def test_automatic_private_failure_is_owned_before_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    server = server_at(tmp_path)
    config = replace(
        server._config,
        compaction=CompactionConfig(min_bytes=0, bloat_ratio_threshold=0.0, check_interval_hours=0),
    )
    server._database.call("fixture", lambda conn: conn.execute("SELECT 1").fetchone())
    original = duckdb.FatalException(f"Failed to delete all rows from index in {site}")

    def fail(*args):
        raise original

    target = {
        "estimate": "_estimate_live_bytes",
        "copy": "_verify_table_count",
        "fresh_verification": "_verify_compact_database",
    }[site]
    monkeypatch.setattr(compaction, target, fail)
    try:
        with pytest.raises(BaseExceptionGroup):
            daemon._maybe_run_auto_compact(server, config)
        observation = server._jobs.first_fatal
        assert observation is not None and observation.graph.failure is original
        batch = read_events(tmp_path)
        assert batch.primary is not None and batch.primary.failure is not None
        assert batch.primary.failure.message == str(original)
        assert server._shutdown_event.is_set()
    finally:
        asyncio.run(server.stop())


def test_ordinary_compact_preserves_spent_state_diagnostics_and_replay(tmp_path: Path) -> None:
    server = server_at(tmp_path)
    config = server._config
    with connect(config) as conn:
        conn.execute("CREATE TABLE witness(k INT, value TEXT)")
        conn.execute("INSERT INTO witness VALUES (1, 'same'), (1, 'same'), (2, NULL)")
        conn.execute("CREATE INDEX witness_non_schema ON witness(value)")
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index"),
        site="source_commit",
    )
    event = import_legacy_events(tmp_path, SchemaFence(41)).primary
    assert event is not None
    run_startup_self_repair(config)
    publish_event(tmp_path, event)
    original = marker_path(tmp_path).read_bytes()
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        before = load_repair_row(fresh)
        history = fresh.execute("SELECT * FROM runtime_state").fetchall()

    result = compaction.compact(config)

    assert result.replaced is True
    assert marker_path(tmp_path).read_bytes() == original
    assert read_events(tmp_path).primary == event
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        assert load_repair_row(fresh) == before
        assert fresh.execute("SELECT * FROM runtime_state").fetchall() == history
        assert fresh.execute("SELECT * FROM witness ORDER BY k").fetchall() == [
            (1, "same"),
            (1, "same"),
            (2, None),
        ]
        assert fresh.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE index_name = 'witness_non_schema'"
        ).fetchall() == [("witness_non_schema",)]
    asyncio.run(server.stop())


def test_previous_failed_backup_is_not_reused_or_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = server_at(tmp_path)
    config = server._config
    with connect(config):
        pass
    prior = Path(str(config.db_path) + ".pre-compact")
    prior.write_bytes(b"retained matched backup from a failed replacement")

    def fail(*args):
        raise OSError("replacement unavailable")

    monkeypatch.setattr(compaction, "_replace_database_files", fail)
    with pytest.raises(compaction.CompactionError):
        compaction.compact(config)
    assert prior.read_bytes() == b"retained matched backup from a failed replacement"
    asyncio.run(server.stop())
