"""Migration policy cannot consume an unobserved originating fatal before folding M1."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall.db import migrations
from recall.db.migrations import Migration
from recall.services.repair_events import read_events
from recall.services.self_repair import marker_path, remember_fatal_failure

from tests.test_services.test_rpc_owned_boundary import server_at


def test_owned_open_observes_migration_fatal_before_fail_soft_translation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = server_at(tmp_path)
    with duckdb.connect(str(server._config.db_path)) as conn:
        conn.execute("CREATE TABLE schema_version(version INTEGER)")
        conn.execute("INSERT INTO schema_version VALUES (36)")
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index in retained M1"),
        site="first",
    )
    original = marker_path(tmp_path).read_bytes()

    def fail(_lease: object) -> bool:
        raise duckdb.FatalException("Failed to delete all rows from index in migration M2")

    monkeypatch.setattr(
        migrations,
        "_MIGRATIONS",
        [Migration("0041_daemon_index_repair", 41, fail)],
    )
    try:
        with pytest.raises(duckdb.FatalException, match="migration M2"):
            server._database.call("startup_open", lambda conn: None)
        assert marker_path(tmp_path).read_bytes() == original
        batch = read_events(tmp_path)
        assert len(batch.secondary) == 1
        assert batch.secondary[0][1].failure is not None
        assert batch.secondary[0][1].failure.message == (
            "Failed to delete all rows from index in migration M2"
        )
        assert server._jobs.first_fatal is not None
        assert server._shutdown_event.is_set()
    finally:
        asyncio.run(server.stop())


@pytest.mark.parametrize(
    "stage",
    [
        "schema_precheck:open",
        "migration:0029:alter",
        "migration_audit:0029_fatal_timestamp",
        "postmigration_open",
    ],
)
def test_owned_open_keeps_m1_and_publishes_each_pre_fold_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    server = server_at(tmp_path)
    if stage != "postmigration_open":
        with duckdb.connect(str(server._config.db_path)) as conn:
            conn.execute("CREATE TABLE schema_version(version INTEGER)")
            conn.execute("INSERT INTO schema_version VALUES (28)")
            conn.execute("CREATE TABLE runtime_state(singleton BOOLEAN)")
    historical = next(m for m in migrations.get_migrations() if m.target_version == 29)
    monkeypatch.setattr(migrations, "_MIGRATIONS", [historical])
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index in retained M1"),
        site="first",
    )
    original = marker_path(tmp_path).read_bytes()
    observed_call = server._jobs.call_db
    error = duckdb.FatalException("Failed to delete all rows from index during pre-fold M2")

    def injected(operation: str, work: Callable[[], Any]) -> Any:
        if operation == stage:

            def fail() -> Any:
                raise error

            return observed_call(operation, fail)
        return observed_call(operation, work)

    monkeypatch.setattr(server._jobs, "call_db", injected)
    try:
        with pytest.raises(duckdb.FatalException, match="pre-fold M2"):
            server._database.call("startup_open", lambda conn: None)
        assert marker_path(tmp_path).read_bytes() == original
        secondary = read_events(tmp_path).secondary
        assert len(secondary) == 1
        failure = secondary[0][1].failure
        assert failure is not None
        assert (failure.site, failure.message) == (stage, str(error))
        assert server._shutdown_event.is_set()
        with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
            if stage == "migration_audit:0029_fatal_timestamp":
                assert fresh.execute("SELECT MAX(version) FROM schema_version").fetchone() == (29,)
                assert fresh.execute("SELECT * FROM schema_migrations").fetchall() == []
            elif stage != "postmigration_open":
                assert fresh.execute("SELECT MAX(version) FROM schema_version").fetchone() == (28,)
    finally:
        asyncio.run(server.stop())


def test_migration_primary_survives_a_separate_close_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = server_at(tmp_path)
    with duckdb.connect(str(server._config.db_path)) as conn:
        conn.execute("CREATE TABLE schema_version(version INTEGER)")
        conn.execute("INSERT INTO schema_version VALUES (36)")
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index in retained M1"),
        site="first",
    )
    original = marker_path(tmp_path).read_bytes()
    primary = duckdb.FatalException("Failed to delete all rows from index in migration M2")
    cleanup = duckdb.FatalException("database has been invalidated during close M3")

    def fail(_lease: object) -> bool:
        raise primary

    monkeypatch.setattr(
        migrations, "_MIGRATIONS", [Migration("0041_daemon_index_repair", 41, fail)]
    )
    cleanup_call = server._jobs.cleanup_db

    def injected(operation: str, work: Callable[[], Any]) -> Any:
        if operation == "database_schema_open:close":

            def close_then_fail() -> Any:
                work()
                raise cleanup

            return cleanup_call(operation, close_then_fail)
        return cleanup_call(operation, work)

    monkeypatch.setattr(server._jobs, "cleanup_db", injected)
    try:
        with pytest.raises(BaseExceptionGroup, match="operation and close failed"):
            server._database.call("startup_open", lambda conn: None)
        assert marker_path(tmp_path).read_bytes() == original
        messages = {
            event.failure.message for _, event in read_events(tmp_path).secondary if event.failure
        }
        assert messages == {str(primary), str(cleanup)}
        assert server._jobs.first_fatal is not None
        assert server._jobs.first_fatal.graph.failure is primary
        with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
            assert fresh.execute("SELECT MAX(version) FROM schema_version").fetchone() == (36,)
    finally:
        asyncio.run(server.stop())


def test_recovery_migration_preserves_primary_when_rollback_reporting_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.db.lease import StandaloneLifetime
    from recall.db.schema import ensure_schema

    database = tmp_path / "recall.duckdb"
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        conn.execute("DROP TABLE daemon_index_repair")
        conn.execute("DELETE FROM schema_version WHERE version>36")
        conn.execute("INSERT INTO schema_version(version) VALUES (36)")
        conn.execute("DELETE FROM schema_migrations WHERE migration_id='0041_daemon_index_repair'")
    lifetime = StandaloneLifetime()
    original_call = lifetime.call_db
    original_cleanup = lifetime.cleanup_db
    primary = ValueError("migration publication interrupted")
    cleanup = OSError("rollback acknowledgement interrupted")
    primary_raised = False

    def call(operation, work):
        nonlocal primary_raised
        result = original_call(operation, work)
        if operation == "repair:audit":
            primary_raised = True
            raise primary
        return result

    def rollback(operation, work):
        result = original_cleanup(operation, work)
        if primary_raised and operation == "migration:0041:rollback":
            raise cleanup
        return result

    monkeypatch.setattr(lifetime, "call_db", call)
    monkeypatch.setattr(lifetime, "cleanup_db", rollback)
    upgrade = next(m for m in migrations.get_migrations() if m.target_version == 41)
    with duckdb.connect(str(database)) as conn, lifetime.lease(conn) as lease:
        with pytest.raises(BaseExceptionGroup) as caught:
            upgrade.upgrade(lease)
        assert caught.value.subgroup(lambda error: error is primary) is not None
        assert caught.value.subgroup(lambda error: error is cleanup) is not None
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute("SELECT max(version) FROM schema_version").fetchone() == (36,)
        assert fresh.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name='daemon_index_repair'"
        ).fetchone() == (0,)
