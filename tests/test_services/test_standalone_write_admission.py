"""A standalone library write or explicit reset cannot bypass a recovery hold."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from recall.db.connection import connect
from recall.db.repair import SchemaFence, load_repair_row
from recall.services import indexer
from recall.services.repair_events import publish_event, read_events
from recall.services.self_repair import (
    DaemonStartupRefused,
    marker_path,
    refusal_marker_path,
    remember_fatal_failure,
    run_startup_self_repair,
)

from tests.test_services.test_rpc_owned_boundary import server_at


def test_confirmed_cli_reset_retains_refusal_and_returns_exit_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.cli import rpc
    from recall.cli.app import app
    from typer.testing import CliRunner

    server = server_at(tmp_path)
    config = server._config
    with connect(config) as conn:
        conn.execute("CREATE TABLE witness(k INT)")
        conn.execute("INSERT INTO witness VALUES (17)")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            f"[sources.{source}]\nroots = []"
            for source in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        )
    )
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RECALL_DB_PATH", str(config.db_path))
    monkeypatch.setenv("RECALL_LOCK_PATH", str(config.lock_path))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(rpc, "_in_process_server", None)
    refusal = refusal_marker_path(tmp_path)
    refusal.write_text("retained armed interruption")
    try:
        result = CliRunner().invoke(app, ["index", "--recreate", "--yes", "--json"])
        assert result.exit_code == 3, result.output
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "RUNTIME"
        assert "retained armed interruption" in error["message"]
        assert refusal.read_text() == "retained armed interruption"
        assert not (tmp_path / "snapshots").exists()
        assert not (tmp_path / "recall.pid").exists()
    finally:
        asyncio.run(server.stop())
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == [(17,)]


def test_invalid_recovery_state_cannot_admit_a_standalone_writer(tmp_path: Path) -> None:
    server = server_at(tmp_path)
    with connect(server._config) as conn:
        conn.execute("CREATE TABLE witness(k INT)")
        conn.execute("INSERT INTO witness VALUES (17)")
        conn.execute("UPDATE daemon_index_repair SET acknowledged_events = '{}'::JSON")
    try:
        with pytest.raises(DaemonStartupRefused, match="recovery state is unavailable"):
            indexer.index_sessions(
                source=None,
                full=False,
                recreate=False,
                verbose=False,
                config=server._config,
                workers=1,
            )
    finally:
        asyncio.run(server.stop())
    with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == [(17,)]
        assert fresh.execute("SELECT acknowledged_events FROM daemon_index_repair").fetchone() == (
            "{}",
        )


def test_failed_reset_rolls_back_the_fence_and_retains_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import recall.db.schema as schema

    server = server_at(tmp_path)
    with connect(server._config) as conn:
        conn.execute("CREATE TABLE witness(k INT)")
        conn.execute("INSERT INTO witness VALUES (17)")
    original = duckdb.FatalException("Failed to delete all rows from index during reset")

    def fail(*args):
        raise original

    monkeypatch.setattr(schema, "_apply_schema", fail)
    try:
        with pytest.raises(duckdb.FatalException) as result:
            server._reset_database(server._config)
        assert result.value is original
        assert server._recreate_backup is not None
        backup = Path(server._recreate_backup)
        assert json.loads((backup / "manifest.json").read_text())["complete"] is True
        assert server._shutdown_event.is_set()
    finally:
        asyncio.run(server.stop())
    with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == [(17,)]
        assert fresh.execute("SELECT max(version) FROM schema_version").fetchone() == (41,)
        assert (load_repair_row(fresh).phase, load_repair_row(fresh).attempt_no) == ("idle", 0)
    with duckdb.connect(str(backup / "recall.duckdb"), read_only=True) as saved:
        assert saved.execute("SELECT * FROM witness").fetchall() == [(17,)]


def test_explicit_settled_reset_retains_matched_recovery_set_and_atomic_fence(
    tmp_path: Path,
) -> None:
    server = server_at(tmp_path)
    config = replace(server._config, config_path=tmp_path / "config.toml")
    config.config_path.write_text("[sources.claude_code]\nroots = []\n")
    with connect(config) as conn:
        conn.execute("CREATE TABLE witness(k INT)")
        conn.execute("INSERT INTO witness VALUES (17)")
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index"),
        site="source_commit",
        fence=SchemaFence(41),
    )
    event = read_events(tmp_path).primary
    assert event is not None
    run_startup_self_repair(config)
    publish_event(tmp_path, event)
    original_marker = marker_path(tmp_path).read_bytes()
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        before = load_repair_row(fresh)
    assert before.oracle_ref is not None
    oracle_files = {
        path.name: path.read_bytes() for path in (tmp_path / before.oracle_ref).iterdir()
    }
    try:
        backup = Path(server._reset_database(config))
    finally:
        asyncio.run(server.stop())

    assert (backup / "config.toml").read_bytes() == config.config_path.read_bytes()
    assert (backup / "daemon-failure.json").read_bytes() == original_marker
    for name, content in oracle_files.items():
        assert (backup / before.oracle_ref / name).read_bytes() == content
    manifest = json.loads((backup / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["recovery"]["schema_version"] == 41
    with duckdb.connect(str(backup / "recall.duckdb"), read_only=True) as saved:
        assert load_repair_row(saved) == before
        assert saved.execute("SELECT * FROM witness").fetchall() == [(17,)]
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        row = load_repair_row(fresh)
        assert (row.phase, row.attempt_no) == ("idle", 0)
        assert row.success_since_attempt is True and row.success_since_failure is True
        assert fresh.execute("SELECT max(version) FROM schema_version").fetchone() == (41,)
        assert fresh.execute(
            "SELECT migration_id FROM schema_migrations "
            "WHERE migration_id = '0041_daemon_index_repair'"
        ).fetchone() == ("0041_daemon_index_repair",)
    assert read_events(tmp_path).primary is None
    assert not refusal_marker_path(tmp_path).exists()


@pytest.mark.parametrize("mode", ["standalone", "recreate", "borrowed", "rpc_reset"])
def test_retained_refusal_prevents_supported_write_or_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    server = server_at(tmp_path)
    config = server._config
    with connect(config) as conn:
        conn.execute("CREATE TABLE witness(k INT)")
        conn.execute("INSERT INTO witness VALUES (17)")
    if mode == "rpc_reset":
        server._database.call("fixture", lambda conn: conn.execute("SELECT 1").fetchone())
    refusal = refusal_marker_path(tmp_path)
    refusal.write_text("retained armed interruption")
    original = refusal.read_bytes()
    monkeypatch.setattr(indexer, "index_sessions_on", lambda *args, **kwargs: "unexpected write")
    borrowed = duckdb.connect(str(config.db_path)) if mode == "borrowed" else None
    try:
        with pytest.raises(DaemonStartupRefused):
            if mode == "rpc_reset":
                server._reset_database(config)
            else:
                indexer.index_sessions(
                    source=None,
                    full=False,
                    recreate=mode == "recreate",
                    verbose=False,
                    config=config,
                    workers=1,
                    conn=borrowed,
                )
        assert refusal.read_bytes() == original
    finally:
        if borrowed is not None:
            borrowed.close()
        asyncio.run(server.stop())
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == [(17,)]
