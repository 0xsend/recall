"""A process interrupted at a fence effect leaves an atomic old or complete schema."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
from recall.db.schema import ensure_schema


@pytest.mark.parametrize("operation", ["upgrade", "fresh", "recreate"])
@pytest.mark.parametrize("fault", ["kill", "error", "fatal"])
@pytest.mark.parametrize("effect", ["create_table", "seed", "audit", "version", "commit"])
def test_fence_effects_are_atomic_across_real_process_loss(
    tmp_path: Path, operation: str, fault: str, effect: str
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    database = data / "recall.duckdb"
    if operation != "fresh":
        with duckdb.connect(str(database)) as conn:
            ensure_schema(conn)
            conn.execute("CREATE TABLE witness(k INT, payload VARCHAR)")
            conn.execute("INSERT INTO witness VALUES (7,'acknowledged'),(7,NULL)")
            if operation == "upgrade":
                conn.execute("DROP TABLE daemon_index_repair")
                conn.execute("DELETE FROM schema_version WHERE version>36")
                conn.execute("INSERT INTO schema_version(version) VALUES (36)")
                conn.execute(
                    "DELETE FROM schema_migrations WHERE migration_id='0041_daemon_index_repair'"
                )
    config = tmp_path / "config.toml"
    config.write_text(
        "\n".join(
            f"[sources.{name}]\nroots = []"
            for name in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        )
    )
    if effect == "commit":
        stage = (
            "migration:0041:commit"
            if operation == "upgrade"
            else "initialize_schema:commit"
            if operation == "fresh"
            else "recreate_schema:commit"
        )
    elif effect == "version":
        stage = "repair:version" if operation == "upgrade" else "schema_version"
    else:
        stage = "repair:" + effect
    result = subprocess.run(
        [sys.executable, "-m", "tests.repair_process", operation, stage, fault],
        env={
            **os.environ,
            "RECALL_CONFIG_PATH": str(config),
            "RECALL_DATA_DIR": str(data),
            "RECALL_DB_PATH": str(database),
            "RECALL_LOCK_PATH": str(data / "recall.lock"),
        },
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert f"checkpoint:{stage}" in result.stdout, result.stderr
    assert result.returncode == (-signal.SIGKILL if fault == "kill" else 1), result.stderr
    committed = effect == "commit"
    with duckdb.connect(str(database), read_only=True) as fresh:
        tables = {
            row[0]
            for row in fresh.execute(
                "SELECT table_name FROM duckdb_tables() WHERE NOT internal"
            ).fetchall()
        }
        assert ("daemon_index_repair" in tables) == (committed or operation == "recreate")
        versions = (
            fresh.execute("SELECT max(version) FROM schema_version").fetchone()
            if "schema_version" in tables
            else (None,)
        )
        assert versions == (
            41 if committed or operation == "recreate" else 36 if operation == "upgrade" else None,
        )
        audit = (
            fresh.execute(
                "SELECT count(*) FROM schema_migrations "
                "WHERE migration_id='0041_daemon_index_repair'"
            ).fetchone()
            if "schema_migrations" in tables
            else (0,)
        )
        assert audit == (int(committed or operation == "recreate"),)
        if operation != "fresh":
            if operation == "recreate" and committed:
                assert "witness" not in tables
            else:
                assert fresh.execute(
                    "SELECT * FROM witness ORDER BY payload NULLS FIRST"
                ).fetchall() == [(7, None), (7, "acknowledged")]
        if operation == "recreate":
            backups = tuple((data / "snapshots").glob("recreate-*/recall.duckdb"))
            assert len(backups) == 1
            with duckdb.connect(str(backups[0]), read_only=True) as backup:
                assert backup.execute(
                    "SELECT * FROM witness ORDER BY payload NULLS FIRST"
                ).fetchall() == [(7, None), (7, "acknowledged")]
    marker = data / "daemon-failure.json"
    if marker.exists() and not committed and operation != "recreate":
        import json

        assert json.loads(marker.read_text())["version"] == 1
