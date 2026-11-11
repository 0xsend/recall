"""Old acknowledged rows and legacy failure identity survive each pre-fence boundary."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
from recall.db.migrations import get_migrations
from recall.services.self_repair import remember_fatal_failure

from tests.test_core.test_schema_cutover import _create_v13_db_with_blocked_0014
from tests.test_services.test_repair_process import _diagnostic_copy

_HISTORICAL = tuple(m for m in get_migrations() if 13 < m.target_version < 41)


def _legacy(database):
    with _create_v13_db_with_blocked_0014() as conn:
        conn.execute("DROP TABLE message_state_new")
        conn.execute("UPDATE runtime_state SET embedding_dimensions=384")
        conn.execute(
            "CREATE TABLE message_embeddings(message_id VARCHAR PRIMARY KEY,"
            "content_embedding FLOAT[384],thinking_embedding FLOAT[384])"
        )
        conn.execute(
            "CREATE TABLE tool_call_embeddings(tool_call_id VARCHAR PRIMARY KEY,"
            "bash_embedding FLOAT[384])"
        )
        conn.execute(
            "INSERT INTO message_embeddings VALUES ('m1',?,NULL)", [[0.0, 0.25, 1.5] + [0.0] * 381]
        )
        conn.execute("CREATE SCHEMA companion")
        conn.execute("CREATE TABLE companion.acknowledged(k VARCHAR,value BIGINT)")
        conn.execute(
            "INSERT INTO companion.acknowledged VALUES ('a',9223372036854775807),('a',NULL)"
        )
        escaped = str(database).replace("'", "''")
        conn.execute(f"ATTACH '{escaped}' AS legacy")
        conn.execute("COPY FROM DATABASE memory TO legacy")
        conn.execute("DETACH legacy")


def _environment(tmp_path, database):
    config = tmp_path / "config.toml"
    config.write_text(
        "\n".join(
            f"[sources.{name}]\nroots=[]"
            for name in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        )
    )
    return {
        **os.environ,
        "RECALL_CONFIG_PATH": str(config),
        "RECALL_DATA_DIR": str(tmp_path),
        "RECALL_DB_PATH": str(database),
        "RECALL_LOCK_PATH": str(tmp_path / "recall.lock"),
    }


@pytest.mark.parametrize("migration", _HISTORICAL, ids=lambda m: m.id)
@pytest.mark.parametrize("effect", ["migration_body", "migration_audit"])
@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("fault", ["kill", "error", "fatal"])
def test_each_historical_boundary_retains_acknowledged_rows_and_legacy_origin(
    tmp_path: Path, migration, effect: str, when: str, fault: str
):
    database = tmp_path / "recall.duckdb"
    _legacy(database)
    remember_fatal_failure(
        tmp_path,
        duckdb.FatalException("Failed to delete all rows from index in retained legacy origin"),
        site="legacy_origin",
    )
    original = (tmp_path / "daemon-failure.json").read_bytes()
    selected = f"{when}:{effect}:{migration.id}"
    result = subprocess.run(
        [sys.executable, "-m", "tests.migration_process", selected, fault, "strict"],
        env=_environment(tmp_path, database),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert f"checkpoint:{selected}" in result.stdout, result.stderr
    assert result.returncode == (-signal.SIGKILL if fault == "kill" else 1), result.stderr
    try:
        fresh = duckdb.connect(str(database), read_only=True)
    except duckdb.IOException:
        fresh = _diagnostic_copy(database, tmp_path / "diagnosis")
    with fresh:
        assert fresh.execute("SELECT id,source,source_path FROM sessions").fetchall() == [
            ("s1", "claude_code", "/tmp/s1")
        ]
        assert fresh.execute(
            "SELECT message_id,content,thinking FROM message_state"
        ).fetchall() == [("m1", "hello", None)]
        vector = fresh.execute(
            "SELECT content_embedding FROM message_embeddings WHERE message_id='m1'"
        ).fetchone()
        assert vector is not None and vector[0] == tuple([0.0, 0.25, 1.5] + [0.0] * 381)
        assert fresh.execute(
            "SELECT * FROM companion.acknowledged ORDER BY value NULLS FIRST"
        ).fetchall() == [("a", None), ("a", 9223372036854775807)]
        version = fresh.execute("SELECT max(version) FROM schema_version").fetchone()
        assert version is not None and 13 <= version[0] <= migration.target_version
        audits = {
            row[0] for row in fresh.execute("SELECT migration_id FROM schema_migrations").fetchall()
        }
        assert {m.id for m in _HISTORICAL if m.target_version < migration.target_version} <= audits
        # The source-path and usage-file migrations commit their own audit
        # with the body; older migrations leave it to the runner.
        body_audited = migration.target_version in {35, 36}
        audited = (effect == "migration_audit" and when == "after") or (
            body_audited and (effect == "migration_audit" or when == "after")
        )
        assert (migration.id in audits) == audited
        assert fresh.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name='daemon_index_repair'"
        ).fetchone() == (0,)
    assert (tmp_path / "daemon-failure.json").read_bytes() == original
    for path in (tmp_path / "daemon-failure.json", *(tmp_path / "fatal-events").glob("*.json")):
        assert json.loads(path.read_text()).get("version", 1) == 1


@pytest.mark.parametrize("mode", ["strict", "lenient"])
def test_supported_v13_chain_reaches_the_committed_fence_with_literal_rows(
    tmp_path: Path, mode: str
):
    database = tmp_path / "recall.duckdb"
    _legacy(database)
    result = subprocess.run(
        [sys.executable, "-m", "tests.migration_process", "none", "kill", mode],
        env=_environment(tmp_path, database),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute("SELECT max(version) FROM schema_version").fetchone() == (41,)
        assert fresh.execute(
            "SELECT content FROM message_state WHERE message_id='m1'"
        ).fetchone() == ("hello",)
        assert fresh.execute("SELECT phase,attempt_no FROM daemon_index_repair").fetchone() == (
            "idle",
            0,
        )
        assert fresh.execute(
            "SELECT count(*) FROM schema_migrations WHERE migration_id='0041_daemon_index_repair'"
        ).fetchone() == (1,)
