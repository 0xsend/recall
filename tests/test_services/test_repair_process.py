"""Real process loss cannot grant another automatic attempt or lose exact rows."""

import hashlib
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.db.repair import load_repair_row
from recall.db.schema import ensure_schema
from recall.services.repair_events import EventFailure, RecoveryEvent, publish_event, read_events

from tests.test_db.test_repair_oracle import populate

_STAGES = (
    "capture_synced",
    "capture_published",
    "capture_directory_synced",
    "before_required_commit",
    "after_required_commit",
    "after_arm_commit",
    "after_drop:main.witness",
    "after_create:main.witness",
    "before_checkpoint",
    "after_checkpoint",
    "after_owned_close",
    "after_fresh_postchecks",
    "before_verified_commit",
    "after_verified_commit",
    "before_event_unlink",
    "after_event_unlink_sync",
)
_PRE_ARM = frozenset(_STAGES[:5])
_SETTLED = frozenset(_STAGES[-3:])


def _child(environment: dict[str, str], operation: str, stage: str, fault: str):
    return subprocess.run(
        [sys.executable, "-m", "tests.repair_process", operation, stage, fault],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _fixture(tmp_path: Path, probe: bool):
    data = tmp_path / "data"
    data.mkdir()
    database = data / "recall.duckdb"
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        populate(conn)
    config = tmp_path / "config.toml"
    config.write_text(
        "\n".join(
            f"[sources.{name}]\nroots = []"
            for name in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        )
    )
    event = RecoveryEvent(
        event_id=UUID(int=1),
        failure=(
            None
            if probe
            else EventFailure(
                site="acknowledged_writer",
                exception_type="FatalException",
                message="Failed to delete all rows from index",
                signature="original",
                failure_class="index-divergence",
                at=datetime(2026, 11, 1),
            )
        ),
        needs_index_verification=probe,
        probe=probe,
    )
    publish_event(data, event)
    return (
        database,
        data,
        {
            **os.environ,
            "RECALL_CONFIG_PATH": str(config),
            "RECALL_DATA_DIR": str(data),
            "RECALL_DB_PATH": str(database),
            "RECALL_LOCK_PATH": str(data / "recall.lock"),
        },
    )


def _assert_content(conn):
    assert conn.execute("SELECT * FROM canonical ORDER BY id").fetchall() == [
        ("a", None, Decimal("123.000000001")),
        ("b", "βeta", None),
    ]
    assert conn.execute("SELECT * FROM companions.vectors").fetchall() == [
        ("a", (1.0, -0.0, None), b"\x00\xff")
    ]
    assert conn.execute("SELECT kind,id FROM intents ORDER BY ALL").fetchall() == [
        ("message", "a"),
        ("message", "a"),
        ("session", "b"),
    ]


def _diagnostic_copy(database: Path, destination: Path):
    destination.mkdir()
    copy = destination / "recall.duckdb"
    shutil.copyfile(database, copy)
    wal = Path(str(database) + ".wal")
    if wal.exists():
        shutil.copyfile(wal, Path(str(copy) + ".wal"))
    # DuckDB's read-only replay may need to write reconstructed index blocks.
    # Only this diagnostic copy grants that authority, never the held original.
    return duckdb.connect(str(copy))


def _file_hashes(database: Path):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (database, Path(str(database) + ".wal"))
        if path.exists()
    }


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("fault", ["kill", "error", "fatal"])
@pytest.mark.parametrize("stage", _STAGES)
def test_automatic_recovery_crash_and_observed_error_boundaries(
    tmp_path: Path, probe: bool, fault: str, stage: str
) -> None:
    database, data, environment = _fixture(tmp_path, probe)
    original = (data / "daemon-failure.json").read_bytes()
    result = _child(environment, "repair", stage, fault)
    assert f"checkpoint:{stage}" in result.stdout, result.stderr
    assert result.returncode == (-signal.SIGKILL if fault == "kill" else 1), result.stderr
    if stage != "after_event_unlink_sync":
        assert (data / "daemon-failure.json").read_bytes() == original
    before_files = _file_hashes(database)
    with _diagnostic_copy(database, tmp_path / "before-diagnosis") as fresh:
        _assert_content(fresh)
        row = load_repair_row(fresh)
        catalog = fresh.execute(
            "SELECT schema_name,index_name,sql FROM duckdb_indexes() ORDER BY ALL"
        ).fetchall()
        if stage not in _PRE_ARM:
            assert row.attempt_no == 1
            assert (row.success_since_attempt, row.success_since_failure) == (False, False)
        if stage in _SETTLED:
            assert row.phase == "verified"
        elif stage not in _PRE_ARM:
            assert row.phase == "running"
        if row.trigger_json is not None:
            assert row.trigger_json.trigger_id == UUID(int=1)
    refusal = (data / "daemon-refused").exists()
    expected_admission = (
        (stage in _PRE_ARM and not (stage == "after_required_commit" and fault == "fatal"))
        or (stage in _SETTLED and fault != "fatal")
    ) and not refusal
    restarted = _child(environment, "startup", "unused", "unused")
    assert restarted.returncode == (0 if expected_admission else 3), restarted.stderr
    if not expected_admission and (row.phase == "running" or refusal):
        assert _file_hashes(database) == before_files
    with _diagnostic_copy(database, tmp_path / "after-diagnosis") as fresh:
        _assert_content(fresh)
        after = load_repair_row(fresh)
        if expected_admission:
            assert (
                after.phase,
                after.attempt_no,
                after.success_since_attempt,
                after.success_since_failure,
            ) == ("verified", 1, False, False)
            assert read_events(data).primary is None and not read_events(data).secondary
            assert fresh.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE index_name='witness'"
            ).fetchone() == ("witness",)
        else:
            assert after.attempt_no == row.attempt_no
            assert (
                fresh.execute(
                    "SELECT schema_name,index_name,sql FROM duckdb_indexes() ORDER BY ALL"
                ).fetchall()
                == catalog
            )
            assert (data / "daemon-refused").exists()
