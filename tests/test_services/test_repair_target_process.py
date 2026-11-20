"""Every actual saved secondary target remains named after process loss."""

import shutil
import signal
from pathlib import Path

import duckdb
import pytest
from recall.db.repair import load_repair_row

from tests.test_services.test_repair_process import (
    _assert_content,
    _child,
    _diagnostic_copy,
    _file_hashes,
    _fixture,
)


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("fault", ["kill", "error", "fatal"])
@pytest.mark.parametrize("effect", ["drop", "create"])
def test_every_saved_target_survives_an_interrupted_catalog(
    tmp_path: Path, probe: bool, fault: str, effect: str
) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    database, data, environment = _fixture(baseline, probe)
    with duckdb.connect(str(database), read_only=True) as original:
        targets = original.execute(
            "SELECT schema_name,index_name FROM duckdb_indexes() WHERE NOT is_primary ORDER BY ALL"
        ).fetchall()
    assert 1 <= len(targets) <= 64
    assert ("main", "witness") in targets
    marker = (data / "daemon-failure.json").read_bytes()
    for ordinal, (schema, index) in enumerate(targets):
        case = tmp_path / f"target-{ordinal}"
        current = case / "data"
        shutil.copytree(data, current)
        path = current / "recall.duckdb"
        child_environment = {
            **environment,
            "RECALL_DATA_DIR": str(current),
            "RECALL_DB_PATH": str(path),
            "RECALL_LOCK_PATH": str(current / "recall.lock"),
        }
        stage = f"after_{effect}:{schema}.{index}"
        result = _child(child_environment, "repair", stage, fault)
        assert f"checkpoint:{stage}" in result.stdout, result.stderr
        assert result.returncode == (-signal.SIGKILL if fault == "kill" else 1), result.stderr
        assert (current / "daemon-failure.json").read_bytes() == marker
        before = _file_hashes(path)
        with _diagnostic_copy(path, case / "diagnosis") as fresh:
            _assert_content(fresh)
            row = load_repair_row(fresh)
            assert (
                row.phase,
                row.attempt_no,
                row.success_since_attempt,
                row.success_since_failure,
            ) == ("running", 1, False, False)
            assert row.target_manifest is not None
            assert {
                (target.schema_name, target.index_name) for target in row.target_manifest.targets
            } == set(targets)
            present = set(
                fresh.execute("SELECT schema_name,index_name FROM duckdb_indexes()").fetchall()
            )
            assert ((schema, index) in present) == (effect == "create")
        restarted = _child(child_environment, "startup", "unused", "unused")
        assert restarted.returncode == 3, restarted.stderr
        assert _file_hashes(path) == before
        assert (current / "daemon-failure.json").read_bytes() == marker
        assert (current / "daemon-refused").exists()
