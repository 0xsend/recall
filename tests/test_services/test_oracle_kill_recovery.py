"""A process killed while sorting leaves a workspace that the next owner reclaims."""

import signal
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.db.repair import load_repair_row
from recall.services.repair_oracles import OracleStore

from tests.test_services.test_repair_process import (
    _assert_content,
    _child,
    _diagnostic_copy,
    _fixture,
)

EVENT = UUID(int=1)

# Small runs and a fan-in of two make every sorter stage reachable on a tiny database.
_CAPTURE_CHILD = """
import os, signal, sys
from pathlib import Path
from uuid import UUID
import duckdb
from recall.db.lease import StandaloneLifetime
from recall.db.oracle_sort import SortShape
from recall.db.repair_oracle import capture_oracle
data = Path(sys.argv[1])
stage = sys.argv[2]
def barrier(reached):
    if reached == stage:
        print("checkpoint:" + reached, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
with duckdb.connect(str(data / "recall.duckdb"), read_only=True) as conn:
    capture_oracle(
        conn,
        data / "repair-oracles" / str(UUID(int=1)),
        UUID(int=1),
        shape=SortShape(run_bytes=64, fan_in=2),
        sidecar=data / "recall.fts.sqlite",
        barrier=barrier,
        observer=StandaloneLifetime(),
    )
"""


def _killed_capture(environment: dict[str, str], data: Path, stage: str) -> None:
    child = subprocess.run(
        [sys.executable, "-c", _CAPTURE_CHILD, str(data), stage],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert f"checkpoint:{stage}" in child.stdout, child.stderr
    assert child.returncode == -signal.SIGKILL, child.stderr


def _workspaces(data: Path) -> list[Path]:
    return sorted((data / "repair-oracles").glob(".compare-*")) + sorted(
        (data / "repair-oracles").glob(".capture-*")
    )


def _assert_repaired(database: Path, data: Path) -> None:
    with _diagnostic_copy(database, database.parent.parent / "after") as fresh:
        _assert_content(fresh)
        row = load_repair_row(fresh)
        assert (row.phase, row.attempt_no) == ("verified", 1)
    assert not (data / "daemon-refused").exists()
    assert _workspaces(data) == []


@pytest.mark.parametrize(
    ("stage", "sorter_directory"),
    [
        ("sort_run_written", True),
        ("sort_level_merged", True),
        ("table_sorted", False),
        ("capture_synced", False),
    ],
    ids=["mid-run", "mid-level-merge", "between-tables", "after-capture-sync"],
)
def test_a_killed_capture_is_recaptured_by_the_next_start(
    tmp_path: Path, stage: str, sorter_directory: bool
) -> None:
    database, data, environment = _fixture(tmp_path, probe=False)
    _killed_capture(environment, data, stage)
    workspace = data / "repair-oracles" / f".capture-{EVENT}"
    assert workspace.is_dir()
    if stage != "capture_synced":
        assert (workspace / "sort").is_dir() == sorter_directory
    restarted = _child(environment, "startup", "unused", "unused")
    assert restarted.returncode == 0, restarted.stderr
    _assert_repaired(database, data)


def test_a_kill_during_the_pre_arm_comparison_resumes_from_the_saved_reference(
    tmp_path: Path,
) -> None:
    database, data, environment = _fixture(tmp_path, probe=False)
    killed = _child(environment, "repair", "pre_arm:sort_run_written", "kill")
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    comparison = list((data / "repair-oracles").glob(".compare-*"))
    assert len(comparison) == 1 and (comparison[0] / "sort").is_dir()
    with _diagnostic_copy(database, tmp_path / "durable") as fresh:
        row = load_repair_row(fresh)
        assert (row.phase, row.attempt_no) == ("required", 0)
        saved = data / str(row.oracle_ref)
        manifest = (saved / "manifest.json").read_bytes()
    restarted = _child(environment, "startup", "unused", "unused")
    assert restarted.returncode == 0, restarted.stderr
    assert (saved / "manifest.json").read_bytes() == manifest
    _assert_repaired(database, data)


def test_a_kill_during_the_post_rebuild_comparison_is_repaired_manually_and_reclaimed_later(
    tmp_path: Path,
) -> None:
    database, data, environment = _fixture(tmp_path, probe=False)
    killed = _child(environment, "repair", "post_rebuild:sort_run_written", "kill")
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    stale = list((data / "repair-oracles").glob(".compare-*"))
    assert len(stale) == 1 and (stale[0] / "sort").is_dir()
    with _diagnostic_copy(database, tmp_path / "durable") as fresh:
        assert load_repair_row(fresh).phase == "running"
    refused = _child(environment, "startup", "unused", "unused")
    assert refused.returncode == 3, refused.stderr
    (data / "daemon-refused").unlink()

    manual = _child(environment, "manual_rebuild", "none", "kill")
    assert manual.returncode == 0, manual.stderr
    # A running manual repair never reconciles the namespace: the stale slot is
    # still there, and only a later reconcile reclaims it.
    assert stale[0].is_dir()
    with duckdb.connect(str(database), read_only=True) as settled:
        _assert_content(settled)
        row = load_repair_row(settled)
    assert row.phase == "verified"
    OracleStore(data).reconcile(row)
    assert not stale[0].exists()
    assert not (data / "daemon-refused").exists()


def test_a_kill_during_the_publication_capture_is_reclaimed_by_the_next_manual_repair(
    tmp_path: Path,
) -> None:
    database, data, environment = _fixture(tmp_path, probe=False)
    interrupted = _child(environment, "repair", "after_drop:main.witness", "kill")
    assert interrupted.returncode == -signal.SIGKILL, interrupted.stderr
    killed = _child(environment, "manual_rebuild", "publication:sort_run_written", "kill")
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    working = list((data / "repair-oracles").glob("*/publication-working"))
    assert len(working) == 1 and (working[0] / "sort").is_dir()
    remedy = _child(environment, "manual_rebuild", "none", "kill")
    assert remedy.returncode == 0, remedy.stderr
    assert not working[0].exists()
    with duckdb.connect(str(database), read_only=True) as settled:
        _assert_content(settled)
        assert load_repair_row(settled).phase == "verified"
