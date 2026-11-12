"""A dead manual actor leaves a held source or a verifiable installed receipt."""

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

_COMMON = (
    "after_manual_arm_commit",
    "after_drop:main.witness",
    "after_create:main.witness",
    "after_manual_fresh_postchecks",
    "after_manual_keyword_publication",
    "after_manual_publication_binding_commit",
    "publication_reference_synced",
    "publication_reference_published",
    "publication_reference_directory_synced",
    "after_manual_intent_retirement",
    "before_manual_completion_commit",
    "after_manual_completion_commit",
    "before_manual_event_unlink",
    "after_manual_event_unlink_sync",
    "before_manual_refusal_unlink",
    "after_manual_refusal_unlink_sync",
)
_COPY = (
    "after_manual_matched_backup",
    "after_manual_database_copy",
    "after_manual_candidate_close",
    "after_manual_original_backup",
    "before_manual_database_replace",
    "after_manual_database_replace",
    "after_manual_database_replace_sync",
)
_VECTORS = tuple(
    (backend, operation, stage)
    for backend in ("sqlite_sidecar", "duckdb")
    for operation in ("manual_rebuild", "manual_compact")
    for stage in _COMMON
    + (() if backend == "sqlite_sidecar" and operation == "manual_rebuild" else _COPY)
)


@pytest.mark.parametrize("backend,operation,stage", _VECTORS)
@pytest.mark.parametrize(
    "probe", [pytest.param(False, id="index_trigger"), pytest.param(True, id="probe_trigger")]
)
@pytest.mark.parametrize("refusal", [pytest.param(False, id="noR"), pytest.param(True, id="heldR")])
@pytest.mark.parametrize("fault", ["kill", "error", "fatal"])
def test_manual_native_process_boundaries(
    tmp_path: Path, backend: str, operation: str, stage: str, probe: bool, fault: str, refusal: bool
) -> None:
    database, data, environment = _fixture(tmp_path, probe)
    config = Path(environment["RECALL_CONFIG_PATH"])
    config.write_text(config.read_text() + f'\n[fts]\nbackend="{backend}"\n')
    with duckdb.connect(str(database)) as seed:
        seed.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) VALUES "
            "('message','outside','delete'),('message','outside','delete')"
        )
        pending = seed.execute(
            "SELECT kind,id,op,queued_at FROM fts_sidecar_pending ORDER BY ALL"
        ).fetchall()
    interrupted = _child(environment, "repair", "after_drop:main.witness", "kill")
    assert interrupted.returncode == -signal.SIGKILL, interrupted.stderr
    if refusal:
        (data / "daemon-refused").write_text("retained manual hold")
    origin = (data / "daemon-failure.json").read_bytes()
    result = _child(environment, operation, stage, fault)
    assert f"checkpoint:{stage}" in result.stdout, result.stderr
    assert result.returncode == (-signal.SIGKILL if fault == "kill" else 3), result.stderr
    try:
        fresh = duckdb.connect(str(database), read_only=True)
    except duckdb.IOException:
        fresh = _diagnostic_copy(database, tmp_path / "diagnosis")
    with fresh:
        _assert_content(fresh)
        row = load_repair_row(fresh)
        assert row.attempt_no == 2
        committed = stage in {
            "after_manual_completion_commit",
            "before_manual_event_unlink",
            "after_manual_event_unlink_sync",
            "before_manual_refusal_unlink",
            "after_manual_refusal_unlink_sync",
        }
        assert row.phase == ("verified" if committed else "running")
        if row.phase == "running":
            assert not row.success_since_attempt and not row.success_since_failure
            assert row.release_certificate is None
            assert (
                fresh.execute(
                    "SELECT kind,id,op,queued_at FROM fts_sidecar_pending ORDER BY ALL"
                ).fetchall()
                == pending
            )
        else:
            assert row.release_certificate is not None
            assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (0,)
    if (data / "daemon-failure.json").exists() and stage not in {
        "after_manual_event_unlink_sync",
        "before_manual_refusal_unlink",
        "after_manual_refusal_unlink_sync",
    }:
        assert (data / "daemon-failure.json").read_bytes() == origin
    hashes = _file_hashes(database)
    startup = _child(environment, "startup", "none", "kill")
    released = stage == "after_manual_refusal_unlink_sync" and fault == "kill"
    assert startup.returncode == (0 if released else 3), startup.stderr
    if not released:
        assert _file_hashes(database) == hashes
    remedy = _child(environment, operation, "none", "kill")
    assert remedy.returncode == 0, remedy.stderr
    with duckdb.connect(str(database), read_only=True) as settled:
        _assert_content(settled)
        row = load_repair_row(settled)
        assert row.phase == "verified" and row.release_certificate is not None
        assert row.success_since_attempt and row.success_since_failure
        assert settled.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE index_name='witness'"
        ).fetchone() == ("witness",)
        assert settled.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (0,)
    assert not (data / "daemon-refused").exists()
    assert not (data / "daemon-failure.json").exists()
