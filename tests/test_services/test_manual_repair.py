"""The existing offline command must qualify exact release after an armed interruption."""

import signal
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.repair import load_repair_row

from tests.test_services.test_repair_process import _assert_content, _child, _fixture


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize(
    "failed_stage",
    [
        None,
        "after_manual_database_copy",
        "after_manual_candidate_close",
        "after_manual_original_backup",
        "after_manual_database_replace",
    ],
)
def test_native_manual_candidate_preserves_rows_and_retries_release(
    tmp_path: Path, failed_stage: str | None, compact: bool
):
    import asyncio
    from dataclasses import replace

    from recall.db.queries import create_fts_indexes, load_fts_extension
    from recall.services.manual_repair import ManualRepair
    from recall.services.self_repair import DaemonStartupRefused

    from tests.test_services.test_rpc_run_completion import _indexed_recompute_setup

    database, server = _indexed_recompute_setup(tmp_path)
    config = replace(server._config, fts=replace(server._config.fts, backend="duckdb"))
    asyncio.run(server.stop())
    with duckdb.connect(str(database)) as conn:
        create_fts_indexes(conn, config.fts)
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) SELECT 'message',id,'upsert' FROM messages"
        )
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) SELECT 'message',id,'upsert' FROM messages"
        )
        before = conn.execute("SELECT * FROM message_state ORDER BY ALL").fetchall()
        pending = conn.execute(
            "SELECT kind,id,op,queued_at FROM fts_sidecar_pending ORDER BY ALL"
        ).fetchall()
        ids = conn.execute("SELECT message_id FROM message_state ORDER BY ALL").fetchall()

    def fail(stage):
        if stage == failed_stage:
            raise OSError("native manual boundary unavailable")

    def run(barrier=lambda stage: None):
        repair = ManualRepair(config, barrier=barrier)
        return repair.compact() if compact else repair.execute()

    if failed_stage is not None:
        with pytest.raises(DaemonStartupRefused, match="boundary unavailable"):
            run(fail)
        with duckdb.connect(str(database), read_only=True) as fresh:
            row = load_repair_row(fresh)
            assert fresh.execute("SELECT * FROM message_state ORDER BY ALL").fetchall() == before
            assert row.phase == "running" and row.release_certificate is None
            assert (
                fresh.execute(
                    "SELECT kind,id,op,queued_at FROM fts_sidecar_pending ORDER BY ALL"
                ).fetchall()
                == pending
            )
        assert (config.data_dir / "daemon-refused").exists()
    run()
    with duckdb.connect(str(database)) as fresh:
        load_fts_extension(fresh)
        row = load_repair_row(fresh)
        assert row.phase == "verified" and row.success_since_attempt and row.success_since_failure
        assert row.release_certificate is not None
        assert fresh.execute("SELECT * FROM message_state ORDER BY ALL").fetchall() == before
        assert (
            fresh.execute("SELECT name FROM fts_main_message_state.docs ORDER BY ALL").fetchall()
            == ids
        )
        assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (0,)
    assert not (config.data_dir / "daemon-refused").exists()
    backups = tuple((config.data_dir / "repair-work").glob("*/original-set/recall.duckdb"))
    assert backups
    for backup in backups:
        with duckdb.connect(str(backup), read_only=True) as old:
            assert load_repair_row(old).phase == "running"
            assert old.execute("SELECT * FROM message_state ORDER BY ALL").fetchall() == before
            assert (
                old.execute(
                    "SELECT kind,id,op,queued_at FROM fts_sidecar_pending ORDER BY ALL"
                ).fetchall()
                == pending
            )


def test_interrupted_orphan_publication_cleanup_can_finish_without_recapturing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from recall.services.manual_repair import ManualRepair
    from recall.services.self_repair import DaemonStartupRefused

    database, data, environment = _fixture(tmp_path, False)
    assert (
        _child(environment, "repair", "after_drop:main.witness", "kill").returncode
        == -signal.SIGKILL
    )
    for name in ("RECALL_CONFIG_PATH", "RECALL_DATA_DIR", "RECALL_DB_PATH", "RECALL_LOCK_PATH"):
        monkeypatch.setenv(name, environment[name])
    config = AppConfig.load()

    def failed(stage):
        if stage == "before_manual_completion_commit":
            raise OSError("completion unavailable")

    with pytest.raises(DaemonStartupRefused, match="completion unavailable"):
        ManualRepair(config, barrier=failed).execute()
    unlink = Path.unlink
    interrupted = False

    def fail_unlink(path, *args, **kwargs):
        nonlocal interrupted
        result = unlink(path, *args, **kwargs)
        if path.parent.name == "publication" and path.name.endswith(".rows") and not interrupted:
            interrupted = True
            raise OSError("cleanup interrupted after unlink")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_unlink)
        with pytest.raises(DaemonStartupRefused, match="cleanup interrupted"):
            ManualRepair(config).execute()
    assert interrupted
    ManualRepair(config).execute()
    with duckdb.connect(str(database), read_only=True) as conn:
        _assert_content(conn)
        row = load_repair_row(conn)
        assert row.phase == "verified" and row.release_certificate is not None
    assert not (data / "daemon-refused").exists()


def test_confirmed_cli_compact_releases_interrupted_recovery_instead_of_threshold_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import json

    from recall.cli.app import app
    from typer.testing import CliRunner

    database, data, environment = _fixture(tmp_path, False)
    assert (
        _child(environment, "repair", "after_drop:main.witness", "kill").returncode
        == -signal.SIGKILL
    )
    for name in ("RECALL_CONFIG_PATH", "RECALL_DATA_DIR", "RECALL_DB_PATH", "RECALL_LOCK_PATH"):
        monkeypatch.setenv(name, environment[name])
    (data / "daemon-refused").write_text("explicit offline remedy required")
    result = CliRunner().invoke(
        app, ["compact", "--yes", "--no-restart", "--threshold", "1000", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["skipped_reason"] is None
    assert payload["tables_copied"]["daemon_index_repair"] == 1
    with duckdb.connect(str(database), read_only=True) as fresh:
        _assert_content(fresh)
        row = load_repair_row(fresh)
        assert row.phase == "verified" and row.attempt_no == 2
        assert row.release_certificate is not None
        assert fresh.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE index_name='witness'"
        ).fetchone() == ("witness",)
    assert not (data / "daemon-refused").exists()
    assert not (data / "daemon-failure.json").exists()


@pytest.mark.parametrize("changed", [False, True])
def test_existing_offline_command_requires_the_saved_content_and_completes_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: bool
) -> None:
    from recall.cli.db import _rebuild_indexes_locally

    database, data, environment = _fixture(tmp_path, False)
    interrupted = _child(environment, "repair", "after_drop:main.witness", "kill")
    assert interrupted.returncode == -signal.SIGKILL, interrupted.stderr
    for name in ("RECALL_CONFIG_PATH", "RECALL_DATA_DIR", "RECALL_DB_PATH", "RECALL_LOCK_PATH"):
        monkeypatch.setenv(name, environment[name])
    config = AppConfig.load()
    refusal = data / "daemon-refused"
    refusal.write_text("retained armed interruption")
    original = (data / "daemon-failure.json").read_bytes()
    if changed:
        with duckdb.connect(str(database)) as conn:
            conn.execute("UPDATE canonical SET text='changed' WHERE id='a'")
        from recall.cli.contract import CliError, ErrorCode

        with pytest.raises(CliError) as rejected:
            _rebuild_indexes_locally(config)
        assert rejected.value.code == ErrorCode.RUNTIME and rejected.value.exit_code == 3
        assert "content" in rejected.value.message or "oracle" in rejected.value.message
        assert (data / "daemon-failure.json").read_bytes() == original
        assert refusal.read_text() == "retained armed interruption"
        with duckdb.connect(str(database), read_only=True) as fresh:
            row = load_repair_row(fresh)
            assert (row.phase, row.attempt_no) == ("running", 1)
            assert fresh.execute(
                "SELECT count(*) FROM duckdb_indexes() WHERE index_name='witness'"
            ).fetchone() == (0,)
    else:
        result = _rebuild_indexes_locally(config)
        assert result.created > 0
        with duckdb.connect(str(database), read_only=True) as fresh:
            _assert_content(fresh)
            row = load_repair_row(fresh)
            assert (
                row.phase,
                row.attempt_no,
                row.success_since_attempt,
                row.success_since_failure,
            ) == ("verified", 2, True, True)
            assert row.release_certificate is not None
            assert fresh.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE index_name='witness'"
            ).fetchone() == ("witness",)
        assert not (data / "daemon-failure.json").exists()
        assert not refusal.exists()


def test_explicit_retry_after_failed_manual_completion_keeps_duplicate_intents_and_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import asyncio

    from recall.services.manual_repair import ManualRepair
    from recall.services.self_repair import DaemonStartupRefused

    from tests.test_services.test_rpc_run_completion import _indexed_recompute_setup

    database, server = _indexed_recompute_setup(tmp_path)
    config = server._config
    asyncio.run(server.stop())
    with duckdb.connect(str(database)) as conn:
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) SELECT 'message',id,'upsert' FROM messages"
        )
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) SELECT 'message',id,'upsert' FROM messages"
        )
        messages = conn.execute("SELECT * FROM messages ORDER BY ALL").fetchall()

    def stop_after_arm(stage):
        if stage == "after_manual_arm_commit":
            raise RuntimeError("manual actor interrupted")

    with pytest.raises(DaemonStartupRefused, match="interrupted"):
        ManualRepair(config, barrier=stop_after_arm).execute()

    origin = OSError("manual completion acknowledgement unavailable")
    executions = 0

    def fail_completion(stage):
        nonlocal executions
        if stage == "before_manual_completion_commit":
            executions += 1
            raise origin

    with pytest.raises(DaemonStartupRefused) as caught:
        ManualRepair(config, barrier=fail_completion).execute()
    assert caught.value.__context__ is origin
    assert executions == 1
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (2,)
        assert fresh.execute("SELECT * FROM messages ORDER BY ALL").fetchall() == messages
        row = load_repair_row(fresh)
        assert (
            row.phase,
            row.attempt_no,
            row.success_since_attempt,
            row.success_since_failure,
        ) == ("running", 3, False, False)
        assert row.release_certificate is None
    ManualRepair(config).execute()
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = load_repair_row(fresh)
        assert (
            row.phase,
            row.attempt_no,
            row.success_since_attempt,
            row.success_since_failure,
        ) == ("verified", 4, True, True)
        assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (0,)
        assert fresh.execute("SELECT * FROM messages ORDER BY ALL").fetchall() == messages
    assert not (tmp_path / "daemon-refused").exists()


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize(
    "stage", ["after_manual_keyword_publication", "before_manual_event_unlink"]
)
def test_a_new_instruction_outside_the_manual_batch_blocks_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: bool, stage: str
):
    from uuid import UUID

    from recall.services.manual_repair import ManualRepair
    from recall.services.repair_events import RecoveryEvent, publish_event, read_events
    from recall.services.self_repair import DaemonStartupRefused

    database, data, environment = _fixture(tmp_path, probe)
    result = _child(environment, "repair", "after_drop:main.witness", "kill")
    assert result.returncode == -signal.SIGKILL, result.stderr
    for name in ("RECALL_CONFIG_PATH", "RECALL_DATA_DIR", "RECALL_DB_PATH", "RECALL_LOCK_PATH"):
        monkeypatch.setenv(name, environment[name])
    config = AppConfig.load()
    original = (data / "daemon-failure.json").read_bytes()
    later = RecoveryEvent(
        event_id=UUID(int=2), failure=None, needs_index_verification=True, probe=True
    )
    emitted = False

    def arrive(at):
        nonlocal emitted
        if at == stage and not emitted:
            emitted = True
            publish_event(data, later)

    with pytest.raises(DaemonStartupRefused, match=r"post-arm|outside|attempt"):
        ManualRepair(config, barrier=arrive).execute()
    assert emitted
    assert (data / "daemon-failure.json").read_bytes() == original
    assert any(event.event_id == later.event_id for event in read_events(data).identified())
    assert (data / "daemon-refused").exists()
    with duckdb.connect(str(database), read_only=True) as fresh:
        _assert_content(fresh)
        row = load_repair_row(fresh)
        assert row.attempt_no == 2
        assert (row.success_since_attempt, row.success_since_failure) == (False, False)
        assert row.trigger_json is not None
        assert later.event_id not in {event.event_id for event in row.trigger_json.events}


def test_manual_release_retry_validates_certificate_after_intent_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from recall.services.manual_repair import ManualRepair
    from recall.services.self_repair import DaemonStartupRefused

    database, data, environment = _fixture(tmp_path, False)
    assert (
        _child(environment, "repair", "after_drop:main.witness", "kill").returncode
        == -signal.SIGKILL
    )
    for name in ("RECALL_CONFIG_PATH", "RECALL_DATA_DIR", "RECALL_DB_PATH", "RECALL_LOCK_PATH"):
        monkeypatch.setenv(name, environment[name])
    config = AppConfig.load()

    def failed(at):
        if at == "before_manual_refusal_unlink":
            raise OSError("refusal unlink unavailable")

    with pytest.raises(DaemonStartupRefused, match="unlink"):
        ManualRepair(config, barrier=failed).execute()
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = load_repair_row(fresh)
        assert row.phase == "verified" and row.attempt_no == 2
        assert row.release_certificate is not None
    assert not (data / "daemon-failure.json").exists()
    assert (data / "daemon-refused").exists()
    result = ManualRepair(config).execute()
    assert result.created == 0 and result.dropped == 0
    assert not (data / "daemon-refused").exists()
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert load_repair_row(fresh).attempt_no == 2
        _assert_content(fresh)


@pytest.mark.parametrize("invalid", [False, True])
def test_cli_external_oracle_must_match_the_unfinished_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: bool
):
    import json

    from recall.cli.app import app
    from typer.testing import CliRunner

    database, data, environment = _fixture(tmp_path, False)
    assert (
        _child(environment, "repair", "after_drop:main.witness", "kill").returncode
        == -signal.SIGKILL
    )
    for name in ("RECALL_CONFIG_PATH", "RECALL_DATA_DIR", "RECALL_DB_PATH", "RECALL_LOCK_PATH"):
        monkeypatch.setenv(name, environment[name])
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = load_repair_row(fresh)
    directory = data / row.oracle_ref
    external = tmp_path / "operator-oracle"
    directory.rename(external)
    if invalid:
        manifest = json.loads((external / "manifest.json").read_text())
        (external / manifest["tables"][0]["filename"]).write_bytes(b"corrupt")
    marker = (data / "daemon-failure.json").read_bytes()
    result = CliRunner().invoke(app, ["db", "rebuild-indexes", "--oracle", str(external), "--json"])
    if invalid:
        assert result.exit_code == 3, result.output
        assert json.loads(result.stdout)["error"]["code"] == "RUNTIME"
        assert (data / "daemon-failure.json").read_bytes() == marker
        with duckdb.connect(str(database), read_only=True) as fresh:
            assert load_repair_row(fresh).attempt_no == 1
    else:
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["created"] > 0
        assert (directory / "manifest.json").read_bytes() == (
            external / "manifest.json"
        ).read_bytes()
        assert not (data / "daemon-refused").exists()
        with duckdb.connect(str(database), read_only=True) as fresh:
            _assert_content(fresh)
            assert load_repair_row(fresh).attempt_no == 2


def test_manual_command_replays_only_an_explicitly_diagnosed_copy_of_a_held_wal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from recall.services.manual_repair import ManualRepair

    database, data, environment = _fixture(tmp_path, True)
    interrupted = _child(environment, "repair", "before_checkpoint", "kill")
    assert interrupted.returncode == -signal.SIGKILL, interrupted.stderr
    for name in ("RECALL_CONFIG_PATH", "RECALL_DATA_DIR", "RECALL_DB_PATH", "RECALL_LOCK_PATH"):
        monkeypatch.setenv(name, environment[name])
    ManualRepair(AppConfig.load()).execute()
    with duckdb.connect(str(database), read_only=True) as fresh:
        _assert_content(fresh)
        row = load_repair_row(fresh)
        assert (
            row.phase,
            row.attempt_no,
            row.success_since_attempt,
            row.success_since_failure,
        ) == ("verified", 2, True, True)
    assert any((path / "recall.duckdb").is_file() for path in (data / "repair-work").iterdir())
    assert not (data / "daemon-refused").exists()
