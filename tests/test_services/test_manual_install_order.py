"""Replacement remains spent/running until its installed-incarnation completion."""

import hashlib
import shutil
import signal
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.repair import load_repair_row
from recall.services import manual_repair
from recall.services.self_repair import DaemonStartupRefused

from tests.test_services.test_repair_process import _child, _fixture


@pytest.mark.parametrize("proof", ["present", "missing", "external", "external_bad"])
@pytest.mark.parametrize("backend", ["sqlite_sidecar", "duckdb"])
def test_interrupted_installed_candidate_keeps_running_and_duplicate_intents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, proof: str
):
    database, data, environment = _fixture(tmp_path, False)
    config_path = Path(environment["RECALL_CONFIG_PATH"])
    config_path.write_text(config_path.read_text() + f'\n[fts]\nbackend="{backend}"\n')
    with duckdb.connect(str(database)) as conn:
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) "
            "VALUES ('message','a','delete'),('message','a','delete')"
        )
    assert (
        _child(environment, "repair", "after_drop:main.witness", "kill").returncode
        == -signal.SIGKILL
    )
    for name in ("RECALL_CONFIG_PATH", "RECALL_DATA_DIR", "RECALL_DB_PATH", "RECALL_LOCK_PATH"):
        monkeypatch.setenv(name, environment[name])
    config = AppConfig.load()

    def failed(stage):
        if stage == "after_manual_database_replace":
            raise OSError("installed completion not yet reached")

    with pytest.raises(DaemonStartupRefused, match="not yet reached"):
        manual_repair.ManualRepair(config, barrier=failed).compact()
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = load_repair_row(fresh)
        assert (row.phase, row.attempt_no) == ("running", 2)
        assert not row.success_since_attempt and not row.success_since_failure
        assert row.release_certificate is None
        assert fresh.execute(
            "SELECT kind,id,op FROM fts_sidecar_pending ORDER BY ALL"
        ).fetchall() == [("message", "a", "delete"), ("message", "a", "delete")]
    assert (data / "daemon-refused").exists()
    external = None
    if proof != "present":
        assert row.oracle_ref is not None
        if proof.startswith("external"):
            external = tmp_path / "matching-oracle"
            shutil.copytree(data / row.oracle_ref, external)
            if proof == "external_bad":
                (external / "publication" / "manifest.json").write_bytes(b"invalid")
        shutil.rmtree(data / row.oracle_ref / "publication")
    if proof in {"missing", "external_bad"}:
        before = hashlib.sha256(database.read_bytes()).hexdigest()
        marker = (data / "daemon-failure.json").read_bytes()
        with pytest.raises(DaemonStartupRefused, match="publication"):
            manual_repair.ManualRepair(config, external_oracle=external).execute()
        with duckdb.connect(str(database), read_only=True) as fresh:
            assert load_repair_row(fresh).attempt_no == 2
            assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (2,)
        assert hashlib.sha256(database.read_bytes()).hexdigest() == before
        assert (data / "daemon-failure.json").read_bytes() == marker
        assert (data / "daemon-refused").exists()
        return
    external_bytes = (
        None
        if external is None
        else {
            str(path.relative_to(external)): path.read_bytes()
            for path in external.rglob("*")
            if path.is_file()
        }
    )
    manual_repair.ManualRepair(config, external_oracle=external).execute()
    if external is not None:
        assert {
            str(path.relative_to(external)): path.read_bytes()
            for path in external.rglob("*")
            if path.is_file()
        } == external_bytes
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = load_repair_row(fresh)
        assert row.phase == "verified" and row.attempt_no == 3
        assert row.release_certificate is not None
        assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (0,)
    assert not (data / "daemon-refused").exists()
