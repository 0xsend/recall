"""A recovery candidate preserves storage format and applies bounded scan policy."""

from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services.manual_repair import ManualRepair

from tests.test_services.test_run_completion import _config


@pytest.mark.parametrize("backend", ["sqlite_sidecar", "duckdb"])
def test_manual_copy_preserves_the_explicit_storage_header(tmp_path: Path, backend: str):
    config = _config(tmp_path, None)
    config = replace(config, fts=replace(config.fts, backend=backend))
    with duckdb.connect(
        str(config.db_path), config={"storage_compatibility_version": "v1.2.0"}
    ) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE acknowledged(id VARCHAR,value BIGINT)")
        conn.execute("INSERT INTO acknowledged VALUES ('a',9223372036854775807)")
    result = ManualRepair(config).compact()
    assert result.replaced
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        header = fresh.execute(
            "SELECT tags['storage_version'] FROM duckdb_databases() "
            "WHERE database_name=current_database()"
        ).fetchone()
        assert header is not None and header[0].removesuffix("+") == "v1.2.0"
        assert fresh.execute("SELECT * FROM acknowledged").fetchall() == [
            ("a", 9223372036854775807)
        ]


def test_every_full_candidate_or_installed_scan_has_runtime_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _config(tmp_path, None)
    with duckdb.connect(str(config.db_path)) as conn:
        ensure_schema(conn)

    connect = duckdb.connect

    def unconfigured(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.execute("SET threads=16")
        return conn

    monkeypatch.setattr(duckdb, "connect", unconfigured)

    class ControlledCandidate(ManualRepair):
        def _source_content(self, lease, row):
            threads = lease.call(
                "policy_test:threads",
                lambda conn: conn.execute("SELECT current_setting('threads')").fetchone(),
            )
            assert threads is not None and 1 <= threads[0] <= 8
            return super()._source_content(lease, row)

    assert ControlledCandidate(config).compact().replaced
