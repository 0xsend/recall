"""Manual reconstruction refuses companion data it cannot preserve."""

from contextlib import closing
from pathlib import Path

import duckdb
import pytest
from recall.db.fts_sidecar import open_sidecar, sidecar_path
from recall.db.repair import load_repair_row
from recall.db.schema import ensure_schema
from recall.services.manual_repair import ManualRepair
from recall.services.self_repair import DaemonStartupRefused

from tests.test_services.test_run_completion import _config


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize(
    "statement,query,expected",
    [
        (
            "CREATE TABLE acknowledged AS SELECT 'keep' AS value",
            "SELECT * FROM acknowledged",
            [("keep",)],
        ),
        (
            "ALTER TABLE bootstrap_progress ADD COLUMN acknowledged TEXT DEFAULT 'keep'",
            "SELECT name FROM pragma_table_info('bootstrap_progress') WHERE name='acknowledged'",
            [("acknowledged",)],
        ),
        (
            "INSERT INTO fts_sidecar_meta VALUES ('acknowledged','keep')",
            "SELECT value FROM fts_sidecar_meta WHERE key='acknowledged'",
            [("keep",)],
        ),
        (
            "INSERT INTO bootstrap_progress VALUES ('acknowledged','keep',NULL)",
            "SELECT last_id FROM bootstrap_progress WHERE kind='acknowledged'",
            [("keep",)],
        ),
    ],
    ids=["table", "column", "metadata", "progress"],
)
def test_unknown_sidecar_content_holds_before_manual_arm(
    tmp_path: Path, compact: bool, statement: str, query: str, expected: list
):
    config = _config(tmp_path, None)
    with duckdb.connect(str(config.db_path)) as conn:
        ensure_schema(conn)
    with closing(open_sidecar(sidecar_path(config.data_dir))) as sidecar:
        sidecar.execute(statement)
        sidecar.commit()
    repair = ManualRepair(config)
    with pytest.raises(DaemonStartupRefused, match=r"unsupported.*sidecar"):
        repair.compact() if compact else repair.execute()
    with duckdb.connect(str(config.db_path), read_only=True) as conn:
        row = load_repair_row(conn)
        assert row.phase == "idle" and row.attempt_no == 0
    with closing(open_sidecar(sidecar_path(config.data_dir))) as sidecar:
        assert sidecar.execute(query).fetchall() == expected
