"""Physical interruption cannot split publication retirement, history and release."""

import asyncio
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pytest

from tests.test_services.test_repair_process import _diagnostic_copy
from tests.test_services.test_rpc_run_completion import _indexed_recompute_setup

_STAGES = (
    "after_completion_seed",
    "after_completion_retirement",
    "after_completion_history",
    "after_completion_counts",
    "after_completion_flags",
    "before_completion_commit",
    "after_completion_commit",
    "before_native_commit",
    "after_native_commit",
    "before_native_rollback",
    "after_native_rollback",
)


@pytest.mark.parametrize("kind", ["qualified", "partial", "failed", "cancelled"])
@pytest.mark.parametrize("stage", _STAGES)
@pytest.mark.parametrize("fault", ["kill", "error", "fatal"])
def test_native_completion_is_atomic_with_real_keyword_publication(
    tmp_path: Path, kind: str, stage: str, fault: str
):
    database, server = _indexed_recompute_setup(tmp_path)
    config = server._config
    asyncio.run(server.stop())
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "\n".join(f"[sources.{name}]\nroots=[]" for name in config.sources)
        + '\n[fts]\nbackend="sqlite_sidecar"\n'
    )
    result = subprocess.run(
        [sys.executable, "-m", "tests.completion_process", kind, stage, fault],
        env={
            **os.environ,
            "RECALL_CONFIG_PATH": str(config_file),
            "RECALL_DATA_DIR": str(config.data_dir),
            "RECALL_DB_PATH": str(database),
            "RECALL_LOCK_PATH": str(config.lock_path),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert f"checkpoint:{stage}" in result.stdout, result.stderr
    assert result.returncode == (-signal.SIGKILL if fault == "kill" else 1), result.stderr
    try:
        fresh = duckdb.connect(str(database), read_only=True)
    except duckdb.IOException:
        fresh = _diagnostic_copy(database, tmp_path / "diagnosis")
    committed = stage in {"after_completion_commit", "after_native_commit"}
    with fresh:
        history = fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone()
        assert history is not None
        successful, failure = history
        released = committed and kind == "qualified"
        assert successful == (datetime(2026, 11, 3) if released else datetime(2026, 11, 1))
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (released, released)
        pending = fresh.execute(
            "SELECT kind,id,op,queued_at FROM fts_sidecar_pending ORDER BY ALL"
        ).fetchall()
        if committed and kind in {"qualified", "partial"}:
            assert pending == []
        else:
            assert json.loads(json.dumps(pending, default=str)) == json.loads(
                (config.data_dir / "published.json").read_text()
            )
        if committed:
            assert (
                failure
                == {
                    "qualified": None,
                    "partial": "incomplete inventory",
                    "failed": "producer failed",
                    "cancelled": "producer cancelled",
                }[kind]
            )
