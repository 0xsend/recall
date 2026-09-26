"""`recall db` maintenance commands and the daemon-status divergence surface.

REQ-RESIL-017: `db rebuild-indexes` rebuilds every index but refuses while the
daemon holds the database. REQ-RESIL-018: `db check-indexes` and `daemon status`
make index/table divergence visible without reading the log.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import duckdb
import pytest
from conftest import _can_acquire_duckdb_lock
from recall.cli.app import app
from recall.cli.contract import CliError, ErrorCode
from recall.cli.manifest import command_manifest, output_fields_for
from recall.core.config import AppConfig
from recall.db import advisory_lock, connect
from typer.testing import CliRunner

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)


def _init_empty_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / ".config/recall/config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    monkeypatch.setenv("RECALL_EMBED_BACKEND", "onnx")
    config = AppConfig.load()
    conn = connect(config)
    conn.close()
    return config


def _no_daemon(*_args: Any, **_kwargs: Any) -> Any:
    raise CliError(code=ErrorCode.RUNTIME, message="daemon not running", exit_code=1)


def _report_dict(*, diverged: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "checked_at": "2026-08-30T12:00:00",
        "indexes_probed": 13,
        "samples_checked": 20,
        "samples_unverifiable": 3,
        "diverged": diverged,
        "diverged_count": len(diverged),
        "complete": True,
        "elapsed_seconds": 0.42,
    }


class TestManifest:
    def test_db_commands_are_in_the_manifest(self) -> None:
        rebuild = command_manifest(["db", "rebuild-indexes"])
        check = command_manifest(["db", "check-indexes"])

        assert rebuild["safety"] == {"mutates": True, "destructive": False, "idempotent": True}
        assert {"dropped", "created", "healed", "elapsed_seconds"} <= set(
            rebuild["output"]["fields"]
        )
        assert check["safety"]["mutates"] is False
        assert {"diverged", "diverged_count", "complete"} <= set(check["output"]["fields"])
        assert any(opt["name"] == "sample" for opt in check["options"])

    def test_daemon_status_manifest_exposes_index_divergence(self) -> None:
        assert "index_divergence" in (output_fields_for("daemon status") or set())

    def test_daemon_status_manifest_exposes_startup_refusal(self) -> None:
        assert "startup_refusal" in (output_fields_for("daemon status") or set())

    def test_daemon_status_manifest_exposes_the_live_daemon_fields(self) -> None:
        fields = output_fields_for("daemon status") or set()
        assert {"daemon_pid", "runtime_unavailable_reason"} <= fields


@_requires_duckdb_lock
class TestRebuildIndexes:
    def test_json_reports_counts_and_heals_missing_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _init_empty_db(tmp_path, monkeypatch)
        conn = duckdb.connect(str(config.db_path))
        conn.execute("DROP INDEX idx_sessions_source")
        conn.close()

        result = CliRunner().invoke(app, ["db", "rebuild-indexes", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["healed"] == ["idx_sessions_source"]
        assert payload["created"] == payload["dropped"] + 1
        assert payload["dropped"] > 0
        assert isinstance(payload["elapsed_seconds"], float)

    def test_text_output_names_the_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _init_empty_db(tmp_path, monkeypatch)

        result = CliRunner().invoke(app, ["db", "rebuild-indexes", "--format", "text"])

        assert result.exit_code == 0, result.output
        assert "Rebuilt" in result.stdout
        assert "healed 0" in result.stdout

    def test_refuses_while_another_holder_owns_the_advisory_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _init_empty_db(tmp_path, monkeypatch)
        acquired = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with advisory_lock(config.lock_path):
                acquired.set()
                release.wait(timeout=10)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        assert acquired.wait(timeout=5)
        try:
            result = CliRunner().invoke(app, ["db", "rebuild-indexes", "--json"])
        finally:
            release.set()
            holder.join(timeout=5)

        assert result.exit_code == 2, result.output
        payload = json.loads(result.stdout)
        assert payload["error"]["code"] == "RUNTIME"
        assert "recall daemon stop" in payload["error"]["message"]

    def test_refuses_on_duckdb_lock_conflict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _init_empty_db(tmp_path, monkeypatch)

        def conflict(*_args: Any, **_kwargs: Any) -> Any:
            raise duckdb.IOException(
                'IO Error: Could not set lock on file "recall.duckdb": Conflicting lock is held '
                "in /usr/bin/python3 (PID 4242)"
            )

        monkeypatch.setattr("recall.cli.db.connect", conflict)

        result = CliRunner().invoke(app, ["db", "rebuild-indexes", "--json"])

        assert result.exit_code == 2, result.output
        payload = json.loads(result.stdout)
        assert "recall daemon stop" in payload["error"]["message"]

    def test_rebuild_clears_fatal_memory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from recall.services.runtime_state import load_runtime_status
        from recall.services.self_repair import (
            read_failure_marker,
            remember_fatal_failure,
        )

        config = _init_empty_db(tmp_path, monkeypatch)
        remember_fatal_failure(
            config.data_dir,
            duckdb.FatalException("Failed to delete all rows from index. Only deleted 0 out of 1"),
            site="catch_up",
        )

        result = CliRunner().invoke(app, ["db", "rebuild-indexes", "--json"])

        assert result.exit_code == 0, result.output
        assert read_failure_marker(config.data_dir) is None
        assert load_runtime_status(config).last_fatal_signature is None


@_requires_duckdb_lock
class TestCheckIndexes:
    def test_falls_back_to_local_probe_when_no_daemon(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _init_empty_db(tmp_path, monkeypatch)
        monkeypatch.setattr("recall.cli.db.rpc_call_or_error", _no_daemon)

        result = CliRunner().invoke(app, ["db", "check-indexes", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["diverged"] == []
        assert payload["diverged_count"] == 0
        assert payload["complete"] is True
        assert payload["indexes_probed"] > 0

    def test_uses_rpc_when_daemon_is_up_and_exits_1_on_divergence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _init_empty_db(tmp_path, monkeypatch)
        calls: list[tuple[str, dict[str, Any]]] = []
        diverged = [
            {
                "table": "session_state",
                "column": "git_repo",
                "key": "/Users/dev/code/app",
                "index_count": 1790,
                "full_count": 1797,
            }
        ]

        def rpc(method: str, params: dict[str, Any], **_kwargs: Any) -> Any:
            calls.append((method, params))
            return _report_dict(diverged=diverged)

        monkeypatch.setattr("recall.cli.db.rpc_call_or_error", rpc)

        result = CliRunner().invoke(app, ["db", "check-indexes", "--sample", "3", "--json"])

        assert result.exit_code == 1, result.output
        assert calls == [("recall.check_indexes", {"sample": 3})]
        payload = json.loads(result.stdout)
        assert payload["diverged"] == diverged
        assert payload["diverged_count"] == 1

    def test_text_output_names_diverged_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _init_empty_db(tmp_path, monkeypatch)
        diverged = [
            {
                "table": "session_state",
                "column": "cwd",
                "key": "/Users/dev/app.worktrees/dev",
                "index_count": 450,
                "full_count": 459,
            }
        ]
        monkeypatch.setattr(
            "recall.cli.db.rpc_call_or_error",
            lambda *_a, **_k: _report_dict(diverged=diverged),
        )

        result = CliRunner().invoke(app, ["db", "check-indexes", "--format", "text"])

        assert result.exit_code == 1, result.output
        assert "session_state.cwd" in result.stdout
        assert "450" in result.stdout and "459" in result.stdout
        assert "recall db rebuild-indexes" in result.stdout

    def test_rejects_out_of_range_sample(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _init_empty_db(tmp_path, monkeypatch)
        monkeypatch.setattr("recall.cli.db.rpc_call_or_error", _no_daemon)

        result = CliRunner().invoke(app, ["db", "check-indexes", "--sample", "0", "--json"])

        assert result.exit_code == 2, result.output
        assert json.loads(result.stdout)["error"]["code"] == "VALIDATION"

    def test_local_lock_conflict_points_at_daemon_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _init_empty_db(tmp_path, monkeypatch)
        monkeypatch.setattr("recall.cli.db.rpc_call_or_error", _no_daemon)

        def conflict(*_args: Any, **_kwargs: Any) -> Any:
            raise duckdb.IOException("Conflicting lock is held in recall (PID 99)")

        monkeypatch.setattr("recall.cli.db.connect_readonly", conflict)

        result = CliRunner().invoke(app, ["db", "check-indexes", "--json"])

        assert result.exit_code == 2, result.output
        assert "recall daemon status" in json.loads(result.stdout)["error"]["message"]


class TestDaemonStatusSurface:
    @pytest.fixture(autouse=True)
    def _config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "recall.cli.daemon.AppConfig.load",
            lambda: type("Config", (), {"daemon": type("Daemon", (), {"interval": 300})()})(),
        )

    @staticmethod
    def _status(index_divergence: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "configured_scheduler": "auto",
            "scheduler": None,
            "installed": False,
            "command": "/tmp/recall daemon --once",
            "config_path": "/tmp/config.toml",
            "artifact_paths": [],
            "runtime_status": {
                "last_failure_message": None,
                "last_fatal_signature": None,
                "fatal_repeat_count": 0,
                "last_index_repair_at": None,
                "needs_index_verification": False,
            },
            "mode": "poll",
            "resolved_mode": "poll",
            "daemon_version": "0.29.1",
            "binary_version": "0.29.1",
            "version_drift": False,
            "index_divergence": index_divergence,
            "startup_refusal": None,
        }

    def test_json_carries_index_divergence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = _report_dict(diverged=[])
        monkeypatch.setattr(
            "recall.cli.daemon.rpc_call_or_error", lambda *_a, **_k: self._status(report)
        )

        result = CliRunner().invoke(app, ["daemon", "status", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["index_divergence"]["indexes_probed"] == 13
        assert payload["index_divergence"]["diverged"] == []

    def test_text_and_stderr_notice_name_divergence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        diverged = [
            {
                "table": "session_state",
                "column": "git_repo",
                "key": "/Users/dev/code/app",
                "index_count": 1790,
                "full_count": 1797,
            }
        ]
        status = self._status(_report_dict(diverged=diverged))
        monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", lambda *_a, **_k: status)

        text = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])
        structured = CliRunner().invoke(app, ["daemon", "status", "--json"])

        assert text.exit_code == 0, text.output
        assert "Index divergence: 1 diverged" in text.stdout
        assert structured.exit_code == 0, structured.output
        assert "Warning" not in structured.stdout
        assert "index" in structured.stderr.lower()
        assert "recall db rebuild-indexes" in structured.stderr

    def test_status_names_the_daemon_it_is_talking_to_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-DAEMON-074: a status the daemon answered names the serving pid,
        not only the pid the local fallback infers from the pid file — and the
        summary header is the one place that says it."""
        status = self._status(None)
        status["daemon_pid"] = 65965
        monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", lambda *_a, **_k: status)

        text = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

        assert text.exit_code == 0, text.output
        assert "pid=65965" in text.stdout
        assert text.stdout.count("65965") == 1

    def test_status_reports_a_starting_daemon_when_rpc_is_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-DAEMON-074: with the RPC not answering and a live daemon holding
        the database, status is a report of that state (exit 0), not a DuckDB
        lock error."""

        def rpc_down(*_a: object, **_k: object) -> dict[str, Any]:
            raise CliError(code=ErrorCode.RUNTIME, message="connection refused", exit_code=1)

        local = self._status(None)
        # The real fleet state: installed, watch mode, runtime fields unread.
        local["installed"] = True
        local["resolved_mode"] = "watch"
        local["daemon_pid"] = 4242
        local["runtime_unavailable_reason"] = "daemon pid 4242 holds the database"
        monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", rpc_down)
        monkeypatch.setattr("recall.cli.daemon.daemon_status", lambda *_a, **_k: local)

        text = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])
        structured = CliRunner().invoke(app, ["daemon", "status", "--json"])

        assert text.exit_code == 0, text.output
        assert "pid 4242" in text.stdout
        assert "RPC not answering" in text.stdout
        assert structured.exit_code == 0, structured.output
        payload = json.loads(structured.stdout)
        assert payload["daemon_pid"] == 4242
        assert payload["runtime_unavailable_reason"] == "daemon pid 4242 holds the database"
        # The unread defaults must not be read as "never ran": that notice names
        # `recall daemon install`, which would restart the daemon mid-startup.
        assert "has not completed a successful run" not in structured.stderr
        assert "recall daemon install" not in structured.stderr

        local["daemon_pid"] = None
        local["runtime_unavailable_reason"] = (
            "another process (PID 77) holds the database read-write"
        )
        text = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])
        assert text.exit_code == 0, text.output
        assert "Database: another process (PID 77)" in text.stdout
        assert "RPC not answering" not in text.stdout

    def test_startup_refusal_shows_in_text_json_and_stderr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-024: while the refusal marker holds the scheduler down, the
        only thing the operator can run is `daemon status`; it must name the
        refusal and the fix on every surface."""
        status = self._status(None)
        status["startup_refusal"] = (
            "refusing to start: DuckDB index/table divergence recurred after an index rebuild. "
            "Manual fix: run `recall daemon stop`, then `recall db rebuild-indexes`."
        )
        monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", lambda *_a, **_k: status)

        text = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])
        structured = CliRunner().invoke(app, ["daemon", "status", "--json"])

        assert text.exit_code == 0, text.output
        assert "Startup refused:" in text.stdout
        assert structured.exit_code == 0, structured.output
        assert json.loads(structured.stdout)["startup_refusal"].startswith("refusing to start")
        assert "refused" in structured.stderr.lower()
        assert "recall db rebuild-indexes" in structured.stderr

    def test_text_shows_last_fatal_and_repair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        status = self._status(None)
        status["runtime_status"] = {
            "last_failure_message": "fatal DuckDB invalidation in usage_harvest: ...",
            "last_fatal_signature": "usage_harvest:FatalException:Failed to delete all rows",
            "fatal_repeat_count": 2,
            "last_index_repair_at": "2026-08-30T12:00:30",
            "needs_index_verification": True,
        }
        monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", lambda *_a, **_k: status)

        result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

        assert result.exit_code == 0, result.output
        assert "Last fatal:" in result.stdout
        assert "x2" in result.stdout
        assert "Index repair: 2026-08-30T12:00:30" in result.stdout
        assert "needs verification" in result.stdout
