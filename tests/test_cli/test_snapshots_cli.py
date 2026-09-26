from __future__ import annotations

import json
import os
import time
from pathlib import Path

import recall.services.snapshots as snapshots_module
from recall.cli.app import app
from recall.services.snapshots import RECALL_SIDECAR_INTRODUCED_AT, SnapshotGcResult
from typer.testing import CliRunner


def _setup_env(tmp_path: Path, monkeypatch) -> Path:
    data_dir = tmp_path / ".local" / "share" / "recall"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(data_dir))
    return data_dir


def _age(path: Path, *, days: int) -> None:
    when = time.time() - (days * 86_400)
    os.utime(path, (when, when), follow_symlinks=False)


def _seed_partial_snapshot(data_dir: Path, monkeypatch) -> Path:
    now = RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 10_000
    monkeypatch.setattr(snapshots_module.time, "time", lambda: now)
    snapshots = data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    partial = snapshots / "partial.duckdb"
    partial.write_text("partial")
    os.utime(partial, (now - 1, now - 1), follow_symlinks=False)
    return partial


def test_snapshots_gc_dry_run_reports_without_removing(tmp_path: Path, monkeypatch) -> None:
    data_dir = _setup_env(tmp_path, monkeypatch)
    snapshots = data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    stale = snapshots / "stale"
    stale.write_text("stale")
    _age(stale, days=10)

    result = CliRunner().invoke(app, ["snapshots", "gc", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["removed_paths"] == [str(stale)]
    assert stale.exists()


def test_snapshots_gc_yes_removes_stale_entries(tmp_path: Path, monkeypatch) -> None:
    data_dir = _setup_env(tmp_path, monkeypatch)
    snapshots = data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    stale = snapshots / "stale"
    stale.write_text("stale")
    _age(stale, days=10)

    result = CliRunner().invoke(app, ["snapshots", "gc", "--days", "7", "--yes", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["removed_paths"] == [str(stale)]
    assert payload["total_bytes_freed"] == 5
    assert not stale.exists()


def test_snapshots_gc_missing_dir_reports_noop(tmp_path: Path, monkeypatch) -> None:
    _setup_env(tmp_path, monkeypatch)

    result = CliRunner().invoke(app, ["snapshots", "gc", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["snapshots_dir_missing"] is True
    assert payload["removed_paths"] == []


def test_snapshots_gc_text_reports_failed_paths(monkeypatch) -> None:
    def gc_snapshots_stub(*_args, **_kwargs) -> SnapshotGcResult:
        return SnapshotGcResult(failed_paths=("foo", "bar"))

    monkeypatch.setattr("recall.cli.snapshots.gc_snapshots", gc_snapshots_stub)

    result = CliRunner().invoke(app, ["snapshots", "gc", "--yes", "--format", "text"])

    assert result.exit_code == 0, result.output
    assert "Failed: 2" in result.stdout
    assert "  - foo" in result.stdout
    assert "  - bar" in result.stdout


def test_snapshots_gc_reports_partial_paths_in_json_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = _setup_env(tmp_path, monkeypatch)
    partial = _seed_partial_snapshot(data_dir, monkeypatch)

    result = CliRunner().invoke(app, ["snapshots", "gc", "--json", "--yes", "--days", "0"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["partial_paths"] == [str(partial)]
    assert payload["removed_paths"] == []
    assert partial.exists()


def test_snapshots_gc_text_warns_about_partial_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = _setup_env(tmp_path, monkeypatch)
    partial = _seed_partial_snapshot(data_dir, monkeypatch)

    result = CliRunner().invoke(
        app,
        ["snapshots", "gc", "--yes", "--days", "0", "--format", "text"],
    )

    assert result.exit_code == 0, result.output
    assert "Partial snapshots: 1" in result.stdout
    assert f"  - {partial}" in result.stdout
    assert partial.exists()


# --- read-only listing (REQ-RECALL-0145-H4) ------------------------------


def test_snapshots_list_reports_size_age_and_retention(tmp_path: Path, monkeypatch) -> None:
    """An operator asking what is on disk must not have to ask `gc` what it would delete."""
    data_dir = _setup_env(tmp_path, monkeypatch)
    snapshots = data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    stale = snapshots / "stale"
    stale.write_text("stale")
    _age(stale, days=10)
    backup = snapshots / "index-migration-abc"
    backup.mkdir()
    (backup / "recall.duckdb").write_text("db")
    _age(backup, days=30)

    result = CliRunner().invoke(app, ["snapshots", "list", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["snapshots_dir"] == str(snapshots)
    assert payload["entry_count"] == 2
    assert payload["total_bytes"] == 7
    by_path = {entry["paths"][0]: entry for entry in payload["entries"]}
    assert by_path[str(stale)]["total_bytes"] == 5
    assert by_path[str(stale)]["retained"] is False
    assert 10 * 86_400 <= by_path[str(stale)]["age_seconds"] < 11 * 86_400
    # The recovery backup gc never prunes is exactly the entry a rollback needs.
    assert by_path[str(backup)]["retained"] is True
    assert by_path[str(backup)]["total_bytes"] == 2
    # Oldest first.
    assert payload["entries"][0]["paths"] == [str(backup)]


def test_snapshots_list_removes_nothing(tmp_path: Path, monkeypatch) -> None:
    data_dir = _setup_env(tmp_path, monkeypatch)
    snapshots = data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    stale = snapshots / "stale"
    stale.write_text("stale")
    _age(stale, days=400)

    result = CliRunner().invoke(app, ["snapshots", "list", "--json"])

    assert result.exit_code == 0, result.output
    assert stale.exists()


def test_snapshots_list_marks_a_partial_snapshot(tmp_path: Path, monkeypatch) -> None:
    """gc refuses partials; a listing that hid them would explain nothing."""
    data_dir = _setup_env(tmp_path, monkeypatch)
    partial = _seed_partial_snapshot(data_dir, monkeypatch)

    result = CliRunner().invoke(app, ["snapshots", "list", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [entry["paths"] for entry in payload["entries"]] == [[str(partial)]]
    assert payload["entries"][0]["partial"] is True


def test_snapshots_list_missing_dir_reports_noop(tmp_path: Path, monkeypatch) -> None:
    _setup_env(tmp_path, monkeypatch)

    result = CliRunner().invoke(app, ["snapshots", "list", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["snapshots_dir_missing"] is True
    assert payload["entries"] == []


def test_snapshots_list_text_summarises_each_entry(tmp_path: Path, monkeypatch) -> None:
    data_dir = _setup_env(tmp_path, monkeypatch)
    snapshots = data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    backup = snapshots / "index-migration-abc"
    backup.mkdir()
    (backup / "recall.duckdb").write_text("db")
    _age(backup, days=30)

    result = CliRunner().invoke(app, ["snapshots", "list", "--format", "text"])

    assert result.exit_code == 0, result.output
    assert "Snapshots: 1 entry" in result.stdout
    assert str(backup) in result.stdout
    assert "30d" in result.stdout
    assert "retained" in result.stdout


def test_snapshots_list_is_registered_in_the_manifest() -> None:
    """REQ-CLI-001: an agent discovers the command and its fields from the manifest."""
    from recall.cli.manifest import COMMANDS, output_fields_for

    assert COMMANDS["snapshots list"]["safety"] == {
        "mutates": False,
        "destructive": False,
        "idempotent": True,
    }
    fields = output_fields_for("snapshots list")
    assert fields is not None
    assert {"entries", "entry_count", "total_bytes", "snapshots_dir"} <= fields
