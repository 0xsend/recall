"""`recall index --json` keeps stdout for the payload and puts progress on stderr.

REQ-INDEX-007 / REQ-INDEX-024: a structured run used to drop progress entirely,
so an operator watching a long `--json` index had no signal at all between the
command starting and its summary. Progress now goes to stderr, where it cannot
reach the payload, and stays bounded: one line per frame the daemon sends.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from recall.cli.app import app
from typer.testing import CliRunner

SUMMARY = {
    "total": 8,
    "indexed": 0,
    "skipped": 8,
    "failed": 0,
    "changed": 0,
    "total_seconds": 0.25,
    "backlog_pending": 0,
}


def _frames() -> list[dict[str, Any]]:
    scanning = [
        {
            "processed": 0,
            "total": 0,
            "status": "scanning",
            "indexed": 0,
            "skipped": 0,
            "failed": 0,
            "inventory_complete": False,
        }
        for _ in range(3)
    ]
    return [
        *scanning,
        {
            "processed": 8,
            "total": 8,
            "status": "done",
            "indexed": 0,
            "skipped": 8,
            "failed": 0,
            "inventory_complete": True,
        },
    ]


@pytest.fixture
def scratch_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))


def _install_fake_rpc(monkeypatch, seen: dict[str, Any]):
    def fake_rpc(method: str, params: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        seen["method"] = method
        on_progress = kwargs.get("on_progress")
        seen["on_progress"] = on_progress
        if on_progress is not None:
            for frame in _frames():
                on_progress(frame)
        return SUMMARY

    monkeypatch.setattr("recall.cli.index.rpc_call_or_error", fake_rpc)


def test_json_index_emits_one_payload_on_stdout_and_bounded_progress_on_stderr(
    scratch_home, monkeypatch
) -> None:
    seen: dict[str, Any] = {}
    _install_fake_rpc(monkeypatch, seen)

    result = CliRunner().invoke(app, ["index", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["skipped"] == 8

    assert seen["on_progress"] is not None, (
        "a --json run passed no progress callback, so nothing can reach stderr"
    )
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert lines, "bounded progress must remain observable on stderr"
    assert len(lines) == len(_frames()), (
        f"stderr carried {len(lines)} lines for {len(_frames())} progress frames"
    )
    assert not any(line.lstrip().startswith("{") for line in lines), (
        "progress must not look like a second JSON payload"
    )


def test_no_progress_keeps_a_json_run_silent_on_stderr(scratch_home, monkeypatch) -> None:
    seen: dict[str, Any] = {}
    _install_fake_rpc(monkeypatch, seen)

    result = CliRunner().invoke(app, ["index", "--json", "--no-progress"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["skipped"] == 8
    assert seen["on_progress"] is None
    assert result.stderr.strip() == ""
