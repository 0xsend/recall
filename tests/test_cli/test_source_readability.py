"""Unreadable source diagnostics through the CLI, service, and real filesystem."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from recall.cli.app import app
from typer.testing import CliRunner

from conftest import set_in_process_server
from tests.test_services.lane_harness import WATCHED_SESSION, build_lane


@pytest.mark.parametrize("cause", ["missing", "permission", "directory", "empty"])
@pytest.mark.parametrize("command", ["show", "show-fresh", "live"])
@pytest.mark.parametrize("output_format", ["text", "json"])
def test_source_readability_and_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cause: str,
    command: str,
    output_format: str,
) -> None:
    lane = build_lane(tmp_path, monkeypatch)
    set_in_process_server(lane.server)
    monkeypatch.setattr(lane.server, "_watch_runtime", lane.watch_runtime([lane.watched]))
    # An empty transcript has only its filename to supply the source session id.
    lane.watched.write_text(lane.watched.read_text().replace(WATCHED_SESSION, lane.watched.stem))
    asyncio.run(lane.server.index_session_now(lane.watched))
    initial = CliRunner().invoke(app, ["show", lane.watched.stem, "--json"])
    assert initial.exit_code == 0, initial.output
    session_id = json.loads(initial.stdout)["id"]
    saved = lane.watched.read_bytes()
    if cause == "missing":
        lane.watched.unlink()
    elif cause == "permission":
        lane.watched.chmod(0)
    elif cause == "directory":
        lane.watched.unlink()
        lane.watched.mkdir()
    else:
        lane.watched.write_bytes(b"")

    args = ["live", "--all"] if command == "live" else ["show", session_id]
    if command == "show-fresh":
        args.append("--fresh")
    args += ["--format", output_format]
    try:
        if cause == "permission":
            with pytest.raises(PermissionError):
                lane.watched.read_bytes()
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        if output_format == "json":
            payload = json.loads(result.stdout)
            if command == "live":
                payload = next(row for row in payload["sessions"] if row["id"] == session_id)
            freshness = payload["freshness"]
            assert ("source_unreadable" in freshness["limitations"]) is (cause != "empty")
            if cause != "empty":
                assert freshness["current"] is False
                assert freshness["validated"] is False
                assert freshness["lag_bytes"] is None
                assert freshness["file_size"] is None
                if command != "live":
                    assert payload["message_count"] == 2
                    assert [m["content"] for m in payload["messages"]] == [
                        "run the test suite",
                        "Running it now.",
                    ]
            elif command == "show-fresh":
                assert freshness["current"] is True
                assert payload["message_count"] == 0
        else:
            diagnostic = "transcript file could not be read; showing stored content."
            stream = result.stdout if command == "live" else result.stderr
            assert (diagnostic in stream) is (cause != "empty")
            if cause != "empty":
                assert "Check that the transcript file exists and is readable." in stream
                if command == "live":
                    row = next(line for line in result.stdout.splitlines() if session_id in line)
                    assert "lag=" not in row
                else:
                    assert "run the test suite" in result.stdout
                    assert "Running it now." in result.stdout
    finally:
        if lane.watched.is_dir():
            lane.watched.rmdir()
        elif lane.watched.exists():
            lane.watched.chmod(0o644)
        lane.watched.write_bytes(saved)
        try:
            restored = CliRunner().invoke(app, ["show", session_id, "--fresh", "--json"])
            assert restored.exit_code == 0, restored.output
            payload = json.loads(restored.stdout)
            assert payload["freshness"]["current"] is True
            assert "source_unreadable" not in payload["freshness"]["limitations"]
            assert payload["message_count"] == 2
        finally:
            asyncio.run(lane.server.stop())
