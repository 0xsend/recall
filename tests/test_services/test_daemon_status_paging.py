"""`daemon status` paging, projection and format flags against a real index.

A status page is a keyset over the source catalog. These run the CLI against
an in-process server over a scratch index (`lane_harness`), and against the
same index with no daemon, so a page answered either way can be compared.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from conftest import set_in_process_server
from lane_harness import Lane, build_lane
from recall.cli.app import app
from recall.cli.rpc import set_in_process_server as set_production_server
from recall.db.source_files import SourceCatalog, SourceSignature
from typer.testing import CliRunner

# Catalog rows the lane adds beside its two indexed claude_code transcripts, so
# a cursor that lands in the wrong place lands visibly in another source.
_EXTRA_ROWS = (
    ("codex", "/elsewhere/codex", "/elsewhere/codex/a.jsonl"),
    ("codex", "/elsewhere/codex", "/elsewhere/codex/b.jsonl"),
    ("grok", "/elsewhere/grok", "/elsewhere/grok/a.jsonl"),
)


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    value = build_lane(tmp_path, monkeypatch)
    catalog = SourceCatalog(value.server._get_conn(), clock=time.time)
    for source, root, path in _EXTRA_ROWS:
        catalog.observe(source, root, path, SourceSignature(1, 2, 3, 4, 5))
    set_in_process_server(value.server)
    yield value
    asyncio.run(value.server.stop())


def _status(*args: str) -> tuple[int, dict[str, Any]]:
    result = CliRunner().invoke(app, ["daemon", "status", "--json", *args])
    return result.exit_code, json.loads(result.stdout)


def _page(*args: str) -> tuple[list[str], str | None]:
    code, body = _status("--fields", "reconciliation", *args)
    assert code == 0, body
    reconciliation = body["reconciliation"]
    return [row["source_path"] for row in reconciliation["source_page"]], reconciliation[
        "next_cursor"
    ]


def _all_paths(lane: Lane) -> list[str]:
    return sorted(
        [str(lane.watched), str(lane.unwatched), *(path for _, _, path in _EXTRA_ROWS)],
        key=lambda path: (_source_of(lane, path), path),
    )


def _source_of(lane: Lane, path: str) -> str:
    return "claude_code" if path in (str(lane.watched), str(lane.unwatched)) else path.split("/")[2]


def _walk(*args: str) -> list[str]:
    paths, cursor = _page("--limit", "2", *args)
    while cursor is not None:
        more, cursor = _page("--limit", "2", "--cursor", cursor, *args)
        paths.extend(more)
    return paths


def test_cursors_the_daemon_did_not_issue_are_rejected(lane: Lane) -> None:
    first, cursor = _page("--limit", "1")
    assert first == [str(lane.watched)]
    assert cursor is not None
    mangled = {
        "visible form, separator lost": cursor.replace("\x1f", ""),
        "garbage": "garbage",
        "unknown source": "zzzz_unknown_source",
        "unknown source with a path": "zzzz\x1f/elsewhere/a.jsonl",
        "empty": "",
        "bare source": "codex",
        "source with no path": "codex\x1f",
        "truncated": cursor[: len(cursor) // 2],
        "last character dropped": cursor[:-1],
        "one-character typo in the path": cursor.replace("live.jsonl", "livf.jsonl"),
        "forged path in a known source": "codex\x1f/nonexistent/path.jsonl",
        "forged path sorting after a source": "claude_code\x1f~",
    }
    for case, value in mangled.items():
        code, body = _status("--limit", "1", "--cursor", value)
        assert code == 2, (case, body)
        assert body["error"]["code"] == "VALIDATION", case


def test_a_valid_cursor_pages_the_whole_catalog_once(lane: Lane) -> None:
    assert _walk() == _all_paths(lane)


def test_a_cursor_whose_row_left_the_catalog_is_rejected(lane: Lane) -> None:
    """Catalog rows are only removed by resetting the index, which ends any walk."""
    _, cursor = _page("--limit", "3")
    assert cursor == "codex\x1f/elsewhere/codex/a.jsonl"
    lane.server._get_conn().execute(
        "DELETE FROM source_files WHERE source_path = '/elsewhere/codex/a.jsonl'"
    )
    code, body = _status("--limit", "2", "--cursor", cursor)
    assert code == 2, body
    assert body["error"]["code"] == "VALIDATION"


def test_the_text_mode_cursor_round_trips_through_a_shell(lane: Lane) -> None:
    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text", "--limit", "1"])
    assert result.exit_code == 0, result.output
    (line,) = [line for line in result.stdout.splitlines() if "--cursor" in line]
    word = line.split("--cursor ", 1)[1]
    assert "\x1f" not in word
    pasted = subprocess.run(
        ["bash", "-c", f"printf %s {word}"], capture_output=True, check=True
    ).stdout.decode()
    assert _page("--limit", "1", "--cursor", pasted)[0] == [str(lane.unwatched)]


def test_offline_pages_equal_the_daemon_pages(lane: Lane) -> None:
    online = _walk()
    asyncio.run(lane.server.stop())
    set_production_server(None)

    assert _walk() == online
    code, body = _status("--fields", "reconciliation")
    assert code == 0, body
    assert body["reconciliation"]["rpc_ready"] is False
    assert len(body["reconciliation"]["source_page"]) == 5
    for forged in ("garbage", "codex\x1f/nonexistent/path.jsonl"):
        code, body = _status("--limit", "1", "--cursor", forged)
        assert code == 2, (forged, body)
        assert body["error"]["code"] == "VALIDATION"


@pytest.mark.parametrize(
    "older_schema",
    ["DROP TABLE source_files", "ALTER TABLE source_files DROP COLUMN normalization_checkpoint"],
)
def test_offline_status_without_catalog_schema(lane: Lane, older_schema: str) -> None:
    lane.server._get_conn().execute("DROP INDEX idx_source_files_source_path")
    lane.server._get_conn().execute(older_schema)
    asyncio.run(lane.server.stop())
    set_production_server(None)

    code, body = _status("--fields", "reconciliation")
    assert code == 0, body
    assert body["reconciliation"]["rpc_ready"] is False
    assert body["reconciliation"]["coverage"] is None
    assert body["reconciliation"]["source_page"] == []
    assert body["reconciliation"]["next_cursor"] is None
    assert "unavailable" in body["reconciliation"]["error"]

    for args in (("--limit", "1"), ("--cursor", "codex\x1f/elsewhere/codex/a.jsonl")):
        code, body = _status(*args)
        assert code == 1, body
        assert body["error"]["code"] == "RUNTIME"


def test_offline_named_page_fails_while_a_daemon_holds_the_database(
    lane: Lane, tmp_path: Path
) -> None:
    import os

    asyncio.run(lane.server.stop())
    set_production_server(None)
    lane.server._config.data_dir.joinpath("recall.pid").write_text(f"{os.getpid()}\n")

    code, body = _status("--limit", "1")

    assert code == 1, body
    assert body["error"]["code"] == "RUNTIME"


def test_every_emitted_field_is_projectable(lane: Lane) -> None:
    code, body = _status()
    assert code == 0, body
    for field in body:
        projected_code, projected = _status("--fields", field)
        assert projected_code == 0, (field, projected)
        assert set(projected) == {field}
    for key in body["reconciliation"]:
        projected_code, projected = _status("--fields", f"reconciliation.{key}")
        assert projected_code == 0, (key, projected)
        assert projected.keys() == {"reconciliation"}
        assert projected["reconciliation"].keys() == {key}


def test_nested_projection_combines_with_top_level_fields(lane: Lane) -> None:
    code, body = _status(
        "--fields", "embed_pending_held,reconciliation.enrichment_ready,reconciliation.pending"
    )
    assert code == 0, body
    assert set(body) == {"embed_pending_held", "reconciliation"}
    assert set(body["reconciliation"]) == {"enrichment_ready", "pending"}


def test_unknown_fields_still_fail_and_point_at_the_nested_field(lane: Lane) -> None:
    code, body = _status("--fields", "enrichment_ready")
    assert code == 2
    assert "reconciliation.enrichment_ready" in body["error"]["message"]
    code, body = _status("--fields", "reconciliation.enrichment_readyy")
    assert code == 2
    assert body["error"]["details"] == {"fields": ["reconciliation.enrichment_readyy"]}


@pytest.mark.parametrize("format_name", ["text", "jsonl", "toon", "bogus"])
@pytest.mark.parametrize("via_params", [False, True])
def test_json_with_another_format_is_a_validation_error(
    lane: Lane, format_name: str, via_params: bool
) -> None:
    args = (
        ["--params", json.dumps({"json": True, "format": format_name})]
        if via_params
        else ["--json", "--format", format_name]
    )
    result = CliRunner().invoke(app, ["daemon", "status", *args])
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"]["code"] == "VALIDATION"


def test_an_unsupported_format_is_a_validation_error_not_a_crash(lane: Lane) -> None:
    result = CliRunner().invoke(app, ["daemon", "status", "--format", "bogus"])
    assert result.exit_code == 2, result.output
    assert "unsupported output format: bogus" in result.output


@pytest.mark.parametrize("format_name", ["json", "auto"])
@pytest.mark.parametrize("via_params", [False, True])
def test_json_with_compatible_format_still_answers(
    lane: Lane, format_name: str, via_params: bool
) -> None:
    args = (
        ["--params", json.dumps({"json": True, "format": format_name})]
        if via_params
        else ["--json", "--format", format_name]
    )
    result = CliRunner().invoke(app, ["daemon", "status", *args, "--fields", "daemon_pid"])
    assert result.exit_code == 0, result.output
    assert set(json.loads(result.stdout)) == {"daemon_pid"}


def test_offline_text_names_readiness_unknown_rather_than_false(lane: Lane) -> None:
    asyncio.run(lane.server.stop())
    set_production_server(None)

    result = CliRunner().invoke(app, ["daemon", "status", "--format", "text"])

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[1] == "  ready: unknown (runtime coverage unavailable; daemon RPC was not reached)"
    (detail,) = [line for line in lines if line.startswith("Reconciliation: ")]
    assert detail == (
        "Reconciliation: rpc_ready=False, catalog_scan_complete=unknown, "
        "live_observation_ready=unknown, raw_indexing_ready=unknown, "
        "keyword_search_ready=unknown, enrichment_ready=unknown"
    )
