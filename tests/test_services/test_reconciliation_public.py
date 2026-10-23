"""Public contract counterexamples against the frozen iteration31 candidate."""

from __future__ import annotations

import asyncio
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from lane_harness import LANE_NOW, build_lane
from recall.core.rpc_types import serialize_rpc_value
from recall.parsers.claude_code import ClaudeCodeParser
from recall.services.coordinator import capture_path, observe_path


@pytest.fixture
def lane(tmp_path, monkeypatch):
    value = build_lane(tmp_path, monkeypatch)
    yield value
    asyncio.run(value.server.stop())


def test_poll_roster_pages_include_observed_unindexed_paths(lane):
    path = lane.watched.with_name("new-unindexed.jsonl")
    path.write_text(lane.watched.read_text().replace("live-mid-tool", "public-unindexed"))
    parser = ClaudeCodeParser()
    observe_path(parser, capture_path(parser, path), conn=lane.server._get_conn())

    async def read_pages():
        cursor = None
        paths = []
        for _ in range(5):
            params = {"all": True, "limit": 1}
            if cursor:
                params["cursor"] = cursor
            page = serialize_rpc_value(await lane.server._handle_live(params, None))
            assert len(page["sessions"]) <= 1
            paths.extend(row["path"] for row in page["sessions"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert str(path) in paths, "public roster silently omitted a known unindexed source"
        assert len(paths) == len(set(paths))

    asyncio.run(read_pages())


def test_show_freshness_ignores_the_parser_build_that_indexed_it(lane):
    conn = lane.server._get_conn()
    session_id = conn.execute(
        "SELECT session_id FROM source_files WHERE source_path = ?", [str(lane.watched)]
    ).fetchone()[0]
    conn.execute(
        "UPDATE source_files SET parser_revision = 'obsolete-parser' WHERE source_path = ?",
        [str(lane.watched)],
    )
    result = serialize_rpc_value(
        asyncio.run(lane.server._handle_show({"session_id": session_id}, None))
    )
    assert result["freshness"]["current"] is True, (
        "an older parser build made indexed content stale"
    )


def test_show_freshness_compares_all_current_sidecars(lane, tmp_path):
    root = tmp_path / ".grok" / "sessions"
    shutil.copytree(Path(__file__).resolve().parents[1] / "fixtures/grok/with_sidecars", root)
    path = next(root.rglob("chat_history.jsonl"))
    asyncio.run(lane.server.index_session_now(path))
    session_id = (
        lane.server._get_conn()
        .execute("SELECT session_id FROM source_files WHERE source_path = ?", [str(path)])
        .fetchone()[0]
    )
    assert session_id is not None
    sidecar = path.with_name("signals.json")
    sidecar.write_text(sidecar.read_text() + "\n")
    result = serialize_rpc_value(
        asyncio.run(lane.server._handle_show({"session_id": session_id}, None))
    )
    assert result["freshness"]["current"] is False, "changed sidecar inputs were reported current"


def test_absent_optional_roots_do_not_prevent_raw_readiness(lane):
    async def status():
        await lane.server._handle_daemon_run({"once": True, "embed": False}, None)
        return await lane.server._handle_daemon_status({}, None)

    result = asyncio.run(status())
    assert result["reconciliation"]["pending"] == 0
    assert result["reconciliation"]["raw_indexing_ready"] is True


def test_foreign_historical_catalog_rows_do_not_block_configured_readiness(lane):
    import time
    from dataclasses import replace

    from recall.core.config import SourceConfig
    from recall.db.source_files import SourceCatalog, SourceSignature

    config = lane.server._config
    lane.server._config = replace(
        config,
        sources={
            "claude_code": SourceConfig(roots=(lane.watched.parent,)),
            "codex": SourceConfig(roots=()),
            "pi_agent": SourceConfig(roots=()),
            "grok": SourceConfig(roots=()),
            "kimi_code": SourceConfig(roots=()),
        },
    )
    SourceCatalog(lane.server._get_conn(), clock=time.time).observe(
        "codex", "/foreign", "/foreign/old.jsonl", SourceSignature(1, 2, 3, 4, 5)
    )

    async def status():
        await lane.server._handle_daemon_run({"once": True, "embed": False}, None)
        return await lane.server._handle_daemon_status({}, None)

    result = asyncio.run(status())
    assert result["reconciliation"]["raw_indexing_ready"] is True


def test_filtered_roster_counts_unknown_metadata_but_keeps_known_source(lane):
    parser = ClaudeCodeParser()
    for number in range(24):
        path = lane.watched.with_name(f"unindexed-{number}.jsonl")
        path.write_text("{}\n")
        observe_path(parser, capture_path(parser, path), conn=lane.server._get_conn())

    async def read():
        page = serialize_rpc_value(
            await lane.server._handle_live({"source": "claude-code", "limit": 1}, None)
        )
        assert page["coverage"]["unindexed_paths"] == 24
        assert page["coverage"]["unknown_count"] == 0
        assert page["next_cursor"] is not None
        filtered = serialize_rpc_value(
            await lane.server._handle_live({"project": "unknown-project", "limit": 1}, None)
        )
        assert filtered["sessions"] == []
        assert filtered["coverage"]["unknown_count"] == 24
        assert len(filtered["coverage"]["filtered_unknown_paths"]) == 16
        assert filtered["coverage"]["complete"] is False

    asyncio.run(read())


@pytest.mark.parametrize("elapsed_seconds, expected_sessions", [(0, 2), (301, 0)])
def test_live_coverage_becomes_complete_after_an_empty_optional_scope_scan(
    lane, monkeypatch, elapsed_seconds, expected_sessions
):
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return LANE_NOW + timedelta(seconds=elapsed_seconds)

    # Roster activity expires after five minutes; coverage does not. Both the
    # transcript timestamps and the RPC clock must be fixed to test that split.
    monkeypatch.setattr("recall.services.rpc_server.datetime", FixedClock)

    async def read():
        before = serialize_rpc_value(await lane.server._handle_live({}, None))
        assert before["coverage"]["complete"] is False
        await lane.server._handle_daemon_run({"once": True, "embed": False}, None)
        after = serialize_rpc_value(await lane.server._handle_live({}, None))
        assert after["coverage"]["complete"] is True
        assert after["coverage"]["catalog_scan_complete"] is True
        assert after["watching"] is False
        assert len(after["sessions"]) == expected_sessions

    asyncio.run(read())


def test_absent_configured_roots_complete_empty_and_are_discovered_when_created(lane, tmp_path):
    from dataclasses import replace

    from recall.core.config import SourceConfig

    root = tmp_path / "configured-claude"
    lane.server._config = replace(
        lane.server._config, sources={"claude_code": SourceConfig(roots=(root,))}
    )

    async def read():
        await lane.server._handle_daemon_run({"once": True, "embed": False}, None)
        return (await lane.server._handle_daemon_status({}, None))["reconciliation"]

    before = asyncio.run(read())
    assert before["raw_indexing_ready"] is True
    assert before["keyword_search_ready"] is True
    coverage = next(row for row in before["coverage"] if row["root_path"] == str(root))
    assert coverage["configuration"] == "configured"
    assert coverage["available"] is False
    assert coverage["scan_complete"] is True
    assert coverage["scan_current_process"] is True
    assert coverage["failure_count"] == 0

    root.mkdir()
    shutil.copyfile(lane.watched, root / "later.jsonl")
    after = asyncio.run(read())
    coverage = next(row for row in after["coverage"] if row["root_path"] == str(root))
    assert coverage["available"] is True
    assert coverage["current"] == 1
    assert after["raw_indexing_ready"] is True


@pytest.mark.parametrize("kind", ["file", "denied"])
def test_configured_root_errors_remain_visible_and_block_readiness(
    lane, tmp_path, monkeypatch, kind
):
    from dataclasses import replace

    from recall.core.config import SourceConfig

    root = tmp_path / "configured-codex"
    if kind == "file":
        root.write_text("not a directory")
    if kind == "denied":
        root.mkdir()
        original_stat = Path.stat

        def deny_selected(path, *args, **kwargs):
            if path == root:
                raise PermissionError("injected configured-root denial")
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", deny_selected)
    lane.server._config = replace(
        lane.server._config, sources={"codex": SourceConfig(roots=(root,))}
    )

    async def read():
        await lane.server._handle_daemon_run({"once": True, "embed": False}, None)
        return await lane.server._handle_daemon_status({}, None)

    result = asyncio.run(read())["reconciliation"]
    assert result["raw_indexing_ready"] is False
    coverage = next(row for row in result["coverage"] if row["root_path"] == str(root))
    assert coverage["failure_count"] == 1
    assert coverage["errors"]


def test_link_ownership_reconfiguration_preserves_identity_and_readable_history(lane, tmp_path):
    from dataclasses import replace

    from recall.core.config import SourceConfig

    roots = [tmp_path / name for name in ("a-root", "b-root")]
    target = tmp_path / "external" / "session.jsonl"
    target.parent.mkdir()
    target.write_text(lane.watched.read_text().replace("live-mid-tool", "delegated-history"))
    for root in roots:
        root.mkdir()
        (root / "linked").symlink_to(target.parent, target_is_directory=True)

    def configure(selected):
        lane.server._config = replace(
            lane.server._config,
            sources={
                source: SourceConfig(roots=tuple(selected) if source == "claude_code" else ())
                for source in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
            },
        )

    async def scan():
        await lane.server._handle_daemon_run({"once": True, "embed": False}, None)
        return await lane.server._handle_daemon_status({}, None)

    configure(reversed(roots))
    asyncio.run(scan())
    conn = lane.server._get_conn()
    before = conn.execute(
        """SELECT session_id, desired_generation, committed_generation, root_path
           FROM source_files WHERE source_path = ?""",
        [str(target)],
    ).fetchone()
    assert before is not None and before[0] is not None
    assert before[3] == str(roots[0])
    configure([roots[1]])
    asyncio.run(scan())
    after = conn.execute(
        """SELECT session_id, desired_generation, committed_generation, root_path
           FROM source_files WHERE source_path = ?""",
        [str(target)],
    ).fetchone()
    assert after == (*before[:3], str(roots[1]))
    (roots[1] / "linked").unlink()
    status = asyncio.run(scan())["reconciliation"]
    assert target.exists()
    assert conn.execute(
        "SELECT missing FROM source_files WHERE source_path = ?", [str(target)]
    ).fetchone() == (True,)
    assert status["raw_indexing_ready"] is True
    shown = asyncio.run(lane.server._handle_show({"session_id": before[0]}, None))
    assert shown is not None
    # Retarget to an identical copy while the original still physically exists.
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    successor = replacement / "session.jsonl"
    successor.write_bytes(target.read_bytes())
    (roots[1] / "linked").symlink_to(replacement, target_is_directory=True)
    asyncio.run(scan())
    identities = conn.execute(
        "SELECT session_id FROM source_files WHERE source_path IN (?, ?) ORDER BY source_path",
        [str(target), str(successor)],
    ).fetchall()
    assert len(identities) == 2 and len({row[0] for row in identities}) == 2
    assert before[0] in {row[0] for row in identities}
