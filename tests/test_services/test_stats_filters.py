"""Tests for filtered analytics, RPC handlers, time overflow, and owned poll snapshots."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.poll_reads import poll_read_request
from recall.core.rpc_types import INVALID_PARAMS, RpcError
from recall.core.stats_filters import normalize_stats_filters
from recall.core.time import parse_since
from recall.core.types import Source
from recall.db import connect
from recall.read_budget import ReadBudget
from recall.services import analytics
from recall.services.poll_reads import read_poll_snapshot
from recall.services.read_handlers import read_overview
from recall.services.rpc_server import ClientConnection, RpcServer


def _config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=data_dir / "config.toml",
        fts=FtsConfig(backend="sqlite_sidecar", fields=("content", "thinking", "bash")),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )


def _seed_analytics_db(config: AppConfig) -> None:
    with connect(config) as conn:
        # Session 1: antigravity_cli, recent (2026-05-10 12:00:00 UTC)
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) "
            "VALUES ('s-agy-new', 'antigravity_cli', '/tmp/agy-new.jsonl')"
        )
        conn.execute(
            "INSERT INTO session_state "
            "(session_id, started_at, ended_at, indexed_at, git_repo, model, host, "
            " input_tokens, cached_input_tokens, output_tokens, message_count, tool_count, "
            " is_complete, file_mtime, file_size) "
            "VALUES ('s-agy-new', TIMESTAMP '2026-05-10 11:00:00', "
            " TIMESTAMP '2026-05-10 12:00:00', TIMESTAMP '2026-05-10 12:05:00', "
            " 'org/repo-b', 'gemini-2.5', 'aem5', 200, 50, 40, 2, 3, TRUE, 1.0, 100)"
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) "
            "VALUES ('m-agy-1', 's-agy-new', 0), ('m-agy-2', 's-agy-new', 1)"
        )
        conn.execute(
            "INSERT INTO tool_calls "
            "(id, session_id, idx, tool_name, bash_command, bash_base, bash_sub, is_compound) "
            "VALUES "
            "('tc-agy-1', 's-agy-new', 0, 'read_file', NULL, NULL, NULL, FALSE), "
            "('tc-agy-2', 's-agy-new', 1, 'read_file', NULL, NULL, NULL, FALSE), "
            "('tc-agy-3', 's-agy-new', 2, 'run_command', 'git status', 'git', 'status', FALSE)"
        )

        # Session 2: antigravity_cli, old (2026-04-01 12:00:00 UTC)
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) "
            "VALUES ('s-agy-old', 'antigravity_cli', '/tmp/agy-old.jsonl')"
        )
        conn.execute(
            "INSERT INTO session_state "
            "(session_id, started_at, ended_at, indexed_at, git_repo, model, host, "
            " input_tokens, cached_input_tokens, output_tokens, message_count, tool_count, "
            " is_complete, file_mtime, file_size) "
            "VALUES ('s-agy-old', TIMESTAMP '2026-04-01 11:00:00', "
            " TIMESTAMP '2026-04-01 12:00:00', TIMESTAMP '2026-04-01 12:05:00', "
            " 'org/repo-a', 'gemini-2.5', 'aem5', 200, NULL, 40, 1, 1, TRUE, 1.0, 100)"
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) VALUES ('m-agy-old', 's-agy-old', 0)"
        )
        conn.execute(
            "INSERT INTO tool_calls "
            "(id, session_id, idx, tool_name, bash_command, bash_base, bash_sub, is_compound) "
            "VALUES ('tc-agy-old', 's-agy-old', 0, 'edit_file', NULL, NULL, NULL, FALSE)"
        )

        # Session 3: codex, recent (2026-05-10 12:30:00 UTC), same token sum as repo-b for tie check
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) "
            "VALUES ('s-codex', 'codex', '/tmp/codex.jsonl')"
        )
        conn.execute(
            "INSERT INTO session_state "
            "(session_id, started_at, ended_at, indexed_at, git_repo, model, host, "
            " input_tokens, cached_input_tokens, output_tokens, message_count, tool_count, "
            " is_complete, file_mtime, file_size) "
            "VALUES ('s-codex', TIMESTAMP '2026-05-10 12:00:00', "
            " TIMESTAMP '2026-05-10 12:30:00', TIMESTAMP '2026-05-10 12:35:00', "
            " 'org/repo-a', 'gpt-5', 'aem5', 200, 100, 40, 1, 2, TRUE, 1.0, 100)"
        )
        conn.execute(
            "INSERT INTO messages (id, session_id, idx) VALUES ('m-codex-1', 's-codex', 0)"
        )
        conn.execute(
            "INSERT INTO tool_calls "
            "(id, session_id, idx, tool_name, bash_command, bash_base, bash_sub, is_compound) "
            "VALUES "
            "('tc-c-1', 's-codex', 0, 'exec_command', 'rm -rf tmp', 'rm', NULL, FALSE), "
            "('tc-c-2', 's-codex', 1, 'exec_command', 'git diff && ls', 'git', 'diff', TRUE)"
        )


def test_parse_since_translates_duration_and_iso_overflow_to_value_error() -> None:
    with pytest.raises(ValueError, match="out of range"):
        parse_since("9999999999d")
    with pytest.raises(ValueError, match="out of range"):
        parse_since("0001-01-01T00:00:00+01:00")


def test_normalize_stats_filters_canonicalizes_and_rejects_blank() -> None:
    fixed_now = datetime(2026, 5, 11, 0, 0, 0, tzinfo=UTC)
    parsed = normalize_stats_filters(source="agy", since="24h", now=fixed_now)
    assert parsed.source is Source.ANTIGRAVITY_CLI
    assert parsed.since == datetime(2026, 5, 10, 0, 0, 0, tzinfo=UTC)
    assert parsed.since_iso == "2026-05-10T00:00:00Z"
    assert parsed.active

    for bad_source in ("", "   ", "unknown_tool"):
        with pytest.raises(ValueError):
            normalize_stats_filters(source=bad_source)
    for bad_since in ("", "   ", "not-a-date", "9999999999d", "0001-01-01T00:00:00+01:00"):
        with pytest.raises(ValueError):
            normalize_stats_filters(since=bad_since)


def test_filtered_analytics_under_non_utc_timezone_and_deterministic_ties(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All filtered analytics bind naive UTC cutoffs (no TZ shift) and deterministic tie-breaks."""
    monkeypatch.setenv("TZ", "America/Chicago")
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        config = _config(tmp_path)
        _seed_analytics_db(config)
        cutoff = parse_since("2026-05-10T11:30:00Z")

        with connect(config) as conn:
            # Filtered overview: antigravity_cli since 11:30Z -> only s-agy-new
            ov = read_overview(config, conn, source=Source.ANTIGRAVITY_CLI, since=cutoff)
            assert ov["sessions"] == 1
            assert ov["messages"] == 2
            assert ov["tool_calls"] == 3
            assert ov["bash_calls"] == 1
            assert ov["last_index_indexed"] is None
            assert ov["last_context_mode"] is None

            # Unfiltered overview preserves runtime_context_stats
            unfiltered_ov = read_overview(config, conn)
            assert unfiltered_ov["sessions"] == 3
            assert unfiltered_ov["last_context_mode"] == "off"

            # Tool usage filtered by source + since
            tools = analytics.tool_usage(
                source=Source.ANTIGRAVITY_CLI, since=cutoff, config=config, conn=conn
            )
            assert [(t.tool_name, t.count) for t in tools] == [
                ("read_file", 2),
                ("run_command", 1),
            ]

            # Bash breakdown & suggestions filtered by source
            bash_agy = analytics.bash_breakdown(
                source=Source.ANTIGRAVITY_CLI, config=config, conn=conn
            )
            assert [(b.bash_base, b.bash_sub, b.count) for b in bash_agy] == [("git", "status", 1)]
            sug, skip = analytics.bash_suggestions(source=Source.CODEX, config=config, conn=conn)
            assert [s.pattern for s in sug] == ["git diff"]
            assert sug[0].confidence == "review"
            assert [s.pattern for s in skip] == ["rm *"]

            # Token usage tie-break: equal sums (240 each) sort org/repo-a before org/repo-b
            toks = analytics.token_usage(since=cutoff, config=config, conn=conn)
            assert [t.repo for t in toks] == ["org/repo-a", "org/repo-b"]

            # Usage by source filtered by source + since; tie-break over (source, model, host)
            usage_agy = analytics.usage_by_source(
                source=Source.ANTIGRAVITY_CLI, since=cutoff, config=config, conn=conn
            )
            assert len(usage_agy) == 1
            assert usage_agy[0].source == "antigravity_cli"
            assert usage_agy[0].input_tokens == 200
            assert usage_agy[0].fresh_input_tokens == 150

            usage_all_recent = analytics.usage_by_source(since=cutoff, config=config, conn=conn)
            # Equal total tokens (240 each): antigravity_cli sorts before codex
            assert [u.source for u in usage_all_recent] == ["antigravity_cli", "codex"]
    finally:
        monkeypatch.delenv("TZ", raising=False)
        if hasattr(time, "tzset"):
            time.tzset()


def test_rpc_server_filtered_stats_and_legacy_usage_validation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _seed_analytics_db(config)
    server = RpcServer(config)

    class _FakeWriter:
        def __init__(self) -> None:
            self.chunks: list[bytes] = []

        def write(self, data: bytes) -> None:
            self.chunks.append(data)

        async def drain(self) -> None:
            return None

    async def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        writer = _FakeWriter()
        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        await server._process_request(
            req.encode("utf-8"),
            cast(asyncio.StreamWriter, writer),
            cast(ClientConnection, None),
        )
        return json.loads(b"".join(writer.chunks))

    async def run() -> None:
        try:
            r1 = await call(
                "recall.stats_filtered", {"source": "agy", "since": "2026-05-01T00:00:00Z"}
            )
            assert r1["result"]["sessions"] == 1
            assert r1["result"]["last_index_indexed"] is None

            r2 = await call("recall.stats_tools_filtered", {"source": "agy"})
            assert {row["tool_name"]: row["count"] for row in r2["result"]} == {
                "read_file": 2,
                "run_command": 1,
                "edit_file": 1,
            }

            r3 = await call("recall.stats_bash_filtered", {"source": "codex", "suggest": True})
            assert [s["pattern"] for s in r3["result"]["skipped"]] == ["rm *"]

            r4 = await call("recall.stats_tokens_filtered", {"source": "codex"})
            assert r4["result"] == [
                {"repo": "org/repo-a", "input_tokens": 200, "output_tokens": 40}
            ]

            r5 = await call("recall.stats_usage_filtered", {"source": "agy"})
            assert all(row["source"] == "antigravity_cli" for row in r5["result"])

            # Legacy recall.stats_usage must reject blank and overflowing since with INVALID_PARAMS
            for bad in ("", "   ", "9999999999d", "0001-01-01T00:00:00+01:00"):
                err = await call("recall.stats_usage", {"since": bad})
                assert err["error"]["code"] == INVALID_PARAMS

            # Filtered methods also reject blank/invalid source and since with INVALID_PARAMS
            bad_src = await call("recall.stats_filtered", {"source": "  "})
            assert bad_src["error"]["code"] == INVALID_PARAMS

            # Versioned filtered methods reject unknown keys (typos never broaden queries)
            for method in (
                "recall.stats_filtered",
                "recall.stats_tools_filtered",
                "recall.stats_bash_filtered",
                "recall.stats_tokens_filtered",
                "recall.stats_usage_filtered",
            ):
                typo = await call(method, {"soruce": "agy"})
                assert typo["error"]["code"] == INVALID_PARAMS
        finally:
            if server._database is not None:
                server._database.close()

    asyncio.run(run())


def test_poll_snapshot_supports_all_filtered_stats_and_rejects_bad_usage_since(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _seed_analytics_db(config)
    before_stat = config.db_path.stat()

    cases = [
        ("recall.stats_filtered", {"source": "agy", "since": "2026-05-01T00:00:00Z"}),
        ("recall.stats_tools_filtered", {"source": "agy"}),
        ("recall.stats_bash_filtered", {"source": "codex", "suggest": False}),
        ("recall.stats_bash_filtered", {"source": "codex", "suggest": True}),
        ("recall.stats_tokens_filtered", {"source": "codex"}),
        ("recall.stats_usage_filtered", {"source": "agy"}),
    ]
    for method, params in cases:
        req = poll_read_request(method, params)
        assert req is not None, f"poll_read_request rejected {method}"
        snap = read_poll_snapshot(config, req, ReadBudget.start(30.0))
        assert snap.data is not None

    # Legacy UsageRead and UsageFilteredRead refuse blank/overflowing since with INVALID_PARAMS
    for method, params in (
        ("recall.stats_usage", {"since": ""}),
        ("recall.stats_usage", {"since": "9999999999d"}),
        ("recall.stats_usage", {"since": "0001-01-01T00:00:00+01:00"}),
        ("recall.stats_usage_filtered", {"source": "  "}),
    ):
        req = poll_read_request(method, params)
        assert req is not None
        with pytest.raises(RpcError) as exc_info:
            read_poll_snapshot(config, req, ReadBudget.start(30.0))
        assert exc_info.value.code == INVALID_PARAMS

    # Unknown keys on filtered poll requests are rejected pre-dispatch
    for method in (
        "recall.stats_filtered",
        "recall.stats_tools_filtered",
        "recall.stats_bash_filtered",
        "recall.stats_tokens_filtered",
        "recall.stats_usage_filtered",
    ):
        assert poll_read_request(method, {"soruce": "agy"}) is None

    after_stat = config.db_path.stat()
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_stat.st_size == before_stat.st_size
    assert not (config.data_dir / "recall.duckdb.wal").exists()


def test_classify_bash_suggestions_coalesces_wildcard_pattern_collisions() -> None:
    from recall.services.analytics import BashStat, classify_bash_suggestions

    suggestions, skipped = classify_bash_suggestions(
        [
            BashStat(bash_base="git", bash_sub=None, count=30, is_compound=False),
            BashStat(bash_base="git", bash_sub="*", count=25, is_compound=False),
        ]
    )
    assert skipped == []
    assert len(suggestions) == 1
    assert suggestions[0].pattern == "git *"
    assert suggestions[0].count == 55
    assert suggestions[0].confidence == "high"
