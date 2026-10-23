"""Independent parser identities through the shared catalog and fresh-read queue."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from lane_harness import lane_config
from recall.core.config import SourceConfig
from recall.core.source_path import SourcePath
from recall.core.types import Source
from recall.services.rpc_server import RpcServer


@pytest.mark.parametrize("order", [("claude_code", "codex"), ("codex", "claude_code")])
def test_shared_path_keeps_independent_history_and_fresh_reads(tmp_path, order):
    target = tmp_path / "external"
    target.mkdir()
    path = target / "rollout-shared.jsonl"
    records = [
        {
            "type": "session_meta",
            "payload": {"id": "codex-shared", "timestamp": "2024-01-16T12:00:00Z", "cwd": "/repo"},
        },
        {
            "type": "event_msg",
            "timestamp": "2024-01-16T12:01:00Z",
            "payload": {"type": "user_message", "message": "codex-only-message"},
        },
        {
            "type": "user",
            "sessionId": "claude-shared",
            "uuid": "claude-record",
            "timestamp": "2024-01-16T12:05:00Z",
            "message": {"role": "user", "content": "claude-only-message"},
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    sources = {source.value: SourceConfig(roots=()) for source in Source}
    for source in order:
        root = tmp_path / source
        root.mkdir()
        (root / "delegated").symlink_to(target, target_is_directory=True)
        sources[source] = SourceConfig(roots=(root,))
    config = replace(lane_config(tmp_path), sources=sources)
    server = RpcServer(config)
    server._executor = ThreadPoolExecutor(max_workers=2)

    async def scenario():
        try:
            for source in order:
                await server.index_session_now(path, source=source)
            sessions = await server._run_readonly(
                lambda conn: conn.execute("SELECT source, id FROM sessions").fetchall()
            )
            ids = dict(sessions)
            assert set(ids) == {"claude_code", "codex"}
            assert len(set(ids.values())) == 2
            initial = {
                source: await server._handle_show({"session_id": ids[source]}, None)
                for source in order
            }
            assert initial["claude_code"]["message_count"] == 1
            assert initial["codex"]["message_count"] == 1
            with path.open("a") as handle:
                for record in (
                    {
                        "type": "user",
                        "sessionId": "claude-shared",
                        "uuid": "claude-record",
                        "timestamp": "2024-01-16T12:06:00Z",
                        "message": {"role": "user", "content": "claude-append"},
                    },
                    {
                        "type": "event_msg",
                        "timestamp": "2024-01-16T12:07:00Z",
                        "payload": {"type": "user_message", "message": "codex-append"},
                    },
                ):
                    handle.write(json.dumps(record) + "\n")
            results = await asyncio.gather(
                *(
                    server._handle_show({"session_id": ids[source], "fresh": True}, None)
                    for source in order
                )
            )
            by_source = dict(zip(order, results, strict=True))
            assert by_source["claude_code"]["message_count"] == 2
            assert by_source["codex"]["message_count"] == 2
            for source, result in by_source.items():
                assert result["id"] == ids[source]
                assert result["freshness"].current
            assert not server._fresh_jobs
            assert not server._raw_requests
            # Source-specific refresh must leave the other source's generation alone.
            with path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "type": "event_msg",
                            "timestamp": "2024-01-16T12:08:00Z",
                            "payload": {"type": "user_message", "message": "only-codex-refresh"},
                        }
                    )
                    + "\n"
                )
            await server._refresh_now([SourcePath("codex", str(path))], timeout=5)
            claude = await server._handle_show({"session_id": ids["claude_code"]}, None)
            codex = await server._handle_show({"session_id": ids["codex"]}, None)
            assert claude["message_count"] == 2
            assert not claude["freshness"].current
            assert codex["message_count"] == 3
            assert codex["freshness"].current
        finally:
            await server.stop()

    asyncio.run(scenario())
