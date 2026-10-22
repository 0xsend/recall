"""Live pages rank narrow identities before hydrating wide payloads."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services.live import _fetch_open_tool_call_groups, _fetch_turn_tails


@pytest.mark.parametrize("kind", ["messages", "tools"])
def test_live_page_bounds_payload_before_ranking(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "live.duckdb"
    with duckdb.connect(str(path)) as c:
        ensure_schema(c)
        c.execute("INSERT INTO sessions VALUES ('s', 'codex', '/live-source', NULL)")
        c.execute(
            "INSERT INTO messages (id, session_id, idx) "
            "SELECT i::TEXT, 's', i FROM range(8192) r(i)"
        )
        c.execute(
            "INSERT INTO message_state (message_id, role, content) "
            "SELECT i::TEXT, 'assistant', repeat(md5(i::TEXT), 512) FROM range(8192) r(i)"
        )
        c.execute("INSERT INTO messages (id, session_id, idx) VALUES ('missing-state', 's', 8192)")
        if kind == "tools":
            c.execute(
                "INSERT INTO tool_calls "
                "(id, session_id, message_id, idx, tool_name, bash_command) "
                "SELECT i::TEXT, 's', i::TEXT, i, 'Bash', repeat(md5(i::TEXT), 512) "
                "FROM range(8192) r(i)"
            )
            c.execute("INSERT INTO tool_use_ids VALUES ('8191', 's', 'call-8191')")
            c.execute(
                "INSERT INTO tool_use_ids "
                "SELECT i::TEXT, 's', 'call-' || i::TEXT FROM range(8191) r(i)"
            )
            c.execute("INSERT INTO tool_results (tool_call_id) VALUES ('8191')")
        c.execute("CHECKPOINT")
    with duckdb.connect(str(path), config={"memory_limit": "32MiB", "threads": 2}) as c:
        if kind == "messages":
            rows = _fetch_turn_tails(c, ["s"])["s"]
            assert [r.idx for r in rows] == list(range(8184, 8192))
            content = rows[-1].content
            assert content is not None
            assert len(content) == 16384
        else:
            rows = _fetch_open_tool_call_groups(c, ["s"], limit=2)["s"]
            assert [r.tool_call.id for r in rows] == ["8190", "8189"]
            command = rows[0].tool_call.bash_command
            assert command is not None
            assert len(command) == 16384
