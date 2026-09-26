"""Regression: agent text truncated mid-surrogate-pair must stay storable.

Agents truncate tool output by character count. When the cut lands inside an
emoji's UTF-16 surrogate pair it strands the high half in the JSONL.
`json.loads` accepts the lone surrogate, but the resulting `str` cannot be
encoded to UTF-8, so DuckDB rejects the bound parameter and the *entire
session* fails to index — observed on a Kimi Code `wire.jsonl` whose output
read `web:build:  \ud83d[...truncated]`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pytest
from recall.core.models import Message, ToolCall
from recall.core.types import Role
from recall.db.queries import insert_messages, insert_tool_calls

# The high half of U+1F680 ROCKET, left alone by a truncated pair.
LONE_HIGH_SURROGATE = "\ud83d"
REPLACEMENT = "�"


def _message(content: str | None = None, thinking: str | None = None) -> Message:
    return Message(
        id="m1",
        session_id="s1",
        idx=0,
        role=Role.ASSISTANT,
        content=content,
        thinking=thinking,
        timestamp=datetime(2026, 7, 24, 23, 12, 15, tzinfo=UTC),
    )


def test_lone_surrogate_in_content_is_replaced() -> None:
    message = _message(content=f"web:build:  {LONE_HIGH_SURROGATE}[...truncated]")

    assert message.content is not None
    assert LONE_HIGH_SURROGATE not in message.content
    assert REPLACEMENT in message.content
    # The real constraint: it must survive the trip to UTF-8.
    message.content.encode("utf-8")


def test_lone_surrogate_in_thinking_and_bash_command_is_replaced() -> None:
    message = _message(thinking=f"planning {LONE_HIGH_SURROGATE} next")
    tool_call = ToolCall(
        id="t1",
        session_id="s1",
        message_id="m1",
        idx=0,
        tool_name="Bash",
        bash_command=f"echo {LONE_HIGH_SURROGATE}",
    )

    assert message.thinking is not None
    assert tool_call.bash_command is not None
    message.thinking.encode("utf-8")
    tool_call.bash_command.encode("utf-8")
    assert REPLACEMENT in message.thinking
    assert REPLACEMENT in tool_call.bash_command


def test_well_formed_surrogate_pair_survives_intact() -> None:
    """Only *unpaired* surrogates are scrubbed; real emoji must round-trip."""

    message = _message(content="ship it 🚀")

    assert message.content == "ship it 🚀"


@pytest.mark.parametrize("field", ["content", "thinking"])
def test_message_with_lone_surrogate_inserts_into_duckdb(field: str) -> None:
    """The end-to-end failure: DuckDB used to reject the bound parameter."""

    text = f"web:build:  {LONE_HIGH_SURROGATE}[...truncated]"
    message = _message(**{field: text})

    conn = duckdb.connect()
    try:
        conn.execute("CREATE TABLE messages (id TEXT, session_id TEXT, idx INTEGER, agent_id TEXT)")
        conn.execute(
            "CREATE TABLE message_embeddings ("
            "message_id TEXT PRIMARY KEY, "
            "content_embedding FLOAT[384], thinking_embedding FLOAT[384])"
        )
        conn.execute(
            """
            CREATE TABLE message_state (
                message_id TEXT PRIMARY KEY, role TEXT, content TEXT, thinking TEXT,
                timestamp TIMESTAMP, has_thinking BOOLEAN, context_text TEXT,
                context_mode TEXT, fts_content TEXT, fts_thinking TEXT
            )
            """
        )
        insert_messages(conn, [message])

        stored = conn.execute(f"SELECT {field} FROM message_state").fetchone()
        assert stored is not None
        assert stored[0] is not None
        assert REPLACEMENT in stored[0]
    finally:
        conn.close()


def test_tool_call_with_lone_surrogate_inserts_into_duckdb() -> None:
    tool_call = ToolCall(
        id="t1",
        session_id="s1",
        message_id="m1",
        idx=0,
        tool_name="Bash",
        tool_input={"command": f"echo {LONE_HIGH_SURROGATE}"},
        bash_command=f"echo {LONE_HIGH_SURROGATE}",
    )

    conn = duckdb.connect()
    try:
        conn.execute(
            """
            CREATE TABLE tool_calls (
                id TEXT, session_id TEXT, message_id TEXT, idx INTEGER,
                tool_name TEXT, tool_input TEXT, bash_command TEXT, bash_base TEXT,
                bash_sub TEXT, is_compound BOOLEAN, agent_id TEXT, subagent_type TEXT,
                subagent_description TEXT, subagent_model TEXT, skill_name TEXT
            )
            """
        )
        insert_tool_calls(conn, [tool_call])

        stored = conn.execute("SELECT bash_command FROM tool_calls").fetchone()
        assert stored is not None
        assert REPLACEMENT in stored[0]
    finally:
        conn.close()
