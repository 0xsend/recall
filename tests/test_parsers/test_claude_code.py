from __future__ import annotations

from pathlib import Path

import pytest
from recall.core.models import TailFacts
from recall.parsers.claude_code import ClaudeCodeParser

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "claude_code"


def test_claude_code_harness_metadata_parses_without_diagnostics() -> None:
    """Harness metadata records do not diagnose, invent messages, or drop session fields."""
    fixture = FIXTURES / "session_harness_metadata.jsonl"
    result = ClaudeCodeParser().parse(fixture)
    session = result.session

    assert result.diagnostics == ()
    assert session.is_complete is True
    assert session.message_count == 2
    assert [message.role for message in session.messages] == ["user", "assistant"]
    assert [message.content for message in session.messages] == ["hello", "hi"]
    assert session.cwd == "/work/demo"
    assert session.source_session_id == "meta-session-001"
    assert session.git_branch == "master"
    assert session.git_repo == "/work/demo"


def test_claude_code_dev_mods_record_is_acknowledged() -> None:
    """dev-mods is harness metadata: no diagnostic, no message, sessionId still read.

    The record points at the harness's internal dev-mods state folder and
    carries no conversation content (REQ-PARSE-031).
    """
    fixture = FIXTURES / "session_dev_mods.jsonl"
    result = ClaudeCodeParser().parse(fixture)
    session = result.session

    assert result.diagnostics == ()
    assert session.is_complete is True
    assert session.message_count == 2
    assert [message.role for message in session.messages] == ["user", "assistant"]
    assert session.source_session_id == "dev-mods-session-001"


def test_claude_code_parser_parses_messages() -> None:
    """Legacy format: tokens at root level, no nested message metadata."""
    fixture = FIXTURES / "session1.jsonl"
    parser = ClaudeCodeParser()
    session = parser.parse(fixture).session

    assert session.message_count == 4
    assert session.input_tokens == 5
    assert session.output_tokens == 7

    # Fields absent from legacy format resolve to None or fallback
    assert session.model is None
    assert session.git_branch is None
    assert session.cwd is None
    # source_session_id falls back to file stem when no sessionId field exists
    assert session.source_session_id == "session1"

    tool_calls = session.messages[1].tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call.tool_name == "bash"
    assert tool_call.bash_command == "git status"
    assert tool_call.bash_base == "git"
    assert tool_call.bash_sub == "status"


# -- Current format (2025+) tests --


@pytest.fixture()
def v2_session():
    fixture = FIXTURES / "session_v2.jsonl"
    parser = ClaudeCodeParser()
    return parser.parse(fixture).session


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        # Model comes from entry.message.model, not the root level.
        ("model", "claude-opus-4-6"),
        # git_branch comes from entry.gitBranch (camelCase root).
        ("git_branch", "feat/new-parser"),
        # Nested message.usage, including cache tokens: input 100 + 80 = 180,
        # cache_creation 500 + 0, cache_read 200 + 300 -> 180 + 500 + 500.
        ("input_tokens", 1180),
        # output_tokens: 50 + 30.
        ("output_tokens", 80),
        ("source_session_id", "v2-session-abc"),
        ("cwd", "/home/dev/project"),
        # git_repo is derived from cwd when no explicit git_root/repo field exists.
        ("git_repo", "/home/dev/project"),
        # All four entries produce messages (2 user + 2 assistant).
        ("message_count", 4),
    ],
)
def test_claude_code_v2_format_session_fields(v2_session, field: str, expected: object) -> None:
    assert getattr(v2_session, field) == expected


def test_claude_code_v2_format_tool_call(v2_session) -> None:
    """Tool use blocks in assistant messages are parsed."""
    # Second message (idx 1) is the assistant with a Bash tool_use
    tool_calls = v2_session.messages[1].tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "Bash"
    assert tool_calls[0].bash_command == "npm run build"
    assert tool_calls[0].bash_base == "npm"


def test_claude_code_v2_format_thinking(v2_session) -> None:
    """Thinking blocks with empty text are handled without error."""
    # The first assistant message has a thinking block with empty string
    msg = v2_session.messages[1]
    # Empty thinking string should not set has_thinking
    assert not msg.has_thinking


def test_tool_use_empty_input_preserved() -> None:
    """Regression: tool_use with input={} must preserve the empty dict, not drop to None."""
    from recall.parsers.common import extract_content_blocks

    content = [{"type": "tool_use", "name": "TodoWrite", "input": {}}]
    tool_calls = extract_content_blocks(content).tool_calls
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "TodoWrite"
    # Empty dict must be preserved, not coerced to None
    assert tool_calls[0].tool_input == {}


# ---------------------------------------------------------------------------
# Milestone 2: build_tool_call subagent/skill extraction
# ---------------------------------------------------------------------------


def test_build_tool_call_no_subagent_fields_for_bash() -> None:
    """build_tool_call does not set subagent fields for non-Agent tools."""
    from recall.parsers.common import build_tool_call

    tc = build_tool_call("Bash", {"command": "git status"})
    assert tc.subagent_type is None
    assert tc.subagent_description is None
    assert tc.subagent_model is None
    assert tc.skill_name is None


# ---------------------------------------------------------------------------
# Milestone 3: agent_progress entries in ClaudeCodeParser
# ---------------------------------------------------------------------------

SUBAGENT_FIXTURE = FIXTURES / "session_subagent.jsonl"


@pytest.fixture()
def subagent_session():
    parser = ClaudeCodeParser()
    return parser.parse(SUBAGENT_FIXTURE).session


def test_subagent_tool_calls_have_agent_id(subagent_session) -> None:
    """Tool calls made by subagent carry the agent_id."""
    all_tool_calls = [tc for m in subagent_session.messages for tc in m.tool_calls]
    subagent_tcs = [tc for tc in all_tool_calls if tc.agent_id is not None]
    assert len(subagent_tcs) == 1
    assert subagent_tcs[0].tool_name == "Bash"
    assert subagent_tcs[0].bash_command == "grep -r 'auth' src/"
    assert subagent_tcs[0].agent_id == "agent_abc123"


def test_agent_dispatch_tool_call_has_subagent_fields(subagent_session) -> None:
    """The Agent tool_use in the main conversation has subagent metadata."""
    all_tool_calls = [tc for m in subagent_session.messages for tc in m.tool_calls]
    agent_dispatches = [tc for tc in all_tool_calls if tc.tool_name == "Agent"]
    assert len(agent_dispatches) == 1
    dispatch = agent_dispatches[0]
    assert dispatch.subagent_type == "Explore"
    assert dispatch.subagent_description == "Find auth middleware"
    assert dispatch.subagent_model == "sonnet"
    assert dispatch.agent_id is None


def test_subagent_tokens_accumulated(subagent_session) -> None:
    """Token usage from subagent progress entries is accumulated into session totals."""
    # Main: input 50+40=90, output 10+15=25
    # Subagent: input 30+25=55, output 20+15=35
    # Total: input 145, output 60
    assert subagent_session.input_tokens == 145
    assert subagent_session.output_tokens == 60


def test_subagent_message_ordering(subagent_session) -> None:
    """Messages are ordered chronologically matching JSONL file order."""
    messages = subagent_session.messages
    assert [m.idx for m in messages] == list(range(8))
    expected_agent_ids = [
        None,
        None,
        "agent_abc123",
        "agent_abc123",
        "agent_abc123",
        "agent_abc123",
        None,
        None,
    ]
    assert [m.agent_id for m in messages] == expected_agent_ids


def test_failed_identity_rewrite_preserves_subagent_data(tmp_path) -> None:
    """A rewrite that fails part-way must leave v13 subagent columns intact.

    Regression: the rewrite used to undo itself by re-inserting rows it had
    saved, using hard-coded column counts and positional indices -- which read
    agent_id as an embedding float and dropped subagent metadata. It is now one
    transaction, so ROLLBACK restores every column without enumerating any.
    """
    import duckdb
    import recall.services.indexer as indexer_module
    from recall.db.schema import ensure_schema
    from recall.services.indexer import _write_session

    parser = ClaudeCodeParser()
    session = parser.parse(SUBAGENT_FIXTURE).session

    conn = duckdb.connect(str(tmp_path / "test.duckdb"))
    ensure_schema(conn)
    _write_session(conn, session, tail_facts=TailFacts())

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("insert failed mid-rewrite")

    # Changing source_session_id routes the next write through the rewrite path.
    renamed = session.model_copy(update={"source_session_id": "rotated-source-id"})
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(indexer_module, "insert_tool_calls", _boom)
    try:
        with pytest.raises(RuntimeError):
            _write_session(conn, renamed, tail_facts=TailFacts())
    finally:
        monkeypatch.undo()

    agent_msgs = conn.execute("SELECT agent_id FROM messages WHERE agent_id IS NOT NULL").fetchall()
    assert len(agent_msgs) == 4
    for row in agent_msgs:
        assert row[0] == "agent_abc123"

    agent_dispatches = conn.execute(
        "SELECT subagent_type, subagent_description, subagent_model "
        "FROM tool_calls WHERE tool_name = 'Agent'"
    ).fetchall()
    assert len(agent_dispatches) == 1
    assert agent_dispatches[0] == ("Explore", "Find auth middleware", "sonnet")

    subagent_tcs = conn.execute(
        "SELECT tool_name, agent_id FROM tool_calls WHERE agent_id IS NOT NULL"
    ).fetchall()
    assert len(subagent_tcs) == 1
    assert subagent_tcs[0] == ("Bash", "agent_abc123")

    # The failed rewrite must not have taken the new identity.
    assert conn.execute(
        "SELECT source_session_id FROM sessions WHERE id = ?", [session.id]
    ).fetchone() != ("rotated-source-id",)

    conn.close()


def test_load_session_returns_subagent_data(tmp_path) -> None:
    """load_session read path includes agent_id and subagent metadata.

    Regression: the sessions.py read path must SELECT and populate the v13
    columns so that recall show / load_session surfaces subagent attribution.
    """
    import duckdb
    from recall.db.queries import insert_messages, insert_tool_calls
    from recall.db.schema import ensure_schema
    from recall.services.sessions import load_session

    parser = ClaudeCodeParser()
    session = parser.parse(SUBAGENT_FIXTURE).session

    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    ensure_schema(conn)

    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
        [session.id, session.source.value, session.source_path, session.source_session_id],
    )
    conn.execute(
        """INSERT INTO session_state (
            session_id, file_mtime, file_size, message_count, tool_count
        ) VALUES (?, ?, ?, ?, ?)""",
        [
            session.id,
            session.file_mtime,
            session.file_size,
            session.message_count,
            session.tool_count,
        ],
    )
    insert_messages(conn, session.messages)
    all_tool_calls = [tc for m in session.messages for tc in m.tool_calls]
    insert_tool_calls(conn, all_tool_calls)

    # Load via the read path
    loaded = load_session(session.id, include_tools=True, conn=conn)
    assert loaded is not None

    # Messages carry agent_id
    agent_msgs = [m for m in loaded.messages if m.agent_id is not None]
    assert len(agent_msgs) == 4
    for m in agent_msgs:
        assert m.agent_id == "agent_abc123"

    # Agent dispatch tool_call has subagent metadata
    all_tcs = [tc for m in loaded.messages for tc in m.tool_calls]
    dispatches = [tc for tc in all_tcs if tc.tool_name == "Agent"]
    assert len(dispatches) == 1
    assert dispatches[0].subagent_type == "Explore"
    assert dispatches[0].subagent_description == "Find auth middleware"
    assert dispatches[0].subagent_model == "sonnet"

    # Subagent tool calls carry agent_id
    subagent_tcs = [tc for tc in all_tcs if tc.agent_id is not None]
    assert len(subagent_tcs) == 1
    assert subagent_tcs[0].tool_name == "Bash"

    conn.close()
