"""`recall list --subagents` reaches the daemon and rows show their tree (REQ-CLI-031)."""

from types import SimpleNamespace

import pytest
from recall.cli import rpc
from recall.cli.app import app
from typer.testing import CliRunner

ROWS = [
    {
        "id": "rootsession",
        "source": "claude_code",
        "started_at": "2026-11-20T10:00:00Z",
        "message_count": 4,
        "tool_count": 1,
        "subagent_count": 2,
        "root_session_id": None,
    },
    {
        "id": "childsession",
        "source": "claude_code",
        "started_at": "2026-11-20T10:01:00Z",
        "message_count": 2,
        "tool_count": 0,
        "subagent_count": 0,
        "root_session_id": "rootsession",
    },
]


@pytest.fixture
def received(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    async def roster(params, client):
        calls.append(dict(params))
        return ROWS

    monkeypatch.setattr(
        rpc, "_in_process_server", SimpleNamespace(_methods={"recall.list": roster})
    )
    return calls


@pytest.mark.parametrize(
    ("arguments", "requested"),
    [([], False), (["--subagents"], True), (["--params", '{"subagents": true}'], True)],
)
def test_subagents_option_reaches_the_daemon_only_when_requested(
    received, arguments, requested
) -> None:
    result = CliRunner().invoke(app, ["list", *arguments, "--json"])

    assert result.exit_code == 0, result.output
    assert [call.get("subagents", False) for call in received] == [requested]
    if not requested:
        assert "subagents" not in received[0]


def test_text_rows_show_a_root_count_and_a_linked_root(received) -> None:
    result = CliRunner().invoke(app, ["list", "--format", "text"])

    assert result.exit_code == 0, result.output
    root_line, child_line = [line for line in result.stdout.splitlines() if "session" in line]
    assert root_line.endswith("subagents=2")
    assert child_line.endswith("root=rootsession")
