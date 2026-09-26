"""`search`, `list` and `stats` warn while reconciliation is behind (REQ-CLI-023).

Search is where an empty answer is most easily misread as "it never happened".
`recall live` already refuses to let a caller infer absence from an incomplete
page; these commands read the same reconciliation coverage and say the same
thing, on stderr in every output mode so structured stdout stays pure data
(REQ-CLI-012).

The commands bind `rpc_call_or_error` at import, while the status path imports
it per call — so patching the module attribute stubs the coverage read without
touching the command's own data call.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from recall.cli import list as list_cli
from recall.cli import search as search_cli
from recall.cli import stats as stats_cli
from recall.cli.app import app
from typer.testing import CliRunner

BACKLOG = "27362 discovered transcripts are not indexed yet"


def _stub_daemon_status(monkeypatch: pytest.MonkeyPatch, *, pending: int) -> None:
    monkeypatch.setenv("RECALL_CLI_STATUS_NOTICES", "true")
    monkeypatch.setattr(
        "recall.cli.rpc.rpc_call_or_error",
        lambda *_args, **_kwargs: {
            "runtime_status": {},
            "reconciliation": {"pending": pending, "catalog_scan_complete": False},
        },
    )


def _stub_command_rpc(monkeypatch: pytest.MonkeyPatch, module: Any, payload: Any) -> None:
    monkeypatch.setattr(module, "rpc_call_or_error", lambda *_args, **_kwargs: payload)


@pytest.mark.parametrize("argv", [["--json"], []])
def test_search_warns_about_the_backlog_on_stderr(monkeypatch, argv) -> None:
    _stub_daemon_status(monkeypatch, pending=27362)
    _stub_command_rpc(monkeypatch, search_cli, [{"lexical_match": True, "session_id": "abc"}])

    result = CliRunner().invoke(app, ["search", "git", *argv])

    assert result.exit_code == 0, result.output
    assert BACKLOG in result.stderr
    assert "inferring absence" in result.stderr
    assert BACKLOG not in result.stdout


@pytest.mark.parametrize("argv", [["--json"], []])
def test_list_warns_about_the_backlog_on_stderr(monkeypatch, argv) -> None:
    _stub_daemon_status(monkeypatch, pending=27362)
    _stub_command_rpc(monkeypatch, list_cli, [])

    result = CliRunner().invoke(app, ["list", *argv])

    assert result.exit_code == 0, result.output
    assert BACKLOG in result.stderr
    assert BACKLOG not in result.stdout


@pytest.mark.parametrize("argv", [["--json"], []])
def test_stats_warns_about_the_backlog_on_stderr(monkeypatch, argv) -> None:
    _stub_daemon_status(monkeypatch, pending=27362)
    _stub_command_rpc(monkeypatch, stats_cli, {"sessions": 1, "messages": 2})

    result = CliRunner().invoke(app, ["stats", *argv])

    assert result.exit_code == 0, result.output
    assert BACKLOG in result.stderr
    assert BACKLOG not in result.stdout


_STATS_SUBCOMMANDS: tuple[tuple[list[str], Any], ...] = (
    (["tools"], [{"tool_name": "Bash", "count": 3}]),
    (["bash"], [{"bash_base": "git", "bash_sub": "status", "count": 2}]),
    (["bash", "--suggest"], {"suggestions": [], "skipped": []}),
    (["tokens"], [{"repo": "recall", "input_tokens": 1, "output_tokens": 2}]),
    (["usage"], [{"source": "codex", "input_tokens": 1, "output_tokens": 2}]),
    (["skills", "--local"], {"coverage": {"scope": "local"}, "rows": []}),
)


@pytest.mark.parametrize("output_argv", [["--json"], []])
@pytest.mark.parametrize(
    "subcommand,payload",
    _STATS_SUBCOMMANDS,
    ids=[" ".join(argv) for argv, _ in _STATS_SUBCOMMANDS],
)
def test_stats_subcommands_warn_about_the_backlog_on_stderr(
    monkeypatch, subcommand, payload, output_argv
) -> None:
    """REQ-CLI-021/023: a subcommand reads the same index the bare command does,
    so a backlog or a drifted daemon makes its numbers just as wrong."""
    _stub_daemon_status(monkeypatch, pending=27362)
    _stub_command_rpc(monkeypatch, stats_cli, payload)

    result = CliRunner().invoke(app, ["stats", *subcommand, *output_argv])

    assert result.exit_code == 0, result.output
    assert BACKLOG in result.stderr
    assert BACKLOG not in result.stdout


def test_stats_subcommand_drops_the_index_freshness_line_in_structured_mode(monkeypatch) -> None:
    """REQ-CLI-021: agents on --json get actionable warnings, not a status line."""
    monkeypatch.setenv("RECALL_CLI_STATUS_NOTICES", "true")
    monkeypatch.setattr(
        "recall.cli.rpc.rpc_call_or_error",
        lambda *_args, **_kwargs: {
            "version_drift": True,
            "daemon_version": "0.29.1",
            "binary_version": "0.29.2",
            "runtime_status": {
                "last_successful_at": "2026-09-18T00:00:00+00:00",
                "last_run_kind": "daemon-scheduled",
            },
        },
    )
    _stub_command_rpc(monkeypatch, stats_cli, [{"tool_name": "Bash", "count": 3}])

    structured = CliRunner().invoke(app, ["stats", "tools", "--json"])
    # Non-TTY stdout defaults to TOON (REQ-CLI-013), so text mode is explicit.
    text = CliRunner().invoke(app, ["stats", "tools", "--format", "text"])

    assert structured.exit_code == 0, structured.output
    assert "0.29.1" in structured.stderr
    assert "Index: last updated" not in structured.stderr
    assert text.exit_code == 0, text.output
    assert "Index: last updated" in text.stderr


def test_commands_are_silent_once_reconciliation_has_caught_up(monkeypatch) -> None:
    _stub_daemon_status(monkeypatch, pending=0)
    _stub_command_rpc(monkeypatch, list_cli, [])

    result = CliRunner().invoke(app, ["list", "--json"])

    assert result.exit_code == 0, result.output
    assert "coverage is incomplete" not in result.stderr


def test_zero_result_search_guidance_names_the_backlog(monkeypatch) -> None:
    """REQ-CLI-019: a backlog is the one reason "no results" can mean "not yet"."""
    _stub_daemon_status(monkeypatch, pending=27362)
    _stub_command_rpc(monkeypatch, search_cli, [])

    result = CliRunner().invoke(app, ["search", "nothing-matches-this", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
    assert "no matches" in result.stderr
    assert "27362 transcripts are discovered but not yet indexed" in result.stderr
    assert "not proof of absence" in result.stderr
