from __future__ import annotations

import json
import re

import pytest
from recall.cli.app import app
from recall.cli.manifest import output_fields_for
from typer.testing import CliRunner

_SUCCESS_SUMMARY = {
    "index_summary": {"indexed": 1, "skipped": 2, "failed": 3},
    "swapped": False,
    "embed_summary": None,
}


def test_text_output_warns_when_record_status_persisted_false(monkeypatch) -> None:
    summary = {**_SUCCESS_SUMMARY, "record_status_persisted": False}
    monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", lambda *_args, **_kwargs: summary)

    result = CliRunner().invoke(app, ["daemon", "--once", "--format", "text"])

    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[0] == (
        "WARN: cycle ran but status metadata was not persisted; "
        "run `recall daemon status` to inspect runtime_state"
    )
    assert lines[1] == "Daemon cycle indexed 1, skipped 2, failed 3."


def test_text_output_no_warning_when_record_status_persisted_true(monkeypatch) -> None:
    summary = {**_SUCCESS_SUMMARY, "record_status_persisted": True}
    monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", lambda *_args, **_kwargs: summary)

    result = CliRunner().invoke(app, ["daemon", "--once", "--format", "text"])

    assert result.exit_code == 0
    assert "status metadata was not persisted" not in result.stdout
    assert "Daemon cycle indexed 1, skipped 2, failed 3." in result.stdout


def test_manifest_exposes_record_status_persisted_field() -> None:
    assert "record_status_persisted" in (output_fields_for("daemon") or set())


def test_daemon_status_manifest_exposes_fts_rebuild_backoff_fields() -> None:
    fields = output_fields_for("daemon status") or set()

    assert "last_fts_rebuild_failure_at" in fields
    assert "last_fts_rebuild_failure_reason" in fields
    assert "fts_rebuild_consecutive_failures" in fields
    assert "fts_rebuild_next_retry_at" in fields


def test_daemon_status_manifest_exposes_embed_loop_liveness_fields() -> None:
    fields = output_fields_for("daemon status") or set()

    assert "embed_loop_iterations" in fields
    assert "embed_loop_last_iteration_at" in fields
    assert "embed_loop_last_trigger" in fields
    assert "embed_loop_stage" in fields
    assert "embed_loop_stage_at" in fields
    assert "embed_loop_last_outcome" in fields
    assert "embed_loop_next_interval" in fields
    assert "embed_requested_cycles" in fields
    assert "embed_requested_at" in fields
    assert "embed_requested_stage" in fields
    assert "embed_requested_stage_at" in fields
    assert "embed_deferred_reason" in fields
    assert "embed_deferred_at" in fields
    assert "embed_pending_at" in fields
    assert "embed_last_error" in fields
    assert "embed_cooldown_sessions" in fields
    assert "embed_cooldown_until" in fields


def test_daemon_status_manifest_exposes_catchup_progress_fields() -> None:
    fields = output_fields_for("daemon status") or set()

    assert "catchup_in_progress" in fields
    assert "catchup_total" in fields
    assert "catchup_done" in fields


@pytest.mark.parametrize(
    ("command", "method", "paused"),
    [("pause", "recall.daemon_pause", True), ("resume", "recall.daemon_resume", False)],
)
def test_pause_and_resume_project_the_confirmed_durable_state(
    monkeypatch: pytest.MonkeyPatch, command: str, method: str, paused: bool
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def rpc(name: str, params: dict[str, object], **_kwargs: object) -> dict[str, bool]:
        calls.append((name, params))
        return {"paused": paused}

    monkeypatch.setattr("recall.cli.daemon.rpc_call_or_error", rpc)
    result = CliRunner().invoke(app, ["daemon", command, "--json"])

    assert result.exit_code == 0, result.output
    assert calls == [(method, {})]
    assert json.loads(result.stdout) == {"paused": paused}
    assert output_fields_for(f"daemon {command}") == {"paused"}


# --- verbosity help on the RPC-routed paths -----------------------


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _help_text(argv: list[str]) -> str:
    """Render `--help` as one line of prose.

    Rich wraps each help column and frames it in box-drawing characters, so a
    sentence arrives split across rows with `│` between the halves. Under
    GitHub Actions (or FORCE_COLOR) Typer also forces ANSI styling, which
    splits words such as `--help` with escape sequences. Drop the escapes and
    the frame, then collapse whitespace, so assertions can name whole phrases.
    """
    result = CliRunner().invoke(app, [*argv, "--help"])
    assert result.exit_code == 0, result.output
    plain = _ANSI_ESCAPE.sub("", result.stdout)
    unframed = plain.translate({ord(char): " " for char in "│╭╮╰╯─"})
    return " ".join(unframed.split())


def test_index_verbose_help_points_at_the_daemon_log_level() -> None:
    """`recall index` is served by the daemon, which ignores the client's -v."""
    text = _help_text(["index"])

    assert "No effect" in text
    assert "daemon.log_level in config.toml" in text
    assert "RECALL_DAEMON_LOG_LEVEL" in text


def test_daemon_verbose_help_scopes_verbosity_to_the_foreground_run() -> None:
    """`--once` is served by the running daemon; only a foreground run is this process."""
    text = _help_text(["daemon"])

    assert "Verbose logging for a foreground daemon" in text
    assert "With --once the running daemon serves the cycle" in text
    assert "daemon.log_level in config.toml" in text
    assert "RECALL_DAEMON_LOG_LEVEL" in text


def test_verbose_help_survives_rich_markup_stripping() -> None:
    """Square brackets are Rich markup: `[daemon] log_level` renders as ` log_level`."""
    for argv in (["index"], ["daemon"]):
        assert "log_level" in _help_text(argv)
        assert "Set  log_level" not in _help_text(argv)
