"""Tests for `recall search` stderr guidance (REQ-CLI-018/019).

The guidance unit tests call `_emit_search_guidance` directly with synthetic
result dicts so they are deterministic and need no database. One end-to-end test
exercises the wiring through the real command in a structured output mode.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import _can_acquire_duckdb_lock, set_in_process_server
from recall.cli import search as search_cli
from recall.cli.app import app
from recall.core.config import AppConfig
from recall.services.indexer import index_sessions
from recall.services.rpc_server import RpcServer
from typer.testing import CliRunner

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)


# ---- _emit_search_guidance unit tests (no DB) ----


def test_guidance_empty_no_tool_emits_hints(capsys) -> None:
    search_cli._emit_search_guidance(
        [], query="foo bar", tool=None, session=None, source=None, mode=None, limit=20
    )
    err = capsys.readouterr().err
    assert "no matches" in err
    assert "recall show" in err
    assert "terminal" in err  # scope note: terminal-only commands aren't indexed
    assert "coverage.unsupported" in err


# ---- zero-result mode hint (REQ-CLI-019) ----


def test_guidance_empty_keyword_mode_does_not_suggest_keyword_mode(capsys) -> None:
    """`--mode keyword` returning nothing must not be told to try --mode keyword."""
    search_cli._emit_search_guidance(
        [], query="foo", tool=None, session=None, source=None, mode="keyword", limit=20
    )
    err = capsys.readouterr().err
    assert "try --mode keyword" not in err
    assert "drop `--mode keyword`" in err


def test_guidance_empty_vector_mode_suggests_keyword(capsys) -> None:
    """A vector miss can still have an exact lexical match, so keyword is the widening step."""
    search_cli._emit_search_guidance(
        [], query="foo", tool=None, session=None, source=None, mode="vector", limit=20
    )
    err = capsys.readouterr().err
    assert "--mode keyword" in err


@pytest.mark.parametrize("mode", ["hybrid", "auto", None])
def test_guidance_empty_widest_mode_suggests_no_mode(capsys, mode: str | None) -> None:
    """Hybrid ran both legs, and auto (or no --mode) already resolved to the widest
    mode this host can serve: naming a mode as the next thing to try is noise."""
    search_cli._emit_search_guidance(
        [], query="foo", tool=None, session=None, source=None, mode=mode, limit=20
    )
    err = capsys.readouterr().err
    assert "--mode" not in err
    assert "no matches" in err


def test_guidance_empty_tool_reprobe_reports_counts(capsys, monkeypatch) -> None:
    fake = [{"lexical_match": True}, {"lexical_match": True}, {"lexical_match": False}]
    monkeypatch.setattr(search_cli, "rpc_call_or_error", lambda method, params: fake)

    search_cli._emit_search_guidance(
        [], query="grpcurl", tool="Bash", session=None, source=None, mode=None, limit=20
    )
    err = capsys.readouterr().err
    assert "--tool Bash" in err
    assert "3 results without it" in err
    assert "2 keyword matches" in err


def test_guidance_empty_tool_reprobe_also_zero(capsys, monkeypatch) -> None:
    monkeypatch.setattr(search_cli, "rpc_call_or_error", lambda method, params: [])

    search_cli._emit_search_guidance(
        [], query="nope", tool="Bash", session=None, source=None, mode=None, limit=20
    )
    err = capsys.readouterr().err
    assert "0 without it" in err
    assert "indexed" in err


def test_guidance_empty_appends_pending_coverage_when_it_applies(capsys) -> None:
    """A backlog is the one reason absence really can mean 'not indexed yet'."""
    search_cli._emit_search_guidance(
        [],
        query="foo bar",
        tool=None,
        session=None,
        source=None,
        mode=None,
        limit=20,
        coverage_note="27362 transcripts are discovered but not yet indexed",
    )
    err = capsys.readouterr().err
    assert "no matches" in err
    assert "27362 transcripts are discovered but not yet indexed" in err


def test_guidance_empty_tool_also_carries_pending_coverage(capsys, monkeypatch) -> None:
    monkeypatch.setattr(search_cli, "rpc_call_or_error", lambda method, params: [])

    search_cli._emit_search_guidance(
        [],
        query="nope",
        tool="Bash",
        session=None,
        source=None,
        mode=None,
        limit=20,
        coverage_note="27362 transcripts are discovered but not yet indexed",
    )
    err = capsys.readouterr().err
    assert "0 without it" in err
    assert "27362 transcripts are discovered but not yet indexed" in err


def test_guidance_omits_coverage_sentence_when_coverage_is_complete(capsys) -> None:
    search_cli._emit_search_guidance(
        [], query="foo", tool=None, session=None, source=None, mode=None, limit=20
    )
    err = capsys.readouterr().err
    assert "not yet indexed" not in err


def test_guidance_semantic_only_banner(capsys) -> None:
    results = [{"lexical_match": False}, {"lexical_match": False}]
    search_cli._emit_search_guidance(
        results, query="foo", tool=None, session=None, source=None, mode=None, limit=20
    )
    err = capsys.readouterr().err
    assert "no keyword matches" in err
    assert "semantically-nearest" in err


def test_guidance_semantic_only_suppressed_in_vector_mode(capsys) -> None:
    results = [{"lexical_match": False}]
    search_cli._emit_search_guidance(
        results, query="foo", tool=None, session=None, source=None, mode="vector", limit=20
    )
    assert capsys.readouterr().err == ""


def test_guidance_silent_when_a_lexical_match_present(capsys) -> None:
    results = [{"lexical_match": True}, {"lexical_match": False}]
    search_cli._emit_search_guidance(
        results, query="foo", tool=None, session=None, source=None, mode=None, limit=20
    )
    assert capsys.readouterr().err == ""


# ---- one daemon_status read per invocation (REQ-CLI-019/021/023) ----


def _backlog_status() -> dict:
    """A daemon status with a backlog old enough to clear the notice's noise floor."""
    return {
        "runtime_status": {},
        "reconciliation": {
            "pending": 27362,
            "catalog_scan_complete": False,
            "coverage": [
                {
                    "source": "claude_code",
                    "configuration": "enabled",
                    "pending": 27362,
                    "oldest_pending_age": 86_400.0,
                }
            ],
        },
    }


def _record_rpc(monkeypatch, calls: list, *, results: list, status: dict) -> None:
    """Route both search and status RPCs through one recording stub.

    `search` binds `rpc_call_or_error` at import; `status_notices` imports it
    lazily from `recall.cli.rpc`. Patching one name alone leaves the other live.
    """

    def call(method: str, params: object = None, **_kwargs: object) -> object:
        calls.append((method, params))
        if method == "recall.search":
            return results
        if method == "recall.daemon_status":
            return status
        raise AssertionError(f"unexpected rpc method: {method}")

    monkeypatch.setattr(search_cli, "rpc_call_or_error", call)
    monkeypatch.setattr("recall.cli.rpc.rpc_call_or_error", call)


def test_search_reads_daemon_status_once_with_a_single_row_page(tmp_path, monkeypatch) -> None:
    """REQ-CLI-019/021: one status read per search, and only one source row."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    calls: list = []
    _record_rpc(monkeypatch, calls, results=[], status=_backlog_status())

    result = CliRunner().invoke(app, ["search", "zqxjkbwvzzznomatch", "--json"])

    assert result.exit_code == 0
    status_calls = [params for method, params in calls if method == "recall.daemon_status"]
    assert status_calls == [{"limit": 1}]


def test_search_prints_the_pending_coverage_fact_once(tmp_path, monkeypatch) -> None:
    """The zero-result clause and the status notice must not both name the backlog."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    calls: list = []
    _record_rpc(monkeypatch, calls, results=[], status=_backlog_status())

    result = CliRunner().invoke(app, ["search", "zqxjkbwvzzznomatch", "--json"])

    assert result.exit_code == 0
    assert result.stderr.count("27362") == 1


def test_search_makes_no_status_rpc_when_notices_are_disabled(tmp_path, monkeypatch) -> None:
    """`[cli] status_notices = false` must also spare the zero-result coverage read."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    monkeypatch.setenv("RECALL_CLI_STATUS_NOTICES", "false")
    calls: list = []
    _record_rpc(monkeypatch, calls, results=[], status=_backlog_status())

    result = CliRunner().invoke(app, ["search", "zqxjkbwvzzznomatch", "--json"])

    assert result.exit_code == 0
    assert [method for method, _ in calls] == ["recall.search"]
    assert "note:" in result.stderr  # the guidance itself still runs
    assert "27362" not in result.stderr


def test_search_with_results_still_warns_about_the_backlog(tmp_path, monkeypatch) -> None:
    """Suppression is scoped to the zero-result clause, not to the notice itself."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    calls: list = []
    _record_rpc(
        monkeypatch,
        calls,
        results=[{"session_id": "a1", "source": "codex", "score": 1.0, "lexical_match": True}],
        status=_backlog_status(),
    )

    result = CliRunner().invoke(app, ["search", "hello", "--json"])

    assert result.exit_code == 0
    assert result.stderr.count("27362") == 1
    assert "coverage is incomplete" in result.stderr


# ---- end-to-end wiring through the real command ----


def _install_codex_fixture(tmp_path: Path) -> None:
    target = tmp_path / ".codex" / "sessions" / "s1"
    target.mkdir(parents=True)
    fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "session1" / "rollout.jsonl"
    )
    shutil.copy(fixture, target / "rollout.jsonl")


def _setup_indexed_env(tmp_path: Path, monkeypatch) -> AppConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    monkeypatch.setattr("recall.services.daemon._read_crontab", lambda: "")
    _install_codex_fixture(tmp_path)
    # embed=False -> keyword-only DB, so a gibberish query is a deterministic empty.
    index_sessions(source=None, full=True, recreate=True, verbose=False)
    config = AppConfig.load()
    set_in_process_server(RpcServer(config=config))
    return config


@_requires_duckdb_lock
def test_search_results_carry_lexical_match(tmp_path, monkeypatch) -> None:
    _setup_indexed_env(tmp_path, monkeypatch)
    runner = CliRunner()

    result = runner.invoke(app, ["search", "hello", "--json"])

    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert rows
    assert all("lexical_match" in row for row in rows)
    assert any(row["lexical_match"] for row in rows)
