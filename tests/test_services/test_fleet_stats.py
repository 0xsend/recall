"""Tests for fail-closed fleet stats aggregation, blank reply rejection, and UTC cutoff pinning."""

from __future__ import annotations

import json

import pytest
from recall.core.fleet import FleetHost
from recall.services.fleet_stats import (
    FleetBashSuggestionsResult,
    fleet_stats_bash,
    fleet_stats_overview,
    fleet_stats_tokens,
    fleet_stats_tools,
)
from recall.services.fleet_transport import FleetRemoteResult
from recall.services.fleet_usage import fleet_stats_usage

from tests.fleet_support import inventory_executor


def test_fleet_stats_overview_sums_and_rejects_malformed_or_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts = (
        FleetHost(name="h1", ssh="h1.example"),
        FleetHost(name="h2", ssh="h2.example"),
        FleetHost(name="bad_bool", ssh="bad1.example"),
        FleetHost(name="blank", ssh="blank.example"),
    )
    seen_argvs: dict[str, list[str]] = {}

    def fake_run(host: FleetHost, remote_argv: list[str], **_kwargs: object) -> FleetRemoteResult:
        seen_argvs[host.name] = list(remote_argv)
        if host.name == "h1":
            return FleetRemoteResult(
                ok=True,
                stdout=json.dumps(
                    {
                        "sessions": 10,
                        "messages": 100,
                        "tool_calls": 40,
                        "bash_calls": 15,
                        "last_index_indexed": 999,
                    }
                ),
                stderr="",
                returncode=0,
            )
        if host.name == "h2":
            return FleetRemoteResult(
                ok=True,
                stdout=json.dumps(
                    {"sessions": 5, "messages": 50, "tool_calls": 20, "bash_calls": 7}
                ),
                stderr="",
                returncode=0,
            )
        if host.name == "bad_bool":
            return FleetRemoteResult(
                ok=True,
                stdout=json.dumps(
                    {"sessions": True, "messages": 1, "tool_calls": 1, "bash_calls": 1}
                ),
                stderr="",
                returncode=0,
            )
        return FleetRemoteResult(ok=True, stdout="   \n", stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", fake_run)
    result = fleet_stats_overview(inventory_executor(hosts), source="agy", since="7d")
    assert result.hosts_ok == 2
    assert result.hosts_failed == 2
    assert result.stats["sessions"] == 15
    assert result.stats["messages"] == 150
    assert result.stats["tool_calls"] == 60
    assert result.stats["bash_calls"] == 22
    assert result.stats["last_index_indexed"] is None
    assert result.stats["last_context_mode"] is None
    # Single resolved absolute UTC cutoff forwarded identically to every host
    cutoffs = {argv[argv.index("--since") + 1] for argv in seen_argvs.values()}
    assert len(cutoffs) == 1
    assert next(iter(cutoffs)).endswith("Z")
    assert all("--source" in argv and "antigravity_cli" in argv for argv in seen_argvs.values())


def test_fleet_stats_tools_merges_and_rejects_oversized_or_duplicate_host_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts = (
        FleetHost(name="h1", ssh="h1"),
        FleetHost(name="h2", ssh="h2"),
        FleetHost(name="oversized", ssh="over"),
        FleetHost(name="dup", ssh="dup"),
    )

    def fake_run(host: FleetHost, _argv: list[str], **_kwargs: object) -> FleetRemoteResult:
        if host.name == "h1":
            rows = [{"tool_name": "read_file", "count": 10}, {"tool_name": "Edit", "count": 5}]
        elif host.name == "h2":
            rows = [{"tool_name": "read_file", "count": 7}, {"tool_name": "Grep", "count": 5}]
        elif host.name == "dup":
            rows = [{"tool_name": "read_file", "count": 3}, {"tool_name": "read_file", "count": 3}]
        else:
            rows = [{"tool_name": f"t{i}", "count": 1} for i in range(51)]
        return FleetRemoteResult(ok=True, stdout=json.dumps(rows), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", fake_run)
    result = fleet_stats_tools(inventory_executor(hosts))
    assert result.hosts_ok == 2
    assert result.hosts_failed == 2
    assert {e["name"] for e in result.errors} == {"oversized", "dup"}
    assert result.rows == [
        {"tool_name": "read_file", "count": 17},
        {"tool_name": "Edit", "count": 5},
        {"tool_name": "Grep", "count": 5},
    ]


def test_fleet_stats_bash_suggest_promotes_tier_after_fleet_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two hosts with 30 `git status` each combine to 60 -> `high` (top-500 preserved)."""
    hosts = (
        FleetHost(name="h1", ssh="h1"),
        FleetHost(name="h2", ssh="h2"),
        FleetHost(name="oversized", ssh="over"),
        FleetHost(name="dup", ssh="dup"),
    )
    seen_argvs: list[list[str]] = []

    def fake_run(host: FleetHost, argv: list[str], **_kwargs: object) -> FleetRemoteResult:
        seen_argvs.append(list(argv))
        if host.name == "h1":
            payload = {
                "suggestions": [
                    {
                        "pattern": "git status",
                        "count": 30,
                        "confidence": "medium",
                        "reason": "No dangerous patterns detected",
                    },
                    {
                        "pattern": "ls *",
                        "count": 20,
                        "confidence": "medium",
                        "reason": "No dangerous patterns detected",
                    },
                ],
                "skipped": [],
            }
        elif host.name == "h2":
            payload = {
                "suggestions": [
                    {
                        "pattern": "git status",
                        "count": 30,
                        "confidence": "medium",
                        "reason": "No dangerous patterns detected",
                    },
                    {
                        "pattern": "ls *",
                        "count": 40,
                        "confidence": "review",
                        "reason": "Contains compound operators",
                    },
                ],
                "skipped": [
                    {"pattern": "rm *", "count": 5, "reason": "Destructive command"},
                ],
            }
        elif host.name == "dup":
            payload = {
                "suggestions": [
                    {
                        "pattern": "git status",
                        "count": 10,
                        "confidence": "medium",
                        "reason": "No dangerous patterns detected",
                    },
                    {
                        "pattern": "git status",
                        "count": 10,
                        "confidence": "medium",
                        "reason": "No dangerous patterns detected",
                    },
                ],
                "skipped": [],
            }
        else:
            payload = {
                "suggestions": [
                    {
                        "pattern": f"cmd{i} *",
                        "count": 1,
                        "confidence": "review",
                        "reason": "Low usage volume",
                    }
                    for i in range(501)
                ],
                "skipped": [],
            }
        return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", fake_run)
    result = fleet_stats_bash(inventory_executor(hosts), suggest=True)
    assert isinstance(result, FleetBashSuggestionsResult)
    assert result.hosts_ok == 2
    assert result.hosts_failed == 2
    assert {e["name"] for e in result.errors} == {"oversized", "dup"}
    assert all("--suggest" in argv for argv in seen_argvs)
    by_pat = {item["pattern"]: item for item in result.payload["suggestions"]}
    assert by_pat["git status"]["count"] == 60
    assert by_pat["git status"]["confidence"] == "high"
    assert by_pat["ls *"]["count"] == 60
    assert by_pat["ls *"]["confidence"] == "review"
    assert [s["pattern"] for s in result.payload["skipped"]] == ["rm *"]


def test_fleet_stats_bash_suggest_accepts_legacy_wildcard_display_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid v0.48.0 host with `(git, None)` and `(git, '*')` emits two `git *` rows."""
    hosts = (FleetHost(name="legacy", ssh="legacy"),)
    payload = {
        "suggestions": [
            {
                "pattern": "git *",
                "count": 30,
                "confidence": "medium",
                "reason": "No dangerous patterns detected",
            },
            {
                "pattern": "git *",
                "count": 25,
                "confidence": "medium",
                "reason": "No dangerous patterns detected",
            },
        ],
        "skipped": [],
    }

    def fake_run(_host: FleetHost, _argv: list[str], **_kwargs: object) -> FleetRemoteResult:
        return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", fake_run)
    result = fleet_stats_bash(inventory_executor(hosts), suggest=True)
    assert isinstance(result, FleetBashSuggestionsResult)
    assert result.hosts_ok == 1
    assert result.hosts_failed == 0
    assert result.payload["suggestions"] == [
        {
            "pattern": "git *",
            "count": 55,
            "confidence": "high",
            "reason": "No dangerous patterns detected",
        }
    ]
    # Filtered replies come only from upgraded coalescing producers, so duplicate
    # wildcard keys must still fail closed.
    filtered = fleet_stats_bash(inventory_executor(hosts), suggest=True, source="agy")
    assert isinstance(filtered, FleetBashSuggestionsResult)
    assert filtered.hosts_ok == 0
    assert filtered.hosts_failed == 1
    assert "duplicate bash pattern" in filtered.errors[0]["error"]


def test_fleet_stats_bash_raw_rejects_duplicate_and_oversized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts = (
        FleetHost(name="h1", ssh="h1"),
        FleetHost(name="dup", ssh="dup"),
        FleetHost(name="over", ssh="over"),
    )

    def fake_run(host: FleetHost, _argv: list[str], **_kwargs: object) -> FleetRemoteResult:
        if host.name == "h1":
            rows = [{"bash_base": "git", "bash_sub": "status", "count": 10, "is_compound": False}]
        elif host.name == "dup":
            rows = [
                {"bash_base": "git", "bash_sub": "status", "count": 5, "is_compound": False},
                {"bash_base": "git", "bash_sub": "status", "count": 5, "is_compound": True},
            ]
        else:
            rows = [
                {"bash_base": f"c{i}", "bash_sub": None, "count": 1, "is_compound": False}
                for i in range(101)
            ]
        return FleetRemoteResult(ok=True, stdout=json.dumps(rows), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", fake_run)
    result = fleet_stats_bash(inventory_executor(hosts), suggest=False)
    assert not isinstance(result, FleetBashSuggestionsResult)
    assert result.hosts_ok == 1
    assert result.hosts_failed == 2
    assert result.rows == [
        {"bash_base": "git", "bash_sub": "status", "count": 10, "is_compound": False}
    ]


def test_fleet_stats_tokens_merges_and_rejects_oversized_and_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts = (
        FleetHost(name="h1", ssh="h1"),
        FleetHost(name="h2", ssh="h2"),
        FleetHost(name="over", ssh="over"),
        FleetHost(name="dup", ssh="dup"),
    )

    def fake_run(host: FleetHost, _argv: list[str], **_kwargs: object) -> FleetRemoteResult:
        if host.name == "h1":
            rows = [{"repo": "org/a", "input_tokens": 100, "output_tokens": 20}]
        elif host.name == "h2":
            rows = [
                {"repo": "org/a", "input_tokens": 50, "output_tokens": 30},
                {"repo": None, "input_tokens": 10, "output_tokens": 5},
            ]
        elif host.name == "dup":
            rows = [
                {"repo": "org/a", "input_tokens": 5, "output_tokens": 1},
                {"repo": "org/a", "input_tokens": 5, "output_tokens": 1},
            ]
        else:
            rows = [{"repo": f"r{i}", "input_tokens": 1, "output_tokens": 1} for i in range(51)]
        return FleetRemoteResult(ok=True, stdout=json.dumps(rows), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", fake_run)
    result = fleet_stats_tokens(inventory_executor(hosts))
    assert result.hosts_ok == 2
    assert result.hosts_failed == 2
    assert {e["name"] for e in result.errors} == {"over", "dup"}
    assert result.rows == [
        {"repo": "org/a", "input_tokens": 150, "output_tokens": 50},
        {"repo": None, "input_tokens": 10, "output_tokens": 5},
    ]


def test_fleet_stats_usage_strict_validation_and_source_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts = (
        FleetHost(name="good", ssh="good"),
        FleetHost(name="wrong_source", ssh="wrong"),
        FleetHost(name="bool_tokens", ssh="bool"),
        FleetHost(name="null_cached", ssh="null_cached"),
        FleetHost(name="missing_cached", ssh="missing_cached"),
        FleetHost(name="dup", ssh="dup"),
        FleetHost(name="blank", ssh="blank"),
        FleetHost(name="over", ssh="over"),
    )

    def fake_run(host: FleetHost, argv: list[str], **_kwargs: object) -> FleetRemoteResult:
        assert "--source" in argv and "antigravity_cli" in argv
        if host.name == "good":
            payload = [
                {
                    "source": "antigravity_cli",
                    "model": "gemini-2.5",
                    "host": "good",
                    "input_tokens": 100,
                    "cached_input_tokens": 40,
                    "output_tokens": 10,
                    "fresh_input_tokens": 60,
                    "session_count": 2,
                }
            ]
            return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)
        if host.name == "wrong_source":
            payload = [
                {
                    "source": "codex",
                    "model": "gpt-5",
                    "host": "wrong",
                    "input_tokens": 999,
                    "cached_input_tokens": 0,
                    "output_tokens": 1,
                    "session_count": 1,
                }
            ]
            return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)
        if host.name == "bool_tokens":
            payload = [
                {
                    "source": "antigravity_cli",
                    "model": "gemini-2.5",
                    "host": "bool",
                    "input_tokens": True,
                    "cached_input_tokens": 0,
                    "output_tokens": 1,
                    "session_count": 1,
                }
            ]
            return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)
        if host.name == "null_cached":
            payload = [
                {
                    "source": "antigravity_cli",
                    "model": "gemini-2.5",
                    "host": "null_cached",
                    "input_tokens": 100,
                    "cached_input_tokens": None,
                    "output_tokens": 10,
                    "fresh_input_tokens": None,
                    "session_count": 1,
                }
            ]
            return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)
        if host.name == "missing_cached":
            payload = [
                {
                    "source": "antigravity_cli",
                    "model": "gemini-2.5",
                    "host": "missing_cached",
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "fresh_input_tokens": None,
                    "session_count": 1,
                }
            ]
            return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)
        if host.name == "dup":
            row = {
                "source": "antigravity_cli",
                "model": "gemini-2.5",
                "host": "dup",
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "output_tokens": 1,
                "fresh_input_tokens": None,
                "session_count": 1,
            }
            return FleetRemoteResult(
                ok=True, stdout=json.dumps([row, dict(row)]), stderr="", returncode=0
            )
        if host.name == "over":
            payload = [
                {
                    "source": "antigravity_cli",
                    "model": f"m{i}",
                    "host": "over",
                    "input_tokens": 1,
                    "cached_input_tokens": 0,
                    "output_tokens": 1,
                    "session_count": 1,
                }
                for i in range(201)
            ]
            return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)
        return FleetRemoteResult(ok=True, stdout="", stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", fake_run)
    result = fleet_stats_usage(inventory_executor(hosts), source="agy")
    assert result.hosts_ok == 1
    assert result.hosts_failed == 7
    assert len(result.rows) == 1
    assert result.rows[0]["source"] == "antigravity_cli"
    assert result.rows[0]["input_tokens"] == 100


def test_fleet_filtered_remote_no_such_option_includes_upgrade_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts = (FleetHost(name="oldpeer", ssh="oldpeer"),)

    def fake_run(_host: FleetHost, _argv: list[str], **_kwargs: object) -> FleetRemoteResult:
        return FleetRemoteResult(
            ok=False,
            stdout="",
            stderr="Error: No such option: --source",
            returncode=2,
            error="Error: No such option: --source",
        )

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", fake_run)
    result = fleet_stats_tools(inventory_executor(hosts), source="agy")
    assert result.hosts_ok == 0
    assert result.hosts_failed == 1
    assert "upgrade recall on this host" in result.errors[0]["error"]
