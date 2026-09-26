"""Fleet stats usage merge and fan-out (REQ-FLEET-MERGE-003 / CMD-002)."""

from __future__ import annotations

import json

import pytest
from recall.core.fleet import FleetHost
from recall.services.fleet_transport import DEFAULT_PER_HOST_TIMEOUT_SECONDS, FleetRemoteResult
from recall.services.fleet_usage import fleet_stats_usage, merge_usage_rows


def test_merge_usage_sums_same_key() -> None:
    rows = [
        {
            "source": "grok",
            "model": "grok-4.5",
            "host": "devbox",
            "input_tokens": 100,
            "cached_input_tokens": 40,
            "output_tokens": 10,
            "fresh_input_tokens": 60,
            "session_count": 2,
        },
        {
            "source": "grok",
            "model": "grok-4.5",
            "host": "devbox",
            "input_tokens": 50,
            "cached_input_tokens": 10,
            "output_tokens": 5,
            "fresh_input_tokens": 40,
            "session_count": 1,
        },
        {
            "source": "claude_code",
            "model": "opus",
            "host": "buildbox",
            "input_tokens": 9,
            "cached_input_tokens": 0,
            "output_tokens": 1,
            "fresh_input_tokens": None,
            "session_count": 1,
        },
    ]
    merged = merge_usage_rows(rows)
    by_key = {(r["source"], r["host"]): r for r in merged}
    g = by_key[("grok", "devbox")]
    assert g["input_tokens"] == 150
    assert g["cached_input_tokens"] == 50
    assert g["output_tokens"] == 15
    assert g["fresh_input_tokens"] == 100
    assert g["session_count"] == 3
    c = by_key[("claude_code", "buildbox")]
    assert c["input_tokens"] == 9
    assert c["fresh_input_tokens"] is None


def test_merge_stamps_inventory_host_when_missing() -> None:
    rows = [
        {"source": "grok", "model": None, "input_tokens": 1, "output_tokens": 0, "session_count": 1}
    ]
    merged = merge_usage_rows(rows, default_host="hub")
    assert merged[0]["host"] == "hub"


def test_merge_overrides_unattributed_sentinel_with_inventory_host() -> None:
    """A remote reporting the `local` sentinel is unattributed, not a host named
    "local" (REQ-FLEET-MERGE-002)."""
    rows = [
        {
            "source": "grok",
            "model": None,
            "host": "local",
            "input_tokens": 1,
            "output_tokens": 0,
            "session_count": 1,
        }
    ]
    merged = merge_usage_rows(rows, default_host="hub")
    assert merged[0]["host"] == "hub"


def test_fleet_stats_usage_stamps_inventory_name_over_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a remote whose usage rows carry the sentinel is attributed to
    its inventory name, not merged into a phantom "local" bucket."""
    hosts = (FleetHost(name="devbox", ssh="devbox.example"),)
    payload = json.dumps(
        [
            {
                "source": "codex",
                "model": "gpt-5",
                "host": "local",
                "input_tokens": 7,
                "output_tokens": 3,
                "session_count": 1,
            }
        ]
    )

    seen_timeouts: list[float] = []

    def fake_run(
        host: FleetHost,
        remote_argv: list[str],
        *,
        timeout: float,
    ) -> FleetRemoteResult:
        seen_timeouts.append(timeout)
        return FleetRemoteResult(ok=True, stdout=payload, stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_usage.run_remote", fake_run)
    result = fleet_stats_usage(hosts)
    assert result.hosts_ok == 1
    assert [r["host"] for r in result.rows] == ["devbox"]
    assert DEFAULT_PER_HOST_TIMEOUT_SECONDS == 45
    assert seen_timeouts == [45]


def test_fleet_stats_usage_merges_and_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts = (
        FleetHost(name="devbox", ssh="devbox.example"),
        FleetHost(name="down", ssh="down.example"),
    )

    def fake_run(host: FleetHost, remote_argv: list[str], **_kwargs: object) -> FleetRemoteResult:
        assert remote_argv[0] == "recall"
        assert "stats" in remote_argv and "usage" in remote_argv
        assert "--json" in remote_argv
        if host.name == "down":
            return FleetRemoteResult(
                ok=False, stdout="", stderr="timeout", returncode=255, error="timeout"
            )
        payload = [
            {
                "source": "grok",
                "model": "grok-4.5",
                "host": "devbox",
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "output_tokens": 2,
                "fresh_input_tokens": 10,
                "session_count": 1,
            }
        ]
        return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_usage.run_remote", fake_run)
    result = fleet_stats_usage(hosts, since="7d")
    assert result.hosts_ok == 1
    assert result.hosts_failed == 1
    assert len(result.rows) == 1
    assert result.rows[0]["input_tokens"] == 10
    assert result.rows[0]["host"] == "devbox"
    assert result.errors[0]["name"] == "down"


def test_fleet_stats_usage_all_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts = (FleetHost(name="a", ssh="a"),)

    def fake_run(*_a: object, **_k: object) -> FleetRemoteResult:
        return FleetRemoteResult(ok=False, stdout="", stderr="no", returncode=1, error="no")

    monkeypatch.setattr("recall.services.fleet_usage.run_remote", fake_run)
    result = fleet_stats_usage(hosts)
    assert result.hosts_ok == 0
    assert result.hosts_failed == 1
    assert result.rows == []
