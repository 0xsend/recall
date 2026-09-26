"""Fleet list/search/show fan-out (REQ-FLEET-CMD-002/003, MERGE-004/005)."""

from __future__ import annotations

import json

import pytest
from recall.core.fleet import FleetHost
from recall.services.fleet_query import (
    fleet_list,
    fleet_live,
    fleet_search,
    fleet_show,
    merge_list_rows,
    merge_live_rows,
    merge_search_rows,
    stamp_host,
)
from recall.services.fleet_transport import FleetRemoteResult


@pytest.mark.parametrize(
    ("case_id", "row_host", "expected"),
    [
        ("HP-01", "devbox", "devbox"),
        ("HP-02", "WorkstationOne", "WorkstationOne"),
        ("BV-01", "local", "inventory-name"),
        ("BV-02", "  local  ", "inventory-name"),
        ("BV-03", "", "inventory-name"),
        ("BV-04", "   ", "inventory-name"),
        ("BV-05", None, "inventory-name"),
        ("EDGE-01", "localhost", "localhost"),
        ("EDGE-02", "local-dev", "local-dev"),
        ("EDGE-03", "LOCAL", "LOCAL"),
    ],
)
def test_stamp_host_treats_unattributed_sentinel_as_unset(
    case_id: str,
    row_host: str | None,
    expected: str,
) -> None:
    """`local` is the REQ-HOST-API-004 fallback a remote emits for rows it could
    not attribute, not a machine identity. Over an SSH hop the inventory name is
    the authoritative provenance, so the sentinel MUST lose to it
    (REQ-FLEET-MERGE-002). Real hostnames that merely contain "local" must not be
    caught by the sentinel check."""
    row: dict[str, object] = {"id": "s1"}
    if row_host is not None:
        row["host"] = row_host
    stamped = stamp_host(row, "inventory-name")
    assert stamped["host"] == expected, case_id
    assert stamped["id"] == "s1", case_id


def test_stamp_host_does_not_mutate_the_input_row() -> None:
    row = {"id": "s1", "host": "local"}
    stamped = stamp_host(row, "devbox")
    assert row["host"] == "local"
    assert stamped["host"] == "devbox"


def test_merge_list_rows_sorts_by_recency_and_limits() -> None:
    rows = [
        {"id": "old", "host": "a", "started_at": "2026-01-01T00:00:00+00:00"},
        {"id": "new", "host": "b", "ended_at": "2026-08-01T00:00:00+00:00"},
        {"id": "mid", "host": "a", "started_at": "2026-06-01T00:00:00+00:00"},
    ]
    merged = merge_list_rows(rows, limit=2)
    assert [r["id"] for r in merged] == ["new", "mid"]


def test_merge_search_rows_sorts_by_score() -> None:
    rows = [
        {"session_id": "a", "score": 0.1, "host": "h1"},
        {"session_id": "b", "score": 0.9, "host": "h2"},
    ]
    merged = merge_search_rows(rows, limit=1)
    assert merged[0]["session_id"] == "b"
    assert merged[0]["host"] == "h2"


def test_fleet_list_stamps_host_and_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts = (
        FleetHost(name="devbox", ssh="devbox"),
        FleetHost(name="down", ssh="down"),
    )

    def fake_run(host: FleetHost, remote_argv: list[str], **_k: object) -> FleetRemoteResult:
        assert remote_argv[:2] == ["recall", "list"]
        assert "--json" in remote_argv
        if host.name == "down":
            return FleetRemoteResult(ok=False, stdout="", stderr="no", returncode=1, error="no")
        payload = [
            {
                "id": "sess1",
                "source": "grok",
                "started_at": "2026-08-01T12:00:00+00:00",
            }
        ]
        return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_query.run_remote", fake_run)
    result = fleet_list(hosts, limit=10)
    assert result.hosts_ok == 1
    assert result.hosts_failed == 1
    assert result.rows[0]["host"] == "devbox"
    assert result.rows[0]["id"] == "sess1"


def test_fleet_search_merges_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts = (
        FleetHost(name="a", ssh="a"),
        FleetHost(name="b", ssh="b"),
    )

    def fake_run(host: FleetHost, remote_argv: list[str], **_k: object) -> FleetRemoteResult:
        assert "search" in remote_argv
        score = 0.2 if host.name == "a" else 0.8
        payload = [
            {
                "kind": "message",
                "session_id": f"s-{host.name}",
                "source": "grok",
                "score": score,
                "host": host.name,
            }
        ]
        return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_query.run_remote", fake_run)
    result = fleet_search(hosts, query="git", limit=5)
    assert result.hosts_ok == 2
    assert result.rows[0]["session_id"] == "s-b"
    assert all(r.get("host") for r in result.rows)


def test_fleet_show_requires_host_when_ambiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts = (
        FleetHost(name="a", ssh="a"),
        FleetHost(name="b", ssh="b"),
    )

    def fake_run(host: FleetHost, remote_argv: list[str], **_k: object) -> FleetRemoteResult:
        payload = {"id": "same", "source": "grok", "host": host.name}
        return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_query.run_remote", fake_run)
    result = fleet_show(hosts, session_id="same")
    assert result.session is None
    assert set(result.ambiguous_hosts) == {"a", "b"}


def test_fleet_show_with_host_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts = (
        FleetHost(name="a", ssh="a"),
        FleetHost(name="b", ssh="b"),
    )
    seen: list[str] = []

    def fake_run(host: FleetHost, remote_argv: list[str], **_k: object) -> FleetRemoteResult:
        seen.append(host.name)
        payload = {"id": "x", "source": "grok"}
        return FleetRemoteResult(ok=True, stdout=json.dumps(payload), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_query.run_remote", fake_run)
    result = fleet_show(hosts, session_id="x", host_name="b")
    assert seen == ["b"]
    assert result.session is not None
    assert result.session["host"] == "b"


def _live_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "path": "/home/a/.claude/projects/p/s.jsonl",
        "liveness": "idle",
        "freshness": {"lag_bytes": 0, "current": True},
        "turn": {"state": "working"},
        "id": "sess1",
        "source": "claude_code",
        "last_activity_at": "2026-09-08T12:00:00+00:00",
    }
    row.update(overrides)
    return row


def test_merge_live_rows_puts_running_agents_above_quiet_ones() -> None:
    """A fleet monitor scans the top of the list; an idle session must not lead it."""
    rows = [
        _live_row(id="idle-recent", liveness="idle", last_activity_at="2026-09-08T12:00:00+00:00"),
        _live_row(id="active-old", liveness="active", last_activity_at="2026-09-08T09:00:00+00:00"),
        _live_row(
            id="ended-newest", liveness="ended", last_activity_at="2026-09-08T13:00:00+00:00"
        ),
    ]

    merged = merge_live_rows(rows, limit=10)

    assert [row["id"] for row in merged] == ["active-old", "idle-recent", "ended-newest"]


def test_merge_live_rows_breaks_ties_by_most_recent_activity() -> None:
    rows = [
        _live_row(id="older", liveness="active", last_activity_at="2026-09-08T09:00:00+00:00"),
        _live_row(id="newer", liveness="active", last_activity_at="2026-09-08T12:00:00+00:00"),
    ]

    merged = merge_live_rows(rows, limit=10)

    assert [row["id"] for row in merged] == ["newer", "older"]


def test_merge_live_rows_applies_the_global_limit_after_ordering() -> None:
    """Per-host limits are local; the caller asked for N across the whole fleet."""
    rows = [
        _live_row(id="idle", liveness="idle"),
        _live_row(id="active", liveness="active"),
    ]

    merged = merge_live_rows(rows, limit=1)

    assert [row["id"] for row in merged] == ["active"]


def test_merge_live_rows_keeps_a_row_with_no_activity_stamp() -> None:
    """An unindexed transcript has no `last_activity_at` yet and is still live."""
    rows = [_live_row(id="unindexed", liveness="active", last_activity_at=None)]

    assert [row["id"] for row in merge_live_rows(rows, limit=10)] == ["unindexed"]


def test_fleet_live_stamps_host_and_skips_a_down_host(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts = (
        FleetHost(name="devbox", ssh="devbox"),
        FleetHost(name="down", ssh="down"),
    )

    def fake_run(host: FleetHost, remote_argv: list[str], **_k: object) -> FleetRemoteResult:
        assert remote_argv[:2] == ["recall", "live"]
        assert "--json" in remote_argv
        # Inventory --host selects machines; the remote must not be given one.
        assert "--host" not in remote_argv
        if host.name == "down":
            return FleetRemoteResult(ok=False, stdout="", stderr="no", returncode=1, error="no")
        return FleetRemoteResult(ok=True, stdout=json.dumps([_live_row()]), stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_query.run_remote", fake_run)
    result = fleet_live(hosts, limit=10)

    assert result.hosts_ok == 1
    assert result.hosts_failed == 1
    assert result.rows[0]["host"] == "devbox"
    assert result.rows[0]["id"] == "sess1"


def test_fleet_live_forwards_all_and_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(host: FleetHost, remote_argv: list[str], **_k: object) -> FleetRemoteResult:
        seen.append(remote_argv)
        return FleetRemoteResult(ok=True, stdout="[]", stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_query.run_remote", fake_run)
    fleet_live(
        (FleetHost(name="a", ssh="a"),),
        include_idle=True,
        source="codex",
        project="recall",
        limit=7,
    )

    assert seen == [
        [
            "recall",
            "live",
            "--json",
            "--limit",
            "7",
            "--all",
            "--source",
            "codex",
            "--project",
            "recall",
        ]
    ]


def test_fleet_live_host_filter_selects_which_machines_to_reach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached: list[str] = []

    def fake_run(host: FleetHost, remote_argv: list[str], **_k: object) -> FleetRemoteResult:
        reached.append(host.name)
        return FleetRemoteResult(ok=True, stdout="[]", stderr="", returncode=0)

    monkeypatch.setattr("recall.services.fleet_query.run_remote", fake_run)
    fleet_live(
        (FleetHost(name="a", ssh="a"), FleetHost(name="b", ssh="b")),
        host_filter="b",
        limit=5,
    )

    assert reached == ["b"]
