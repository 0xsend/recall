from __future__ import annotations

import json
from typing import Any, cast

import pytest
from recall.core.fleet import FleetHost
from recall.core.types import Source
from recall.services.fleet_skills import fleet_stats_skills
from recall.services.fleet_transport import FleetRemoteResult

ALL_SOURCES = [source.value for source in Source]


def _payload(
    host: str,
    *,
    source: str = "codex",
    covered_sources: list[str] | None = None,
    bounded: bool = False,
) -> dict[str, object]:
    control = {
        "considered_sessions": 8,
        "attributed_invocations": 3,
        "unattributed_candidates": 2,
    }
    if not bounded:
        control = {
            "considered_sessions": 4,
            "attributed_invocations": 2,
            "unattributed_candidates": 1,
        }
    return {
        "rows": [
            {
                "skill_name": "engineering-practices:code-law",
                "source": source,
                "host": host,
                "invocations": 2,
                "sessions": 1,
            }
        ],
        "coverage": {
            "scope": "local",
            "expected_hosts": [host],
            "successful_hosts": [host],
            "covered_sources": covered_sources or ALL_SOURCES,
            "considered_sessions": 4,
            "attributed_invocations": 2,
            "unattributed_candidates": 1,
            "control": control,
        },
    }


def _invalid_payload(kind: str) -> str:
    payload = _payload("edge")
    coverage = cast(dict[str, Any], payload["coverage"])
    if kind == "multiple-local-hosts":
        coverage["expected_hosts"] = ["edge", "other"]
        coverage["successful_hosts"] = ["edge", "other"]
    elif kind == "row-total-mismatch":
        payload["rows"] = []
        coverage["attributed_invocations"] = 99
    elif kind == "unbounded-control-mismatch":
        coverage["control"] = {
            "considered_sessions": 8,
            "attributed_invocations": 3,
            "unattributed_candidates": 2,
        }
    else:
        raise AssertionError(f"unknown invalid payload kind: {kind}")
    return json.dumps(payload)


def test_fleet_skills_merges_local_and_all_remote_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hosts = (
        FleetHost(name="edge-a", ssh="a.example"),
        FleetHost(name="edge-b", ssh="b.example"),
    )
    seen: list[list[str]] = []
    seen_timeouts: list[float] = []

    def fake_run(
        host: FleetHost,
        remote_argv: list[str],
        *,
        timeout: float,
    ) -> FleetRemoteResult:
        seen.append(remote_argv)
        seen_timeouts.append(timeout)
        return FleetRemoteResult(
            ok=True,
            stdout=json.dumps(
                _payload(
                    f"reported-{host.name}",
                    covered_sources=["codex", "grok"],
                    bounded=True,
                )
            ),
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("recall.services.fleet_skills.run_remote", fake_run)
    result = fleet_stats_skills(
        hosts,
        local_payload=_payload(
            "control",
            covered_sources=["codex", "grok"],
            bounded=True,
        ),
        since="7d",
        sources=(Source.CODEX, Source.GROK),
    )

    assert result.errors == ()
    assert result.payload is not None
    assert result.payload["coverage"] == {
        "scope": "local+fleet",
        "expected_hosts": ["control", "edge-a", "edge-b"],
        "successful_hosts": ["control", "edge-a", "edge-b"],
        "covered_sources": ["codex", "grok"],
        "considered_sessions": 12,
        "attributed_invocations": 6,
        "unattributed_candidates": 3,
        "control": {
            "considered_sessions": 24,
            "attributed_invocations": 9,
            "unattributed_candidates": 6,
        },
    }
    assert {row["host"] for row in result.payload["rows"]} == {
        "control",
        "edge-a",
        "edge-b",
    }
    assert all("--local" in argv and "--json" in argv for argv in seen)
    assert all(argv.count("--source") == 2 for argv in seen)
    assert all(argv[-2:] == ["--since", "7d"] for argv in seen)
    assert seen_timeouts == [660.0, 660.0]


@pytest.mark.parametrize(
    "remote_result",
    [
        FleetRemoteResult(ok=False, stdout="", stderr="down", returncode=255, error="down"),
        FleetRemoteResult(ok=True, stdout="not-json", stderr="", returncode=0),
        FleetRemoteResult(
            ok=True,
            stdout=json.dumps(
                {
                    **_payload("edge"),
                    "coverage": {
                        **cast(dict[str, Any], _payload("edge")["coverage"]),
                        "covered_sources": ["codex"],
                    },
                }
            ),
            stderr="",
            returncode=0,
        ),
        FleetRemoteResult(
            ok=True,
            stdout=_invalid_payload("multiple-local-hosts"),
            stderr="",
            returncode=0,
        ),
        FleetRemoteResult(
            ok=True,
            stdout=_invalid_payload("row-total-mismatch"),
            stderr="",
            returncode=0,
        ),
        FleetRemoteResult(
            ok=True,
            stdout=_invalid_payload("unbounded-control-mismatch"),
            stderr="",
            returncode=0,
        ),
    ],
    ids=[
        "remote-failure",
        "invalid-json",
        "incomplete-sources",
        "multiple-local-hosts",
        "row-total-mismatch",
        "unbounded-control-mismatch",
    ],
)
def test_fleet_skills_returns_no_payload_on_any_remote_failure(
    monkeypatch: pytest.MonkeyPatch,
    remote_result: FleetRemoteResult,
) -> None:
    monkeypatch.setattr(
        "recall.services.fleet_skills.run_remote",
        lambda *_args, **_kwargs: remote_result,
    )

    result = fleet_stats_skills(
        (FleetHost(name="edge", ssh="edge.example"),),
        local_payload=_payload("control"),
    )

    assert result.payload is None
    assert len(result.errors) == 1
    assert result.errors[0]["name"] == "edge"


def test_fleet_skills_rejects_local_inventory_host_name_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "recall.services.fleet_skills.run_remote",
        lambda *_args, **_kwargs: FleetRemoteResult(
            ok=True,
            stdout=json.dumps(_payload("control")),
            stderr="",
            returncode=0,
        ),
    )

    result = fleet_stats_skills(
        (FleetHost(name="control", ssh="edge.example"),),
        local_payload=_payload("control"),
    )

    assert result.payload is None
    assert result.errors == (
        {"name": "fleet", "error": "endpoint host names must be unique: control"},
    )
