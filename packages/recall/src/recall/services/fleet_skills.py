"""Fail-closed census over the shared resolved fleet scope."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from recall.core.rpc_params import SkillStatsParams
from recall.core.types import Source
from recall.services.fleet_endpoints import EndpointSuccess, FleetExecutor, FleetOperation
from recall.services.fleet_transport import DEFAULT_CONCURRENCY

SKILL_CENSUS_PER_HOST_TIMEOUT_SECONDS = 660.0


@dataclass(frozen=True)
class FleetSkillsResult:
    payload: dict[str, Any] | None
    errors: tuple[dict[str, str], ...]


def fleet_stats_skills(
    executor: FleetExecutor,
    *,
    since: str | None = None,
    sources: Sequence[Source] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = SKILL_CENSUS_PER_HOST_TIMEOUT_SECONDS,
) -> FleetSkillsResult:
    selected = tuple(sources) if sources else tuple(Source)
    expected_sources = [source.value for source in selected]
    argv = ["recall", "stats", "skills", "--local", "--json"]
    for source in selected:
        argv.extend(["--source", source.value])
    if since:
        argv.extend(["--since", since])

    def decode(value: Any) -> dict[str, Any]:
        return _validate_payload(
            value, expected_sources=expected_sources, bounded=since is not None
        )

    outcomes = executor.execute(
        FleetOperation(
            "recall.stats_skills",
            SkillStatsParams(since=since, source=expected_sources),
            argv,
            decode,
            local_seconds=600.0,
            remote_seconds=timeout,
        ),
        concurrency=concurrency,
    )
    errors = tuple(
        {"name": outcome.name, "error": outcome.error}
        for outcome in outcomes
        if not isinstance(outcome, EndpointSuccess)
    )
    if errors:
        return FleetSkillsResult(None, errors)
    payloads = [
        (outcome.name, outcome.value)
        for outcome in outcomes
        if isinstance(outcome, EndpointSuccess)
    ]
    names = [name for name, _ in payloads]
    population_fields = ("considered_sessions", "attributed_invocations", "unattributed_candidates")
    coverage: dict[str, Any] = {
        "scope": "local+fleet",
        "expected_hosts": names,
        "successful_hosts": list(names),
        "covered_sources": expected_sources,
    }
    for field in population_fields:
        coverage[field] = sum(int(payload["coverage"][field]) for _, payload in payloads)
    coverage["control"] = {
        field: sum(int(payload["coverage"]["control"][field]) for _, payload in payloads)
        for field in population_fields
    }
    return FleetSkillsResult({"rows": _merge_rows(payloads), "coverage": coverage}, ())


def _validate_payload(
    value: object,
    *,
    expected_sources: list[str],
    bounded: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"expected object, got {type(value).__name__}")
    rows = value.get("rows")
    coverage = value.get("coverage")
    if not isinstance(rows, list) or not isinstance(coverage, dict):
        raise ValueError("payload requires rows array and coverage object")
    if coverage.get("scope") != "local":
        raise ValueError("remote coverage scope must be local")
    covered_sources = coverage.get("covered_sources")
    if covered_sources != expected_sources:
        raise ValueError(f"covered sources {covered_sources!r} != requested {expected_sources!r}")
    expected_hosts = _string_list(coverage.get("expected_hosts"), "expected_hosts")
    successful_hosts = _string_list(coverage.get("successful_hosts"), "successful_hosts")
    if len(expected_hosts) != 1 or successful_hosts != expected_hosts:
        raise ValueError("local coverage must contain exactly one successful host")

    population: dict[str, int] = {}
    for field in (
        "considered_sessions",
        "attributed_invocations",
        "unattributed_candidates",
    ):
        population[field] = _nonnegative_int(coverage.get(field), field)
    control = coverage.get("control")
    if not isinstance(control, dict):
        raise ValueError("coverage.control must be an object")
    control_population: dict[str, int] = {}
    for field in (
        "considered_sessions",
        "attributed_invocations",
        "unattributed_candidates",
    ):
        control_population[field] = _nonnegative_int(control.get(field), f"control.{field}")
        if bounded and control_population[field] < population[field]:
            raise ValueError(f"control.{field} cannot be smaller than window {field}")
        if not bounded and control_population[field] != population[field]:
            raise ValueError(f"control.{field} must equal unbounded window {field}")

    normalized_rows: list[dict[str, Any]] = []
    row_keys: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{index}] must be an object")
        skill_name = row.get("skill_name")
        source = row.get("source")
        host = row.get("host")
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError(f"rows[{index}].skill_name must be non-empty")
        if not isinstance(source, str) or source not in expected_sources:
            raise ValueError(f"rows[{index}].source is outside requested coverage")
        if not isinstance(host, str) or not host.strip():
            raise ValueError(f"rows[{index}].host must be non-empty")
        if host.strip() != expected_hosts[0]:
            raise ValueError(f"rows[{index}].host is outside local host coverage")
        invocations = _positive_int(row.get("invocations"), f"rows[{index}].invocations")
        sessions = _positive_int(row.get("sessions"), f"rows[{index}].sessions")
        if sessions > invocations:
            raise ValueError(f"rows[{index}].sessions cannot exceed invocations")
        key = (skill_name.strip(), source, host.strip())
        if key in row_keys:
            raise ValueError(f"rows[{index}] duplicates skill/source/host granularity")
        row_keys.add(key)
        normalized_rows.append(
            {
                "skill_name": skill_name.strip(),
                "source": source,
                "host": host.strip(),
                "invocations": invocations,
                "sessions": sessions,
            }
        )

    row_invocations = sum(int(row["invocations"]) for row in normalized_rows)
    if row_invocations != population["attributed_invocations"]:
        raise ValueError("coverage.attributed_invocations does not match row totals")

    return {
        "rows": normalized_rows,
        "coverage": {
            **coverage,
            "expected_hosts": expected_hosts,
            "successful_hosts": successful_hosts,
            "covered_sources": list(expected_sources),
        },
    }


def _merge_rows(payloads: Sequence[tuple[str | None, dict[str, Any]]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for endpoint_host, payload in payloads:
        for row in payload["rows"]:
            host = endpoint_host or str(row["host"])
            key = (str(row["skill_name"]), str(row["source"]), host)
            bucket = buckets.setdefault(
                key,
                {
                    "skill_name": key[0],
                    "source": key[1],
                    "host": key[2],
                    "invocations": 0,
                    "sessions": 0,
                },
            )
            bucket["invocations"] += int(row["invocations"])
            bucket["sessions"] += int(row["sessions"])
    return [buckets[key] for key in sorted(buckets)]


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"coverage.{field} must be an array of non-empty strings")
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"coverage.{field} must be an array of non-empty strings")
        strings.append(item.strip())
    return strings


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: object, field: str) -> int:
    parsed = _nonnegative_int(value, field)
    if parsed == 0:
        raise ValueError(f"{field} must be positive")
    return parsed
