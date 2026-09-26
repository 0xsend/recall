"""Fail-closed fleet fan-out for the cross-harness skill census."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from recall.core.fleet import FleetHost
from recall.core.types import Source
from recall.services.fleet_transport import (
    DEFAULT_CONCURRENCY,
    FleetRemoteResult,
    run_remote,
)

logger = logging.getLogger("recall.fleet")

SKILL_CENSUS_PER_HOST_TIMEOUT_SECONDS = 660.0


@dataclass(frozen=True)
class FleetSkillsResult:
    payload: dict[str, Any] | None
    errors: tuple[dict[str, str], ...]


def fleet_stats_skills(
    hosts: Iterable[FleetHost],
    *,
    local_payload: dict[str, Any],
    since: str | None = None,
    sources: Sequence[Source] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = SKILL_CENSUS_PER_HOST_TIMEOUT_SECONDS,
) -> FleetSkillsResult:
    """Merge local and remote skill payloads only when every endpoint is complete."""
    host_list = list(hosts)
    selected = tuple(sources) if sources else tuple(Source)
    expected_sources = [source.value for source in selected]
    errors: list[dict[str, str]] = []

    try:
        local = _validate_payload(
            local_payload,
            expected_sources=expected_sources,
            bounded=since is not None,
        )
    except ValueError as err:
        return FleetSkillsResult(payload=None, errors=({"name": "local", "error": str(err)},))

    if not host_list:
        return FleetSkillsResult(
            payload=None,
            errors=({"name": "fleet", "error": "no configured remote hosts"},),
        )

    local_host = str(local["coverage"]["expected_hosts"][0])
    endpoint_names = [local_host, *(host.name for host in host_list)]
    collisions = sorted(name for name in set(endpoint_names) if endpoint_names.count(name) > 1)
    if collisions:
        return FleetSkillsResult(
            payload=None,
            errors=(
                {
                    "name": "fleet",
                    "error": f"endpoint host names must be unique: {', '.join(collisions)}",
                },
            ),
        )

    remote_payloads: dict[str, dict[str, Any]] = {}
    workers = max(1, min(concurrency, len(host_list)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _fetch_for_host,
                host,
                since=since,
                sources=selected,
                expected_sources=expected_sources,
                timeout=timeout,
            ): host
            for host in host_list
        }
        for future in as_completed(futures):
            host = futures[future]
            try:
                result, payload = future.result()
            except Exception as err:
                logger.exception("fleet stats skills crashed for %s", host.name)
                errors.append({"name": host.name, "error": str(err)})
                continue
            if not result.ok or payload is None:
                errors.append(
                    {"name": host.name, "error": result.error or "remote skill query failed"}
                )
                continue
            remote_payloads[host.name] = payload

    if errors:
        errors.sort(key=lambda item: item["name"])
        return FleetSkillsResult(payload=None, errors=tuple(errors))

    ordered_payloads = [(None, local)] + [
        (host.name, remote_payloads[host.name]) for host in host_list
    ]
    merged_rows = _merge_rows(ordered_payloads)
    endpoint_hosts = endpoint_names
    population_fields = (
        "considered_sessions",
        "attributed_invocations",
        "unattributed_candidates",
    )
    coverage: dict[str, Any] = {
        "scope": "local+fleet",
        "expected_hosts": endpoint_hosts,
        "successful_hosts": list(endpoint_hosts),
        "covered_sources": expected_sources,
    }
    for field in population_fields:
        coverage[field] = sum(
            int(payload["coverage"][field]) for _host, payload in ordered_payloads
        )
    coverage["control"] = {
        field: sum(
            int(payload["coverage"]["control"][field]) for _host, payload in ordered_payloads
        )
        for field in population_fields
    }
    return FleetSkillsResult(
        payload={"rows": merged_rows, "coverage": coverage},
        errors=(),
    )


def _fetch_for_host(
    host: FleetHost,
    *,
    since: str | None,
    sources: Sequence[Source],
    expected_sources: list[str],
    timeout: float,
) -> tuple[FleetRemoteResult, dict[str, Any] | None]:
    remote_argv = ["recall", "stats", "skills", "--local", "--json"]
    for source in sources:
        remote_argv.extend(["--source", source.value])
    if since:
        remote_argv.extend(["--since", since])

    result = run_remote(host, remote_argv, timeout=timeout)
    if not result.ok:
        return result, None
    try:
        decoded = json.loads(result.stdout)
        payload = _validate_payload(
            decoded,
            expected_sources=expected_sources,
            bounded=since is not None,
        )
    except (json.JSONDecodeError, ValueError) as err:
        failed = FleetRemoteResult(
            ok=False,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            error=f"invalid stats skills payload: {err}",
        )
        return failed, None
    return result, payload


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
