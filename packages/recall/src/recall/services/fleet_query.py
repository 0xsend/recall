"""Fleet query decoding and merge rules over resolved endpoints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from recall.core.rpc_params import ListParams, LiveParams, SearchParams, ShowParams
from recall.core.types import is_attributed_host
from recall.services.fleet_endpoints import (
    EndpointFailure,
    EndpointNotFound,
    EndpointSuccess,
    FleetExecutor,
    FleetOperation,
)
from recall.services.fleet_transport import DEFAULT_CONCURRENCY, DEFAULT_PER_HOST_TIMEOUT_SECONDS


@dataclass(frozen=True)
class FleetQueryResult:
    rows: list[dict[str, Any]]
    errors: list[dict[str, str]] = field(default_factory=list)
    hosts_ok: int = 0
    hosts_failed: int = 0


@dataclass(frozen=True)
class FleetLiveResult(FleetQueryResult):
    coverage: dict[str, Any] = field(default_factory=dict)


def stamp_host(row: dict[str, Any], inventory_name: str) -> dict[str, Any]:
    """Preserve genuine stored origins, otherwise use the queried endpoint label.

    The `local` sentinel means unattributed in that database's frame of reference;
    it cannot identify a machine in a merged result.
    """
    if is_attributed_host(row.get("host")):
        return row
    return {**row, "host": inventory_name}


def _recency_key(row: dict[str, Any]) -> datetime:
    for key in ("ended_at", "started_at", "indexed_at"):
        raw = row.get(key)
        if not raw:
            continue
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
    return datetime.min


def merge_list_rows(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Sort by recency desc and apply global limit (MERGE-004)."""
    ordered = sorted(rows, key=_recency_key, reverse=True)
    if limit <= 0:
        return list(ordered)
    return ordered[:limit]


# A fleet monitor reads the top of the list to answer "what is running right
# now", so liveness outranks recency across hosts: an idle session on one
# machine must not push a working agent on another below the fold.
_LIVENESS_ORDER = {"active": 0, "idle": 1, "unknown": 2, "ended": 3}
_LIVENESS_LAST = len(_LIVENESS_ORDER)


def _live_sort_key(row: dict[str, Any]) -> tuple[int, float]:
    """Rank by liveness, then most recent activity first.

    Activity is compared as seconds since the epoch rather than as a
    `datetime`: rows arrive from several hosts and one may stamp an aware value
    while another stamps a naive one, which raises on comparison. A row with no
    stamp at all -- a transcript the daemon promoted but has not indexed yet --
    is still live, so it sorts last within its rank instead of being dropped.
    """
    liveness = str(row.get("liveness") or "unknown")
    rank = _LIVENESS_ORDER.get(liveness, _LIVENESS_LAST)
    raw = row.get("last_activity_at")
    if isinstance(raw, datetime):
        activity: datetime | None = raw
    elif isinstance(raw, str):
        try:
            activity = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            activity = None
    else:
        activity = None
    return rank, float("inf") if activity is None else -activity.timestamp()


def merge_live_rows(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Order by liveness, then most recent activity, then apply the global limit."""
    ordered = sorted(rows, key=_live_sort_key)
    if limit <= 0:
        return list(ordered)
    return ordered[:limit]


def merge_search_rows(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Sort by score desc and apply global limit (MERGE-005)."""

    def score_key(row: dict[str, Any]) -> float:
        try:
            return float(row.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    ordered = sorted(rows, key=score_key, reverse=True)
    if limit <= 0:
        return list(ordered)
    return ordered[:limit]


def decode_array(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("expected JSON array of objects")
    return value


def decode_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _arrays(
    executor: FleetExecutor, operation: FleetOperation, concurrency: int
) -> FleetQueryResult:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    successful = 0
    for outcome in executor.execute(operation, concurrency=concurrency):
        if isinstance(outcome, EndpointSuccess):
            successful += 1
            rows.extend(stamp_host(row, outcome.name) for row in outcome.value)
        else:
            errors.append({"name": outcome.name, "error": outcome.error})
    return FleetQueryResult(rows, errors, successful, len(errors))


def fleet_list(
    executor: FleetExecutor,
    *,
    since: str | None = None,
    source: str | None = None,
    project: str | None = None,
    limit: int = 50,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetQueryResult:
    argv = ["recall", "list", "--json", "--limit", str(limit)]
    for flag, value in (("--since", since), ("--source", source), ("--project", project)):
        if value:
            argv.extend([flag, value])
    raw = _arrays(
        executor,
        FleetOperation(
            "recall.list",
            ListParams(since=since, source=source, project=project, limit=limit),
            argv,
            decode_array,
            local_seconds=timeout,
            remote_seconds=timeout,
        ),
        concurrency,
    )
    return FleetQueryResult(
        merge_list_rows(raw.rows, limit=limit), raw.errors, raw.hosts_ok, raw.hosts_failed
    )


def fleet_search(
    executor: FleetExecutor,
    *,
    query: str,
    tool: str | None = None,
    source: str | None = None,
    mode: str | None = None,
    limit: int = 20,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetQueryResult:
    argv = ["recall", "search", query, "--json", "--limit", str(limit)]
    for flag, value in (("--tool", tool), ("--source", source), ("--mode", mode)):
        if value:
            argv.extend([flag, value])
    raw = _arrays(
        executor,
        FleetOperation(
            "recall.search",
            SearchParams(query=query, tool=tool, source=source, mode=mode, limit=limit),
            argv,
            decode_array,
            local_seconds=timeout,
            remote_seconds=timeout,
        ),
        concurrency,
    )
    return FleetQueryResult(
        merge_search_rows(raw.rows, limit=limit), raw.errors, raw.hosts_ok, raw.hosts_failed
    )


def _decode_live(value: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(value, list):
        rows = decode_array(value)
        metadata = {"schema_version": 1, "watching": None, "coverage": None, "next_cursor": None}
    elif isinstance(value, dict) and value.get("schema_version") == 2:
        rows = decode_array(value.get("sessions"))
        coverage, cursor = value.get("coverage"), value.get("next_cursor")
        if not isinstance(coverage, dict) or not isinstance(coverage.get("complete"), bool):
            raise ValueError("invalid live coverage")
        if not isinstance(value.get("watching"), bool) or not (
            cursor is None or isinstance(cursor, str)
        ):
            raise ValueError("invalid live observation or continuation")
        metadata = {
            key: value.get(key)
            for key in (
                "schema_version",
                "watching",
                "coverage",
                "next_cursor",
            )
        }
    else:
        raise ValueError("unsupported live output; expected version 2 envelope or legacy array")
    if len(rows) > 256:
        raise ValueError("live page exceeds 256 rows")
    return rows, metadata


def fleet_live(
    executor: FleetExecutor,
    *,
    include_idle: bool = False,
    source: str | None = None,
    project: str | None = None,
    limit: int = 50,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetLiveResult:
    if not 1 <= limit <= 256:
        raise ValueError("live limit must be 1..256")
    argv = ["recall", "live", "--json", "--limit", str(limit)]
    if include_idle:
        argv.append("--all")
    for flag, value in (("--source", source), ("--project", project)):
        if value:
            argv.extend([flag, value])
    operation = FleetOperation(
        "recall.live",
        LiveParams(all=include_idle, source=source, project=project, limit=limit),
        argv,
        _decode_live,
        local_seconds=2.0,
        remote_seconds=timeout,
    )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    reports: list[dict[str, Any]] = []
    outcomes = executor.execute(operation, concurrency=concurrency)
    for outcome in outcomes:
        if isinstance(outcome, EndpointSuccess):
            page, metadata = outcome.value
            rows.extend(stamp_host(row, outcome.name) for row in page)
            reports.append({"name": outcome.name, **metadata})
        else:
            errors.append({"name": outcome.name, "error": outcome.error})
            reports.append({"name": outcome.name, "coverage": None, "error": outcome.error})
    merged = merge_live_rows(rows, limit=limit)
    truncated = len(merged) < len(rows)
    complete = (
        bool(outcomes)
        and not errors
        and not truncated
        and all(
            report.get("coverage") is not None
            and report["coverage"].get("complete") is True
            and report.get("next_cursor") is None
            for report in reports
        )
    )
    successful = len(outcomes) - len(errors)
    return FleetLiveResult(
        merged,
        errors,
        successful,
        len(errors),
        {
            "complete": complete,
            "hosts": reports,
            "received": len(rows),
            "returned": len(merged),
            "truncated": truncated,
            "hosts_ok": successful,
            "hosts_failed": len(errors),
        },
    )


@dataclass(frozen=True)
class FleetShowResult:
    session: dict[str, Any] | None
    errors: list[dict[str, str]] = field(default_factory=list)
    hosts_tried: int = 0
    ambiguous_hosts: tuple[str, ...] = ()


def fleet_show(
    executor: FleetExecutor,
    *,
    session_id: str,
    tools: bool = False,
    message_limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetShowResult:
    if not executor.scope.endpoints and executor.scope.selector is not None:
        return FleetShowResult(
            session=None,
            errors=[{"name": executor.scope.selector, "error": "not in selected fleet"}],
        )
    argv = ["recall", "show", session_id, "--json"]
    if tools:
        argv.append("--tools")
    if message_limit is not None:
        argv.extend(["--message-limit", str(message_limit)])
    outcomes = executor.execute(
        FleetOperation(
            "recall.show",
            ShowParams(session_id=session_id, tools=tools, message_limit=message_limit),
            argv,
            decode_object,
            local_seconds=timeout,
            remote_seconds=timeout,
            not_found=True,
        ),
        concurrency=concurrency,
    )
    found: list[tuple[str, dict[str, Any]]] = []
    errors: list[dict[str, str]] = []
    for outcome in outcomes:
        if isinstance(outcome, EndpointSuccess):
            found.append((outcome.name, stamp_host(outcome.value, outcome.name)))
        elif isinstance(outcome, EndpointFailure) or (
            isinstance(outcome, EndpointNotFound) and executor.scope.selector is not None
        ):
            errors.append({"name": outcome.name, "error": outcome.error})
    if len(found) == 1:
        return FleetShowResult(found[0][1], errors, len(outcomes))
    return FleetShowResult(None, errors, len(outcomes), tuple(sorted(name for name, _ in found)))
