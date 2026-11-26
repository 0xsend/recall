"""Fleet query decoding and merge rules over resolved endpoints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from recall.core.host_text import HostValue
from recall.core.rpc_params import (
    ListParams,
    LiveParams,
    SearchFilteredParams,
    SearchParams,
    ShowParams,
)
from recall.core.search_filters import has_filtered_search_params
from recall.core.time import format_utc_iso, to_utc
from recall.core.types import Role, is_attributed_host
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
    # Canonical answering endpoint per returned row (parallel to `rows`),
    # distinct from the preserved stored origin in `row["host"]`.
    endpoints: tuple[HostValue, ...] = field(default=(), kw_only=True)


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


def _parse_fleet_timestamp(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return to_utc(raw)
    if isinstance(raw, str) and raw:
        try:
            return to_utc(datetime.fromisoformat(raw.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _normalize_timestamp_value(value: Any) -> Any:
    parsed = _parse_fleet_timestamp(value)
    return format_utc_iso(parsed) if parsed is not None else value


_TOP_LEVEL_TIMESTAMP_KEYS = (
    "started_at",
    "ended_at",
    "indexed_at",
    "last_activity_at",
    "timestamp",
)


def _normalize_fleet_timestamps(row: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize typed timestamp fields to RFC3339 UTC (`...Z`) at the fleet reader boundary."""
    updated: dict[str, Any] | None = None
    for key in _TOP_LEVEL_TIMESTAMP_KEYS:
        if key in row and row[key] is not None:
            normalized = _normalize_timestamp_value(row[key])
            if normalized != row[key]:
                if updated is None:
                    updated = dict(row)
                updated[key] = normalized

    turn = row.get("turn")
    if isinstance(turn, dict):
        turn_updated: dict[str, Any] | None = None
        for key in ("last_user_at", "last_assistant_at"):
            if key in turn and turn[key] is not None:
                normalized = _normalize_timestamp_value(turn[key])
                if normalized != turn[key]:
                    if turn_updated is None:
                        turn_updated = dict(turn)
                    turn_updated[key] = normalized
        running_tool = turn.get("running_tool")
        if isinstance(running_tool, dict) and running_tool.get("started_at") is not None:
            normalized = _normalize_timestamp_value(running_tool["started_at"])
            if normalized != running_tool["started_at"]:
                if turn_updated is None:
                    turn_updated = dict(turn)
                turn_updated["running_tool"] = {**running_tool, "started_at": normalized}
        if turn_updated is not None:
            if updated is None:
                updated = dict(row)
            updated["turn"] = turn_updated

    messages = row.get("messages")
    if isinstance(messages, list):
        messages_changed = False
        normalized_messages: list[Any] = []
        for msg in messages:
            if not isinstance(msg, dict):
                normalized_messages.append(msg)
                continue
            msg_updated: dict[str, Any] | None = None
            if msg.get("timestamp") is not None:
                normalized = _normalize_timestamp_value(msg["timestamp"])
                if normalized != msg["timestamp"]:
                    msg_updated = dict(msg)
                    msg_updated["timestamp"] = normalized
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                tc_changed = False
                normalized_tcs: list[Any] = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        normalized_tcs.append(tc)
                        continue
                    tc_updated: dict[str, Any] | None = None
                    for key in ("completed_at", "result_completed_at"):
                        if tc.get(key) is not None:
                            normalized = _normalize_timestamp_value(tc[key])
                            if normalized != tc[key]:
                                if tc_updated is None:
                                    tc_updated = dict(tc)
                                tc_updated[key] = normalized
                    if tc_updated is not None:
                        tc_changed = True
                        normalized_tcs.append(tc_updated)
                    else:
                        normalized_tcs.append(tc)
                if tc_changed:
                    if msg_updated is None:
                        msg_updated = dict(msg)
                    msg_updated["tool_calls"] = normalized_tcs
            if msg_updated is not None:
                messages_changed = True
                normalized_messages.append(msg_updated)
            else:
                normalized_messages.append(msg)
        if messages_changed:
            if updated is None:
                updated = dict(row)
            updated["messages"] = normalized_messages

    return updated if updated is not None else row


def _recency_key(row: dict[str, Any]) -> datetime:
    for key in ("ended_at", "started_at", "indexed_at"):
        parsed = _parse_fleet_timestamp(row.get(key))
        if parsed is not None:
            return parsed
    return datetime.min.replace(tzinfo=UTC)


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

    Activity is compared as seconds since the epoch in UTC rather than as a
    mixed naive/aware `datetime`: rows arrive from peers across versions and
    timezones. A row with no stamp at all -- a transcript the daemon promoted
    but has not indexed yet -- is still live, so it sorts last within its rank
    instead of being dropped.
    """
    liveness = str(row.get("liveness") or "unknown")
    rank = _LIVENESS_ORDER.get(liveness, _LIVENESS_LAST)
    activity = _parse_fleet_timestamp(row.get("last_activity_at"))
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
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array, got {type(value).__name__}")
    if any(not isinstance(row, dict) for row in value):
        raise ValueError("expected JSON array of objects")
    return [decode_object(row) for row in value]


def decode_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object, got {type(value).__name__}")
    normalized = _normalize_fleet_timestamps(value)
    host = normalized.get("host")
    return {**normalized, "host": HostValue(host)} if isinstance(host, str) else normalized


def _arrays(
    executor: FleetExecutor, operation: FleetOperation, concurrency: int
) -> FleetQueryResult:
    rows: list[dict[str, Any]] = []
    endpoints: list[HostValue] = []
    errors: list[dict[str, str]] = []
    successful = 0
    for outcome in executor.execute(operation, concurrency=concurrency):
        if isinstance(outcome, EndpointSuccess):
            successful += 1
            for row in outcome.value:
                rows.append(stamp_host(row, outcome.name))
                endpoints.append(outcome.name)
        else:
            errors.append({"name": outcome.name, "error": outcome.error})
    return FleetQueryResult(rows, errors, successful, len(errors), endpoints=tuple(endpoints))


def _merge_with_endpoints(
    rows: Sequence[dict[str, Any]],
    endpoints: Sequence[HostValue],
    *,
    key: Any,
    limit: int,
) -> tuple[list[dict[str, Any]], tuple[HostValue, ...]]:
    paired = list(zip(rows, endpoints, strict=True))
    ordered = sorted(paired, key=lambda item: key(item[0]), reverse=True)
    if limit > 0:
        ordered = ordered[:limit]
    return [row for row, _ in ordered], tuple(ep for _, ep in ordered)


def fleet_list(
    executor: FleetExecutor,
    *,
    since: str | None = None,
    source: str | None = None,
    project: str | None = None,
    limit: int = 50,
    subagents: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetQueryResult:
    argv = ["recall", "list", "--json", "--limit", str(limit)]
    for flag, value in (("--since", since), ("--source", source), ("--project", project)):
        if value:
            argv.extend([flag, value])
    if subagents:
        argv.append("--subagents")
    raw = _arrays(
        executor,
        FleetOperation(
            "recall.list",
            ListParams(
                since=since, source=source, project=project, limit=limit, subagents=subagents
            ),
            argv,
            decode_array,
            local_seconds=timeout,
            remote_seconds=timeout,
        ),
        concurrency,
    )
    merged_rows, merged_endpoints = _merge_with_endpoints(
        raw.rows, raw.endpoints, key=_recency_key, limit=limit
    )
    return FleetQueryResult(
        merged_rows, raw.errors, raw.hosts_ok, raw.hosts_failed, endpoints=merged_endpoints
    )


def _parse_finite_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("search row score must be a finite number")
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError) as err:
        raise ValueError("search row score must be a finite number") from err
    if not math.isfinite(score):
        raise ValueError("search row score must be finite")
    return score


def _decode_filtered_search_array(
    value: Any,
    *,
    expected_roles: frozenset[str] | None,
    per_session: int | None,
) -> list[dict[str, Any]]:
    """Validate filtered search rows fail-closed before fleet merging."""
    rows = decode_array(value)
    validated: list[dict[str, Any]] = []
    session_counts: dict[str, int] = {}
    for row in rows:
        kind = row.get("kind")
        if not isinstance(kind, str) or kind not in {"message", "tool_call"}:
            raise ValueError("filtered search row missing valid kind")
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("filtered search row missing valid session_id")
        score = _parse_finite_score(row.get("score"))
        role = row.get("role")
        if expected_roles is not None and (
            kind != "message" or not isinstance(role, str) or role not in expected_roles
        ):
            raise ValueError("filtered search reply violated --role filter")
        if per_session is not None:
            seen = session_counts.get(session_id, 0) + 1
            if seen > per_session:
                raise ValueError("filtered search reply exceeded --per-session cap")
            session_counts[session_id] = seen
        validated.append({**row, "score": score})
    return validated


def _safe_search_score(row: dict[str, Any]) -> float:
    raw = row.get("score")
    if isinstance(raw, bool):
        return 0.0
    try:
        score = float(raw or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _merge_search_with_endpoints(
    rows: Sequence[dict[str, Any]],
    endpoints: Sequence[HostValue],
    *,
    limit: int,
    per_session: int | None = None,
) -> tuple[list[dict[str, Any]], tuple[HostValue, ...]]:
    paired = list(zip(rows, endpoints, strict=True))
    ordered = sorted(paired, key=lambda item: _safe_search_score(item[0]), reverse=True)
    if per_session is None:
        if limit > 0:
            ordered = ordered[:limit]
        return [row for row, _ in ordered], tuple(ep for _, ep in ordered)
    kept_rows: list[dict[str, Any]] = []
    kept_endpoints: list[HostValue] = []
    counts: dict[tuple[str, str], int] = {}
    for row, endpoint in ordered:
        session_key = (endpoint.raw(), str(row.get("session_id", "")))
        seen = counts.get(session_key, 0)
        if seen >= per_session:
            continue
        counts[session_key] = seen + 1
        kept_rows.append(row)
        kept_endpoints.append(endpoint)
        if limit > 0 and len(kept_rows) >= limit:
            break
    return kept_rows, tuple(kept_endpoints)


def fleet_search(
    executor: FleetExecutor,
    *,
    query: str,
    tool: str | None = None,
    source: str | None = None,
    mode: str | None = None,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: str | None = None,
    per_session: int | None = None,
    limit: int = 20,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetQueryResult:
    argv = ["recall", "search", query, "--json", "--limit", str(limit)]
    for flag, value in (
        ("--tool", tool),
        ("--source", source),
        ("--mode", mode),
        ("--project", project),
        ("--since", since),
    ):
        if value:
            argv.extend([flag, value])
    if roles:
        for role in roles:
            argv.extend(["--role", role.value])
    if per_session is not None:
        argv.extend(["--per-session", str(per_session)])

    use_filtered = has_filtered_search_params(
        roles=roles,
        project=project,
        since=since,
        per_session=per_session,
        fleet=True,
    )
    if use_filtered:
        expected_roles = frozenset(role.value for role in roles) if roles else None
        operation = FleetOperation(
            "recall.search_filtered",
            SearchFilteredParams(
                query=query,
                tool=tool,
                source=source,
                mode=mode,
                limit=limit,
                role=[role.value for role in roles] if roles else None,
                project=project,
                since=since,
                per_session=per_session,
            ),
            argv,
            lambda value: _decode_filtered_search_array(
                value,
                expected_roles=expected_roles,
                per_session=per_session,
            ),
            local_seconds=timeout,
            remote_seconds=timeout,
        )
    else:
        operation = FleetOperation(
            "recall.search",
            SearchParams(query=query, tool=tool, source=source, mode=mode, limit=limit),
            argv,
            decode_array,
            local_seconds=timeout,
            remote_seconds=timeout,
        )

    raw = _arrays(executor, operation, concurrency)
    merged_rows, merged_endpoints = _merge_search_with_endpoints(
        raw.rows, raw.endpoints, limit=limit, per_session=per_session
    )
    return FleetQueryResult(
        merged_rows, raw.errors, raw.hosts_ok, raw.hosts_failed, endpoints=merged_endpoints
    )


def _decode_live_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("invalid live sessions")
    return decode_array(value)


def _decode_live(value: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(value, list):
        rows = _decode_live_rows(value)
        metadata = {"schema_version": 1, "watching": None, "coverage": None, "next_cursor": None}
    elif isinstance(value, dict) and value.get("schema_version") == 2:
        rows = _decode_live_rows(value.get("sessions"))
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
        invalid_reply_prefix="",
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
    endpoint: HostValue | None = None


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
            errors=[{"name": executor.scope.selector, "error": "not in fleet inventory"}],
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
    found: list[tuple[HostValue, dict[str, Any]]] = []
    errors: list[dict[str, str]] = []
    for outcome in outcomes:
        if isinstance(outcome, EndpointSuccess):
            found.append((outcome.name, stamp_host(outcome.value, outcome.name)))
        elif isinstance(outcome, EndpointFailure) or (
            isinstance(outcome, EndpointNotFound) and executor.scope.selector is not None
        ):
            errors.append({"name": outcome.name, "error": outcome.error})
    if len(found) == 1:
        return FleetShowResult(found[0][1], errors, len(outcomes), endpoint=found[0][0])
    return FleetShowResult(None, errors, len(outcomes), tuple(sorted(name for name, _ in found)))
