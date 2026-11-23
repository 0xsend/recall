"""Fleet fan-out for stats usage (REQ-FLEET-CMD-002 / MERGE-002/003)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from recall.core.host_text import HostValue
from recall.core.rpc_params import Params, StatsUsageFilteredParams, UsageStatsParams
from recall.core.stats_filters import normalize_stats_filters
from recall.core.types import UNATTRIBUTED_HOST, is_attributed_host
from recall.services.analytics import DEFAULT_USAGE_STATS_LIMIT
from recall.services.fleet_endpoints import FleetExecutor, FleetOperation
from recall.services.fleet_query import _arrays, decode_array
from recall.services.fleet_transport import (
    DEFAULT_CONCURRENCY,
    DEFAULT_PER_HOST_TIMEOUT_SECONDS,
)

_USAGE_NUMERIC_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "session_count",
)


@dataclass(frozen=True)
class FleetUsageResult:
    rows: list[dict[str, Any]]
    errors: list[dict[str, str]] = field(default_factory=list)
    hosts_ok: int = 0
    hosts_failed: int = 0


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def merge_usage_rows(
    rows: Sequence[dict[str, Any]],
    *,
    default_host: str | None = None,
) -> list[dict[str, Any]]:
    """Merge usage rows by (source, model, host); sum token counts (MERGE-003)."""
    buckets: dict[tuple[str, str | None, str], dict[str, Any]] = {}
    fresh_sum: dict[tuple[str, str | None, str], int] = {}
    fresh_any: dict[tuple[str, str | None, str], bool] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source") or "unknown")
        model_raw = row.get("model")
        model = str(model_raw) if model_raw is not None and str(model_raw) else None
        host_raw = row.get("host")
        if is_attributed_host(host_raw):
            host = HostValue(
                host_raw.raw().strip() if isinstance(host_raw, HostValue) else str(host_raw).strip()
            )
        elif default_host:
            host = default_host
        else:
            host = UNATTRIBUTED_HOST
        key = (source, model, host)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = {
                "source": source,
                "model": model,
                "host": host,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "fresh_input_tokens": None,
                "session_count": 0,
            }
            buckets[key] = bucket
            fresh_sum[key] = 0
            fresh_any[key] = False
        for field_name in _USAGE_NUMERIC_KEYS:
            bucket[field_name] = _as_int(bucket.get(field_name)) + _as_int(row.get(field_name))
        fresh = _as_optional_int(row.get("fresh_input_tokens"))
        if fresh is not None:
            fresh_sum[key] += fresh
            fresh_any[key] = True

    merged: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        if fresh_any[key]:
            bucket["fresh_input_tokens"] = fresh_sum[key]
        else:
            bucket["fresh_input_tokens"] = None
        merged.append(bucket)

    merged.sort(
        key=lambda r: (
            str(r.get("source") or ""),
            str(r.get("model") or ""),
            str.__str__(r.get("host") or ""),
        )
    )
    return merged


def _require_nonneg_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_optional_nonneg_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_nonneg_int(value, field_name)


def _decode_usage(
    value: Any,
    *,
    expected_source: str | None = None,
    max_rows: int = DEFAULT_USAGE_STATS_LIMIT,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"unexpected stats usage payload type: {type(value).__name__}")
    if len(value) > max_rows:
        raise ValueError(f"stats usage reply exceeds per-host limit of {max_rows} rows")
    rows = decode_array(value)
    validated: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str | None, str | None]] = set()
    for index, row in enumerate(rows):
        source = row.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"rows[{index}].source must be a non-empty string")
        if expected_source is not None and source != expected_source:
            raise ValueError(
                f"rows[{index}].source {source!r} violated --source {expected_source!r} filter"
            )
        model = row.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError(f"rows[{index}].model must be a string or null")
        host = row.get("host")
        if host is not None and not isinstance(host, (str, HostValue)):
            raise ValueError(f"rows[{index}].host must be a string or null")
        host_key = (
            host.raw() if isinstance(host, HostValue) else (host if isinstance(host, str) else None)
        )
        group_key = (source, model, host_key)
        if group_key in seen_keys:
            raise ValueError(f"duplicate usage group key {group_key!r} at rows[{index}]")
        seen_keys.add(group_key)
        input_tokens = _require_nonneg_int(row.get("input_tokens"), f"rows[{index}].input_tokens")
        output_tokens = _require_nonneg_int(
            row.get("output_tokens"), f"rows[{index}].output_tokens"
        )
        session_count = _require_nonneg_int(
            row.get("session_count"), f"rows[{index}].session_count"
        )
        cached_input_tokens = _require_nonneg_int(
            row.get("cached_input_tokens"), f"rows[{index}].cached_input_tokens"
        )
        fresh_input_tokens = _require_optional_nonneg_int(
            row.get("fresh_input_tokens"), f"rows[{index}].fresh_input_tokens"
        )
        validated.append(
            {
                "source": source,
                "model": model,
                "host": host,
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "output_tokens": output_tokens,
                "fresh_input_tokens": fresh_input_tokens,
                "session_count": session_count,
            }
        )
    return validated


def fleet_stats_usage(
    executor: FleetExecutor,
    *,
    source: str | None = None,
    since: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetUsageResult:
    filters = normalize_stats_filters(source=source, since=since)
    source_value = filters.source.value if filters.source is not None else None
    argv = ["recall", "stats", "usage", "--json"]
    if source_value is not None:
        argv.extend(["--source", source_value])
    if filters.since_iso is not None:
        argv.extend(["--since", filters.since_iso])
    if source_value is not None:
        method = "recall.stats_usage_filtered"
        params: Params = StatsUsageFilteredParams(source=source_value, since=filters.since_iso)
    else:
        method = "recall.stats_usage"
        params = UsageStatsParams(since=filters.since_iso)
    result = _arrays(
        executor,
        FleetOperation(
            method,
            params,
            argv,
            lambda value: _decode_usage(value, expected_source=source_value),
            local_seconds=timeout,
            remote_seconds=timeout,
            invalid_reply_prefix="invalid stats usage JSON: ",
            allow_blank_reply=False,
        ),
        concurrency,
    )
    return FleetUsageResult(
        merge_usage_rows(result.rows), result.errors, result.hosts_ok, result.hosts_failed
    )
