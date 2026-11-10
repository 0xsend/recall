"""Fleet fan-out for stats usage (REQ-FLEET-CMD-002 / MERGE-002/003)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from recall.core.rpc_params import UsageStatsParams
from recall.core.types import UNATTRIBUTED_HOST, is_attributed_host
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
            host = str(host_raw).strip()
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
            str(r.get("host") or ""),
        )
    )
    return merged


def fleet_stats_usage(
    executor: FleetExecutor,
    *,
    since: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetUsageResult:
    argv = ["recall", "stats", "usage", "--json"]
    if since:
        argv.extend(["--since", since])
    result = _arrays(
        executor,
        FleetOperation(
            "recall.stats_usage",
            UsageStatsParams(since=since),
            argv,
            decode_array,
            local_seconds=timeout,
            remote_seconds=timeout,
        ),
        concurrency,
    )
    return FleetUsageResult(
        merge_usage_rows(result.rows), result.errors, result.hosts_ok, result.hosts_failed
    )
