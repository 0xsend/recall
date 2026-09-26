"""Fleet fan-out for stats usage (REQ-FLEET-CMD-002 / MERGE-002/003)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from recall.core.fleet import FleetHost
from recall.core.types import UNATTRIBUTED_HOST, is_attributed_host
from recall.services.fleet_query import stamp_host
from recall.services.fleet_transport import (
    DEFAULT_CONCURRENCY,
    DEFAULT_PER_HOST_TIMEOUT_SECONDS,
    FleetRemoteResult,
    run_remote,
)

logger = logging.getLogger("recall.fleet")

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


def _parse_usage_json(stdout: str) -> list[dict[str, Any]]:
    text = stdout.strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        # CTA envelope
        return [row for row in data["data"] if isinstance(row, dict)]
    raise ValueError(f"unexpected stats usage payload type: {type(data).__name__}")


def _fetch_usage_for_host(
    host: FleetHost,
    *,
    since: str | None,
    timeout: float,
) -> tuple[FleetHost, FleetRemoteResult, list[dict[str, Any]]]:
    remote_argv = ["recall", "stats", "usage", "--json"]
    if since:
        remote_argv.extend(["--since", since])
    result = run_remote(host, remote_argv, timeout=timeout)
    if not result.ok:
        return host, result, []
    try:
        rows = _parse_usage_json(result.stdout)
    except (json.JSONDecodeError, ValueError) as err:
        failed = FleetRemoteResult(
            ok=False,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            error=f"invalid stats usage JSON: {err}",
        )
        return host, failed, []
    # Stamp inventory name when the remote omitted host or reported the
    # unattributed sentinel (MERGE-002).
    return host, result, [stamp_host(row, host.name) for row in rows]


def fleet_stats_usage(
    hosts: Iterable[FleetHost],
    *,
    since: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetUsageResult:
    """Fan-out `recall stats usage --json` and merge (REQ-FLEET-CMD-002)."""
    host_list = list(hosts)
    if not host_list:
        return FleetUsageResult(rows=[], errors=[], hosts_ok=0, hosts_failed=0)

    workers = max(1, min(concurrency, len(host_list)))
    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    hosts_ok = 0
    hosts_failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_usage_for_host, host, since=since, timeout=timeout): host
            for host in host_list
        }
        for future in as_completed(futures):
            host = futures[future]
            try:
                _host, result, rows = future.result()
            except Exception as err:
                logger.exception("fleet stats usage crashed for %s", host.name)
                hosts_failed += 1
                errors.append({"name": host.name, "error": str(err)})
                continue
            if not result.ok:
                hosts_failed += 1
                errors.append({"name": host.name, "error": result.error or "remote failed"})
                continue
            hosts_ok += 1
            all_rows.extend(rows)

    # Preserve inventory name order in errors already insertion-unordered; sort.
    errors.sort(key=lambda e: e.get("name") or "")
    return FleetUsageResult(
        rows=merge_usage_rows(all_rows),
        errors=errors,
        hosts_ok=hosts_ok,
        hosts_failed=hosts_failed,
    )
