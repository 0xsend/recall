"""Fleet fan-out for list, search, and show (REQ-FLEET-CMD-002/003, MERGE-004/005)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from recall.core.fleet import FleetHost
from recall.core.types import is_attributed_host
from recall.services.fleet_transport import (
    DEFAULT_CONCURRENCY,
    DEFAULT_PER_HOST_TIMEOUT_SECONDS,
    FleetRemoteResult,
    run_remote,
)

logger = logging.getLogger("recall.fleet")


@dataclass(frozen=True)
class FleetQueryResult:
    rows: list[dict[str, Any]]
    errors: list[dict[str, str]] = field(default_factory=list)
    hosts_ok: int = 0
    hosts_failed: int = 0


@dataclass(frozen=True)
class FleetLiveResult(FleetQueryResult):
    coverage: dict[str, Any] = field(default_factory=dict)


def _fetch_live_page(
    host: FleetHost, argv: Sequence[str], *, timeout: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Accept current envelopes and label old array-only hosts as unknown coverage."""
    result = run_remote(host, argv, timeout=timeout)
    if not result.ok:
        raise ValueError(result.error or "remote failed")
    data = parse_json_payload(result.stdout)
    if isinstance(data, list):
        rows = data
        metadata = {"schema_version": 1, "watching": None, "coverage": None, "next_cursor": None}
    elif isinstance(data, dict) and data.get("schema_version") == 2:
        rows = data.get("sessions")
        coverage = data.get("coverage")
        cursor = data.get("next_cursor")
        if not isinstance(coverage, dict) or not isinstance(coverage.get("complete"), bool):
            raise ValueError("invalid live coverage")
        if not isinstance(data.get("watching"), bool) or not (
            cursor is None or isinstance(cursor, str)
        ):
            raise ValueError("invalid live observation or continuation")
        metadata = {
            key: data.get(key) for key in ("schema_version", "watching", "coverage", "next_cursor")
        }
    else:
        raise ValueError("unsupported live output; expected version 2 envelope or legacy array")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("invalid live sessions")
    if len(rows) > 256:
        raise ValueError("live page exceeds 256 rows")
    return [stamp_host(row, host.name) for row in rows], {"name": host.name, **metadata}


def parse_json_payload(stdout: str) -> list[dict[str, Any]] | dict[str, Any]:
    """Parse recall --json stdout (bare value or CTA envelope)."""
    text = stdout.strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict) and "data" in data:
        return data["data"]  # type: ignore[return-value]
    return data  # type: ignore[return-value]


def parse_json_array(stdout: str) -> list[dict[str, Any]]:
    data = parse_json_payload(stdout)
    if data is None:
        return []
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    raise ValueError(f"expected JSON array, got {type(data).__name__}")


def parse_json_object(stdout: str) -> dict[str, Any]:
    data = parse_json_payload(stdout)
    if isinstance(data, dict):
        return data
    raise ValueError(f"expected JSON object, got {type(data).__name__}")


def stamp_host(row: dict[str, Any], inventory_name: str) -> dict[str, Any]:
    """Attribute a remote row to its SSH hop (REQ-FLEET-MERGE-002).

    A remote's own `local` label is the unattributed sentinel in *that* host's
    frame of reference and says nothing to the control host, so it loses to the
    inventory name just as a missing label does. Only a real hostname the remote
    reports (`WorkstationOne`) is preserved.
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


def _fetch_json_array(
    host: FleetHost,
    remote_argv: Sequence[str],
    *,
    timeout: float,
) -> tuple[FleetHost, FleetRemoteResult, list[dict[str, Any]]]:
    result = run_remote(host, remote_argv, timeout=timeout)
    if not result.ok:
        return host, result, []
    try:
        rows = parse_json_array(result.stdout)
    except (json.JSONDecodeError, ValueError) as err:
        failed = FleetRemoteResult(
            ok=False,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            error=f"invalid JSON: {err}",
        )
        return host, failed, []
    return host, result, [stamp_host(row, host.name) for row in rows]


def _fan_out_arrays(
    hosts: Sequence[FleetHost],
    remote_argv_for: Callable[[FleetHost], Sequence[str]],
    *,
    concurrency: int,
    timeout: float,
) -> FleetQueryResult:
    if not hosts:
        return FleetQueryResult(rows=[], errors=[], hosts_ok=0, hosts_failed=0)

    workers = max(1, min(concurrency, len(hosts)))
    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    hosts_ok = 0
    hosts_failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _fetch_json_array,
                host,
                remote_argv_for(host),
                timeout=timeout,
            ): host
            for host in hosts
        }
        for future in as_completed(futures):
            host = futures[future]
            try:
                _host, result, rows = future.result()
            except Exception as err:
                logger.exception("fleet query crashed for %s", host.name)
                hosts_failed += 1
                errors.append({"name": host.name, "error": str(err)})
                continue
            if not result.ok:
                hosts_failed += 1
                errors.append({"name": host.name, "error": result.error or "remote failed"})
                continue
            hosts_ok += 1
            all_rows.extend(rows)

    errors.sort(key=lambda e: e.get("name") or "")
    return FleetQueryResult(
        rows=all_rows,
        errors=errors,
        hosts_ok=hosts_ok,
        hosts_failed=hosts_failed,
    )


def fleet_list(
    hosts: Iterable[FleetHost],
    *,
    since: str | None = None,
    source: str | None = None,
    project: str | None = None,
    host_filter: str | None = None,
    limit: int = 50,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetQueryResult:
    """Fan-out list; per-host limit then global re-limit (MERGE-004)."""
    host_list = list(hosts)
    if host_filter:
        host_list = [h for h in host_list if h.name == host_filter]

    def remote_argv(_host: FleetHost) -> list[str]:
        # Inventory --host only selects which machines to SSH to. Do not pass
        # --host to the remote CLI: edges may still be on builds without that
        # flag (pre-fleet), and local host labels (e.g. WorkstationOne) may not
        # match the inventory name.
        argv = ["recall", "list", "--json", "--limit", str(limit)]
        if since:
            argv.extend(["--since", since])
        if source:
            argv.extend(["--source", source])
        if project:
            argv.extend(["--project", project])
        return argv

    raw = _fan_out_arrays(
        host_list,
        remote_argv,
        concurrency=concurrency,
        timeout=timeout,
    )
    return FleetQueryResult(
        rows=merge_list_rows(raw.rows, limit=limit),
        errors=raw.errors,
        hosts_ok=raw.hosts_ok,
        hosts_failed=raw.hosts_failed,
    )


def fleet_live(
    hosts: Iterable[FleetHost],
    *,
    include_idle: bool = False,
    source: str | None = None,
    project: str | None = None,
    host_filter: str | None = None,
    limit: int = 50,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetLiveResult:
    """Merge bounded remote pages while preserving each host's coverage and cursor."""
    if not 1 <= limit <= 256:
        raise ValueError("live limit must be 1..256")
    host_list = list(hosts)
    if host_filter:
        host_list = [h for h in host_list if h.name == host_filter]

    def remote_argv(_host: FleetHost) -> list[str]:
        # As in `fleet_list`: inventory --host only selects which machines to
        # SSH to, and a remote's own host label need not match the inventory
        # name, so --host is never forwarded.
        argv = ["recall", "live", "--json", "--limit", str(limit)]
        if include_idle:
            argv.append("--all")
        if source:
            argv.extend(["--source", source])
        if project:
            argv.extend(["--project", project])
        return argv

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(host_list)))) as pool:
        futures = {
            pool.submit(_fetch_live_page, host, remote_argv(host), timeout=timeout): host
            for host in host_list
        }
        for future in as_completed(futures):
            host = futures[future]
            try:
                remote_rows, report = future.result()
            except Exception as error:
                errors.append({"name": host.name, "error": str(error)})
                reports.append({"name": host.name, "coverage": None, "error": str(error)})
            else:
                rows.extend(remote_rows)
                reports.append(report)
    reports.sort(key=lambda report: report["name"])
    errors.sort(key=lambda error: error["name"])
    merged = merge_live_rows(rows, limit=limit)
    truncated = len(merged) < len(rows)
    complete = (
        bool(host_list)
        and not errors
        and not truncated
        and all(
            report.get("coverage") is not None
            and report["coverage"].get("complete") is True
            and report.get("next_cursor") is None
            for report in reports
        )
    )
    return FleetLiveResult(
        rows=merged,
        errors=errors,
        hosts_ok=len(host_list) - len(errors),
        hosts_failed=len(errors),
        coverage={
            "complete": complete,
            "hosts": reports,
            "received": len(rows),
            "returned": len(merged),
            "truncated": truncated,
            "hosts_ok": len(host_list) - len(errors),
            "hosts_failed": len(errors),
        },
    )


def fleet_search(
    hosts: Iterable[FleetHost],
    *,
    query: str,
    tool: str | None = None,
    source: str | None = None,
    mode: str | None = None,
    limit: int = 20,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetQueryResult:
    """Fan-out search; merge by score (MERGE-005)."""
    host_list = list(hosts)

    def remote_argv(_host: FleetHost) -> list[str]:
        argv = ["recall", "search", query, "--json", "--limit", str(limit)]
        if tool:
            argv.extend(["--tool", tool])
        if source:
            argv.extend(["--source", source])
        if mode:
            argv.extend(["--mode", mode])
        return argv

    raw = _fan_out_arrays(
        host_list,
        remote_argv,
        concurrency=concurrency,
        timeout=timeout,
    )
    return FleetQueryResult(
        rows=merge_search_rows(raw.rows, limit=limit),
        errors=raw.errors,
        hosts_ok=raw.hosts_ok,
        hosts_failed=raw.hosts_failed,
    )


@dataclass(frozen=True)
class FleetShowResult:
    session: dict[str, Any] | None
    errors: list[dict[str, str]] = field(default_factory=list)
    hosts_tried: int = 0
    ambiguous_hosts: tuple[str, ...] = ()


def fleet_show(
    hosts: Iterable[FleetHost],
    *,
    session_id: str,
    host_name: str | None = None,
    tools: bool = False,
    message_limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetShowResult:
    """Show a session across the fleet (CMD-003).

    If ``host_name`` is set, only that inventory host is queried.
    Otherwise all hosts are probed; a single hit wins; multiple hits → ambiguous.
    """
    host_list = list(hosts)
    if host_name:
        host_list = [h for h in host_list if h.name == host_name]
        if not host_list:
            return FleetShowResult(
                session=None,
                errors=[{"name": host_name, "error": "not in fleet inventory"}],
                hosts_tried=0,
            )

    def remote_argv(_host: FleetHost) -> list[str]:
        argv = ["recall", "show", session_id, "--json"]
        if tools:
            argv.append("--tools")
        if message_limit is not None:
            argv.extend(["--message-limit", str(message_limit)])
        return argv

    workers = max(1, min(concurrency, len(host_list))) if host_list else 1
    found: list[tuple[str, dict[str, Any]]] = []
    errors: list[dict[str, str]] = []
    tried = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_remote, host, remote_argv(host), timeout=timeout): host
            for host in host_list
        }
        for future in as_completed(futures):
            host = futures[future]
            tried += 1
            try:
                result = future.result()
            except Exception as err:
                errors.append({"name": host.name, "error": str(err)})
                continue
            if not result.ok:
                # Not found is expected when probing; keep soft unless sole target.
                err = result.error or "remote failed"
                if host_name or "not found" not in err.lower():
                    errors.append({"name": host.name, "error": err})
                continue
            try:
                obj = parse_json_object(result.stdout)
            except (json.JSONDecodeError, ValueError) as err:
                errors.append({"name": host.name, "error": f"invalid JSON: {err}"})
                continue
            found.append((host.name, stamp_host(obj, host.name)))

    errors.sort(key=lambda e: e.get("name") or "")
    if len(found) == 1:
        return FleetShowResult(session=found[0][1], errors=errors, hosts_tried=tried)
    if len(found) > 1:
        names = tuple(sorted(name for name, _ in found))
        return FleetShowResult(
            session=None,
            errors=errors,
            hosts_tried=tried,
            ambiguous_hosts=names,
        )
    return FleetShowResult(session=None, errors=errors, hosts_tried=tried)
