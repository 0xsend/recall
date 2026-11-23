"""Fail-closed fleet fan-out and additive reducers for stats commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from recall.core.rpc_params import (
    BashStatsParams,
    Params,
    StatsBashFilteredParams,
    StatsFilteredParams,
)
from recall.core.stats_filters import normalize_stats_filters
from recall.services.analytics import (
    DEFAULT_BASH_STATS_LIMIT,
    DEFAULT_BASH_SUGGESTIONS_LIMIT,
    DEFAULT_TOKEN_STATS_LIMIT,
    DEFAULT_TOOL_STATS_LIMIT,
    BashStat,
    PermissionSkipped,
    PermissionSuggestion,
)
from recall.services.fleet_endpoints import EndpointSuccess, FleetExecutor, FleetOperation
from recall.services.fleet_transport import (
    DEFAULT_CONCURRENCY,
    DEFAULT_PER_HOST_TIMEOUT_SECONDS,
)

_OVERVIEW_COUNTER_KEYS = ("sessions", "messages", "tool_calls", "bash_calls")
_EMPTY_LAST_RUN_FIELDS: dict[str, None] = {
    "last_index_total": None,
    "last_index_indexed": None,
    "last_index_skipped": None,
    "last_index_failed": None,
    "last_context_messages": None,
    "last_context_mode": None,
    "last_context_input_tokens": None,
    "last_context_output_tokens": None,
    "last_context_model": None,
}


@dataclass(frozen=True)
class FleetStatsOverviewResult:
    stats: dict[str, Any]
    errors: list[dict[str, str]] = field(default_factory=list)
    hosts_ok: int = 0
    hosts_failed: int = 0


@dataclass(frozen=True)
class FleetStatsRowsResult:
    rows: list[dict[str, Any]]
    errors: list[dict[str, str]] = field(default_factory=list)
    hosts_ok: int = 0
    hosts_failed: int = 0


@dataclass(frozen=True)
class FleetBashSuggestionsResult:
    payload: dict[str, Any]
    errors: list[dict[str, str]] = field(default_factory=list)
    hosts_ok: int = 0
    hosts_failed: int = 0


def _require_nonneg_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    return value


def _require_nonempty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _decode_overview(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object, got {type(value).__name__}")
    return {key: _require_nonneg_int(value.get(key), key) for key in _OVERVIEW_COUNTER_KEYS}


_VALID_SUGGESTION_CONFIDENCE = frozenset({"high", "medium", "review"})
_COMPOUND_REASON = "Contains compound operators"


def _decode_tool_rows(
    value: Any, *, max_rows: int = DEFAULT_TOOL_STATS_LIMIT
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array, got {type(value).__name__}")
    if len(value) > max_rows:
        raise ValueError(f"stats tools reply exceeds per-host limit of {max_rows} rows")
    decoded: list[dict[str, Any]] = []
    seen_tools: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{index}] must be an object")
        tool_name = _require_nonempty_str(row.get("tool_name"), f"rows[{index}].tool_name")
        if tool_name in seen_tools:
            raise ValueError(f"duplicate tool_name {tool_name!r} at rows[{index}]")
        seen_tools.add(tool_name)
        decoded.append(
            {
                "tool_name": tool_name,
                "count": _require_nonneg_int(row.get("count"), f"rows[{index}].count"),
            }
        )
    return decoded


def _decode_bash_rows(
    value: Any, *, max_rows: int = DEFAULT_BASH_STATS_LIMIT
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array, got {type(value).__name__}")
    if len(value) > max_rows:
        raise ValueError(f"stats bash reply exceeds per-host limit of {max_rows} rows")
    decoded: list[dict[str, Any]] = []
    seen_keys: set[tuple[str | None, str | None]] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{index}] must be an object")
        bash_base = _require_optional_str(row.get("bash_base"), f"rows[{index}].bash_base")
        bash_sub = _require_optional_str(row.get("bash_sub"), f"rows[{index}].bash_sub")
        key = (bash_base, bash_sub)
        if key in seen_keys:
            raise ValueError(f"duplicate bash key {key!r} at rows[{index}]")
        seen_keys.add(key)
        decoded.append(
            {
                "bash_base": bash_base,
                "bash_sub": bash_sub,
                "count": _require_nonneg_int(row.get("count"), f"rows[{index}].count"),
                "is_compound": _require_bool(row.get("is_compound"), f"rows[{index}].is_compound"),
            }
        )
    return decoded


def _decode_bash_suggestions(
    value: Any,
    *,
    max_rows: int = DEFAULT_BASH_SUGGESTIONS_LIMIT,
    allow_legacy_wildcard_collisions: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object, got {type(value).__name__}")
    raw_suggestions = value.get("suggestions")
    raw_skipped = value.get("skipped")
    if not isinstance(raw_suggestions, list):
        raise ValueError("suggestions must be a JSON array")
    if not isinstance(raw_skipped, list):
        raise ValueError("skipped must be a JSON array")
    if len(raw_suggestions) + len(raw_skipped) > max_rows:
        raise ValueError(f"stats bash --suggest reply exceeds per-host limit of {max_rows} rows")
    decoded_suggestions: list[dict[str, Any]] = []
    decoded_skipped: list[dict[str, Any]] = []
    seen_unique_suggestions: set[str] = set()
    seen_suggestion_patterns: set[str] = set()
    seen_unique_skipped: set[str] = set()
    for index, row in enumerate(raw_suggestions):
        if not isinstance(row, dict):
            raise ValueError(f"suggestions[{index}] must be an object")
        pattern = _require_nonempty_str(row.get("pattern"), f"suggestions[{index}].pattern")
        # Non-wildcard patterns map 1-to-1 to SQL (bash_base, bash_sub), and upgraded
        # (filtered) producers coalesce all display keys before returning. Only unfiltered
        # legacy v0.48.0 replies may emit colliding `... *` rows (e.g. `(git, NULL)` and
        # `(git, "*")`), which coalesce additively during merge.
        require_unique = not (allow_legacy_wildcard_collisions and pattern.endswith(" *"))
        if require_unique and pattern in seen_unique_suggestions:
            raise ValueError(f"duplicate bash pattern {pattern!r} at suggestions[{index}]")
        if require_unique:
            seen_unique_suggestions.add(pattern)
        seen_suggestion_patterns.add(pattern)
        count = _require_nonneg_int(row.get("count"), f"suggestions[{index}].count")
        confidence = _require_nonempty_str(
            row.get("confidence"), f"suggestions[{index}].confidence"
        )
        if confidence not in _VALID_SUGGESTION_CONFIDENCE:
            raise ValueError(
                f"suggestions[{index}].confidence must be one of high/medium/review, "
                f"got {confidence!r}"
            )
        reason = _require_nonempty_str(row.get("reason"), f"suggestions[{index}].reason")
        decoded_suggestions.append(
            {
                "pattern": pattern,
                "count": count,
                "is_compound": reason == _COMPOUND_REASON,
            }
        )
    for index, row in enumerate(raw_skipped):
        if not isinstance(row, dict):
            raise ValueError(f"skipped[{index}] must be an object")
        pattern = _require_nonempty_str(row.get("pattern"), f"skipped[{index}].pattern")
        if pattern in seen_suggestion_patterns:
            raise ValueError(f"bash pattern {pattern!r} appeared in both suggestions and skipped")
        require_unique = not (allow_legacy_wildcard_collisions and pattern.endswith(" *"))
        if require_unique and pattern in seen_unique_skipped:
            raise ValueError(f"duplicate bash pattern {pattern!r} at skipped[{index}]")
        if require_unique:
            seen_unique_skipped.add(pattern)
        count = _require_nonneg_int(row.get("count"), f"skipped[{index}].count")
        reason = _require_nonempty_str(row.get("reason"), f"skipped[{index}].reason")
        decoded_skipped.append({"pattern": pattern, "count": count, "reason": reason})
    return {"suggestions": decoded_suggestions, "skipped": decoded_skipped}


def _decode_token_rows(
    value: Any, *, max_rows: int = DEFAULT_TOKEN_STATS_LIMIT
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array, got {type(value).__name__}")
    if len(value) > max_rows:
        raise ValueError(f"stats tokens reply exceeds per-host limit of {max_rows} rows")
    decoded: list[dict[str, Any]] = []
    seen_repos: set[str | None] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{index}] must be an object")
        repo = _require_optional_str(row.get("repo"), f"rows[{index}].repo")
        if repo in seen_repos:
            raise ValueError(f"duplicate repo {repo!r} at rows[{index}]")
        seen_repos.add(repo)
        decoded.append(
            {
                "repo": repo,
                "input_tokens": _require_nonneg_int(
                    row.get("input_tokens"), f"rows[{index}].input_tokens"
                ),
                "output_tokens": _require_nonneg_int(
                    row.get("output_tokens"), f"rows[{index}].output_tokens"
                ),
            }
        )
    return decoded


def _filter_argv(source_value: str | None, since_iso: str | None) -> list[str]:
    argv: list[str] = []
    if source_value is not None:
        argv.extend(["--source", source_value])
    if since_iso is not None:
        argv.extend(["--since", since_iso])
    return argv


def fleet_stats_overview(
    executor: FleetExecutor,
    *,
    source: str | None = None,
    since: str | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetStatsOverviewResult:
    filters = normalize_stats_filters(source=source, since=since)
    source_value = filters.source.value if filters.source is not None else None
    argv = ["recall", "stats", "--json", *_filter_argv(source_value, filters.since_iso)]
    if filters.active:
        method = "recall.stats_filtered"
        params: Params = StatsFilteredParams(source=source_value, since=filters.since_iso)
    else:
        method = "recall.stats"
        params = Params()

    outcomes = executor.execute(
        FleetOperation(
            method,
            params,
            argv,
            _decode_overview,
            local_seconds=timeout,
            remote_seconds=timeout,
            invalid_reply_prefix="invalid stats JSON: ",
            allow_blank_reply=False,
        ),
        concurrency=concurrency,
    )
    totals = {key: 0 for key in _OVERVIEW_COUNTER_KEYS}
    errors: list[dict[str, str]] = []
    hosts_ok = 0
    for outcome in outcomes:
        if isinstance(outcome, EndpointSuccess):
            hosts_ok += 1
            for key in _OVERVIEW_COUNTER_KEYS:
                totals[key] += outcome.value[key]
        else:
            errors.append({"name": outcome.name, "error": outcome.error})
    return FleetStatsOverviewResult(
        stats={**totals, **_EMPTY_LAST_RUN_FIELDS},
        errors=errors,
        hosts_ok=hosts_ok,
        hosts_failed=len(errors),
    )


def fleet_stats_tools(
    executor: FleetExecutor,
    *,
    source: str | None = None,
    since: str | None = None,
    limit: int = DEFAULT_TOOL_STATS_LIMIT,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetStatsRowsResult:
    filters = normalize_stats_filters(source=source, since=since)
    source_value = filters.source.value if filters.source is not None else None
    argv = ["recall", "stats", "tools", "--json", *_filter_argv(source_value, filters.since_iso)]
    if filters.active:
        method = "recall.stats_tools_filtered"
        params: Params = StatsFilteredParams(source=source_value, since=filters.since_iso)
    else:
        method = "recall.stats_tools"
        params = Params()

    outcomes = executor.execute(
        FleetOperation(
            method,
            params,
            argv,
            _decode_tool_rows,
            local_seconds=timeout,
            remote_seconds=timeout,
            invalid_reply_prefix="invalid stats tools JSON: ",
            allow_blank_reply=False,
        ),
        concurrency=concurrency,
    )
    counts: dict[str, int] = {}
    errors: list[dict[str, str]] = []
    hosts_ok = 0
    for outcome in outcomes:
        if isinstance(outcome, EndpointSuccess):
            hosts_ok += 1
            for row in outcome.value:
                name = row["tool_name"]
                counts[name] = counts.get(name, 0) + row["count"]
        else:
            errors.append({"name": outcome.name, "error": outcome.error})
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if limit > 0:
        ordered = ordered[:limit]
    rows = [{"tool_name": name, "count": count} for name, count in ordered]
    return FleetStatsRowsResult(
        rows=rows, errors=errors, hosts_ok=hosts_ok, hosts_failed=len(errors)
    )


def _merge_bash_outcomes(
    outcomes: list[Any],
) -> tuple[list[BashStat], list[dict[str, str]], int]:
    buckets: dict[tuple[str | None, str | None], tuple[int, bool]] = {}
    errors: list[dict[str, str]] = []
    hosts_ok = 0
    for outcome in outcomes:
        if isinstance(outcome, EndpointSuccess):
            hosts_ok += 1
            for row in outcome.value:
                key = (row["bash_base"], row["bash_sub"])
                prev_count, prev_compound = buckets.get(key, (0, False))
                buckets[key] = (prev_count + row["count"], prev_compound or row["is_compound"])
        else:
            errors.append({"name": outcome.name, "error": outcome.error})
    ordered = sorted(
        buckets.items(),
        key=lambda item: (
            -item[1][0],
            item[0][0] is None,
            item[0][0] or "",
            item[0][1] is None,
            item[0][1] or "",
        ),
    )
    stats = [
        BashStat(bash_base=base, bash_sub=sub, count=count, is_compound=compound)
        for (base, sub), (count, compound) in ordered
    ]
    return stats, errors, hosts_ok


def _merge_bash_suggestion_outcomes(
    outcomes: list[Any],
    *,
    high_threshold: int = 50,
    medium_threshold: int = 10,
) -> FleetBashSuggestionsResult:
    suggestion_buckets: dict[str, tuple[int, bool]] = {}
    skipped_buckets: dict[str, tuple[int, str]] = {}
    errors: list[dict[str, str]] = []
    hosts_ok = 0
    for outcome in outcomes:
        if isinstance(outcome, EndpointSuccess):
            hosts_ok += 1
            for row in outcome.value["suggestions"]:
                pat = row["pattern"]
                prev_count, prev_compound = suggestion_buckets.get(pat, (0, False))
                suggestion_buckets[pat] = (
                    prev_count + row["count"],
                    prev_compound or row["is_compound"],
                )
            for row in outcome.value["skipped"]:
                pat = row["pattern"]
                prev_count, prev_reason = skipped_buckets.get(pat, (0, row["reason"]))
                skipped_buckets[pat] = (prev_count + row["count"], prev_reason)
        else:
            errors.append({"name": outcome.name, "error": outcome.error})

    # Destructive/skipped always takes precedence if any host skipped a pattern.
    for pat in list(skipped_buckets):
        if pat in suggestion_buckets:
            extra_count, _ = suggestion_buckets.pop(pat)
            prev_count, reason = skipped_buckets[pat]
            skipped_buckets[pat] = (prev_count + extra_count, reason)

    suggestions: list[PermissionSuggestion] = []
    for pat, (count, is_compound) in sorted(
        suggestion_buckets.items(), key=lambda item: (-item[1][0], item[0])
    ):
        if is_compound:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pat,
                    count=count,
                    confidence="review",
                    reason=_COMPOUND_REASON,
                )
            )
        elif count >= high_threshold:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pat,
                    count=count,
                    confidence="high",
                    reason="No dangerous patterns detected",
                )
            )
        elif count >= medium_threshold:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pat,
                    count=count,
                    confidence="medium",
                    reason="No dangerous patterns detected",
                )
            )
        else:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pat,
                    count=count,
                    confidence="review",
                    reason="Low usage volume",
                )
            )

    skipped = [
        PermissionSkipped(pattern=pat, count=count, reason=reason)
        for pat, (count, reason) in sorted(
            skipped_buckets.items(), key=lambda item: (-item[1][0], item[0])
        )
    ]
    return FleetBashSuggestionsResult(
        payload={
            "suggestions": [asdict(item) for item in suggestions],
            "skipped": [asdict(item) for item in skipped],
        },
        errors=errors,
        hosts_ok=hosts_ok,
        hosts_failed=len(errors),
    )


def fleet_stats_bash(
    executor: FleetExecutor,
    *,
    suggest: bool = False,
    source: str | None = None,
    since: str | None = None,
    limit: int = DEFAULT_BASH_STATS_LIMIT,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetStatsRowsResult | FleetBashSuggestionsResult:
    """Fan out bash breakdowns or suggestions across the fleet and merge."""
    filters = normalize_stats_filters(source=source, since=since)
    source_value = filters.source.value if filters.source is not None else None
    filter_args = _filter_argv(source_value, filters.since_iso)
    if suggest:
        # Query `--suggest` on each host so per-host replies include up to
        # DEFAULT_BASH_SUGGESTIONS_LIMIT (500) patterns rather than truncating at 100.
        argv = ["recall", "stats", "bash", "--suggest", "--json", *filter_args]
        if filters.active:
            method = "recall.stats_bash_filtered"
            params: Params = StatsBashFilteredParams(
                suggest=True, source=source_value, since=filters.since_iso
            )
        else:
            method = "recall.stats_bash"
            params = BashStatsParams(suggest=True)
        allow_legacy_collisions = not filters.active
        outcomes = executor.execute(
            FleetOperation(
                method,
                params,
                argv,
                lambda value: _decode_bash_suggestions(
                    value,
                    allow_legacy_wildcard_collisions=allow_legacy_collisions,
                ),
                local_seconds=timeout,
                remote_seconds=timeout,
                invalid_reply_prefix="invalid stats bash JSON: ",
                allow_blank_reply=False,
            ),
            concurrency=concurrency,
        )
        return _merge_bash_suggestion_outcomes(outcomes)

    argv = ["recall", "stats", "bash", "--json", *filter_args]
    if filters.active:
        method = "recall.stats_bash_filtered"
        params = StatsBashFilteredParams(
            suggest=False, source=source_value, since=filters.since_iso
        )
    else:
        method = "recall.stats_bash"
        params = BashStatsParams(suggest=False)

    outcomes = executor.execute(
        FleetOperation(
            method,
            params,
            argv,
            _decode_bash_rows,
            local_seconds=timeout,
            remote_seconds=timeout,
            invalid_reply_prefix="invalid stats bash JSON: ",
            allow_blank_reply=False,
        ),
        concurrency=concurrency,
    )
    merged, errors, hosts_ok = _merge_bash_outcomes(outcomes)
    capped = merged[:limit] if limit > 0 else merged
    return FleetStatsRowsResult(
        rows=[asdict(item) for item in capped],
        errors=errors,
        hosts_ok=hosts_ok,
        hosts_failed=len(errors),
    )


def fleet_stats_tokens(
    executor: FleetExecutor,
    *,
    source: str | None = None,
    since: str | None = None,
    limit: int = DEFAULT_TOKEN_STATS_LIMIT,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetStatsRowsResult:
    filters = normalize_stats_filters(source=source, since=since)
    source_value = filters.source.value if filters.source is not None else None
    argv = ["recall", "stats", "tokens", "--json", *_filter_argv(source_value, filters.since_iso)]
    if filters.active:
        method = "recall.stats_tokens_filtered"
        params: Params = StatsFilteredParams(source=source_value, since=filters.since_iso)
    else:
        method = "recall.stats_tokens"
        params = Params()

    outcomes = executor.execute(
        FleetOperation(
            method,
            params,
            argv,
            _decode_token_rows,
            local_seconds=timeout,
            remote_seconds=timeout,
            invalid_reply_prefix="invalid stats tokens JSON: ",
            allow_blank_reply=False,
        ),
        concurrency=concurrency,
    )
    totals: dict[str | None, tuple[int, int]] = {}
    errors: list[dict[str, str]] = []
    hosts_ok = 0
    for outcome in outcomes:
        if isinstance(outcome, EndpointSuccess):
            hosts_ok += 1
            for row in outcome.value:
                repo = row["repo"]
                prev_in, prev_out = totals.get(repo, (0, 0))
                totals[repo] = (prev_in + row["input_tokens"], prev_out + row["output_tokens"])
        else:
            errors.append({"name": outcome.name, "error": outcome.error})
    ordered = sorted(
        totals.items(),
        key=lambda item: (-(item[1][0] + item[1][1]), item[0] is None, item[0] or ""),
    )
    if limit > 0:
        ordered = ordered[:limit]
    rows = [
        {"repo": repo, "input_tokens": inp, "output_tokens": out} for repo, (inp, out) in ordered
    ]
    return FleetStatsRowsResult(
        rows=rows, errors=errors, hosts_ok=hosts_ok, hosts_failed=len(errors)
    )
