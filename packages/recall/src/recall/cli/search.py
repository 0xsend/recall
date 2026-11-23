from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

import typer

from recall.cli.contract import (
    CliError,
    ErrorCode,
    OutputFormat,
    emit_data_with_cta,
    emit_error,
    parse_fields,
    parse_params,
    render_cta_hints,
    resolve_bool_param,
    resolve_int_param,
    resolve_output_format_early,
    resolve_output_format_from_params,
    resolve_param,
    validate_source,
    validate_structured_fields,
)
from recall.cli.cta import cta_for_search
from recall.cli.fleet_help import FLEET_QUERY_HELP, INVENTORY_ONLY_HELP
from recall.cli.manifest import output_fields_for
from recall.cli.rpc import rpc_call_or_error
from recall.cli.status_notices import (
    fetch_daemon_status,
    pending_coverage_guidance,
    render_status_notice_for_status,
)
from recall.cli.utils import with_fleet_host
from recall.core.config import AppConfig
from recall.core.search_filters import (
    ROLE_HELP,
    has_filtered_search_params,
    normalize_search_filters,
)
from recall.core.snippets import build_search_snippet, sanitize_snippet
from recall.core.tool_format import format_tool_call_line
from recall.core.types import SOURCE_HELP


def _normalize_search_snippets(
    results: list[dict[str, Any]],
    *,
    query: str,
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Sanitize/bound server-supplied snippets and backfill missing ones from older peers."""
    normalized: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            normalized.append(row)
            continue
        try:
            raw_snippet = row.get("snippet")
            if isinstance(raw_snippet, str):
                snippet = sanitize_snippet(raw_snippet)
            else:
                snippet = build_search_snippet(
                    query=query,
                    kind=str(row.get("kind") or "message"),
                    content=row.get("content") if isinstance(row.get("content"), str) else None,
                    thinking=row.get("thinking") if isinstance(row.get("thinking"), str) else None,
                    role=row.get("role") if isinstance(row.get("role"), str) else None,
                    tool_name=(
                        row.get("tool_name") if isinstance(row.get("tool_name"), str) else None
                    ),
                    bash_command=(
                        row.get("bash_command")
                        if isinstance(row.get("bash_command"), str)
                        else None
                    ),
                    tool_input=(
                        row.get("tool_input")
                        if isinstance(row.get("tool_input"), Mapping)
                        else None
                    ),
                    fields=fields,
                )
        except Exception:
            snippet = ""
        normalized.append({**row, "snippet": snippet})
    return normalized


def _render_search_progress(method: str, payload: dict[str, Any]) -> None:
    """Keep advisory search state on stderr, including for structured callers."""
    if method != "search_progress":
        return
    attempt = payload.get("attempt")
    if attempt not in (1, 2):
        return
    prefix = "search" if attempt == 1 else "search (attempt 2)"
    if payload.get("state") == "waiting":
        limit = payload.get("preference_limit_ms")
        if not isinstance(limit, (int, float)) or limit < 0:
            return
        line = (
            f"{prefix}: waiting for embedding drain "
            f"(preference up to {limit / 1000:g}s; request deadline 30s)"
        )
    elif payload.get("state") == "released":
        wait = payload.get("wait_ms")
        reason = {"drain_idle": "drain idle", "preference_elapsed": "limit reached"}.get(
            payload.get("reason")
        )
        if not isinstance(wait, (int, float)) or wait < 0 or reason is None:
            return
        line = (
            f"{prefix}: embedding preference ended after {wait / 1000:.2f}s "
            f"({reason}); joining scan admission"
        )
    else:
        return
    typer.echo(line, err=True)
    sys.stderr.flush()


def _widening_mode_hint(mode: str | None) -> str:
    """Name a mode that could return more than the one the caller just ran.

    The advice used to say "try --mode keyword" whatever was asked for, which
    recommends the mode already in use for `--mode keyword`, and for every other
    mode names a strict subset of what already ran. Only two transitions widen:
    keyword to the auto default, which adds the semantic leg when the host has
    embeddings, and vector to keyword, which can still match exactly where the
    nearest neighbours did not. Hybrid and auto already ran both legs.
    """
    if mode == "keyword":
        return " Widen the query, or drop `--mode keyword` so auto adds the semantic leg."
    if mode == "vector":
        return " Widen the query or try --mode keyword to require exact matches."
    return " Widen the query."


@dataclass(frozen=True)
class NoProbe:
    pass


@dataclass(frozen=True)
class UnavailableProbe:
    pass


@dataclass(frozen=True)
class CountedProbe:
    total: int
    lexical: int


ToolProbe = NoProbe | UnavailableProbe | CountedProbe
NO_PROBE = NoProbe()


def _acquire_tool_probe(
    results: list[dict],
    *,
    query: str | None,
    tool: str | None,
    session: str | None,
    source: str | None,
    mode: str | None,
    project: str | None = None,
    since: str | None = None,
    host: str | None = None,
    per_session: int | None = None,
    limit: int,
    keyword_index_unready: bool = False,
) -> ToolProbe:
    from recall.core.read_scope import AggregateRead, current_invocation

    if (
        results
        or not tool
        or keyword_index_unready
        or isinstance(current_invocation().capability, AggregateRead)
    ):
        return NoProbe()
    params: dict[str, object] = {"query": query, "limit": limit}
    if session is not None:
        params["session"] = session
    if source:
        params["source"] = source
    if mode:
        params["mode"] = mode
    if project is not None:
        params["project"] = project
    if since is not None:
        params["since"] = since
    if host is not None:
        params["host"] = host
    if per_session is not None:
        params["per_session"] = per_session
    method = (
        "recall.search_filtered"
        if has_filtered_search_params(
            project=project,
            since=since,
            host=host,
            per_session=per_session,
            fleet=False,
        )
        else "recall.search"
    )
    try:
        unfiltered = rpc_call_or_error(method, params)
    except CliError:
        return UnavailableProbe()
    return CountedProbe(len(unfiltered), sum(1 for row in unfiltered if row.get("lexical_match")))


def _emit_search_guidance(
    results: list[dict],
    *,
    query: str | None,
    tool: str | None,
    session: str | None,
    source: str | None,
    mode: str | None,
    limit: int,
    coverage_note: str | None = None,
    keyword_index_unready: bool = False,
    probe: ToolProbe = NO_PROBE,
) -> None:
    """Emit human guidance about search matches and coverage to stderr.

    Guidance is advisory and goes to stderr in every output mode so structured
    stdout stays pure data (REQ-CLI-012/018/019). The `lexical_match` field on
    each row carries the same signal in stdout for callers that drop stderr.

    `coverage_note` is the caller's reading of reconciliation coverage; it rides
    on every zero-result branch because a backlog can mean "not indexed yet".
    Known-unready keyword coverage replaces match guidance even with results:
    neither absence nor semantic-only hits establish a lexical miss then
    (REQ-CLI-019/023).
    """

    def emit_guidance(note: str) -> None:
        typer.echo(f"{note} {coverage_note}" if coverage_note else note, err=True)

    if keyword_index_unready:
        emit_guidance(
            "Warning: keyword index is not ready; keyword results may be incomplete. "
            "Retry after keyword_search_ready becomes true in recall daemon status."
        )
        return

    if not results:
        if tool:
            reprobe_total = probe.total if isinstance(probe, CountedProbe) else None
            reprobe_lexical = probe.lexical if isinstance(probe, CountedProbe) else 0
            if reprobe_total:
                emit_guidance(
                    f"note: 0 results with --tool {tool} (restricts to {tool} tool calls, "
                    f"excludes message text); {reprobe_total} results without it "
                    f"({reprobe_lexical} keyword matches). Drop --tool to search "
                    "conversation content."
                )
            elif reprobe_total == 0:
                emit_guidance(
                    f"note: 0 results with --tool {tool} and 0 without it — nothing matches "
                    f"{query!r}. Confirm the session is indexed (not just unmatched): "
                    "recall show <id> / recall list."
                )
            else:
                emit_guidance(
                    f"note: 0 results with --tool {tool}. The filter restricts to {tool} tool "
                    "calls and excludes message text — retry without --tool."
                )
        else:
            emit_guidance(
                f"note: no matches for {query!r}.{_widening_mode_hint(mode)} To "
                "confirm a session is indexed (vs unmatched): recall show <id> / recall list. "
                "Parked unsupported sources are a recall parser gap — see daemon status "
                "coverage.unsupported; file a recall issue or fix the parser. "
                "recall does not index commands run directly in a terminal — check shell "
                "history for those."
            )
        return

    # REQ-CLI-018: results exist but none matched the lexical leg — they are
    # semantic neighbors that may be unrelated. `--mode vector` opts into that
    # explicitly, so don't nag there.
    if mode != "vector" and not any(row.get("lexical_match") for row in results):
        typer.echo(
            f"note: no keyword matches for {query!r} — showing {len(results)} "
            "semantically-nearest results that may be unrelated. Refine terms, or try "
            "--mode keyword to require exact matches.",
            err=True,
        )


def command(
    ctx: typer.Context,
    query: str | None = typer.Argument(None, help="Search query"),
    tool: str | None = typer.Option(
        None,
        "--tool",
        help="Restrict to one tool's calls (any tool name, e.g. Bash); excludes message text",
    ),
    role: Annotated[
        list[str] | None,
        typer.Option("--role", help=ROLE_HELP),
    ] = None,
    project: str | None = typer.Option(
        None, "--project", help="Filter by git_repo substring (case-insensitive)"
    ),
    since: str | None = typer.Option(None, "--since", help="Time window e.g. 7d, 24h, 2026-01-01"),
    host: str | None = typer.Option(
        None, "--host", help="Filter by session host (or select fleet endpoint with --fleet)"
    ),
    per_session: int | None = typer.Option(
        None, "--per-session", help="Maximum hits to return per session"
    ),
    session: str | None = typer.Option(
        None, "--session", help="Recall session ID, source session ID, or unique recall ID prefix"
    ),
    source: str | None = typer.Option(None, "--source", help=SOURCE_HELP),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="keyword|vector|hybrid|auto (auto picks hybrid when embeddings exist, else keyword)",
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum results to return"),
    fleet: bool = typer.Option(False, "--fleet", help=FLEET_QUERY_HELP),
    fleet_inventory_only: bool = typer.Option(
        False,
        "--fleet-inventory-only",
        help=INVENTORY_ONLY_HELP,
    ),
    fleet_config: str | None = typer.Option(
        None, "--fleet-config", help="Fleet inventory path (default: ~/.config/recall/fleet.toml)"
    ),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
) -> None:
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params,
            allowed={
                "query",
                "tool",
                "role",
                "project",
                "since",
                "host",
                "per_session",
                "session",
                "source",
                "mode",
                "limit",
                "fleet",
                "fleet_inventory_only",
                "fleet_config",
                "fields",
                "format",
                "json",
                "cta",
            },
            types={
                "query": "string",
                "tool": "string",
                "role": "string[]",
                "project": "string",
                "since": "string",
                "host": "string",
                "per_session": "integer",
                "session": "string",
                "source": "string",
                "mode": "string",
                "limit": "integer",
                "fleet": "boolean",
                "fleet_inventory_only": "boolean",
                "fleet_config": "string",
                "fields": "string",
                "format": "string",
                "json": "boolean",
                "cta": "boolean",
            },
        )
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        query = resolve_param(ctx, payload, "query", query)
        tool = resolve_param(ctx, payload, "tool", tool)
        role_values = resolve_param(ctx, payload, "role", role)
        project = resolve_param(ctx, payload, "project", project)
        since = resolve_param(ctx, payload, "since", since)
        host = resolve_param(ctx, payload, "host", host)
        per_session = resolve_param(ctx, payload, "per_session", per_session)
        session = resolve_param(ctx, payload, "session", session)
        source = resolve_param(ctx, payload, "source", source)
        validate_source(source)
        mode = resolve_param(ctx, payload, "mode", mode)
        if mode is not None and mode not in {"auto", "keyword", "vector", "hybrid"}:
            raise ValueError(f"unsupported search mode: {mode}")
        limit = resolve_int_param(ctx, payload, "limit", limit)
        fleet = resolve_bool_param(ctx, payload, "fleet", fleet)
        fleet_inventory_only = resolve_bool_param(
            ctx,
            payload,
            "fleet_inventory_only",
            fleet_inventory_only,
        )
        if fleet_inventory_only and not fleet:
            raise ValueError("--fleet-inventory-only requires --fleet")
        fleet_config = resolve_param(ctx, payload, "fleet_config", fleet_config)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        if limit <= 0:
            raise ValueError("limit must be positive")
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        if not query:
            raise CliError(
                code=ErrorCode.VALIDATION,
                message="search query is required",
                details={"field": "query"},
                exit_code=2,
            )
        filters = normalize_search_filters(
            tool=tool,
            role=role_values,
            project=project,
            since=since,
            host=host,
            per_session=per_session,
        )
        since_str = since.strip() if since is not None else None

        top_endpoint: str | None = None
        if fleet:
            if session is not None:
                raise CliError(
                    code=ErrorCode.VALIDATION,
                    message="--session is not supported with --fleet search",
                    exit_code=2,
                )
            from recall.cli.fleet import run_fleet_search

            results, top_endpoint = run_fleet_search(
                query=query,
                tool=tool,
                source=source,
                mode=mode,
                roles=filters.roles,
                project=filters.project,
                since=since_str,
                host=filters.host,
                per_session=filters.per_session,
                limit=limit,
                fleet_config=fleet_config,
                fleet_inventory_only=fleet_inventory_only,
            )
        else:
            rpc_params: dict[str, object] = {"query": query, "limit": limit}
            if tool:
                rpc_params["tool"] = tool
            if session is not None:
                rpc_params["session"] = session
            if source:
                rpc_params["source"] = source
            if mode:
                rpc_params["mode"] = mode
            if filters.roles:
                rpc_params["role"] = [r.value for r in filters.roles]
            if filters.project is not None:
                rpc_params["project"] = filters.project
            if since_str is not None:
                rpc_params["since"] = since_str
            if filters.host is not None:
                rpc_params["host"] = filters.host
            if filters.per_session is not None:
                rpc_params["per_session"] = filters.per_session

            method = (
                "recall.search_filtered"
                if has_filtered_search_params(
                    roles=filters.roles,
                    project=filters.project,
                    since=filters.since,
                    host=filters.host,
                    per_session=filters.per_session,
                    fleet=False,
                )
                else "recall.search"
            )
            try:
                results = rpc_call_or_error(
                    method, rpc_params, on_notification=_render_search_progress
                )
            except CliError as err:
                if (
                    method == "recall.search_filtered"
                    and "method not found: recall.search_filtered" in err.message
                ):
                    raise CliError(
                        code=err.code,
                        message=(
                            "running daemon does not support search filters "
                            f"({err.message}); upgrade or run `recall daemon restart`"
                        ),
                        details=err.details,
                        exit_code=err.exit_code,
                    ) from None
                raise
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None

    # One daemon_status read serves both stderr consumers below: the
    # zero-result coverage clause (REQ-CLI-019) and the health notice
    # (REQ-CLI-021/023). The read honours `[cli] status_notices` and never
    # forks a daemon, so a search cannot start one to decorate its stderr.
    config = AppConfig.load()
    results = _normalize_search_snippets(results, query=query, fields=config.fts.fields)
    status = fetch_daemon_status(config)
    coverage_note = pending_coverage_guidance(status) if not results else None
    reconciliation = status.get("reconciliation") if status is not None else None
    uses_keyword = mode != "vector" or bool(tool)
    keyword_index_unready = (
        not fleet
        and uses_keyword
        and isinstance(reconciliation, dict)
        and reconciliation.get("keyword_search_ready") is False
    )

    # REQ-CLI-018/019: stderr guidance for empty or semantic-only results, in
    # every output mode (runs before the structured early-return below).
    probe = _acquire_tool_probe(
        results,
        query=query,
        tool=tool,
        session=session,
        source=source,
        mode=mode,
        project=filters.project,
        since=since_str,
        host=filters.host,
        per_session=filters.per_session,
        limit=limit,
        keyword_index_unready=keyword_index_unready,
    )
    _emit_search_guidance(
        results,
        query=query,
        tool=tool,
        session=session,
        source=source,
        mode=mode,
        limit=limit,
        coverage_note=coverage_note,
        keyword_index_unready=keyword_index_unready,
        probe=probe,
    )

    def status_notice(*, include_informational: bool) -> str | None:
        if status is None:
            return None
        # The coverage clause above already stated the backlog; repeating it
        # here would put the same fact on stderr twice (REQ-CLI-019).
        return render_status_notice_for_status(
            status,
            config=config,
            include_informational=include_informational,
            include_pending_coverage=coverage_note is None,
        )

    ctas = (
        cta_for_search(results, fleet=fleet, endpoint=top_endpoint)
        if (cta or output_format == OutputFormat.TEXT)
        else []
    )

    if output_format != OutputFormat.TEXT:
        # Warnings-only on stderr so agents using --json still see health issues
        # (version drift, bloat) without the human-oriented index-freshness line
        # (REQ-CLI-021/022, REQ-CLI-012).
        health_notice = status_notice(include_informational=False)
        if health_notice is not None:
            typer.echo(health_notice, err=True)
        try:
            emit_data_with_cta(
                results,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("search"),
            )
        except CliError as err:
            emit_error(err, output_format=output_format)
            raise typer.Exit(code=err.exit_code) from None
        return

    notice = status_notice(include_informational=True)
    if notice is not None:
        typer.echo(notice, err=True)

    if not results:
        typer.echo("No results.")
        render_cta_hints(ctas)
        return

    for result in results:
        score = result.get("score", 0)
        session_id = sanitize_snippet(str(result.get("session_id", "?")), max_chars=256)
        source_val = sanitize_snippet(str(result.get("source", "?")), max_chars=128)
        header = f"[{score:.2f}] {session_id} ({source_val})"
        typer.echo(with_fleet_host(header, result, fleet=fleet))
        if result.get("kind") == "message":
            snippet = result.get("snippet") or ""
            role_label = sanitize_snippet(str(result.get("role") or "unknown"), max_chars=64)
            typer.echo(f"  {role_label}: {snippet}")
        else:
            raw_tool_snippet = result.get("snippet")
            tool_line = (
                raw_tool_snippet
                if isinstance(raw_tool_snippet, str) and raw_tool_snippet
                else sanitize_snippet(format_tool_call_line(result))
            )
            typer.echo(f"  {tool_line}")
        if result.get("source_path"):
            safe_path = sanitize_snippet(str(result["source_path"]), max_chars=1024)
            typer.echo(f"  source: {safe_path}")
        if result.get("timestamp"):
            from recall.cli.utils import format_datetime

            ts_raw = result["timestamp"]
            formatted_ts = format_datetime(
                ts_raw if isinstance(ts_raw, (str, type(None))) else str(ts_raw)
            )
            safe_ts = sanitize_snippet(formatted_ts, max_chars=128)
            typer.echo(f"  time: {safe_ts}")
        typer.echo("")

    render_cta_hints(ctas)
