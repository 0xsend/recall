from __future__ import annotations

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
from recall.cli.manifest import output_fields_for
from recall.cli.rpc import rpc_call_or_error
from recall.cli.status_notices import (
    fetch_daemon_status,
    pending_coverage_guidance,
    render_status_notice_for_status,
)
from recall.core.config import AppConfig


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
) -> None:
    """Emit human guidance to stderr when a search is empty or semantic-only.

    Guidance is advisory and goes to stderr in every output mode so structured
    stdout stays pure data (REQ-CLI-012/018/019). The `lexical_match` field on
    each row carries the same signal in stdout for callers that drop stderr.

    `coverage_note` is the caller's reading of reconciliation coverage; it rides
    on every zero-result branch because a backlog is the one condition under
    which "no results" really can mean "not indexed yet" (REQ-CLI-019/023).
    """

    def emit_empty(note: str) -> None:
        typer.echo(f"{note} {coverage_note}" if coverage_note else note, err=True)

    if not results:
        if tool:
            # REQ-CLI-019: a zero-result --tool query is ambiguous — over-restrictive
            # filter or genuinely unmatched? Re-probe once without the filter
            # (keeping the other filters) and report the delta so the caller knows.
            reprobe_total: int | None = None
            reprobe_lexical = 0
            try:
                probe_params: dict[str, object] = {"query": query, "limit": limit}
                if session:
                    probe_params["session"] = session
                if source:
                    probe_params["source"] = source
                if mode:
                    probe_params["mode"] = mode
                unfiltered = rpc_call_or_error("recall.search", probe_params)
                reprobe_total = len(unfiltered)
                reprobe_lexical = sum(1 for row in unfiltered if row.get("lexical_match"))
            except CliError:
                reprobe_total = None
            if reprobe_total:
                emit_empty(
                    f"note: 0 results with --tool {tool} (restricts to {tool} tool calls, "
                    f"excludes message text); {reprobe_total} results without it "
                    f"({reprobe_lexical} keyword matches). Drop --tool to search "
                    "conversation content."
                )
            elif reprobe_total == 0:
                emit_empty(
                    f"note: 0 results with --tool {tool} and 0 without it — nothing matches "
                    f"{query!r}. Confirm the session is indexed (not just unmatched): "
                    "recall show <id> / recall list."
                )
            else:
                emit_empty(
                    f"note: 0 results with --tool {tool}. The filter restricts to {tool} tool "
                    "calls and excludes message text — retry without --tool."
                )
        else:
            emit_empty(
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
    session: str | None = typer.Option(None, "--session", help="Filter by session ID"),
    source: str | None = typer.Option(
        None, "--source", help="claude-code, codex, pi-agent, grok, kimi-code"
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="keyword|vector|hybrid|auto (auto picks hybrid when embeddings exist, else keyword)",
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum results to return"),
    fleet: bool = typer.Option(
        False, "--fleet", help="Fan-out search to fleet.toml hosts and merge hits"
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
                "session",
                "source",
                "mode",
                "limit",
                "fleet",
                "fleet_config",
                "fields",
                "format",
                "json",
                "cta",
            },
            types={
                "query": "string",
                "tool": "string",
                "session": "string",
                "source": "string",
                "mode": "string",
                "limit": "integer",
                "fleet": "boolean",
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
        session = resolve_param(ctx, payload, "session", session)
        source = resolve_param(ctx, payload, "source", source)
        validate_source(source)
        mode = resolve_param(ctx, payload, "mode", mode)
        if mode is not None and mode not in {"auto", "keyword", "vector", "hybrid"}:
            raise ValueError(f"unsupported search mode: {mode}")
        limit = resolve_int_param(ctx, payload, "limit", limit)
        fleet = resolve_bool_param(ctx, payload, "fleet", fleet)
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

        if fleet:
            if session:
                raise CliError(
                    code=ErrorCode.VALIDATION,
                    message="--session is not supported with --fleet search",
                    exit_code=2,
                )
            from recall.cli.fleet import run_fleet_search

            results = run_fleet_search(
                query=query,
                tool=tool,
                source=source,
                mode=mode,
                limit=limit,
                fleet_config=fleet_config,
            )
        else:
            rpc_params: dict[str, object] = {"query": query, "limit": limit}
            if tool:
                rpc_params["tool"] = tool
            if session:
                rpc_params["session"] = session
            if source:
                rpc_params["source"] = source
            if mode:
                rpc_params["mode"] = mode

            results = rpc_call_or_error("recall.search", rpc_params)
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
    status = fetch_daemon_status(config)
    coverage_note = pending_coverage_guidance(status) if not results else None

    # REQ-CLI-018/019: stderr guidance for empty or semantic-only results, in
    # every output mode (runs before the structured early-return below).
    _emit_search_guidance(
        results,
        query=query,
        tool=tool,
        session=session,
        source=source,
        mode=mode,
        limit=limit,
        coverage_note=coverage_note,
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

    ctas = cta_for_search(results) if (cta or output_format == OutputFormat.TEXT) else []

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
        session_id = result.get("session_id", "?")
        source_val = result.get("source", "?")
        header = f"[{score:.2f}] {session_id} ({source_val})"
        typer.echo(header)
        if result.get("kind") == "message":
            content = result.get("content")
            snippet = content.strip() if isinstance(content, str) else str(content or "")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            role = result.get("role") or "unknown"
            typer.echo(f"  {role}: {snippet}")
        else:
            tool_name = result.get("tool_name") or "tool"
            bash_cmd = result.get("bash_command") or ""
            typer.echo(f"  [{tool_name}] {bash_cmd}")
        if result.get("source_path"):
            typer.echo(f"  source: {result['source_path']}")
        if result.get("timestamp"):
            typer.echo(f"  time: {result['timestamp']}")
        typer.echo("")

    render_cta_hints(ctas)
