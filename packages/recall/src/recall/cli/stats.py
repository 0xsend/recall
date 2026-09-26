from __future__ import annotations

from typing import Annotated

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
    resolve_output_format_early,
    resolve_output_format_from_params,
    resolve_param,
    validate_structured_fields,
)
from recall.cli.cta import cta_for_stats
from recall.cli.manifest import output_fields_for, output_fields_for_variant
from recall.cli.rpc import rpc_call_or_error
from recall.cli.status_notices import render_status_notice
from recall.core.types import parse_source

app = typer.Typer(help="Analytics commands")

SKILL_CENSUS_RPC_TIMEOUT_SECONDS = 600.0


def _emit_status_notice(output_format: OutputFormat) -> None:
    """Surface daemon/database health on stderr (REQ-CLI-021/023).

    Every subcommand counts the same index the bare command does, so a drifted
    daemon or a reconciliation backlog makes `stats tools` as wrong as `stats`.
    Structured callers drop the human index-freshness line and keep the
    actionable warnings, and nothing reaches stdout (REQ-CLI-012).
    """
    notice = render_status_notice(include_informational=(output_format == OutputFormat.TEXT))
    if notice is not None:
        typer.echo(notice, err=True)


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params,
            allowed={"fields", "format", "json", "cta"},
            types={
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
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        stats = rpc_call_or_error("recall.stats", {})
        _emit_status_notice(output_format)
        ctas = cta_for_stats(None) if (cta or output_format == OutputFormat.TEXT) else []
        if output_format != OutputFormat.TEXT:
            emit_data_with_cta(
                stats,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("stats"),
            )
            return
        typer.echo("Overview")
        typer.echo(f"  Sessions: {stats.get('sessions', 0)}")
        typer.echo(f"  Messages: {stats.get('messages', 0)}")
        typer.echo(f"  Tool calls: {stats.get('tool_calls', 0)}")
        typer.echo(f"  Bash calls: {stats.get('bash_calls', 0)}")
        typer.echo("Last index run:")
        typer.echo(f"  Messages indexed:                 {stats.get('last_index_indexed', 0):,}")
        typer.echo(
            f"  Messages contextualized:           {stats.get('last_context_messages', 0):,}"
        )
        typer.echo(f"  Context backend:                   {stats.get('last_context_mode', 'off')}")
        typer.echo(
            f"  Context tokens (input):            {stats.get('last_context_input_tokens', 0):,}"
        )
        typer.echo(
            f"  Context tokens (output):           {stats.get('last_context_output_tokens', 0):,}"
        )
        typer.echo(
            f"  Context model:                     {stats.get('last_context_model') or '(none)'}"
        )
        render_cta_hints(ctas)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None


@app.command("tools")
def tools(
    ctx: typer.Context,
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
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
            allowed={"fields", "format", "json", "cta"},
            types={
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
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        stats = rpc_call_or_error("recall.stats_tools", {})
        _emit_status_notice(output_format)
        ctas = cta_for_stats("tools") if (cta or output_format == OutputFormat.TEXT) else []
        if output_format != OutputFormat.TEXT:
            emit_data_with_cta(
                stats,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("stats tools"),
            )
            return
        for stat in stats:
            typer.echo(f"{stat.get('tool_name', '?')}: {stat.get('count', 0)}")
        render_cta_hints(ctas)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None


@app.command("bash")
def bash(
    ctx: typer.Context,
    suggest: bool = typer.Option(False, "--suggest", help="Generate permission suggestions"),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
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
            allowed={"suggest", "fields", "format", "json", "cta"},
            types={
                "suggest": "boolean",
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
        suggest = resolve_bool_param(ctx, payload, "suggest", suggest)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        ctas = cta_for_stats("bash") if (cta or output_format == OutputFormat.TEXT) else []
        if suggest:
            result = rpc_call_or_error("recall.stats_bash", {"suggest": True})
            _emit_status_notice(output_format)
            if output_format != OutputFormat.TEXT:
                emit_data_with_cta(
                    result,
                    ctas,
                    output_format=output_format,
                    include_cta=cta,
                    fields=fields_value,
                    allowed_fields=output_fields_for_variant(
                        "stats bash",
                        {"suggest": True},
                    ),
                )
                return
            suggestions = result.get("suggestions", []) if isinstance(result, dict) else []
            skipped = result.get("skipped", []) if isinstance(result, dict) else []
            typer.echo("Suggested Bash Permissions")
            typer.echo("=" * 26)
            for suggestion in suggestions:
                typer.echo(
                    f"{suggestion.get('confidence', '?')}: "
                    f"{suggestion.get('pattern', '?')} "
                    f"({suggestion.get('count', 0)} uses)"
                )
            if skipped:
                typer.echo("Skipped")
                for item in skipped:
                    typer.echo(
                        f"- {item.get('pattern', '?')} "
                        f"({item.get('count', 0)} uses): "
                        f"{item.get('reason', '?')}"
                    )
            render_cta_hints(ctas)
            return

        stats = rpc_call_or_error("recall.stats_bash", {})
        _emit_status_notice(output_format)
        if output_format != OutputFormat.TEXT:
            emit_data_with_cta(
                stats,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for_variant(
                    "stats bash",
                    {"suggest": False},
                ),
            )
            return
        for stat in stats:
            base = stat.get("bash_base") or "unknown"
            sub = stat.get("bash_sub") or "*"
            suffix = " (compound)" if stat.get("is_compound") else ""
            typer.echo(f"{base} {sub}: {stat.get('count', 0)}{suffix}")
        render_cta_hints(ctas)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None


@app.command("tokens")
def tokens(
    ctx: typer.Context,
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
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
            allowed={"fields", "format", "json", "cta"},
            types={
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
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        stats = rpc_call_or_error("recall.stats_tokens", {})
        _emit_status_notice(output_format)
        ctas = cta_for_stats("tokens") if (cta or output_format == OutputFormat.TEXT) else []
        if output_format != OutputFormat.TEXT:
            emit_data_with_cta(
                stats,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("stats tokens"),
            )
            return
        for stat in stats:
            label = stat.get("repo") or "unknown"
            typer.echo(
                f"{label}: {stat.get('input_tokens', 0)} in / {stat.get('output_tokens', 0)} out"
            )
        render_cta_hints(ctas)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None


@app.command("usage")
def usage(
    ctx: typer.Context,
    since: str | None = typer.Option(None, "--since", help="Time window e.g. 30d, 12h"),
    fleet: bool = typer.Option(
        False,
        "--fleet",
        help="Fan-out to hosts in fleet.toml and merge usage (REQ-FLEET-CMD-002)",
    ),
    fleet_config: str | None = typer.Option(
        None,
        "--fleet-config",
        help="Fleet inventory path (default: ~/.config/recall/fleet.toml)",
    ),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    """Token usage by source x model x host (fleet ledger)."""
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params,
            allowed={"since", "fleet", "fleet_config", "fields", "format", "json", "cta"},
            types={
                "since": "string",
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
        since = resolve_param(ctx, payload, "since", since)
        fleet = resolve_bool_param(ctx, payload, "fleet", fleet)
        fleet_config = resolve_param(ctx, payload, "fleet_config", fleet_config)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        if fleet:
            from recall.cli.fleet import run_fleet_stats_usage

            stats = run_fleet_stats_usage(since=since, fleet_config=fleet_config)
        else:
            rpc_params: dict[str, object] = {}
            if since is not None:
                rpc_params["since"] = since
            stats = rpc_call_or_error("recall.stats_usage", rpc_params)

        _emit_status_notice(output_format)
        ctas = cta_for_stats("usage") if (cta or output_format == OutputFormat.TEXT) else []
        if output_format != OutputFormat.TEXT:
            emit_data_with_cta(
                stats,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("stats usage"),
            )
            return
        if not stats:
            typer.echo("No sessions with token usage in range.")
            render_cta_hints(ctas)
            return
        for stat in stats:
            source = stat.get("source") or "unknown"
            model = stat.get("model") or "-"
            host = stat.get("host") or "-"
            fresh = stat.get("fresh_input_tokens")
            cached = stat.get("cached_input_tokens", 0)
            parts = [
                f"{source}",
                f"model={model}",
                f"host={host}",
                f"in={stat.get('input_tokens', 0)}",
            ]
            if cached:
                parts.append(f"cached={cached}")
            if fresh is not None:
                parts.append(f"fresh={fresh}")
            parts.append(f"out={stat.get('output_tokens', 0)}")
            parts.append(f"sessions={stat.get('session_count', 0)}")
            typer.echo(" ".join(parts))
        render_cta_hints(ctas)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None


@app.command("skills")
def skills(
    ctx: typer.Context,
    since: str | None = typer.Option(None, "--since", help="Time window e.g. 30d, 12h"),
    source: Annotated[
        list[str] | None,
        typer.Option(
            "--source",
            help="Source to include; repeat for multiple sources",
        ),
    ] = None,
    local: bool = typer.Option(
        False,
        "--local",
        help="Query only the current daemon instead of the configured fleet",
    ),
    fleet_config: str | None = typer.Option(
        None,
        "--fleet-config",
        help="Fleet inventory path (default: ~/.config/recall/fleet.toml)",
    ),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    cta: bool = typer.Option(False, "--cta", help="Include call-to-action suggestions"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    """Skill invocations with complete local or local-plus-fleet coverage."""
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
                "since",
                "source",
                "local",
                "fleet_config",
                "fields",
                "format",
                "json",
                "cta",
            },
            types={
                "since": "string",
                "source": "string[]",
                "local": "boolean",
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
        since = resolve_param(ctx, payload, "since", since)
        source_values = resolve_param(ctx, payload, "source", source or [])
        local = resolve_bool_param(ctx, payload, "local", local)
        fleet_config = resolve_param(ctx, payload, "fleet_config", fleet_config)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        selected_sources = tuple(
            dict.fromkeys(parse_source(value) for value in (source_values or []))
        )
        rpc_params: dict[str, object] = {}
        if since is not None:
            rpc_params["since"] = since
        if selected_sources:
            rpc_params["source"] = [source.value for source in selected_sources]
        census = rpc_call_or_error(
            "recall.stats_skills",
            rpc_params,
            idle_timeout=SKILL_CENSUS_RPC_TIMEOUT_SECONDS,
        )
        if not isinstance(census, dict):
            raise CliError(
                code=ErrorCode.RUNTIME,
                message="local stats skills returned a non-object payload",
            )
        if not local:
            from recall.cli.fleet import run_fleet_stats_skills

            census = run_fleet_stats_skills(
                local_payload=census,
                since=since,
                sources=selected_sources,
                fleet_config=fleet_config,
            )

        _emit_status_notice(output_format)
        ctas = cta_for_stats("skills") if (cta or output_format == OutputFormat.TEXT) else []
        if output_format != OutputFormat.TEXT:
            emit_data_with_cta(
                census,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("stats skills"),
            )
            return

        coverage = census.get("coverage")
        rows = census.get("rows")
        coverage = coverage if isinstance(coverage, dict) else {}
        rows = rows if isinstance(rows, list) else []
        successful_hosts = coverage.get("successful_hosts")
        successful_hosts = successful_hosts if isinstance(successful_hosts, list) else []
        expected_hosts = coverage.get("expected_hosts")
        expected_hosts = expected_hosts if isinstance(expected_hosts, list) else []
        typer.echo(f"scope: {coverage.get('scope', 'unknown')}")
        typer.echo(f"hosts: {len(successful_hosts)}/{len(expected_hosts)}")
        typer.echo(
            f"sessions: {coverage.get('considered_sessions', 0)}; "
            f"attributed: {coverage.get('attributed_invocations', 0)}; "
            f"unattributed candidates: {coverage.get('unattributed_candidates', 0)}"
        )
        if not rows:
            typer.echo("No attributed skill invocations in range.")
        for row in rows:
            if not isinstance(row, dict):
                continue
            typer.echo(
                f"{row.get('skill_name', '?')}  {row.get('source', '?')}  "
                f"{row.get('host', '?')}  {row.get('invocations', 0)} invocations  "
                f"{row.get('sessions', 0)} sessions"
            )
        render_cta_hints(ctas)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None
