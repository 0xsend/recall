from __future__ import annotations

from typing import Any, cast

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
from recall.cli.cta import cta_for_list
from recall.cli.manifest import output_fields_for
from recall.cli.rpc import rpc_call_or_error
from recall.cli.status_notices import render_status_notice
from recall.cli.utils import format_datetime


def command(
    ctx: typer.Context,
    source: str | None = typer.Option(
        None, "--source", help="claude-code, codex, pi-agent, grok, kimi-code"
    ),
    since: str | None = typer.Option(None, "--since", help="Time window (7d, 24h, 2024-01-01)"),
    project: str | None = typer.Option(None, "--project", help="Filter by git repo path"),
    host: str | None = typer.Option(
        None, "--host", help="Filter by session host label (exact match)"
    ),
    fleet: bool = typer.Option(
        False, "--fleet", help="Fan-out to fleet.toml hosts and merge sessions"
    ),
    fleet_config: str | None = typer.Option(
        None, "--fleet-config", help="Fleet inventory path (default: ~/.config/recall/fleet.toml)"
    ),
    limit: int = typer.Option(50, "--limit", help="Maximum sessions to return"),
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
                "source",
                "since",
                "project",
                "host",
                "fleet",
                "fleet_config",
                "limit",
                "fields",
                "format",
                "json",
                "cta",
            },
            types={
                "source": "string",
                "since": "string",
                "project": "string",
                "host": "string",
                "fleet": "boolean",
                "fleet_config": "string",
                "limit": "integer",
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
        source = resolve_param(ctx, payload, "source", source)
        validate_source(source)
        since = resolve_param(ctx, payload, "since", since)
        project = resolve_param(ctx, payload, "project", project)
        host = resolve_param(ctx, payload, "host", host)
        fleet = resolve_bool_param(ctx, payload, "fleet", fleet)
        fleet_config = resolve_param(ctx, payload, "fleet_config", fleet_config)
        limit = resolve_int_param(ctx, payload, "limit", limit)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        if limit <= 0:
            raise ValueError("limit must be positive")
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        if fleet:
            from recall.cli.fleet import run_fleet_list

            sessions_raw: Any = run_fleet_list(
                since=since,
                source=source,
                project=project,
                host=host,
                limit=limit,
                fleet_config=fleet_config,
            )
        else:
            rpc_params: dict[str, object] = {"limit": limit}
            if source:
                rpc_params["source"] = source
            if since:
                rpc_params["since"] = since
            if project:
                rpc_params["project"] = project
            if host:
                rpc_params["host"] = host

            sessions_raw = rpc_call_or_error("recall.list", rpc_params)
        sessions = cast(list[dict[str, Any]], sessions_raw)
        ctas = cta_for_list(sessions) if (cta or output_format == OutputFormat.TEXT) else []

        if output_format != OutputFormat.TEXT:
            # Warnings-only on stderr so agents using --json still see health
            # issues (version drift, bloat) without the human index-freshness
            # line (REQ-CLI-021/022, REQ-CLI-012).
            health_notice = render_status_notice(include_informational=False)
            if health_notice is not None:
                typer.echo(health_notice, err=True)
            emit_data_with_cta(
                sessions,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("list"),
            )
            return

        notice = render_status_notice()
        if notice is not None:
            typer.echo(notice, err=True)

        if not sessions:
            typer.echo("No sessions found.")
            render_cta_hints(ctas)
            return

        for session in sessions:
            started_raw = session.get("started_at")
            started = format_datetime(
                started_raw if isinstance(started_raw, (str, type(None))) else str(started_raw)
            )
            project_label = session.get("git_repo") or session.get("cwd") or "unknown"
            sid = session.get("id", "?")
            src = session.get("source", "?")
            msg_count = session.get("message_count", 0)
            tool_count = session.get("tool_count", 0)
            typer.echo(
                f"[{started}] {sid} ({src}) {project_label} messages={msg_count} tools={tool_count}"
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
