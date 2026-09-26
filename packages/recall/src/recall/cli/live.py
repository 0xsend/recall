"""`recall live` — sessions an agent is running right now (REQ-LIVE-002)."""

from __future__ import annotations

from typing import Any, cast

import typer

from recall.cli.contract import (
    CliError,
    ErrorCode,
    OutputFormat,
    emit_data,
    emit_error,
    parse_fields,
    parse_params,
    project_fields,
    resolve_bool_param,
    resolve_int_param,
    resolve_optional_int_param,
    resolve_output_format_early,
    resolve_output_format_from_params,
    resolve_param,
    validate_source,
    validate_structured_fields,
)
from recall.cli.manifest import output_fields_for
from recall.cli.rpc import rpc_call_or_error
from recall.cli.utils import format_datetime, stale_after_fresh_note

app = typer.Typer(
    help=(
        "Inspect current agent activity, turn state, and index freshness.\n\n"
        "The default roster is active sessions observed in watch or poll mode."
        " Use --all for recent idle and ended sessions. Structured output preserves coverage"
        " and next_cursor; --fields projects"
        " session fields inside that envelope. Use --cursor to continue a local page.\n\n"
        "Examples:\n\n"
        "  recall live --project /path/to/repo --json\n\n"
        "  recall live --all --json\n\n"
        "Then read a selected session: recall show <session-id> --tail 20 --fresh --json"
    ),
    short_help="Inspect current agent activity and freshness.",
    invoke_without_command=True,
)

NOT_WATCHING_NOTE = (
    "note: the daemon is not running a live watcher; activity comes from periodic"
    " source observations"
)


@app.callback(invoke_without_command=True)
def command(
    ctx: typer.Context,
    all_sessions: bool = typer.Option(
        False, "--all", help="Also include recent idle and ended sessions (default window 24 h)"
    ),
    source: str | None = typer.Option(
        None, "--source", help="claude-code, codex, pi-agent, grok, kimi-code"
    ),
    project: str | None = typer.Option(None, "--project", help="Filter by git repo path"),
    host: str | None = typer.Option(
        None, "--host", help="Filter by session host label (exact match)"
    ),
    limit: int = typer.Option(50, "--limit", help="Maximum sessions to return"),
    cursor: str | None = typer.Option(None, "--cursor", help="Continue a local live page"),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Catch up already-indexed rows on this page; does not first-index",
    ),
    fleet: bool = typer.Option(
        False, "--fleet", help="Fan-out to fleet.toml hosts and merge live sessions"
    ),
    fleet_config: str | None = typer.Option(
        None, "--fleet-config", help="Fleet inventory path (default: ~/.config/recall/fleet.toml)"
    ),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(
        None, "--fields", help="Project session fields inside the coverage envelope"
    ),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
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
            allowed={
                "all",
                "source",
                "project",
                "host",
                "limit",
                "cursor",
                "fresh",
                "fleet",
                "fleet_config",
                "fields",
                "format",
                "json",
            },
            types={
                "all": "boolean",
                "fresh": "boolean",
                "source": "string",
                "project": "string",
                "host": "string",
                "limit": "integer",
                "cursor": "string",
                "fleet": "boolean",
                "fleet_config": "string",
                "fields": "string",
                "format": "string",
                "json": "boolean",
            },
        )
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        all_sessions = resolve_bool_param(ctx, payload, "all", all_sessions)
        fresh = resolve_bool_param(ctx, payload, "fresh", fresh)
        source = resolve_param(ctx, payload, "source", source)
        validate_source(source)
        project = resolve_param(ctx, payload, "project", project)
        host = resolve_param(ctx, payload, "host", host)
        fleet = resolve_bool_param(ctx, payload, "fleet", fleet)
        fleet_config = resolve_param(ctx, payload, "fleet_config", fleet_config)
        limit = resolve_int_param(ctx, payload, "limit", limit)
        if not 1 <= limit <= 256:
            raise ValueError("limit must be 1..256")
        cursor = resolve_param(ctx, payload, "cursor", cursor)
        if cursor and fleet:
            raise ValueError("--cursor is local; continue each fleet host using its next_cursor")
        if fresh and fleet:
            # No remote is told to index: `recall live --fresh` is not forwarded
            # over the hop, because an edge on an older build would fail the
            # whole host on an unknown flag. Answering anyway would hand back
            # rows that are stale by construction under a flag that promises the
            # opposite, so the combination is refused instead.
            raise ValueError("--fresh cannot be combined with --fleet; no remote is asked to index")
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        if fleet:
            from recall.cli.fleet import run_fleet_live

            result = run_fleet_live(
                include_idle=all_sessions,
                source=source,
                project=project,
                host=host,
                limit=limit,
                fleet_config=fleet_config,
            )
            sessions = cast(list[dict[str, Any]], result["sessions"])
        else:
            rpc_params: dict[str, object] = {"limit": limit, "all": all_sessions}
            if cursor:
                rpc_params["cursor"] = cursor
            if source:
                rpc_params["source"] = source
            if project:
                rpc_params["project"] = project
            if host:
                rpc_params["host"] = host
            if fresh:
                rpc_params["fresh"] = True

            result = cast(dict[str, Any], rpc_call_or_error("recall.live", rpc_params))
            sessions = cast(list[dict[str, Any]], result.get("sessions", []))
            if not result.get("watching", False):
                typer.echo(NOT_WATCHING_NOTE, err=True)
            if fresh:
                note = stale_after_fresh_note(sessions)
                if note is not None:
                    typer.echo(note, err=True)

        if result.get("next_cursor"):
            typer.echo(
                f"note: more sessions; continue with --cursor {result['next_cursor']}", err=True
            )
        coverage = result.get("coverage")
        if not isinstance(coverage, dict) or not coverage.get("complete", False):
            typer.echo(
                "note: coverage is incomplete; inspect coverage before inferring absence", err=True
            )
        if output_format != OutputFormat.TEXT:
            if fields_value is not None:
                result = {
                    **result,
                    "sessions": project_fields(
                        sessions, fields=fields_value, allowed_fields=output_fields_for("live")
                    ),
                }
            emit_data(result, output_format=output_format)
            return

        if not sessions:
            typer.echo("No matching sessions on this page.")
            return
        for session in sessions:
            typer.echo(_render_row(session))
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None


def _render_row(session: dict[str, Any]) -> str:
    turn = session.get("turn") or {}
    freshness = session.get("freshness") or {}
    running = turn.get("running_tool") or {}
    activity_raw = session.get("last_activity_at")
    activity = format_datetime(
        activity_raw if isinstance(activity_raw, (str, type(None))) else str(activity_raw)
    )
    parts = [
        f"[{activity}]",
        str(session.get("liveness", "unknown")),
        str(session.get("id") or "unindexed"),
        f"({session.get('source') or '?'})",
        str(session.get("git_repo") or session.get("cwd") or session.get("path")),
        f"turn={turn.get('state', 'unknown')}",
    ]
    if running.get("name"):
        parts.append(f"tool={running['name']}")
    lag_bytes = freshness.get("lag_bytes")
    parts.append("fresh" if freshness.get("current") else f"lag={lag_bytes if lag_bytes else '?'}")
    return " ".join(parts)


# A mark is enrichment (REQ-LIVE-008): the harness hook that writes one runs
# inside the agent's own session start, so a daemon that cannot take it must
# not turn into a failing hook. The command says what it could not do and
# exits 0.
MARK_SKIPPED_NOTE = "note: not marked ({reason}); liveness falls back to what recall can derive"


@app.command("mark")
def mark_command(
    ctx: typer.Context,
    session: str | None = typer.Option(
        None, "--session", help="The harness's own session id (source_session_id)"
    ),
    pid: int | None = typer.Option(
        None, "--pid", help="Process id of the agent writing this session"
    ),
    source: str | None = typer.Option(
        None, "--source", help="claude-code (default), codex, pi-agent, grok, kimi-code"
    ),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    """Record the actual writer PID observed for a harness session.

    Use the harness source session ID, not recall's generated ID. A mark adds
    process-exit evidence; it does not control the agent.
    Inspect marked in JSON: an unavailable daemon yields false with exit 0.

    Example:

      recall live mark --session <source-session-id> --pid <writer-pid>
      --source codex --json
    """
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params,
            allowed={"session", "pid", "source", "fields", "format", "json"},
            types={
                "session": "string",
                "pid": "integer",
                "source": "string",
                "fields": "string",
                "format": "string",
                "json": "boolean",
            },
        )
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        session = resolve_param(ctx, payload, "session", session)
        if not session:
            raise ValueError("--session is required")
        source = resolve_param(ctx, payload, "source", source)
        validate_source(source)
        pid = resolve_optional_int_param(ctx, payload, "pid", pid)
        if pid is None:
            raise ValueError("--pid is required")
        if pid <= 0:
            raise ValueError("pid must be positive")
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        rpc_params: dict[str, object] = {"session": session, "pid": pid}
        if source:
            rpc_params["source"] = source

        try:
            # No `auto_fork`: forking a daemon here would put a cold start, and
            # possibly an index pass, inside the agent's session start.
            result = cast(
                dict[str, Any],
                rpc_call_or_error("recall.live_mark", rpc_params, auto_fork=False),
            )
        except CliError as err:
            typer.echo(MARK_SKIPPED_NOTE.format(reason=err.message), err=True)
            result = {
                "marked": False,
                "source_session_id": session,
                "pid": pid,
                "reason": err.message,
            }

        if output_format != OutputFormat.TEXT:
            emit_data(
                result,
                output_format=output_format,
                fields=fields_value,
                allowed_fields=output_fields_for("live mark"),
            )
            return
        typer.echo(_render_mark(result))
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None


def _render_mark(result: dict[str, Any]) -> str:
    verb = "marked" if result.get("marked") else "not marked"
    return " ".join([verb, str(result.get("source_session_id")), f"pid={result.get('pid')}"])
