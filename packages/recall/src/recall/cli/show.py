from __future__ import annotations

import json
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
    resolve_optional_float_param,
    resolve_optional_int_param,
    resolve_output_format_early,
    resolve_output_format_from_params,
    resolve_param,
    validate_structured_fields,
)
from recall.cli.cta import cta_for_show
from recall.cli.manifest import output_fields_for
from recall.cli.rpc import rpc_call_or_error
from recall.cli.utils import format_datetime, stale_after_fresh_note

# A follower is read line by line — `while read line; do ... done` — so the
# stream is NDJSON whatever `--format` says. A TOON table or a pretty-printed
# JSON array would never terminate a line-oriented reader.
NOT_WATCHING_NOTE = (
    "note: the daemon is not running a live watcher for this session; follow"
    " receives committed updates from periodic reconciliation"
)


# Kept in step with `rpc_server.DEFAULT_FOLLOW_TIMEOUT`; the CLI cannot import
# the service layer (REQ-RPC-004), so it carries its own copy of the default it
# has to size the socket's idle bound against.
DEFAULT_TIMEOUT = 60.0
FOLLOW_IDLE_MARGIN = 30.0


def _emit_ndjson(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, separators=(",", ":"), default=str))


# What a summary row keeps: enough to locate the turn and re-read it, plus the
# signal that thinking exists. Everything else on a payload-free turn is either
# null or constant across the read, so it is weight without information.
_SUMMARY_FIELDS = ("id", "idx", "role", "timestamp", "has_thinking")


def _omission_summary(message: dict[str, Any], *, tools: bool, thinking: bool) -> str:
    """Name what this read withheld, and the flag that would reveal it."""
    omitted: list[str] = []
    if not tools:
        omitted.append("tool calls (--tools)")
    if not thinking and message.get("has_thinking"):
        omitted.append("thinking (--thinking)")
    if not omitted:
        return "no message text, thinking, or tool calls"
    return "no message text; omitted: " + ", ".join(omitted)


def _view_message(message: dict[str, Any], *, tools: bool, thinking: bool) -> dict[str, Any]:
    """Project one message onto what the caller asked to see (REQ-CLI-024)."""
    if not thinking:
        message = {
            key: value
            for key, value in message.items()
            if key not in {"thinking", "thinking_embedding"}
        }
    tool_calls = message.get("tool_calls")
    tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
    if message.get("content") or message.get("thinking") or tool_call_count:
        return message
    summary = {key: message[key] for key in _SUMMARY_FIELDS if key in message}
    if tools:
        # Only a read that asked for tool calls can honestly report their count.
        summary["tool_call_count"] = tool_call_count
    summary["summary"] = _omission_summary(message, tools=tools, thinking=thinking)
    return summary


def _view_messages(payload: dict[str, Any], *, tools: bool, thinking: bool) -> dict[str, Any]:
    """Apply the message view to a `show` response or one follow frame.

    A payload without a message list — a `closed` frame — passes through, so the
    single read and the stream share one projection.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload
    return {
        **payload,
        "messages": [
            _view_message(message, tools=tools, thinking=thinking)
            if isinstance(message, dict)
            else message
            for message in messages
        ],
    }


def _live_read_error(reason: str) -> CliError:
    return CliError(
        code=ErrorCode.RUNTIME,
        message=(
            f"The daemon response cannot satisfy the requested live read: {reason}."
            " Use a daemon built from the same recall revision as this client."
        ),
    )


def _validate_live_read_response(session: object, *, windowed: bool, tail: int | None) -> None:
    """An older daemon can accept and silently ignore unknown show parameters."""
    if not isinstance(session, dict):
        raise _live_read_error("missing session object")
    freshness = session.get("freshness")
    if not isinstance(freshness, dict) or not isinstance(freshness.get("current"), bool):
        raise _live_read_error("missing freshness observation")
    if not windowed:
        return
    cursor = session.get("cursor")
    if not isinstance(cursor, str) or not cursor:
        raise _live_read_error("missing continuation cursor")
    messages = session.get("messages")
    if not isinstance(messages, list):
        raise _live_read_error("missing message window")
    if tail is not None and len(messages) > tail:
        raise _live_read_error(f"returned {len(messages)} messages for a tail of {tail}")


def _run_follow(params: dict[str, object], *, tools: bool, thinking: bool) -> None:
    """Print each delta as it arrives, then the one closing line.

    The socket's idle bound is the stream's own deadline plus a margin: the
    daemon promises a closing frame by then, so silence past it means the daemon
    died rather than that the session went quiet. Without the bound a follower
    outlives the daemon it was watching.
    """
    deadline = params.get("timeout")
    idle_timeout = (float(deadline) if isinstance(deadline, (int, float)) else DEFAULT_TIMEOUT) + (
        FOLLOW_IDLE_MARGIN
    )
    closed = cast(
        dict[str, Any],
        rpc_call_or_error(
            "recall.show_follow",
            params,
            on_notification=lambda _method, payload: _emit_ndjson(
                _view_messages(payload, tools=tools, thinking=thinking)
            ),
            idle_timeout=idle_timeout,
        ),
    )
    _emit_ndjson(_view_messages(closed, tools=tools, thinking=thinking))
    if not closed.get("watching", False):
        typer.echo(NOT_WATCHING_NOTE, err=True)


def command(
    ctx: typer.Context,
    session_id: str | None = typer.Argument(None, help="Session ID"),
    tools: bool = typer.Option(False, "--tools", help="Include tool calls"),
    thinking: bool = typer.Option(False, "--thinking", help="Include thinking blocks"),
    message_limit: int | None = typer.Option(
        None,
        "--message-limit",
        help="Limit returned messages",
    ),
    host: str | None = typer.Option(
        None,
        "--host",
        help="Fleet host name when using --fleet (required if ambiguous)",
    ),
    tail: int | None = typer.Option(
        None, "--tail", help="Read only the last N messages instead of the whole session"
    ),
    after: str | None = typer.Option(
        None, "--after", help="Read only what landed after a cursor from an earlier read"
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="Have the daemon index pending bytes before answering"
    ),
    follow: bool = typer.Option(
        False, "--follow", help="Stream new messages as NDJSON until the deadline"
    ),
    timeout: float | None = typer.Option(
        None, "--timeout", help="Seconds to follow before closing the stream (default 60)"
    ),
    fleet: bool = typer.Option(False, "--fleet", help="Resolve session via fleet.toml hosts"),
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
    """Read current or historical session messages, with bounded live updates.

    Find current sessions with recall live; locate past work with list/search.
    Use --tail for a bounded read, then pass its opaque cursor to --after.
    Check freshness.current and stderr even with --fresh: a refresh can time out.
    Follow emits NDJSON deltas and a closed event; without --after it starts now.

    Examples:

      recall show <session-id> --tail 20 --fresh --json

      recall show <session-id> --after '<cursor>' --fresh --json

      recall show <session-id> --after '<cursor>' --follow --timeout 60
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
            allowed={
                "session_id",
                "tools",
                "thinking",
                "message_limit",
                "tail",
                "after",
                "fresh",
                "follow",
                "timeout",
                "host",
                "fleet",
                "fleet_config",
                "fields",
                "format",
                "json",
                "cta",
            },
            types={
                "tail": "integer",
                "after": "string",
                "fresh": "boolean",
                "follow": "boolean",
                "timeout": "number",
                "session_id": "string",
                "tools": "boolean",
                "thinking": "boolean",
                "message_limit": "integer",
                "host": "string",
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
        session_id = resolve_param(ctx, payload, "session_id", session_id)
        tools = resolve_bool_param(ctx, payload, "tools", tools)
        thinking = resolve_bool_param(ctx, payload, "thinking", thinking)
        fresh = resolve_bool_param(ctx, payload, "fresh", fresh)
        follow = resolve_bool_param(ctx, payload, "follow", follow)
        timeout = resolve_optional_float_param(ctx, payload, "timeout", timeout)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        host = resolve_param(ctx, payload, "host", host)
        fleet = resolve_bool_param(ctx, payload, "fleet", fleet)
        fleet_config = resolve_param(ctx, payload, "fleet_config", fleet_config)
        message_limit = resolve_optional_int_param(ctx, payload, "message_limit", message_limit)
        if message_limit is not None and message_limit <= 0:
            raise ValueError("message_limit must be positive")
        tail = resolve_optional_int_param(ctx, payload, "tail", tail)
        after = resolve_param(ctx, payload, "after", after)
        if tail is not None and tail <= 0:
            raise ValueError("tail must be positive")
        windowed = tail is not None or after is not None
        if windowed and message_limit is not None:
            raise ValueError(
                "--message-limit reads from the start of the session;"
                " --tail and --after read from the end. Use one or the other."
            )
        if windowed and fleet:
            raise ValueError("--tail and --after are not available with --fleet")
        if follow and fleet:
            raise ValueError("--follow is not available with --fleet")
        if follow and tail is not None:
            raise ValueError("--follow starts from --after or from now; it has no tail window")
        if timeout is not None and not follow:
            raise ValueError("--timeout only applies to --follow")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        if not session_id:
            raise CliError(
                code=ErrorCode.VALIDATION,
                message="session_id is required",
                details={"field": "session_id"},
                exit_code=2,
            )

        if follow:
            follow_params: dict[str, object] = {"session_id": session_id, "tools": tools}
            if after:
                follow_params["after"] = after
            if timeout is not None:
                follow_params["timeout"] = timeout
            _run_follow(follow_params, tools=tools, thinking=thinking)
            return

        if fleet:
            from recall.cli.fleet import run_fleet_show

            session_raw: Any = run_fleet_show(
                session_id=session_id,
                host=host,
                tools=tools,
                message_limit=message_limit,
                fleet_config=fleet_config,
            )
        else:
            rpc_params: dict[str, object] = {"session_id": session_id, "tools": tools}
            if message_limit is not None:
                rpc_params["message_limit"] = message_limit
            if tail is not None:
                rpc_params["tail"] = tail
            if after:
                rpc_params["after"] = after
            if fresh:
                rpc_params["fresh"] = True
            session_raw = rpc_call_or_error("recall.show", rpc_params)
        if windowed or fresh:
            _validate_live_read_response(session_raw, windowed=windowed, tail=tail)
        # Validation reads the daemon's answer; rendering reads what was asked
        # for. The view runs after, so a withheld field never fails the check.
        session = _view_messages(cast(dict[str, Any], session_raw), tools=tools, thinking=thinking)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        err_msg = str(err)
        code = (
            ErrorCode.NOT_FOUND
            if err_msg.startswith(("session not found", "session not indexed"))
            else ErrorCode.VALIDATION
        )
        exit_code = 1 if code == ErrorCode.NOT_FOUND else 2
        emit_error(
            CliError(code=code, message=str(err), exit_code=exit_code),
            output_format=output_format,
        )
        raise typer.Exit(code=exit_code) from None

    if session.get("cursor_reset"):
        typer.echo(
            "note: cursor reset after source content was rewritten; this is a replacement window",
            err=True,
        )
    if fresh:
        note = stale_after_fresh_note([session])
        if note is not None:
            typer.echo(note, err=True)

    ctas = cta_for_show(session) if (cta or output_format == OutputFormat.TEXT) else []

    if output_format != OutputFormat.TEXT:
        try:
            emit_data_with_cta(
                session,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("show"),
            )
        except CliError as err:
            emit_error(err, output_format=output_format)
            raise typer.Exit(code=err.exit_code) from None
        return

    started_raw = session.get("started_at")
    started = format_datetime(
        started_raw if isinstance(started_raw, (str, type(None))) else str(started_raw)
    )
    sid = session.get("id", "?")
    source = session.get("source", "?")
    typer.echo(f"[{started}] Session {sid} ({source})")
    if session.get("git_repo") or session.get("cwd"):
        typer.echo(f"Project: {session.get('git_repo') or session.get('cwd')}")
    duration = session.get("duration_seconds") or 0
    msg_count = session.get("message_count", 0)
    tool_count = session.get("tool_count", 0)
    typer.echo(f"Duration: {duration}s | Messages: {msg_count} | Tools: {tool_count}")
    typer.echo("")

    messages = session.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        ts_raw = message.get("timestamp")
        timestamp = format_datetime(
            ts_raw if isinstance(ts_raw, (str, type(None))) else str(ts_raw)
        )
        role = message.get("role", "unknown")
        summary = message.get("summary")
        if isinstance(summary, str):
            # A collapsed turn is one line and no trailing blank, so a run of
            # them reads as a block rather than as a wall of empty headers.
            typer.echo(f"[{timestamp}] {role}: {summary}")
            continue
        typer.echo(f"[{timestamp}] {role}:")
        if message.get("content"):
            typer.echo(message["content"])
        # The view above already withheld thinking unless it was asked for.
        if message.get("thinking"):
            typer.echo("[thinking]")
            typer.echo(message["thinking"])
        tool_calls = message.get("tool_calls") if tools else None
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                label = tool_call.get("tool_name", "tool")
                detail = tool_call.get("bash_command") or ""
                typer.echo(f"  [{label}] {detail}")
        typer.echo("")

    orphans = session.get("orphan_tool_calls") if tools else None
    if isinstance(orphans, list) and orphans:
        typer.echo("Orphan tool calls:")
        for tool_call in orphans:
            if not isinstance(tool_call, dict):
                continue
            label = tool_call.get("tool_name", "tool")
            detail = tool_call.get("bash_command") or ""
            typer.echo(f"  [{label}] {detail}")

    render_cta_hints(ctas)
