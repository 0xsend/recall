"""Fleet status diagnostic and read-only aggregate query adapters.

Imports services for client orchestration only (no local DuckDB). Allowed by
tests/test_cli/test_cli_import_lint.py for the same reason as daemon.py.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from recall.cli.contract import (
    CliError,
    ErrorCode,
    OutputFormat,
    emit_data,
    emit_error,
    resolve_output_format,
    resolve_output_format_early,
)
from recall.core.fleet import (
    FleetConfig,
    FleetConfigError,
    default_fleet_path,
    load_fleet_config,
)
from recall.core.host_text import HostMessage, HostText, format_host
from recall.core.types import Source
from recall.services.fleet_transport import fleet_status

app = typer.Typer(help="Diagnose inventory SSH routes to edge daemons")


def _config_error(error: FleetConfigError) -> CliError:
    return CliError(
        code=ErrorCode.VALIDATION,
        message=error.host_message.raw(),
        exit_code=2,
        host_message=error.host_message,
    )


def _load_inventory(fleet_config: str | None) -> FleetConfig:
    path = Path(fleet_config).expanduser() if fleet_config else default_fleet_path()
    try:
        config = load_fleet_config(path)
    except FleetConfigError as err:
        raise _config_error(err) from err
    return config


def _query(
    fleet_config: str | None,
    *,
    inventory_only: bool = False,
    host: str | None = None,
    census: bool = False,
):
    from recall.core.read_scope import bind_aggregate
    from recall.services.fleet_endpoints import FleetExecutor

    inventory = _load_inventory(fleet_config)
    try:
        executor = FleetExecutor(
            inventory, inventory_only=inventory_only, selector=host, census=census
        )
    except FleetConfigError as err:
        raise _config_error(err) from err
    bind_aggregate(executor.local_read)
    return executor


def _emit_skips(errors: list[dict[str, str]]) -> None:
    for err in errors:
        typer.echo(
            f"fleet: skip {format_host(err.get('name', 'None'))}: {err.get('error')}", err=True
        )


def _require_any_ok(hosts_ok: int, *, verb: str) -> None:
    if hosts_ok == 0:
        raise CliError(
            code=ErrorCode.RUNTIME,
            message=f"fleet {verb}: all hosts failed",
            exit_code=1,
        )


def run_fleet_stats_usage(
    *,
    since: str | None,
    fleet_config: str | None,
    fleet_inventory_only: bool = False,
) -> list[dict[str, object]]:
    """Resolved endpoint usage; emit skip warnings on stderr."""
    from recall.services.fleet_usage import fleet_stats_usage

    executor = _query(fleet_config, inventory_only=fleet_inventory_only)
    result = fleet_stats_usage(executor, since=since)
    _emit_skips(result.errors)
    _require_any_ok(result.hosts_ok, verb="stats usage")
    return result.rows  # type: ignore[return-value]


def run_fleet_stats_skills(
    *,
    since: str | None,
    sources: tuple[Source, ...],
    fleet_config: str | None,
) -> dict[str, object]:
    """Fail-closed endpoint query for the skill census."""
    from recall.services.fleet_skills import fleet_stats_skills

    executor = _query(fleet_config, census=True)
    result = fleet_stats_skills(
        executor,
        since=since,
        sources=sources,
    )
    if result.payload is None:
        _emit_skips(list(result.errors))
        parts: list[str | HostText] = ["fleet stats skills incomplete: "]
        for index, error in enumerate(result.errors):
            if index:
                parts.append("; ")
            parts.extend(
                [HostText(value=error.get("name", "None")), ": ", error.get("error", "None")]
            )
        message = HostMessage(parts=tuple(parts))
        raise CliError(
            code=ErrorCode.RUNTIME,
            message=message.raw(),
            exit_code=1,
            host_message=message,
        )
    return result.payload


def run_fleet_list(
    *,
    since: str | None,
    source: str | None,
    project: str | None,
    host: str | None,
    limit: int,
    fleet_config: str | None,
    fleet_inventory_only: bool = False,
) -> list[dict[str, object]]:
    from recall.services.fleet_query import fleet_list

    executor = _query(fleet_config, inventory_only=fleet_inventory_only, host=host)
    result = fleet_list(
        executor,
        since=since,
        source=source,
        project=project,
        limit=limit,
    )
    _emit_skips(result.errors)
    _require_any_ok(result.hosts_ok, verb="list")
    return result.rows  # type: ignore[return-value]


def run_fleet_live(
    *,
    include_idle: bool,
    source: str | None,
    project: str | None,
    host: str | None,
    limit: int,
    fleet_config: str | None,
    fleet_inventory_only: bool = False,
) -> dict[str, Any]:
    from recall.services.fleet_query import fleet_live

    executor = _query(fleet_config, inventory_only=fleet_inventory_only, host=host)
    result = fleet_live(
        executor,
        include_idle=include_idle,
        source=source,
        project=project,
        limit=limit,
    )
    _emit_skips(result.errors)
    _require_any_ok(result.hosts_ok, verb="live")
    return {
        "schema_version": 2,
        "watching": None,
        "sessions": result.rows,
        "next_cursor": None,
        "coverage": result.coverage,
    }


def run_fleet_search(
    *,
    query: str,
    tool: str | None,
    source: str | None,
    mode: str | None,
    limit: int,
    fleet_config: str | None,
    fleet_inventory_only: bool = False,
) -> list[dict[str, object]]:
    from recall.services.fleet_query import fleet_search

    executor = _query(fleet_config, inventory_only=fleet_inventory_only)
    result = fleet_search(
        executor,
        query=query,
        tool=tool,
        source=source,
        mode=mode,
        limit=limit,
    )
    _emit_skips(result.errors)
    _require_any_ok(result.hosts_ok, verb="search")
    return result.rows  # type: ignore[return-value]


def run_fleet_show(
    *,
    session_id: str,
    host: str | None,
    tools: bool,
    message_limit: int | None,
    fleet_config: str | None,
    fleet_inventory_only: bool = False,
) -> dict[str, object]:
    from recall.services.fleet_query import fleet_show

    executor = _query(fleet_config, inventory_only=fleet_inventory_only, host=host)
    result = fleet_show(
        executor,
        session_id=session_id,
        tools=tools,
        message_limit=message_limit,
    )
    _emit_skips(result.errors)
    if result.ambiguous_hosts:
        parts: list[str | HostText] = [f"session {session_id!r} found on multiple hosts ("]
        for index, name in enumerate(result.ambiguous_hosts):
            if index:
                parts.append(", ")
            parts.append(HostText(value=name))
        parts.append("); pass --host <name> to disambiguate")
        message = HostMessage(parts=tuple(parts))
        raise CliError(
            code=ErrorCode.VALIDATION,
            message=message.raw(),
            exit_code=2,
            host_message=message,
        )
    if result.session is None:
        raise CliError(
            code=ErrorCode.NOT_FOUND,
            message=f"session not found on fleet: {session_id}",
            exit_code=1,
        )
    return result.session  # type: ignore[return-value]


@app.command("status")
def status_command(
    ctx: typer.Context,
    config: str | None = typer.Option(
        None,
        "--config",
        help="Fleet inventory TOML (default: ~/.config/recall/fleet.toml)",
    ),
    format_name: str | None = typer.Option(
        None, "--format", help="Output format: auto, text, json, jsonl, toon"
    ),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """Probe each inventory host: reachability, binary/daemon version, drift."""
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=None,
    )
    try:
        output_format = resolve_output_format(json_output=json_output, format_name=format_name)
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None
    path = Path(config).expanduser() if config else default_fleet_path()
    try:
        fleet_config = load_fleet_config(path)
    except FleetConfigError as err:
        err_format = output_format if output_format != OutputFormat.TEXT else OutputFormat.JSON
        emit_error(
            _config_error(err),
            output_format=err_format,
        )
        raise typer.Exit(code=2) from None

    rows = fleet_status(fleet_config.hosts)
    payload = [asdict(row) for row in rows]

    if output_format != OutputFormat.TEXT:
        emit_data(payload, output_format=output_format)
        return

    if not fleet_config.hosts:
        typer.echo(f"No fleet hosts configured ({path}). Add [[host]] entries to fleet.toml.")
        return

    for row in rows:
        if row.ok:
            drift = (
                "drift"
                if row.version_drift is True
                else ("ok" if row.version_drift is False else "unknown")
            )
            daemon = row.daemon_version or "?"
            binary = row.binary_version or "?"
            line = f"{row.name}\t{row.ssh}\tok\tbinary={binary}\tdaemon={daemon}\t{drift}"
            if row.error:
                line += f"\tnote={row.error}"
            typer.echo(line)
        else:
            typer.echo(f"{row.name}\t{row.ssh}\tFAIL\t{row.error or 'unreachable'}")
