"""Fleet commands: status probe and (later) fan-out query verbs.

Imports services for SSH fan-out only (no local DuckDB). Allowed by
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
    resolve_output_format_early,
)
from recall.core.fleet import (
    FleetConfig,
    FleetConfigError,
    default_fleet_path,
    load_fleet_config,
)
from recall.core.types import Source
from recall.services.fleet_transport import fleet_status

app = typer.Typer(help="Fleet query fan-out (read-only SSH to edge daemons)")


def _load_inventory(fleet_config: str | None) -> FleetConfig:
    path = Path(fleet_config).expanduser() if fleet_config else default_fleet_path()
    try:
        config = load_fleet_config(path)
    except FleetConfigError as err:
        raise CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2) from err
    if not config.hosts:
        raise CliError(
            code=ErrorCode.VALIDATION,
            message=(
                f"no fleet hosts configured ({path}); add [[host]] name/ssh entries to fleet.toml"
            ),
            exit_code=2,
        )
    return config


def _emit_skips(errors: list[dict[str, str]]) -> None:
    for err in errors:
        typer.echo(f"fleet: skip {err.get('name')}: {err.get('error')}", err=True)


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
) -> list[dict[str, object]]:
    """SSH fan-out for stats usage; emit skip warnings on stderr."""
    from recall.services.fleet_usage import fleet_stats_usage

    config = _load_inventory(fleet_config)
    result = fleet_stats_usage(config.hosts, since=since)
    _emit_skips(result.errors)
    _require_any_ok(result.hosts_ok, verb="stats usage")
    return result.rows  # type: ignore[return-value]


def run_fleet_stats_skills(
    *,
    local_payload: dict[str, object],
    since: str | None,
    sources: tuple[Source, ...],
    fleet_config: str | None,
) -> dict[str, object]:
    """Fail-closed SSH fan-out for the skill census."""
    from recall.services.fleet_skills import fleet_stats_skills

    config = _load_inventory(fleet_config)
    result = fleet_stats_skills(
        config.hosts,
        local_payload=local_payload,
        since=since,
        sources=sources,
    )
    if result.payload is None:
        detail = "; ".join(f"{error.get('name')}: {error.get('error')}" for error in result.errors)
        raise CliError(
            code=ErrorCode.RUNTIME,
            message=f"fleet stats skills incomplete: {detail}",
            exit_code=1,
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
) -> list[dict[str, object]]:
    from recall.services.fleet_query import fleet_list

    config = _load_inventory(fleet_config)
    result = fleet_list(
        config.hosts,
        since=since,
        source=source,
        project=project,
        host_filter=host,
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
) -> dict[str, Any]:
    from recall.services.fleet_query import fleet_live

    config = _load_inventory(fleet_config)
    result = fleet_live(
        config.hosts,
        include_idle=include_idle,
        source=source,
        project=project,
        host_filter=host,
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
) -> list[dict[str, object]]:
    from recall.services.fleet_query import fleet_search

    config = _load_inventory(fleet_config)
    result = fleet_search(
        config.hosts,
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
) -> dict[str, object]:
    from recall.services.fleet_query import fleet_show

    config = _load_inventory(fleet_config)
    result = fleet_show(
        config.hosts,
        session_id=session_id,
        host_name=host,
        tools=tools,
        message_limit=message_limit,
    )
    _emit_skips(result.errors)
    if result.ambiguous_hosts:
        names = ", ".join(result.ambiguous_hosts)
        raise CliError(
            code=ErrorCode.VALIDATION,
            message=(
                f"session {session_id!r} found on multiple hosts ({names}); "
                "pass --host <name> to disambiguate"
            ),
            exit_code=2,
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
    path = Path(config).expanduser() if config else default_fleet_path()
    try:
        fleet_config = load_fleet_config(path)
    except FleetConfigError as err:
        err_format = output_format if output_format != OutputFormat.TEXT else OutputFormat.JSON
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
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
