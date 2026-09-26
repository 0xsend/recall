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
    require_confirmation,
    resolve_bool_param,
    resolve_int_param,
    resolve_output_format_early,
    resolve_output_format_from_params,
    resolve_param,
    validate_structured_fields,
)
from recall.cli.manifest import output_fields_for
from recall.core.config import AppConfig
from recall.services.snapshots import (
    DEFAULT_GC_DAYS,
    SnapshotGcResult,
    SnapshotInventory,
    gc_snapshots,
    list_snapshots,
)

app = typer.Typer(help="Snapshot maintenance commands")

PARAM_TYPES = {
    "days": "integer",
    "dry_run": "boolean",
    "yes": "boolean",
    "fields": "string",
    "format": "string",
    "json": "boolean",
}

LIST_PARAM_TYPES = {
    "fields": "string",
    "format": "string",
    "json": "boolean",
}


@app.command("list")
def list_command(
    ctx: typer.Context,
    format_name: str | None = typer.Option(
        None,
        "--format",
        help="Output format: auto, text, json, jsonl, toon",
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    """Report the snapshot entries on disk with their size and age."""
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(params, allowed=set(LIST_PARAM_TYPES), types=LIST_PARAM_TYPES)
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        inventory = list_snapshots(AppConfig.load())
        if output_format == OutputFormat.TEXT:
            _emit_inventory_text(inventory)
            return
        emit_data_with_cta(
            inventory,
            [],
            output_format=output_format,
            include_cta=False,
            fields=fields_value,
            allowed_fields=output_fields_for("snapshots list"),
        )
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None


@app.command("gc")
def gc_command(
    ctx: typer.Context,
    days: int = typer.Option(
        DEFAULT_GC_DAYS,
        "--days",
        help="Delete snapshot entries older than N days",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report what would be deleted without removing entries",
    ),
    yes: bool = typer.Option(False, "--yes", help="Confirm snapshot deletion"),
    format_name: str | None = typer.Option(
        None,
        "--format",
        help="Output format: auto, text, json, jsonl, toon",
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(params, allowed=set(PARAM_TYPES), types=PARAM_TYPES)
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        days = resolve_int_param(ctx, payload, "days", days)
        if days < 0:
            raise ValueError("days must be non-negative")
        dry_run = resolve_bool_param(ctx, payload, "dry_run", dry_run)
        yes = resolve_bool_param(ctx, payload, "yes", yes)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        require_confirmation(
            should_confirm=True,
            confirmed=yes,
            dry_run=dry_run,
            message="`recall snapshots gc` deletes stale snapshot artifacts.",
        )

        result = gc_snapshots(AppConfig.load(), days=days, dry_run=dry_run)
        _emit_result(result, output_format=output_format, fields=fields_value)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None


def _format_bytes(total: int) -> str:
    size = float(total)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable: the GiB branch always returns")


def _format_age(age_seconds: float) -> str:
    if age_seconds < 3600:
        return f"{int(age_seconds // 60)}m"
    if age_seconds < 86_400:
        return f"{int(age_seconds // 3600)}h"
    return f"{int(age_seconds // 86_400)}d"


def _emit_inventory_text(inventory: SnapshotInventory) -> None:
    if inventory.snapshots_dir_missing:
        typer.echo(f"Snapshots directory: missing ({inventory.snapshots_dir})")
        return
    noun = "entry" if inventory.entry_count == 1 else "entries"
    typer.echo(
        f"Snapshots: {inventory.entry_count} {noun}, "
        f"{_format_bytes(inventory.total_bytes)} in {inventory.snapshots_dir}"
    )
    for entry in inventory.entries:
        # `retained` and `partial` are the two reasons `snapshots gc` leaves an
        # entry alone, so naming them here answers "why is this still here".
        reasons = (("retained", entry.retained), ("partial", entry.partial))
        notes = [note for note, applies in reasons if applies]
        suffix = f" ({', '.join(notes)})" if notes else ""
        typer.echo(
            f"  {', '.join(entry.paths)}  "
            f"{_format_bytes(entry.total_bytes)}, {_format_age(entry.age_seconds)} old{suffix}"
        )
    if inventory.unreadable_paths:
        typer.echo("Unreadable paths:")
        for path in inventory.unreadable_paths:
            typer.echo(f"  - {path}")


def _emit_result(
    result: SnapshotGcResult,
    *,
    output_format: OutputFormat,
    fields: tuple[str, ...] | None,
) -> None:
    if output_format == OutputFormat.TEXT:
        _emit_text(result)
        return
    emit_data_with_cta(
        result,
        [],
        output_format=output_format,
        include_cta=False,
        fields=fields,
        allowed_fields=output_fields_for("snapshots gc"),
    )


def _emit_text(result: SnapshotGcResult) -> None:
    if result.snapshots_dir_missing:
        typer.echo("Snapshots directory: missing")
    typer.echo(f"Removed paths: {len(result.removed_paths)}")
    typer.echo(f"Kept paths: {len(result.kept_paths)}")
    typer.echo(f"Failed: {len(result.failed_paths)}")
    typer.echo(f"Partial snapshots: {len(result.partial_paths)}")
    typer.echo(f"Total bytes freed: {result.total_bytes_freed}")
    if result.dry_run:
        typer.echo("Dry run: true")
    if result.partial_paths:
        typer.echo("Partial snapshot paths:")
        for path in result.partial_paths[:5]:
            typer.echo(f"  - {path}")
        if len(result.partial_paths) > 5:
            typer.echo(f"  ... and {len(result.partial_paths) - 5} more")
    if result.failed_paths:
        typer.echo("Failed paths:")
        for path in result.failed_paths[:5]:
            typer.echo(f"  - {path}")
        if len(result.failed_paths) > 5:
            typer.echo(f"  ... and {len(result.failed_paths) - 5} more")
