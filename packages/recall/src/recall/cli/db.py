"""`recall db` -- local DuckDB index maintenance (REQ-RESIL-017, REQ-RESIL-018).

Like `compact`, these commands own local database access instead of routing
through the RPC client: `rebuild-indexes` must run while the daemon is stopped
(it needs the exclusive DuckDB lock), and `check-indexes` falls back to a
read-only local probe when no daemon is serving.
"""

from __future__ import annotations

from typing import Any

import duckdb
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
    resolve_int_param,
    resolve_output_format_early,
    resolve_output_format_from_params,
    resolve_param,
    validate_structured_fields,
)
from recall.cli.manifest import output_fields_for
from recall.cli.rpc import rpc_call_or_error
from recall.core.config import AppConfig
from recall.db import RecallLockError, advisory_lock, connect, connect_readonly, is_lock_conflict
from recall.db.maintenance import (
    PROBE_SAMPLE_DEFAULT,
    PROBE_SAMPLE_MAX,
    IndexRebuildResult,
    probe_index_divergence,
    rebuild_indexes,
)
from recall.services.moves import MovedDuplicate, apply_moved_duplicates, plan_moved_duplicates
from recall.services.self_repair import clear_fatal_memory

app = typer.Typer(help="Local database maintenance commands")

REBUILD_PARAM_TYPES = {
    "fields": "string",
    "format": "string",
    "json": "boolean",
}

CHECK_PARAM_TYPES = {
    "sample": "integer",
    "fields": "string",
    "format": "string",
    "json": "boolean",
}

SUPERSEDE_PARAM_TYPES = {
    "apply": "boolean",
    "fields": "string",
    "format": "string",
    "json": "boolean",
}

SUPERSEDE_DAEMON_MESSAGE = (
    "cannot supersede moved transcripts while another process holds the database "
    "(usually the recall daemon); run `recall daemon stop`, rerun this command, "
    "then `recall daemon start`"
)

DAEMON_HOLDS_DB_MESSAGE = (
    "cannot rebuild indexes while another process holds the database "
    "(usually the recall daemon); run `recall daemon stop`, rerun this command, "
    "then `recall daemon start`"
)

READONLY_BLOCKED_MESSAGE = (
    "cannot open the database read-only while another process holds it; "
    "if the daemon is running, `recall daemon status` carries its startup probe, "
    "or run `recall daemon stop` first"
)


@app.command("rebuild-indexes")
def rebuild_indexes_command(
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
    """Drop and recreate every ART index, healing schema drift (REQ-RESIL-017)."""
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(params, allowed=set(REBUILD_PARAM_TYPES), types=REBUILD_PARAM_TYPES)
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        result = _rebuild_indexes_locally(AppConfig.load())
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None

    if output_format == OutputFormat.TEXT:
        _emit_rebuild_text(result)
        return
    emit_data(
        _rebuild_payload(result),
        output_format=output_format,
        fields=fields_value,
        allowed_fields=output_fields_for("db rebuild-indexes"),
    )


@app.command("check-indexes")
def check_indexes_command(
    ctx: typer.Context,
    sample: int = typer.Option(
        PROBE_SAMPLE_DEFAULT,
        "--sample",
        help=f"Recent keys to probe per index (1-{PROBE_SAMPLE_MAX})",
    ),
    format_name: str | None = typer.Option(
        None,
        "--format",
        help="Output format: auto, text, json, jsonl, toon",
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    """Probe index/table divergence; exit 1 when any sampled key diverges (REQ-RESIL-018)."""
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(params, allowed=set(CHECK_PARAM_TYPES), types=CHECK_PARAM_TYPES)
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        sample = resolve_int_param(ctx, payload, "sample", sample)
        if not 1 <= sample <= PROBE_SAMPLE_MAX:
            raise ValueError(f"--sample must be between 1 and {PROBE_SAMPLE_MAX}, got {sample}")
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)

        report = _probe_indexes(sample)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None

    if output_format == OutputFormat.TEXT:
        _emit_probe_text(report)
    else:
        emit_data(
            report,
            output_format=output_format,
            fields=fields_value,
            allowed_fields=output_fields_for("db check-indexes"),
        )
    if _diverged_count(report) > 0:
        raise typer.Exit(code=1)


def _rebuild_indexes_locally(config: AppConfig) -> IndexRebuildResult:
    """Rebuild under the advisory lock and the exclusive DuckDB lock.

    Both locks are held by a running daemon; either failing means the daemon
    (or another indexer) owns the database, and the operator must stop it
    rather than have this command race the writer.
    """
    # CliError is a frozen dataclass; it must be raised outside the
    # `advisory_lock` context manager, whose generator-based __exit__ assigns
    # `exc.__traceback__` and would trip FrozenInstanceError on the way out.
    try:
        with advisory_lock(config.lock_path):
            conn = connect(config, lenient_schema=True)
            try:
                result = rebuild_indexes(conn)
                clear_fatal_memory(config, conn)
            finally:
                conn.close()
    except RecallLockError as err:
        raise CliError(
            code=ErrorCode.RUNTIME, message=DAEMON_HOLDS_DB_MESSAGE, exit_code=2
        ) from err
    except duckdb.IOException as err:
        if is_lock_conflict(err):
            raise CliError(
                code=ErrorCode.RUNTIME, message=DAEMON_HOLDS_DB_MESSAGE, exit_code=2
            ) from None
        raise CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=2) from err
    return result


def _probe_indexes(sample: int) -> dict[str, Any]:
    """Prefer the daemon's shared connection; fall back to a local read-only probe."""
    try:
        report = rpc_call_or_error("recall.check_indexes", {"sample": sample}, auto_fork=False)
    except CliError:
        return _probe_indexes_locally(AppConfig.load(), sample)
    if not isinstance(report, dict):
        raise CliError(
            code=ErrorCode.RUNTIME,
            message=f"daemon returned an unexpected check_indexes payload: {type(report).__name__}",
            exit_code=2,
        )
    return report


def _probe_indexes_locally(config: AppConfig, sample: int) -> dict[str, Any]:
    try:
        conn = connect_readonly(config)
    except duckdb.IOException as err:
        if is_lock_conflict(err):
            raise CliError(
                code=ErrorCode.RUNTIME, message=READONLY_BLOCKED_MESSAGE, exit_code=2
            ) from None
        raise CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=2) from err
    try:
        return probe_index_divergence(conn, sample=sample).to_payload()
    finally:
        conn.close()


def _rebuild_payload(result: IndexRebuildResult) -> dict[str, Any]:
    return {
        "dropped": result.dropped,
        "created": result.created,
        "healed": list(result.healed),
        "elapsed_seconds": result.elapsed_seconds,
    }


def _emit_rebuild_text(result: IndexRebuildResult) -> None:
    typer.echo(
        f"Rebuilt {result.created} index(es): dropped {result.dropped}, "
        f"healed {len(result.healed)} in {result.elapsed_seconds:.2f}s"
    )
    for name in result.healed:
        typer.echo(f"  healed missing index: {name}")


def _diverged_count(report: dict[str, Any]) -> int:
    count = report.get("diverged_count")
    if isinstance(count, int):
        return count
    diverged = report.get("diverged")
    return len(diverged) if isinstance(diverged, list) else 0


def _emit_probe_text(report: dict[str, Any]) -> None:
    completeness = "complete" if report.get("complete", True) else "incomplete (budget exhausted)"
    typer.echo(
        f"Index probe: {report.get('indexes_probed', 0)} index(es), "
        f"{report.get('samples_checked', 0)} key(s) checked, "
        f"{report.get('samples_unverifiable', 0)} unverifiable, {completeness} "
        f"in {float(report.get('elapsed_seconds', 0.0)):.2f}s"
    )
    diverged = report.get("diverged")
    if not isinstance(diverged, list) or not diverged:
        typer.echo("No index/table divergence detected.")
        return
    typer.echo(f"Diverged keys ({len(diverged)}):")
    for entry in diverged:
        if not isinstance(entry, dict):
            continue
        typer.echo(
            f"  {entry.get('table')}.{entry.get('column')} = {entry.get('key')!r}: "
            f"index={entry.get('index_count')} full_scan={entry.get('full_count')}"
        )
    typer.echo(
        "The ART index disagrees with the table. Run `recall daemon stop`, then "
        "`recall db rebuild-indexes`, then `recall daemon start`."
    )


@app.command("supersede-moved")
def supersede_moved_command(
    ctx: typer.Context,
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Remove the superseded rows; without it, only report them",
    ),
    format_name: str | None = typer.Option(
        None,
        "--format",
        help="Output format: auto, text, json, jsonl, toon",
    ),
    fields: str | None = typer.Option(None, "--fields", help="Project structured output fields"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    params: str | None = typer.Option(None, "--params", help="Raw JSON input payload"),
) -> None:
    """Remove rows for transcripts indexed again after their directory moved (REQ-INDEX-027)."""
    output_format = resolve_output_format_early(
        ctx,
        json_output=json_output,
        format_name=format_name,
        raw_params=params,
    )
    try:
        payload = parse_params(
            params, allowed=set(SUPERSEDE_PARAM_TYPES), types=SUPERSEDE_PARAM_TYPES
        )
        output_format = resolve_output_format_from_params(
            ctx,
            payload,
            json_output=json_output,
            format_name=format_name,
        )
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        apply_value = bool(resolve_param(ctx, payload, "apply", apply))
        if fields_value is not None:
            # Reject bad fields before any work is committed, not after.
            project_fields(
                {"applied": False, "superseded": 0, "moves": []},
                fields=fields_value,
                allowed_fields=output_fields_for("db supersede-moved"),
            )

        plans, superseded = _supersede_moved_locally(AppConfig.load(), apply=apply_value)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None

    result = {
        "applied": apply_value,
        "superseded": superseded,
        "moves": [
            {
                "predecessor_path": plan.predecessor_path,
                "successor_path": plan.successor_path,
            }
            for plan in plans
        ],
    }
    if output_format == OutputFormat.TEXT:
        verb = "Superseded" if apply_value else "Would supersede"
        typer.echo(f"{verb} {len(plans)} moved transcript row(s)")
        for plan in plans:
            typer.echo(f"  {plan.predecessor_path} -> {plan.successor_path}")
        return
    emit_data(
        result,
        output_format=output_format,
        fields=fields_value,
        allowed_fields=output_fields_for("db supersede-moved"),
    )


def _supersede_moved_locally(config: AppConfig, *, apply: bool) -> tuple[list[MovedDuplicate], int]:
    """Plan under the database locks; apply only when asked.

    Planning also takes the exclusive locks: a plan made beside a live writer
    could be stale by the time anyone acts on it.
    """
    from recall.core.types import default_session_host

    try:
        with advisory_lock(config.lock_path):
            conn = connect(config)
            try:
                plans = plan_moved_duplicates(conn, host=default_session_host())
                superseded = 0
                if apply:
                    superseded = apply_moved_duplicates(conn, plans, queue_sidecar_deletes=False)
            finally:
                conn.close()
    except RecallLockError as err:
        raise CliError(
            code=ErrorCode.RUNTIME, message=SUPERSEDE_DAEMON_MESSAGE, exit_code=2
        ) from err
    except duckdb.IOException as err:
        if is_lock_conflict(err):
            raise CliError(
                code=ErrorCode.RUNTIME, message=SUPERSEDE_DAEMON_MESSAGE, exit_code=2
            ) from None
        raise CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=2) from err
    except duckdb.Error as err:
        raise CliError(code=ErrorCode.RUNTIME, message=str(err), exit_code=2) from err
    return plans, superseded
