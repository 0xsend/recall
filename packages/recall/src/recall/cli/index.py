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
    render_dry_run,
    require_confirmation,
    resolve_bool_param,
    resolve_optional_bool_param,
    resolve_output_format_early,
    resolve_output_format_from_params,
    resolve_param,
    validate_source,
    validate_structured_fields,
)
from recall.cli.cta import cta_for_dry_run, cta_for_index
from recall.cli.manifest import COMMANDS, output_fields_for
from recall.cli.rpc import rpc_call_or_error, stderr_progress, structured_progress


def command(
    ctx: typer.Context,
    full: bool = typer.Option(
        False,
        "--full",
        help=(
            "Force full reindex. On an llm-local/llm-remote host pair this with "
            "--context template, or every message without stored LLM context is summarized"
        ),
    ),
    source: str | None = typer.Option(
        None, "--source", help="claude-code, codex, pi-agent, grok, kimi-code"
    ),
    recreate: bool = typer.Option(
        False,
        "--recreate",
        help="Backup and rebuild database (destructive, requires --yes or confirmation)",
    ),
    yes: bool = typer.Option(False, "--yes", help="Confirm destructive operations"),
    embed: bool | None = typer.Option(
        None,
        "--embed/--no-embed",
        help="Generate embeddings during indexing",
    ),
    workers: str = typer.Option(
        "auto",
        "--workers",
        help="Changed-file preparation workers or 'auto'",
    ),
    context: str | None = typer.Option(
        None,
        "--context",
        help=(
            "Context mode for this run: off, template, llm-local, llm-remote, llm-codex "
            "(overrides config)"
        ),
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help="Limit to sessions whose git repo path matches (chunk a --full backfill)",
    ),
    recompute_context: bool = typer.Option(
        False,
        "--recompute-context",
        help="Rebuild stored context for existing CONTENT/THINKING rows without reparsing JSONL",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help=(
            "Limit to transcripts modified since e.g. 30d, 12h "
            "(scopes --full and --recompute-context)"
        ),
    ),
    only_mode: str | None = typer.Option(
        None,
        "--only-mode",
        help=(
            "Only recompute rows currently marked: off, template, llm-local, llm-remote, llm-codex"
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        # Rich renders square brackets as markup, so the TOML table is named in
        # dotted form rather than as `[daemon] log_level`.
        help=(
            "No effect: the daemon runs the index and keeps its own level. "
            "Set daemon.log_level in config.toml or RECALL_DAEMON_LOG_LEVEL, "
            "then restart the daemon"
        ),
    ),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show indexing progress"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and print the resolved request",
    ),
    root: str | None = typer.Option(
        None,
        "--root",
        help="Alternate home directory for multi-host ingest (layout like $HOME)",
    ),
    host: str | None = typer.Option(
        None,
        "--host",
        help="Host label stored on sessions (default: short hostname; basename of --root)",
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
    fields_value: tuple[str, ...] | None = None
    try:
        payload = parse_params(
            params,
            allowed={
                "full",
                "source",
                "recreate",
                "yes",
                "embed",
                "workers",
                "context",
                "project",
                "recompute_context",
                "since",
                "only_mode",
                "verbose",
                "progress",
                "dry_run",
                "root",
                "host",
                "format",
                "fields",
                "json",
                "cta",
            },
            types={
                "full": "boolean",
                "source": "string",
                "recreate": "boolean",
                "yes": "boolean",
                "embed": "boolean",
                "workers": "string",
                "context": "string",
                "project": "string",
                "recompute_context": "boolean",
                "since": "string",
                "only_mode": "string",
                "verbose": "boolean",
                "progress": "boolean",
                "dry_run": "boolean",
                "root": "string",
                "host": "string",
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
        full = resolve_bool_param(ctx, payload, "full", full)
        source = resolve_param(ctx, payload, "source", source)
        validate_source(source)
        recreate = resolve_bool_param(ctx, payload, "recreate", recreate)
        yes = resolve_bool_param(ctx, payload, "yes", yes)
        embed = resolve_optional_bool_param(ctx, payload, "embed", embed)
        workers = str(resolve_param(ctx, payload, "workers", workers))
        context = resolve_param(ctx, payload, "context", context)
        project = resolve_param(ctx, payload, "project", project)
        recompute_context = resolve_bool_param(ctx, payload, "recompute_context", recompute_context)
        since = resolve_param(ctx, payload, "since", since)
        only_mode = resolve_param(ctx, payload, "only_mode", only_mode)
        verbose = resolve_bool_param(ctx, payload, "verbose", verbose)
        progress = resolve_bool_param(ctx, payload, "progress", progress)
        dry_run = resolve_bool_param(ctx, payload, "dry_run", dry_run)
        root = resolve_param(ctx, payload, "root", root)
        host = resolve_param(ctx, payload, "host", host)
        cta = resolve_bool_param(ctx, payload, "cta", cta)
        fields_value = parse_fields(resolve_param(ctx, payload, "fields", fields))
        validate_structured_fields(output_format=output_format, fields=fields_value)
        if root and not host:
            # REQ-MULTIHOST-002: default host to basename of --root
            from pathlib import Path as _Path

            host = _Path(root).expanduser().resolve().name or "remote"
        require_confirmation(
            should_confirm=recreate,
            confirmed=yes,
            dry_run=dry_run,
            message="`recall index --recreate` rebuilds the live database.",
        )

        if dry_run:
            render_dry_run(
                command="index",
                request={
                    "full": full,
                    "source": source,
                    "recreate": recreate,
                    "yes": yes,
                    "embed": embed,
                    "workers": workers,
                    "context": context,
                    "project": project,
                    "recompute_context": recompute_context,
                    "since": since,
                    "only_mode": only_mode,
                    "verbose": verbose,
                    "progress": progress,
                    "root": root,
                    "host": host,
                },
                safety=COMMANDS["index"]["safety"],
                output_format=output_format,
                fields=fields_value,
                ctas=(
                    cta_for_dry_run("index")
                    if (cta or output_format == OutputFormat.TEXT)
                    else None
                ),
                include_cta=cta,
            )
            return

        rpc_params: dict[str, object] = {
            "full": full,
            "recreate": recreate,
            "verbose": verbose,
            "workers": workers,
        }
        if source:
            rpc_params["source"] = source
        if embed is not None:
            rpc_params["embed"] = embed
        if context is not None:
            rpc_params["context"] = context
        if project is not None:
            rpc_params["project"] = project
        if recompute_context:
            rpc_params["recompute_context"] = True
        if since is not None:
            rpc_params["since"] = since
        if only_mode is not None:
            rpc_params["only_mode"] = only_mode
        if root is not None:
            rpc_params["root"] = root
        if host is not None:
            rpc_params["host"] = host
        if recreate:
            rpc_params["confirmed"] = True

        if full and context is None:
            _warn_if_full_reparse_will_summarize()

        # Structured runs keep stdout for the payload and get bounded progress
        # on stderr instead of no progress at all (REQ-INDEX-007/REQ-INDEX-024).
        on_progress = None
        if progress:
            on_progress = (
                stderr_progress if output_format == OutputFormat.TEXT else structured_progress
            )
        summary = rpc_call_or_error("recall.index", rpc_params, on_progress=on_progress)
    except CliError as err:
        emit_error(err, output_format=output_format)
        raise typer.Exit(code=err.exit_code) from None
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=output_format,
        )
        raise typer.Exit(code=2) from None

    summary_dict = summary if isinstance(summary, dict) else {}
    ctas = cta_for_index(summary_dict) if (cta or output_format == OutputFormat.TEXT) else []

    if output_format != OutputFormat.TEXT:
        try:
            emit_data_with_cta(
                summary,
                ctas,
                output_format=output_format,
                include_cta=cta,
                fields=fields_value,
                allowed_fields=output_fields_for("index"),
            )
        except CliError as err:
            emit_error(err, output_format=output_format)
            raise typer.Exit(code=err.exit_code) from None
        return

    indexed = summary_dict.get("indexed", 0)
    skipped = summary_dict.get("skipped", 0)
    failed = summary_dict.get("failed", 0)
    changed = summary_dict.get("changed", 0)
    total = summary_dict.get("total", 0)
    total_seconds = summary_dict.get("total_seconds", 0.0)
    typer.echo(
        f"Indexed {indexed} sessions, skipped {skipped}, "
        f"failed {failed} (changed {changed} of {total}) "
        f"in {total_seconds:.2f}s."
    )
    backlog_line = _backlog_line(summary_dict)
    if backlog_line is not None:
        typer.echo(backlog_line)
    render_cta_hints(ctas)


def _backlog_line(summary: dict[str, object]) -> str | None:
    """Report what the daemon still owes, so silence is never the answer.

    A plain `recall index` hands the sources that were already pending to the
    shared drain (REQ-RECON-025); without this line its summary would count
    them in neither `indexed` nor `failed` and read as work that vanished.
    """
    pending = summary.get("backlog_pending", 0)
    if not isinstance(pending, int) or pending <= 0:
        return None
    rate = summary.get("backlog_drain_per_minute")
    draining = (
        f"the daemon is draining ~{rate:.0f}/min"
        if isinstance(rate, int | float) and rate > 0
        else "the daemon is draining it"
    )
    return f"Reconciliation backlog: {pending} transcripts pending; {draining}."


def _warn_if_full_reparse_will_summarize() -> None:
    """Warn that a bare --full under an LLM context mode summarizes everything.

    Reusing stored context (REQ-INDEX-021) prevents the loss, not the cost of
    rows that never had any: on a corpus where most messages carry `off` or
    `template` context, a bare --full is hundreds of hours of model calls.  The
    warning is advisory -- automation that means it is not blocked -- and never
    raises, because a diagnostic must not be able to fail the command.
    """
    try:
        from recall.core.config import AppConfig

        mode = AppConfig.load().embedding.context.mode
    except Exception:
        return
    if mode not in {"llm-local", "llm-remote", "llm-codex"}:
        return
    typer.secho(
        f"warning: --full with context mode {mode!r} will summarize every message "
        "that has no stored LLM context, which can take hundreds of hours. "
        "Pass --context template to re-parse without summarizing "
        "(stored LLM context is reused either way).",
        err=True,
        fg=typer.colors.YELLOW,
    )
