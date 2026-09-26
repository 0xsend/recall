from __future__ import annotations

import typer

from recall.cli import compact as compact_cmd
from recall.cli import daemon as daemon_cmd
from recall.cli import db as db_cmd
from recall.cli import fleet as fleet_cmd
from recall.cli import index as index_cmd
from recall.cli import list as list_cmd
from recall.cli import live as live_cmd
from recall.cli import search as search_cmd
from recall.cli import show as show_cmd
from recall.cli import snapshots as snapshots_cmd
from recall.cli.contract import CliError, ErrorCode, OutputFormat, emit_data, emit_error
from recall.cli.manifest import command_manifest, full_manifest
from recall.cli.stats import app as stats_app

app = typer.Typer(add_completion=False, invoke_without_command=True)
SCHEMA_COMMAND_ARGUMENT = typer.Argument(
    None,
    help="Command path to inspect, for example: search or daemon install",
)


@app.callback()
def root(
    ctx: typer.Context,
    llms: bool = typer.Option(False, "--llms", help="Emit the full agent manifest as JSON"),
    version: bool = typer.Option(False, "--version", help="Print recall version and exit"),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if version:
        from importlib.metadata import version as _pkg_version

        typer.echo(_pkg_version("recall"))
        raise typer.Exit()
    if not llms:
        return
    emit_data(full_manifest(), output_format=OutputFormat.JSON)
    raise typer.Exit()


@app.command("schema")
def schema_command(
    command_path: list[str] | None = SCHEMA_COMMAND_ARGUMENT,
) -> None:
    try:
        emit_data(command_manifest(command_path or []), output_format=OutputFormat.JSON)
    except ValueError as err:
        emit_error(
            CliError(code=ErrorCode.VALIDATION, message=str(err), exit_code=2),
            output_format=OutputFormat.JSON,
        )
        raise typer.Exit(code=2) from None


app.command("index")(index_cmd.command)
app.command("compact")(compact_cmd.command)
app.command("search")(search_cmd.command)
app.command("list")(list_cmd.command)
app.add_typer(live_cmd.app, name="live")
app.command("show")(show_cmd.command)
app.add_typer(daemon_cmd.app, name="daemon")
app.add_typer(db_cmd.app, name="db")
app.add_typer(snapshots_cmd.app, name="snapshots")
app.add_typer(stats_app, name="stats")
app.add_typer(fleet_cmd.app, name="fleet")


if __name__ == "__main__":
    app()
