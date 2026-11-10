from __future__ import annotations

import importlib
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from dataclasses import dataclass
from typing import Any

import typer

# Typer vendors click; its group and command classes come from this copy.
from typer import _click as click
from typer.core import TyperGroup

from recall.cli.contract import CliError, ErrorCode, OutputFormat, emit_data, emit_error
from recall.cli.manifest import command_manifest, full_manifest


@dataclass(frozen=True)
class _CommandSource:
    """Where a top-level command is defined: a module attribute that is either
    a command function or a Typer group."""

    module: str
    attribute: str


# Top-level commands in help order: Typer lists command functions before groups.
# A CLI invocation imports only the module of the command it runs: every module
# costs import CPU before any command can answer, and on a small host that cost
# is most of a responsive read.
_COMMANDS: dict[str, _CommandSource] = {
    "index": _CommandSource("recall.cli.index", "command"),
    "compact": _CommandSource("recall.cli.compact", "command"),
    "search": _CommandSource("recall.cli.search", "command"),
    "list": _CommandSource("recall.cli.list", "command"),
    "show": _CommandSource("recall.cli.show", "command"),
    "live": _CommandSource("recall.cli.live", "app"),
    "daemon": _CommandSource("recall.cli.daemon", "app"),
    "db": _CommandSource("recall.cli.db", "app"),
    "snapshots": _CommandSource("recall.cli.snapshots", "app"),
    "stats": _CommandSource("recall.cli.stats", "app"),
    "fleet": _CommandSource("recall.cli.fleet", "app"),
}


def _load_command(name: str, source: _CommandSource) -> click.Command:
    """Convert the command exactly as registering it on the root would."""
    target = getattr(importlib.import_module(source.module), source.attribute)
    registration = typer.Typer(add_completion=False)
    if isinstance(target, typer.Typer):
        registration.add_typer(target, name=name)
    else:
        registration.command(name)(target)
    return typer.main.get_group(registration).commands[name]


class _CommandTable(dict[str, click.Command]):
    """Every top-level command name; a command is converted on first read.

    Click and every reader of `Group.commands` see the whole tree, while a CLI
    invocation imports only the module of the command it reads.
    """

    def __missing__(self, name: str) -> click.Command:
        source = _COMMANDS.get(name)
        if source is None:
            raise KeyError(name)
        command = _load_command(name, source)
        self[name] = command
        return command

    def __contains__(self, name: object) -> bool:
        return name in _COMMANDS or super().__contains__(name)

    def __iter__(self) -> Iterator[str]:
        eager = [name for name in super().__iter__() if name not in _COMMANDS]
        return iter([*eager, *_COMMANDS])

    def __len__(self) -> int:
        return len(list(iter(self)))

    # dict's own lookups and views read only the stored entries. These read
    # through `__iter__` and `__missing__`, so unconverted commands are present;
    # their signatures narrow dict's, which the type checker reports.
    def get(  # ty: ignore[invalid-method-override]
        self, key: str, default: click.Command | None = None, /
    ) -> click.Command | None:
        if key not in self:
            return default
        return self[key]

    def keys(self) -> KeysView[str]:  # ty: ignore[invalid-method-override]
        return KeysView(self)

    def items(self) -> ItemsView[str, click.Command]:  # ty: ignore[invalid-method-override]
        return ItemsView(self)

    def values(self) -> ValuesView[click.Command]:  # ty: ignore[invalid-method-override]
        return ValuesView(self)


class _LazyCommandGroup(TyperGroup):
    """Root group whose command table imports each command's module on first read."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commands = _CommandTable(self.commands)


app = typer.Typer(add_completion=False, invoke_without_command=True, cls=_LazyCommandGroup)
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
    from recall.cli.poll_snapshot import begin_invocation, end_invocation
    from recall.core import read_scope

    read_token = read_scope.begin_invocation()
    ctx.call_on_close(lambda: read_scope.end_invocation(read_token))
    token = begin_invocation()
    ctx.call_on_close(lambda: end_invocation(token))
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


if __name__ == "__main__":
    app()
