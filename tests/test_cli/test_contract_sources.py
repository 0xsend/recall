"""Explicit-source detection must survive typer vendoring click.

typer >= 0.22 ships its own copy of click as ``typer._click``, so
``ctx.get_parameter_source`` returns members of a ParameterSource enum that
is a *different class* from standalone click's. Identity-based membership
against a set built from ``click.core.ParameterSource`` silently fails for
every parameter, which makes explicitly-passed CLI flags read as defaulted
and lets ``--params`` payload values override them (observed as
``recall daemon --json`` starting a foreground daemon instead of exiting 2).
"""

from __future__ import annotations

import enum
from typing import cast

import typer
from recall.cli.contract import resolve_bool_param, resolve_param


class _VendoredParameterSource(enum.Enum):
    """Stands in for typer._click's ParameterSource: same member names,
    different enum class than click.core.ParameterSource."""

    COMMANDLINE = enum.auto()
    DEFAULT = enum.auto()


class _StubCtx:
    def __init__(self, source: _VendoredParameterSource) -> None:
        self._source = source

    def get_parameter_source(self, name: str) -> _VendoredParameterSource:
        return self._source


def _ctx(source: _VendoredParameterSource) -> typer.Context:
    return cast(typer.Context, _StubCtx(source))


def test_explicit_flag_wins_over_payload_with_vendored_click_enum() -> None:
    ctx = _ctx(_VendoredParameterSource.COMMANDLINE)
    # CLI passed --once (current=True); payload says false. Explicit CLI
    # input must win even when the source enum comes from a vendored click.
    assert resolve_bool_param(ctx, {"once": False}, "once", True) is True


def test_explicit_string_param_wins_over_payload_with_vendored_click_enum() -> None:
    ctx = _ctx(_VendoredParameterSource.COMMANDLINE)
    assert resolve_param(ctx, {"source": "codex"}, "source", "claude-code") == "claude-code"


def test_defaulted_param_still_resolves_from_payload() -> None:
    ctx = _ctx(_VendoredParameterSource.DEFAULT)
    # DEFAULT is not an explicit source; payload must still apply.
    assert resolve_bool_param(ctx, {"once": True}, "once", False) is True
