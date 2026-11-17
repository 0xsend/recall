"""Render `--help` output as plain prose for assertions."""

import re

from recall.cli.app import app
from typer.testing import CliRunner

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def help_text(argv: list[str]) -> str:
    """Render `--help` as one line of prose.

    Rich wraps each help column and frames it in box-drawing characters, so a
    sentence arrives split across rows with `│` between the halves. Under
    GitHub Actions (or FORCE_COLOR) Typer also forces ANSI styling, which
    splits words such as `--help` with escape sequences. Drop the escapes and
    the frame, then collapse whitespace, so assertions can name whole phrases.
    """
    result = CliRunner().invoke(app, [*argv, "--help"])
    assert result.exit_code == 0, result.output
    plain = _ANSI_ESCAPE.sub("", result.stdout)
    unframed = plain.translate({ord(char): " " for char in "│╭╮╰╯─"})
    return " ".join(unframed.split())
