"""Lightweight executable entrypoint."""

import sys

from recall.read_budget import cli_read_budget


def main() -> None:
    if sys.argv[1:2] == ["live"] or sys.argv[1:3] == ["daemon", "status"]:
        with cli_read_budget():
            from recall.cli.app import app

            app()
    else:
        from recall.cli.app import app

        app()
