from __future__ import annotations

import shlex
from dataclasses import dataclass

BASH_SUBCOMMAND_TOOLS = {
    "git",
    "kubectl",
    "docker",
    "npm",
    "yarn",
    "pnpm",
    "cargo",
    "go",
    "uv",
    "pip",
    "brew",
    "apt",
    "systemctl",
}

COMPOUND_TOKENS = ("&&", "||", "|", ";", "\n")


@dataclass(frozen=True)
class BashCommand:
    command: str
    base: str | None
    sub: str | None
    is_compound: bool


def parse_bash_command(command: str | None) -> BashCommand | None:
    if command is None:
        return None
    stripped = command.strip()
    if not stripped:
        return None
    is_compound = any(token in stripped for token in COMPOUND_TOKENS)
    first_segment = stripped
    for token in ("&&", "||", "|", ";"):
        if token in first_segment:
            first_segment = first_segment.split(token, 1)[0]
    try:
        parts = shlex.split(first_segment)
    except ValueError:
        parts = first_segment.split()
    base = parts[0] if parts else None
    sub = None
    if base in BASH_SUBCOMMAND_TOOLS and len(parts) > 1:
        sub = parts[1]
    return BashCommand(command=stripped, base=base, sub=sub, is_compound=is_compound)
