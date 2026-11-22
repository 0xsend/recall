"""Bounded single-line tool-call summaries shared across harnesses."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

TOOL_DETAIL_MAX_CHARS = 160
_MAX_FALLBACK_ITEMS = 3
_MAX_PATCH_TARGETS = 3
_PATCH_FILE_RE = re.compile(r"^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s+(.+?)\s*$", re.MULTILINE)

_PATH_KEYS = (
    "file_path",
    "path",
    "notebook_path",
    "target_file",
    "directory",
    "dir_path",
    "cwd",
)
_PATTERN_KEYS = ("pattern", "query", "search_term", "glob")
_URL_KEYS = ("url", "uri")
_COMMAND_KEYS = ("command", "cmd", "commandline", "command_line")
_MESSAGE_KEYS = ("description", "summary", "toolsummary", "message", "prompt", "question", "target")


def format_tool_detail(
    tool_name: str,
    tool_input: Mapping[str, Any] | None,
    bash_command: str | None = None,
    *,
    max_chars: int = TOOL_DETAIL_MAX_CHARS,
) -> str:
    """Return a bounded, single-line summary of a tool call's target or arguments."""
    assert max_chars >= 4, f"max_chars must be >= 4, got {max_chars}"
    if bash_command:
        normalized = _collapse_ws(bash_command)
        if normalized:
            return _truncate(normalized, max_chars)

    if not isinstance(tool_input, Mapping) or not tool_input:
        return ""

    lowered = {str(key).lower(): value for key, value in tool_input.items()}
    name_lower = tool_name.strip().lower()

    detail = (
        _format_subagent(name_lower, lowered)
        or _format_skill(name_lower, lowered)
        or _format_patch(name_lower, lowered)
        or _format_primary_target(lowered)
        or _format_fallback_kv(tool_input)
    )
    return _truncate(_collapse_ws(detail), max_chars) if detail else ""


def format_tool_call_line(
    tool_call: Mapping[str, Any],
    *,
    max_chars: int = TOOL_DETAIL_MAX_CHARS,
) -> str:
    """Render `[tool_name] <detail>` with no trailing space when detail is empty."""
    label = str(tool_call.get("tool_name") or "tool")
    raw_input = tool_call.get("tool_input")
    tool_input = raw_input if isinstance(raw_input, Mapping) else None
    raw_bash = tool_call.get("bash_command")
    bash_command = raw_bash if isinstance(raw_bash, str) else None
    detail = format_tool_detail(label, tool_input, bash_command, max_chars=max_chars)
    return f"[{label}] {detail}" if detail else f"[{label}]"


def _format_subagent(name_lower: str, lowered: Mapping[str, Any]) -> str:
    if name_lower not in {"agent", "invoke_subagent", "spawn_agent"}:
        return ""
    kind = _first_str(lowered, ("subagent_type", "typename", "agent_type"))
    desc = _first_str(lowered, ("description", "role", "prompt"))
    if kind and desc:
        return f"{kind}: {desc}"
    return kind or desc or ""


def _format_skill(name_lower: str, lowered: Mapping[str, Any]) -> str:
    if name_lower != "skill":
        return ""
    return _first_str(lowered, ("skill", "name")) or ""


def _format_patch(name_lower: str, lowered: Mapping[str, Any]) -> str:
    if name_lower != "apply_patch":
        return ""
    raw = _first_str(lowered, ("input", "patch"))
    if not raw:
        return ""
    targets = _PATCH_FILE_RE.findall(raw)
    if targets:
        unique = list(dict.fromkeys(targets))
        shown = ", ".join(unique[:_MAX_PATCH_TARGETS])
        if len(unique) > _MAX_PATCH_TARGETS:
            shown = f"{shown} (+{len(unique) - _MAX_PATCH_TARGETS} more)"
        return shown
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _format_primary_target(lowered: Mapping[str, Any]) -> str:
    pattern = _first_str(lowered, _PATTERN_KEYS)
    path = _first_str(lowered, _PATH_KEYS)
    if pattern and path:
        return f"{pattern} in {path}"
    if path:
        return path
    if pattern:
        return pattern
    url = _first_str(lowered, _URL_KEYS)
    if url:
        return url
    command = _first_str(lowered, _COMMAND_KEYS)
    if command:
        return command
    return _first_str(lowered, _MESSAGE_KEYS) or ""


def _format_fallback_kv(tool_input: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, value in tool_input.items():
        if value is None or isinstance(value, (dict, list)):
            continue
        rendered = _collapse_ws(str(value))
        if not rendered:
            continue
        parts.append(f"{key}={rendered}")
        if len(parts) >= _MAX_FALLBACK_ITEMS:
            break
    return " ".join(parts)


def _first_str(lowered: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = lowered.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
