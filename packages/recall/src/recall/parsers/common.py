"""Shared helpers for JSONL session parsers.

Consolidates content extraction, tool call construction, timestamp parsing,
and cascading field lookup utilities used across Claude Code, Codex, and
Pi Agent parsers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from recall.core.bash import parse_bash_command
from recall.core.models import ToolCall, ToolResult
from recall.parsers.skills import derive_skill_name

# Alternate home for multi-host ingest (REQ-MULTIHOST-001). discover/watch_roots
# consult resolve_home(); parse workers do not re-discover, so main-thread set
# is enough for index runs.
_HOME_ROOT: ContextVar[Path | None] = ContextVar("recall_home_root", default=None)


def resolve_home() -> Path:
    """Return the active home root (override or ``Path.home()``)."""
    override = _HOME_ROOT.get()
    return override if override is not None else Path.home()


def home_root_overridden() -> bool:
    """True when multi-host ``use_home_root`` is active."""
    return _HOME_ROOT.get() is not None


@contextmanager
def use_home_root(path: Path | None) -> Iterator[None]:
    """Temporarily treat *path* as $HOME for parser discovery."""
    token = _HOME_ROOT.set(path)
    try:
        yield
    finally:
        _HOME_ROOT.reset(token)


# ---------------------------------------------------------------------------
# Cascading field extraction
# ---------------------------------------------------------------------------
# Each path is either a str (flat key lookup) or a tuple of str (nested key
# traversal). Paths are tried in order — newest format first, oldest last.
# When the upstream format changes, prepend one path to the caller's list.


def first_of(entry: dict[str, Any], *paths: str | tuple[str, ...]) -> str | None:
    """Return the first non-None string value found at any of *paths* in *entry*.

    A path can be a single string key (flat lookup) or a tuple of keys
    (nested traversal).  Paths are tried left-to-right; the first hit wins.
    """
    for path in paths:
        value = _resolve_path(entry, path)
        if isinstance(value, str) and value:
            return value
    return None


def first_int(entry: dict[str, Any], *paths: str | tuple[str, ...]) -> int | None:
    """Like :func:`first_of` but coerces the resolved value to ``int``.

    Returns ``None`` if no path resolves or the value cannot be coerced.
    """
    for path in paths:
        value = _resolve_path(entry, path)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_path(entry: dict[str, Any], path: str | tuple[str, ...]) -> Any:
    """Walk *entry* along *path* and return the leaf value, or ``None``."""
    if isinstance(path, str):
        return entry.get(path)
    # Tuple path — nested traversal
    current: Any = entry
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp string, tolerating trailing ``Z``."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Token accumulation
# ---------------------------------------------------------------------------


def accumulate_metric(existing: int | None, value: Any) -> int | None:
    """Accumulate an integer metric: ``None + None = None``, ``None + N = N``, ``M + N = M + N``."""
    if value is None:
        return existing
    try:
        number = int(value)
    except (TypeError, ValueError):
        return existing
    if existing is None:
        return number
    return existing + number


# ---------------------------------------------------------------------------
# Content block extraction
# ---------------------------------------------------------------------------

# Recognized text block types across all providers
_TEXT_TYPES = frozenset({"text", "input_text", "output_text"})

# Recognized tool-use block types across all providers
_TOOL_USE_TYPES = frozenset({"tool_use", "toolCall"})

# Recognized tool-result block types across all providers
_TOOL_RESULT_TYPES = frozenset({"tool_result", "toolResult"})

# REQ-LIVE-006: a stored result is a summary for turn state, not the payload.
TOOL_RESULT_SUMMARY_MAX_CHARS = 1024


@dataclass(frozen=True)
class ContentBlocks:
    """Everything one message's content blocks yield.

    A value object rather than a tuple: the block vocabulary grows (tool
    results arrived with REQ-LIVE-005) and every growth would otherwise
    re-order an unlabelled tuple at five call sites.
    """

    text: list[str]
    thinking: list[str]
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]


def summarize_tool_result(content: Any) -> str:
    """Flatten a tool_result payload to a bounded summary string.

    Harnesses put either a plain string or a list of text blocks here, and the
    payload is unbounded — a single `Read` result can be megabytes — so it is
    truncated at ``TOOL_RESULT_SUMMARY_MAX_CHARS``.
    """
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        text = "\n".join(parts) if parts else ""
    elif content is None:
        text = ""
    else:
        text = str(content)
    return text[:TOOL_RESULT_SUMMARY_MAX_CHARS]


def extract_content_blocks(content: Any) -> ContentBlocks:
    """Extract text, thinking, tool-call, and tool-result blocks from message content.

    Handles all known content-block schemas across Claude Code, Codex, and
    Pi Agent: ``tool_use`` and ``toolCall`` types, ``text`` and ``thinking``
    keys in thinking blocks, ``tool_result`` and ``toolResult`` results.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_results: list[ToolResult] = []
    blocks = ContentBlocks(text_parts, thinking_parts, tool_calls, tool_results)

    if content is None:
        return blocks

    if isinstance(content, str):
        text_parts.append(content)
        return blocks

    if isinstance(content, dict):
        content = [content]

    if not isinstance(content, Iterable):
        return blocks

    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type in _TEXT_TYPES:
            text = item.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        elif item_type == "thinking":
            # Pi Agent uses "thinking" key; Claude Code uses "text"
            thinking = item.get("thinking") or item.get("text")
            if isinstance(thinking, str) and thinking:
                thinking_parts.append(thinking)
        elif item_type in _TOOL_USE_TYPES:
            tool_name = item.get("name")
            tool_name = str(tool_name) if tool_name else ""
            # tool_use uses "input" key; toolCall uses "arguments" key.
            # Check key presence explicitly — "input": {} is valid and must
            # not be dropped by falsy short-circuit.
            if "input" in item:
                tool_input = item["input"]
            else:
                tool_input = _parse_json_arguments(item.get("arguments"))
            tool_use_id = item.get("id") or item.get("toolCallId")
            tool_calls.append(
                build_tool_call(
                    tool_name,
                    tool_input,
                    tool_use_id=str(tool_use_id) if tool_use_id else None,
                )
            )
        elif item_type in _TOOL_RESULT_TYPES:
            # Only a result naming its call can be paired; an unnamed one has
            # no join key and is dropped rather than guessed at by position.
            answered_id = item.get("tool_use_id") or item.get("toolCallId")
            if answered_id:
                tool_results.append(
                    ToolResult(
                        tool_use_id=str(answered_id),
                        result_summary=summarize_tool_result(item.get("content")),
                        is_error=bool(item.get("is_error", False)),
                    )
                )

    return blocks


def _parse_json_arguments(raw: Any) -> dict[str, Any] | None:
    """Parse a JSON-encoded arguments string (Codex/Pi Agent ``toolCall`` format)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# Tool call construction
# ---------------------------------------------------------------------------

# Union of all bash-like tool names across providers
_BASH_TOOL_NAMES = frozenset({"bash", "shell", "exec_command", "shell_command"})


def build_tool_call(tool_name: str, tool_input: Any, *, tool_use_id: str | None = None) -> ToolCall:
    """Construct a ``ToolCall`` with bash command parsing when applicable.

    For Agent tool calls, subagent_type/description/model are extracted from
    tool_input so the DB can answer "what subagents were launched?" without
    re-parsing raw JSON blobs.  Similarly, Skill calls carry skill_name.

    Only Claude Code has a typed Skill tool; every other harness loads a
    skill by reading its SKILL.md, so skill_name is derived from the path
    when the call did not name a skill itself (REQ-PARSE-017).
    """
    bash_command = extract_bash_command(tool_name, tool_input)
    parsed = parse_bash_command(bash_command) if bash_command else None

    # Extract Agent / Skill metadata when present; guard against non-dict inputs
    subagent_type: str | None = None
    subagent_description: str | None = None
    subagent_model: str | None = None
    skill_name: str | None = None

    if isinstance(tool_input, dict):
        if tool_name == "Agent":
            subagent_type = tool_input.get("subagent_type") or None
            subagent_description = tool_input.get("description") or None
            subagent_model = tool_input.get("model") or None
        elif tool_name == "Skill":
            skill_name = tool_input.get("skill") or None

    if skill_name is None:
        skill_name = derive_skill_name(tool_name, tool_input, bash_command)

    # A payload that is a bare string has nowhere to live in a dict-typed
    # column.  Codex sends `apply_patch` that way -- `input` is the raw patch
    # -- so discarding it left the row recording only that a call happened
    # (REQ-PARSE-018).  A string already recovered as a bash command is not
    # stored twice.
    stored_input: dict[str, Any] | None
    if isinstance(tool_input, dict):
        stored_input = tool_input
    elif isinstance(tool_input, str) and bash_command is None:
        stored_input = {"input": tool_input}
    else:
        stored_input = None

    return ToolCall(
        id="",
        session_id="",
        message_id=None,
        idx=0,
        tool_name=tool_name,
        tool_input=stored_input,
        bash_command=parsed.command if parsed else None,
        bash_base=parsed.base if parsed else None,
        bash_sub=parsed.sub if parsed else None,
        is_compound=parsed.is_compound if parsed else False,
        subagent_type=subagent_type,
        subagent_description=subagent_description,
        subagent_model=subagent_model,
        skill_name=skill_name,
        tool_use_id=tool_use_id,
    )


def extract_bash_command(tool_name: str, tool_input: Any) -> str | None:
    """Extract the bash command string from a tool invocation, if applicable.

    Recognizes ``bash``, ``shell``, ``exec_command``, and ``shell_command``
    tool names.  Handles ``command``, ``cmd``, and ``commands`` (list) input
    keys, plus bare-string input.
    """
    if tool_name.lower() not in _BASH_TOOL_NAMES:
        return None
    if isinstance(tool_input, str):
        return tool_input
    if isinstance(tool_input, dict):
        if isinstance(tool_input.get("command"), str):
            return tool_input["command"]
        if isinstance(tool_input.get("cmd"), str):
            return tool_input["cmd"]
        if isinstance(tool_input.get("commands"), list):
            return " && ".join(str(cmd) for cmd in tool_input["commands"] if cmd)
    return None
