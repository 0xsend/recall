from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from recall.core.ids import message_id as make_message_id
from recall.core.ids import session_id as make_session_id
from recall.core.ids import tool_call_id as make_tool_call_id
from recall.core.models import (
    Message,
    ParseDiagnostic,
    ParseResult,
    Session,
    StopMarker,
    TailFacts,
    ToolCall,
    ToolResult,
)
from recall.core.types import Role, Source
from recall.parsers.capture import JsonlCapture
from recall.parsers.checkpoint import read_resume_state, resume_checkpoint
from recall.parsers.common import (
    accumulate_metric,
    extract_content_blocks,
    first_int,
    first_of,
    parse_timestamp,
    resolve_home,
)
from recall.parsers.protocol import (
    default_discover,
    default_live_candidates,
    default_watch_roots,
)
from recall.parsers.revision import parser_revision

# Metadata types still yield sessionId/cwd/gitBranch. They must not diagnose
# as unsupported and must not become conversation messages.
_KNOWN_RECORD_TYPES = frozenset(
    {
        "user",
        "assistant",
        "message",
        "progress",
        "system",
        "summary",
        "queue-operation",
        "file-history-snapshot",
        "last-prompt",
        "custom-title",
        "attachment",
        "atis-latch",
        "ai-title",
        "mode",
        "permission-mode",
        "file-history-delta",
        "cost-state",
        "pr-link",
        "agent-name",
        "relocated",
        "worktree-state",
        "agent-color",
        "bridge-session",
        "fork-context-ref",
        "continued-in",
        "agent-setting",
        "dev-mods",
    }
)


@dataclass
class ClaudeCodeParser:
    source: Source = Source.CLAUDE_CODE
    # Discovery roots from `[sources.<name>] roots`; None = default_roots(),
    # an empty tuple = configured to scan nothing.
    roots: tuple[Path, ...] | None = None

    @property
    def file_pattern(self) -> str:
        return "*.jsonl"

    def default_roots(self) -> list[Path]:
        return [resolve_home() / ".claude/projects"]

    def watch_roots(self) -> list[Path]:
        return default_watch_roots(self)

    def discover(self) -> list[Path]:
        return default_discover(self)

    def sidecar_paths(self, path: Path) -> list[Path]:
        """No sidecars: all metadata comes from the session file itself."""
        return []

    def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
        return default_live_candidates(self, now=now, idle_threshold=idle_threshold)

    def parse(
        self,
        path: Path,
        *,
        offset: int = 0,
        message_idx_base: int = 0,
        orphan_tool_call_idx_base: int = 0,
        resume_state: Mapping[str, Any] | None = None,
    ) -> ParseResult:
        # Every record is self-contained: a message never waits on a later one
        # to complete it, so the boundary carries nothing (REQ-INDEX-026).
        read_resume_state(resume_state, offset=offset, supported=frozenset())
        absolute_path = str(path.expanduser().resolve())
        session_id_value = make_session_id(self.source.value, absolute_path)
        stat = path.stat()
        file_mtime = stat.st_mtime
        file_size = stat.st_size
        is_full_parse = offset == 0

        messages: list[Message] = []
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        stop_markers: list[StopMarker] = []
        is_complete = True
        started_at: datetime | None = None
        ended_at: datetime | None = None
        model: str | None = None
        cwd: str | None = None
        git_repo: str | None = None
        git_branch: str | None = None
        source_session_id: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        with JsonlCapture(path, offset=offset) as capture:
            diagnostics = capture.diagnostics
            for entry in capture.records():
                if entry.get("type") not in _KNOWN_RECORD_TYPES and "role" not in entry:
                    diagnostics.append(
                        ParseDiagnostic(
                            "unsupported_record",
                            capture.record_start,
                            f"record: {entry.get('type')!r}",
                        )
                    )
                    continue
                # -- Timestamp: try root-level fields --
                timestamp = parse_timestamp(first_of(entry, "timestamp", "created_at"))
                if timestamp is not None:
                    if is_full_parse:
                        started_at = timestamp if started_at is None else min(started_at, timestamp)
                    ended_at = timestamp if ended_at is None else max(ended_at, timestamp)

                # -- Session-level metadata: cascading extraction --
                # Each field tries newest format first, falling back to legacy.
                # All four are first-wins over the *file*, and nearly every
                # assistant record carries model and gitBranch. A suffix would
                # elect its own first value, which the incremental merge's
                # COALESCE then writes over the committed one -- so only a
                # parse that started at byte zero may elect them (REQ-INDEX-026).
                if is_full_parse:
                    if model is None:
                        model = first_of(entry, ("message", "model"), "model", "model_name")
                    if cwd is None:
                        cwd = first_of(entry, "cwd", "working_directory")
                    if git_repo is None:
                        git_repo = first_of(entry, "git_root", "repo")
                    if git_branch is None:
                        git_branch = first_of(entry, "gitBranch", ("git", "branch"), "git_branch")
                if source_session_id is None:
                    source_session_id = first_of(entry, "sessionId")

                # -- Token accumulation: try nested usage first, then legacy root --
                input_tokens = accumulate_metric(
                    input_tokens,
                    first_int(entry, ("message", "usage", "input_tokens"), "inputTokens"),
                )
                output_tokens = accumulate_metric(
                    output_tokens,
                    first_int(entry, ("message", "usage", "output_tokens"), "outputTokens"),
                )
                # Cache tokens are separate counters in the Claude API usage
                # model — they are NOT subsets of input_tokens. Accumulating
                # them gives total billable input tokens. No legacy equivalent.
                input_tokens = accumulate_metric(
                    input_tokens,
                    first_int(entry, ("message", "usage", "cache_creation_input_tokens")),
                )
                input_tokens = accumulate_metric(
                    input_tokens,
                    first_int(entry, ("message", "usage", "cache_read_input_tokens")),
                )

                # -- Subagent progress: extract nested messages with agent_id --
                # Progress entries have a top-level `message` key containing
                # request metadata, NOT a message payload. The actual message
                # lives at entry["data"]["message"]["message"]. Guard against
                # accidentally treating this as a main-conversation message.
                if (
                    isinstance(entry, dict)
                    and entry.get("type") == "progress"
                    and isinstance(entry.get("data"), dict)
                    and entry["data"].get("type") == "agent_progress"
                ):
                    agent_id = entry["data"].get("agentId")
                    nested = entry["data"].get("message", {})
                    nested_payload = nested.get("message") if isinstance(nested, dict) else None
                    if nested_payload and isinstance(nested_payload, dict):
                        progress_ts = parse_timestamp(entry.get("timestamp"))
                        if progress_ts is not None:
                            if is_full_parse:
                                started_at = (
                                    progress_ts
                                    if started_at is None
                                    else min(started_at, progress_ts)
                                )
                            ended_at = (
                                progress_ts if ended_at is None else max(ended_at, progress_ts)
                            )

                        # Accumulate subagent token usage — these are billed
                        # separately from main-conversation tokens.
                        usage = nested_payload.get("usage", {})
                        if isinstance(usage, dict):
                            input_tokens = accumulate_metric(
                                input_tokens, usage.get("input_tokens")
                            )
                            output_tokens = accumulate_metric(
                                output_tokens, usage.get("output_tokens")
                            )
                            input_tokens = accumulate_metric(
                                input_tokens,
                                usage.get("cache_creation_input_tokens"),
                            )
                            input_tokens = accumulate_metric(
                                input_tokens, usage.get("cache_read_input_tokens")
                            )

                        message, message_tool_results = _parse_message(
                            message_payload=nested_payload,
                            session_id=session_id_value,
                            idx=message_idx_base + len(messages),
                            timestamp=progress_ts,
                        )
                        message.agent_id = agent_id
                        tool_results.extend(message_tool_results)
                        _record_stop_marker(stop_markers, nested_payload, message.idx)
                        messages.append(message)
                        for tool_idx, tool_call in enumerate(message.tool_calls):
                            tool_call.idx = tool_idx
                            tool_call.id = make_tool_call_id(message.id, tool_idx)
                            tool_call.session_id = session_id_value
                            tool_call.message_id = message.id
                            tool_call.agent_id = agent_id
                            tool_calls.append(tool_call)

                # -- Message extraction (main conversation only) --
                # Use elif to ensure progress entries never fall through here;
                # their top-level `message` key is metadata, not a payload.
                elif isinstance(entry, dict):
                    message_payload = entry.get("message")
                    if message_payload is None and "role" in entry:
                        message_payload = entry
                    if message_payload:
                        if not isinstance(message_payload, dict) or (
                            message_payload.get("role", message_payload.get("sender", "user"))
                            not in {"user", "assistant", "system"}
                        ):
                            diagnostics.append(
                                ParseDiagnostic(
                                    "unsupported_record",
                                    capture.record_start,
                                    "unrecognized message payload or role",
                                )
                            )
                            continue
                        message, message_tool_results = _parse_message(
                            message_payload=message_payload,
                            session_id=session_id_value,
                            idx=message_idx_base + len(messages),
                            timestamp=timestamp,
                        )
                        tool_results.extend(message_tool_results)
                        _record_stop_marker(stop_markers, message_payload, message.idx)
                        messages.append(message)
                        for tool_idx, tool_call in enumerate(message.tool_calls):
                            tool_call.idx = tool_idx
                            tool_call.id = make_tool_call_id(message.id, tool_idx)
                            tool_call.session_id = session_id_value
                            tool_call.message_id = message.id
                            tool_calls.append(tool_call)

        # Current format has no explicit git_root/repo field — derive from
        # cwd as best-effort. Note: cwd may be a subdirectory of the actual
        # repo root. If Claude Code adds a gitRoot field, prepend it to the
        # first_of chain above.
        if git_repo is None and cwd is not None:
            git_repo = cwd

        message_count = len(messages)
        tool_count = len(tool_calls)
        duration_seconds = None
        if started_at and ended_at:
            duration_seconds = int((ended_at - started_at).total_seconds())

        next_byte_offset = capture.next_byte_offset
        is_complete = is_complete and not diagnostics
        session = Session(
            id=session_id_value,
            source=self.source,
            source_path=absolute_path,
            source_session_id=source_session_id or path.stem,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            model=model,
            cwd=cwd,
            git_repo=git_repo,
            git_branch=git_branch,
            message_count=message_count,
            tool_count=tool_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            is_complete=is_complete,
            file_mtime=file_mtime,
            file_size=file_size,
            messages=messages,
        )
        return ParseResult(
            session=session,
            next_byte_offset=next_byte_offset,
            diagnostics=tuple(diagnostics),
            captured_prefix_sha256=capture.captured_prefix_sha256,
            initial_prefix_sha256=capture.initial_prefix_sha256,
            source_dev=capture.source_dev,
            source_inode=capture.source_inode,
            captured_size=capture.captured_size,
            committed_prefix_sha256=capture.committed_prefix_sha256,
            is_full_parse=is_full_parse,
            normalization_checkpoint=resume_checkpoint(
                capture,
                parser_revision=parser_revision(type(self)),
                resumable=True,
                diagnostics=diagnostics,
                message_idx_base=message_idx_base + message_count,
                orphan_tool_call_idx_base=orphan_tool_call_idx_base,
            ),
            tail_facts=TailFacts(
                tool_results=tuple(tool_results),
                stop_markers=tuple(stop_markers),
                # Claude Code writes no session-end record: a census of 60 live
                # transcripts found only conversation, metadata and checkpoint
                # entry types. `ended` comes from the pid stamp (REQ-LIVE-008)
                # or the idle window instead.
                session_ended=False,
            ),
        )


# Claude Code's end-of-turn stop reasons. `tool_use` is the third value it
# emits and means mid-turn: the assistant stopped to run a tool and owes the
# conversation another message. Censused over 40 recent transcripts:
# tool_use 21,545x, end_turn 721x, stop_sequence 8x.
_TURN_ENDING_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})


def _record_stop_marker(
    stop_markers: list[StopMarker], message_payload: dict[str, Any], idx: int
) -> None:
    """Record the harness stop reason for one message, verbatim, when present."""
    stop_reason = message_payload.get("stop_reason")
    if isinstance(stop_reason, str) and stop_reason:
        stop_markers.append(
            StopMarker(
                idx=idx,
                reason=stop_reason,
                ends_turn=stop_reason in _TURN_ENDING_STOP_REASONS,
            )
        )


def _parse_message(
    message_payload: dict[str, Any],
    session_id: str,
    idx: int,
    timestamp: datetime | None,
) -> tuple[Message, list[ToolResult]]:
    """Build the message and hand back the tool results its content answered."""
    role_value = message_payload.get("role") or message_payload.get("sender") or "user"
    try:
        role = Role(role_value)
    except ValueError:
        role = Role.USER

    content = message_payload.get("content")
    blocks = extract_content_blocks(content)
    for tool_result in blocks.tool_results:
        tool_result.completed_at = timestamp

    message_id_value = make_message_id(session_id, idx)
    message = Message(
        id=message_id_value,
        session_id=session_id,
        idx=idx,
        role=role,
        content="\n".join(blocks.text) if blocks.text else None,
        thinking="\n".join(blocks.thinking) if blocks.thinking else None,
        timestamp=timestamp,
        has_thinking=bool(blocks.thinking),
        tool_calls=blocks.tool_calls,
    )
    return message, blocks.tool_results
