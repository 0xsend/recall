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
    build_tool_call,
    extract_content_blocks,
    parse_timestamp,
    resolve_home,
    summarize_tool_result,
)
from recall.parsers.protocol import (
    default_discover,
    default_live_candidates,
    default_watch_roots,
)
from recall.parsers.revision import parser_revision


@dataclass
class PiAgentParser:
    source: Source = Source.PI_AGENT
    # Discovery roots from `[sources.<name>] roots`; None = default_roots(),
    # an empty tuple = configured to scan nothing.
    roots: tuple[Path, ...] | None = None

    @property
    def file_pattern(self) -> str:
        return "*.jsonl"

    def default_roots(self) -> list[Path]:
        return [resolve_home() / ".pi" / "agent" / "sessions"]

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
        # One record is one message; nothing accumulates across records, so
        # the boundary carries nothing (REQ-INDEX-026).
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
        source_session_id: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        with JsonlCapture(path, offset=offset) as capture:
            diagnostics = capture.diagnostics
            for entry in capture.records():
                timestamp = parse_timestamp(entry.get("timestamp"))
                if timestamp is not None:
                    if is_full_parse:
                        started_at = timestamp if started_at is None else min(started_at, timestamp)
                    ended_at = timestamp if ended_at is None else max(ended_at, timestamp)

                entry_type = entry.get("type")
                if entry_type == "session":
                    source_session_id = _coerce_str(entry.get("id")) or source_session_id
                    cwd = _coerce_str(entry.get("cwd")) or cwd
                elif entry_type == "model_change":
                    model = _coerce_str(entry.get("modelId")) or model
                elif entry_type == "message":
                    message_payload = entry.get("message")
                    if (
                        isinstance(message_payload, dict)
                        and message_payload.get("role") == "bashExecution"
                    ):
                        message, message_tool_results = _parse_bash_execution(
                            message_payload,
                            session_id=session_id_value,
                            idx=message_idx_base + len(messages),
                            timestamp=timestamp,
                        )
                    elif not isinstance(message_payload, dict) or message_payload.get(
                        "role"
                    ) not in {
                        "user",
                        "assistant",
                        "system",
                        "toolResult",
                    }:
                        diagnostics.append(
                            ParseDiagnostic(
                                "unsupported_record",
                                capture.record_start,
                                "unrecognized message payload or role",
                            )
                        )
                        continue
                    else:
                        message, message_tool_results = _parse_message(
                            message_payload=message_payload,
                            session_id=session_id_value,
                            idx=message_idx_base + len(messages),
                            timestamp=timestamp,
                        )
                    tool_results.extend(message_tool_results)
                    messages.append(message)
                    _record_pi_stop_marker(stop_markers, message_payload, message)
                    for tool_idx, tool_call in enumerate(message.tool_calls):
                        tool_call.idx = tool_idx
                        tool_call.id = make_tool_call_id(message.id, tool_idx)
                        tool_call.session_id = session_id_value
                        tool_call.message_id = message.id
                        tool_calls.append(tool_call)

                    usage = message_payload.get("usage")
                    if isinstance(usage, dict):
                        input_tokens = accumulate_metric(input_tokens, usage.get("input"))
                        output_tokens = accumulate_metric(output_tokens, usage.get("output"))
                elif entry_type == "custom_message":
                    content = _coerce_str(entry.get("content"))
                    if content is not None:
                        messages.append(
                            _plain_message(
                                role=Role.SYSTEM,
                                content=content,
                                session_id=session_id_value,
                                idx=message_idx_base + len(messages),
                                timestamp=timestamp,
                            )
                        )
                elif entry_type == "compaction":
                    summary = _coerce_str(entry.get("summary"))
                    if summary is not None:
                        messages.append(
                            _plain_message(
                                role=Role.SYSTEM,
                                content=summary,
                                session_id=session_id_value,
                                idx=message_idx_base + len(messages),
                                timestamp=timestamp,
                            )
                        )
                # A context_edit omits or replaces a prior entry in the model's
                # future context only; the recorded history it targets stays
                # authoritative, so the edit itself is not conversation content.
                elif entry_type not in {
                    "thinking_level_change",
                    "label",
                    "session_info",
                    "context_edit",
                }:
                    diagnostics.append(
                        ParseDiagnostic(
                            "unsupported_record", capture.record_start, f"record: {entry_type!r}"
                        )
                    )

        duration_seconds = None
        if started_at and ended_at:
            duration_seconds = int((ended_at - started_at).total_seconds())

        next_byte_offset = capture.next_byte_offset
        is_complete = is_complete and not diagnostics
        session = Session(
            id=session_id_value,
            source=self.source,
            source_path=absolute_path,
            source_session_id=source_session_id,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            model=model,
            cwd=cwd,
            git_repo=None,
            git_branch=None,
            message_count=len(messages),
            tool_count=len(tool_calls),
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
                message_idx_base=message_idx_base + len(messages),
                orphan_tool_call_idx_base=orphan_tool_call_idx_base,
            ),
            tail_facts=TailFacts(
                tool_results=tuple(tool_results),
                stop_markers=tuple(stop_markers),
            ),
        )


def _plain_message(
    *,
    role: Role,
    content: str,
    session_id: str,
    idx: int,
    timestamp: datetime | None,
) -> Message:
    return Message(
        id=make_message_id(session_id, idx),
        session_id=session_id,
        idx=idx,
        role=role,
        content=content,
        timestamp=timestamp,
    )


def _parse_bash_execution(
    payload: dict[str, Any],
    *,
    session_id: str,
    idx: int,
    timestamp: datetime | None,
) -> tuple[Message, list[ToolResult]]:
    command = _coerce_str(payload.get("command")) or ""
    output = payload.get("output")
    output_text = output if isinstance(output, str) else ""
    exit_code = payload.get("exitCode")
    is_error = bool(payload.get("cancelled")) or (isinstance(exit_code, int) and exit_code != 0)
    tool_call = build_tool_call("bash", {"command": command})
    tool_call.idx = 0
    prefix = f"$ {command}" if command else ""
    content = "\n".join(part for part in (prefix, output_text) if part) or None
    message = Message(
        id=make_message_id(session_id, idx),
        session_id=session_id,
        idx=idx,
        role=Role.SYSTEM,
        content=content,
        timestamp=timestamp,
        tool_calls=[tool_call],
    )
    results: list[ToolResult] = []
    if tool_call.tool_use_id:
        results.append(
            ToolResult(
                tool_use_id=tool_call.tool_use_id,
                result_summary=summarize_tool_result(output_text),
                is_error=is_error,
                completed_at=timestamp,
            )
        )
    return message, results


def _parse_message(
    message_payload: dict[str, Any],
    session_id: str,
    idx: int,
    timestamp: datetime | None,
) -> tuple[Message, list[ToolResult]]:
    """Build the message and hand back the tool results its content answered."""
    role = _parse_role(message_payload.get("role"))
    blocks = extract_content_blocks(message_payload.get("content"))
    # A Pi Agent result is a whole message with role "toolResult" naming the
    # call it answers, not a block inside another message.
    answered_id = _coerce_str(message_payload.get("toolCallId"))
    if answered_id:
        blocks.tool_results.append(
            ToolResult(
                tool_use_id=answered_id,
                result_summary=summarize_tool_result(message_payload.get("content")),
                is_error=bool(message_payload.get("isError", False)),
            )
        )
    for tool_result in blocks.tool_results:
        tool_result.completed_at = timestamp
    message = Message(
        id=make_message_id(session_id, idx),
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


# Pi Agent's end-of-turn stop reasons. `toolUse` (raw `tool_use`) is mid-turn:
# the assistant stopped to run a tool and owes another message. Censused on
# live sessions: toolUse is the working loop; `stop` is a finished reply;
# `aborted` / `error` close the turn without a reply.
_TURN_ENDING_STOP_REASONS = frozenset({"stop", "aborted", "error"})


def _record_pi_stop_marker(
    stop_markers: list[StopMarker], message_payload: dict[str, Any], message: Message
) -> None:
    """Record Pi's stopReason, or the protocol fallback when the field is absent."""
    if message.role is not Role.ASSISTANT:
        return
    reason = message_payload.get("stopReason")
    if isinstance(reason, str) and reason:
        stop_markers.append(
            StopMarker(
                idx=message.idx,
                reason=reason,
                ends_turn=reason in _TURN_ENDING_STOP_REASONS,
            )
        )
        return
    if message.tool_calls:
        stop_markers.append(StopMarker(idx=message.idx, reason="toolUse", ends_turn=False))
    elif message.content:
        stop_markers.append(StopMarker(idx=message.idx, reason="stop", ends_turn=True))


def _parse_role(value: Any) -> Role:
    normalized = _coerce_str(value) or "user"
    match normalized:
        case "assistant":
            return Role.ASSISTANT
        case "toolResult" | "system":
            return Role.SYSTEM
        case _:
            return Role.USER


def _coerce_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
