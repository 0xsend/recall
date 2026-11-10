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
    ClaudeCounters,
    ClaudeResponseUsage,
    ClaudeUsageFull,
    ClaudeUsageSuffix,
    Message,
    NormalizationCheckpoint,
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
from recall.parsers.checkpoint import (
    ChunkBudget,
    ChunkTarget,
    UnsupportedResumeState,
    read_resume_metadata,
    read_resume_state,
    resume_checkpoint,
)
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
        "tools-used",
    }
)

# The harness's log of the API requests it sent: request parameters, the
# system/tool shape, and content-addressed copies of the prompt blocks. A blob
# has a `message` envelope with a role, but it is wire payload, not a turn: the
# conversation records already carry the turns it copies, and the rest is
# harness-injected context the adapter does not index (REQ-PARSE-033). These
# records are routed away before any message, tool or token normalization.
_REQUEST_LOG_RECORD_TYPES = frozenset({"api-request", "api-request-blob", "api-request-shape"})


class _ResponseUsage:
    """File-local last-supplied patches and anonymous additive counters.

    Content blocks repeat API usage. The last supplied counter is authoritative,
    including zero or a downward correction; an absent counter supplies no update.
    The same response can reappear after other responses or inside agent progress.
    Anonymous legacy records have no safe identity and remain additive.
    """

    def __init__(self) -> None:
        self.responses: dict[str, ClaudeCounters] = {}
        self.anonymous = ClaudeCounters()
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def add(self, entry: dict[str, Any]) -> None:
        patch = ClaudeCounters(
            first_int(entry, ("message", "usage", "input_tokens"), "inputTokens"),
            first_int(entry, ("message", "usage", "cache_creation_input_tokens")),
            first_int(entry, ("message", "usage", "cache_read_input_tokens")),
            first_int(entry, ("message", "usage", "output_tokens"), "outputTokens"),
        )
        if all(value is None for value in patch.cells()):
            return
        response_id = first_of(entry, ("message", "id"))
        previous = self.responses.get(response_id, ClaudeCounters())
        for index, value in enumerate(patch.cells()):
            if value is None:
                continue
            delta = value - (previous.cells()[index] or 0)
            if index == 3:
                self.output_tokens = accumulate_metric(self.output_tokens, delta)
            else:
                self.input_tokens = accumulate_metric(self.input_tokens, delta)
        if response_id is not None:
            self.responses[response_id] = previous.patched(patch)
        else:
            self.anonymous = ClaudeCounters(
                *(
                    old if new is None else (old or 0) + new
                    for old, new in zip(self.anonymous.cells(), patch.cells(), strict=True)
                )
            )

    def projection(self) -> tuple[ClaudeResponseUsage, ...]:
        return tuple(ClaudeResponseUsage(key, value) for key, value in self.responses.items())


def _usage_scope(state: Mapping[str, Any], *, offset: int) -> str | None:
    if offset == 0:
        return None
    raw = state.get("usage_scope")
    if (
        not isinstance(raw, dict)
        or set(raw) != {"format", "token"}
        or type(raw["format"]) is not int
        or raw["format"] != 1
        or (raw["token"] is not None and (not isinstance(raw["token"], str) or not raw["token"]))
    ):
        raise UnsupportedResumeState("unsupported Claude usage scope")
    return raw["token"]


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
        target: ChunkTarget | None = None,
    ) -> ParseResult:
        state = read_resume_state(resume_state, offset=offset, supported=frozenset({"usage_scope"}))
        scope_token = _usage_scope(state, offset=offset)
        usage = _ResponseUsage()
        metadata = read_resume_metadata(state, offset=offset)
        absolute_path = str(path.expanduser().resolve())
        session_id_value = make_session_id(self.source.value, absolute_path)
        stat = path.stat()
        file_mtime = stat.st_mtime
        file_size = stat.st_size
        is_full_parse = offset == 0
        budget = ChunkBudget(target if not is_full_parse else None)

        messages: list[Message] = []
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        stop_markers: list[StopMarker] = []
        is_complete = True

        with JsonlCapture(path, offset=offset) as capture:
            diagnostics = capture.diagnostics

            def checkpoint() -> NormalizationCheckpoint | None:
                return resume_checkpoint(
                    capture,
                    diagnostics=diagnostics,
                    metadata=metadata,
                    adapter_state={"usage_scope": {"format": 1, "token": scope_token}},
                    parser_revision=parser_revision(type(self)),
                    resumable=_resumable(),
                    message_idx_base=message_idx_base + len(messages),
                    orphan_tool_call_idx_base=orphan_tool_call_idx_base,
                )

            for entry in capture.records(
                stop=lambda: budget.stop(
                    capture,
                    identities=len(messages)
                    + len(tool_calls)
                    + len(tool_results)
                    + len(stop_markers),
                    checkpoint=checkpoint,
                )
            ):
                record_type = entry.get("type")
                if record_type in _REQUEST_LOG_RECORD_TYPES:
                    continue
                if record_type not in _KNOWN_RECORD_TYPES and "role" not in entry:
                    diagnostics.append(
                        ParseDiagnostic(
                            "unsupported_record",
                            capture.record_start,
                            f"record: {record_type!r}",
                        )
                    )
                    continue
                # -- Timestamp: try root-level fields --
                timestamp = parse_timestamp(first_of(entry, "timestamp", "created_at"))
                if timestamp is not None:
                    metadata.started_at = (
                        timestamp
                        if metadata.started_at is None
                        else min(metadata.started_at, timestamp)
                    )
                    metadata.ended_at = (
                        timestamp
                        if metadata.ended_at is None
                        else max(metadata.ended_at, timestamp)
                    )

                # -- Session-level metadata: cascading extraction --
                # Each field tries newest format first, falling back to legacy.
                # Elect into the native accumulator, including when the first
                # value follows a checkpoint. Suffixes restore the same values.
                if metadata.model is None:
                    metadata.model = first_of(entry, ("message", "model"), "model", "model_name")
                if metadata.cwd is None:
                    metadata.cwd = first_of(entry, "cwd", "working_directory")
                if metadata.git_repo is None:
                    metadata.git_repo = first_of(entry, "git_root", "repo")
                if metadata.git_branch is None:
                    metadata.git_branch = first_of(
                        entry, "gitBranch", ("git", "branch"), "git_branch"
                    )
                if metadata.source_session_id is None:
                    metadata.source_session_id = first_of(entry, "sessionId")

                usage.add(entry)

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
                            metadata.started_at = (
                                progress_ts
                                if metadata.started_at is None
                                else min(metadata.started_at, progress_ts)
                            )
                            metadata.ended_at = (
                                progress_ts
                                if metadata.ended_at is None
                                else max(metadata.ended_at, progress_ts)
                            )

                        usage.add({"message": nested_payload})

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
        published = metadata.model_copy()
        if published.git_repo is None and published.cwd is not None:
            published.git_repo = published.cwd

        published.source_session_id = metadata.source_session_id or path.stem

        message_count = len(messages)
        tool_count = len(tool_calls)
        duration_seconds = None
        if metadata.started_at and metadata.ended_at:
            duration_seconds = int((metadata.ended_at - metadata.started_at).total_seconds())

        next_byte_offset = capture.next_byte_offset
        is_complete = is_complete and not diagnostics
        session = Session(
            **published.model_dump(),
            id=session_id_value,
            source=self.source,
            source_path=absolute_path,
            duration_seconds=duration_seconds,
            message_count=message_count,
            tool_count=tool_count,
            input_tokens=usage.input_tokens if is_full_parse else None,
            output_tokens=usage.output_tokens if is_full_parse else None,
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
            reached_end=capture.reached_end,
            committed_prefix_sha256=capture.committed_prefix_sha256,
            is_full_parse=is_full_parse,
            normalization_checkpoint=checkpoint(),
            claude_usage=(
                ClaudeUsageFull(usage.projection())
                if is_full_parse
                else ClaudeUsageSuffix(
                    scope_token,
                    offset,
                    capture.initial_prefix_sha256,
                    usage.projection(),
                    usage.anonymous,
                )
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


def _resumable() -> bool:
    return True
