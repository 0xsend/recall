from __future__ import annotations

import contextlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
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
    home_root_overridden,
    resolve_home,
    summarize_tool_result,
)
from recall.parsers.protocol import (
    default_discover,
    default_live_candidates,
    default_watch_roots,
)
from recall.parsers.revision import parser_revision

# Kimi Code's `step.end` finish reason for a completed turn. Every other
# value it emits (`tool_calls`, `length`) stops the step for a reason that
# leaves the agent owing the conversation another message.
_TURN_ENDING_FINISH_REASON = "stop"

# Control and telemetry records. Acknowledged without diagnostics; they are
# not conversation content. turn.prompt / turn.steer / prompt.steered
# duplicate context.append_message (a steered message is re-recorded there).
# agent.turn.started only opens the turn that agent.turn.ended closes.
# subagent.spawned / .started / .completed are delegation lifecycle: the
# delegation is the parent's Agent tool call, the completed record's
# resultSummary is verbatim inside that call's tool.result output, and its
# usage belongs to the subagent's own wire file (indexed as its own session),
# so nothing is indexed from them here. full_compaction.begin / .complete
# only bracket the context.apply_compaction record that carries the summary.
# context.undo drops turns from the model's future context, not from the
# recorded transcript, so the undone turns stay indexed. Unknown types stay
# diagnostic.
_CONTROL_RECORD_TYPES = frozenset(
    {
        "agent.turn.started",
        "config.update",
        "context.undo",
        "file_history.checkpoint",
        "file_history.tracked",
        "full_compaction.begin",
        "full_compaction.complete",
        "interaction.request",
        "interaction.resolved",
        "interruptionReminder.recorded",
        "llm.tools_snapshot",
        "mcp.tools_discovered",
        "metadata",
        "permission.record_approval_result",
        "permission.set_mode",
        "plan_mode.cancel",
        "plan_mode.enter",
        "plan_mode.exit",
        "plugin.session_start",
        "profile.bind",
        "prompt.aborted",
        "prompt.accepted",
        "prompt.completed",
        "prompt.steered",
        "runtime.set_binding",
        "staleGuard.recorded",
        "subagent.completed",
        "subagent.spawned",
        "subagent.started",
        "swarm_mode.enter",
        "swarm_mode.exit",
        "task.started",
        "task.terminated",
        "task.waitDelivered",
        "token_counting.measured",
        "token_counting.truncated",
        "token_counting.turn_recorded",
        "tools.set_active_tools",
        "tools.update_store",
        "turn.cancel",
        "turn.ended",
        "turn.prompt",
        "turn.steer",
        "turn.step.interrupted",
        "turn.step.retrying",
    }
)


@dataclass
class KimiCodeParser:
    """Parser for Kimi Code CLI session wire logs.

    Sessions live at $KIMI_CODE_HOME/sessions/<workDirKey>/<sessionId>/agents/<agent>/wire.jsonl
    (default $KIMI_CODE_HOME = ~/.kimi-code). Sub-agents get their own
    agents/agent-N/wire.jsonl; every wire file is indexed as its own session.
    The session directory's state.json carries the real cwd ("workDir") — the
    workDirKey path component is a non-reversible slug+hash.

    Each line is one JSON event with a millisecond epoch "time". Relevant
    event types:

    - context.append_message: user message, message.content = [{"type":"text","text":...}]
    - context.append_loop_event: LLM step stream — step.begin/step.end bracket a
      step; content.part carries {"type":"think"|"text"} parts; tool.call carries
      {"name", "args", "toolCallId"}; tool.result carries {"toolCallId", "result"}
    - llm.request / profile.bind: model identity ("modelAlias", "model").
      profile.bind.systemPrompt is not ingested as a message.
    - usage.record: per-step token deltas
      {"inputOther", "inputCacheRead", "inputCacheCreation", "output"}.
      Deltas are additive, so kimi_code is deliberately NOT in
      ABSOLUTE_TOKEN_SOURCES (which is for cumulative-total sources).
    - context.apply_compaction: context-window compaction "summary", ingested
      as a SYSTEM message. Messages recorded before it stay indexed, and the
      kept user messages Kimi re-sends after it are indexed as recorded.
    - agent.message.appended: one append to the agent's message store
      (message.message = {role, content, toolCalls, toolCallId},
      message.meta.source = input|llm|tool|notify). input/llm/tool records
      mirror context.append_message and the step stream, so they are deduped
      against what those already indexed; a record with no wire mirror
      (observed: "notify" background-task notifications the stream never
      carried) is indexed from this record. Incremental suffix parses skip
      the record entirely: dedup state cannot see the committed prefix, the
      wire stream carries the mirrored content, and anything unmirrored is
      picked up by the next full parse.
    - agent.turn.started: turn lifecycle marker (turnId, queueItemId);
      nothing to index. agent.turn.ended closes the turn — its outcome
      ("done", "failed") becomes an ends_turn stop marker on the last
      message.
    - subagent.spawned / .started / .completed: delegation lifecycle.
      spawned's routing metadata (subagentId, taskId, model) adds nothing
      to the parent's Agent tool call, which the step stream already
      indexed with description/prompt/subagent_type. completed's
      resultSummary is verbatim inside that call's tool.result output, and
      its usage/contextTokens are the subagent's own — the subagent's
      wire file is indexed as its own session, so counting them here would
      double-count. None of the three is indexed.

    Control and telemetry records are skipped without diagnostics.
    turn.prompt / turn.steer / prompt.steered duplicate context.append_message.
    """

    source: Source = Source.KIMI_CODE
    # Discovery roots from `[sources.<name>] roots`; None = default_roots(),
    # an empty tuple = configured to scan nothing.
    roots: tuple[Path, ...] | None = None

    @property
    def file_pattern(self) -> str:
        return "wire.jsonl"

    def default_roots(self) -> list[Path]:
        return [_sessions_root()]

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
        # An assistant step accumulates parts across records until a boundary
        # flushes them into one message. The adapter declines at an open step
        # instead of carrying that half-built message (REQ-INDEX-026).
        read_resume_state(resume_state, offset=offset, supported=frozenset())
        absolute_path = str(path.expanduser().resolve())
        session_id_value = make_session_id(self.source.value, absolute_path)
        stat = path.stat()
        file_mtime = stat.st_mtime
        file_size = stat.st_size
        is_full_parse = offset == 0

        # .../sessions/<workDirKey>/<sessionId>/agents/<agent>/wire.jsonl
        agent_name = path.parent.name
        agent_id = agent_name if agent_name != "main" else None
        session_dir = path.parents[2] if len(path.parents) > 2 else None
        source_session_id = session_dir.name if session_dir is not None else None

        messages: list[Message] = []
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []
        stop_markers: list[StopMarker] = []
        is_complete = True
        started_at: datetime | None = None
        ended_at: datetime | None = None
        model: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        # Open assistant step: parts accumulate until the step boundary flushes
        # them into a single ASSISTANT message (one LLM response = one message).
        step_text: list[str] = []
        step_thinking: list[str] = []
        step_tool_calls: list[ToolCall] = []
        step_timestamp: datetime | None = None
        # `step.begin` seen without its `step.end`. Tracked apart from the
        # accumulators because a step that has produced no part yet is just as
        # open as one mid-sentence, and flushing clears the accumulators.
        step_open = False
        # Mirror dedup for agent.message.appended (full parses only; a suffix
        # parse skips those records because this state cannot see the
        # committed prefix). Fingerprints of flushed steps, (role, content)
        # of plain messages each stream indexed, and recorded tool result ids.
        step_fingerprints: Counter[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = (
            Counter()
        )
        context_plain: Counter[tuple[str, str]] = Counter()
        agent_plain: Counter[tuple[str, str]] = Counter()
        tool_result_ids: set[str] = set()

        def commit_step_message(
            texts: list[str],
            thinking: list[str],
            calls: list[ToolCall],
            timestamp: datetime | None,
        ) -> None:
            idx = message_idx_base + len(messages)
            msg = Message(
                id=make_message_id(session_id_value, idx),
                session_id=session_id_value,
                idx=idx,
                role=Role.ASSISTANT,
                content="\n".join(texts) if texts else None,
                thinking="\n".join(thinking) if thinking else None,
                timestamp=timestamp,
                has_thinking=bool(thinking),
                agent_id=agent_id,
                tool_calls=list(calls),
            )
            messages.append(msg)
            for tool_idx, tool_call in enumerate(calls):
                tool_call.idx = tool_idx
                tool_call.id = make_tool_call_id(msg.id, tool_idx)
                tool_call.session_id = session_id_value
                tool_call.message_id = msg.id
                tool_call.agent_id = agent_id
                tool_calls.append(tool_call)
            step_fingerprints[_step_fingerprint(texts, thinking, calls)] += 1

        def flush_step() -> None:
            nonlocal step_timestamp
            if not (step_text or step_thinking or step_tool_calls):
                return
            commit_step_message(step_text, step_thinking, step_tool_calls, step_timestamp)
            step_text.clear()
            step_thinking.clear()
            step_tool_calls.clear()
            step_timestamp = None

        def append_plain(role: Role, content: str | None, timestamp: datetime | None) -> None:
            idx = message_idx_base + len(messages)
            messages.append(
                Message(
                    id=make_message_id(session_id_value, idx),
                    session_id=session_id_value,
                    idx=idx,
                    role=role,
                    content=content,
                    thinking=None,
                    timestamp=timestamp,
                    has_thinking=False,
                    agent_id=agent_id,
                    tool_calls=[],
                )
            )

        with JsonlCapture(path, offset=offset) as capture:
            diagnostics = capture.diagnostics
            for entry in capture.records():
                timestamp = _ms_to_datetime(entry.get("time"))
                if timestamp is None:
                    timestamp = _ms_to_datetime(entry.get("created_at"))
                if timestamp is not None:
                    if is_full_parse:
                        started_at = timestamp if started_at is None else min(started_at, timestamp)
                    ended_at = timestamp if ended_at is None else max(ended_at, timestamp)

                entry_type = entry.get("type")

                if entry_type == "context.append_message":
                    flush_step()
                    payload = entry.get("message")
                    if not isinstance(payload, dict):
                        continue
                    try:
                        role = Role(str(payload.get("role") or "user"))
                    except ValueError:
                        diagnostics.append(
                            ParseDiagnostic(
                                "unsupported_record", capture.record_start, "unknown message role"
                            )
                        )
                        continue
                    content = _extract_text(payload.get("content"))
                    if content:
                        key = (role.value, content)
                        if agent_plain[key] > 0:
                            # agent.message.appended (which precedes this
                            # mirror in the file) already indexed it.
                            agent_plain[key] -= 1
                            continue
                        context_plain[key] += 1
                        append_plain(role, content, timestamp)

                elif entry_type == "context.append_loop_event":
                    event = entry.get("event")
                    if not isinstance(event, dict):
                        continue
                    event_type = event.get("type")

                    if event_type == "step.begin" or event_type == "step.end":
                        step_open = event_type == "step.begin"
                        flush_step()
                        finish_reason = event.get("finishReason")
                        if isinstance(finish_reason, str) and finish_reason:
                            # A step.end closes the step whose messages were
                            # just flushed, so it anchors to the last message.
                            stop_markers.append(
                                StopMarker(
                                    idx=message_idx_base + len(messages) - 1,
                                    reason=finish_reason,
                                    ends_turn=finish_reason == _TURN_ENDING_FINISH_REASON,
                                )
                            )
                    elif event_type == "content.part":
                        part = event.get("part")
                        if not isinstance(part, dict):
                            continue
                        if step_timestamp is None:
                            step_timestamp = timestamp
                        if part.get("type") == "think":
                            think = part.get("think")
                            if isinstance(think, str) and think:
                                step_thinking.append(think)
                        elif part.get("type") == "text":
                            text = part.get("text")
                            if isinstance(text, str) and text:
                                step_text.append(text)
                    elif event_type == "tool.call":
                        if step_timestamp is None:
                            step_timestamp = timestamp
                        name = str(event.get("name") or "")
                        args = event.get("args")
                        call_id = event.get("toolCallId")
                        step_tool_calls.append(
                            build_tool_call(
                                name, args, tool_use_id=str(call_id) if call_id else None
                            )
                        )
                    elif event_type == "tool.result":
                        flush_step()
                        output = _tool_result_text(event.get("result"))
                        tcid = event.get("toolCallId")
                        if isinstance(tcid, str) and tcid:
                            tool_result_ids.add(tcid)
                            tool_results.append(
                                ToolResult(
                                    tool_use_id=tcid,
                                    result_summary=summarize_tool_result(output),
                                    is_error=bool(event.get("isError", False)),
                                    completed_at=timestamp,
                                )
                            )
                            prefix = f"[tool_call_id: {tcid}]"
                            output = f"{prefix}\n{output}" if output else prefix
                        if output:
                            append_plain(Role.SYSTEM, output, timestamp)
                    else:
                        diagnostics.append(
                            ParseDiagnostic(
                                "unsupported_record",
                                capture.record_start,
                                f"loop event: {event_type!r}",
                            )
                        )

                elif entry_type in {"llm.request", "profile.bind"}:
                    # First-wins over the file: a suffix that re-elected the
                    # model would overwrite the committed one through the
                    # incremental COALESCE merge (REQ-INDEX-026).
                    if is_full_parse and model is None:
                        candidate = entry.get("modelAlias") or entry.get("model")
                        if isinstance(candidate, str) and candidate:
                            model = candidate

                elif entry_type == "usage.record":
                    usage = entry.get("usage")
                    if isinstance(usage, dict):
                        # Cache tokens are separate counters in the Kimi usage
                        # model, not subsets of inputOther — sum all three for
                        # total billable input tokens (same rule as Claude Code).
                        for key in ("inputOther", "inputCacheRead", "inputCacheCreation"):
                            input_tokens = accumulate_metric(input_tokens, usage.get(key))
                        output_tokens = accumulate_metric(output_tokens, usage.get("output"))

                elif entry_type == "context.apply_compaction":
                    flush_step()
                    summary = entry.get("summary")
                    if isinstance(summary, str) and summary:
                        append_plain(Role.SYSTEM, summary, timestamp)

                elif entry_type == "agent.message.appended":
                    payload = entry.get("message")
                    inner = payload.get("message") if isinstance(payload, dict) else None
                    if not isinstance(inner, dict):
                        continue
                    if not is_full_parse:
                        # Every observed record mirrors a wire record that may
                        # live in the committed prefix, which this parse's
                        # dedup state cannot see. The wire stream indexes the
                        # mirrored content; the rare unmirrored record (a
                        # "notify" the stream never carried) is picked up by
                        # the next full parse.
                        continue
                    raw_role = str(inner.get("role") or "user")
                    if raw_role == "assistant":
                        flush_step()
                        texts: list[str] = []
                        thinking: list[str] = []
                        record_calls: list[ToolCall] = []
                        content = inner.get("content")
                        if isinstance(content, list):
                            for part in content:
                                if not isinstance(part, dict):
                                    continue
                                if part.get("type") == "think":
                                    think = part.get("think")
                                    if isinstance(think, str) and think:
                                        thinking.append(think)
                                elif part.get("type") == "text":
                                    text = part.get("text")
                                    if isinstance(text, str) and text:
                                        texts.append(text)
                        record_tool_calls = inner.get("toolCalls")
                        if isinstance(record_tool_calls, list):
                            for call in record_tool_calls:
                                if not isinstance(call, dict):
                                    continue
                                # arguments is a JSON-encoded string, unlike
                                # the loop stream's already-decoded args.
                                arguments = call.get("arguments")
                                if isinstance(arguments, str):
                                    with contextlib.suppress(json.JSONDecodeError):
                                        arguments = json.loads(arguments)
                                call_id = call.get("id")
                                record_calls.append(
                                    build_tool_call(
                                        str(call.get("name") or ""),
                                        arguments,
                                        tool_use_id=str(call_id) if call_id else None,
                                    )
                                )
                        fingerprint = _step_fingerprint(texts, thinking, record_calls)
                        if step_fingerprints[fingerprint] > 0:
                            # The step stream already indexed this message.
                            step_fingerprints[fingerprint] -= 1
                            continue
                        if texts or thinking or record_calls:
                            commit_step_message(texts, thinking, record_calls, timestamp)
                    elif raw_role == "tool":
                        flush_step()
                        tcid = inner.get("toolCallId")
                        if not (isinstance(tcid, str) and tcid):
                            diagnostics.append(
                                ParseDiagnostic(
                                    "unsupported_record",
                                    capture.record_start,
                                    "agent message: tool result without toolCallId",
                                )
                            )
                            continue
                        if tcid in tool_result_ids:
                            continue
                        tool_result_ids.add(tcid)
                        output = _extract_text(inner.get("content")) or ""
                        tool_results.append(
                            ToolResult(
                                tool_use_id=tcid,
                                result_summary=summarize_tool_result(output),
                                is_error=False,
                                completed_at=timestamp,
                            )
                        )
                        prefix = f"[tool_call_id: {tcid}]"
                        output = f"{prefix}\n{output}" if output else prefix
                        append_plain(Role.SYSTEM, output, timestamp)
                    else:
                        try:
                            role = Role(raw_role)
                        except ValueError:
                            diagnostics.append(
                                ParseDiagnostic(
                                    "unsupported_record",
                                    capture.record_start,
                                    "unknown agent message role",
                                )
                            )
                            continue
                        content = _extract_text(inner.get("content"))
                        if not content:
                            continue
                        key = (role.value, content)
                        if context_plain[key] > 0:
                            # context.append_message already indexed it.
                            context_plain[key] -= 1
                            continue
                        flush_step()
                        agent_plain[key] += 1
                        append_plain(role, content, timestamp)

                elif entry_type == "agent.turn.ended":
                    # The record itself declares the turn boundary, whatever
                    # the outcome ("done", "failed"); outcome is the harness's
                    # own vocabulary for the marker reason.
                    flush_step()
                    outcome = entry.get("outcome")
                    if (
                        isinstance(outcome, str)
                        and outcome
                        and message_idx_base + len(messages) > 0
                    ):
                        stop_markers.append(
                            StopMarker(
                                idx=message_idx_base + len(messages) - 1,
                                reason=outcome,
                                ends_turn=True,
                            )
                        )

                elif entry_type not in _CONTROL_RECORD_TYPES:
                    diagnostics.append(
                        ParseDiagnostic(
                            "unsupported_record", capture.record_start, f"record: {entry_type!r}"
                        )
                    )

            # Snapshot before the final flush clears the accumulators.
            step_open_at_boundary = step_open or bool(step_text or step_thinking or step_tool_calls)
            flush_step()

        duration_seconds = None
        if started_at is not None and ended_at is not None:
            duration_seconds = max(0, int((ended_at - started_at).total_seconds()))

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
            cwd=_read_work_dir(path),
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
                resumable=not step_open_at_boundary,
                diagnostics=diagnostics,
                message_idx_base=message_idx_base + len(messages),
                orphan_tool_call_idx_base=orphan_tool_call_idx_base,
            ),
            tail_facts=TailFacts(
                tool_results=tuple(tool_results),
                stop_markers=tuple(stop_markers),
                # `step.end` closes a step, not the session.
                session_ended=False,
            ),
        )


def _step_fingerprint(
    texts: list[str],
    thinking: list[str],
    calls: list[ToolCall],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Content identity of a flushed step, matched by agent.message.appended mirrors."""
    return (tuple(texts), tuple(thinking), tuple(call.tool_use_id or "" for call in calls))


def _sessions_root() -> Path:
    """Kimi Code sessions root under active home / KIMI_CODE_HOME.

    Multi-host ``use_home_root`` wins so fleet trees use ``<root>/.kimi-code``.
    Outside that, ``KIMI_CODE_HOME`` relocates the install (single-host).
    """
    if home_root_overridden():
        return resolve_home() / ".kimi-code" / "sessions"
    home_override = os.environ.get("KIMI_CODE_HOME")
    base = Path(home_override) if home_override else resolve_home() / ".kimi-code"
    return base / "sessions"


def _ms_to_datetime(value: Any) -> datetime | None:
    """Convert a millisecond epoch timestamp to a tz-aware UTC datetime."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _extract_text(content: Any) -> str | None:
    """Extract plain text from a Kimi message content list of text parts."""
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        parts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
            and item["text"]
        ]
        joined = "\n".join(parts).strip()
        return joined if joined else None
    return None


def _tool_result_text(result: Any) -> str:
    """Flatten a tool.result payload to display text."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("output", "error", "content"):
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
        return json.dumps(result, ensure_ascii=False)
    if result is None:
        return ""
    return str(result)


def _read_work_dir(wire_path: Path) -> str | None:
    """Read the real cwd from the session's state.json ("workDir"), best effort."""
    try:
        state_path = wire_path.parents[2] / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, IndexError):
        return None
    work_dir = state.get("workDir") if isinstance(state, dict) else None
    return work_dir if isinstance(work_dir, str) and work_dir else None
