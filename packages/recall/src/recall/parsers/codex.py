from __future__ import annotations

import hashlib
import json
import shlex
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
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
from recall.parsers.checkpoint import (
    UnsupportedResumeState,
    read_resume_flag,
    read_resume_state,
    resume_checkpoint,
)
from recall.parsers.common import (
    build_tool_call,
    extract_content_blocks,
    first_int,
    parse_timestamp,
    resolve_home,
    summarize_tool_result,
)
from recall.parsers.js_object import parse_exec_wrapper
from recall.parsers.protocol import (
    default_discover,
    default_live_candidates,
    default_watch_roots,
)
from recall.parsers.revision import parser_revision

# Keys of the state a Codex boundary hands to its suffix.
_TURN_IN_PROGRESS = "turn_in_progress"
_CORRELATION = "correlation"

# Execution views of a wrapper program's decoded calls are named in their own
# identity namespace (REQ-PARSE-018), which is what lets a suffix reason about
# them without the whole file's tool identities.
_EXEC_IDENTITY_PREFIX = "exec-"


def _require(condition: bool, detail: str) -> None:
    """Reject a stored correlation shape this build cannot honor."""
    if not condition:
        raise UnsupportedResumeState(f"codex correlation state: {detail}")


def _str_list(raw: Any, detail: str) -> list[str]:
    _require(isinstance(raw, list), detail)
    _require(all(isinstance(item, str) for item in raw), detail)
    return list(raw)


def _fixed_strings(raw: Any, size: int, detail: str) -> list[str]:
    _require(isinstance(raw, list) and len(raw) == size, detail)
    _require(all(isinstance(item, str) for item in raw), detail)
    return list(raw)


def _read_anchor(raw: Any) -> tuple[str, str, str] | None:
    if raw is None:
        return None
    turn, digest, timestamp = _fixed_strings(raw, 3, "anchor must be 3 strings")
    return turn, digest, timestamp


def _read_pending_alias(raw: Any) -> tuple[str, str, str, str] | None:
    if raw is None:
        return None
    alias, canonical, digest, timestamp = _fixed_strings(raw, 4, "alias_pending must be 4 strings")
    return alias, canonical, digest, timestamp


@dataclass
class _CodexCorrelation:
    """The rollout normalizer's cross-record pairing state (REQ-INDEX-026).

    A modern rollout describes one conversation twice -- as SDK
    ``response_item`` records and as native ``event_msg`` mirrors -- and the
    normalizer pairs them off so only one copy becomes a message. Every
    sampled real rollout is mixed, so declining on mixed format would decline
    on every modern transcript. Instead the boundary carries the pairing that
    is still *open*, which is what a suffix cannot rebuild.

    Only unresolved state is carried. A message key is held as a digest rather
    than its content, and a turn/key pair whose two representations have
    balanced leaves no entry at all, so the envelope tracks the conversation's
    open correlations rather than its length. Adapter state is compressed when
    needed before applying the 64 KiB envelope backstop; genuinely larger open
    state declines with an operator-visible warning.

    ``balances`` is the signed ``responses - mirrors`` count per (turn, key).
    It reproduces the pairwise counter comparison exactly: a response
    duplicates when the balance is already negative, a mirror when it is
    already positive, and each record moves the balance one step toward zero
    or away from it.
    """

    turn_id: str | None = None
    aliases: dict[str, str] = field(default_factory=dict)
    balances: Counter[tuple[str, str]] = field(default_factory=Counter)
    correlated_ids: set[str] = field(default_factory=set)
    user_anchor: tuple[str, str, str] | None = None
    pending_alias: tuple[str, str, str, str] | None = None
    # Turns that ran an exec wrapper, so a later execution view of it is not
    # counted as a second top-level invocation.
    exec_turns: set[str] = field(default_factory=set)
    # An SDK record claimed an identity in the execution-view namespace. Never
    # observed, and it is the one thing that would make an `exec-` identity
    # ambiguous, so a suffix stops trusting the namespace once it happens.
    exec_namespace_claimed: bool = False

    def to_state(self) -> dict[str, Any]:
        """The open pairing, as JSON data; empty when nothing is open."""
        state: dict[str, Any] = {}
        if self.turn_id is not None:
            state["turn"] = self.turn_id
        if self.aliases:
            state["aliases"] = dict(self.aliases)
        if self.balances:
            state["balances"] = [
                [turn, digest, balance] for (turn, digest), balance in sorted(self.balances.items())
            ]
        if self.correlated_ids:
            state["ids"] = sorted(self.correlated_ids)
        if self.user_anchor is not None:
            state["anchor"] = list(self.user_anchor)
        if self.pending_alias is not None:
            state["alias_pending"] = list(self.pending_alias)
        if self.exec_turns:
            state["exec_turns"] = sorted(self.exec_turns)
        if self.exec_namespace_claimed:
            state["exec_claimed"] = True
        return state

    @classmethod
    def from_state(cls, raw: Any) -> _CodexCorrelation:
        """Rebuild the open pairing, refusing any shape this build cannot read."""
        _require(isinstance(raw, dict), "expected an object")
        unsupported = sorted(
            key
            for key in raw
            if key
            not in {
                "turn",
                "aliases",
                "balances",
                "ids",
                "anchor",
                "alias_pending",
                "exec_turns",
                "exec_claimed",
            }
        )
        _require(not unsupported, f"unsupported keys {unsupported}")

        turn_id = raw.get("turn")
        _require(turn_id is None or isinstance(turn_id, str), "turn must be a string")

        aliases = raw.get("aliases", {})
        _require(isinstance(aliases, dict), "aliases must be an object")
        _require(
            all(isinstance(key, str) and isinstance(value, str) for key, value in aliases.items()),
            "aliases must map strings to strings",
        )

        raw_balances = raw.get("balances", [])
        _require(isinstance(raw_balances, list), "balances must be a list")
        balances: Counter[tuple[str, str]] = Counter()
        for item in raw_balances:
            _require(isinstance(item, list) and len(item) == 3, "balance must be a triple")
            turn, digest, count = item
            _require(isinstance(turn, str) and isinstance(digest, str), "balance names strings")
            _require(
                len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
                "balance digest must be a lowercase SHA-256",
            )
            _require(isinstance(count, int) and not isinstance(count, bool), "balance is an int")
            # A zero balance is indistinguishable from an absent one, so
            # storing one means the writer disagrees with this build's rules.
            _require(count != 0, "balance must be non-zero")
            balance_key = (turn, digest)
            _require(balance_key not in balances, "balance keys must be unique")
            balances[balance_key] = count

        exec_claimed = raw.get("exec_claimed", False)
        _require(isinstance(exec_claimed, bool), "exec_claimed must be a boolean")
        exec_turns = _str_list(raw.get("exec_turns", []), "exec_turns must be strings")
        correlated_ids = _str_list(raw.get("ids", []), "ids must be strings")

        anchor = raw.get("anchor")
        pending_alias = raw.get("alias_pending")
        return cls(
            exec_turns=set(exec_turns),
            exec_namespace_claimed=exec_claimed,
            turn_id=turn_id,
            aliases=dict(aliases),
            balances=balances,
            correlated_ids=set(correlated_ids),
            user_anchor=_read_anchor(anchor),
            pending_alias=_read_pending_alias(pending_alias),
        )


@dataclass
class CodexParser:
    source: Source = Source.CODEX
    # Discovery roots from `[sources.<name>] roots`; None = default_roots(),
    # an empty tuple = configured to scan nothing.
    roots: tuple[Path, ...] | None = None

    @property
    def file_pattern(self) -> str:
        return "rollout*.jsonl"

    def default_roots(self) -> list[Path]:
        return [resolve_home() / ".codex/sessions"]

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
        carried = read_resume_state(
            resume_state, offset=offset, supported=frozenset({_TURN_IN_PROGRESS, _CORRELATION})
        )
        correlation = _CodexCorrelation.from_state(carried.get(_CORRELATION, {}))
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
        # A `task_started` before the boundary still governs the first appended
        # assistant message's marker, so the checkpoint carries it forward.
        turn_in_progress = read_resume_flag(carried, _TURN_IN_PROGRESS)
        orphan_tool_calls: list[ToolCall] = []
        orphan_positions: list[int] = []
        tool_result_positions: list[int] = []
        is_complete = True
        started_at: datetime | None = None
        ended_at: datetime | None = None
        cwd: str | None = None
        git_branch: str | None = None
        git_repo: str | None = None
        source_session_id: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        model: str | None = None

        with JsonlCapture(path, offset=offset) as capture:
            diagnostics = capture.diagnostics
            for entry in _conversation_records(
                capture, correlation=correlation, resumed=not is_full_parse
            ):
                orphan_start = len(orphan_tool_calls)
                tool_result_start = len(tool_results)
                message_start = len(messages)
                position = int(entry["_record_position"])
                timestamp = parse_timestamp(entry.get("timestamp"))
                if timestamp is not None:
                    if is_full_parse:
                        started_at = timestamp if started_at is None else min(started_at, timestamp)
                    ended_at = timestamp if ended_at is None else max(ended_at, timestamp)

                entry_type = entry.get("type")
                if entry_type == "session_meta":
                    payload = entry.get("payload", {})
                    meta_id = payload.get("id")
                    # A forked subagent rollout opens with its own meta and then
                    # embeds its parent's; the parent's describes another thread.
                    if source_session_id is not None and meta_id not in (None, source_session_id):
                        continue
                    source_session_id = meta_id or source_session_id
                    cwd = payload.get("cwd") or cwd
                    git_raw = payload.get("git")
                    git_info = git_raw if isinstance(git_raw, dict) else {}
                    git_branch = git_info.get("branch") or git_branch
                    git_repo = git_info.get("root") or git_repo
                    meta_ts = parse_timestamp(payload.get("timestamp"))
                    if meta_ts is not None:
                        if is_full_parse:
                            started_at = meta_ts if started_at is None else min(started_at, meta_ts)
                        ended_at = meta_ts if ended_at is None else max(ended_at, meta_ts)
                elif entry_type == "token_usage_record":
                    total = entry.get("payload", {}).get("thread_token_usage")
                    if isinstance(total, dict):
                        ti = first_int(total, "input_tokens")
                        to = first_int(total, "output_tokens")
                        if ti is not None:
                            input_tokens = ti if input_tokens is None else max(input_tokens, ti)
                        if to is not None:
                            output_tokens = to if output_tokens is None else max(output_tokens, to)
                elif entry_type == "turn_context":
                    payload = entry.get("payload", {})
                    model = payload.get("model") or model
                    cwd = payload.get("cwd") or cwd
                elif entry_type == "event_msg":
                    payload = entry.get("payload", {})
                    payload_type = payload.get("type")
                    if payload_type == "user_message":
                        message = _build_plain_message(
                            role=Role.USER,
                            text=str(payload.get("message", "")),
                            session_id=session_id_value,
                            idx=message_idx_base + len(messages),
                            timestamp=timestamp,
                        )
                        messages.append(message)
                    elif payload_type == "agent_message":
                        message = _build_plain_message(
                            role=Role.ASSISTANT,
                            text=str(payload.get("message", "")),
                            session_id=session_id_value,
                            idx=message_idx_base + len(messages),
                            timestamp=timestamp,
                        )
                        messages.append(message)
                    elif payload_type == "function_call":
                        tool_name = str(payload.get("name", ""))
                        tool_input = payload.get("parameters")
                        tool_call = build_tool_call(tool_name, tool_input)
                        orphan_tool_calls.append(tool_call)
                    elif payload_type == "task_started":
                        turn_in_progress = True
                    elif payload_type == "task_complete":
                        turn_in_progress = False
                        # Codex's end-of-turn marker. It is an event, not a
                        # message, so it anchors to the last message parsed.
                        stop_markers.append(
                            StopMarker(
                                idx=message_idx_base + len(messages) - 1,
                                reason="task_complete",
                                ends_turn=True,
                            )
                        )
                    elif payload_type == "turn_aborted":
                        turn_in_progress = False
                        # Interrupted turns never emit task_complete. The
                        # harness reason is passed through; the turn is over.
                        if messages:
                            reason = payload.get("reason")
                            if not isinstance(reason, str) or not reason:
                                reason = "turn_aborted"
                            stop_markers.append(
                                StopMarker(
                                    idx=message_idx_base + len(messages) - 1,
                                    reason=reason,
                                    ends_turn=True,
                                )
                            )
                    elif payload_type == "token_count":
                        info = payload.get("info")
                        total = info.get("total_token_usage") if isinstance(info, dict) else None
                        if isinstance(total, dict):
                            ti = first_int(total, "input_tokens")
                            if ti is not None:
                                input_tokens = ti if input_tokens is None else max(input_tokens, ti)
                            to = first_int(total, "output_tokens")
                            if to is not None:
                                output_tokens = (
                                    to if output_tokens is None else max(output_tokens, to)
                                )
                    elif payload_type not in {
                        "thread_settings_applied",
                        "thread_goal_updated",
                        "context_compacted",
                        "agent_reasoning",
                        "item_started",
                    }:
                        diagnostics.append(
                            ParseDiagnostic(
                                "unsupported_record",
                                capture.record_start,
                                f"event_msg: {payload_type!r}",
                            )
                        )
                elif entry_type == "response_item":
                    payload = entry.get("payload", {})
                    payload_type = payload.get("type")
                    if payload_type == "message":
                        role_value = payload.get("role", "user")
                        try:
                            role = Role(role_value)
                        except ValueError:
                            is_complete = False
                            diagnostics.append(
                                ParseDiagnostic(
                                    "unsupported_record",
                                    capture.record_start,
                                    f"unknown role: {role_value!r}",
                                )
                            )
                            continue
                        blocks = extract_content_blocks(payload.get("content"))
                        messages.append(
                            Message(
                                id=make_message_id(
                                    session_id_value, message_idx_base + len(messages)
                                ),
                                session_id=session_id_value,
                                idx=message_idx_base + len(messages),
                                role=role,
                                content="\n".join(blocks.text) if blocks.text else None,
                                thinking="\n".join(blocks.thinking) if blocks.thinking else None,
                                timestamp=timestamp,
                                has_thinking=bool(blocks.thinking),
                                tool_calls=blocks.tool_calls,
                                agent_id=payload.get("author"),
                            )
                        )
                    elif payload_type == "reasoning":
                        # Reasoning items are internal model traces. They are
                        # neither a user-visible turn nor an unsupported
                        # record; assistant message content carries any
                        # durable thinking blocks we can safely index.
                        continue
                    elif payload_type in {"compaction", "compaction_summary"}:
                        summary = _compaction_summary(payload)
                        if summary is not None:
                            messages.append(
                                _build_plain_message(
                                    role=Role.SYSTEM,
                                    text=summary,
                                    session_id=session_id_value,
                                    idx=message_idx_base + len(messages),
                                    timestamp=timestamp,
                                )
                            )
                    elif payload_type == "function_call":
                        tool_name = str(payload.get("name", ""))
                        tool_input = _parse_function_call_arguments(payload.get("arguments"))
                        orphan_tool_calls.append(
                            build_tool_call(tool_name, tool_input, tool_use_id=_call_id(payload))
                        )
                    elif payload_type == "custom_tool_call":
                        tool_name = str(payload.get("name", ""))
                        raw_input = payload.get("input")
                        if tool_name == _EXEC_WRAPPER_TOOL_NAME and isinstance(raw_input, str):
                            # One wrapper program fans out to many calls, so no
                            # single one owns the wrapper's call_id.
                            orphan_tool_calls.extend(_exec_wrapper_tool_calls(raw_input))
                        else:
                            tool_input: dict[str, Any] | str | None = (
                                raw_input if isinstance(raw_input, (dict, str)) else None
                            )
                            orphan_tool_calls.append(
                                build_tool_call(
                                    tool_name, tool_input, tool_use_id=_call_id(payload)
                                )
                            )
                    elif payload_type in _TOOL_OUTPUT_PAYLOAD_TYPES:
                        call_id = _call_id(payload)
                        if call_id:
                            tool_results.append(
                                _codex_tool_result(call_id, payload.get("output"), timestamp)
                            )
                    elif payload_type == "tool_search_call":
                        orphan_tool_calls.append(
                            build_tool_call(
                                "tool_search",
                                _parse_function_call_arguments(payload.get("arguments")),
                                tool_use_id=_call_id(payload),
                            )
                        )
                    elif payload_type == "tool_search_output":
                        call_id = _call_id(payload)
                        if call_id:
                            tool_results.append(
                                _codex_tool_result(
                                    call_id, {"tools": payload.get("tools", [])}, timestamp
                                )
                            )
                    elif payload_type == "web_search_call":
                        action = payload.get("action", {})
                        # A search action carries `query`, not `url`; reading
                        # only `url` recorded nothing for every actual search
                        # (REQ-PARSE-019).  Keep the whole action.
                        orphan_tool_calls.append(
                            build_tool_call(
                                "web_search",
                                dict(action) if isinstance(action, dict) and action else None,
                                tool_use_id=_call_id(payload) or payload.get("id"),
                            )
                        )
                    elif payload_type not in _TOOL_OUTPUT_PAYLOAD_TYPES:
                        is_complete = False
                        diagnostics.append(
                            ParseDiagnostic(
                                "unsupported_record",
                                capture.record_start,
                                f"response_item: {payload_type!r}",
                            )
                        )
                elif entry_type == "message":
                    payload = entry.get("payload", {})
                    role_value = payload.get("role", "user")
                    try:
                        role = Role(role_value)
                    except ValueError:
                        is_complete = False
                        diagnostics.append(
                            ParseDiagnostic(
                                "unsupported_record",
                                capture.record_start,
                                f"unknown legacy role: {role_value!r}",
                            )
                        )
                        continue
                    content = payload.get("content")
                    blocks = extract_content_blocks(content)
                    text_parts = blocks.text
                    thinking_parts = blocks.thinking
                    msg_tool_calls = blocks.tool_calls
                    for tool_result in blocks.tool_results:
                        tool_result.completed_at = timestamp
                    tool_results.extend(blocks.tool_results)
                    message = Message(
                        id=make_message_id(session_id_value, message_idx_base + len(messages)),
                        session_id=session_id_value,
                        idx=message_idx_base + len(messages),
                        role=role,
                        content="\n".join(text_parts) if text_parts else None,
                        thinking="\n".join(thinking_parts) if thinking_parts else None,
                        timestamp=timestamp,
                        has_thinking=bool(thinking_parts),
                        tool_calls=msg_tool_calls,
                    )
                    messages.append(message)
                elif entry_type not in {
                    "rollout_item",
                    "world_state",
                    "inter_agent_communication_metadata",
                }:
                    is_complete = False
                    diagnostics.append(
                        ParseDiagnostic(
                            "unsupported_record", capture.record_start, f"record: {entry_type!r}"
                        )
                    )

                if (
                    turn_in_progress
                    and len(messages) > message_start
                    and messages[-1].role is Role.ASSISTANT
                    and not messages[-1].agent_id
                ):
                    # A paired tool result is not a completed task. Carry the
                    # observed task start through each main-thread assistant
                    # record until a native completion/abort closes it.
                    stop_markers.append(
                        StopMarker(idx=messages[-1].idx, reason="task_started", ends_turn=False)
                    )
                orphan_positions.extend([position] * (len(orphan_tool_calls) - orphan_start))
                tool_result_positions.extend([position] * (len(tool_results) - tool_result_start))

        assert len(orphan_positions) == len(orphan_tool_calls)
        orphan_tool_calls = [
            call
            for _, call in sorted(
                zip(orphan_positions, orphan_tool_calls, strict=True), key=lambda pair: pair[0]
            )
        ]
        # Native completion items are held back to the end of the walk, so the
        # records they produce arrive out of file order. Orphan calls were
        # already restored to that order; tool results are restored here for
        # the same reason, and because a suffix's results must concatenate
        # onto a prefix's exactly as a full parse would have ordered them.
        assert len(tool_result_positions) == len(tool_results)
        tool_results = [
            result
            for _, result in sorted(
                zip(tool_result_positions, tool_results, strict=True), key=lambda pair: pair[0]
            )
        ]

        for message in messages:
            for tool_idx, tool_call in enumerate(message.tool_calls):
                tool_call.idx = tool_idx
                tool_call.id = make_tool_call_id(message.id, tool_idx)
                tool_call.session_id = session_id_value
                tool_call.message_id = message.id
                tool_calls.append(tool_call)

        for orphan_idx, tool_call in enumerate(orphan_tool_calls, start=orphan_tool_call_idx_base):
            tool_call.idx = orphan_idx
            tool_call.id = make_tool_call_id(None, orphan_idx, session_id_value=session_id_value)
            tool_call.session_id = session_id_value
            tool_call.message_id = None
            tool_calls.append(tool_call)

        message_count = len(messages)
        tool_count = len(tool_calls)
        duration_seconds = None
        if started_at and ended_at:
            duration_seconds = int((ended_at - started_at).total_seconds())

        # Real codex session_meta git payloads carry branch/commit_hash/
        # repository_url but no local root key, so git_info.get("root") never
        # fires; fall back to cwd (mirroring the Claude parser) instead of
        # indexing every codex session with git_repo=NULL.
        if git_repo is None and cwd is not None:
            git_repo = cwd

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
            orphan_tool_calls=orphan_tool_calls,
        )
        return ParseResult(
            session=session,
            next_byte_offset=next_byte_offset,
            is_full_parse=is_full_parse,
            normalization_checkpoint=resume_checkpoint(
                capture,
                parser_revision=parser_revision(type(self)),
                # Every boundary is offered; `resume_checkpoint` declines the
                # ones whose open correlation no longer fits the envelope.
                resumable=True,
                diagnostics=diagnostics,
                message_idx_base=message_idx_base + message_count,
                orphan_tool_call_idx_base=orphan_tool_call_idx_base + len(orphan_tool_calls),
                adapter_state=_codex_adapter_state(turn_in_progress, correlation),
            ),
            tail_facts=TailFacts(
                tool_results=tuple(tool_results),
                # Several lifecycle events can follow the same message. The
                # last observed event, not the first, governs that position.
                stop_markers=tuple({marker.idx: marker for marker in stop_markers}.values()),
                # Codex ends a turn, not a session: `task_complete` says the
                # agent is idle, not that the transcript is closed.
                session_ended=False,
            ),
            diagnostics=tuple(diagnostics),
            captured_prefix_sha256=capture.captured_prefix_sha256,
            initial_prefix_sha256=capture.initial_prefix_sha256,
            source_dev=capture.source_dev,
            source_inode=capture.source_inode,
            captured_size=capture.captured_size,
            committed_prefix_sha256=capture.committed_prefix_sha256,
        )


def _reasoning_text(payload: dict[str, Any]) -> str:
    """Keep only plaintext supplied by the harness; encrypted traces stay opaque."""
    parts: list[str] = []
    for name in ("summary", "summary_text", "content", "raw_content"):
        value = payload.get(name)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(
                item if isinstance(item, str) else item.get("text", "")
                for item in value
                if isinstance(item, (str, dict))
            )
    return "\n".join(part for part in parts if isinstance(part, str) and part)


def _normalized_message(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if payload.get("type") == "reasoning":
        normalized.update(
            type="message",
            role="assistant",
            content=[{"type": "thinking", "thinking": _reasoning_text(payload)}],
        )
    elif payload.get("type") == "agent_message":
        normalized.update(type="message", role="assistant")
    if normalized.get("role") == "developer":
        normalized["role"] = "system"
    content = normalized.get("content")
    if isinstance(content, dict) and content.get("type") == "Text":
        normalized["content"] = {**content, "type": "text"}
    elif isinstance(content, list):
        normalized["content"] = [
            {**block, "type": "text"}
            if isinstance(block, dict) and block.get("type") == "Text"
            else block
            for block in content
        ]
    return normalized


def _suffix_can_defer(item: dict[str, Any], correlation: _CodexCorrelation) -> bool:
    """Whether a suffix can match this completion item without the full file.

    An execution view carries an identity in its own namespace, so no SDK
    record before the boundary can have claimed it and the only state the
    match needs -- the turn that ran the wrapper -- rides in the checkpoint.
    Every other completion item is matched against the identities of SDK tool
    records anywhere in the file, which a suffix does not have.
    """
    if correlation.exec_namespace_claimed:
        return False
    identity = item.get("id")
    return isinstance(identity, str) and identity.startswith(_EXEC_IDENTITY_PREFIX)


def _codex_adapter_state(turn_in_progress: bool, correlation: _CodexCorrelation) -> dict[str, Any]:
    """The boundary's carried state; empty when the adapter owes nothing."""
    state: dict[str, Any] = {}
    if turn_in_progress:
        state[_TURN_IN_PROGRESS] = True
    open_correlation = correlation.to_state()
    if open_correlation:
        state[_CORRELATION] = open_correlation
    return state


def _message_key(payload: dict[str, Any]) -> str:
    """Identify a message by its normalized content, as a fixed-size digest.

    Digested rather than kept verbatim because an unresolved key travels in the
    normalization checkpoint: holding message bodies there would make the
    envelope grow with the transcript instead of with its open correlations.
    """
    blocks = extract_content_blocks(payload.get("content"))
    identity = json.dumps(
        [
            payload.get("role"),
            blocks.text,
            blocks.thinking,
            payload.get("author"),
            [tool.model_dump(mode="json") for tool in blocks.tool_calls],
        ],
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _conversation_records(
    capture: JsonlCapture, *, correlation: _CodexCorrelation, resumed: bool
) -> Iterator[dict[str, Any]]:
    """Normalize rollout representations with one-to-one correlated mirror matching.

    Equal messages in the same representation remain distinct. Cross-format
    matching requires an item id or an observed turn id. Compaction history is
    an explicit snapshot, so occurrences already in history are not replayed.

    `correlation` carries the open pairing across a resume boundary and is
    advanced in place, so the adapter can hand the same state to the next
    suffix. `resumed` says this walk started past the head of the file, and so
    holds neither the replaced-message history a compaction snapshot is matched
    against nor the tool identities a deferred completion item is matched
    against; both are diagnosed rather than guessed at.
    """
    turn_id = correlation.turn_id
    turn_aliases = correlation.aliases
    user_anchor = correlation.user_anchor
    pending_alias = correlation.pending_alias
    ids = correlation.correlated_ids
    balances = correlation.balances
    exec_turns = correlation.exec_turns
    history: Counter[str] = Counter()
    call_ids: set[str] = set()
    output_ids: set[str] = set()
    pending_tools: list[tuple[dict[str, Any], dict[str, Any], str | None]] = []
    for original in capture.records():
        entry = {**original, "_record_position": capture.record_start}
        record_type = original.get("type")
        if record_type is None:
            # Typeless session header: older rollouts omit "type".
            entry = {**entry, "type": "session_meta", "payload": dict(original)}
        elif record_type in {"function_call", "function_call_output", "reasoning"}:
            entry = {**entry, "type": "response_item", "payload": dict(original)}
        payload = entry.get("payload", {})
        if not isinstance(payload, dict):
            capture.diagnostics.append(
                ParseDiagnostic("unsupported_record", capture.record_start, "non-object payload")
            )
            continue
        raw_turn = payload.get("turn_id")
        if isinstance(raw_turn, str):
            turn_id = raw_turn
        metadata = payload.get("internal_chat_message_metadata_passthrough")
        message_turn = metadata.get("turn_id") if isinstance(metadata, dict) else None
        message_turn = message_turn if isinstance(message_turn, str) else turn_id
        message_turn = turn_aliases.get(message_turn, message_turn) if message_turn else None
        if entry.get("type") == "event_msg" and payload.get("type") == "task_started":
            pending_alias = None
            if user_anchor and isinstance(raw_turn, str):
                anchor_turn, anchor_key, anchor_timestamp = user_anchor
                if entry.get("timestamp") == anchor_timestamp:
                    pending_alias = (raw_turn, anchor_turn, anchor_key, anchor_timestamp)
            user_anchor = None
        representation = "response"
        if entry.get("type") == "compacted":
            if resumed:
                # A compaction snapshot replays the conversation it replaced,
                # minus whatever this walk already emitted. A suffix has
                # emitted none of it, so replaying here would duplicate the
                # whole prefix. Diagnose instead: the capture stops
                # acknowledging at this record and the caller re-reads from
                # zero.
                capture.diagnostics.append(
                    ParseDiagnostic(
                        "unsupported_record",
                        capture.record_start,
                        "compaction snapshot needs the full parse history",
                    )
                )
                continue
            replacement = payload.get("replacement_history", [])
            if not isinstance(replacement, list):
                capture.diagnostics.append(
                    ParseDiagnostic(
                        "unsupported_record", capture.record_start, "invalid replacement history"
                    )
                )
                continue
            remaining = history.copy()
            summary = payload.get("message")
            if isinstance(summary, str) and summary:
                replacement = [
                    *replacement,
                    {
                        "type": "message",
                        "role": "system",
                        "content": summary,
                    },
                ]
            for item in replacement:
                if item == {"type": "other"}:
                    # Observed compaction placeholders contain no recoverable
                    # payload. Additional unknown fields still fail closed.
                    continue
                if isinstance(item, dict) and item.get("type") == "compaction":
                    # The compacted snapshot can carry an opaque model state
                    # envelope beside its readable replacement messages.
                    summary = _compaction_summary(item)
                    if summary is None:
                        continue
                    item = {"type": "message", "role": "system", "content": summary}
                if not isinstance(item, dict) or item.get("type") not in {
                    "message",
                    "agent_message",
                    "reasoning",
                }:
                    capture.diagnostics.append(
                        ParseDiagnostic(
                            "unsupported_record",
                            capture.record_start,
                            "non-message replacement history item",
                        )
                    )
                    continue
                normalized = _normalized_message(item)
                key = _message_key(normalized)
                if remaining[key]:
                    remaining[key] -= 1
                    continue
                history[key] += 1
                yield {**entry, "type": "response_item", "payload": normalized}
            continue
        if entry.get("type") == "event_msg":
            event_type = payload.get("type")
            if event_type in {
                "entered_review_mode",
                "exited_review_mode",
                "web_search_end",
                "mcp_tool_call_end",
            }:
                try:
                    payload = {**payload, "item": _legacy_completed_item(payload)}
                except ValueError as error:
                    capture.diagnostics.append(
                        ParseDiagnostic("unsupported_record", capture.record_start, str(error))
                    )
                    continue
                event_type = "item_completed"
            if event_type == "item_completed":
                item = payload.get("item", {})
                kind = item.get("type") if isinstance(item, dict) else None
                if kind in {"UserMessage", "AgentMessage"}:
                    content = item.get("content")
                    blocks = content if isinstance(content, list) else [content]
                    unsupported = [
                        block.get("type")
                        for block in blocks
                        if isinstance(block, dict)
                        and isinstance(block.get("text"), str)
                        and block["text"]
                        and block.get("type")
                        not in {"Text", "text", "input_text", "output_text", "thinking"}
                    ]
                    if unsupported:
                        capture.diagnostics.append(
                            ParseDiagnostic(
                                "unsupported_record",
                                capture.record_start,
                                f"completed {kind} text blocks: {unsupported!r}",
                            )
                        )
                        continue
                if kind in {"WebSearch", "McpToolCall", "CommandExecution", "ImageView"}:
                    if resumed and not _suffix_can_defer(item, correlation):
                        # Only an execution view names an identity no SDK
                        # record can have claimed, so only it can be matched
                        # without the whole file's tool identities. Anything
                        # else is diagnosed here, inside this record's capture
                        # window, so the capture stops acknowledging at this
                        # record and the caller re-reads the source from zero.
                        capture.diagnostics.append(
                            ParseDiagnostic(
                                "unsupported_record",
                                capture.record_start,
                                "completion item needs the full parse tool correlation",
                            )
                        )
                        continue
                    try:
                        _completed_tool_records(item)
                    except ValueError as error:
                        capture.diagnostics.append(
                            ParseDiagnostic("unsupported_record", capture.record_start, str(error))
                        )
                        continue
                    pending_tools.append((entry, item, message_turn))
                    continue
                if kind == "HookPrompt":
                    item = {
                        **item,
                        "content": "\n".join(
                            fragment["text"]
                            for fragment in item.get("fragments", [])
                            if isinstance(fragment, dict) and isinstance(fragment.get("text"), str)
                        ),
                    }
                elif kind == "Plan":
                    item = {**item, "content": item.get("text", "")}
                elif kind == "EnteredReviewMode":
                    item = {
                        **item,
                        "content": json.dumps(
                            {
                                "target": item.get("target"),
                                "user_facing_hint": item.get("user_facing_hint"),
                            },
                            sort_keys=True,
                        ),
                    }
                elif kind == "ExitedReviewMode":
                    if item.get("review_output") is None:
                        continue
                    item = {
                        **item,
                        "content": json.dumps(item.get("review_output"), sort_keys=True),
                    }
                if kind in {
                    "ContextCompaction",
                    "SubAgentActivity",
                    "FileChange",
                    "Extension",
                    "CollabAgentToolCall",
                }:
                    # Rendered activity items summarize execution/UI state;
                    # response_item calls remain the tool invocation records.
                    continue
                if kind not in {
                    "UserMessage",
                    "AgentMessage",
                    "Reasoning",
                    "HookPrompt",
                    "Plan",
                    "EnteredReviewMode",
                    "ExitedReviewMode",
                }:
                    capture.diagnostics.append(
                        ParseDiagnostic(
                            "unsupported_record",
                            capture.record_start,
                            f"completed item: {kind!r}",
                        )
                    )
                    continue
                payload = {
                    **item,
                    "type": "reasoning" if kind == "Reasoning" else "message",
                    "role": (
                        "user"
                        if kind == "UserMessage"
                        else "system"
                        if kind in {"HookPrompt", "EnteredReviewMode"}
                        else "assistant"
                    ),
                }
            elif event_type in {"user_message", "agent_message", "agent_reasoning"}:
                payload = {
                    **payload,
                    "type": "message",
                    "role": "user" if event_type == "user_message" else "assistant",
                    "content": payload.get("message", payload.get("text", "")),
                }
                if event_type == "agent_reasoning":
                    payload["content"] = [{"type": "thinking", "thinking": payload["content"]}]
            else:
                yield entry
                continue
            entry["type"] = "response_item"
            representation = "mirror"
        if entry.get("type") == "response_item":
            kind = payload.get("type")
            identity = _call_id(payload) or (
                payload.get("id") if kind == "web_search_call" else None
            )
            if identity and kind in {
                "function_call",
                "custom_tool_call",
                "web_search_call",
                "tool_search_call",
            }:
                call_ids.add(str(identity))
            if identity and kind in _TOOL_OUTPUT_PAYLOAD_TYPES | {"tool_search_output"}:
                output_ids.add(str(identity))
            if isinstance(identity, str) and identity.startswith(_EXEC_IDENTITY_PREFIX):
                correlation.exec_namespace_claimed = True
            if kind == "custom_tool_call" and payload.get("name") == "exec" and message_turn:
                exec_turns.add(message_turn)
        if entry.get("type") in {"message", "response_item"} and payload.get("type") in {
            None,
            "message",
            "agent_message",
            "reasoning",
        }:
            if payload.get("type") == "reasoning" and not _reasoning_text(payload):
                continue
            payload = _normalized_message(payload)
            key = _message_key(payload)
            if pending_alias:
                alias, canonical_turn, user_key, alias_timestamp = pending_alias
                # Rewritten rollout prefixes can name the same turn differently
                # in SDK metadata and native activity. Only the matching user
                # message at the shared task-start timestamp establishes an alias.
                if (
                    representation == "mirror"
                    and payload.get("role") == "user"
                    and message_turn == alias
                    and key == user_key
                    and entry.get("timestamp") == alias_timestamp
                ):
                    turn_aliases[alias] = canonical_turn
                    message_turn = canonical_turn
                pending_alias = None
            timestamp = entry.get("timestamp")
            user_anchor = (
                (message_turn, key, timestamp)
                if representation == "response"
                and payload.get("role") == "user"
                and message_turn
                and isinstance(timestamp, str)
                else None
            )
            raw_id = payload.get("id") or payload.get("item_id") or payload.get("message_id")
            correlated_id = str(raw_id) if raw_id is not None else None
            duplicate_id = correlated_id in ids if correlated_id else False
            if correlated_id:
                ids.add(correlated_id)
            duplicate_mirror = False
            if message_turn:
                balance_key = (message_turn, key)
                balance = balances[balance_key]
                if representation == "mirror":
                    duplicate_mirror = balance > 0
                    balance -= 1
                else:
                    duplicate_mirror = balance < 0
                    balance += 1
                # A balanced pair is indistinguishable from an unseen one, so
                # dropping it keeps the carried state proportional to the
                # conversation's open correlations rather than its length.
                if balance:
                    balances[balance_key] = balance
                else:
                    del balances[balance_key]
            if duplicate_id or duplicate_mirror:
                continue
            history[key] += 1
            entry["payload"] = payload
        yield entry

    # `turn_id`, `user_anchor` and `pending_alias` are rebound rather than
    # mutated, so they are published here; the containers above are shared.
    correlation.turn_id = turn_id
    correlation.user_anchor = user_anchor
    correlation.pending_alias = pending_alias
    for entry, item, item_turn in pending_tools:
        identity = str(item.get("id") or "")
        # exec-* items are execution views of an observed wrapper program's
        # decoded calls, not additional top-level invocations (REQ-PARSE-018).
        if identity.startswith("exec-") and item_turn in exec_turns:
            continue
        for payload in _completed_tool_records(item):
            is_output = payload["type"] in _TOOL_OUTPUT_PAYLOAD_TYPES
            seen = output_ids if is_output else call_ids
            if identity and identity in seen:
                continue
            if identity:
                seen.add(identity)
            yield {**entry, "type": "response_item", "payload": payload}


def _legacy_completed_item(payload: dict[str, Any]) -> dict[str, Any]:
    """Older native completion events use the same invocation identity and content."""
    match payload["type"]:
        case "entered_review_mode":
            return {**payload, "type": "EnteredReviewMode"}
        case "exited_review_mode":
            return {**payload, "type": "ExitedReviewMode"}
        case "web_search_end":
            return {**payload, "type": "WebSearch", "id": payload.get("call_id")}
        case "mcp_tool_call_end":
            invocation = payload.get("invocation")
            result = payload.get("result")
            if not isinstance(invocation, dict):
                raise ValueError("mcp_tool_call_end has no invocation")
            if not isinstance(result, dict) or set(result) not in ({"Ok"}, {"Err"}):
                raise ValueError("mcp_tool_call_end has no recognized result")
            if "Ok" in result and not isinstance(result["Ok"], dict):
                raise ValueError("mcp_tool_call_end has a non-object success result")
            return {
                **invocation,
                "type": "McpToolCall",
                "id": payload.get("call_id"),
                "result": result.get("Ok"),
                "error": result.get("Err"),
                "status": "failed" if "Err" in result else "completed",
            }
        case _:
            raise AssertionError(f"not a legacy completion event: {payload['type']!r}")


def _completed_tool_records(item: dict[str, Any]) -> list[dict[str, Any]]:
    identity = item.get("id")
    kind = item["type"]
    if not isinstance(identity, str) or not identity:
        raise ValueError(f"completed {kind} has no invocation id")
    if kind == "WebSearch":
        action = item.get("action")
        if not isinstance(action, dict):
            if not isinstance(item.get("query"), str):
                raise ValueError("completed WebSearch has no action or query")
            action = {"query": item.get("query")}
        records = [
            {"type": "web_search_call", "id": identity, "call_id": identity, "action": action}
        ]
        if "results" in item:
            if not isinstance(item["results"], list):
                raise ValueError("completed WebSearch has non-list results")
            records.append(
                {
                    "type": "function_call_output",
                    "call_id": identity,
                    "output": {"content": json.dumps(item["results"], sort_keys=True)},
                }
            )
        return records
    if kind == "ImageView":
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("completed ImageView has no path")
        return [
            {
                "type": "function_call",
                "call_id": identity,
                "name": "view_image",
                "arguments": {"path": path},
            }
        ]
    if kind == "McpToolCall":
        if not all(isinstance(item.get(key), str) and item[key] for key in ("server", "tool")):
            raise ValueError("completed McpToolCall has no server/tool name")
        name = f"mcp__{item.get('server', '')}__{item.get('tool', '')}"
        arguments = item.get("arguments")
        result = item.get("result")
        error = item.get("error")
        output = {
            "content": json.dumps(
                result if result is not None else {"error": error}, sort_keys=True
            ),
            "success": not (
                error is not None
                or item.get("status") == "failed"
                or (isinstance(result, dict) and result.get("isError"))
            ),
        }
    else:
        assert kind == "CommandExecution"
        name = "exec_command"
        command = item.get("command")
        if isinstance(command, list) and command and all(isinstance(part, str) for part in command):
            command = shlex.join(command)
        if not isinstance(command, str) or not command:
            raise ValueError("completed CommandExecution has no command")
        arguments = {"cmd": command, "cwd": item.get("cwd"), "source": item.get("source")}
        output = {
            "content": item.get("aggregated_output"),
            "success": item.get("exit_code") in (None, 0),
        }
    return [
        {"type": "function_call", "call_id": identity, "name": name, "arguments": arguments},
        {"type": "function_call_output", "call_id": identity, "output": output},
    ]


def _build_plain_message(
    role: Role, text: str, session_id: str, idx: int, timestamp: datetime | None
) -> Message:
    message_id_value = make_message_id(session_id, idx)
    return Message(
        id=message_id_value,
        session_id=session_id,
        idx=idx,
        role=role,
        content=text or None,
        thinking=None,
        timestamp=timestamp,
        has_thinking=False,
        tool_calls=[],
    )


def _compaction_summary(payload: dict[str, Any]) -> str | None:
    """Return a readable Codex compaction summary without guessing structure."""
    for key in ("summary", "content", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
        blocks = extract_content_blocks(value)
        if blocks.text:
            return "\n".join(blocks.text)
    return None


def _parse_function_call_arguments(raw: Any) -> dict[str, Any] | None:
    """Parse a JSON-encoded arguments string from a Codex function_call."""
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


# Codex >=2026-07-09 stops emitting one function_call per tool and instead
# ships a small JS program under a single custom_tool_call named "exec".
# The real calls live inside it as ``tools.<name>({...})``, and one program
# may batch several of them via ``Promise.all([...])``.
_EXEC_WRAPPER_TOOL_NAME = "exec"


def _exec_wrapper_tool_calls(source: str) -> list[ToolCall]:
    """Build the tool calls an ``exec`` wrapper stands for.

    A wrapper that invokes nothing recognizable still yields one row, because
    dropping the source is what left 183k codex rows with no searchable text.
    Non-literal arguments keep the whole wrapper source rather
    than a slice: finding a call's own extent needs a JS parser, and the
    surrounding program is where its inputs (patch bodies, heredocs) live.
    """
    inner = parse_exec_wrapper(source)
    if not inner:
        return [build_tool_call(_EXEC_WRAPPER_TOOL_NAME, {"source": source})]
    return [
        build_tool_call(name, arguments if arguments is not None else {"source": source})
        for name, arguments in inner
    ]


# Codex names a tool result by the `call_id` of the call it answers, across
# both the typed-function and custom-tool shapes.
_TOOL_OUTPUT_PAYLOAD_TYPES = frozenset(
    {"function_call_output", "custom_tool_call_output", "local_shell_call_output"}
)


def _call_id(payload: dict[str, Any]) -> str | None:
    value = payload.get("call_id")
    return str(value) if value else None


def _codex_tool_result(call_id: str, output: Any, timestamp: datetime | None) -> ToolResult:
    """Build a tool result from a codex `*_output` payload.

    The typed-function shape wraps the text in a dict that also carries a
    `success` flag; the custom-tool shape is a bare string with no error
    signal, so a missing flag means "not known to have failed".
    """
    if isinstance(output, dict):
        text = output.get("content") if isinstance(output.get("content"), str) else None
        is_error = output.get("success") is False
        summary = summarize_tool_result(text if text is not None else output)
    else:
        is_error = False
        summary = summarize_tool_result(output)
    return ToolResult(
        tool_use_id=call_id,
        result_summary=summary,
        is_error=is_error,
        completed_at=timestamp,
    )
