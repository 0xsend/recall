from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

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
    build_tool_call,
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
from recall.parsers.skills import derive_skill_name


@dataclass
class GrokParser:
    """Parser for Grok (xAI Grok Build / Grok CLI) session JSONL logs.

    Sessions live at ~/.grok/sessions/<encoded-workspace>/<session-uuid>/chat_history.jsonl
    Each line is a top-level event: system, user, assistant, tool_result,
    reasoning, or backend_tool_call.

    - User content is list[{"type":"text", "text": "..."}]
    - Assistant has optional "content" (str), nested "reasoning":{"text": "..."}
      for thinking, and "tool_calls": [{"id", "name", "arguments"}]
    - Top-level reasoning joins summary_text parts as assistant thinking.
      encrypted_content is opaque and is never used as text.
    - backend_tool_call becomes a ToolCall named from kind.tool_type; the
      action (query, urls) is kept as tool_input.
    - Tool results are separate events with "tool_call_id" and "content".
    """

    source: Source = Source.GROK
    # Discovery roots from `[sources.<name>] roots`; None = default_roots(),
    # an empty tuple = configured to scan nothing.
    roots: tuple[Path, ...] | None = None

    @property
    def file_pattern(self) -> str:
        return "chat_history.jsonl"

    def default_roots(self) -> list[Path]:
        return [resolve_home() / ".grok" / "sessions"]

    def watch_roots(self) -> list[Path]:
        return default_watch_roots(self)

    def discover(self) -> list[Path]:
        return default_discover(self)

    def sidecar_paths(self, path: Path) -> list[Path]:
        """summary.json / signals.json, the files _read_grok_sidecars consumes.

        Kept in lockstep with REQ-PARSE-014: metadata sourced from a sidecar
        must also be fingerprinted from it, or the row never refreshes.
        """
        session_dir = path.parent
        return [session_dir / "summary.json", session_dir / "signals.json"]

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
        # Reasoning and backend-tool records attach to a *neighbouring*
        # assistant message rather than standing alone, so the adapter carries
        # nothing forward: it declines instead (REQ-INDEX-026).
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
        is_complete = True
        model: str | None = None
        cwd: str | None = None
        source_session_id: str | None = None
        pending_thinking: list[str] = []
        pending_backend_calls: list[ToolCall] = []

        def emit_assistant(
            content: str | None,
            thinking: str | None,
            msg_tool_calls: list[ToolCall],
        ) -> None:
            msg = Message(
                id=make_message_id(session_id_value, message_idx_base + len(messages)),
                session_id=session_id_value,
                idx=message_idx_base + len(messages),
                role=Role.ASSISTANT,
                content=content,
                thinking=thinking,
                timestamp=None,
                has_thinking=bool(thinking),
                tool_calls=msg_tool_calls,
            )
            messages.append(msg)
            for tool_idx, tool_call in enumerate(msg_tool_calls):
                tool_call.idx = tool_idx
                tool_call.id = make_tool_call_id(msg.id, tool_idx)
                tool_call.session_id = session_id_value
                tool_call.message_id = msg.id
                tool_calls.append(tool_call)

        def flush_pending() -> None:
            if not pending_thinking and not pending_backend_calls:
                return
            thinking = "\n".join(pending_thinking) if pending_thinking else None
            calls = list(pending_backend_calls)
            pending_thinking.clear()
            pending_backend_calls.clear()
            emit_assistant(None, thinking, calls)

        with JsonlCapture(path, offset=offset) as capture:
            diagnostics = capture.diagnostics
            for entry in capture.records():
                entry_type = entry.get("type")

                if entry_type == "system":
                    flush_pending()
                    content = self._extract_text_content(entry.get("content"))
                    if content:
                        msg = self._make_plain_message(
                            role=Role.SYSTEM,
                            content=content,
                            session_id=session_id_value,
                            idx=message_idx_base + len(messages),
                        )
                        messages.append(msg)

                elif entry_type == "user":
                    flush_pending()
                    content = self._extract_text_content(entry.get("content"))
                    if content:
                        msg = self._make_plain_message(
                            role=Role.USER,
                            content=content,
                            session_id=session_id_value,
                            idx=message_idx_base + len(messages),
                        )
                        messages.append(msg)

                elif entry_type == "assistant":
                    text_content = self._extract_text_content(entry.get("content"))
                    reasoning = entry.get("reasoning") or {}
                    nested = reasoning.get("text") if isinstance(reasoning, dict) else None
                    if isinstance(nested, str) and nested:
                        pending_thinking.append(nested)
                    thinking = "\n".join(pending_thinking) if pending_thinking else None
                    pending_thinking.clear()

                    if not model:
                        mid = entry.get("model_id")
                        if isinstance(mid, str) and mid:
                            model = mid

                    raw_tcs = entry.get("tool_calls") or []
                    msg_tool_calls = list(pending_backend_calls)
                    pending_backend_calls.clear()
                    for tc in raw_tcs:
                        if not isinstance(tc, dict):
                            continue
                        name = str(tc.get("name", ""))
                        args_raw = tc.get("arguments")
                        tool_input = self._parse_json_args(args_raw)
                        call_id = tc.get("id")
                        msg_tool_calls.append(
                            build_tool_call(
                                name,
                                tool_input,
                                tool_use_id=str(call_id) if call_id else None,
                            )
                        )

                    emit_assistant(text_content, thinking, msg_tool_calls)

                elif entry_type == "tool_result":
                    flush_pending()
                    # Map tool output to a SYSTEM message (matches Pi Agent behavior for toolResult)
                    content = str(entry.get("content", "") or "")
                    tcid = entry.get("tool_call_id")
                    if isinstance(tcid, str) and tcid:
                        tool_results.append(
                            ToolResult(
                                tool_use_id=tcid,
                                result_summary=summarize_tool_result(content),
                                is_error=bool(entry.get("is_error", False)),
                            )
                        )
                    if content or tcid:
                        text = content
                        if tcid and isinstance(tcid, str):
                            prefix = f"[tool_call_id: {tcid}]"
                            text = f"{prefix}\n{content}" if content else prefix
                        msg = self._make_plain_message(
                            role=Role.SYSTEM,
                            content=text,
                            session_id=session_id_value,
                            idx=message_idx_base + len(messages),
                        )
                        messages.append(msg)

                elif entry_type == "reasoning":
                    summary = _reasoning_summary_text(entry)
                    if summary:
                        pending_thinking.append(summary)

                elif entry_type == "backend_tool_call":
                    call = self._backend_tool_call(entry)
                    if call is None:
                        continue
                    last = messages[-1] if messages else None
                    if last is not None and last.role == Role.ASSISTANT:
                        tool_idx = len(last.tool_calls)
                        call.idx = tool_idx
                        call.id = make_tool_call_id(last.id, tool_idx)
                        call.session_id = session_id_value
                        call.message_id = last.id
                        last.tool_calls.append(call)
                        tool_calls.append(call)
                    else:
                        pending_backend_calls.append(call)

                else:
                    diagnostics.append(
                        ParseDiagnostic(
                            "unsupported_record", capture.record_start, f"record: {entry_type!r}"
                        )
                    )

            had_pending = bool(pending_thinking or pending_backend_calls)
            flush_pending()

        # A trailing assistant message is an open boundary: a following
        # `backend_tool_call` amends its tool calls in place, and a suffix
        # parsed on its own has no earlier message to amend (REQ-INDEX-026).
        # Pending reasoning or backend-tool state is the same boundary seen
        # one record earlier, because flushing it emits that assistant.
        trailing_assistant = bool(messages) and messages[-1].role is Role.ASSISTANT
        assert trailing_assistant or not had_pending

        # Derive cwd from the encoded workspace directory in the path:
        # ~/.grok/sessions/<encoded-cwd>/<uuid>/chat_history.jsonl
        try:
            uuid_dir = path.parent
            source_session_id = uuid_dir.name
            encoded_workspace = uuid_dir.parent.name
            cwd = unquote(encoded_workspace) if encoded_workspace else None
        except Exception:
            source_session_id = None
            cwd = None

        # REQ-PARSE-014 / REQ-GROK-TS-*: enrich from sibling sidecars when present.
        # Tokens stay None here — chat_history has no usage; harvest fills them
        # (REQ-PARSE-012 / REQ-USAGE-015). Never map contextTokensUsed → tokens.
        started_at: datetime | None = None
        ended_at: datetime | None = None
        duration_seconds: int | None = None
        git_repo: str | None = None
        git_branch: str | None = None
        sidecar = _read_grok_sidecars(path.parent)
        if sidecar.cwd:
            cwd = sidecar.cwd
        if sidecar.model:
            model = sidecar.model
        started_at = sidecar.started_at
        ended_at = sidecar.ended_at
        duration_seconds = sidecar.duration_seconds
        git_repo = sidecar.git_repo
        git_branch = sidecar.git_branch

        # Grok may load agent-profile skills from the source tree configured as
        # a runtime skill directory. Attribution needs the resolved session cwd
        # so a working-copy read remains authoring (REQ-PARSE-017).
        for tool_call in tool_calls:
            if tool_call.skill_name is None:
                tool_call.skill_name = derive_skill_name(
                    tool_call.tool_name,
                    tool_call.tool_input,
                    tool_call.bash_command,
                    cwd=cwd,
                )

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
            message_count=len(messages),
            tool_count=len(tool_calls),
            # Grok chat_history.jsonl records no token usage fields; keep this
            # unknown rather than fabricating zeros or derived estimates.
            # Exact tokens come from the unified.jsonl harvester (REQ-USAGE-*).
            input_tokens=None,
            output_tokens=None,
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
                resumable=not trailing_assistant,
                diagnostics=diagnostics,
                message_idx_base=message_idx_base + len(messages),
                orphan_tool_call_idx_base=orphan_tool_call_idx_base,
            ),
            # Grok has no named stop field. An assistant with tool_calls is
            # mid-turn; a text-only assistant is the end of the turn.
            tail_facts=TailFacts(
                tool_results=tuple(tool_results),
                stop_markers=tuple(_grok_stop_markers(messages)),
            ),
        )

    def _extract_text_content(self, content: Any) -> str | None:
        """Extract plain text from Grok user/assistant content (str or list of text parts)."""
        if content is None:
            return None
        if isinstance(content, str):
            return content if content.strip() else None
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text")
                    if isinstance(t, str) and t:
                        parts.append(t)
                elif isinstance(item, str):
                    parts.append(item)
            joined = "\n".join(parts).strip()
            return joined if joined else None
        return None

    def _parse_json_args(self, raw: Any) -> dict[str, Any] | None:
        """Parse arguments which may be dict or JSON-encoded string (Grok format)."""
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

    def _backend_tool_call(self, entry: dict[str, Any]) -> ToolCall | None:
        kind = entry.get("kind")
        if not isinstance(kind, dict):
            return None
        tool_type = kind.get("tool_type")
        if not isinstance(tool_type, str) or not tool_type:
            return None
        action = kind.get("action")
        if isinstance(action, dict) and action:
            tool_input: dict[str, Any] | str | None = dict(action)
        else:
            raw_input = kind.get("input")
            parsed = self._parse_json_args(raw_input)
            tool_input = parsed if parsed is not None else raw_input
        call_id = kind.get("id") or kind.get("call_id") or entry.get("id")
        return build_tool_call(
            tool_type,
            tool_input,
            tool_use_id=str(call_id) if call_id else None,
        )

    def _make_plain_message(
        self, role: Role, content: str | None, session_id: str, idx: int
    ) -> Message:
        return Message(
            id=make_message_id(session_id, idx),
            session_id=session_id,
            idx=idx,
            role=role,
            content=content,
            thinking=None,
            timestamp=None,
            has_thinking=False,
            tool_calls=[],
        )


def _grok_stop_markers(messages: list[Message]) -> list[StopMarker]:
    """Read Grok's turn vocabulary off the assistant event after tools attach."""
    markers: list[StopMarker] = []
    for message in messages:
        if message.role is not Role.ASSISTANT:
            continue
        if message.tool_calls:
            markers.append(StopMarker(idx=message.idx, reason="tool_calls", ends_turn=False))
        elif message.content:
            markers.append(StopMarker(idx=message.idx, reason="stop", ends_turn=True))
    return markers


def _reasoning_summary_text(entry: dict[str, Any]) -> str | None:
    """Join readable summary_text parts; encrypted_content stays opaque."""
    summary = entry.get("summary")
    parts: list[str] = []
    if isinstance(summary, str) and summary:
        parts.append(summary)
    elif isinstance(summary, list):
        for item in summary:
            if isinstance(item, str) and item:
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "summary_text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
    joined = "\n".join(parts).strip()
    return joined if joined else None


@dataclass(frozen=True)
class _GrokSidecarMeta:
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: int | None = None
    cwd: str | None = None
    git_repo: str | None = None
    git_branch: str | None = None
    model: str | None = None


def _read_grok_sidecars(session_dir: Path) -> _GrokSidecarMeta:
    """Load summary.json / signals.json; failures are non-fatal (REQ-GROK-TS-004)."""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: int | None = None
    cwd: str | None = None
    git_repo: str | None = None
    git_branch: str | None = None
    model: str | None = None

    summary_path = session_dir / "summary.json"
    if summary_path.is_file():
        try:
            raw = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            raw = None
        if isinstance(raw, dict):
            started_at = parse_timestamp(raw.get("created_at"))
            ended_at = parse_timestamp(raw.get("last_active_at"))
            info = raw.get("info")
            if isinstance(info, dict):
                info_cwd = info.get("cwd")
                if isinstance(info_cwd, str) and info_cwd:
                    cwd = info_cwd
            git_root = raw.get("git_root_dir")
            if isinstance(git_root, str) and git_root:
                git_repo = git_root
            branch = raw.get("head_branch")
            if isinstance(branch, str) and branch:
                git_branch = branch
            mid = raw.get("current_model_id")
            if isinstance(mid, str) and mid:
                model = mid

    signals_path = session_dir / "signals.json"
    if signals_path.is_file():
        try:
            raw_signals = json.loads(signals_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            raw_signals = None
        if isinstance(raw_signals, dict):
            duration = raw_signals.get("sessionDurationSeconds")
            if isinstance(duration, bool):
                duration = None
            if isinstance(duration, (int, float)) and duration >= 0:
                duration_seconds = int(duration)

    return _GrokSidecarMeta(
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration_seconds,
        cwd=cwd,
        git_repo=git_repo,
        git_branch=git_branch,
        model=model,
    )
