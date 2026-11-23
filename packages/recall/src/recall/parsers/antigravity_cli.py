from __future__ import annotations

import re
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from recall.core.ids import message_id as make_message_id
from recall.core.ids import session_id as make_session_id
from recall.core.ids import tool_call_id as make_tool_call_id
from recall.core.models import (
    Message,
    NormalizationCheckpoint,
    ParseDiagnostic,
    ParseResult,
    Session,
    StopMarker,
    TailFacts,
    ToolCall,
)
from recall.core.source_io import SidecarRefusedError, open_source
from recall.core.types import Role, Source
from recall.parsers.capture import JsonlCapture
from recall.parsers.checkpoint import (
    ChunkBudget,
    ChunkTarget,
    read_resume_metadata,
    read_resume_state,
    resume_checkpoint,
)
from recall.parsers.common import (
    accumulate_metric,
    build_tool_call,
    parse_timestamp,
    resolve_home,
)
from recall.parsers.protocol import (
    default_discover,
    default_live_candidates,
    default_watch_roots,
)
from recall.parsers.revision import parser_revision
from recall.parsers.skills import derive_skill_name

_USER_REQUEST_RE = re.compile(r"(?:^|\n)<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", re.DOTALL)
_SYSTEM_MESSAGE_RE = re.compile(r"(?:^|\n)<SYSTEM_MESSAGE>\s*(.*?)\s*</SYSTEM_MESSAGE>", re.DOTALL)


@dataclass
class AntigravityCliParser:
    """Parser for Google Antigravity CLI (``agy``) session transcripts.

    Sessions live at:
      ``~/.gemini/antigravity-cli/brain/<conversation-id>/.system_generated/logs/transcript_full.jsonl``

    Each JSONL line is one self-contained trajectory step with ``step_index``,
    ``source``, ``type``, ``status``, and ``created_at``:

    - ``USER_INPUT``: user prompt (typically wrapped in ``<USER_REQUEST>`` with
      trailing metadata tags).
    - ``PLANNER_RESPONSE``: assistant response with optional ``thinking``,
      ``content``, ``tool_calls`` (``[{"name", "args"}]``), ``model``, and
      per-call additive token deltas (``input_tokens``, ``cache_read_tokens``,
      ``output_tokens``). ``input_tokens`` is uncached input, so billable input
      is ``input_tokens + cache_read_tokens`` and ``cache_read_tokens`` fills
      ``cached_input_tokens`` (same shape as Claude Code / Pi Agent).
    - ``GENERIC``: tool execution output (``status`` may be ``DONE``,
      ``RUNNING``, or ``ERROR``; optional ``error`` message) → SYSTEM message.
    - ``SYSTEM_MESSAGE``: harness/background-task/subagent notification →
      SYSTEM message.
    - ``ERROR_MESSAGE``: API or runtime error → SYSTEM message.
    - ``CHECKPOINT``: context-compaction summary (also carries per-call token
      usage) → SYSTEM message.

    Per-session workspace and git metadata (``cwd``, ``git_repo``,
    ``git_branch``) live in the sibling SQLite sidecar at
    ``~/.gemini/antigravity-cli/conversations/<conversation-id>.db``
    (``trajectory_metadata_blob``). Malformed sidecar contents are non-fatal;
    unreadable or non-regular sidecar files fail closed via
    ``SidecarRefusedError``.
    """

    source: Source = Source.ANTIGRAVITY_CLI
    # Discovery roots from `[sources.<name>] roots`; None = default_roots(),
    # an empty tuple = configured to scan nothing.
    roots: tuple[Path, ...] | None = None

    @property
    def file_pattern(self) -> str:
        return "transcript_full.jsonl"

    def default_roots(self) -> list[Path]:
        return [resolve_home() / ".gemini" / "antigravity-cli" / "brain"]

    def watch_roots(self) -> list[Path]:
        return default_watch_roots(self)

    def discover(self) -> list[Path]:
        return default_discover(self)

    def sidecar_paths(self, path: Path) -> list[Path]:
        """Per-session conversation DB and WAL carrying workspace and git metadata.

        Declared in lockstep with ``_read_conversation_metadata`` so both main DB
        and uncheckpointed WAL updates participate in staleness fingerprinting
        (REQ-PARSE-015 / REQ-INDEX-018).
        """
        db_path = _conversation_db_path(path)
        if db_path is None:
            return []
        return [db_path, _wal_path(db_path)]

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
        # Message records are independent; native session metadata still
        # accumulates across them (REQ-INDEX-026).
        carried = read_resume_state(resume_state, offset=offset, supported=frozenset())
        metadata = read_resume_metadata(carried, offset=offset)
        absolute_path = str(path.expanduser().resolve())
        session_id_value = make_session_id(self.source.value, absolute_path)
        stat = path.stat()
        file_mtime = stat.st_mtime
        file_size = stat.st_size
        is_full_parse = offset == 0
        budget = ChunkBudget(target if not is_full_parse else None)

        messages: list[Message] = []
        tool_calls: list[ToolCall] = []
        stop_markers: list[StopMarker] = []
        is_complete = True
        input_tokens: int | None = None
        output_tokens: int | None = None
        cached_input_tokens: int | None = None

        with JsonlCapture(path, offset=offset) as capture:
            diagnostics = capture.diagnostics

            def checkpoint() -> NormalizationCheckpoint | None:
                return resume_checkpoint(
                    capture,
                    diagnostics=diagnostics,
                    metadata=metadata,
                    parser_revision=parser_revision(type(self)),
                    resumable=True,
                    message_idx_base=message_idx_base + len(messages),
                    orphan_tool_call_idx_base=orphan_tool_call_idx_base,
                )

            for entry in capture.records(
                stop=lambda: budget.stop(
                    capture,
                    identities=len(messages) + len(tool_calls) + len(stop_markers),
                    checkpoint=checkpoint,
                )
            ):
                timestamp = parse_timestamp(entry.get("created_at"))
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

                entry_type = entry.get("type")

                if entry_type == "USER_INPUT":
                    content = _extract_user_request(entry.get("content"))
                    if content:
                        messages.append(
                            _plain_message(
                                role=Role.USER,
                                content=content,
                                session_id=session_id_value,
                                idx=message_idx_base + len(messages),
                                timestamp=timestamp,
                            )
                        )

                elif entry_type == "PLANNER_RESPONSE":
                    if metadata.model is None:
                        metadata.model = _coerce_str(entry.get("model"))

                    input_tokens, output_tokens, cached_input_tokens = _accumulate_usage(
                        input_tokens,
                        output_tokens,
                        cached_input_tokens,
                        entry,
                    )

                    content = _coerce_str(entry.get("content"))
                    if content is None and entry.get("status") == "ERROR":
                        content = _coerce_str(entry.get("error"))
                    thinking = _coerce_str(entry.get("thinking"))
                    msg_tool_calls = _parse_tool_calls(entry.get("tool_calls"))

                    if content or thinking or msg_tool_calls:
                        msg = Message(
                            id=make_message_id(session_id_value, message_idx_base + len(messages)),
                            session_id=session_id_value,
                            idx=message_idx_base + len(messages),
                            role=Role.ASSISTANT,
                            content=content,
                            thinking=thinking,
                            timestamp=timestamp,
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
                        _record_stop_marker(stop_markers, entry, msg)
                    elif entry.get("status") == "ERROR" and (message_idx_base + len(messages)) > 0:
                        stop_markers.append(
                            StopMarker(
                                idx=message_idx_base + len(messages) - 1,
                                reason="error",
                                ends_turn=True,
                            )
                        )

                elif entry_type == "GENERIC":
                    content = _coerce_str(entry.get("content")) or _coerce_str(entry.get("error"))
                    if content:
                        content = _strip_generic_timestamp_preamble(content)
                        messages.append(
                            _plain_message(
                                role=Role.SYSTEM,
                                content=content,
                                session_id=session_id_value,
                                idx=message_idx_base + len(messages),
                                timestamp=timestamp,
                            )
                        )

                elif entry_type == "SYSTEM_MESSAGE":
                    content = _extract_system_message(entry.get("content"))
                    if content:
                        messages.append(
                            _plain_message(
                                role=Role.SYSTEM,
                                content=content,
                                session_id=session_id_value,
                                idx=message_idx_base + len(messages),
                                timestamp=timestamp,
                            )
                        )

                elif entry_type == "ERROR_MESSAGE":
                    content = _coerce_str(entry.get("error")) or _coerce_str(entry.get("content"))
                    if content:
                        messages.append(
                            _plain_message(
                                role=Role.SYSTEM,
                                content=content,
                                session_id=session_id_value,
                                idx=message_idx_base + len(messages),
                                timestamp=timestamp,
                            )
                        )

                elif entry_type == "CHECKPOINT":
                    input_tokens, output_tokens, cached_input_tokens = _accumulate_usage(
                        input_tokens,
                        output_tokens,
                        cached_input_tokens,
                        entry,
                    )
                    content = _coerce_str(entry.get("content"))
                    if content:
                        messages.append(
                            _plain_message(
                                role=Role.SYSTEM,
                                content=content,
                                session_id=session_id_value,
                                idx=message_idx_base + len(messages),
                                timestamp=timestamp,
                            )
                        )

                else:
                    diagnostics.append(
                        ParseDiagnostic(
                            "unsupported_record",
                            capture.record_start,
                            f"record: {entry_type!r}",
                        )
                    )

        published = metadata.model_copy()
        path_conversation_id = _conversation_id_from_path(path)
        if path_conversation_id is not None:
            published.source_session_id = path_conversation_id

        sidecar = _read_conversation_metadata(_conversation_db_path(path))
        if sidecar.cwd is not None:
            published.cwd = sidecar.cwd
        if sidecar.git_repo is not None:
            published.git_repo = sidecar.git_repo
        elif published.cwd is not None:
            published.git_repo = published.cwd
        if sidecar.git_branch is not None:
            published.git_branch = sidecar.git_branch
        if published.source_session_id is None and sidecar.conversation_id is not None:
            published.source_session_id = sidecar.conversation_id

        # Project-relative skill reads need the resolved session cwd
        # (REQ-PARSE-017).
        for tool_call in tool_calls:
            if tool_call.skill_name is None:
                tool_call.skill_name = derive_skill_name(
                    tool_call.tool_name,
                    tool_call.tool_input,
                    tool_call.bash_command,
                    cwd=published.cwd,
                )

        duration_seconds = None
        if published.started_at and published.ended_at:
            duration_seconds = int((published.ended_at - published.started_at).total_seconds())

        next_byte_offset = capture.next_byte_offset
        is_complete = is_complete and not diagnostics
        session = Session(
            **published.model_dump(),
            id=session_id_value,
            source=self.source,
            source_path=absolute_path,
            duration_seconds=duration_seconds,
            message_count=len(messages),
            tool_count=len(tool_calls),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
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
            normalization_checkpoint=resume_checkpoint(
                capture,
                metadata=metadata,
                parser_revision=parser_revision(type(self)),
                resumable=True,
                diagnostics=diagnostics,
                message_idx_base=message_idx_base + len(messages),
                orphan_tool_call_idx_base=orphan_tool_call_idx_base,
            ),
            tail_facts=TailFacts(
                stop_markers=tuple(stop_markers),
            ),
        )


_GENERIC_TIMESTAMP_LINE_RE = re.compile(
    r"^(?:Created|Completed) At:\s*"
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\r?\n?"
)


def _strip_generic_timestamp_preamble(text: str) -> str:
    """Strip leading duplicated Created At / Completed At ISO timestamp lines on GENERIC steps.

    Falls back to the original text if stripping leaves nothing so message indexing never shifts.
    """
    remaining = text
    while True:
        match = _GENERIC_TIMESTAMP_LINE_RE.match(remaining)
        if match is None:
            break
        remaining = remaining[match.end() :]
    stripped = remaining.lstrip("\r\n")
    return stripped if stripped else text


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


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        tool_name = str(name) if name else ""
        tool_input = item.get("args")
        call_id = item.get("id")
        calls.append(
            build_tool_call(
                tool_name,
                tool_input,
                tool_use_id=str(call_id) if call_id else None,
            )
        )
    return calls


def _record_stop_marker(
    stop_markers: list[StopMarker],
    entry: dict[str, Any],
    message: Message,
) -> None:
    """Derive turn stop state from a PLANNER_RESPONSE step."""
    if entry.get("status") == "ERROR":
        stop_markers.append(StopMarker(idx=message.idx, reason="error", ends_turn=True))
    elif message.tool_calls:
        stop_markers.append(StopMarker(idx=message.idx, reason="tool_calls", ends_turn=False))
    elif message.content:
        stop_markers.append(StopMarker(idx=message.idx, reason="stop", ends_turn=True))


def _accumulate_usage(
    input_tokens: int | None,
    output_tokens: int | None,
    cached_input_tokens: int | None,
    entry: dict[str, Any],
) -> tuple[int | None, int | None, int | None]:
    """Fold per-step Antigravity CLI token counters into session totals.

    Antigravity CLI records uncached ``input_tokens`` and ``cache_read_tokens``
    as separate per-call deltas, so billable input is their sum and
    ``cache_read_tokens`` fills ``cached_input_tokens`` (matching Claude Code /
    Pi Agent).
    """
    input_tokens = accumulate_metric(input_tokens, entry.get("input_tokens"))
    input_tokens = accumulate_metric(input_tokens, entry.get("cache_read_tokens"))
    output_tokens = accumulate_metric(output_tokens, entry.get("output_tokens"))
    cached_input_tokens = accumulate_metric(cached_input_tokens, entry.get("cache_read_tokens"))
    return input_tokens, output_tokens, cached_input_tokens


def _extract_user_request(raw: Any) -> str | None:
    text = _coerce_str(raw)
    if text is None:
        return None
    match = _USER_REQUEST_RE.search(text)
    if match is not None and match.group(1):
        return match.group(1)
    return text


def _extract_system_message(raw: Any) -> str | None:
    text = _coerce_str(raw)
    if text is None:
        return None
    match = _SYSTEM_MESSAGE_RE.search(text)
    if match is not None and match.group(1):
        return match.group(1)
    return text


def _coerce_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _conversation_id_from_path(path: Path) -> str | None:
    """Return ``<conversation-id>`` for ``.../brain/<id>/.system_generated/logs/<file>``."""
    resolved = path.expanduser()
    if (
        len(resolved.parents) >= 4
        and resolved.parent.name == "logs"
        and resolved.parents[1].name == ".system_generated"
        and resolved.parents[3].name == "brain"
    ):
        cid = resolved.parents[2].name
        return cid if cid else None
    return None


def _conversation_db_path(path: Path) -> Path | None:
    """Return ``<app-data-dir>/conversations/<id>.db`` when in standard brain layout."""
    resolved = path.expanduser()
    if (
        len(resolved.parents) >= 5
        and resolved.parent.name == "logs"
        and resolved.parents[1].name == ".system_generated"
        and resolved.parents[3].name == "brain"
    ):
        cid = resolved.parents[2].name
        if cid:
            return resolved.parents[4] / "conversations" / f"{cid}.db"
    return None


@dataclass(frozen=True)
class _ConversationMetadata:
    cwd: str | None = None
    git_repo: str | None = None
    git_branch: str | None = None
    conversation_id: str | None = None


def _wal_path(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.name}-wal")


def _read_optional_regular_bytes(path: Path) -> bytes | None:
    """Read a regular sidecar file via ``open_source``; None if absent, refuse on error."""
    try:
        with open_source(path) as stream:
            return stream.read()
    except FileNotFoundError:
        return None
    except OSError as err:
        raise SidecarRefusedError(str(err)) from err


def _read_conversation_metadata(db_path: Path | None) -> _ConversationMetadata:
    """Read workspace and git metadata from ``conversations/<id>.db`` (+ ``-wal``).

    Missing files return empty metadata. Non-regular files (e.g., FIFOs) and
    unreadable files (e.g., ``chmod 000``) on either the main DB or its WAL are
    opened through ``open_source`` and propagate ``SidecarRefusedError`` so the
    coordinator fails closed rather than mistaking a refused sidecar for absent
    metadata. SQLite or protobuf decode errors on a readable regular file are
    non-fatal.
    """
    if db_path is None:
        return _ConversationMetadata()
    # Read the WAL before the main DB so an intervening checkpoint cannot drop
    # newly committed frames (if a checkpoint runs in between, the main DB
    # already holds those pages).
    wal_bytes = _read_optional_regular_bytes(_wal_path(db_path))
    db_bytes = _read_optional_regular_bytes(db_path)
    if not db_bytes:
        return _ConversationMetadata()

    with tempfile.TemporaryDirectory(prefix="recall-agy-sidecar-") as tmp_dir:
        tmp_db = Path(tmp_dir) / "conversation.db"
        tmp_db.write_bytes(db_bytes)
        if wal_bytes:
            _wal_path(tmp_db).write_bytes(wal_bytes)
        try:
            conn = sqlite3.connect(str(tmp_db))
        except sqlite3.Error:
            return _ConversationMetadata()
        try:
            row = conn.execute(
                "SELECT data FROM trajectory_metadata_blob WHERE id = 'main' LIMIT 1"
            ).fetchone()
            if row is None:
                row = conn.execute("SELECT data FROM trajectory_metadata_blob LIMIT 1").fetchone()
        except sqlite3.Error:
            return _ConversationMetadata()
        finally:
            conn.close()
    if row is None or not isinstance(row[0], (bytes, bytearray)):
        return _ConversationMetadata()
    return _decode_trajectory_metadata(bytes(row[0]))


def _decode_trajectory_metadata(blob: bytes) -> _ConversationMetadata:
    """Extract workspace/git fields from protobuf ``trajectory_metadata_blob``.

    Observed wire schema:
      - field 1 (repeated message, WorkspaceInfo):
          - field 1 (string): workspace ``file://`` URI → ``cwd``
          - field 2 (string): git root ``file://`` URI → ``git_repo``
          - field 4 (string): git branch → ``git_branch``
      - field 6 (string): conversation / cascade ID
      - field 7 (repeated string): workspace ``file://`` URIs (fallback ``cwd``)
    """
    cwd: str | None = None
    git_repo: str | None = None
    git_branch: str | None = None
    conversation_id: str | None = None

    for field_num, wire_type, payload in _iter_proto_fields(blob):
        if wire_type != 2 or not isinstance(payload, bytes):
            continue
        if field_num == 1:
            ws_cwd, ws_git_repo, ws_branch = _decode_workspace_info(payload)
            if cwd is None and ws_cwd is not None:
                cwd = ws_cwd
            if git_repo is None and ws_git_repo is not None:
                git_repo = ws_git_repo
            if git_branch is None and ws_branch is not None:
                git_branch = ws_branch
        elif field_num == 6 and conversation_id is None:
            conversation_id = _decode_utf8(payload)
        elif field_num == 7 and cwd is None:
            cwd = _file_uri_to_path(_decode_utf8(payload))

    return _ConversationMetadata(
        cwd=cwd,
        git_repo=git_repo,
        git_branch=git_branch,
        conversation_id=conversation_id,
    )


def _decode_workspace_info(blob: bytes) -> tuple[str | None, str | None, str | None]:
    cwd: str | None = None
    git_repo: str | None = None
    git_branch: str | None = None
    for field_num, wire_type, payload in _iter_proto_fields(blob):
        if wire_type != 2 or not isinstance(payload, bytes):
            continue
        if field_num == 1 and cwd is None:
            cwd = _file_uri_to_path(_decode_utf8(payload))
        elif field_num == 2 and git_repo is None:
            git_repo = _file_uri_to_path(_decode_utf8(payload))
        elif field_num == 4 and git_branch is None:
            git_branch = _decode_utf8(payload)
    return cwd, git_repo, git_branch


def _iter_proto_fields(data: bytes) -> list[tuple[int, int, int | bytes]]:
    """Parse top-level protobuf wire fields, stopping cleanly on malformed input."""
    fields: list[tuple[int, int, int | bytes]] = []
    pos = 0
    length = len(data)
    while pos < length:
        key, pos = _read_varint(data, pos)
        if key is None:
            break
        field_num = key >> 3
        wire_type = key & 7
        if field_num == 0:
            break
        if wire_type == 0:
            value, pos = _read_varint(data, pos)
            if value is None:
                break
            fields.append((field_num, wire_type, value))
        elif wire_type == 1:
            if pos + 8 > length:
                break
            fields.append((field_num, wire_type, data[pos : pos + 8]))
            pos += 8
        elif wire_type == 2:
            size, pos = _read_varint(data, pos)
            if size is None or pos + size > length:
                break
            fields.append((field_num, wire_type, data[pos : pos + size]))
            pos += size
        elif wire_type == 5:
            if pos + 4 > length:
                break
            fields.append((field_num, wire_type, data[pos : pos + 4]))
            pos += 4
        else:
            break
    return fields


def _read_varint(data: bytes, pos: int) -> tuple[int | None, int]:
    shift = 0
    result = 0
    length = len(data)
    while pos < length and shift < 64:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7
    return None, pos


def _decode_utf8(raw: bytes) -> str | None:
    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    return text if text else None


def _file_uri_to_path(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("file://"):
        parsed = urlparse(value)
        if parsed.scheme == "file":
            path = unquote(parsed.path)
            return path if path else None
    if value.startswith("/"):
        return value
    return None
