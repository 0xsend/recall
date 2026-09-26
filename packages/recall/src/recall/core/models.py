from __future__ import annotations

import base64
import json
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from recall.core.types import UNATTRIBUTED_HOST, Role, Source

_LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")


def _scrub_lone_surrogates(value: Any) -> Any:
    """Replace unpaired UTF-16 surrogates with U+FFFD.

    Agents truncate tool output by character count, which can split an emoji's
    surrogate pair and strand its high half in the JSONL. `json.loads` accepts
    the lone surrogate, but the resulting `str` cannot be encoded to UTF-8, so
    DuckDB rejects the parameter — and it takes the *whole session* down with
    it, not just the one message. Scrubbing here keeps the invariant at the
    model boundary, so every parser gets it and a new one cannot forget to.
    """

    if not isinstance(value, str) or value.isascii():
        return value
    if _LONE_SURROGATE_RE.search(value) is None:
        return value
    return _LONE_SURROGATE_RE.sub("�", value)


# Free text that originates in an agent transcript and may have been truncated
# mid-codepoint. Storage-bound fields whose values recall itself generates do
# not need this.
AgentText = Annotated[str, BeforeValidator(_scrub_lone_surrogates)]
OptionalAgentText = Annotated[str | None, BeforeValidator(_scrub_lone_surrogates)]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    message_id: str | None
    idx: int
    tool_name: str
    tool_input: dict[str, Any] | None = None

    bash_command: OptionalAgentText = None
    bash_base: OptionalAgentText = None
    bash_sub: OptionalAgentText = None
    is_compound: bool = False

    # Subagent fields — populated when tool_name == "Agent" or "Skill"
    agent_id: str | None = None
    subagent_type: str | None = None
    subagent_description: OptionalAgentText = None
    subagent_model: str | None = None
    skill_name: str | None = None

    # Harness-native id of the tool_use block (REQ-LIVE-005). `id` above is a
    # positional hash of (message_id, idx), so this is the only key a
    # tool_result arriving in a later incremental chunk can be paired on. It is
    # not a `tool_calls` column: see the `tool_use_ids` side table.
    tool_use_id: str | None = None

    bash_embedding: list[float] | None = None


class ToolResult(BaseModel):
    """A harness tool_result, paired to the tool_use it answers (REQ-LIVE-006)."""

    model_config = ConfigDict(extra="forbid")

    tool_use_id: str
    result_summary: AgentText = ""
    is_error: bool = False
    completed_at: datetime | None = None


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    idx: int
    role: Role
    content: OptionalAgentText = None
    thinking: OptionalAgentText = None
    timestamp: datetime | None = None
    has_thinking: bool = False
    # Present when the message belongs to a subagent invocation rather than
    # the main conversation; None for top-level messages.
    agent_id: str | None = None
    context_text: AgentText = ""
    context_mode: str = "off"
    tool_calls: list[ToolCall] = Field(default_factory=list)

    content_embedding: list[float] | None = None
    thinking_embedding: list[float] | None = None


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: Source
    source_path: str
    source_session_id: str | None = None

    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: int | None = None

    model: str | None = None
    cwd: str | None = None
    git_repo: str | None = None
    git_branch: str | None = None
    # REQ-HOST-API-002: fleet/agent surfaces; stamped at index (session_state.host).
    host: str = UNATTRIBUTED_HOST

    message_count: int = 0
    tool_count: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None

    is_complete: bool = True
    file_mtime: float
    file_size: int
    # Newest mtime across the parser's declared sidecars; 0.0 when it has none.
    sidecar_mtime: float = 0.0
    indexed_at: datetime | None = None

    messages: list[Message] = Field(default_factory=list)
    orphan_tool_calls: list[ToolCall] = Field(default_factory=list)


@dataclass(frozen=True)
class StopMarker:
    """One harness stop marker, anchored to the message it closes (REQ-LIVE-005).

    ``reason`` is the harness's own vocabulary, passed through unnormalized.
    ``ends_turn`` is the parser's reading of that vocabulary: only the parser
    knows that Claude Code's ``end_turn`` closes a turn while ``tool_use`` does
    not, and keeping that knowledge here leaves the read-time derivation
    generic over every harness.
    """

    idx: int
    reason: str
    ends_turn: bool


@dataclass(frozen=True)
class TailFacts:
    """What the parsed lines say about the current turn (REQ-LIVE-005).

    The parser only surfaces markers; ``services/live.py`` derives turn state
    from them at read time. A harness that emits none of these leaves every
    field empty, so the derivation resolves ``unknown`` rather than guessing.
    """

    tool_results: tuple[ToolResult, ...] = ()
    stop_markers: tuple[StopMarker, ...] = ()
    session_ended: bool = False


@dataclass(frozen=True)
class ParseDiagnostic:
    """A source record the parser could not safely commit."""

    kind: Literal[
        "unterminated_tail",
        "malformed_record",
        "unsupported_record",
        "resource_limit",
        "source_changed",
    ]
    byte_offset: int
    detail: str


NORMALIZATION_CHECKPOINT_VERSION = 3
# The envelope rides in one catalog row and is rewritten by every acknowledgement.
# Compressible adapter state stays within the durable bound; genuinely larger
# state declines rather than growing the hot catalog row without limit.
NORMALIZATION_CHECKPOINT_BYTES_MAX = 64 * 1024
_NORMALIZATION_CHECKPOINT_STATE_BYTES_MAX = 4 * 1024 * 1024
_COMPRESSED_ADAPTER_STATE_KEY = "_recall_zlib_v1"
_NORMALIZATION_CHECKPOINT_NESTING_MAX = 64

_SHA256_RE = re.compile("[0-9a-f]{64}")


def _checkpoint_state_nesting_within_bound(value: object) -> bool:
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, dict):
            if depth >= _NORMALIZATION_CHECKPOINT_NESTING_MAX and item:
                return False
            stack.extend((nested, depth + 1) for nested in item.values())
        elif isinstance(item, list):
            if depth >= _NORMALIZATION_CHECKPOINT_NESTING_MAX and item:
                return False
            stack.extend((nested, depth + 1) for nested in item)
    return True


@dataclass(frozen=True)
class NormalizationCheckpoint:
    """Proof that the parse boundary at ``offset`` can be resumed (REQ-INDEX-025).

    The encoded envelope is written to ``source_files.normalization_checkpoint``
    by the same acknowledgement that commits the rows it describes. Storage keeps
    it opaque; ``decode`` validates shape only, and every claim it makes about the
    source -- the parser revision, the offset, the prefix digest -- is checked
    against the catalog before a resume is allowed.
    """

    parser_revision: str
    offset: int
    prefix_sha256: str
    message_idx_base: int
    orphan_tool_call_idx_base: int
    source_dev: int
    source_inode: int
    adapter_state: dict[str, Any] = field(default_factory=dict)
    version: int = NORMALIZATION_CHECKPOINT_VERSION

    def encode(self) -> str:
        """Serialize the envelope, compressing carried state before declining."""
        if self.source_dev < 0 or self.source_inode < 0:
            raise ValueError("normalization checkpoint file identity must be non-negative")
        if set(self.adapter_state) == {_COMPRESSED_ADAPTER_STATE_KEY}:
            raise ValueError("normalization checkpoint state uses a reserved key")
        if not _checkpoint_state_nesting_within_bound(self.adapter_state):
            raise ValueError("normalization checkpoint state exceeds the nesting bound")
        envelope = {
            "version": self.version,
            "parser_revision": self.parser_revision,
            "offset": self.offset,
            "prefix_sha256": self.prefix_sha256,
            "message_idx_base": self.message_idx_base,
            "orphan_tool_call_idx_base": self.orphan_tool_call_idx_base,
            "source_dev": self.source_dev,
            "source_inode": self.source_inode,
            "adapter_state": self.adapter_state,
        }
        encoded = json.dumps(envelope, separators=(",", ":"))
        if len(encoded) > NORMALIZATION_CHECKPOINT_BYTES_MAX:
            raw_state = json.dumps(self.adapter_state, separators=(",", ":")).encode()
            if len(raw_state) > _NORMALIZATION_CHECKPOINT_STATE_BYTES_MAX:
                raise ValueError("normalization checkpoint state exceeds the compression bound")
            compressed = base64.b64encode(zlib.compress(raw_state, level=9)).decode("ascii")
            envelope["adapter_state"] = {_COMPRESSED_ADAPTER_STATE_KEY: compressed}
            encoded = json.dumps(envelope, separators=(",", ":"))
        if len(encoded) > NORMALIZATION_CHECKPOINT_BYTES_MAX:
            raise ValueError("normalization checkpoint exceeds the durable size bound")
        return encoded

    @classmethod
    def decode(cls, raw: str | None) -> NormalizationCheckpoint | None:
        """Return the stored envelope, or None when it carries no usable proof.

        Decoding fails closed: a legacy NULL, a truncated or oversized value, a
        version this build does not know, and any field of the wrong shape all
        resolve to "no resume proof", which sends the caller down the full parse
        path instead of raising into it.
        """
        if raw is None or len(raw) > NORMALIZATION_CHECKPOINT_BYTES_MAX:
            return None
        try:
            envelope = json.loads(raw)
        except (ValueError, RecursionError):
            return None
        if not isinstance(envelope, dict):
            return None
        counters = [
            envelope.get(name)
            for name in (
                "version",
                "offset",
                "message_idx_base",
                "orphan_tool_call_idx_base",
                "source_dev",
                "source_inode",
            )
        ]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in counters):
            return None
        version, offset, message_idx_base, orphan_tool_call_idx_base, source_dev, source_inode = (
            counters
        )
        if version != NORMALIZATION_CHECKPOINT_VERSION:
            return None
        if (
            offset < 0
            or message_idx_base < 0
            or orphan_tool_call_idx_base < 0
            or source_dev < 0
            or source_inode < 0
        ):
            return None
        parser_revision = envelope.get("parser_revision")
        if not isinstance(parser_revision, str) or not parser_revision:
            return None
        prefix_sha256 = envelope.get("prefix_sha256")
        if not isinstance(prefix_sha256, str) or _SHA256_RE.fullmatch(prefix_sha256) is None:
            return None
        adapter_state = envelope.get("adapter_state")
        if not isinstance(adapter_state, dict):
            return None
        if set(adapter_state) == {_COMPRESSED_ADAPTER_STATE_KEY}:
            compressed = adapter_state[_COMPRESSED_ADAPTER_STATE_KEY]
            if not isinstance(compressed, str):
                return None
            try:
                packed = base64.b64decode(compressed, validate=True)
                decompressor = zlib.decompressobj()
                unpacked = decompressor.decompress(
                    packed, _NORMALIZATION_CHECKPOINT_STATE_BYTES_MAX + 1
                )
                if (
                    len(unpacked) > _NORMALIZATION_CHECKPOINT_STATE_BYTES_MAX
                    or decompressor.unconsumed_tail
                    or decompressor.unused_data
                    or not decompressor.eof
                ):
                    return None
                adapter_state = json.loads(unpacked)
            except (ValueError, UnicodeError, zlib.error, RecursionError):
                return None
            if not isinstance(adapter_state, dict):
                return None
        if not _checkpoint_state_nesting_within_bound(adapter_state):
            return None
        if any(not isinstance(key, str) for key in adapter_state):
            return None
        return cls(
            parser_revision=parser_revision,
            offset=offset,
            prefix_sha256=prefix_sha256,
            message_idx_base=message_idx_base,
            orphan_tool_call_idx_base=orphan_tool_call_idx_base,
            source_dev=source_dev,
            source_inode=source_inode,
            adapter_state=adapter_state,
            version=version,
        )


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing a session file, supporting byte-offset incremental parsing.

    When ``is_full_parse`` is True (offset=0), ``session`` contains all messages.
    When False (offset>0), ``session`` contains only newly parsed messages and
    metadata from the new lines — the caller merges into the existing DB row.
    """

    session: Session
    next_byte_offset: int
    is_full_parse: bool
    tail_facts: TailFacts = field(default_factory=TailFacts)
    diagnostics: tuple[ParseDiagnostic, ...] = ()
    # SHA-256 of exactly the bytes captured for this parse, never a later
    # reread of a concurrently written source.
    captured_prefix_sha256: str | None = None
    # SHA-256 of the bytes before the requested parse offset, read from the
    # same file handle as the suffix. This closes the verification/parse race.
    initial_prefix_sha256: str | None = None
    source_dev: int | None = None
    source_inode: int | None = None
    captured_size: int = 0
    committed_prefix_sha256: str | None = None
    # The adapter's own declaration that the boundary it just captured can be
    # resumed (REQ-INDEX-026). None means "re-read this source from zero"; it
    # is the only safe reading of an open normalization boundary.
    normalization_checkpoint: NormalizationCheckpoint | None = None
