"""Adapter declarations about resuming normalization (REQ-INDEX-026).

An adapter finishes a parse holding whatever normalization state its format
leaves open -- an unterminated turn, an assistant message a later record may
amend, an accumulating step. Only the adapter knows whether the next suffix
can be normalized to the same rows a full parse would produce, so only the
adapter may issue the checkpoint that permits a resume.

Declining is the correct answer at an open boundary, never a defect: the
caller re-reads that source from zero and loses time, not rows.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from recall.core.models import NormalizationCheckpoint, ParseDiagnostic, SessionMetadata
from recall.parsers.capture import JsonlCapture

logger = logging.getLogger(__name__)


class UnsupportedResumeState(ValueError):
    """A stored resume state this build's adapter cannot honor.

    Distinct from every other parse failure on purpose. A caller that meets it
    knows the checkpoint -- not the transcript, and not the adapter -- is the
    unusable part, so it can fall back to a full parse deliberately instead of
    treating an arbitrary parser defect as a routine fallback.
    """


def read_resume_state(
    state: Mapping[str, Any] | None, *, offset: int, supported: frozenset[str]
) -> dict[str, Any]:
    """Return the carried state, refusing any shape this adapter cannot read.

    A state offered at offset zero is refused too: there is no prefix for it to
    describe, so honoring it would normalize the head of the file as if records
    the parse never saw had already been consumed.
    """
    if state is None:
        return {}
    if not isinstance(state, Mapping):
        raise UnsupportedResumeState(f"resume state must be a JSON object, got {type(state)!r}")
    if state and offset == 0:
        raise UnsupportedResumeState("resume state carries no meaning at offset zero")
    unsupported = sorted(key for key in state if key not in supported | {"metadata"})
    if unsupported:
        raise UnsupportedResumeState(f"unsupported resume state keys: {unsupported}")
    return dict(state)


def read_resume_metadata(state: Mapping[str, Any], *, offset: int) -> SessionMetadata:
    """Restore the native accumulator, refusing absent or unsupported fields.

    The envelope bounds carried state, including metadata. Its shared boundary
    verdict declines oversized accumulators rather than truncating elections.
    """
    if offset == 0:
        return SessionMetadata()
    raw = state.get("metadata")
    if not isinstance(raw, dict) or set(raw) != set(SessionMetadata.model_fields):
        raise UnsupportedResumeState("resume state lacks complete native metadata")
    try:
        # JSON mode accepts the serialized timestamps while retaining strict
        # string/null types and refusing unknown fields.
        return SessionMetadata.model_validate_json(json.dumps(raw), strict=True)
    except (ValidationError, TypeError) as err:
        raise UnsupportedResumeState("unsupported native metadata accumulator") from err


def read_resume_flag(state: Mapping[str, Any], key: str, *, required: bool = False) -> bool:
    """Read one carried boolean, refusing a value of any other type."""
    if required and key not in state:
        raise UnsupportedResumeState(f"resume state lacks required flag {key!r}")
    value = state.get(key, False)
    if not isinstance(value, bool):
        raise UnsupportedResumeState(f"resume state {key!r} must be a boolean, got {value!r}")
    return value


def resume_checkpoint(
    capture: JsonlCapture,
    *,
    parser_revision: str,
    resumable: bool,
    diagnostics: Sequence[ParseDiagnostic],
    message_idx_base: int,
    orphan_tool_call_idx_base: int,
    metadata: SessionMetadata,
    adapter_state: dict[str, Any] | None = None,
) -> NormalizationCheckpoint | None:
    """Stamp the boundary the adapter just captured, or decline to.

    A diagnostic other than an unterminated tail declines. A malformed or
    oversized record stopped the capture before the boundary. An unsupported
    record was acknowledged, but a suffix resumed past it would not carry its
    diagnostic, so only a full parse can keep the gap visible and report the
    session incomplete; every append to such a source takes the reference
    path. An unterminated tail was never yielded, so the boundary is the same
    one a clean parse ending at the last complete record would declare.
    """
    assert message_idx_base >= 0 and orphan_tool_call_idx_base >= 0
    assert capture.source_dev is not None and capture.source_inode is not None
    checkpoint = NormalizationCheckpoint(
        parser_revision=parser_revision,
        offset=capture.next_byte_offset,
        prefix_sha256=capture.committed_prefix_sha256,
        message_idx_base=message_idx_base,
        orphan_tool_call_idx_base=orphan_tool_call_idx_base,
        source_dev=capture.source_dev,
        source_inode=capture.source_inode,
        adapter_state={**(adapter_state or {}), "metadata": metadata.model_dump(mode="json")},
    )
    verdict = boundary_verdict(
        checkpoint, resumable=resumable, diagnostics=diagnostics, path=capture.path
    )
    assert not capture.stopped or verdict is BoundaryVerdict.RESUMABLE
    return checkpoint if verdict is BoundaryVerdict.RESUMABLE else None


class BoundaryVerdict(Enum):
    RESUMABLE = "resumable"
    NOT_RESUMABLE = "not_resumable"
    DIAGNOSED = "diagnosed"


def boundary_verdict(
    checkpoint: NormalizationCheckpoint,
    *,
    resumable: bool,
    diagnostics: Sequence[ParseDiagnostic],
    path: Path,
) -> BoundaryVerdict:
    """One eligibility decision for both EOF and artificial record boundaries."""
    if not all(diagnostic.is_unterminated_tail for diagnostic in diagnostics):
        return BoundaryVerdict.DIAGNOSED
    if not resumable:
        return BoundaryVerdict.NOT_RESUMABLE
    try:
        checkpoint.encode()
    except ValueError as err:
        logger.warning(
            "normalization checkpoint declined path=%s offset=%s error=%s",
            path,
            checkpoint.offset,
            err,
        )
        return BoundaryVerdict.NOT_RESUMABLE
    return BoundaryVerdict.RESUMABLE


@dataclass(frozen=True)
class ChunkTarget:
    """Stop after either amount of accepted normalized work or source bytes."""

    identities: int
    bytes: int

    def __post_init__(self) -> None:
        if self.identities <= 0 or self.bytes <= 0:
            raise ValueError("chunk targets must be positive")


@dataclass
class ChunkBudget:
    target: ChunkTarget | None
    deferred_identities: int = 0

    def accept_deferred(self, identities: int) -> None:
        assert identities >= 0
        self.deferred_identities += identities

    def reached(self, capture: JsonlCapture, identities: int) -> bool:
        if self.target is None:
            return False
        return (
            identities + self.deferred_identities >= self.target.identities
            or capture.next_byte_offset - capture.offset >= self.target.bytes
        )

    def stop(
        self,
        capture: JsonlCapture,
        *,
        identities: int,
        checkpoint: Callable[[], NormalizationCheckpoint | None],
    ) -> bool:
        return self.reached(capture, identities) and checkpoint() is not None
