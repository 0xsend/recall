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

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from recall.core.models import NormalizationCheckpoint, ParseDiagnostic
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
    unsupported = sorted(key for key in state if key not in supported)
    if unsupported:
        raise UnsupportedResumeState(f"unsupported resume state keys: {unsupported}")
    return dict(state)


def read_resume_flag(state: Mapping[str, Any], key: str) -> bool:
    """Read one carried boolean, refusing a value of any other type."""
    value = state.get(key, False)
    if not isinstance(value, bool):
        raise UnsupportedResumeState(f"resume state {key!r} must be a boolean, got {value!r}")
    return value


# A final line the writer has not finished yet. The capture stops before it
# without acknowledging it and never yields it, so the committed offset, the
# committed digest and the adapter's carried state all still describe the last
# complete record -- the same boundary a clean parse ending there would
# declare. Every other diagnostic either damages the prefix or is raised by an
# adapter mid-record, which pins the offset behind state that has moved on.
_RESUMABLE_DIAGNOSTIC = "unterminated_tail"


def resume_checkpoint(
    capture: JsonlCapture,
    *,
    parser_revision: str,
    resumable: bool,
    diagnostics: Sequence[ParseDiagnostic],
    message_idx_base: int,
    orphan_tool_call_idx_base: int,
    adapter_state: dict[str, Any] | None = None,
) -> NormalizationCheckpoint | None:
    """Stamp the boundary the adapter just captured, or decline to.

    A diagnostic other than an unterminated tail declines: the capture refuses
    to acknowledge past a diagnosed record, so the committed prefix no longer
    describes a boundary the adapter reasoned about.
    """
    assert message_idx_base >= 0 and orphan_tool_call_idx_base >= 0
    assert capture.source_dev is not None and capture.source_inode is not None
    if not resumable:
        return None
    if any(diagnostic.kind != _RESUMABLE_DIAGNOSTIC for diagnostic in diagnostics):
        return None
    checkpoint = NormalizationCheckpoint(
        parser_revision=parser_revision,
        offset=capture.next_byte_offset,
        prefix_sha256=capture.committed_prefix_sha256,
        message_idx_base=message_idx_base,
        orphan_tool_call_idx_base=orphan_tool_call_idx_base,
        source_dev=capture.source_dev,
        source_inode=capture.source_inode,
        adapter_state=adapter_state or {},
    )
    try:
        checkpoint.encode()
    except ValueError as err:
        # Correctness still wins over an unbounded hot catalog row, but this
        # degradation must be visible: otherwise an operator sees only a
        # permanent return to whole-file reparses with every status green.
        logger.warning("normalization checkpoint declined path=%s error=%s", capture.path, err)
        return None
    return checkpoint
