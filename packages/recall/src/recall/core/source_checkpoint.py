"""Desired source location and committed capture proofs have separate ownership."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace

from pydantic import TypeAdapter, ValidationError

from recall.core.models import NORMALIZATION_CHECKPOINT_BYTES_MAX, NormalizationCheckpoint
from recall.core.source import DecodedPrefix, SessionSource

logger = logging.getLogger(__name__)
_SOURCE = TypeAdapter(SessionSource)


@dataclass(frozen=True)
class SourceCheckpoint:
    locator: SessionSource | None = None
    resume: NormalizationCheckpoint | None = None
    decoded: DecodedPrefix | None = None

    @classmethod
    def decode(cls, raw: str | None) -> SourceCheckpoint:
        if raw is None or len(raw) > NORMALIZATION_CHECKPOINT_BYTES_MAX:
            return cls()
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return cls()
            locator = (
                _SOURCE.validate_python(data["source_locator"])
                if "source_locator" in data
                else None
            )
            decoded = (
                DecodedPrefix.model_validate(data["decoded_prefix"])
                if "decoded_prefix" in data
                else None
            )
        except (ValueError, ValidationError):
            return cls()
        return cls(locator, NormalizationCheckpoint.decode(raw), decoded)

    def encode(self) -> str | None:
        if self.locator is None and self.resume is None and self.decoded is None:
            return None
        # Existing readers ignore optional fields on version-3 resume proofs.
        # A metadata/content-only envelope is version 0 and cannot authorize resume.
        data = json.loads(self.resume.encode()) if self.resume is not None else {"version": 0}
        if self.locator is not None:
            data["source_locator"] = _SOURCE.dump_python(self.locator, mode="json")
        if self.decoded is not None:
            data["decoded_prefix"] = self.decoded.model_dump()
        encoded = json.dumps(data, separators=(",", ":"))
        if len(encoded) > NORMALIZATION_CHECKPOINT_BYTES_MAX:
            if self.resume is not None:
                logger.warning("resume proof declined: source envelope exceeds durable bound")
                return replace(self, resume=None).encode()
            raise ValueError("source checkpoint exceeds durable size bound")
        return encoded
