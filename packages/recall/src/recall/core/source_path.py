from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recall.core.transcripts import plain_transcript_path, transcript_identity


@dataclass(frozen=True, order=True)
class SourcePath:
    """A parser's identity for an absolute canonical transcript pathname."""

    source: str
    path: str

    @classmethod
    def resolve(cls, source: str, path: str | Path) -> SourcePath:
        expanded = Path(path).expanduser()
        try:
            canonical = transcript_identity(expanded)
        except OSError:
            canonical = plain_transcript_path(expanded).resolve()
        return cls(source, str(canonical))
