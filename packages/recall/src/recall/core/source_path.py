from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class SourcePath:
    """A parser's identity for an absolute canonical transcript pathname."""

    source: str
    path: str

    @classmethod
    def resolve(cls, source: str, path: str | Path) -> SourcePath:
        return cls(source, str(Path(path).expanduser().resolve()))
