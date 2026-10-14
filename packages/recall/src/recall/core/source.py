"""Session identity and its current filesystem representation."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SessionSource:
    canonical_path: Path
    physical_path: Path
    encoding: Literal["jsonl", "zstd"]

    @classmethod
    def resolve(cls, source: str, path: Path) -> SessionSource:
        expanded = path.expanduser()
        if source == "codex" and expanded.name.endswith((".jsonl", ".jsonl.zst")):
            plain = expanded.with_suffix("") if expanded.name.endswith(".zst") else expanded
            compressed = plain.with_name(plain.name + ".zst")
            if os.path.lexists(plain):
                expanded = plain
            elif os.path.lexists(compressed):
                expanded = compressed
        return cls.from_resolved_path(source, expanded.resolve())

    @classmethod
    def from_resolved_path(cls, source: str, path: Path) -> SessionSource:
        """Select a representation for an absolute path already resolved by its caller."""
        assert path.is_absolute()
        if source != "codex":
            return cls(path, path, "jsonl")
        canonical = path.with_suffix("") if path.name.endswith(".jsonl.zst") else path
        if os.path.lexists(canonical):
            return cls(canonical, canonical, _codex_encoding(canonical))
        compressed = canonical.with_name(canonical.name + ".zst")
        if os.path.lexists(compressed):
            return cls(canonical, compressed, "zstd")
        return cls(canonical, canonical, "jsonl")

    @property
    def present(self) -> bool:
        return os.path.lexists(self.physical_path)


def matches_session_file(source: str, name: str, pattern: str) -> bool:
    if source == "codex":
        return name.startswith("rollout") and name.endswith((".jsonl", ".jsonl.zst"))
    return fnmatch.fnmatchcase(name, pattern)


def _codex_encoding(path: Path) -> Literal["jsonl", "zstd"]:
    # Aliases retain their resolved legacy identity, whose name may not carry
    # the rollout suffix. The bounded header keeps encoding recoverable from
    # that persisted path after restart or representation replacement.
    try:
        with path.open("rb") as source:
            header = source.read(4)
    except FileNotFoundError:
        return "jsonl"
    magic = int.from_bytes(header, "little")
    compressed = magic == 0xFD2FB528 or 0x184D2A50 <= magic <= 0x184D2A5F
    return "zstd" if compressed else "jsonl"
