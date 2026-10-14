"""Canonical session identity and its selected filesystem representation."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_DECODED_BYTES = 512 * 1024 * 1024
Encoding = Literal["jsonl", "zstd"]


class DecodedPrefix(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    size: Annotated[int, Field(ge=0, le=MAX_DECODED_BYTES)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


@dataclass(frozen=True)
class SessionSource:
    canonical_path: Path
    physical_path: Path
    encoding: Encoding
    source: str = ""
    root_path: Path | None = None
    discovery_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.canonical_path.is_absolute() or not self.physical_path.is_absolute():
            raise ValueError("source paths must be absolute")
        if any(
            path is not None and not path.is_absolute()
            for path in (self.root_path, self.discovery_path)
        ):
            raise ValueError("source locator paths must be absolute")

    @classmethod
    def resolve(cls, source: str, path: Path) -> SessionSource:
        """Select a discovery input before resolving aliases."""
        expanded = preferred_file(source, path.expanduser().absolute())
        compressed = source == "codex" and expanded.name.endswith(".jsonl.zst")
        physical = expanded.resolve()
        canonical = (
            physical.with_suffix("") if compressed and physical.suffix == ".zst" else physical
        )
        encoding = (
            _encoding(physical, "zstd" if compressed else "jsonl") if source == "codex" else "jsonl"
        )
        return cls(
            canonical,
            physical,
            encoding,
            source,
            discovery_path=expanded if physical != expanded else None,
        )

    @classmethod
    def for_session(
        cls, source: str, canonical: Path, locator: SessionSource | None = None
    ) -> SessionSource:
        """Resolve a persisted identity without interpreting its filename again."""
        canonical = canonical.expanduser().absolute()
        if source != "codex":
            return cls(canonical, canonical, "jsonl", source)
        if locator is not None and locator.discovery_path is not None:
            selected = preferred_file(source, locator.discovery_path).resolve()
            if selected in representation_paths(canonical, locator) and file_exists(selected):
                return cls(
                    canonical,
                    selected,
                    _encoding(selected, locator.encoding),
                    source,
                    locator.root_path,
                    locator.discovery_path,
                )
        if file_exists(canonical):
            physical = canonical
            fallback = (
                locator.encoding if locator and locator.physical_path == canonical else "jsonl"
            )
        elif locator is not None and file_exists(locator.physical_path):
            physical = locator.physical_path
            fallback = locator.encoding
        elif file_exists(canonical.with_name(canonical.name + ".zst")):
            physical = canonical.with_name(canonical.name + ".zst")
            fallback = "zstd"
        elif canonical.suffix == ".zst" and file_exists(canonical.with_suffix("")):
            physical = canonical.with_suffix("")
            fallback = "jsonl"
        else:
            physical = locator.physical_path if locator is not None else canonical
            fallback = locator.encoding if locator is not None else "jsonl"
        return cls(
            canonical,
            physical,
            _encoding(physical, fallback),
            source,
            locator.root_path if locator else None,
            locator.discovery_path if locator else None,
        )

    @property
    def present(self) -> bool:
        return file_exists(self.physical_path)


def file_exists(path: Path) -> bool:
    """Follow aliases; only actual absence is a negative filesystem observation."""
    try:
        path.stat()
    except FileNotFoundError:
        return False
    return True


def preferred_file(source: str, path: Path) -> Path:
    if source != "codex" or not path.name.endswith((".jsonl", ".jsonl.zst")):
        return path
    plain = path.with_suffix("") if path.name.endswith(".zst") else path
    if file_exists(plain):
        return plain
    packed = plain.with_name(plain.name + ".zst")
    return packed if file_exists(packed) else path


def representation_paths(canonical: Path, locator: SessionSource) -> frozenset[Path]:
    paths = {canonical, canonical.with_name(canonical.name + ".zst"), locator.physical_path}
    if canonical.suffix == ".zst":
        paths.add(canonical.with_suffix(""))
    return frozenset(paths)


def _encoding(path: Path, fallback: Encoding) -> Encoding:
    try:
        with path.open("rb") as source:
            header = source.read(4)
    except FileNotFoundError:
        return fallback
    magic = int.from_bytes(header, "little")
    if magic == 0xFD2FB528 or 0x184D2A50 <= magic <= 0x184D2A5F:
        return "zstd"
    if header and (header.lstrip().startswith((b"{", b"[")) or header.isspace()):
        return "jsonl"
    return fallback


def matches_session_file(source: str, name: str, pattern: str) -> bool:
    if source == "codex":
        return name.startswith("rollout") and name.endswith((".jsonl", ".jsonl.zst"))
    return fnmatch.fnmatchcase(name, pattern)
