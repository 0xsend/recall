"""Logical transcript names and their selected on-disk representations."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

CODEX_PATTERN = "rollout*.jsonl"
MAX_COMPRESSED_CAPTURE_BYTES = 96 * 1024 * 1024


def transcript_name_matches(name: str, pattern: str) -> bool:
    if pattern == CODEX_PATTERN:
        name = name.removesuffix(".zst")
    return fnmatch.fnmatchcase(name, pattern)


def transcript_identity(path: Path) -> Path:
    """Resolve aliases and retain the plain name across Codex compression."""
    physical = Path(os.path.realpath(transcript_file(path.expanduser()), strict=True))
    plain = plain_transcript_path(physical)
    try:
        return Path(os.path.realpath(plain, strict=True))
    except FileNotFoundError:
        # An absent or dangling plain sibling must not redirect compressed bytes.
        return plain


def plain_transcript_path(path: Path) -> Path:
    """The lexical identity, for directory walkers that already resolved aliases."""
    if path.name.endswith(".jsonl.zst") and transcript_name_matches(path.name, CODEX_PATTERN):
        return path.with_suffix("")
    return path


def transcript_file(path: Path) -> Path:
    """Select plain first; a missing Codex plain path may have a Zstd sibling.

    Only Codex rollout names admit a compressed representation. IO errors
    other than absence propagate rather than silently selecting older bytes.
    """
    if not transcript_name_matches(path.name, CODEX_PATTERN):
        return path
    plain = plain_transcript_path(path)
    try:
        plain.stat()
    except FileNotFoundError:
        return plain.with_suffix(".jsonl.zst")
    return plain


def transcript_path_key(path: str | Path) -> str:
    """Canonical channel spelling, retaining Codex's plain name across compression.

    For accessible files this matches `sessions.source_path`, because this
    is both the channel key and the key the index is queried by. The watcher
    sees the path the harness writes, a caller reads the path the index stored,
    and on a host whose `$HOME` or `~/.claude` is a symlink — or on macOS,
    where `/tmp` is one — those are different strings for the same file.
    Without one spelling, `--fresh` waits out its whole timeout on a session
    that was just indexed.

    An inaccessible representation still needs a key for notification and
    retry settlement. Best-effort alias resolution retains its plain spelling;
    content capture independently requires successful physical selection and
    reports IO failures. A channel key never establishes source currentness.
    """
    expanded = Path(path).expanduser()
    try:
        return str(transcript_identity(expanded))
    except OSError:
        return os.path.realpath(plain_transcript_path(expanded))
