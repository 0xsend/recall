"""Fingerprint the normalization code that owns durable parser checkpoints."""

from __future__ import annotations

import hashlib
import inspect
from functools import cache
from pathlib import Path


@cache
def _normalization_paths() -> tuple[Path, ...]:
    """Include the complete core/parser boundary, including newly added modules.

    Parsers depend only on core and their own package. Fingerprinting that
    boundary makes adding a normalization dependency invalidate checkpoints
    without maintaining a second list of the modules imports already name.
    Unrelated core/parser edits may also decline resume; a full reference
    parse is safer than combining rows normalized by different code.
    """
    package = Path(__file__).resolve().parent.parent
    return tuple(
        path for name in ("core", "parsers") for path in sorted((package / name).rglob("*.py"))
    )


@cache
def parser_revision(parser_type: type) -> str:
    """Hash the adapter and the code boundary containing its normalization."""
    adapter = Path(inspect.getfile(parser_type))
    digest = hashlib.sha256(adapter.read_bytes())
    for path in _normalization_paths():
        digest.update(path.read_bytes())
    return digest.hexdigest()
