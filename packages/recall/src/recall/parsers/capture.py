"""Finite JSONL reads whose digests describe the bytes actually parsed."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Self

from recall.core.models import ParseDiagnostic

HASH_BUFFER_BYTES = 1024 * 1024
MAX_RECORD_BYTES = 64 * 1024 * 1024


class JsonlCapture:
    """Read only the prefix present when opened, retaining an exact checkpoint.

    The source path is never replaced with a temporary filename, so adapters
    retain their normal session identity and sidecar lookup. A caller verifies
    the captured digest against the source before publishing derived rows.
    """

    def __init__(
        self, path: Path, *, offset: int = 0, max_record_bytes: int = MAX_RECORD_BYTES
    ) -> None:
        if offset < 0 or not 0 < max_record_bytes <= MAX_RECORD_BYTES:
            raise ValueError("offset must be non-negative and record limit within 1..64 MiB")
        self.path = path
        self.offset = offset
        self.max_record_bytes = max_record_bytes
        self.next_byte_offset = 0
        self.captured_size = 0
        self.record_start = 0
        self.diagnostics: list[ParseDiagnostic] = []
        self._digest = hashlib.sha256()
        self._committed_digest = hashlib.sha256()
        self._initial_prefix_sha256: str | None = None
        self.source_dev: int | None = None
        self.source_inode: int | None = None
        self._handle: BinaryIO | None = None
        self._end = 0
        self._iterated = False
        self._checkpoint_blocked = False

    @property
    def captured_prefix_sha256(self) -> str:
        return self._digest.hexdigest()

    @property
    def committed_prefix_sha256(self) -> str:
        return self._committed_digest.hexdigest()

    @property
    def initial_prefix_sha256(self) -> str:
        """Digest of the bytes before the requested parse offset on this handle."""
        if self._initial_prefix_sha256 is None:
            raise RuntimeError("capture must be opened before reading its initial prefix digest")
        return self._initial_prefix_sha256

    def __enter__(self) -> Self:
        if self._handle is not None or self._iterated:
            raise RuntimeError("a capture can be opened only once")
        handle = self.path.open("rb")
        try:
            source_stat = os.fstat(handle.fileno())
            self._end = source_stat.st_size
            self.source_dev = source_stat.st_dev
            self.source_inode = source_stat.st_ino
            if self.offset > self._end:
                raise ValueError("parse offset exceeds captured source size")
            while self.captured_size < self.offset:
                chunk = handle.read(min(HASH_BUFFER_BYTES, self.offset - self.captured_size))
                if not chunk:
                    raise OSError("source shrank while capturing checkpoint prefix")
                self._digest.update(chunk)
                self.captured_size += len(chunk)
            self._committed_digest = self._digest.copy()
            self._initial_prefix_sha256 = self._digest.hexdigest()
            self.next_byte_offset = self.offset
        except BaseException:
            handle.close()
            raise
        self._handle = handle
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self._handle is not None
        self._handle.close()
        self._handle = None

    def records(self) -> Iterator[dict[str, Any]]:
        """Yield decoded objects; a partial or invalid record stops the prefix."""
        if self._handle is None or self._iterated:
            raise RuntimeError("capture records require a single open reader")
        self._iterated = True
        while self.captured_size < self._end:
            self.record_start = self.captured_size
            remaining = self._end - self.captured_size
            raw = self._handle.readline(min(remaining, self.max_record_bytes + 1))
            if not raw:
                self.diagnostics.append(
                    ParseDiagnostic("source_changed", self.record_start, "source shortened")
                )
                break
            self._digest.update(raw)
            self.captured_size += len(raw)
            if len(raw) > self.max_record_bytes:
                self.diagnostics.append(
                    ParseDiagnostic(
                        "resource_limit", self.record_start, "record exceeds byte limit"
                    )
                )
                break
            if not raw.endswith(b"\n"):
                self.diagnostics.append(
                    ParseDiagnostic("unterminated_tail", self.record_start, "no complete newline")
                )
                break
            try:
                line = raw.decode("utf-8").strip()
                entry = json.loads(line) if line else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.diagnostics.append(
                    ParseDiagnostic("malformed_record", self.record_start, "invalid UTF-8 or JSON")
                )
                break
            if line and not isinstance(entry, dict):
                self.diagnostics.append(
                    ParseDiagnostic("malformed_record", self.record_start, "expected JSON object")
                )
                break
            previous_offset = self.next_byte_offset
            previous_digest = self._committed_digest.copy()
            if not self._checkpoint_blocked:
                self.next_byte_offset = self.captured_size
                self._committed_digest = self._digest.copy()
            if entry is not None:
                diagnostic_count = len(self.diagnostics)
                yield entry
                # Adapters can diagnose an unsupported semantic record through
                # this shared list; later records cannot acknowledge past it.
                if len(self.diagnostics) > diagnostic_count:
                    self.next_byte_offset = previous_offset
                    self._committed_digest = previous_digest
                    self._checkpoint_blocked = True
