"""Finite, fully validated Zstandard captures for cold Codex rollouts."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

from recall.core.transcripts import MAX_COMPRESSED_CAPTURE_BYTES
from recall.parsers.capture import HASH_BUFFER_BYTES, JsonlCapture


class CompressedCapture(JsonlCapture):
    """A compressed file is acknowledged only as one complete physical capture.

    Decoding uses at most 1 MiB input/output blocks and a 64 MiB Zstd window.
    Input and decoded JSONL each have a 96 MiB ceiling; the temporary decoded
    file closes on every exit. Physical digests and offsets describe the bytes
    on disk, never the temporary name or decompressed byte positions.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._physical_digest = hashlib.sha256()
        self._physical_stat: os.stat_result | None = None

    @contextmanager
    def _open(self) -> Iterator[BinaryIO]:
        with tempfile.TemporaryFile() as decoded:
            with self.path.open("rb") as source:
                self._physical_stat = os.fstat(source.fileno())
                remaining = self._physical_stat.st_size
                if not 0 < remaining <= MAX_COMPRESSED_CAPTURE_BYTES:
                    raise ValueError("compressed rollout input exceeds 96 MiB or is empty")
                decoder = zstd.ZstdDecompressor(
                    options={zstd.DecompressionParameter.window_log_max: 26}
                )
                total = 0
                frames = 0
                while remaining:
                    chunk = source.read(min(HASH_BUFFER_BYTES, remaining))
                    if not chunk:
                        raise OSError("compressed rollout shortened during capture")
                    remaining -= len(chunk)
                    self._physical_digest.update(chunk)
                    while chunk or not decoder.needs_input:
                        output = decoder.decompress(chunk, max_length=HASH_BUFFER_BYTES)
                        total += len(output)
                        if total > MAX_COMPRESSED_CAPTURE_BYTES:
                            raise ValueError("compressed rollout decoded content exceeds 96 MiB")
                        decoded.write(output)
                        chunk = b""
                        if decoder.eof:
                            frames += 1
                            if frames > 4096:
                                raise ValueError("compressed rollout exceeds 4096 frames")
                            chunk = decoder.unused_data
                            if chunk or remaining:
                                decoder = zstd.ZstdDecompressor(
                                    options={zstd.DecompressionParameter.window_log_max: 26}
                                )
                            else:
                                break
                if not decoder.eof:
                    raise ValueError("compressed rollout ends in an incomplete Zstandard frame")
            decoded.seek(0)
            yield decoded

    def __enter__(self) -> Self:
        super().__enter__()
        assert self._physical_stat is not None
        self.source_dev = self._physical_stat.st_dev
        self.source_inode = self._physical_stat.st_ino
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        super().__exit__(exc_type, exc, traceback)
        if exc_type is not None:
            return
        if self.captured_size != self._end or any(
            diagnostic.kind != "unsupported_record" for diagnostic in self.diagnostics
        ):
            kinds = ", ".join(dict.fromkeys(item.kind for item in self.diagnostics))
            raise ValueError(f"compressed rollout JSONL is incomplete: {kinds}")
        assert self._physical_stat is not None
        self.captured_size = self._physical_stat.st_size
        self.next_byte_offset = self.captured_size
        self._digest = self._physical_digest
        self._committed_digest = self._physical_digest.copy()
