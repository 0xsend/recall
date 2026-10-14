"""Bounded Zstandard JSONL capture with whole-file physical acknowledgment."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from recall.core.models import ParseDiagnostic
from recall.parsers.capture import HASH_BUFFER_BYTES, MAX_RECORD_BYTES, JsonlCapture

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

MAX_DECODED_BYTES = 512 * 1024 * 1024
MAX_WINDOW_LOG = 27  # 128 MiB


class CompressedJsonlCapture(JsonlCapture):
    def __init__(
        self,
        path: Path,
        *,
        offset: int = 0,
        max_record_bytes: int = MAX_RECORD_BYTES,
        physical_limit: int | None = None,
    ) -> None:
        super().__init__(path, offset=offset, max_record_bytes=max_record_bytes)
        self.physical_limit = physical_limit
        self.decoded_size = 0
        self._decoded_digest = hashlib.sha256()

    @property
    def decoded_sha256(self) -> str:
        return self._decoded_digest.hexdigest()

    def decoded_chunks(self) -> Iterator[bytes]:
        return self._decoded_chunks()

    @property
    def incremental(self) -> bool:
        return False

    def records(self) -> Iterator[dict[str, Any]]:
        if self.offset:
            raise ValueError("compressed captures cannot resume")
        yield from super().records()
        if not self.diagnostics:
            self.next_byte_offset = self.captured_size
            self._committed_digest = self._digest.copy()

    def _lines(self) -> Iterator[bytes]:
        raw = bytearray()
        decoded_offset = 0
        for block in self._decoded_chunks():
            start = 0
            while start < len(block):
                newline = block.find(b"\n", start)
                end = len(block) if newline < 0 else newline + 1
                if len(raw) + end - start > self.max_record_bytes:
                    self.diagnostics.append(
                        ParseDiagnostic(
                            "resource_limit", decoded_offset, "record exceeds byte limit"
                        )
                    )
                    return
                raw.extend(block[start:end])
                start = end
                if newline >= 0:
                    self.record_start = decoded_offset
                    decoded_offset += len(raw)
                    yield bytes(raw)
                    raw.clear()
        if raw and not self.diagnostics:
            self.record_start = decoded_offset
            yield bytes(raw)

    def _decoded_chunks(self) -> Iterator[bytes]:
        assert self._handle is not None
        decoder = zstd.ZstdDecompressor(
            options={zstd.DecompressionParameter.window_log_max: MAX_WINDOW_LOG}
        )
        decoded_size = 0
        end = min(self._end, self.physical_limit) if self.physical_limit is not None else self._end
        frame_complete = False
        data = b""
        try:
            while True:
                if decoder.needs_input and not data:
                    remaining = end - self.captured_size
                    if not remaining:
                        if not frame_complete:
                            self.diagnostics.append(
                                ParseDiagnostic(
                                    "malformed_record", decoded_size, "truncated Zstandard frame"
                                )
                            )
                        return
                    data = self._handle.read(min(HASH_BUFFER_BYTES, remaining))
                    if not data:
                        self.diagnostics.append(
                            ParseDiagnostic("source_changed", decoded_size, "source shortened")
                        )
                        return
                    self._digest.update(data)
                    self.captured_size += len(data)
                frame_complete = False
                block = decoder.decompress(
                    data, max_length=min(HASH_BUFFER_BYTES, MAX_DECODED_BYTES - decoded_size + 1)
                )
                data = b""
                decoded_size += len(block)
                if decoded_size > MAX_DECODED_BYTES:
                    self.diagnostics.append(
                        ParseDiagnostic(
                            "resource_limit", decoded_size, "decoded rollout exceeds 512 MiB"
                        )
                    )
                    return
                if block:
                    self.decoded_size = decoded_size
                    self._decoded_digest.update(block)
                    yield block
                if decoder.eof:
                    data = decoder.unused_data
                    frame_complete = True
                    decoder = zstd.ZstdDecompressor(
                        options={zstd.DecompressionParameter.window_log_max: MAX_WINDOW_LOG}
                    )
        except zstd.ZstdError as error:
            self.diagnostics.append(
                ParseDiagnostic(
                    "malformed_record", decoded_size, f"invalid Zstandard frame: {error}"
                )
            )
