"""Regular-file admission for transcript and sibling-file content readers."""

from __future__ import annotations

import errno
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import BinaryIO


class NonRegularSourceError(OSError):
    """An existing path or opened descriptor has the wrong file kind."""


def require_regular_source(path: Path, status: os.stat_result) -> None:
    """Refuse special files through the ordinary unreadable-source IO channel."""
    if not S_ISREG(status.st_mode):
        raise NonRegularSourceError(errno.EINVAL, "source is not a regular file", str(path))


@contextmanager
def open_source(path: Path) -> Iterator[BinaryIO]:
    """Open only a regular source, without waiting for a FIFO writer.

    Follow supported source aliases. The path check avoids opening known
    devices; the descriptor check refuses replacements between stat and open.
    O_NONBLOCK bounds FIFO open, not IO on a stalled regular/network file.
    """
    require_regular_source(path, path.stat())
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        require_regular_source(path, os.fstat(descriptor))
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        yield stream


class SidecarRefusedError(RuntimeError):
    """A sibling exists but cannot supply metadata; it is never absent metadata."""


@dataclass(frozen=True)
class SidecarText:
    text: str | None


@dataclass(frozen=True)
class RefusedSidecar:
    error: OSError

    @property
    def text(self) -> str:
        raise SidecarRefusedError(str(self.error)) from self.error


def read_sidecar(path: Path) -> SidecarText | RefusedSidecar:
    """Separate absent metadata from a refused read before best-effort decoding.

    Accessing refused text propagates outside the OSError/JSON decoding channel,
    so parsers cannot turn a refused sibling into an absent metadata value.
    """
    try:
        with open_source(path) as stream:
            return SidecarText(stream.read().decode("utf-8"))
    except FileNotFoundError:
        return SidecarText(None)
    except OSError as error:
        return RefusedSidecar(error)
