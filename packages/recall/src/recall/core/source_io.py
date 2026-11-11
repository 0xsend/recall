"""Regular-file admission for transcript and sibling-file content readers."""

from __future__ import annotations

import errno
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from stat import S_ISREG
from typing import BinaryIO


def require_regular_source(path: Path, status: os.stat_result) -> None:
    """Refuse special files through the ordinary unreadable-source IO channel."""
    if not S_ISREG(status.st_mode):
        raise OSError(errno.EINVAL, "source is not a regular file", str(path))


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


def read_source_text(path: Path) -> str:
    """Read UTF-8 sibling metadata through the same source admission boundary."""
    with open_source(path) as stream:
        return stream.read().decode("utf-8")
