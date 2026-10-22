"""Validate filesystem Unix socket addresses before startup creates state."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def validate_socket_path(path: Path) -> None:
    # sockaddr_un.sun_path includes a terminating NUL byte.
    capacity = 104 if sys.platform == "darwin" else 108
    length = len(os.fsencode(path))
    if length >= capacity:
        raise ValueError(
            f"AF_UNIX socket path is {length} bytes; the limit is {capacity - 1} bytes "
            f"({capacity} including the terminating NUL): {path}. "
            "Choose a shorter RECALL_DATA_DIR."
        )
