"""Cooperating daemon and offline-reader ownership of one database pathname."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path

import duckdb

from recall.core.config import AppConfig, create_private_dir
from recall.db.connection import _apply_runtime_pragmas

logger = logging.getLogger("recall.ownership")
STARTUP_OWNERSHIP_TIMEOUT_SECONDS = 5.0
_OWNERSHIP_POLL_SECONDS = 0.05


class DatabaseOwnershipError(RuntimeError):
    """Ownership could not be acquired without disturbing another owner."""


@dataclass(frozen=True)
class SnapshotUnavailable:
    reason: str


def ownership_path(db_path: Path) -> Path:
    """Aliases through symlinks and relative paths select the same stable inode.

    Hard-linked aliases and noncooperating binaries are outside this protocol.
    The gate must never be unlinked, including on shutdown or owner death.
    """
    return Path(f"{db_path.expanduser().resolve()}.ownership")


@contextmanager
def _gate(db_path: Path) -> Iterator[tuple[Path, int]]:
    path = db_path
    try:
        path = ownership_path(db_path)
        create_private_dir(path.parent)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    except OSError as err:
        raise DatabaseOwnershipError(
            f"cannot open database ownership gate {path} for {db_path}: {err}; "
            "check the path and directory permissions, then retry"
        ) from err
    try:
        yield path, descriptor
    finally:
        # Closing also releases flock after exceptions and cancellation. Never
        # unlink: a waiter may already have opened this same inode.
        os.close(descriptor)


def _try_acquire(path: Path, descriptor: int) -> bool:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    except OSError as err:
        raise DatabaseOwnershipError(f"cannot lock database ownership gate {path}: {err}") from err
    return True


@asynccontextmanager
async def daemon_ownership(db_path: Path) -> AsyncIterator[None]:
    """Wait asynchronously for a bounded interval, then own until all users drain."""
    with _gate(db_path) as (path, descriptor):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + STARTUP_OWNERSHIP_TIMEOUT_SECONDS
        waiting = False
        while not _try_acquire(path, descriptor):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise DatabaseOwnershipError(
                    f"database ownership busy for {db_path} (gate {path}); "
                    f"waited {STARTUP_OWNERSHIP_TIMEOUT_SECONDS:g} seconds; "
                    "another daemon or offline status snapshot owns it; retry after it exits"
                )
            if not waiting:
                logger.info("waiting for database ownership: database=%s gate=%s", db_path, path)
                waiting = True
            await asyncio.sleep(min(_OWNERSHIP_POLL_SECONDS, remaining))
        yield


def read_offline[T](
    config: AppConfig, read: Callable[[duckdb.DuckDBPyConnection], T]
) -> T | SnapshotUnavailable:
    """Read one existing database under nonblocking ownership; no handle escapes.

    The callback returns detached data. This is the only local status opener;
    it never creates a DuckDB file or installs/migrates its schema.
    """
    try:
        with _gate(config.db_path) as (path, descriptor):
            if not _try_acquire(path, descriptor):
                return SnapshotUnavailable(
                    f"database ownership busy at {path}; daemon startup, runtime or teardown, "
                    "or another offline snapshot owns it; runtime fields unavailable; "
                    "retry after the current owner finishes"
                )
            if not config.db_path.exists():
                return SnapshotUnavailable(f"database does not exist: {config.db_path}")
            conn = duckdb.connect(str(config.db_path), read_only=True)
            try:
                _apply_runtime_pragmas(conn, config)
                return read(conn)
            finally:
                conn.close()
    except DatabaseOwnershipError as err:
        return SnapshotUnavailable(str(err))
