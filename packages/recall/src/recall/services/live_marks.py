"""The writer-pid mark a harness hook writes for a session (REQ-LIVE-008).

Enrichment, and only enrichment: recall never infers a mark from a process
tree, and a session without one derives exactly what it derived before. What a
mark adds is the single fact no transcript can carry: the process that was
writing it is gone.

The `live_marks.surface_key` column is no longer written or read; it stays
until the next schema migration drops it.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import duckdb

logger = logging.getLogger("recall.live_marks")


def upsert_live_mark(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str,
    source_session_id: str,
    host: str,
    pid: int | None,
    marked_at: datetime,
) -> None:
    """Record the mark for one session, replacing any earlier one.

    A `SessionStart` hook fires again every time a harness resumes into the same
    transcript, and each firing carries a *new* pid. The mark is the current
    fact about the session, not a log of them, so the latest write wins.
    """
    conn.execute(
        """
        INSERT INTO live_marks (source, source_session_id, host, pid, marked_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (source, source_session_id, host) DO UPDATE SET
            pid = excluded.pid,
            marked_at = excluded.marked_at
        """,
        [source, source_session_id, host, pid, marked_at],
    )


def pid_alive(pid: int) -> bool:
    """Whether a process with this id exists on this machine.

    `kill(pid, 0)` asks the kernel without delivering anything. `EPERM` means
    the process exists and belongs to someone else, which is still alive — a
    harness running as another user is still a live writer.

    Non-positive ids are refused rather than probed: `kill(0, 0)` signals the
    caller's whole process group and `kill(-1, 0)` every process the caller may
    signal, so both would read as "alive" for a mark that names no process.

    A pid the kernel has recycled onto an unrelated process reads as alive.
    Nothing cheap distinguishes the two, and the failure is the safe direction:
    a session stays `idle` instead of being wrongly declared `ended`.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as err:
        logger.debug("pid probe failed for %s: %s", pid, err)
        return False
    return True
