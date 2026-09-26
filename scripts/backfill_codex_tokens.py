#!/usr/bin/env python3
"""One-time repair: backfill Codex session token totals from source files.

Codex sessions indexed before the ``token_count`` parser fix (REQ-PARSE-011)
carry ``NULL`` ``input_tokens``/``output_tokens`` in
``session_state`` because the old parser hard-coded both to ``None``. The daemon
indexes incrementally — it only reparses files whose mtime/size/offset changed —
so historical Codex sessions never pick up tokens without a full reindex. A full
reindex is the wrong tool here: ``recall index --full`` reparses messages
(delete+reinsert), which would drop the embeddings and context summaries on every
Codex message.

This script does the surgical thing instead: it re-reads each affected rollout
file with the *current* ``CodexParser`` and writes ONLY the two token columns.

Why it is safe:
  * Tokens live in ``session_state``, which nothing FK-references (``messages`` /
    ``tool_calls`` reference ``sessions(id)``, the identity table). A targeted
    ``UPDATE session_state`` touches no other rows — no FK churn, no embedding or
    context loss.
  * Codex is an ``ABSOLUTE_TOKEN_SOURCE`` (cumulative totals), so the canonical
    merge — mirrored from ``indexer.py`` — is ``GREATEST(existing, parsed)``. That
    makes this idempotent and safe to re-run.

The recall daemon holds the DuckDB write lock for its whole lifetime, so this
script must run with the daemon stopped:

    recall daemon stop
    uv run python scripts/backfill_codex_tokens.py --dry-run   # preview
    uv run python scripts/backfill_codex_tokens.py             # apply
    recall daemon start

New Codex sessions indexed after that fix already get tokens automatically;
this only backfills the pre-fix backlog.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
from recall.core.config import AppConfig
from recall.core.models import Session
from recall.core.types import Source
from recall.db.connection import RecallLockError, advisory_lock, connect
from recall.parsers import CodexParser

logger = logging.getLogger("recall.backfill_codex_tokens")

# A single planned write: (input_tokens, output_tokens, session_id). Either token
# may be None, in which case _UPDATE_SQL leaves that column untouched.
TokenUpdate = tuple[int | None, int | None, str]

# Sessions that still need a backfill: Codex source with at least one NULL token
# column. Joined to session_state because that is where the mutable token totals
# live (sessions holds only immutable identity).
_SELECT_TARGETS_SQL = """
    SELECT s.id, s.source_path
    FROM sessions s
    JOIN session_state ss ON ss.session_id = s.id
    WHERE s.source = ?
      AND (ss.input_tokens IS NULL OR ss.output_tokens IS NULL)
    ORDER BY s.source_path
"""

# Mirrors the absolute-source branch of indexer.py's incremental merge: Codex
# totals are cumulative, so GREATEST(existing, parsed) — never additive. A column
# whose freshly parsed value is NULL is left untouched so a file with no
# token_count events can never clobber an existing value with 0.
_UPDATE_SQL = """
    UPDATE session_state
    SET
        input_tokens = CASE
            WHEN ? IS NULL THEN input_tokens
            ELSE GREATEST(COALESCE(input_tokens, 0), ?)
        END,
        output_tokens = CASE
            WHEN ? IS NULL THEN output_tokens
            ELSE GREATEST(COALESCE(output_tokens, 0), ?)
        END
    WHERE session_id = ?
"""

_DAEMON_RUNNING_HINT = (
    "could not open the recall database read-write — the daemon almost certainly "
    "holds the lock.\nStop it first, then re-run:\n"
    "    recall daemon stop\n"
    "    uv run python scripts/backfill_codex_tokens.py\n"
    "    recall daemon start"
)


@dataclass(frozen=True)
class Target:
    """A Codex session row that is missing token totals."""

    session_id: str
    source_path: str


@dataclass
class Stats:
    """Tally of what the backfill touched. ``updated`` is also the dry-run count."""

    scanned: int = 0
    updated: int = 0
    no_file: int = 0  # source file deleted/moved since indexing
    no_tokens: int = 0  # file parsed but carried no token_count events
    parse_errors: int = 0
    id_mismatch: int = 0  # parsed session id != stored id (path drift) — skipped


def _parse_session(path: Path) -> Session:
    """Re-derive session metadata (incl. tokens) from a Codex rollout file.

    Delegates to the real ``CodexParser`` so token extraction can never drift
    from how the indexer computes it.
    """
    return CodexParser().parse(path).session


def _plan_updates(
    conn: duckdb.DuckDBPyConnection, *, limit: int | None, stats: Stats
) -> list[TokenUpdate]:
    """Parse every target file and build the (input, output, session_id) writes.

    Parsing (slow, I/O-bound) is fully separated from writing (a single fast
    transaction) so the write lock is held for as short a window as possible.
    """
    rows = conn.execute(_SELECT_TARGETS_SQL, [Source.CODEX.value]).fetchall()
    targets = [Target(session_id=str(row[0]), source_path=str(row[1])) for row in rows]
    if limit is not None:
        targets = targets[:limit]
    logger.info("codex sessions missing tokens: %d", len(targets))

    updates: list[TokenUpdate] = []
    for target in targets:
        stats.scanned += 1
        path = Path(target.source_path)
        if not path.exists():
            stats.no_file += 1
            logger.debug("skip (source file gone): %s", target.source_path)
            continue
        try:
            session = _parse_session(path)
        except Exception:
            # A single corrupt/partial rollout must not abort the whole repair.
            stats.parse_errors += 1
            logger.warning("parse failed, skipping: %s", target.source_path, exc_info=True)
            continue
        # Defensive: the parser recomputes the session id from source+path. If it
        # disagrees with the stored id we must NOT write parsed tokens to the
        # wrong row — skip and surface it instead.
        if session.id != target.session_id:
            stats.id_mismatch += 1
            logger.warning(
                "session id mismatch (stored=%s parsed=%s) for %s — skipping",
                target.session_id,
                session.id,
                target.source_path,
            )
            continue
        if session.input_tokens is None and session.output_tokens is None:
            stats.no_tokens += 1
            continue
        updates.append((session.input_tokens, session.output_tokens, target.session_id))
    return updates


def backfill(*, dry_run: bool = False, limit: int | None = None) -> Stats:
    """Backfill Codex token totals on the configured database. Returns a tally.

    Raises ``SystemExit`` with actionable guidance if the database is locked by a
    running daemon.
    """
    config = AppConfig.load()
    stats = Stats()
    try:
        with advisory_lock(config.lock_path):
            try:
                conn = connect(config)
            except duckdb.Error as err:
                if "lock" in str(err).lower():
                    raise SystemExit(_DAEMON_RUNNING_HINT) from err
                raise
            try:
                updates = _plan_updates(conn, limit=limit, stats=stats)
                if dry_run:
                    stats.updated = len(updates)
                    logger.info("[dry-run] would update %d sessions", stats.updated)
                    return stats
                conn.execute("BEGIN")
                try:
                    for input_tokens, output_tokens, session_id in updates:
                        conn.execute(
                            _UPDATE_SQL,
                            [input_tokens, input_tokens, output_tokens, output_tokens, session_id],
                        )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                stats.updated = len(updates)
                return stats
            finally:
                conn.close()
    except RecallLockError as err:
        raise SystemExit(_DAEMON_RUNNING_HINT) from err


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill Codex session token totals without a full reindex.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many sessions would be updated without writing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N sessions (useful for a smoke test).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    stats = backfill(dry_run=args.dry_run, limit=args.limit)
    verb = "would update" if args.dry_run else "updated"
    logger.info(
        "done: scanned=%d %s=%d  skipped(no_file=%d no_tokens=%d parse_errors=%d id_mismatch=%d)",
        stats.scanned,
        verb,
        stats.updated,
        stats.no_file,
        stats.no_tokens,
        stats.parse_errors,
        stats.id_mismatch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
