"""Read-only inventory of historical move candidates and repair effects.

Timestamp and declared-id groups are lookup candidates, never identity proof.
Only the move planner decides which rows can be superseded.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Final

import duckdb

from recall.core.types import UNATTRIBUTED_HOST
from recall.services.moves import (
    CARRIED_CONTEXTS_SELECT,
    CARRIED_MESSAGE_VECTORS_SELECT,
    CARRIED_TOOL_VECTORS_SELECT,
    MovedDuplicate,
    _groups,
    _has_departed,
    _local_rows,
    _MoveGroup,
    _Row,
    _survivor_proof,
    plan_moved_duplicates,
)

# An explicit error, never a truncated inventory presented as complete.
_CENSUS_SESSIONS_MAX: Final = 1_000_000


@dataclass(frozen=True)
class MoveCensus:
    total_sessions: int
    host: str
    sessions: int
    timestamp_less_sessions: int
    duplicate_key_groups: int
    excess_rows: int
    classes: dict[str, int]
    groups: tuple[dict[str, Any], ...]
    changed_timestamp_candidates: tuple[dict[str, Any], ...]
    pairs: tuple[dict[str, Any], ...]

    @property
    def excluded_sessions(self) -> int:
        return self.total_sessions - self.sessions

    def payload(self) -> dict[str, Any]:
        values = asdict(self)
        host = values.pop("host")
        return {
            "version": 1,
            **values,
            "excluded_sessions": self.excluded_sessions,
            "host_filter": {"hosts": [host, UNATTRIBUTED_HOST], "include_null": True},
        }


def census_moved_duplicates(
    conn: duckdb.DuckDBPyConnection, *, host: str
) -> tuple[list[MovedDuplicate], MoveCensus]:
    """Inventory the local candidate namespace; count overlapping classes per group.

    A repeated declared id at different timestamps is reported separately. It
    cannot prove a timestamp changed, particularly for forks and subagents.
    """
    count = conn.execute("SELECT count(*) FROM sessions").fetchone()
    assert count is not None
    if int(count[0]) > _CENSUS_SESSIONS_MAX:
        raise ValueError(f"census supports at most {_CENSUS_SESSIONS_MAX} sessions")
    plans = plan_moved_duplicates(conn, host=host)
    rows = _local_rows(conn, host=host)
    keys: Counter[tuple[object, ...]] = Counter()
    identities: dict[tuple[object, ...], list[_Row]] = defaultdict(list)
    for row in rows:
        keys[(row.host, row.source, row.source_session_id, row.started_at)] += 1
        if row.source_session_id is not None:
            identities[(row.host, row.source, row.source_session_id)].append(row)
    groups = [_describe_group(group) for group in _groups(rows).values() if len(group) > 1]
    changed = tuple(
        {"class": "changed_timestamp_candidates", "rows": [_describe_row(row) for row in group]}
        for group in identities.values()
        if len({row.started_at for row in group}) > 1
    )
    classes: Counter[str] = Counter()
    for group in groups:
        classes.update(group["classes"])
    return plans, MoveCensus(
        total_sessions=int(count[0]),
        host=host,
        sessions=len(rows),
        timestamp_less_sessions=sum(row.started_at is None for row in rows),
        duplicate_key_groups=sum(n > 1 for n in keys.values()),
        excess_rows=sum(n - 1 for n in keys.values() if n > 1),
        classes=dict(sorted(classes.items())),
        groups=tuple(groups),
        changed_timestamp_candidates=changed,
        pairs=tuple(_describe_pair(conn, plan, rows) for plan in plans),
    )


def _describe_row(row: _Row) -> dict[str, Any]:
    return {
        **asdict(row),
        "started_at": row.started_at.isoformat() if row.started_at is not None else None,
        "departed": _has_departed(row.source_path),
    }


def _describe_group(rows: list[_Row]) -> dict[str, Any]:
    classes: set[str] = set()
    if rows[0].started_at is None:
        classes.add("missing_timestamp")
    departed = [row for row in rows if _has_departed(row.source_path)]
    survivors = [row for row in rows if not _has_departed(row.source_path)]
    group = _MoveGroup.of(rows)
    if not departed:
        classes.add("surviving_copies")
    elif not survivors:
        classes.add("no_survivor")
    elif group is None:
        classes.add("candidate_overflow")
    else:
        for predecessor in group.departed:
            proofs = [_survivor_proof(survivor, predecessor) for survivor in group.survivors]
            holders = proofs.count(("proven_move",))
            if holders == 1 and group.sole_holder(predecessor) is None:
                classes.add("ambiguous_predecessors")
            elif holders == 1:
                classes.add("proven_move")
            elif holders > 1:
                classes.add("multiple_proven_survivors")
            else:
                classes.update(reason for proof in proofs for reason in proof)
    return {
        "source": rows[0].source,
        "started_at": rows[0].started_at.isoformat() if rows[0].started_at is not None else None,
        "classes": sorted(classes),
        "rows": [_describe_row(row) for row in rows],
    }


def _describe_pair(
    conn: duckdb.DuckDBPyConnection, plan: MovedDuplicate, rows: list[_Row]
) -> dict[str, Any]:
    by_id = {row.session_id: row for row in rows}
    predecessor = plan.predecessor_id
    counts: dict[str, int] = {}
    # These are selectors for the maintenance operation's documented effects,
    # not a second implementation of the deletion procedure.
    selectors = {
        "sessions": "id = ?",
        "session_state": "session_id = ?",
        "messages": "session_id = ?",
        "message_state": "message_id IN (SELECT id FROM messages WHERE session_id = ?)",
        "message_embeddings": "message_id IN (SELECT id FROM messages WHERE session_id = ?)",
        "tool_calls": "session_id = ?",
        "tool_call_embeddings": "tool_call_id IN (SELECT id FROM tool_calls WHERE session_id = ?)",
        "tool_use_ids": "session_id = ?",
        "tool_results": (
            "tool_call_id IN (SELECT tool_call_id FROM tool_use_ids WHERE session_id = ?)"
        ),
        "session_stop_markers": "session_id = ?",
        "usage_events": "session_id = ?",
        "source_files": "session_id = ?",
    }
    for table, predicate in selectors.items():
        count = conn.execute(
            f"SELECT count(*) FROM {table} WHERE {predicate}", [predecessor]
        ).fetchone()
        assert count is not None
        counts[table] = int(count[0])
    contexts = conn.execute(CARRIED_CONTEXTS_SELECT, [plan.successor_id, predecessor]).fetchall()
    messages = conn.execute(
        "SELECT successor_message FROM (" + CARRIED_MESSAGE_VECTORS_SELECT + ")",
        [plan.successor_id, predecessor],
    ).fetchall()
    tools = conn.execute(
        "SELECT successor_tool_call FROM (" + CARRIED_TOOL_VECTORS_SELECT + ")",
        [plan.successor_id, predecessor],
    ).fetchall()
    return {
        **asdict(plan),
        "predecessor": _describe_row(by_id[predecessor]),
        "successor": _describe_row(by_id[plan.successor_id]),
        "predecessor_rows_affected": counts,
        "successor_context_message_ids": sorted(str(row[0]) for row in contexts),
        "successor_vector_message_ids": sorted(str(row[0]) for row in messages),
        "successor_vector_tool_call_ids": sorted(str(row[0]) for row in tools),
    }
