"""Resolve public session selectors before reading or marking a session."""

from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb

from recall.core.types import UNATTRIBUTED_HOST

_HEX_ID_PREFIX_RE = re.compile(r"[0-9a-f]{6,31}")
_CANDIDATE_LIMIT = 20


class UnknownSessionError(ValueError):
    """No indexed session matches the selector in its source/host scope."""


@dataclass(frozen=True)
class SessionIdentity:
    id: str
    source: str
    source_session_id: str | None
    source_path: str
    host: str


def resolve_session_identifier(
    conn: duckdb.DuckDBPyConnection,
    identifier: str,
    *,
    source: str | None = None,
    host: str | None = None,
) -> SessionIdentity:
    """Resolve exact recall ID, exact source ID, then a 6-31 hex recall prefix.

    Source and host restrict the lookup. Ambiguity reports bounded candidates
    with canonical IDs callers can use to select one without guessing.
    """
    scope: list[str] = []
    values: list[object] = []
    if source is not None:
        scope.append("s.source = ?")
        values.append(source)
    if host is not None:
        scope.append("ss.host = ?")
        values.append(host)
    predicates = [("s.id = ?", identifier, "session id")]
    predicates.append(("s.source_session_id = ?", identifier, "source_session_id"))
    if _HEX_ID_PREFIX_RE.fullmatch(identifier):
        predicates.append(("s.id LIKE ?", identifier + "%", "session id prefix"))
    for predicate, value, kind in predicates:
        exact_id = kind == "session id"
        where = " AND ".join([predicate, *([] if exact_id else scope)])
        rows = conn.execute(
            f"""SELECT s.id, s.source, s.source_session_id, s.source_path, ss.host
                FROM sessions s LEFT JOIN session_state ss ON ss.session_id = s.id
                WHERE {where} ORDER BY s.id LIMIT ?""",
            [value, *([] if exact_id else values), _CANDIDATE_LIMIT + 1],
        ).fetchall()
        if not rows and scope and kind == "source_session_id":
            excluded = conn.execute(
                """SELECT s.id, s.source, ss.host FROM sessions s
                    LEFT JOIN session_state ss ON ss.session_id = s.id
                    WHERE s.source_session_id = ? ORDER BY s.id LIMIT 1""",
                [identifier],
            ).fetchone()
            if excluded is not None:
                raise _outside_scope(excluded[0], excluded[1], excluded[2])
        if len(rows) == 1:
            row = rows[0]
            if exact_id and (
                (source is not None and row[1] != source) or (host is not None and row[4] != host)
            ):
                raise _outside_scope(row[0], row[1], row[4])
            raw_host = row[4]
            label = (
                raw_host.strip()
                if isinstance(raw_host, str) and raw_host.strip()
                else UNATTRIBUTED_HOST
            )
            return SessionIdentity(row[0], row[1], row[2], row[3], label)
        if rows:
            candidates = ", ".join(
                f"{row[0]} (source={row[1]}, host={row[4] or UNATTRIBUTED_HOST})"
                for row in rows[:_CANDIDATE_LIMIT]
            )
            more = "; more candidates omitted" if len(rows) > _CANDIDATE_LIMIT else ""
            raise ValueError(
                f"ambiguous {kind}: {identifier} matches multiple sessions: {candidates}{more}; "
                "use an exact recall session id or narrow source/host filters"
            )
    raise UnknownSessionError(
        f"session not found in indexed metadata: {identifier}; unindexed sources were not searched"
    )


def _outside_scope(session_id: str, source: str, host: str | None) -> ValueError:
    return ValueError(
        f"session {session_id} (source={source}, host={host or UNATTRIBUTED_HOST}) "
        "is outside the requested source/host scope"
    )
