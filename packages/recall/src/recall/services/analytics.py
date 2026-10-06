from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, cast

import duckdb

from recall.core.config import AppConfig
from recall.core.types import Source, default_session_host
from recall.db import connect_readonly
from recall.parsers.skills import derive_skill_names, is_skill_candidate

DANGEROUS_BASES = {
    "rm",
    "sudo",
    "chmod",
    "chown",
    "dd",
    "mkfs",
    "mount",
    "umount",
    "shutdown",
    "reboot",
    "kill",
    "killall",
}


@dataclass(frozen=True)
class OverviewStats:
    sessions: int
    messages: int
    tool_calls: int
    bash_calls: int


@dataclass(frozen=True)
class ToolStat:
    tool_name: str
    count: int


@dataclass(frozen=True)
class BashStat:
    bash_base: str | None
    bash_sub: str | None
    count: int
    is_compound: bool


@dataclass(frozen=True)
class PermissionSuggestion:
    pattern: str
    count: int
    confidence: str
    reason: str


@dataclass(frozen=True)
class PermissionSkipped:
    pattern: str
    count: int
    reason: str


@dataclass(frozen=True)
class TokenStat:
    repo: str | None
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class UsageStat:
    source: str
    model: str | None
    host: str | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    fresh_input_tokens: int | None
    session_count: int


@dataclass(frozen=True)
class SkillStat:
    skill_name: str
    source: str
    host: str
    invocations: int
    sessions: int


@dataclass(frozen=True)
class SkillPopulation:
    considered_sessions: int
    attributed_invocations: int
    unattributed_candidates: int


@dataclass(frozen=True)
class SkillCoverage:
    scope: str
    expected_hosts: tuple[str, ...]
    successful_hosts: tuple[str, ...]
    covered_sources: tuple[str, ...]
    considered_sessions: int
    attributed_invocations: int
    unattributed_candidates: int
    control: SkillPopulation


@dataclass(frozen=True)
class SkillUsageResult:
    rows: tuple[SkillStat, ...]
    coverage: SkillCoverage


def overview(
    *, config: AppConfig | None = None, conn: duckdb.DuckDBPyConnection | None = None
) -> OverviewStats:
    config = config or AppConfig.load()
    owned_conn = conn is None
    conn = conn or connect_readonly(config)
    try:
        sessions = _count(conn, "sessions")
        messages = _count(conn, "messages")
        tool_calls = _count(conn, "tool_calls")
        bash_row = conn.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE bash_command IS NOT NULL"
        ).fetchone()
        bash_calls = int(bash_row[0]) if bash_row else 0
        return OverviewStats(
            sessions=sessions,
            messages=messages,
            tool_calls=tool_calls,
            bash_calls=bash_calls,
        )
    finally:
        if owned_conn:
            conn.close()


def tool_usage(
    limit: int = 50,
    *,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[ToolStat]:
    config = config or AppConfig.load()
    owned_conn = conn is None
    conn = conn or connect_readonly(config)
    try:
        rows = conn.execute(
            """
            SELECT tool_name, COUNT(*) AS count
            FROM tool_calls
            GROUP BY tool_name
            ORDER BY count DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [ToolStat(tool_name=row[0], count=int(row[1])) for row in rows]
    finally:
        if owned_conn:
            conn.close()


def bash_breakdown(
    limit: int = 100,
    *,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[BashStat]:
    config = config or AppConfig.load()
    owned_conn = conn is None
    conn = conn or connect_readonly(config)
    try:
        rows = conn.execute(
            """
            SELECT bash_base, bash_sub, COUNT(*) AS count, MAX(is_compound) AS is_compound
            FROM tool_calls
            WHERE bash_command IS NOT NULL
            GROUP BY bash_base, bash_sub
            ORDER BY count DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [
            BashStat(
                bash_base=row[0],
                bash_sub=row[1],
                count=int(row[2]),
                is_compound=bool(row[3]),
            )
            for row in rows
        ]
    finally:
        if owned_conn:
            conn.close()


def bash_suggestions(
    high_threshold: int = 50,
    medium_threshold: int = 10,
    *,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> tuple[list[PermissionSuggestion], list[PermissionSkipped]]:
    suggestions: list[PermissionSuggestion] = []
    skipped: list[PermissionSkipped] = []
    for stat in bash_breakdown(limit=500, config=config, conn=conn):
        base = (stat.bash_base or "").strip()
        if not base:
            continue
        pattern = _format_pattern(base, stat.bash_sub)
        if base in DANGEROUS_BASES:
            skipped.append(
                PermissionSkipped(pattern=pattern, count=stat.count, reason="Destructive command")
            )
            continue
        if stat.is_compound:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pattern,
                    count=stat.count,
                    confidence="review",
                    reason="Contains compound operators",
                )
            )
            continue
        if stat.count >= high_threshold:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pattern,
                    count=stat.count,
                    confidence="high",
                    reason="No dangerous patterns detected",
                )
            )
        elif stat.count >= medium_threshold:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pattern,
                    count=stat.count,
                    confidence="medium",
                    reason="No dangerous patterns detected",
                )
            )
        else:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pattern,
                    count=stat.count,
                    confidence="review",
                    reason="Low usage volume",
                )
            )
    return suggestions, skipped


def token_usage(
    limit: int = 50,
    *,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[TokenStat]:
    config = config or AppConfig.load()
    owned_conn = conn is None
    conn = conn or connect_readonly(config)
    try:
        rows = conn.execute(
            """
            SELECT ss.git_repo, SUM(COALESCE(ss.input_tokens, 0)) AS input_tokens,
                   SUM(COALESCE(ss.output_tokens, 0)) AS output_tokens
            FROM session_state ss
            GROUP BY ss.git_repo
            ORDER BY
                SUM(COALESCE(ss.input_tokens, 0)) + SUM(COALESCE(ss.output_tokens, 0)) DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [
            TokenStat(
                repo=row[0],
                input_tokens=int(row[1] or 0),
                output_tokens=int(row[2] or 0),
            )
            for row in rows
        ]
    finally:
        if owned_conn:
            conn.close()


def usage_by_source(
    *,
    since: datetime | None = None,
    limit: int = 200,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[UsageStat]:
    """Token totals by source x model x host (REQ-USAGE-020-023).

    Excludes sessions where both input_tokens and output_tokens are NULL so
    unharvested Grok rows do not pollute totals as zeros.
    """
    config = config or AppConfig.load()
    owned_conn = conn is None
    conn = conn or connect_readonly(config)
    try:
        has_host = _column_exists(conn, "session_state", "host")
        has_cached = _column_exists(conn, "session_state", "cached_input_tokens")
        host_expr = "ss.host" if has_host else "CAST(NULL AS VARCHAR)"
        # SUM ignores NULLs; all-NULL → NULL (unknown cache). Do not COALESCE to 0
        # before SUM — that invents zero-cache for Claude/Kimi (REQ-USAGE-021).
        cached_expr = "SUM(ss.cached_input_tokens)" if has_cached else "CAST(NULL AS BIGINT)"
        where = [
            "NOT (ss.input_tokens IS NULL AND ss.output_tokens IS NULL)",
        ]
        params: list[object] = []
        if since is not None:
            where.append("COALESCE(ss.ended_at, ss.started_at, ss.indexed_at) >= ?")
            params.append(since)
        where_sql = " AND ".join(where)
        rows = conn.execute(
            f"""
            SELECT
                s.source,
                ss.model,
                {host_expr} AS host,
                SUM(COALESCE(ss.input_tokens, 0)) AS input_tokens,
                {cached_expr} AS cached_input_tokens,
                SUM(COALESCE(ss.output_tokens, 0)) AS output_tokens,
                COUNT(*) AS session_count
            FROM sessions s
            JOIN session_state ss ON ss.session_id = s.id
            WHERE {where_sql}
            GROUP BY s.source, ss.model, {host_expr}
            ORDER BY
                SUM(COALESCE(ss.input_tokens, 0)) + SUM(COALESCE(ss.output_tokens, 0)) DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        stats: list[UsageStat] = []
        for row in rows:
            input_tokens = int(row[3] or 0)
            cached_raw = row[4]
            cached_known = cached_raw is not None
            cached = int(cached_raw or 0) if cached_known else 0
            fresh: int | None = None
            if cached_known and cached <= input_tokens:
                fresh = input_tokens - cached
            stats.append(
                UsageStat(
                    source=str(row[0]),
                    model=row[1],
                    host=row[2],
                    input_tokens=input_tokens,
                    cached_input_tokens=cached if cached_known else 0,
                    output_tokens=int(row[5] or 0),
                    fresh_input_tokens=fresh,
                    session_count=int(row[6] or 0),
                )
            )
        return stats
    finally:
        if owned_conn:
            conn.close()


def skill_usage(
    *,
    sources: Sequence[Source] | None = None,
    since: datetime | None = None,
    local_host: str | None = None,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> SkillUsageResult:
    """Skill invocations by name x source x host with fail-closed coverage data."""
    config = config or AppConfig.load()
    owned_conn = conn is None
    conn = conn or connect_readonly(config)
    selected = tuple(sources) if sources else tuple(Source)
    host = (local_host or default_session_host()).strip() or default_session_host()
    try:
        control_rows, control = _skill_population(
            conn,
            sources=selected,
            since=None,
            local_host=host,
        )
        if since is None:
            rows, window = control_rows, control
        else:
            rows, window = _skill_population(
                conn,
                sources=selected,
                since=since,
                local_host=host,
            )
        return SkillUsageResult(
            rows=rows,
            coverage=SkillCoverage(
                scope="local",
                expected_hosts=(host,),
                successful_hosts=(host,),
                covered_sources=tuple(source.value for source in selected),
                considered_sessions=window.considered_sessions,
                attributed_invocations=window.attributed_invocations,
                unattributed_candidates=window.unattributed_candidates,
                control=control,
            ),
        )
    finally:
        if owned_conn:
            conn.close()


# A transcript's path below its project or cwd directory survives a rename of
# that directory, so a transcript indexed again after `~/old/repo` became
# `~/new/repo` shares this key with its earlier row.  Codex dates its rollouts
# instead, and the date directories never move.
_TRANSCRIPT_KEY_SQL: Final = (
    "COALESCE(NULLIF(regexp_extract(s.source_path, "
    "'^.*/(?:projects|sessions)/[^/]+/(.+)$', 1), ''), s.source_path)"
)


def _skill_population(
    conn: duckdb.DuckDBPyConnection,
    *,
    sources: Sequence[Source],
    since: datetime | None,
    local_host: str,
) -> tuple[tuple[SkillStat, ...], SkillPopulation]:
    source_values = [source.value for source in sources]
    placeholders = ", ".join("?" for _ in source_values)
    time_clause = ""
    params: list[object] = list(source_values)
    if since is not None:
        time_clause = " AND COALESCE(ss.ended_at, ss.started_at, ss.indexed_at) >= ?"
        params.append(since)

    # One row per transcript: a moved transcript is indexed once per path it
    # was seen at, and counting every row counts its history once per move.
    # The copy still on disk wins, then the one that ran latest.
    transcripts = f"""
        WITH transcripts AS (
            SELECT s.id, s.source, ss.cwd
            FROM sessions s
            JOIN session_state ss ON ss.session_id = s.id
            WHERE s.source IN ({placeholders}){time_clause}
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY s.source, COALESCE(ss.host, ''), {_TRANSCRIPT_KEY_SQL}
                ORDER BY
                    EXISTS (
                        SELECT 1 FROM source_files f
                        WHERE f.source = s.source
                          AND f.source_path = s.source_path
                          AND NOT f.missing
                    ) DESC,
                    COALESCE(ss.ended_at, ss.started_at, ss.indexed_at) DESC,
                    s.id
            ) = 1
        )
    """
    count_row = conn.execute(f"{transcripts} SELECT COUNT(*) FROM transcripts", params).fetchone()
    considered_sessions = int(count_row[0] or 0) if count_row else 0

    candidate_rows = conn.execute(
        f"""
        {transcripts}
        SELECT
            tc.tool_name,
            CAST(tc.tool_input AS VARCHAR),
            tc.bash_command,
            t.id,
            t.source,
            t.cwd
        FROM tool_calls tc
        JOIN transcripts t ON t.id = tc.session_id
        WHERE (
              NULLIF(TRIM(tc.skill_name), '') IS NOT NULL
              OR LOWER(tc.tool_name) = 'skill'
              OR POSITION('SKILL.md' IN COALESCE(tc.bash_command, '')) > 0
              OR POSITION('SKILL.md' IN COALESCE(CAST(tc.tool_input AS VARCHAR), '')) > 0
          )
        """,
        params,
    ).fetchall()

    calls = [_SkillCandidate.from_row(row) for row in candidate_rows]
    buckets: dict[tuple[str, str, str], tuple[int, set[str]]] = {}
    unattributed_candidates = 0
    attributed_invocations = 0
    for call in calls:
        skill_names = call.skill_names
        if not skill_names:
            if is_skill_candidate(call.tool_name, call.tool_input, call.bash_command):
                unattributed_candidates += 1
            continue

        for skill_name in skill_names:
            key = (skill_name, call.source.value, local_host)
            invocations, session_ids = buckets.get(key, (0, set()))
            session_ids.add(call.session_id)
            buckets[key] = (invocations + 1, session_ids)
        attributed_invocations += len(skill_names)

    stats = tuple(
        SkillStat(
            skill_name=key[0],
            source=key[1],
            host=key[2],
            invocations=value[0],
            sessions=len(value[1]),
        )
        for key, value in sorted(buckets.items())
    )
    return stats, SkillPopulation(
        considered_sessions=considered_sessions,
        attributed_invocations=attributed_invocations,
        unattributed_candidates=unattributed_candidates,
    )


@dataclass(frozen=True)
class _SkillCandidate:
    tool_name: str
    tool_input: Any
    bash_command: str | None
    session_id: str
    source: Source
    skill_names: tuple[str, ...]

    @classmethod
    def from_row(cls, row: tuple[object, ...]) -> _SkillCandidate:
        tool_name = str(row[0] or "")
        tool_input = _parse_tool_input(row[1])
        bash_command = str(row[2]) if row[2] is not None else None
        cwd = str(row[5]) if row[5] is not None else None
        return cls(
            tool_name=tool_name,
            tool_input=tool_input,
            bash_command=bash_command,
            session_id=str(row[3]),
            source=Source(str(row[4])),
            # Re-derived rather than read from the stored column, which holds
            # one name from whichever attribution rules indexed the row.
            skill_names=derive_skill_names(tool_name, tool_input, bash_command, cwd=cwd),
        )


def _parse_tool_input(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if not isinstance(value, str) or not value:
        return None
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _column_exists(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return any(str(row[1]) == column for row in rows)


def _format_pattern(base: str, sub: str | None) -> str:
    if sub:
        return f"{base} {sub}"
    return f"{base} *"


def _count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0
