from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, cast

import duckdb

from recall.core.config import AppConfig
from recall.core.time import to_naive_utc
from recall.core.types import Source, default_session_host
from recall.db.lease import DbLease
from recall.db.ownership import standalone_read_lease
from recall.parsers.skills import derive_skill_names, is_skill_candidate

DEFAULT_TOOL_STATS_LIMIT = 50
DEFAULT_BASH_STATS_LIMIT = 100
DEFAULT_BASH_SUGGESTIONS_LIMIT = 500
DEFAULT_TOKEN_STATS_LIMIT = 50
DEFAULT_USAGE_STATS_LIMIT = 200

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


def _session_filter_cte(
    source: Source | None,
    since: datetime | None,
) -> tuple[str, list[object]]:
    """Build a narrow materialized session-ID filter CTE when source or since is active."""
    where: list[str] = []
    params: list[object] = []
    if source is not None:
        where.append("s.source = ?")
        params.append(source.value)
    if since is not None:
        where.append("COALESCE(ss.ended_at, ss.started_at, ss.indexed_at) >= ?")
        params.append(to_naive_utc(since))
    assert where, "_session_filter_cte requires at least one filter"
    where_sql = " AND ".join(where)
    cte = f"""
        WITH filtered AS MATERIALIZED (
            SELECT s.id
            FROM sessions s
            JOIN session_state ss ON ss.session_id = s.id
            WHERE {where_sql}
        )
    """
    return cte, params


def overview(
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> OverviewStats:
    config = config or AppConfig.load()
    with standalone_read_lease(config, conn) as lease:
        return overview_on(lease, source=source, since=since, config=config)


def overview_on(
    lease: DbLease,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig,
) -> OverviewStats:
    return lease.call(
        "analytics:overview",
        lambda conn: _overview(conn, source=source, since=since, config=config),
    )


def _overview(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig,
) -> OverviewStats:
    if source is None and since is None:
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

    cte, params = _session_filter_cte(source, since)
    sessions_row = conn.execute(f"{cte} SELECT COUNT(*) FROM filtered", params).fetchone()
    messages_row = conn.execute(
        f"{cte} SELECT COUNT(*) FROM messages m JOIN filtered f ON f.id = m.session_id",
        params,
    ).fetchone()
    tools_row = conn.execute(
        f"""
        {cte}
        SELECT
            COUNT(*),
            COUNT(*) FILTER (WHERE tc.bash_command IS NOT NULL)
        FROM tool_calls tc
        JOIN filtered f ON f.id = tc.session_id
        """,
        params,
    ).fetchone()
    return OverviewStats(
        sessions=int(sessions_row[0]) if sessions_row else 0,
        messages=int(messages_row[0]) if messages_row else 0,
        tool_calls=int(tools_row[0]) if tools_row else 0,
        bash_calls=int(tools_row[1]) if tools_row else 0,
    )


def tool_usage(
    limit: int = DEFAULT_TOOL_STATS_LIMIT,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[ToolStat]:
    config = config or AppConfig.load()
    with standalone_read_lease(config, conn) as lease:
        return tool_usage_on(lease, limit=limit, source=source, since=since, config=config)


def tool_usage_on(
    lease: DbLease,
    limit: int = DEFAULT_TOOL_STATS_LIMIT,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig,
) -> list[ToolStat]:
    return lease.call(
        "analytics:tool_usage",
        lambda conn: _tool_usage(conn, limit=limit, source=source, since=since, config=config),
    )


def _tool_usage(
    conn: duckdb.DuckDBPyConnection,
    limit: int = DEFAULT_TOOL_STATS_LIMIT,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig,
) -> list[ToolStat]:
    if source is None and since is None:
        rows = conn.execute(
            """
            SELECT tool_name, COUNT(*) AS count
            FROM tool_calls
            GROUP BY tool_name
            ORDER BY count DESC, tool_name ASC NULLS LAST
            LIMIT ?
            """,
            [limit],
        ).fetchall()
    else:
        cte, params = _session_filter_cte(source, since)
        rows = conn.execute(
            f"""
            {cte}
            SELECT tc.tool_name, COUNT(*) AS count
            FROM tool_calls tc
            JOIN filtered f ON f.id = tc.session_id
            GROUP BY tc.tool_name
            ORDER BY count DESC, tc.tool_name ASC NULLS LAST
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
    return [ToolStat(tool_name=row[0], count=int(row[1])) for row in rows]


def bash_breakdown(
    limit: int = DEFAULT_BASH_STATS_LIMIT,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[BashStat]:
    config = config or AppConfig.load()
    with standalone_read_lease(config, conn) as lease:
        return bash_breakdown_on(lease, limit=limit, source=source, since=since, config=config)


def bash_breakdown_on(
    lease: DbLease,
    limit: int = DEFAULT_BASH_STATS_LIMIT,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig,
) -> list[BashStat]:
    return lease.call(
        "analytics:bash_breakdown",
        lambda conn: _bash_breakdown(conn, limit=limit, source=source, since=since, config=config),
    )


def _bash_breakdown(
    conn: duckdb.DuckDBPyConnection,
    limit: int = DEFAULT_BASH_STATS_LIMIT,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig,
) -> list[BashStat]:
    if source is None and since is None:
        rows = conn.execute(
            """
            SELECT bash_base, bash_sub, COUNT(*) AS count, MAX(is_compound) AS is_compound
            FROM tool_calls
            WHERE bash_command IS NOT NULL
            GROUP BY bash_base, bash_sub
            ORDER BY count DESC, bash_base ASC NULLS LAST, bash_sub ASC NULLS LAST
            LIMIT ?
            """,
            [limit],
        ).fetchall()
    else:
        cte, params = _session_filter_cte(source, since)
        rows = conn.execute(
            f"""
            {cte}
            SELECT tc.bash_base, tc.bash_sub, COUNT(*) AS count, MAX(tc.is_compound) AS is_compound
            FROM tool_calls tc
            JOIN filtered f ON f.id = tc.session_id
            WHERE tc.bash_command IS NOT NULL
            GROUP BY tc.bash_base, tc.bash_sub
            ORDER BY count DESC, tc.bash_base ASC NULLS LAST, tc.bash_sub ASC NULLS LAST
            LIMIT ?
            """,
            [*params, limit],
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


def classify_bash_suggestions(
    stats: Sequence[BashStat],
    *,
    high_threshold: int = 50,
    medium_threshold: int = 10,
) -> tuple[list[PermissionSuggestion], list[PermissionSkipped]]:
    """Classify bash command breakdown rows into permission suggestions and skipped commands."""
    suggestion_buckets: dict[str, tuple[int, bool]] = {}
    skipped_buckets: dict[str, int] = {}
    for stat in stats:
        base = (stat.bash_base or "").strip()
        if not base:
            continue
        pattern = _format_pattern(base, stat.bash_sub)
        if base in DANGEROUS_BASES:
            skipped_buckets[pattern] = skipped_buckets.get(pattern, 0) + stat.count
            continue
        prev_count, prev_compound = suggestion_buckets.get(pattern, (0, False))
        suggestion_buckets[pattern] = (
            prev_count + stat.count,
            prev_compound or stat.is_compound,
        )

    suggestions: list[PermissionSuggestion] = []
    for pattern, (count, is_compound) in sorted(
        suggestion_buckets.items(), key=lambda item: (-item[1][0], item[0])
    ):
        if is_compound:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pattern,
                    count=count,
                    confidence="review",
                    reason="Contains compound operators",
                )
            )
        elif count >= high_threshold:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pattern,
                    count=count,
                    confidence="high",
                    reason="No dangerous patterns detected",
                )
            )
        elif count >= medium_threshold:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pattern,
                    count=count,
                    confidence="medium",
                    reason="No dangerous patterns detected",
                )
            )
        else:
            suggestions.append(
                PermissionSuggestion(
                    pattern=pattern,
                    count=count,
                    confidence="review",
                    reason="Low usage volume",
                )
            )

    skipped = [
        PermissionSkipped(pattern=pattern, count=count, reason="Destructive command")
        for pattern, count in sorted(skipped_buckets.items(), key=lambda item: (-item[1], item[0]))
    ]
    return suggestions, skipped


def bash_suggestions(
    high_threshold: int = 50,
    medium_threshold: int = 10,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> tuple[list[PermissionSuggestion], list[PermissionSkipped]]:
    config = config or AppConfig.load()
    with standalone_read_lease(config, conn) as lease:
        return bash_suggestions_on(
            lease,
            high_threshold,
            medium_threshold,
            source=source,
            since=since,
            config=config,
        )


def bash_suggestions_on(
    lease: DbLease,
    high_threshold: int = 50,
    medium_threshold: int = 10,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig,
) -> tuple[list[PermissionSuggestion], list[PermissionSkipped]]:
    return lease.call(
        "analytics:bash_suggestions",
        lambda conn: _bash_suggestions(
            high_threshold,
            medium_threshold,
            source=source,
            since=since,
            config=config,
            conn=conn,
        ),
    )


def _bash_suggestions(
    high_threshold: int = 50,
    medium_threshold: int = 10,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig,
    conn: duckdb.DuckDBPyConnection,
) -> tuple[list[PermissionSuggestion], list[PermissionSkipped]]:
    stats = _bash_breakdown(
        conn,
        limit=DEFAULT_BASH_SUGGESTIONS_LIMIT,
        source=source,
        since=since,
        config=config,
    )
    return classify_bash_suggestions(
        stats,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
    )


def token_usage(
    limit: int = DEFAULT_TOKEN_STATS_LIMIT,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[TokenStat]:
    config = config or AppConfig.load()
    with standalone_read_lease(config, conn) as lease:
        return token_usage_on(lease, limit=limit, source=source, since=since, config=config)


def token_usage_on(
    lease: DbLease,
    limit: int = DEFAULT_TOKEN_STATS_LIMIT,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig,
) -> list[TokenStat]:
    return lease.call(
        "analytics:token_usage",
        lambda conn: _token_usage(conn, limit=limit, source=source, since=since, config=config),
    )


def _token_usage(
    conn: duckdb.DuckDBPyConnection,
    limit: int = DEFAULT_TOKEN_STATS_LIMIT,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    config: AppConfig,
) -> list[TokenStat]:
    where: list[str] = []
    params: list[object] = []
    join_sql = ""
    if source is not None:
        join_sql = "JOIN sessions s ON s.id = ss.session_id"
        where.append("s.source = ?")
        params.append(source.value)
    if since is not None:
        where.append("COALESCE(ss.ended_at, ss.started_at, ss.indexed_at) >= ?")
        params.append(to_naive_utc(since))
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    rows = conn.execute(
        f"""
        SELECT ss.git_repo, SUM(COALESCE(ss.input_tokens, 0)) AS input_tokens,
               SUM(COALESCE(ss.output_tokens, 0)) AS output_tokens
        FROM session_state ss
        {join_sql}
        {where_sql}
        GROUP BY ss.git_repo
        ORDER BY
            SUM(COALESCE(ss.input_tokens, 0)) + SUM(COALESCE(ss.output_tokens, 0)) DESC,
            ss.git_repo ASC NULLS LAST
        LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    return [
        TokenStat(
            repo=row[0],
            input_tokens=int(row[1] or 0),
            output_tokens=int(row[2] or 0),
        )
        for row in rows
    ]


def usage_by_source(
    *,
    source: Source | None = None,
    since: datetime | None = None,
    limit: int = DEFAULT_USAGE_STATS_LIMIT,
    config: AppConfig | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> list[UsageStat]:
    """Token totals by source x model x host (REQ-USAGE-020-023).

    Excludes sessions where both input_tokens and output_tokens are NULL so
    unharvested Grok rows do not pollute totals as zeros.
    """
    config = config or AppConfig.load()
    with standalone_read_lease(config, conn) as lease:
        return usage_by_source_on(lease, source=source, since=since, limit=limit, config=config)


def usage_by_source_on(
    lease: DbLease,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    limit: int = DEFAULT_USAGE_STATS_LIMIT,
    config: AppConfig,
) -> list[UsageStat]:
    return lease.call(
        "analytics:usage_by_source",
        lambda conn: _usage_by_source(conn, source=source, since=since, limit=limit, config=config),
    )


def _usage_by_source(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: Source | None = None,
    since: datetime | None = None,
    limit: int = DEFAULT_USAGE_STATS_LIMIT,
    config: AppConfig,
) -> list[UsageStat]:
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
    if source is not None:
        where.append("s.source = ?")
        params.append(source.value)
    if since is not None:
        where.append("COALESCE(ss.ended_at, ss.started_at, ss.indexed_at) >= ?")
        params.append(to_naive_utc(since))
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
            -- One per spawn tree: a root and its linked sessions share the key
            -- (REQ-CLI-031); tokens still sum over every session.
            COUNT(DISTINCT COALESCE(s.root_source_session_id, s.source_session_id, s.id))
                AS session_count
        FROM sessions s
        JOIN session_state ss ON ss.session_id = s.id
        WHERE {where_sql}
        GROUP BY s.source, ss.model, {host_expr}
        ORDER BY
            SUM(COALESCE(ss.input_tokens, 0)) + SUM(COALESCE(ss.output_tokens, 0)) DESC,
            s.source ASC NULLS LAST,
            ss.model ASC NULLS LAST,
            {host_expr} ASC NULLS LAST
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
    with standalone_read_lease(config, conn) as lease:
        return skill_usage_on(
            lease, sources=sources, since=since, local_host=local_host, config=config
        )


def skill_usage_on(
    lease: DbLease,
    *,
    sources: Sequence[Source] | None = None,
    since: datetime | None = None,
    local_host: str | None = None,
    config: AppConfig,
) -> SkillUsageResult:
    return lease.call(
        "analytics:skill_usage",
        lambda conn: _skill_usage(
            conn, sources=sources, since=since, local_host=local_host, config=config
        ),
    )


def _skill_usage(
    conn: duckdb.DuckDBPyConnection,
    *,
    sources: Sequence[Source] | None = None,
    since: datetime | None = None,
    local_host: str | None = None,
    config: AppConfig,
) -> SkillUsageResult:
    selected = tuple(sources) if sources else tuple(Source)
    host = (local_host or default_session_host()).strip() or default_session_host()
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
        params.append(to_naive_utc(since))

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
