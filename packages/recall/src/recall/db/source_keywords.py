"""Scoped keyword obligations tied to acknowledged canonical source bindings."""

from dataclasses import dataclass

import duckdb

from recall.core.source_path import SourcePath
from recall.db.lease import DbLease, TransactionCapability

SOURCE_SCOPE_LIMIT = 256
KEYWORD_OWNERS = """
keyword_owners AS (
    SELECT s.id AS session_id
    FROM fts_sidecar_pending p
    LEFT JOIN messages m ON p.kind='message' AND m.id=p.id
    LEFT JOIN tool_calls t ON p.kind='tool_call' AND t.id=p.id
    LEFT JOIN sessions s ON s.id=coalesce(m.session_id,t.session_id)
), unattributed AS (
    SELECT EXISTS(SELECT 1 FROM keyword_owners WHERE session_id IS NULL) AS unknown
)
"""


@dataclass(frozen=True)
class SettledKeywords:
    path: SourcePath


@dataclass(frozen=True)
class PendingKeywords:
    path: SourcePath
    reason: str


type KeywordMembership = SettledKeywords | PendingKeywords


def keyword_membership(
    path: SourcePath, *, bound: bool, pending: bool, unknown: bool
) -> KeywordMembership:
    if not bound:
        return PendingKeywords(path, "captured source lacks its acknowledged canonical session")
    if unknown:
        return PendingKeywords(
            path, "unattributed keyword intents prevent proving the selected scope"
        )
    if pending:
        return PendingKeywords(
            path, "the selected source scope still has unresolved keyword intents"
        )
    return SettledKeywords(path)


@dataclass(frozen=True)
class SourceSelection:
    paths: tuple[SourcePath, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.paths) <= SOURCE_SCOPE_LIMIT or len(set(self.paths)) != len(
            self.paths
        ):
            raise ValueError("source selection requires 1..256 distinct identities")

    def inspect_on(self, unit: DbLease | TransactionCapability) -> tuple[KeywordMembership, ...]:
        return unit.call("source:keyword_membership", self.inspect)

    def inspect(self, conn: duckdb.DuckDBPyConnection) -> tuple[KeywordMembership, ...]:
        rows = conn.execute(
            f"""WITH {KEYWORD_OWNERS}, selected AS (
                SELECT unnest(?::VARCHAR[]) AS source, unnest(?::VARCHAR[]) AS source_path
            )
            SELECT selected.source, selected.source_path, s.id IS NOT NULL AS bound,
                   EXISTS(SELECT 1 FROM keyword_owners k WHERE k.session_id=s.id) AS pending,
                   unattributed.unknown
            FROM selected
            LEFT JOIN source_files sf USING(source,source_path)
            LEFT JOIN sessions s ON s.id=sf.session_id
            CROSS JOIN unattributed
            ORDER BY selected.source,selected.source_path""",
            [[path.source for path in self.paths], [path.path for path in self.paths]],
        ).fetchall()
        assert len(rows) == len(self.paths), "canonical source binding is not unique"
        return tuple(
            keyword_membership(
                SourcePath(source, path), bound=bound, pending=pending, unknown=unknown
            )
            for source, path, bound, pending, unknown in rows
        )
