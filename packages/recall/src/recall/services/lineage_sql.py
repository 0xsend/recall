"""Session lineage SQL for the schema a connection holds (REQ-SCHEMA-007).

Read-only opens may see a database that the daemon has not yet upgraded, and
reads there degrade rather than fail. Every reader takes its lineage
expressions from `lineage_sql`, which returns the stored columns when they
exist and NULLs otherwise, so a schema without lineage reads as a schema
whose sessions are all roots.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

_NULL_TEXT = "CAST(NULL AS TEXT)"


@dataclass(frozen=True)
class LineageSql:
    """Expressions over the `s` alias of `sessions` (joined to `ss`, its `session_state`)."""

    root: str
    agent_id: str
    agent_type: str
    agent_description: str

    @property
    def columns(self) -> str:
        """The projection `sessions.lineage_from_row` decodes."""
        return f"{self.root}, {self.agent_id}, {self.agent_type}, {self.agent_description}"

    @property
    def roots(self) -> str:
        """Root sessions keyed the way a lineage names them.

        Copies sharing a key (a moved transcript beside its original) resolve
        to the smallest recall id.
        """
        return f"""
            SELECT s.source, s.source_session_id, ss.host, MIN(s.id) AS root_id
            FROM sessions s JOIN session_state ss ON ss.session_id = s.id
            WHERE {self.root} IS NULL AND s.source_session_id IS NOT NULL
            GROUP BY s.source, s.source_session_id, ss.host
        """

    @property
    def children(self) -> str:
        """Linked-session counts per root key."""
        return f"""
            SELECT s.source, {self.root} AS root_source_session_id, ss.host,
                   COUNT(*) AS subagent_count
            FROM sessions s JOIN session_state ss ON ss.session_id = s.id
            WHERE {self.root} IS NOT NULL
            GROUP BY s.source, {self.root}, ss.host
        """


_STORED = LineageSql(
    root="s.root_source_session_id",
    agent_id="s.agent_id",
    agent_type="s.agent_type",
    agent_description="s.agent_description",
)
_ABSENT = LineageSql(
    root=_NULL_TEXT, agent_id=_NULL_TEXT, agent_type=_NULL_TEXT, agent_description=_NULL_TEXT
)


def lineage_sql(conn: duckdb.DuckDBPyConnection) -> LineageSql:
    rows = conn.execute(
        "SELECT 1 FROM duckdb_columns() WHERE database_name = current_database()"
        " AND schema_name = current_schema() AND table_name = 'sessions'"
        " AND column_name = 'root_source_session_id'"
    ).fetchall()
    return _STORED if rows else _ABSENT
