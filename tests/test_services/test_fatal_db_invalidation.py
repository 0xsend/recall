"""A fatally invalidated DuckDB instance must stop the daemon, not loop forever.

DuckDB marks the whole *database instance* invalid after certain index faults,
and `duckdb.connect()` returns that same cached instance for the life of the
process -- so close+reopen cannot heal it. Background loops that log-and-continue
therefore spin on a dead database indefinitely (observed on a live host: 4,891
identical errors and a 46 MB log while `daemon status` still read healthy).
Every loop that touches the shared connection owes the same terminal behaviour
the RPC dispatcher already has (REQ-RESIL-011).
"""

from __future__ import annotations

import logging

import duckdb
import pytest
from recall.core.models import TailFacts
from recall.db.fatal import is_fatal_db_invalidation

FATAL_MESSAGE = "database has been invalidated because of a previous fatal error"


class TestFatalClassification:
    def test_duckdb_fatal_exception_is_fatal(self) -> None:
        assert is_fatal_db_invalidation(duckdb.FatalException(FATAL_MESSAGE))

    def test_wrapped_message_without_duckdb_type_is_fatal(self) -> None:
        assert is_fatal_db_invalidation(RuntimeError(f"catch-up failed: {FATAL_MESSAGE}"))

    def test_ordinary_error_is_not_fatal(self) -> None:
        assert not is_fatal_db_invalidation(RuntimeError("boom"))

    def test_constraint_violation_is_not_fatal(self) -> None:
        assert not is_fatal_db_invalidation(duckdb.ConstraintException('duplicate key "a"'))


class TestUsageHarvestPropagates:
    """`_maybe_harvest_usage` swallows every exception by design -- a harvest
    failure must not abort a watch tick. A dead database is not a harvest
    failure, so it is the one error that must escape."""

    def test_ordinary_harvest_failure_is_still_swallowed(self, monkeypatch) -> None:
        import recall.services.usage_harvest as harvest_module
        from recall.services.watcher import _maybe_harvest_usage

        def _boom(conn: object) -> None:
            raise RuntimeError("grok log unreadable")

        monkeypatch.setattr(harvest_module, "harvest_grok_unified_log", _boom)
        _maybe_harvest_usage(duckdb.connect(":memory:"))  # must not raise

    def test_swallowed_failure_carries_traceback(self, monkeypatch, caplog) -> None:
        """REQ-RESIL-023: the swallowed error is logged with its
        traceback so the failing statement is attributable without a
        reproduction."""
        import recall.services.usage_harvest as harvest_module
        from recall.services.watcher import _maybe_harvest_usage

        def _short_row(conn: object) -> None:
            raise IndexError("tuple index out of range")

        monkeypatch.setattr(harvest_module, "harvest_grok_unified_log", _short_row)
        with caplog.at_level(logging.WARNING, logger="recall.watcher"):
            _maybe_harvest_usage(duckdb.connect(":memory:"))

        record = next(r for r in caplog.records if "usage harvest failed" in r.getMessage())
        assert record.exc_info is not None
        assert record.exc_info[0] is IndexError

    def test_fatal_invalidation_escapes(self, monkeypatch) -> None:
        import recall.services.usage_harvest as harvest_module
        from recall.services.watcher import _maybe_harvest_usage

        def _fatal(conn: object) -> None:
            raise duckdb.FatalException(FATAL_MESSAGE)

        monkeypatch.setattr(harvest_module, "harvest_grok_unified_log", _fatal)
        with pytest.raises(duckdb.FatalException):
            _maybe_harvest_usage(duckdb.connect(":memory:"))


class TestIdentityRewriteIsAtomic:
    """`_rewrite_session_with_new_identity` runs outside any transaction, so it
    hand-rolls its own rollback: delete the session again, then re-insert the
    rows it saved. That compensation is both lossy (it re-inserts session_state
    without every column) and unsafe (the second delete can fault the index and
    invalidate the whole instance). A real transaction is the correct instrument.
    """

    @staticmethod
    def _session(source_session_id: str | None):
        from recall.core.models import Message, Session, ToolCall
        from recall.core.types import Role, Source

        return Session(
            id="s1",
            source=Source.CODEX,
            source_path="/tmp/atomic.jsonl",
            source_session_id=source_session_id,
            file_mtime=1.0,
            file_size=100,
            messages=[
                Message(
                    id="m1",
                    session_id="s1",
                    idx=0,
                    role=Role.ASSISTANT,
                    content="hello",
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            session_id="s1",
                            message_id="m1",
                            idx=0,
                            tool_name="bash",
                            bash_command="git status",
                        )
                    ],
                )
            ],
        )

    def test_failed_rewrite_preserves_every_session_state_column(self, monkeypatch) -> None:
        import recall.services.indexer as indexer_module
        from recall.db.schema import ensure_schema
        from recall.services.indexer import _write_session

        conn = duckdb.connect(":memory:")
        ensure_schema(conn)
        _write_session(conn, self._session(None), tail_facts=TailFacts())
        # sidecar_mtime is set by a later pass, so a rewrite must carry it through.
        conn.execute("UPDATE session_state SET sidecar_mtime = 1234.5 WHERE session_id = 's1'")
        before = conn.execute("SELECT * FROM session_state WHERE session_id = 's1'").fetchone()

        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("insert failed mid-rewrite")

        monkeypatch.setattr(indexer_module, "insert_tool_calls", _boom)
        with pytest.raises(RuntimeError):
            _write_session(conn, self._session("new-source-id"), tail_facts=TailFacts())

        after = conn.execute("SELECT * FROM session_state WHERE session_id = 's1'").fetchone()
        assert after == before, "a failed rewrite must leave session_state untouched"

    def test_failed_rewrite_preserves_identity_and_children(self, monkeypatch) -> None:
        import recall.services.indexer as indexer_module
        from recall.db.schema import ensure_schema
        from recall.services.indexer import _write_session

        conn = duckdb.connect(":memory:")
        ensure_schema(conn)
        _write_session(conn, self._session(None), tail_facts=TailFacts())
        before_identity = conn.execute("SELECT * FROM sessions WHERE id = 's1'").fetchone()

        def _boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("insert failed mid-rewrite")

        monkeypatch.setattr(indexer_module, "insert_tool_calls", _boom)
        with pytest.raises(RuntimeError):
            _write_session(conn, self._session("new-source-id"), tail_facts=TailFacts())

        assert conn.execute("SELECT * FROM sessions WHERE id = 's1'").fetchone() == before_identity
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone() == (1,)
