"""Keyword transaction cleanup retains the originating fatal and rolls rows back."""

import asyncio
from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall.core.repair_state import PublicationRole
from recall.db.fts_sidecar_stage import KeywordTurn, OwnedKeywordTurn
from recall.services.owned_jobs import DaemonJobs


@pytest.mark.parametrize("stage", ["unregister", "rollback", "failure_diagnostic"])
def test_transaction_observes_primary_before_independent_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    published: list[tuple[BaseException, PublicationRole]] = []
    owner = DaemonJobs(lambda error, site, role: published.append((error, role)), lambda: None)
    primary = duckdb.FatalException("Failed to delete all rows from index in keyword write")
    cleanup = duckdb.FatalException("database has been invalidated during keyword cleanup")
    db_path = tmp_path / "keyword.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE witness(value INT)")
    conn.execute("CREATE TABLE fts_sidecar_pending(kind TEXT, id TEXT, op TEXT)")
    observed_cleanup = owner.cleanup_db
    observed_call = owner.call_db

    def fail() -> None:
        raise primary

    def injected_call(operation: str, work: Any) -> Any:
        if stage == "unregister" and operation == "keyword:stage_intents":
            return observed_call(operation, fail)
        return observed_call(operation, work)

    def injected_cleanup(operation: str, work: Any) -> Any:
        if operation == f"keyword:{stage}":

            def apply_then_fail() -> Any:
                work()
                assert published[0] == (primary, PublicationRole.PRIMARY)
                raise cleanup

            return observed_cleanup(operation, apply_then_fail)
        return observed_cleanup(operation, work)

    def diagnostic(error: BaseException) -> None:
        if stage == "failure_diagnostic":
            assert error is primary
            assert published[0] == (primary, PublicationRole.PRIMARY)
            raise cleanup

    monkeypatch.setattr(owner, "call_db", injected_call)
    monkeypatch.setattr(owner, "cleanup_db", injected_cleanup)
    try:
        with owner.lease(conn) as lease, pytest.raises(BaseExceptionGroup):
            turn = OwnedKeywordTurn(lease, KeywordTurn(tmp_path / "keyword.sqlite", ("content",)))
            with turn.transaction(conn, on_failure=diagnostic) as pending:
                conn.execute("INSERT INTO witness VALUES (17)")
                if stage == "unregister":
                    pending.messages((("m1", "hello", ""),))
                else:
                    raise primary
        assert published == [
            (primary, PublicationRole.PRIMARY),
            (cleanup, PublicationRole.SECONDARY),
        ]
    finally:
        conn.close()
        asyncio.run(owner.close())
    with duckdb.connect(str(db_path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM witness").fetchall() == []
        assert fresh.execute("SELECT * FROM fts_sidecar_pending").fetchall() == []
