"""Requested RPC work obtains release only from its actual settled scope."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import cast

import duckdb
import pytest
from recall.services.rpc_server import ClientConnection

from tests.test_services.test_raw_run_completion import _raw_setup


def test_actual_rpc_empty_recompute_releases_after_complete_target_capture(tmp_path: Path) -> None:
    database, server, _, _ = _raw_setup(tmp_path)

    async def scenario() -> None:
        try:
            summary = await server._recompute_context_request(
                config=server._config,
                since=None,
                only_mode=None,
                embed=False,
                client=None,
            )
            assert summary.total == 0
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone()
        assert row is not None and row[0] > datetime(2026, 11, 1)
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (True, True)


def _indexed_recompute_setup(tmp_path: Path):
    from dataclasses import replace

    from recall.core.config import ContextConfig
    from recall.services.coordinator import capture_path, observe_path, prepare_raw_sources

    database, server, path, parser = _raw_setup(tmp_path)
    path.write_text(
        path.read_text().replace('"id":"source"', '"id":"source","cwd":"/fixture/project"')
    )
    server._config = replace(
        server._config,
        embedding=replace(
            server._config.embedding,
            context=ContextConfig(mode="off"),
        ),
    )
    item = server._database.call(
        "fixture:observe", lambda conn: observe_path(parser, capture_path(parser, path), conn=conn)
    )
    prepared = prepare_raw_sources((item,), {"codex": parser})
    server._commit_raw_sources(prepared, server._config)
    server._database.call(
        "fixture:spent",
        lambda conn: conn.execute(
            "UPDATE daemon_index_repair SET success_since_attempt=false, "
            "success_since_failure=false"
        ).fetchone(),
    )
    server._database.call(
        "fixture:prior_success",
        lambda conn: conn.execute(
            "UPDATE runtime_state SET last_successful_at = '2026-11-01'"
        ).fetchone(),
    )
    return database, server


def test_actual_rpc_recompute_with_deferred_publication_keeps_spent_budget(tmp_path: Path) -> None:
    import sqlite3
    from dataclasses import replace

    from recall.core.config import ContextConfig
    from recall.db.fts_sidecar import sidecar_path

    database, server = _indexed_recompute_setup(tmp_path)
    config = replace(
        server._config,
        embedding=replace(
            server._config.embedding,
            context=ContextConfig(mode="template"),
        ),
    )

    async def scenario() -> None:
        holder = sqlite3.connect(sidecar_path(config.data_dir), isolation_level=None, timeout=0)
        try:
            holder.execute("BEGIN EXCLUSIVE")
            result = await server._recompute_context_request(
                config=config,
                since=None,
                only_mode=None,
                embed=False,
                client=None,
            )
            assert result.total == 1 and result.changed == 1 and result.failed == 0
        finally:
            holder.rollback()
            holder.close()
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (1,)
        assert fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone() == (
            datetime(2026, 11, 1),
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


@pytest.mark.parametrize("selected", [False, True])
def test_recompute_roster_distinguishes_selected_and_outside_keyword_obligations(
    tmp_path: Path,
    selected: bool,
) -> None:
    database, server = _indexed_recompute_setup(tmp_path)
    server._database.call(
        "fixture:pending",
        lambda conn: conn.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) "
            + (
                "SELECT 'message',id,'upsert' FROM messages"
                if selected
                else "VALUES ('message','outside-the-frozen-roster','delete')"
            )
        ).fetchone(),
    )

    async def scenario() -> None:
        try:
            result = await server._recompute_context_request(
                config=server._config,
                since=None,
                only_mode=None,
                embed=False,
                client=None,
            )
            assert result.total == 1
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone()
        assert row is not None
        assert (row[0] == datetime(2026, 11, 1)) == selected
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (not selected, not selected)
        assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (1,)


@pytest.mark.parametrize("cancelled", [False, True])
def test_recompute_records_failed_or_cancelled_without_releasing_budget(
    tmp_path: Path,
    cancelled: bool,
) -> None:
    database, server = _indexed_recompute_setup(tmp_path)
    origin = (
        asyncio.CancelledError("client departed") if cancelled else ValueError("progress failed")
    )

    class Client:
        async def send_progress(self, **kwargs) -> None:
            raise origin

    async def scenario() -> None:
        try:
            with pytest.raises(type(origin)) as caught:
                await server._recompute_context_request(
                    config=server._config,
                    since=None,
                    only_mode=None,
                    embed=False,
                    client=cast(ClientConnection, Client()),
                )
            assert caught.value is origin
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone()
        assert row is not None and row[0] == datetime(2026, 11, 1)
        assert row[1] == (
            "cancelled: requested context recompute departed" if cancelled else "progress failed"
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


@pytest.mark.parametrize("admitted_first", [False, True])
def test_rpc_departure_orders_against_the_actual_completion_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admitted_first: bool,
) -> None:
    import threading

    from recall.services import run_completion

    database, server, _, _ = _raw_setup(tmp_path)
    completion_entered = threading.Event()
    cancellation_entered = threading.Event()
    release = threading.Event()
    original_record = run_completion.record_completion_on
    original_cancel = run_completion.RunScope.cancel

    def cancel(scope, reason):
        if admitted_first:
            cancellation_entered.set()
        result = original_cancel(scope, reason)
        if not admitted_first:
            cancellation_entered.set()
        return result

    def record(*args, **kwargs):
        if not admitted_first:
            completion_entered.set()
            assert release.wait(5)
        else:

            def barrier(stage):
                if stage == "after_completion_history":
                    completion_entered.set()
                    assert release.wait(5)

            kwargs["barrier"] = barrier
        return original_record(*args, **kwargs)

    monkeypatch.setattr(run_completion, "record_completion_on", record)
    monkeypatch.setattr(run_completion.RunScope, "cancel", cancel)

    async def scenario() -> None:
        task = asyncio.create_task(
            server._recompute_context_request(
                config=server._config,
                since=None,
                only_mode=None,
                embed=False,
                client=None,
            )
        )
        try:
            assert await asyncio.to_thread(completion_entered.wait, 5)
            task.cancel()
            assert await asyncio.to_thread(cancellation_entered.wait, 5)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone()
        assert row is not None
        if admitted_first:
            assert row[0] > datetime(2026, 11, 1) and row[1] is None
        else:
            assert row == (datetime(2026, 11, 1), "cancelled: requested run departed")
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (admitted_first, admitted_first)


@pytest.mark.parametrize(
    "stage", ["after_completion_history", "before_completion_commit", "after_completion_commit"]
)
def test_rpc_recompute_completion_failure_executes_once_and_keeps_atomic_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    from recall.services import run_completion

    database, server, _, _ = _raw_setup(tmp_path)
    original = run_completion.record_completion_on
    origin = OSError("completion acknowledgement failed")
    executions = 0

    def record(*args, **kwargs):
        nonlocal executions
        executions += 1

        def fail(at):
            if at == stage:
                raise origin

        return original(*args, **kwargs, barrier=fail)

    monkeypatch.setattr(run_completion, "record_completion_on", record)

    async def scenario() -> None:
        try:
            with pytest.raises(
                run_completion.CompletionCommitUncertain
                if stage == "after_completion_commit"
                else OSError
            ) as caught:
                await server._recompute_context_request(
                    config=server._config,
                    since=None,
                    only_mode=None,
                    embed=False,
                    client=None,
                )
            if stage == "after_completion_commit":
                assert caught.value.__cause__ is origin
            else:
                assert caught.value is origin
        finally:
            await server.stop()

    asyncio.run(scenario())
    assert executions == 1
    committed = stage == "after_completion_commit"
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = fresh.execute(
            "SELECT last_successful_at,last_index_total FROM runtime_state"
        ).fetchone()
        assert row is not None
        assert (row[0] > datetime(2026, 11, 1)) == committed
        assert row[1] == (0 if committed else None)
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (committed, committed)


def test_rpc_recompute_reporting_failure_keeps_the_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recall.services import embed_phase, run_completion

    database, server = _indexed_recompute_setup(tmp_path)
    origin = ValueError("target capture unavailable")
    reports = 0

    def prepare(*args, **kwargs):
        raise origin

    def record(*args, **kwargs):
        nonlocal reports
        reports += 1
        raise OSError("reporting unavailable")

    monkeypatch.setattr(embed_phase, "prepare_context_recompute_on", prepare)
    monkeypatch.setattr(run_completion, "record_completion_on", record)

    async def scenario() -> None:
        try:
            with pytest.raises(ValueError) as caught:
                await server._recompute_context_request(
                    config=server._config,
                    since=None,
                    only_mode=None,
                    embed=False,
                    client=None,
                )
            assert caught.value is origin
        finally:
            await server.stop()

    asyncio.run(scenario())
    assert reports == 1
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone() == (
            datetime(2026, 11, 1),
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


def test_failed_rpc_completion_closes_the_parent_before_later_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recall.services import run_completion

    _, server, _, _ = _raw_setup(tmp_path)
    previous = server._database._connection
    assert previous is not None
    record = run_completion.record_completion_on
    origin = OSError("completion write failed")

    def fail(*args, **kwargs):
        def barrier(stage):
            if stage == "before_completion_commit":
                raise origin

        return record(*args, **kwargs, barrier=barrier)

    monkeypatch.setattr(run_completion, "record_completion_on", fail)

    async def scenario() -> None:
        try:
            with pytest.raises(OSError) as caught:
                await server._recompute_context_request(
                    config=server._config,
                    since=None,
                    only_mode=None,
                    embed=False,
                    client=None,
                )
            assert caught.value is origin
            with pytest.raises(duckdb.ConnectionException, match="closed"):
                previous.execute("SELECT 1")
            assert server._database.call(
                "after-resolution", lambda conn: conn.execute("SELECT 1").fetchone()
            ) == (1,)
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_failed_fresh_resolution_holds_admission_and_preserves_both_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recall.services import run_completion
    from recall.services.owned_jobs import WorkNotAdmitted

    database, server, _, _ = _raw_setup(tmp_path)
    record = run_completion.record_completion_on
    origin = OSError("completion write failed")
    diagnosis = OSError("fresh diagnostic read unavailable")

    def fail(*args, **kwargs):
        def barrier(stage):
            if stage == "before_completion_commit":
                raise origin

        return record(*args, **kwargs, barrier=barrier)

    def unreadable(read):
        raise diagnosis

    monkeypatch.setattr(run_completion, "record_completion_on", fail)
    monkeypatch.setattr(server._recovery_admission, "_read_retained", unreadable)

    async def scenario() -> None:
        try:
            with pytest.raises(BaseExceptionGroup) as caught:
                await server._recompute_context_request(
                    config=server._config,
                    since=None,
                    only_mode=None,
                    embed=False,
                    client=None,
                )
            assert caught.value.subgroup(lambda error: error is origin) is not None
            assert caught.value.subgroup(lambda error: error is diagnosis) is not None
            assert server._shutdown_event.is_set()
            assert (tmp_path / "daemon-refused").exists()
            with pytest.raises(WorkNotAdmitted):
                server._database.call(
                    "late-write", lambda conn: conn.execute("DELETE FROM runtime_state")
                )
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone() == (
            datetime(2026, 11, 1),
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)
