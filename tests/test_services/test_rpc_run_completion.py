"""Requested RPC work obtains release only from its actual settled scope."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

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
            "cancelled: the client that asked for it departed" if cancelled else "progress failed"
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


@pytest.mark.parametrize("kind", ["recompute", "index", "daemon"])
@pytest.mark.parametrize("admitted_first", [False, True])
def test_rpc_departure_orders_against_the_actual_completion_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    admitted_first: bool,
    kind: str,
) -> None:
    import threading

    from recall.services import run_completion

    database, server, _, _ = _inventory_server(tmp_path, monkeypatch, roots=False)
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

    async def requested():
        if kind == "recompute":
            return await server._recompute_context_request(
                config=server._config,
                since=None,
                only_mode=None,
                embed=False,
                client=None,
            )
        if kind == "index":
            from recall.core.types import Source
            from recall.services.coordinator import IndexRequestScope

            return await server._manual_index_request(
                config=server._config,
                source=Source.CODEX,
                client=None,
                scope=IndexRequestScope(),
            )
        return await server._handle_daemon_run({"once": True, "embed": False}, None)

    async def scenario() -> None:
        task = asyncio.create_task(requested())
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
            assert row == (
                datetime(2026, 11, 1),
                "cancelled: the client that asked for it departed",
            )
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


@pytest.mark.parametrize("daemon", [False, True])
def test_partial_rpc_inventory_never_records_a_successful_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    daemon: bool,
) -> None:
    from dataclasses import replace

    from recall.core.config import SourceConfig
    from recall.core.types import Source
    from recall.parsers.codex import CodexParser
    from recall.services.coordinator import IndexRequestScope

    database, server, _, _ = _raw_setup(tmp_path)
    config = replace(
        server._config,
        sources={
            name: SourceConfig(roots=(tmp_path,) if name == "codex" else ())
            for name in ("codex", "claude_code", "pi_agent", "grok", "kimi_code")
        },
        compaction=replace(server._config.compaction, auto_trigger=False),
    )
    server._config = config
    monkeypatch.setattr(server, "_load_runtime_config", lambda: config)

    def fail(*args, **kwargs):
        raise ValueError("captured transcript parse failed")

    monkeypatch.setattr(CodexParser, "parse", fail)

    async def scenario() -> None:
        try:
            if daemon:
                result = await server._handle_daemon_run({"once": True, "embed": False}, None)
                summary = result.index_summary
            else:
                summary = await server._manual_index_request(
                    config=config,
                    source=Source.CODEX,
                    client=None,
                    scope=IndexRequestScope(),
                )
            assert summary.failed == 1
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


def _inventory_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, roots: bool = True):
    from dataclasses import replace

    from recall.core.config import SourceConfig

    database, server, path, _ = _raw_setup(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    path = path.rename(root / path.name)
    from recall.parsers.codex import CodexParser

    parser = CodexParser(roots=(root,))
    config = replace(
        server._config,
        sources={
            name: SourceConfig(roots=(root,) if roots and name == "codex" else ())
            for name in ("codex", "claude_code", "pi_agent", "grok", "kimi_code")
        },
        compaction=replace(server._config.compaction, auto_trigger=False),
    )
    server._config = config
    monkeypatch.setattr(server, "_load_runtime_config", lambda: config)
    return database, server, path, parser


@pytest.mark.parametrize("daemon", [False, True])
def test_actual_empty_rpc_inventory_can_release_a_spent_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    daemon: bool,
) -> None:
    from recall.core.types import Source
    from recall.services.coordinator import IndexRequestScope

    database, server, _, _ = _inventory_server(tmp_path, monkeypatch, roots=False)

    async def scenario() -> None:
        try:
            if daemon:
                summary = (
                    await server._handle_daemon_run({"once": True, "embed": False}, None)
                ).index_summary
            else:
                summary = await server._manual_index_request(
                    config=server._config,
                    source=Source.CODEX,
                    client=None,
                    scope=IndexRequestScope(),
                )
            assert summary.total == 0 and summary.failed == 0
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = fresh.execute(
            "SELECT last_successful_at,last_index_total,last_failure_message FROM runtime_state"
        ).fetchone()
        assert row is not None and row[0] > datetime(2026, 11, 1) and row[1:] == (0, None)
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (True, True)


@pytest.mark.parametrize("failure", ["discovery", "usage"])
def test_zero_item_rpc_payload_does_not_hide_a_failed_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    from recall.core.types import Source
    from recall.parsers.discovery import DiscoveryScope
    from recall.services import usage_harvest
    from recall.services.coordinator import IndexRequestScope

    database, server, _, _ = _inventory_server(tmp_path, monkeypatch, roots=failure == "discovery")
    entries = DiscoveryScope._entries

    def unavailable(scope, directory, owner):
        if directory == tmp_path / "sessions":
            scope.fail(owner, PermissionError("selected root unavailable"))
            return None
        return entries(scope, directory, owner)

    def usage_unavailable(*args, **kwargs):
        raise OSError("selected usage capture unavailable")

    if failure == "discovery":
        monkeypatch.setattr(DiscoveryScope, "_entries", unavailable)
    else:
        monkeypatch.setattr(usage_harvest, "read_grok_usage_batch_on", usage_unavailable)

    async def scenario() -> None:
        try:
            summary = await server._manual_index_request(
                config=server._config,
                source=Source.CODEX,
                client=None,
                scope=IndexRequestScope(),
            )
            assert summary.total == 0
            assert server._jobs.first_fatal is None
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone()
        assert row is not None and row[0] == datetime(2026, 11, 1) and row[1]
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


def test_unchanged_rpc_scope_keeps_its_older_keyword_obligation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from recall.core.types import Source
    from recall.db.fts_sidecar import sidecar_path
    from recall.services.coordinator import IndexRequestScope

    database, server, _, _ = _inventory_server(tmp_path, monkeypatch)

    async def scenario() -> None:
        holder = sqlite3.connect(sidecar_path(tmp_path), isolation_level=None, timeout=0)
        try:
            holder.execute("BEGIN EXCLUSIVE")
            first = await server._manual_index_request(
                config=server._config,
                source=Source.CODEX,
                client=None,
                scope=IndexRequestScope(),
            )
            assert first.total == 1 and first.failed == 0
            holder.rollback()
            second = await server._manual_index_request(
                config=server._config,
                source=Source.CODEX,
                client=None,
                scope=IndexRequestScope(),
            )
            assert second.skipped == 1
        finally:
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


def test_rpc_captured_generation_keeps_its_receipt_when_a_later_append_defers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    from recall.core.types import RunKind, Source
    from recall.db.fts_sidecar import sidecar_path
    from recall.services.coordinator import IndexRequestScope, capture_path, observe_path

    database, server, path, parser = _inventory_server(tmp_path, monkeypatch)
    serve = server._serve_raw_turn
    injected = False
    earlier_success: datetime | None = None

    async def serve_with_later_append() -> bool:
        nonlocal injected, earlier_success
        worked = await serve()
        if worked and not injected:
            injected = True
            earlier = server._database.call(
                "fixture:earlier_success",
                lambda conn: conn.execute(
                    "SELECT last_successful_at FROM runtime_state"
                ).fetchone(),
            )
            assert earlier is not None
            earlier_success = earlier[0]
            with path.open("a") as stream:
                stream.write(
                    '{"type":"response_item","payload":{"type":"message","role":"user","content":"laterappend"}}\n'
                )
            await server._writer_call(
                lambda: server._database.call(
                    "fixture:later_observation",
                    lambda conn: observe_path(parser, capture_path(parser, path), conn=conn),
                ),
                "later source observation",
            )
            holder = sqlite3.connect(sidecar_path(tmp_path), isolation_level=None, timeout=0)
            try:
                holder.execute("BEGIN EXCLUSIVE")
                assert await serve()
            finally:
                holder.rollback()
                holder.close()
        return worked

    monkeypatch.setattr(server, "_serve_raw_turn", serve_with_later_append)

    async def scenario() -> None:
        try:
            summary = await server._manual_index_request(
                config=server._config,
                source=Source.CODEX,
                client=None,
                scope=IndexRequestScope(),
            )
            assert summary.indexed == 1 and summary.failed == 0
        finally:
            await server.stop()

    asyncio.run(scenario())
    assert injected and earlier_success is not None
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (1,)
        row = fresh.execute(
            "SELECT last_successful_at,last_failure_message,last_run_kind FROM runtime_state"
        ).fetchone()
        assert (
            row is not None and row[0] > earlier_success and row[1:] == (None, RunKind.INDEX.value)
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (True, True)


@pytest.mark.parametrize("daemon", [False, True])
@pytest.mark.parametrize("failure_class", ["other", "disk-full"])
def test_actual_idle_phase_noop_can_complete_after_diagnostic_folding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    daemon: bool,
    failure_class: Literal["other", "disk-full"],
) -> None:
    from dataclasses import replace
    from uuid import UUID

    from recall.core.config import SourceConfig
    from recall.core.types import RunKind, Source
    from recall.db.schema import ensure_schema
    from recall.services.daemon import run_startup_fts_sidecar_sync_on
    from recall.services.owned_jobs import DaemonJobs
    from recall.services.repair_events import EventFailure, RecoveryEvent, publish_event
    from recall.services.repair_protocol import RepairProtocol
    from recall.services.runtime_state import record_run_success

    from tests.test_services.test_rpc_owned_boundary import server_at

    database = tmp_path / "recall.duckdb"
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        record_run_success(conn, run_kind=RunKind.INDEX, successful_at=datetime(2026, 11, 1))
        publish_event(
            tmp_path,
            RecoveryEvent(
                event_id=UUID(int=100),
                failure=EventFailure(
                    site="diagnostic",
                    exception_type="Error",
                    message="retained diagnostic",
                    signature="diagnostic",
                    failure_class=failure_class,
                    at=datetime(2026, 11, 2),
                ),
                needs_index_verification=failure_class == "disk-full",
            ),
        )
        with jobs.lease(conn) as lease:
            RepairProtocol(tmp_path).begin_start(lease)
        assert conn.execute(
            "SELECT phase,success_since_failure FROM daemon_index_repair"
        ).fetchone() == ("idle", False)
    server = server_at(tmp_path)
    config = replace(
        server._config,
        sources={
            name: SourceConfig(roots=())
            for name in ("codex", "claude_code", "pi_agent", "grok", "kimi_code")
        },
        compaction=replace(server._config.compaction, auto_trigger=False),
    )
    server._config = config
    monkeypatch.setattr(server, "_load_runtime_config", lambda: config)
    outcome = server._database.run(
        "fixture:startup_keywords", lambda lease: run_startup_fts_sidecar_sync_on(lease, config)
    )
    server._record_keyword_sync(outcome, config)

    async def scenario() -> None:
        try:
            if daemon:
                summary = (
                    await server._handle_daemon_run({"once": True, "embed": False}, None)
                ).index_summary
            else:
                from recall.services.coordinator import IndexRequestScope

                summary = await server._manual_index_request(
                    config=config, source=Source.CODEX, client=None, scope=IndexRequestScope()
                )
            assert summary.total == 0 and summary.failed == 0
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        row = fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone()
        assert row is not None and row[0] > datetime(2026, 11, 1) and row[1] is None
        assert fresh.execute(
            "SELECT phase,success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == ("idle", True, True)


def test_rpc_unchanged_scope_can_complete_with_attributed_outside_keyword_backlog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3
    from dataclasses import replace

    from recall.core.config import SourceConfig
    from recall.core.types import Source
    from recall.db.fts_sidecar import sidecar_path
    from recall.parsers.codex import CodexParser
    from recall.services.coordinator import IndexRequestScope
    from recall.services.rpc_server import KeywordRetryGate

    from tests.test_services.test_raw_run_completion import _prepare

    database, server, path, parser = _raw_setup(tmp_path)
    now = [0.0]
    server._keyword_gate = KeywordRetryGate(clock=lambda: now[0])
    root = tmp_path / "selected"
    root.mkdir()
    selected = root / "rollout-selected.jsonl"
    selected.write_text(path.read_text().replace('"source"', '"selected"'))
    selected_parser = CodexParser(roots=(root,))
    config = replace(
        server._config,
        sources={
            name: SourceConfig(roots=(root,) if name == "codex" else ())
            for name in ("codex", "claude_code", "pi_agent", "grok", "kimi_code")
        },
        compaction=replace(server._config.compaction, auto_trigger=False),
    )
    server._config = config
    monkeypatch.setattr(server, "_load_runtime_config", lambda: config)

    async def scenario() -> None:
        try:
            holder = sqlite3.connect(sidecar_path(tmp_path), isolation_level=None, timeout=0)
            try:
                holder.execute("BEGIN EXCLUSIVE")
                server._commit_raw_sources(_prepare(server, parser, path), config)
            finally:
                holder.rollback()
                holder.close()
            now[0] = 10.0
            server._commit_raw_sources(_prepare(server, selected_parser, selected), config)
            before = server._database.call(
                "fixture:before_scope",
                lambda conn: conn.execute(
                    "SELECT last_successful_at FROM runtime_state"
                ).fetchone(),
            )
            summary = await server._manual_index_request(
                config=config, source=Source.CODEX, client=None, scope=IndexRequestScope()
            )
            assert summary.skipped == 1
            after = server._database.call(
                "fixture:after_scope",
                lambda conn: conn.execute(
                    "SELECT last_successful_at,last_failure_message FROM runtime_state"
                ).fetchone(),
            )
            assert (
                before is not None
                and after is not None
                and after[0] > before[0]
                and after[1] is None
            )
        finally:
            await server.stop()

    asyncio.run(scenario())
    with duckdb.connect(str(database), read_only=True) as fresh:
        assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (1,)
