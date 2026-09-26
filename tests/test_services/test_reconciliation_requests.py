"""Operator-requested cycles use the same captured inputs and writer as the daemon."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from lane_harness import Lane, build_lane
from recall.core.config import ContextConfig
from recall.core.rpc_types import APP_LOCKED, RpcError
from recall.core.types import RunKind, Source
from recall.db import open_sidecar, sidecar_path
from recall.db.source_files import SourceCatalog
from recall.parsers.claude_code import ClaudeCodeParser
from recall.services import embed_phase, runtime_state
from recall.services.context_backends import ContextResult
from recall.services.coordinator import set_paused
from recall.services.system_state import EmbedPreconditionResult


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Lane]:
    lane = build_lane(tmp_path, monkeypatch)
    server = lane.server
    server._config = replace(
        server._config,
        daemon=replace(server._config.daemon, embed_idle_session=0),
        compaction=replace(server._config.compaction, auto_trigger=False),
    )
    monkeypatch.setattr(
        "recall.services.system_state.check_power", lambda: EmbedPreconditionResult(ok=True)
    )
    monkeypatch.setattr(
        "recall.services.system_state.check_load", lambda _: EmbedPreconditionResult(ok=True)
    )
    yield lane
    asyncio.run(server.stop())


def test_once_reconciles_same_size_changes_and_a_second_cycle_is_idle(lane: Lane) -> None:
    path = lane.watched
    original_stat = path.stat()
    original = path.read_text()
    updated = original.replace("run the test suite", "run our test suite")
    assert updated != original and len(updated) == len(original)
    path.write_text(updated)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    server = lane.server

    async def scenario() -> None:
        result = await server._handle_daemon_run({"once": True, "embed": False}, None)
        assert result.index_summary.total == 2
        assert result.index_summary.indexed == 1
        assert server._get_conn().execute(
            "SELECT COUNT(*) FROM message_state WHERE content = 'run our test suite'"
        ).fetchone() == (1,)
        item = SourceCatalog(server._get_conn(), clock=time.time).get(
            Source.CLAUDE_CODE.value, str(path)
        )
        assert item is not None and item.current
        second = await server._handle_daemon_run({"once": True, "embed": False}, None)
        assert second.index_summary.indexed == 0
        assert second.index_summary.skipped == 2
        status = runtime_state.load_runtime_status_from_conn(server._get_conn())
        assert status.last_run_kind is RunKind.DAEMON_ONCE
        assert status.last_index_summary is not None and status.last_index_summary.skipped == 2

    asyncio.run(scenario())


def test_once_prepares_outside_writer_and_coalesces_with_fresh_request(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = lane.server
    lane.append(lane.watched, "new request content", uuid="request-content")
    entered, release = threading.Event(), threading.Event()
    parse = ClaudeCodeParser.parse
    calls = 0

    def blocked(self, path, **kwargs):
        nonlocal calls
        if path == lane.watched:
            calls += 1
            entered.set()
            assert release.wait(5)
        return parse(self, path, **kwargs)

    monkeypatch.setattr(ClaudeCodeParser, "parse", blocked)

    async def scenario() -> None:
        cycle = asyncio.create_task(server._handle_daemon_run({"once": True, "embed": False}, None))
        refresh = None
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert not server._write_lock.locked(), (
                "filesystem preparation owns the database writer"
            )
            refresh = asyncio.create_task(server.index_session_now(lane.watched))
            await server._writer_call(
                lambda: server._get_conn().execute("UPDATE session_state SET host = 'test-host'"),
                "another writer during requested capture",
            )
        finally:
            release.set()
            await cycle
            if refresh is not None:
                await refresh
        assert calls == 1
        assert server._get_conn().execute(
            "SELECT COUNT(*) FROM message_state WHERE content = 'new request content'"
        ).fetchone() == (1,)

    asyncio.run(scenario())


def test_once_embeds_outside_writer_and_preserves_embedding_options(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = lane.server
    for path in (lane.watched, lane.unwatched):
        os.utime(path, (1, 1))
    entered, release = threading.Event(), threading.Event()
    configs = []

    class Backend:
        dimensions = server._config.embedding.dimensions
        model_id = "request-verifier"
        query_prefix = ""

        def embed(self, texts: list[str]) -> list[list[float]]:
            entered.set()
            assert release.wait(5)
            return [[0.25] * self.dimensions for _ in texts]

    def backend(config):
        configs.append(config)
        return Backend()

    monkeypatch.setattr("recall.services.embeddings.get_backend", backend)

    async def scenario() -> None:
        cycle = asyncio.create_task(
            server._handle_daemon_run({"once": True, "embed": True, "batch_size": 32}, None)
        )
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert not server._write_lock.locked(), "model generation owns the database writer"
            lane.append(
                lane.watched, "raw advances during requested embedding", uuid="during-model"
            )
            await asyncio.wait_for(server.index_session_now(lane.watched), timeout=2)
            assert server._get_conn().execute(
                "SELECT COUNT(*) FROM message_state "
                "WHERE content = 'raw advances during requested embedding'"
            ).fetchone() == (1,)
        finally:
            release.set()
            await cycle
        assert configs and configs[0].batch_size == 32
        expected = server._config.embedding
        assert configs[0].backend == expected.backend
        assert configs[0].model == expected.model
        assert configs[0].context == expected.context
        assert configs[0].dimensions == expected.dimensions

    asyncio.run(scenario())


def test_once_honors_pause_before_any_request_write(lane: Lane) -> None:
    server = lane.server
    lane.append(lane.watched, "paused append", uuid="paused")
    before = runtime_state.load_runtime_status_from_conn(server._get_conn())
    set_paused(server._config, True)
    with pytest.raises(RpcError) as error:
        asyncio.run(server._handle_daemon_run({"once": True, "embed": False}, None))
    assert "paused" in error.value.message
    assert runtime_state.load_runtime_status_from_conn(server._get_conn()) == before
    assert server._get_conn().execute(
        "SELECT COUNT(*) FROM message_state WHERE content = 'paused append'"
    ).fetchone() == (0,)


@pytest.mark.parametrize("stage", ["attempt", "success"])
def test_once_runtime_metadata_recovers_aborted_transactions(
    lane: Lane, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    server = lane.server
    original = getattr(runtime_state, f"record_run_{stage}")
    attempts = 0

    def abort(conn, **kwargs):
        nonlocal attempts
        if kwargs["run_kind"] is RunKind.DAEMON_ONCE:
            attempts += 1
            if stage == "success" or attempts == 1:
                conn.execute("BEGIN")
                conn.execute("SELECT CAST('injected metadata failure' AS INTEGER)")
        return original(conn, **kwargs)

    monkeypatch.setattr(runtime_state, f"record_run_{stage}", abort)
    result = asyncio.run(server._handle_daemon_run({"once": True, "embed": False}, None))
    assert attempts == 2
    assert result.record_status_persisted is (stage == "attempt")
    assert server._get_conn().execute("SELECT 1").fetchone() == (1,)


@pytest.mark.parametrize("failure_record_aborts", [False, True])
def test_once_inventory_failure_recovers_before_failure_metadata_and_retry(
    lane: Lane, monkeypatch: pytest.MonkeyPatch, failure_record_aborts: bool
) -> None:
    server = lane.server
    # An unchanged corpus is compared in memory and never reaches the catalog
    # writer (REQ-INDEX-023), so the failure this test injects needs a real
    # observation to abort.
    lane.append(lane.watched, "inventory failure retry", uuid="inventory-failure-retry")
    original = SourceCatalog.observe_batch
    attempts = 0
    record_failure = runtime_state.record_run_failure
    failure_attempts = 0

    def record(conn, **kwargs):
        nonlocal failure_attempts
        failure_attempts += 1
        if failure_record_aborts and failure_attempts == 1:
            conn.execute("BEGIN")
            conn.execute("SELECT CAST('injected failure metadata error' AS INTEGER)")
        return record_failure(conn, **kwargs)

    def abort(self, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            self._conn.execute("BEGIN")
            self._conn.execute("SELECT CAST('injected inventory failure' AS INTEGER)")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SourceCatalog, "observe_batch", abort)
    monkeypatch.setattr(runtime_state, "record_run_failure", record)
    with pytest.raises(duckdb.ConversionException):
        asyncio.run(server._handle_daemon_run({"once": True, "embed": False}, None))
    status = runtime_state.load_runtime_status_from_conn(server._get_conn())
    assert failure_attempts == (2 if failure_record_aborts else 1)
    assert (
        status.last_failure_message and "injected inventory failure" in status.last_failure_message
    )
    result = asyncio.run(server._handle_daemon_run({"once": True, "embed": False}, None))
    assert result.index_summary.total == 2
    assert server._get_conn().execute("SELECT 1").fetchone() == (1,)


def test_once_embedding_commit_failure_preserves_raw_history_and_recovers(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = lane.server
    for path in (lane.watched, lane.unwatched):
        os.utime(path, (1, 1))

    class Backend:
        dimensions = server._config.embedding.dimensions
        model_id = "request-recovery-verifier"
        query_prefix = ""

        def embed(self, texts):
            return [[0.25] * self.dimensions for _ in texts]

    monkeypatch.setattr("recall.services.embeddings.get_backend", lambda _: Backend())
    commit = embed_phase.commit_prepared_embed_cycle

    def abort(_prepared, *, conn, **_kwargs):
        conn.execute("BEGIN")
        conn.execute("SELECT CAST('injected embedding commit failure' AS INTEGER)")

    monkeypatch.setattr(embed_phase, "commit_prepared_embed_cycle", abort)

    async def scenario() -> None:
        before = server._get_conn().execute("SELECT COUNT(*) FROM messages").fetchone()
        result = await server._handle_daemon_run({"once": True, "embed": True}, None)
        assert result.embed_summary is None
        assert server._embed_state is not None and server._embed_state.last_error
        assert "injected embedding commit failure" in server._embed_state.last_error
        assert server._get_conn().execute("SELECT COUNT(*) FROM messages").fetchone() == before
        assert server._get_conn().execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (
            0,
        )
        monkeypatch.setattr(embed_phase, "commit_prepared_embed_cycle", commit)
        result = await server._handle_daemon_run({"once": True, "embed": True}, None)
        assert result.embed_summary and result.embed_summary["embedded"] > 0
        assert server._embed_state.last_error is None

    asyncio.run(scenario())


def test_once_uses_configured_source_and_an_explicit_override(lane: Lane) -> None:
    server = lane.server
    server._config = replace(
        server._config, daemon=replace(server._config.daemon, source=Source.CLAUDE_CODE)
    )
    path = Path.home() / ".codex" / "sessions" / "rollout-requested.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "requested-codex"}})
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": "only requested source",
                },
            }
        )
        + "\n"
    )

    async def scenario() -> None:
        first = await server._handle_daemon_run({"once": True, "embed": False}, None)
        assert first.index_summary.total == 2
        assert server._get_conn().execute(
            "SELECT COUNT(*) FROM sessions WHERE source='codex'"
        ).fetchone() == (0,)
        second = await server._handle_daemon_run(
            {"once": True, "embed": False, "source": "codex"}, None
        )
        assert second.index_summary.total == 1 and second.index_summary.indexed == 1
        assert server._get_conn().execute(
            "SELECT COUNT(*) FROM sessions WHERE source='codex'"
        ).fetchone() == (1,)

    asyncio.run(scenario())


def test_once_reports_the_raw_context_mode_when_llm_work_is_deferred(lane: Lane) -> None:
    server = lane.server
    server._config = replace(
        server._config,
        embedding=replace(server._config.embedding, context=ContextConfig(mode="llm-local")),
    )
    lane.append(lane.watched, "raw context mode", uuid="raw-context-mode")
    result = asyncio.run(server._handle_daemon_run({"once": True, "embed": False}, None))
    assert result.index_summary.context_mode == "template"
    assert server._get_conn().execute(
        "SELECT context_mode FROM message_state WHERE content = 'raw context mode'"
    ).fetchone() == ("template",)


@pytest.mark.parametrize("change", ["content", "metadata", "cancel"])
def test_recompute_models_release_writer_and_reject_changed_inputs(
    lane: Lane, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    server = lane.server
    entered, release = threading.Event(), threading.Event()

    class Backend:
        def is_available(self):
            return True

        def generate_prefix(self, session, message):
            entered.set()
            assert release.wait(5)
            return ContextResult(prefix="[obsolete generated context] ", mode="llm-local")

    monkeypatch.setattr("recall.services.context_backends.get_context_backend", lambda _: Backend())

    async def scenario() -> None:
        task = asyncio.create_task(
            server._handle_index(
                {
                    "recompute_context": True,
                    "context": "llm-local",
                    "embed": False,
                },
                None,
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert not server._write_lock.locked(), "context model owns the raw writer"
            if change == "cancel":
                task.cancel()
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(task), timeout=0.02)
            else:
                statement = (
                    "UPDATE message_state SET content = 'changed input'"
                    if change == "content"
                    else "UPDATE session_state SET cwd = '/changed/input'"
                )
                await server._writer_call(
                    lambda: server._get_conn().execute(statement),
                    "raw input changes while context model runs",
                )
        finally:
            release.set()
            done = await asyncio.gather(task, return_exceptions=True)
        if change == "cancel":
            assert isinstance(done[0], asyncio.CancelledError)
        else:
            assert isinstance(done[0], RpcError)
            assert "changed" in done[0].message
        assert server._get_conn().execute(
            "SELECT COUNT(*) FROM message_state "
            "WHERE context_text = '[obsolete generated context] '"
        ).fetchone() == (0,)
        if change == "content":
            assert server._get_conn().execute(
                "SELECT DISTINCT content FROM message_state"
            ).fetchall() == [("changed input",)]
        assert (
            server._get_conn()
            .execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE '_recall_recompute_%'"
            )
            .fetchall()
            == []
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("embed", [False, True])
def test_recompute_preserves_non_targets_and_tool_vectors(
    lane: Lane, monkeypatch: pytest.MonkeyPatch, embed: bool
) -> None:
    server = lane.server
    conn = server._get_conn()
    target = conn.execute(
        "SELECT message_id FROM message_state WHERE content IS NOT NULL ORDER BY message_id LIMIT 1"
    ).fetchone()[0]
    conn.execute("UPDATE message_state SET context_text = '[keep] ', context_mode = 'llm-local'")
    conn.execute(
        "UPDATE message_state SET context_text = '', context_mode = 'off' WHERE message_id = ?",
        [target],
    )
    vector = [0.5] * server._config.embedding.dimensions
    conn.execute("UPDATE session_state SET git_repo = 'recomputesearchneedle'")
    for (message_id,) in conn.execute("SELECT id FROM messages").fetchall():
        conn.execute("INSERT INTO message_embeddings VALUES (?, ?, NULL)", [message_id, vector])
    for (tool_id,) in conn.execute("SELECT id FROM tool_calls").fetchall():
        conn.execute("INSERT INTO tool_call_embeddings VALUES (?, ?)", [tool_id, vector])
    before = conn.execute("SELECT * FROM tool_call_embeddings ORDER BY tool_call_id").fetchall()
    messages_before = conn.execute(
        "SELECT * FROM message_embeddings WHERE message_id != ? ORDER BY message_id", [target]
    ).fetchall()
    state_before = conn.execute(
        "SELECT * FROM message_state WHERE message_id != ? ORDER BY message_id", [target]
    ).fetchall()

    class Backend:
        dimensions = server._config.embedding.dimensions
        model_id = "recompute-vector-verifier"
        query_prefix = ""

        def embed(self, texts):
            return [[0.25] * self.dimensions for _ in texts]

    monkeypatch.setattr("recall.services.embeddings.get_backend", lambda _: Backend())
    assert conn.execute(
        "SELECT COUNT(*) FROM message_state WHERE context_mode = 'off' "
        "AND (content IS NOT NULL OR thinking IS NOT NULL)"
    ).fetchone() == (1,)
    result = asyncio.run(
        server._handle_index(
            {
                "recompute_context": True,
                "context": "template",
                "only_mode": "off",
                "embed": embed,
            },
            None,
        )
    )
    assert result.total == result.changed == result.context_messages == 1
    assert conn.execute(
        "SELECT context_mode FROM message_state WHERE message_id = ?", [target]
    ).fetchone() == ("template",)
    assert conn.execute(
        "SELECT DISTINCT context_text, context_mode FROM message_state WHERE message_id != ?",
        [target],
    ).fetchall() == [("[keep] ", "llm-local")]
    assert (
        conn.execute("SELECT * FROM tool_call_embeddings ORDER BY tool_call_id").fetchall()
        == before
    )
    assert (
        conn.execute(
            "SELECT * FROM message_embeddings WHERE message_id != ? ORDER BY message_id", [target]
        ).fetchall()
        == messages_before
    )
    assert (
        conn.execute(
            "SELECT * FROM message_state WHERE message_id != ? ORDER BY message_id", [target]
        ).fetchall()
        == state_before
    )
    target_vectors = conn.execute(
        "SELECT content_embedding FROM message_embeddings WHERE message_id = ?", [target]
    ).fetchall()
    assert len(target_vectors) == int(embed)
    if embed:
        assert list(target_vectors[0][0]) == [0.25] * server._config.embedding.dimensions
    with closing(open_sidecar(sidecar_path(server._config.data_dir))) as sidecar:
        assert sidecar.execute(
            "SELECT COUNT(*) FROM message_fts WHERE message_fts MATCH 'recomputesearchneedle'"
        ).fetchone() == (1,)
    assert conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone() == (0,)


def test_index_request_cut_short_records_why_it_failed(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RESIL-026: a failure stamp always carries a reason.

    A restart that interrupted an index request failed it with an `RpcError`,
    whose dataclass shape never populated `args` -- so `str(err)` was empty and
    the host recorded `last_failure_at` with `last_failure_message: ""`.
    """
    server = lane.server

    async def shutting_down(**_kwargs: object) -> None:
        raise RpcError(code=APP_LOCKED, message="reconciliation is paused or shutting down")

    monkeypatch.setattr(server, "_reconcile_index_request", shutting_down)

    with pytest.raises(RpcError):
        asyncio.run(server._handle_index({"embed": False}, None))

    status = runtime_state.load_runtime_status_from_conn(server._get_conn())
    assert status.last_failure_at is not None
    assert status.last_failure_message == "reconciliation is paused or shutting down"


def test_recompute_honors_pause_without_touching_context(lane: Lane) -> None:
    server = lane.server
    conn = server._get_conn()
    before = conn.execute("SELECT * FROM message_state ORDER BY message_id").fetchall()
    set_paused(server._config, True)
    with pytest.raises(RpcError) as error:
        asyncio.run(
            server._handle_index(
                {
                    "recompute_context": True,
                    "context": "template",
                    "embed": False,
                },
                None,
            )
        )
    assert "paused" in error.value.message
    assert conn.execute("SELECT * FROM message_state ORDER BY message_id").fetchall() == before


def test_recompute_recovers_an_aborted_snapshot_and_retries(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = lane.server
    prepare = embed_phase.prepare_context_recompute

    def abort(*_args, conn, **_kwargs):
        conn.execute("BEGIN")
        conn.execute("SELECT CAST('injected recompute snapshot failure' AS INTEGER)")

    monkeypatch.setattr(embed_phase, "prepare_context_recompute", abort)
    params = {"recompute_context": True, "context": "template", "embed": False}
    with pytest.raises(duckdb.ConversionException):
        asyncio.run(server._handle_index(params, None))
    conn = server._get_conn()
    assert conn.execute("SELECT 1").fetchone() == (1,)
    status = runtime_state.load_runtime_status_from_conn(conn)
    assert status.last_failure_message and "injected recompute" in status.last_failure_message
    assert (
        conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name LIKE '_recall_recompute_%'"
        ).fetchall()
        == []
    )
    monkeypatch.setattr(embed_phase, "prepare_context_recompute", prepare)
    result = asyncio.run(server._handle_index(params, None))
    assert result.changed > 0


def test_recompute_time_scope_uses_session_time(lane: Lane) -> None:
    server = lane.server
    conn = server._get_conn()
    session_id = conn.execute("SELECT id FROM sessions ORDER BY id LIMIT 1").fetchone()[0]
    conn.execute("UPDATE session_state SET started_at = '2000-01-01', ended_at = '2000-01-01'")
    conn.execute(
        "UPDATE session_state SET ended_at = current_timestamp WHERE session_id = ?", [session_id]
    )
    expected = conn.execute(
        "SELECT COUNT(*) FROM messages m JOIN message_state ms ON ms.message_id = m.id "
        "WHERE m.session_id = ? AND (ms.content IS NOT NULL OR ms.thinking IS NOT NULL)",
        [session_id],
    ).fetchone()[0]
    result = asyncio.run(
        server._handle_index(
            {
                "recompute_context": True,
                "context": "template",
                "embed": False,
                "since": "1h",
            },
            None,
        )
    )
    assert result.total == result.changed == expected > 0
    assert conn.execute(
        "SELECT DISTINCT ms.context_mode FROM messages m "
        "JOIN message_state ms ON ms.message_id=m.id "
        "WHERE m.session_id != ?",
        [session_id],
    ).fetchall() == [("off",)]


def test_recompute_reloads_a_changed_embedding_model(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = lane.server
    models: list[str] = []

    class Backend:
        dimensions = server._config.embedding.dimensions
        query_prefix = ""

        def __init__(self, model_id: str):
            self.model_id = model_id

        def embed(self, texts):
            return [[0.25] * self.dimensions for _ in texts]

    def get_backend(config):
        models.append(config.model)
        return Backend(config.model)

    monkeypatch.setattr("recall.services.embeddings.get_backend", get_backend)
    original_model = server._config.embedding.model
    for model in (original_model, "verifier-new-model"):
        server._config = replace(
            server._config, embedding=replace(server._config.embedding, model=model)
        )
        result = asyncio.run(
            server._handle_index(
                {
                    "recompute_context": True,
                    "context": "template",
                    "embed": True,
                },
                None,
            )
        )
        assert result.changed > 0
    assert models == [original_model, "verifier-new-model"]


def test_manual_index_captures_same_size_edits_and_full_advances_catalog(lane: Lane) -> None:
    server = lane.server
    path = lane.watched
    stat = path.stat()
    path.write_text(path.read_text().replace("run the test suite", "run our test suite"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    catalog = SourceCatalog(server._get_conn(), clock=time.time)

    async def scenario() -> None:
        result = await server._handle_index({"embed": False}, None)
        assert result.indexed == 1
        assert server._get_conn().execute(
            "SELECT COUNT(*) FROM message_state WHERE content = 'run our test suite'"
        ).fetchone() == (1,)
        before = catalog.get(Source.CLAUDE_CODE.value, str(path))
        assert before is not None and before.current
        full = await server._handle_index({"full": True, "embed": False}, None)
        assert full.indexed == 2
        after = catalog.get(Source.CLAUDE_CODE.value, str(path))
        assert after is not None and after.current
        assert after.committed_generation > before.committed_generation
        again = await server._handle_index({"embed": False}, None)
        assert again.indexed == 0 and again.skipped == 2

    asyncio.run(scenario())


def test_manual_index_honors_pause(lane: Lane) -> None:
    server = lane.server
    lane.append(lane.watched, "manual pause", uuid="manual-pause")
    set_paused(server._config, True)
    with pytest.raises(RpcError) as error:
        asyncio.run(server._handle_index({"full": True, "embed": False}, None))
    assert "paused" in error.value.message
    assert server._get_conn().execute(
        "SELECT COUNT(*) FROM message_state WHERE content = 'manual pause'"
    ).fetchone() == (0,)


def test_manual_root_host_and_full_scope_do_not_widen_the_request(
    lane: Lane, tmp_path: Path
) -> None:
    server = lane.server
    remote = tmp_path / "remote-host"
    path = remote / ".codex" / "sessions" / "rollout-remote.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "remote-native",
                    "cwd": "/srv/remote-project",
                    "git": {"repository_url": "example/remote-project"},
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": "remote transcript",
                },
            }
        )
        + "\n"
    )

    async def scenario() -> None:
        result = await server._handle_index(
            {
                "root": str(remote),
                "host": "imported-host",
                "source": "codex",
                "context": "template",
                "embed": False,
            },
            None,
        )
        assert result.total == result.indexed == 1
        row = (
            server._get_conn()
            .execute(
                "SELECT s.id, ss.host FROM sessions s JOIN session_state ss ON ss.session_id=s.id "
                "WHERE s.source_path = ?",
                [str(path)],
            )
            .fetchone()
        )
        assert row is not None and row[1] == "imported-host"
        item = SourceCatalog(server._get_conn(), clock=time.time).get("codex", str(path))
        assert item is not None and item.current and item.root_path == str(path.parent)
        lane.append(lane.watched, "local remains outside remote request", uuid="local-scope")
        second = await server._handle_index(
            {
                "root": str(remote),
                "host": "imported-host",
                "source": "codex",
                "full": True,
                "embed": False,
            },
            None,
        )
        assert second.total == second.indexed == 1
        assert server._get_conn().execute(
            "SELECT COUNT(*) FROM message_state "
            "WHERE content = 'local remains outside remote request'"
        ).fetchone() == (0,)

    asyncio.run(scenario())


def test_manual_preparation_and_models_release_the_shared_writer(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = lane.server
    entered, release = threading.Event(), threading.Event()

    class Backend:
        def is_available(self):
            return True

        def generate_prefix(self, session, message):
            entered.set()
            assert release.wait(5)
            return ContextResult(
                prefix="[manual context] ", mode="llm-local", input_tokens=3, output_tokens=2
            )

    monkeypatch.setattr("recall.services.context_backends.get_context_backend", lambda _: Backend())
    lane.append(lane.watched, "manual model request", uuid="manual-model-request")

    async def scenario() -> None:
        task = asyncio.create_task(
            server._handle_index(
                {
                    "context": "llm-local",
                    "embed": False,
                },
                None,
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert not server._write_lock.locked()
            lane.append(lane.unwatched, "independent raw request", uuid="independent-raw")
            await asyncio.wait_for(server.index_session_now(lane.unwatched), timeout=2)
        finally:
            release.set()
            result = await task
        assert result.context_messages > 0
        assert result.context_input_tokens > 0 and result.context_output_tokens > 0
        stats = (
            server._get_conn()
            .execute(
                "SELECT last_context_mode, last_context_messages, last_context_input_tokens, "
                "last_context_output_tokens FROM runtime_state"
            )
            .fetchone()
        )
        assert stats is not None and stats[0] == "llm-local"
        assert all(value > 0 for value in stats[1:])
        assert server._get_conn().execute(
            "SELECT COUNT(*) FROM message_state WHERE content = 'independent raw request'"
        ).fetchone() == (1,)
        assert server._get_conn().execute(
            "SELECT context_mode FROM message_state WHERE content = 'manual model request'"
        ).fetchone() == ("llm-local",)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["template", "off", "llm-local"])
def test_manual_full_retains_stored_llm_context_and_reports_reuse(
    lane: Lane, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    server = lane.server
    conn = server._get_conn()
    conn.execute(
        "UPDATE message_state SET context_text = '[stored llm] ', context_mode = 'llm-local' "
        "WHERE content IS NOT NULL OR thinking IS NOT NULL"
    )
    stored = conn.execute(
        "SELECT message_id, context_text, context_mode FROM message_state "
        "WHERE context_mode = 'llm-local' ORDER BY message_id"
    ).fetchall()
    expected = sum(row[2] == "llm-local" for row in stored)

    from recall.services.context_backends import get_context_backend

    def unexpected(config):
        if config.mode.startswith("llm-"):
            pytest.fail("equivalent stored context must not load a model")
        return get_context_backend(config)

    monkeypatch.setattr("recall.services.context_backends.get_context_backend", unexpected)
    result = asyncio.run(
        server._handle_index({"full": True, "context": mode, "embed": False}, None)
    )
    assert result.indexed == 2
    assert result.context_reused == expected
    assert (
        conn.execute(
            "SELECT message_id, context_text, context_mode FROM message_state "
            "WHERE context_mode = 'llm-local' ORDER BY message_id"
        ).fetchall()
        == stored
    )


def test_manual_since_and_project_filter_catalog_generations(lane: Lane) -> None:
    server = lane.server
    conn = server._get_conn()
    conn.execute(
        "UPDATE session_state SET git_repo = 'target-repository' WHERE session_id = "
        "(SELECT id FROM sessions WHERE source_path = ?)",
        [str(lane.watched)],
    )
    other = SourceCatalog(conn, clock=time.time).get(Source.CLAUDE_CODE.value, str(lane.unwatched))
    assert other is not None
    lane.append(lane.watched, "in scoped request", uuid="scope-in")
    lane.append(lane.unwatched, "outside scoped request", uuid="scope-out")
    os.utime(lane.watched, (1_789_000_000, 1_789_000_000))
    result = asyncio.run(
        server._handle_index(
            {"full": True, "since": "2026-01-01", "project": "target-repository", "embed": False},
            None,
        )
    )
    assert result.total == result.indexed == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM message_state WHERE content = 'in scoped request'"
    ).fetchone() == (1,)
    assert conn.execute(
        "SELECT COUNT(*) FROM message_state WHERE content = 'outside scoped request'"
    ).fetchone() == (0,)
    after = SourceCatalog(conn, clock=time.time).get(Source.CLAUDE_CODE.value, str(lane.unwatched))
    assert after == other
    os.utime(lane.watched, (1, 1))
    result = asyncio.run(
        server._handle_index(
            {"full": True, "since": "2026-01-01", "project": "target-repository", "embed": False},
            None,
        )
    )
    assert result.total == 0


def test_fresh_reconciliation_preserves_imported_host(lane: Lane) -> None:
    conn = lane.server._get_conn()
    conn.execute("UPDATE session_state SET host = 'imported-host'")
    lane.append(lane.watched, "new imported content", uuid="imported-append")
    asyncio.run(lane.server.index_session_now(lane.watched))
    assert conn.execute("SELECT DISTINCT host FROM session_state").fetchall() == [
        ("imported-host",)
    ]


def test_manual_inventory_failure_recovers_and_records_failure(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = lane.server
    # Only a source whose observation would change the catalog reaches the
    # per-path writer turn at all (REQ-INDEX-023).
    lane.append(lane.watched, "manual inventory retry", uuid="manual-inventory-retry")
    original = SourceCatalog.observe
    attempts = 0

    def abort(self, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            self._conn.execute("BEGIN")
            self._conn.execute("SELECT CAST('manual inventory failure' AS INTEGER)")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(SourceCatalog, "observe", abort)
    with pytest.raises(duckdb.ConversionException):
        asyncio.run(server._handle_index({"embed": False}, None))
    status = runtime_state.load_runtime_status_from_conn(server._get_conn())
    assert status.last_failure_message and "manual inventory failure" in status.last_failure_message
    assert status.last_run_kind is RunKind.INDEX
    result = asyncio.run(server._handle_index({"embed": False}, None))
    assert result.total == 2
    assert server._get_conn().execute("SELECT 1").fetchone() == (1,)


def test_recreate_keeps_consistent_backup_and_uses_catalog(lane: Lane) -> None:
    server = lane.server
    conn = server._get_conn()
    before = conn.execute(
        "SELECT message_id, content FROM message_state ORDER BY message_id"
    ).fetchall()
    lane.append(lane.watched, "recreated content", uuid="recreated-content")
    result = asyncio.run(
        server._handle_index(
            {"recreate": True, "confirmed": True, "context": "template", "embed": False}, None
        )
    )
    backup_path = getattr(result, "backup_path", None)
    assert backup_path is not None, (
        "recreate destroyed the prior database without reporting a backup"
    )
    with closing(
        duckdb.connect(str(Path(backup_path) / "recall.duckdb"), read_only=True)
    ) as backup:
        assert (
            backup.execute(
                "SELECT message_id, content FROM message_state ORDER BY message_id"
            ).fetchall()
            == before
        )
    assert (Path(backup_path) / "recall.fts.sqlite").is_file()
    assert result.indexed == result.total == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM source_files WHERE desired_generation = committed_generation"
    ).fetchone() == (2,)
    assert conn.execute(
        "SELECT COUNT(*) FROM message_state WHERE content = 'recreated content'"
    ).fetchone() == (1,)


def test_recreate_honors_pause_before_backup_or_deletion(lane: Lane) -> None:
    server = lane.server
    before = (
        server._get_conn()
        .execute("SELECT message_id, content FROM message_state ORDER BY message_id")
        .fetchall()
    )
    set_paused(server._config, True)
    with pytest.raises(RpcError) as error:
        asyncio.run(
            server._handle_index({"recreate": True, "confirmed": True, "embed": False}, None)
        )
    assert "paused" in error.value.message
    assert (
        server._get_conn()
        .execute("SELECT message_id, content FROM message_state ORDER BY message_id")
        .fetchall()
        == before
    )
    assert not (server._config.data_dir / "snapshots").exists()


def test_recreate_does_not_hold_writer_during_models(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = lane.server
    entered, release = threading.Event(), threading.Event()

    class Backend:
        def is_available(self):
            return True

        def generate_prefix(self, session, message):
            entered.set()
            assert release.wait(5)
            return ContextResult(prefix="[recreate context] ", mode="llm-local")

    monkeypatch.setattr("recall.services.context_backends.get_context_backend", lambda _: Backend())

    async def scenario() -> None:
        task = asyncio.create_task(
            server._handle_index(
                {"recreate": True, "confirmed": True, "context": "llm-local", "embed": False}, None
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert not server._write_lock.locked()
            assert await server._run_readonly(lambda conn: conn.execute("SELECT 1").fetchone()) == (
                1,
            )
        finally:
            release.set()
            await task

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["backup", "schema"])
def test_recreate_failure_retains_prior_rows_and_usable_connection(
    lane: Lane, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    server = lane.server
    conn = server._get_conn()
    before = conn.execute(
        "SELECT message_id, content FROM message_state ORDER BY message_id"
    ).fetchall()
    from recall.services import recreation

    if stage == "backup":

        def fail_copy(*_args):
            raise OSError("injected backup failure")

        monkeypatch.setattr(recreation, "_clone_or_copy", fail_copy)
        with pytest.raises(OSError, match="injected backup failure"):
            asyncio.run(
                server._handle_index({"recreate": True, "confirmed": True, "embed": False}, None)
            )
    else:

        def fail_schema(database, *_args):
            database.execute("SELECT CAST('injected schema failure' AS INTEGER)")

        monkeypatch.setattr(recreation, "_apply_schema", fail_schema)
        with pytest.raises(RpcError) as error:
            asyncio.run(
                server._handle_index({"recreate": True, "confirmed": True, "embed": False}, None)
            )
        assert error.value.data["backup_path"] in error.value.message
        assert (Path(error.value.data["backup_path"]) / "manifest.json").is_file()
    assert (
        conn.execute("SELECT message_id, content FROM message_state ORDER BY message_id").fetchall()
        == before
    )
    conn.execute("UPDATE session_state SET host = 'still-writable'")
    assert conn.execute("SELECT DISTINCT host FROM session_state").fetchall() == [
        ("still-writable",)
    ]


def test_recreate_invalidates_inventory_captured_before_reset(
    lane: Lane, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = lane.server
    from recall.services import coordinator
    from recall.services.reconciler import InventoryBatch

    prepare = coordinator.prepare_raw_cycle
    entered, release = threading.Event(), threading.Event()
    captured = False

    def paused_inventory(config, **kwargs):
        nonlocal captured
        for event in prepare(config, **kwargs):
            if isinstance(event.event, InventoryBatch) and not captured:
                captured = True
                entered.set()
                assert release.wait(5)
            yield event

    monkeypatch.setattr(coordinator, "prepare_raw_cycle", paused_inventory)

    async def scenario() -> None:
        task = asyncio.create_task(server._run_inventory_loop())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            result = await server._handle_index(
                {"recreate": True, "confirmed": True, "embed": False}, None
            )
            assert result.indexed == 2
            release.set()
            deadline = asyncio.get_running_loop().time() + 3
            while not server._inventory_complete and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.02)
            assert server._inventory_complete
            assert server._inventory_error is None
            assert server._get_conn().execute(
                "SELECT COUNT(*) FROM source_files "
                "WHERE missing OR desired_generation != committed_generation"
            ).fetchone() == (0,)
        finally:
            release.set()
            server._shutdown_event.set()
            await task

    asyncio.run(scenario())
