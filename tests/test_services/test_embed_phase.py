from __future__ import annotations

import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    ContextConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.models import Message, Session, TailFacts, ToolCall
from recall.core.types import DaemonMode, Role, Source
from recall.db.fts_sidecar import open_sidecar
from recall.db.schema import ensure_schema
from recall.services.embed_phase import (
    EmbedPhaseState,
    commit_prepared_embed_cycle,
    find_pending_embeds,
    generate_prepared_embed_cycle,
    prepare_embed_cycle,
)
from recall.services.fts_sidecar_reconcile import reconcile_sidecar
from recall.services.indexer import SessionContextStats, index_sessions


def _app_config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(mode=DaemonMode.POLL),
        cli=CliConfig(),
    )


class RecordingBackend:
    dimensions = 384
    model_id = "test-model"
    query_prefix = ""

    def __init__(self, value: float = 1.0) -> None:
        self.calls: list[list[str]] = []
        self.value = value

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[self.value] * self.dimensions for _ in texts]


class FakeMlxCoreModule(types.ModuleType):
    def clear_cache(self) -> None:
        return None


class FakeMlxModule(types.ModuleType):
    core: FakeMlxCoreModule


class TestFindPendingEmbeds:
    def test_respects_idle_threshold(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        codex_dir = tmp_path / ".codex" / "sessions" / "s1"
        codex_dir.mkdir(parents=True)
        fixture = (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "codex"
            / "session1"
            / "rollout.jsonl"
        )
        dest = codex_dir / "rollout.jsonl"
        shutil.copy(fixture, dest)

        index_sessions(source=None, full=False, recreate=True, verbose=False)

        config = _app_config(tmp_path)
        conn = duckdb.connect(str(config.db_path), read_only=True)
        try:
            # Very large threshold = no sessions qualify as idle
            result = find_pending_embeds(conn, idle_threshold=999999)
            assert result.message_count == 0
        finally:
            conn.close()

    def test_orders_oldest_idle_session_first(self) -> None:
        from recall.services.indexer import _write_session

        conn = duckdb.connect(":memory:")
        ensure_schema(conn)
        for session_id, file_mtime in (
            ("newest", 300.0),
            ("oldest", 100.0),
            ("middle", 200.0),
        ):
            session = Session(
                id=session_id,
                source=Source.CODEX,
                source_path=f"/tmp/{session_id}.jsonl",
                file_mtime=file_mtime,
                file_size=1,
                messages=[
                    Message(
                        id=f"{session_id}-m0",
                        session_id=session_id,
                        idx=0,
                        role=Role.ASSISTANT,
                        content=f"content {session_id}",
                    )
                ],
                message_count=1,
            )
            _write_session(conn, session, tail_facts=TailFacts())
        try:
            result = find_pending_embeds(conn, idle_threshold=0)
            assert result.session_ids == ["oldest", "middle", "newest"]
        finally:
            conn.close()


def test_embed_cycle_uses_each_message_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recall.services.indexer import _write_session
    from recall.services.system_state import EmbedPreconditionResult

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    session = Session(
        id="mixed-context-session",
        source=Source.CODEX,
        source_path="/tmp/mixed-context.jsonl",
        file_mtime=1.0,
        file_size=1,
        messages=[
            Message(
                id="off-message",
                session_id="mixed-context-session",
                idx=0,
                role=Role.USER,
                content="hello",
            ),
            Message(
                id="template-message",
                session_id="mixed-context-session",
                idx=1,
                role=Role.ASSISTANT,
                content="world",
            ),
        ],
        message_count=2,
    )
    _write_session(conn, session, tail_facts=TailFacts())
    conn.execute(
        """
        UPDATE message_state
        SET
            context_text = ?,
            context_mode = ?,
            fts_content = ? || COALESCE(content, ''),
            fts_thinking = ? || COALESCE(thinking, '')
        WHERE message_id = ?
        """,
        ["[repo branch] ", "template", "[repo branch] ", "[repo branch] ", "template-message"],
    )
    backend = RecordingBackend()
    state = EmbedPhaseState(_config=config)
    mlx_module = FakeMlxModule("mlx")
    mlx_core_module = FakeMlxCoreModule("mlx.core")
    mlx_module.core = mlx_core_module
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core_module)
    monkeypatch.setattr(state, "load_backend", lambda: backend)
    monkeypatch.setattr(
        "recall.services.system_state.check_power",
        lambda: EmbedPreconditionResult(ok=True),
    )
    monkeypatch.setattr(
        "recall.services.system_state.check_load",
        lambda _threshold: EmbedPreconditionResult(ok=True),
    )

    import asyncio

    from recall.services.rpc_server import RpcServer

    server = RpcServer(config)
    server._conn = conn
    embedded = asyncio.run(server._run_embed_batch(config, state))

    assert embedded == 2
    assert backend.calls == [["hello", "[repo branch] world"]]


def _idle_session(session_id: str, content: str) -> Session:
    return Session(
        id=session_id,
        source=Source.CODEX,
        source_path=f"/tmp/{session_id}.jsonl",
        file_mtime=1.0,
        file_size=1,
        messages=[
            Message(
                id=f"{session_id}-m0",
                session_id=session_id,
                idx=0,
                role=Role.ASSISTANT,
                content=content,
            )
        ],
        message_count=1,
    )


def _allow_embedding(monkeypatch: pytest.MonkeyPatch, state: EmbedPhaseState, backend: Any) -> None:
    from recall.services.system_state import EmbedPreconditionResult

    mlx_module = FakeMlxModule("mlx")
    mlx_core_module = FakeMlxCoreModule("mlx.core")
    mlx_module.core = mlx_core_module
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core_module)
    monkeypatch.setattr(state, "load_backend", lambda: backend)
    monkeypatch.setattr(
        "recall.services.system_state.check_power",
        lambda: EmbedPreconditionResult(ok=True),
    )
    monkeypatch.setattr(
        "recall.services.system_state.check_load",
        lambda _threshold: EmbedPreconditionResult(ok=True),
    )


def test_staleness_check_compares_inputs_by_identity_not_row_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staleness check compares the same input generation by identity, not
    by incidental row order.

    tool_calls.idx is not unique within a session, so ORDER BY idx leaves ties
    whose physical order is plan-dependent. A snapshot whose tool inputs are
    arranged in a different tie order than the commit-time query is still the
    same generation and must match.
    """
    from recall.services.embed_phase import PreparedEmbedSession, _inputs_still_match
    from recall.services.indexer import _write_session

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    session = _idle_session("ties", "body")
    for position in range(3):
        session.messages[0].tool_calls.append(
            ToolCall(
                id=f"ties-t{position}",
                session_id=session.id,
                message_id="ties-m0",
                idx=0,  # shared idx: ORDER BY idx cannot order these rows
                tool_name="Read",
                tool_input={"path": f"f{position}"},
            )
        )
    _write_session(conn, session, tail_facts=TailFacts())
    state = EmbedPhaseState(_config=config)
    prepared = prepare_embed_cycle(config, state, conn=conn)
    item = prepared.sessions[0]

    # Rebuild the snapshot's tool inputs in a rotated version of the
    # commit-time query's tie order: identical rows, different incidental order.
    db_id_order = [
        str(row[0])
        for row in conn.execute(
            "SELECT id FROM tool_calls WHERE session_id = ? ORDER BY idx", [session.id]
        ).fetchall()
    ]
    rotated_order = db_id_order[1:] + db_id_order[:1]
    assert rotated_order != db_id_order
    by_id = {entry.tool_call_id: entry for entry in item.tool_inputs}
    shuffled = PreparedEmbedSession(
        session=item.session,
        source=item.source,
        inputs=item.inputs,
        tool_inputs=tuple(by_id[tool_id] for tool_id in rotated_order),
        context_targets=item.context_targets,
        context_metadata=item.context_metadata,
        embedding_targets=item.embedding_targets,
    )

    assert _inputs_still_match(conn, shuffled)


def test_session_with_duplicate_tool_idx_drains_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session written with duplicate tool_calls.idx values is the same
    generation at snapshot and commit, so it must be committable."""
    from recall.services.embed_phase import publish_prepared_embed_cycle
    from recall.services.indexer import _write_session

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    session = _idle_session("dup-idx", "body")
    for position in range(4):
        session.messages[0].tool_calls.append(
            ToolCall(
                id=f"dup-idx-t{position}",
                session_id=session.id,
                message_id="dup-idx-m0",
                idx=position % 2,  # two idx values shared by two tool calls each
                tool_name="Read",
                tool_input={"path": f"f{position}"},
            )
        )
    _write_session(conn, session, tail_facts=TailFacts())
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())

    prepared = prepare_embed_cycle(config, state, conn=conn)
    generate_prepared_embed_cycle(config, state, prepared)
    drained = publish_prepared_embed_cycle(prepared, config, conn=conn)

    assert drained == 5  # 1 message + 4 tool calls
    assert find_pending_embeds(conn, idle_threshold=0).total == 0


def test_uncommittable_session_is_deprioritized_and_drain_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One session that can never commit must not stop the drain.

    After repeated commit rejections the offender is skipped for a bounded
    cooldown, healthy sessions keep draining, and the skip is visible in the
    phase state.
    """
    import asyncio

    import recall.services.embed_phase as embed_phase
    from recall.services.indexer import _write_session
    from recall.services.rpc_server import RpcServer

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("poisoned", "stale"), tail_facts=TailFacts())
    _write_session(conn, _idle_session("healthy-1", "one"), tail_facts=TailFacts())
    _write_session(conn, _idle_session("healthy-2", "two"), tail_facts=TailFacts())
    # The poisoned session heads the oldest-first roster.
    conn.execute("UPDATE session_state SET file_mtime = 0.5 WHERE session_id = 'poisoned'")
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())

    real_check = embed_phase._inputs_still_match

    def reject_poisoned(check_conn: duckdb.DuckDBPyConnection, item: Any) -> bool:
        if item.session.id == "poisoned":
            return False
        return real_check(check_conn, item)

    monkeypatch.setattr(embed_phase, "_inputs_still_match", reject_poisoned)

    server = RpcServer(config)
    server._conn = conn

    first = asyncio.run(server._run_embed_batch(config, state))
    assert first == 0
    assert find_pending_embeds(conn, idle_threshold=0).total == 3

    # Expire the no-progress backoff so the next cycle runs immediately.
    state.stalled_until = time.monotonic() - 1
    second = asyncio.run(server._run_embed_batch(config, state))
    assert second == 0
    assert "poisoned" in state.cooled_session_ids()
    cooldown_count, cooldown_until = state.cooldown_status()
    assert cooldown_count == 1
    assert cooldown_until is not None and cooldown_until > time.time()

    # Healthy sessions drain around the deprioritized one.
    assert asyncio.run(server._run_embed_batch(config, state)) == 1
    assert asyncio.run(server._run_embed_batch(config, state)) == 1
    remaining = find_pending_embeds(conn, idle_threshold=0)
    assert remaining.session_ids == ["poisoned"]
    assert conn.execute(
        "SELECT COUNT(*) FROM message_embeddings WHERE message_id LIKE 'healthy-%'"
    ).fetchone() == (2,)

    # With only the cooled-down session left, the loop reports the wait instead
    # of spinning the model.
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.last_outcome == "cooldown-wait"
    assert "poisoned" in state.cooled_session_ids()


def test_cooldown_counts_rejections_per_session_not_the_pending_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Escalation follows the session, not the backlog it sits in.

    The pending signature carries global window totals, so on a live host every
    new idle session moves it between cycles. Rejections are counted per
    session, so the session that keeps failing its commit is the session that
    gets deprioritized — and a session that never failed one never is.
    """
    import asyncio

    import recall.services.embed_phase as embed_phase
    from recall.services.indexer import _write_session
    from recall.services.rpc_server import RpcServer

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("poisoned", "stale"), tail_facts=TailFacts())
    _write_session(conn, _idle_session("healthy", "one"), tail_facts=TailFacts())
    # The poisoned session heads the oldest-first roster.
    conn.execute("UPDATE session_state SET file_mtime = 0.5 WHERE session_id = 'poisoned'")
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())

    real_check = embed_phase._inputs_still_match

    def reject_poisoned(check_conn: duckdb.DuckDBPyConnection, item: Any) -> bool:
        if item.session.id == "poisoned":
            return False
        return real_check(check_conn, item)

    monkeypatch.setattr(embed_phase, "_inputs_still_match", reject_poisoned)

    server = RpcServer(config)
    server._conn = conn

    # Cycle 1: the head session's result is discarded at commit time.
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.cooled_session_ids() == frozenset()
    signature_one = state.stalled_pending_signature

    # A new session crosses the idle threshold, changing the global pending
    # counts and therefore the signature the stall pacing keys on.
    _write_session(conn, _idle_session("newcomer", "two"), tail_facts=TailFacts())
    assert find_pending_embeds(conn, idle_threshold=0).total == 3

    # Expire the no-progress backoff so the next cycle runs, as a real loop
    # would after daemon.embed_backoff.
    state.stalled_until = time.monotonic() - 1

    # Cycle 2: the same session is rejected again. The changed signature must
    # not lose the rejection the session already carries.
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.stalled_pending_signature != signature_one, (
        "precondition: the pending signature changed between cycles"
    )
    assert state.cooled_session_ids() == frozenset({"poisoned"})
    cooldown_count, cooldown_until = state.cooldown_status()
    assert cooldown_count == 1
    assert cooldown_until is not None and cooldown_until > time.time()

    # Healthy sessions drain around the deprioritized one without ever being
    # deprioritized themselves, and the skip stays visible through the state
    # the status payload reads.
    assert asyncio.run(server._run_embed_batch(config, state)) == 1
    assert asyncio.run(server._run_embed_batch(config, state)) == 1
    assert find_pending_embeds(conn, idle_threshold=0).session_ids == ["poisoned"]
    assert state.cooled_session_ids() == frozenset({"poisoned"})


def test_commit_rejection_count_resets_when_the_session_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed session starts over.

    A rejection followed by a commit of the same session must clear its count,
    and a rejection of one session must never escalate another.
    """
    import asyncio

    import recall.services.embed_phase as embed_phase
    from recall.services.indexer import _write_session
    from recall.services.rpc_server import RpcServer

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("flaky", "stale"), tail_facts=TailFacts())
    _write_session(conn, _idle_session("healthy", "one"), tail_facts=TailFacts())
    conn.execute("UPDATE session_state SET file_mtime = 0.5 WHERE session_id = 'flaky'")
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())

    real_check = embed_phase._inputs_still_match
    rejected: set[str] = {"flaky"}

    def reject_selected(check_conn: duckdb.DuckDBPyConnection, item: Any) -> bool:
        if item.session.id in rejected:
            return False
        return real_check(check_conn, item)

    monkeypatch.setattr(embed_phase, "_inputs_still_match", reject_selected)

    server = RpcServer(config)
    server._conn = conn

    # Cycle 1: the head session is rejected once.
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.cooled_session_ids() == frozenset()

    # Cycle 2: the same session now commits, which clears its count.
    rejected.clear()
    state.stalled_until = time.monotonic() - 1
    assert asyncio.run(server._run_embed_batch(config, state)) == 1

    # A single rejection of the other session must not deprioritize either one.
    rejected.add("healthy")
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.cooled_session_ids() == frozenset()

    # The backstop still fires once that session's own count reaches the
    # threshold, and it takes only that session.
    state.stalled_until = time.monotonic() - 1
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.cooled_session_ids() == frozenset({"healthy"})

    # The session that drained earlier must need two fresh rejections of its
    # own, not one, to be deprioritized.
    rejected.add("flaky")
    _write_session(conn, _idle_session("flaky", "changed"), tail_facts=TailFacts())
    conn.execute("UPDATE session_state SET file_mtime = 0.5 WHERE session_id = 'flaky'")
    assert find_pending_embeds(conn, idle_threshold=0).session_ids[0] == "flaky", (
        "precondition: the re-written session heads the roster again"
    )
    state.stalled_until = time.monotonic() - 1
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.cooled_session_ids() == frozenset({"healthy"})


def test_backoff_pacing_tick_is_not_a_commit_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cycle short-circuited by the no-progress backoff is pacing, not evidence.

    It attempts no commit and discards no result, so counting it would make a
    threshold of two fire on one real rejection plus one idle timer tick
    (`embed_interval` is shorter than `embed_backoff`, so that tick always
    arrives), and re-marking the stall would push the deadline forward forever.
    """
    import asyncio

    import recall.services.embed_phase as embed_phase
    from recall.services.indexer import _write_session
    from recall.services.rpc_server import RpcServer

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("poisoned", "stale"), tail_facts=TailFacts())
    _write_session(conn, _idle_session("healthy", "one"), tail_facts=TailFacts())
    conn.execute("UPDATE session_state SET file_mtime = 0.5 WHERE session_id = 'poisoned'")
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())

    real_check = embed_phase._inputs_still_match

    def reject_poisoned(check_conn: duckdb.DuckDBPyConnection, item: Any) -> bool:
        if item.session.id == "poisoned":
            return False
        return real_check(check_conn, item)

    monkeypatch.setattr(embed_phase, "_inputs_still_match", reject_poisoned)

    server = RpcServer(config)
    server._conn = conn

    # Cycle 1: one real commit attempt whose result the staleness check discards.
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.cooled_session_ids() == frozenset()
    paced_until = state.stalled_until

    # Cycle 2: the backoff has not expired, so the cycle short-circuits before
    # generating or committing anything.
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.last_outcome == "stalled"
    assert state.cooled_session_ids() == frozenset()
    assert state.stalled_until == paced_until

    # Cycle 3: the backoff expired, so a second real rejection deprioritizes it.
    state.stalled_until = time.monotonic() - 1
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert "poisoned" in state.cooled_session_ids()


def test_stall_pacing_holds_when_a_cooled_session_heads_the_backlog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stall signature must be read through the cooldown filter.

    A deprioritized session keeps its place in the raw oldest-first order, so a
    signature taken from the unfiltered roster can never match the next cycle's
    filtered snapshot: the backoff is bypassed and every tick reloads the model.
    """
    import asyncio

    import recall.services.embed_phase as embed_phase
    from recall.services.indexer import _write_session
    from recall.services.rpc_server import RpcServer

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    for session_id, content in (("oldest", "a"), ("poisoned", "b"), ("healthy", "c")):
        _write_session(conn, _idle_session(session_id, content), tail_facts=TailFacts())
    conn.execute("UPDATE session_state SET file_mtime = 0.2 WHERE session_id = 'oldest'")
    conn.execute("UPDATE session_state SET file_mtime = 0.5 WHERE session_id = 'poisoned'")
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())

    real_check = embed_phase._inputs_still_match

    def reject_selected(check_conn: duckdb.DuckDBPyConnection, item: Any) -> bool:
        if item.session.id in {"oldest", "poisoned"}:
            return False
        return real_check(check_conn, item)

    monkeypatch.setattr(embed_phase, "_inputs_still_match", reject_selected)

    server = RpcServer(config)
    server._conn = conn

    # Two real rejections deprioritize the head session.
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    state.stalled_until = time.monotonic() - 1
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert "oldest" in state.cooled_session_ids()

    # The next session in the filtered roster now makes no progress either.
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.last_outcome == "no-progress"

    # The tick that follows must pace on that stall, not regenerate the batch.
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.last_outcome == "stalled"


def test_cooldown_state_is_safe_to_read_while_a_cycle_mutates_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status reader and the embed cycle touch the cooldown map concurrently.

    `daemon status` reads it on the event loop while the embed snapshot prunes
    it on the writer executor, so an unguarded dict raises "dictionary changed
    size during iteration" or a KeyError on the expired entries both drop.
    """
    import sys
    import threading

    import recall.services.embed_phase as embed_phase

    config = _app_config(tmp_path)
    state = EmbedPhaseState(_config=config)
    monkeypatch.setattr(embed_phase, "SESSION_COOLDOWN_LIMIT", 500)
    monkeypatch.setattr(embed_phase, "SESSION_REJECTION_LIMIT", 500)
    session_ids = [f"s{index}" for index in range(400)]
    failures: list[Exception] = []
    stop = threading.Event()

    def read_status() -> None:
        try:
            while not stop.is_set():
                state.cooldown_status()
                state.cooled_session_ids()
        except Exception as err:
            failures.append(err)

    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # widen the window both threads interleave in
    reader = threading.Thread(target=read_status)
    reader.start()
    try:
        for _ in range(200):
            # Two rejections deprioritize the whole roster; the short cooldown
            # and the reset then make the next round rebuild it, so the reader
            # meets a map that is growing, shrinking, and being pruned.
            state.note_commit_rejections(committed=(), rejected=session_ids, cooldown=0.001)
            state.note_commit_rejections(committed=(), rejected=session_ids, cooldown=0.001)
            state.cooled_session_ids()
            state.clear_cooldowns()
    except Exception as err:
        failures.append(err)
    finally:
        stop.set()
        reader.join(timeout=30)
        sys.setswitchinterval(switch_interval)

    assert not reader.is_alive()
    assert failures == []


def test_backend_change_clears_cooldowns_with_the_stall_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different model is a different generation of work: nothing carries over."""
    import asyncio
    from dataclasses import replace

    import recall.services.embed_phase as embed_phase
    from recall.services.indexer import _write_session
    from recall.services.rpc_server import RpcServer

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("poisoned", "stale"), tail_facts=TailFacts())
    _write_session(conn, _idle_session("healthy", "one"), tail_facts=TailFacts())
    conn.execute("UPDATE session_state SET file_mtime = 0.5 WHERE session_id = 'poisoned'")
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())

    real_check = embed_phase._inputs_still_match

    def reject_poisoned(check_conn: duckdb.DuckDBPyConnection, item: Any) -> bool:
        if item.session.id == "poisoned":
            return False
        return real_check(check_conn, item)

    monkeypatch.setattr(embed_phase, "_inputs_still_match", reject_poisoned)

    server = RpcServer(config)
    server._conn = conn

    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    state.stalled_until = time.monotonic() - 1
    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert "poisoned" in state.cooled_session_ids()

    state.configure(replace(config, embedding=EmbeddingConfig(model="other-model")))

    assert state.cooled_session_ids() == frozenset()
    assert state.cooldown_status() == (0, None)
    assert state.stalled_pending_signature is None


def test_empty_backlog_clears_stall_state_when_a_pause_lands_mid_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pause must not strand stall state the empty backlog would have cleared.

    The pause is re-checked after the snapshot, so an operator pausing during a
    cycle used to leave a stall signature no later cycle can match.
    """
    import asyncio

    from recall.services.embed_phase import PendingEmbeds
    from recall.services.rpc_server import RpcServer

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())
    state.mark_stalled(PendingEmbeds(1, 0, ["gone"]), 300.0)

    checks = 0

    def pause_after_the_snapshot(_config: AppConfig) -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    monkeypatch.setattr("recall.services.coordinator.is_paused", pause_after_the_snapshot)

    server = RpcServer(config)
    server._conn = conn

    assert asyncio.run(server._run_embed_batch(config, state)) == 0
    assert state.last_outcome == "paused"
    assert state.stalled_pending_signature is None
    assert state.stalled_until == 0.0


def test_prepared_generation_rejects_raw_rewrite_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model result can never overwrite a newer raw input generation."""
    from recall.services.indexer import _write_session

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("race", "before"), tail_facts=TailFacts())
    state = EmbedPhaseState(_config=config)
    backend = RecordingBackend()
    _allow_embedding(monkeypatch, state, backend)

    prepared = prepare_embed_cycle(config, state, conn=conn)
    # This is the raw writer's commit while model work is still running.
    conn.execute("UPDATE message_state SET content = 'after' WHERE message_id = 'race-m0'")
    generate_prepared_embed_cycle(config, state, prepared)

    assert commit_prepared_embed_cycle(prepared, conn=conn) == 0
    assert conn.execute(
        "SELECT content FROM message_state WHERE message_id = 'race-m0'"
    ).fetchone() == ("after",)
    assert conn.execute("SELECT count(*) FROM message_embeddings").fetchone() == (0,)


def test_prepared_commit_publishes_matching_effective_context_and_fts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services.indexer import _write_session

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("effective", "body"), tail_facts=TailFacts())
    conn.execute(
        """
        UPDATE message_state SET context_text = '[repo main] ', context_mode = 'template'
        WHERE message_id = 'effective-m0'
        """
    )
    state = EmbedPhaseState(_config=config)
    backend = RecordingBackend()
    _allow_embedding(monkeypatch, state, backend)

    prepared = prepare_embed_cycle(config, state, conn=conn)
    generate_prepared_embed_cycle(config, state, prepared)

    assert commit_prepared_embed_cycle(prepared, conn=conn) == 1
    assert backend.calls == [["[repo main] body"]]
    assert conn.execute(
        "SELECT context_text, fts_content FROM message_state WHERE message_id = 'effective-m0'"
    ).fetchone() == ("[repo main] ", "[repo main] body")


def test_prepared_stop_prevents_publication_and_repeat_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services.indexer import _write_session

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("stop", "body"), tail_facts=TailFacts())
    state = EmbedPhaseState(_config=config)
    backend = RecordingBackend()
    _allow_embedding(monkeypatch, state, backend)

    prepared = prepare_embed_cycle(config, state, conn=conn)
    generate_prepared_embed_cycle(config, state, prepared)
    assert commit_prepared_embed_cycle(prepared, conn=conn, should_stop=lambda: True) == 0
    assert find_pending_embeds(conn, idle_threshold=0).total == 1
    assert commit_prepared_embed_cycle(prepared, conn=conn) == 1
    assert prepare_embed_cycle(config, state, conn=conn).sessions == []


def test_prepared_generation_failure_is_returned_without_touching_raw_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services.indexer import _write_session

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("failure", "raw survives"), tail_facts=TailFacts())
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())
    monkeypatch.setattr(
        state, "load_backend", lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )

    prepared = generate_prepared_embed_cycle(
        config, state, prepare_embed_cycle(config, state, conn=conn)
    )

    assert prepared.error == "RuntimeError: offline"
    assert commit_prepared_embed_cycle(prepared, conn=conn) == 0
    assert conn.execute(
        "SELECT content FROM message_state WHERE message_id = 'failure-m0'"
    ).fetchone() == ("raw survives",)


@pytest.mark.parametrize(
    ("statement", "params"),
    [
        ("UPDATE message_state SET role = ? WHERE message_id = ?", ["user", "provenance-m0"]),
        ("UPDATE messages SET agent_id = ? WHERE id = ?", ["subagent-2", "provenance-m0"]),
        ("UPDATE tool_calls SET tool_name = ? WHERE id = ?", ["Write", "provenance-t0"]),
        (
            "UPDATE tool_calls SET tool_input = ? WHERE id = ?",
            ['{"path":"changed"}', "provenance-t0"],
        ),
    ],
)
def test_prepared_generation_discards_changed_context_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    statement: str,
    params: list[str],
) -> None:
    """A detached context result cannot publish after any rendered input changes."""
    from recall.services.indexer import _write_session

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    session = _idle_session("provenance", "body")
    session.messages[0].tool_calls.append(
        ToolCall(
            id="provenance-t0",
            session_id=session.id,
            message_id="provenance-m0",
            idx=0,
            tool_name="Read",
            tool_input={"path": "original"},
            bash_command="cat original",
        )
    )
    _write_session(conn, session, tail_facts=TailFacts())
    state = EmbedPhaseState(_config=config)
    backend = RecordingBackend()
    _allow_embedding(monkeypatch, state, backend)

    prepared = prepare_embed_cycle(config, state, conn=conn)
    conn.execute(statement, params)
    generate_prepared_embed_cycle(config, state, prepared)

    assert commit_prepared_embed_cycle(prepared, conn=conn) == 0
    assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (0,)


def test_prepared_context_commit_enqueues_and_repairs_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detached context publication reaches SQLite FTS through durable pending work."""
    from recall.services.indexer import _write_session

    config = _app_config(tmp_path)
    config = AppConfig(
        data_dir=config.data_dir,
        db_path=config.db_path,
        lock_path=config.lock_path,
        config_path=config.config_path,
        fts=config.fts,
        embedding=EmbeddingConfig(context=ContextConfig(mode="llm-local")),
        daemon=config.daemon,
        cli=config.cli,
    )
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("sidecar", "body"), tail_facts=TailFacts())
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())

    monkeypatch.setattr(
        "recall.services.indexer._prepare_context_run",
        lambda _config: types.SimpleNamespace(config=_config, backend=None),
    )

    def contextualize(
        session: Session,
        _config: ContextConfig,
        _backend: object,
        *,
        targets: list[Message],
    ) -> SessionContextStats:
        for message in targets:
            message.context_text = "[generated] "
            message.context_mode = "llm-local"
        return SessionContextStats(messages=len(targets))

    monkeypatch.setattr("recall.services.indexer._resolve_session_message_contexts", contextualize)

    prepared = generate_prepared_embed_cycle(
        config, state, prepare_embed_cycle(config, state, conn=conn)
    )

    assert commit_prepared_embed_cycle(prepared, conn=conn) == 1
    assert conn.execute("SELECT kind, id, op FROM fts_sidecar_pending").fetchall() == [
        ("message", "sidecar-m0", "upsert")
    ]

    sidecar_conn = open_sidecar(tmp_path / "recall.fts.sqlite")
    try:
        stats = reconcile_sidecar(conn, sidecar_conn)
        assert stats.pending_drained["message"] == 1
        assert sidecar_conn.execute(
            """
            SELECT message_id FROM message_fts_rowid
            JOIN message_fts ON message_fts.rowid = message_fts_rowid.rowid
            WHERE message_fts MATCH 'generated'
            """
        ).fetchall() == [("sidecar-m0",)]
    finally:
        sidecar_conn.close()


def test_prepared_context_replaces_vector_for_new_context_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A context change replaces a pre-existing vector for the old effective text."""
    from recall.services.indexer import _write_session

    base_config = _app_config(tmp_path)
    config = AppConfig(
        data_dir=base_config.data_dir,
        db_path=base_config.db_path,
        lock_path=base_config.lock_path,
        config_path=base_config.config_path,
        fts=base_config.fts,
        embedding=EmbeddingConfig(context=ContextConfig(mode="llm-local")),
        daemon=base_config.daemon,
        cli=base_config.cli,
    )
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("vector", "body"), tail_facts=TailFacts())
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend(value=2.0))

    prepared = prepare_embed_cycle(config, state, conn=conn)
    conn.execute(
        "INSERT INTO message_embeddings VALUES (?, ?, ?)",
        ["vector-m0", [1.0] * 384, None],
    )
    monkeypatch.setattr(
        "recall.services.indexer._prepare_context_run",
        lambda _config: types.SimpleNamespace(config=_config, backend=None),
    )

    def contextualize(
        session: Session,
        _config: ContextConfig,
        _backend: object,
        *,
        targets: list[Message],
    ) -> SessionContextStats:
        for message in targets:
            message.context_text = "[fresh] "
            message.context_mode = "llm-local"
        return SessionContextStats(messages=len(targets))

    monkeypatch.setattr("recall.services.indexer._resolve_session_message_contexts", contextualize)
    generate_prepared_embed_cycle(config, state, prepared)

    assert commit_prepared_embed_cycle(prepared, conn=conn) == 1
    assert conn.execute(
        "SELECT content_embedding[1] FROM message_embeddings WHERE message_id = ?",
        ["vector-m0"],
    ).fetchone() == (2.0,)


def test_interrupted_prepared_generation_publishes_only_completed_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopping between sessions never marks an ungenerated session embedded."""
    from recall.services.indexer import _write_session

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("first-prepared", "first"), tail_facts=TailFacts())
    _write_session(conn, _idle_session("second-prepared", "second"), tail_facts=TailFacts())
    state = EmbedPhaseState(_config=config)
    backend = RecordingBackend()
    _allow_embedding(monkeypatch, state, backend)

    prepared = prepare_embed_cycle(config, state, conn=conn)
    generate_prepared_embed_cycle(config, state, prepared, should_stop=lambda: bool(backend.calls))

    assert commit_prepared_embed_cycle(prepared, conn=conn) == 1
    assert find_pending_embeds(conn, idle_threshold=0).session_ids == ["second-prepared"]


class TestEmbedPhaseState:
    def test_maybe_unload_after_timeout(self) -> None:
        config = _app_config(Path("/tmp/test"))
        # Override embed_model_timeout to 0 for immediate unload
        config = AppConfig(
            data_dir=config.data_dir,
            db_path=config.db_path,
            lock_path=config.lock_path,
            config_path=config.config_path,
            fts=config.fts,
            embedding=config.embedding,
            daemon=DaemonConfig(embed_model_timeout=0),
            cli=config.cli,
        )
        state = EmbedPhaseState(_config=config)
        # Simulate a loaded backend
        state.backend = cast(Any, object())
        state.last_used_at = time.monotonic() - 1  # 1 second ago
        state.maybe_unload()
        assert state.backend is None

    def test_maybe_unload_skips_when_recently_used(self) -> None:
        config = _app_config(Path("/tmp/test"))
        state = EmbedPhaseState(_config=config)
        state.backend = cast(Any, object())
        state.last_used_at = time.monotonic()  # just now
        state.maybe_unload()
        assert state.backend is not None  # 600s timeout not reached

    def test_loop_liveness_reports_the_stage_it_entered(self) -> None:
        """A stalled loop must be distinguishable from an idle one by stage age."""
        state = EmbedPhaseState(_config=_app_config(Path("/tmp/test")))

        before = time.time()
        state.begin_timer_cycle()
        state.enter_stage("snapshot")
        after = time.time()

        assert state.loop_iterations == 1
        assert before <= state.last_iteration_at <= after
        assert state.last_trigger == "timer"
        assert state.stage == "snapshot"
        assert state.last_iteration_at <= state.stage_at <= after

        state.record_outcome("drained")
        assert state.last_outcome == "drained"

    def test_a_deferral_reason_carries_the_time_it_was_evaluated(self) -> None:
        """A reason with no timestamp reads the same live as hours stale."""
        state = EmbedPhaseState(_config=_app_config(Path("/tmp/test")))

        assert state.deferred_at == 0.0

        before = time.time()
        state.deferred_reason = "load 41.0 > battery threshold 5.4 (on battery: Battery Power)"

        assert before <= state.deferred_at <= time.time()

        state.deferred_reason = None

        assert state.deferred_at == 0.0


def test_embed_cycle_records_its_outcome_and_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every cycle outcome is observable, including the idle path that stayed silent."""
    import asyncio

    from recall.services.indexer import _write_session
    from recall.services.rpc_server import RpcServer

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("outcome-session", "hello"), tail_facts=TailFacts())
    backend = RecordingBackend()
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, backend)
    server = RpcServer(config)
    server._conn = conn

    drained = asyncio.run(server._run_embed_batch(config, state))

    assert drained == 1
    assert state.last_outcome == "drained"
    assert state.stage == "commit"

    drained = asyncio.run(server._run_embed_batch(config, state))

    assert drained == 0
    assert state.last_outcome == "no-eligible-sessions"
    assert state.stage == "unload"


def test_requested_enrichment_records_its_own_cycle_beside_the_timers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requested work records its outcome without wearing the timer's stage.

    The timer loop and per-session index enrichment share one EmbedPhaseState.
    The trigger, outcome and batch must move whichever path committed, or an
    operator reading `outcome=deferred` beside an advancing `embed_last_batch_at`
    cannot tell which one describes the daemon. The timer's own pacing fields
    must not: a requested cycle runs while the timer sleeps, so writing its
    stage leaves `daemon status` naming a stage the daemon already left with an
    age that grows until the timer next wakes, and counting it as an iteration
    reads a `--full` reparse as thousands of timer cycles (REQ-ADAPT-012).
    """
    import asyncio

    from recall.services.embed_phase import prepare_index_enrichment
    from recall.services.indexer import _write_session
    from recall.services.rpc_server import RpcServer
    from recall.services.system_state import EmbedPreconditionResult

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("requested-session", "hello"), tail_facts=TailFacts())
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())
    monkeypatch.setattr(
        "recall.services.system_state.check_load",
        lambda _threshold: EmbedPreconditionResult(
            ok=False, reason="load 41.0 > battery threshold 5.4 (on battery: Battery Power)"
        ),
    )
    server = RpcServer(config)
    server._conn = conn
    server._embed_state = state

    timer_stage_at = 0.0

    async def observe() -> dict[str, Any]:
        nonlocal timer_stage_at
        state.begin_timer_cycle()
        assert await server._run_embed_batch(config, state) == -1
        assert state.last_outcome == "deferred"
        timer_stage_at = state.stage_at
        await server._generate_requested_enrichment(
            config,
            lambda: prepare_index_enrichment(config, "requested-session", conn=conn, embed=True),
        )
        return await server._handle_daemon_status({}, None)

    status = asyncio.run(observe())

    assert state.last_trigger == "requested"
    assert state.last_outcome == "drained"
    assert state.last_batch_size == 1
    # The requested cycle is counted and staged in its own fields.
    assert state.requested_cycles == 1
    assert state.requested_stage == "commit"
    # The timer's iteration count and stage are exactly where its own last
    # cycle left them.
    assert state.loop_iterations == 1
    assert state.stage == "preconditions"
    assert state.stage_at == timer_stage_at
    assert status["embed_loop_iterations"] == 1
    assert status["embed_loop_stage"] == "preconditions"
    assert status["embed_requested_cycles"] == 1
    assert status["embed_requested_stage"] == "commit"
    assert status["embed_loop_last_trigger"] == "requested"
    assert status["embed_loop_last_outcome"] == "drained"


def test_failed_requested_enrichment_closes_its_cycle_with_an_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A requested cycle that raises before committing still records an outcome.

    Preparation runs inside the writer. When it raises, an unclosed cycle leaves
    the requested fields naming a stage the daemon already left with no outcome
    at all, so `daemon status` reports work that is not running (REQ-ADAPT-012).
    """
    import asyncio

    from recall.services.embed_phase import PreparedEmbedCycle
    from recall.services.rpc_server import RpcServer

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())
    server = RpcServer(config)
    server._conn = conn
    server._embed_state = state
    state.enter_stage("wait")
    timer_stage_at = state.stage_at

    def prepare() -> PreparedEmbedCycle:
        raise RuntimeError("transcript vanished mid-request")

    with pytest.raises(RuntimeError, match="transcript vanished mid-request"):
        asyncio.run(server._generate_requested_enrichment(config, prepare))

    assert state.requested_cycles == 1
    assert state.last_trigger == "requested"
    assert state.last_outcome == "failed"
    assert state.requested_stage == "snapshot"
    assert state.stage == "wait"
    assert state.stage_at == timer_stage_at


def test_deferred_cycle_reports_its_reason_and_the_age_of_its_pending_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferral must be readable beside the outcome, and both of them dated.

    `embed_pending` is served from the last snapshot the phase took, so during a
    backoff window it stays byte-identical while watch indexing keeps adding
    messages. Without the observation time, "frozen because idle" and "frozen
    because nobody looked" read the same. The deferral reason needs the same
    treatment: an operator whose `pmset` says AC while the reason names the
    battery threshold cannot otherwise tell a stale reason from a live one
    (REQ-ADAPT-006, REQ-ADAPT-012).
    """
    import asyncio

    from recall.services.indexer import _write_session
    from recall.services.rpc_server import RpcServer
    from recall.services.system_state import EmbedPreconditionResult

    config = _app_config(tmp_path)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    _write_session(conn, _idle_session("pending-session", "hello"), tail_facts=TailFacts())
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())
    server = RpcServer(config)
    server._conn = conn
    server._embed_state = state

    reason = "load 41.0 > battery threshold 5.4 (on battery: Battery Power)"

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        assert await server._run_embed_batch(config, state) == 1
        observed = await server._handle_daemon_status({}, None)
        _write_session(conn, _idle_session("later-session", "later"), tail_facts=TailFacts())
        monkeypatch.setattr(
            "recall.services.system_state.check_load",
            lambda _threshold: EmbedPreconditionResult(ok=False, reason=reason),
        )
        assert await server._run_embed_batch(config, state) == -1
        return observed, await server._handle_daemon_status({}, None)

    before_deferral = time.time()
    observed, deferred = asyncio.run(scenario())

    assert observed["embed_deferred_reason"] is None
    assert observed["embed_deferred_at"] is None
    assert observed["embed_pending_at"] == pytest.approx(state.last_pending_at)
    # The deferral short-circuits before the snapshot, so the pending count and
    # its observation time both stand still while the backlog grows.
    assert deferred["embed_pending"] == observed["embed_pending"]
    assert deferred["embed_pending_at"] == observed["embed_pending_at"]
    assert deferred["embed_deferred_reason"] == reason
    assert before_deferral <= deferred["embed_deferred_at"] <= time.time()
    # The reconciliation view keeps carrying the reason for existing readers.
    assert deferred["reconciliation"]["enrichment_deferred"] == reason
