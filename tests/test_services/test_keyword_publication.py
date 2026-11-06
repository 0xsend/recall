"""Keyword publication follows DuckDB COMMIT, once per raw writer turn.

`REQ-FTS-SIDECAR-006` and `REQ-FTS-SIDECAR-018`: the sidecar never holds a
change DuckDB has not committed, a raw turn publishes its keyword changes in
one SQLite commit after its last DuckDB COMMIT, and a busy or locked sidecar
never fails or stalls the DuckDB write beyond one bounded deadline.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pytest
from lane_harness import Lane, build_lane
from recall.core import telemetry
from recall.core.config import AppConfig
from recall.db import connect, fts_sidecar
from recall.db.fts_sidecar import (
    content_fingerprint,
    open_sidecar,
    scope_message_fts_columns,
    search_messages_fts,
    sidecar_path,
)
from recall.db.source_files import SourceCatalog
from recall.parsers.codex import CodexParser
from recall.services.coordinator import (
    RawCommitResult,
    capture_path,
    commit_prepared_raw_sources,
    observe_path,
    prepare_raw_sources,
)
from recall.services.daemon import run_startup_fts_sidecar_sync
from recall.services.embed_phase import PreparedEmbedCycle

SESSION_META: dict[str, object] = {
    "type": "session_meta",
    "payload": {
        "id": "keyword-turn",
        "timestamp": "2026-11-06T12:00:00Z",
        "cwd": "/work/keyword-repo",
        "git": {"branch": "main", "root": "/work/keyword-repo"},
    },
}


def _record(record: dict[str, object]) -> bytes:
    return (json.dumps(record) + "\n").encode()


def _user(text: str) -> bytes:
    return _record(
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": text}}
    )


def _shell(call_id: str, command: str) -> bytes:
    return _record(
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "shell_command",
                "arguments": json.dumps({"cmd": command}),
                "call_id": call_id,
            },
        }
    )


def _transcript(*records: bytes) -> bytes:
    return _record(SESSION_META) + b"".join(records)


@dataclass
class RawTurns:
    """One Codex transcript committed through the daemon's raw-turn boundary."""

    config: AppConfig
    conn: duckdb.DuckDBPyConnection
    path: Path
    parser: CodexParser

    @property
    def sidecar(self) -> Path:
        return sidecar_path(self.config.data_dir)

    def turn(self, content: bytes) -> RawCommitResult:
        self.path.write_bytes(content)
        item = observe_path(self.parser, capture_path(self.parser, self.path), conn=self.conn)
        prepared = prepare_raw_sources((item,), {self.parser.source.value: self.parser})
        return commit_prepared_raw_sources(prepared, self.config, conn=self.conn)

    def message_id(self, text: str) -> str:
        row = self.conn.execute(
            "SELECT message_id FROM message_state WHERE content = ?", [text]
        ).fetchone()
        assert row is not None, text
        return str(row[0])

    def pending(self) -> list[tuple[str, str]]:
        return [
            (str(kind), str(entity))
            for kind, entity in self.conn.execute(
                "SELECT kind, id FROM fts_sidecar_pending ORDER BY kind, id"
            ).fetchall()
        ]


def _raw_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    environment = {
        "HOME": str(tmp_path),
        "RECALL_DATA_DIR": str(tmp_path / "data"),
        "RECALL_CONFIG_PATH": str(tmp_path / "config.toml"),
        "RECALL_CONTEXT_MODE": "off",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    return environment


def _start_keyword_index(config: AppConfig) -> None:
    """Create and bootstrap the keyword index as daemon startup does before serving.

    A completed bootstrap is what makes a later startup sync a reconciliation
    pass rather than a full rebuild that would hide drift.
    """
    result = run_startup_fts_sidecar_sync(config)
    assert result.error is None


@pytest.fixture
def raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[RawTurns]:
    _raw_environment(tmp_path, monkeypatch)
    config = AppConfig.load()
    _start_keyword_index(config)
    conn = connect(config)
    try:
        yield RawTurns(
            config, conn, tmp_path / "rollout-keyword.jsonl", CodexParser(roots=(tmp_path,))
        )
    finally:
        conn.close()


def _change_counter(path: Path) -> int:
    """SQLite's file change counter: one increment per committed write transaction."""
    with path.open("rb") as handle:
        handle.seek(24)
        return int.from_bytes(handle.read(4), "big")


def _sidecar_fingerprints(sidecar: Path) -> tuple[dict[str, str], dict[str, str]]:
    with closing(sqlite3.connect(sidecar)) as conn:
        messages = dict(
            conn.execute("SELECT message_id, content_fingerprint FROM message_fts_rowid")
        )
        tools = dict(
            conn.execute("SELECT tool_call_id, content_fingerprint FROM tool_calls_fts_rowid")
        )
    return messages, tools


def _assert_sidecar_matches_duckdb(conn: duckdb.DuckDBPyConnection, sidecar: Path) -> None:
    """Membership both ways and every id's fingerprint equal DuckDB's scoped columns."""
    messages = {
        str(message_id): content_fingerprint(
            scope_message_fts_columns(content or "", thinking or "", None)
        )
        for message_id, content, thinking in conn.execute(
            "SELECT message_id, fts_content, fts_thinking FROM message_state"
        ).fetchall()
    }
    tools = {
        str(tool_id): content_fingerprint((command,))
        for tool_id, command in conn.execute(
            "SELECT id, bash_command FROM tool_calls WHERE bash_command IS NOT NULL"
        ).fetchall()
    }
    assert _sidecar_fingerprints(sidecar) == (messages, tools)


def _keyword_hits(sidecar: Path, query: str) -> list[str]:
    with closing(sqlite3.connect(sidecar)) as conn:
        return [key for key, _score in search_messages_fts(conn, query, [], 10)]


def _counter(recorder: telemetry.Recorder, name: str) -> int:
    return recorder.counters.get(name, 0)


@pytest.fixture
def recorder() -> Iterator[telemetry.Recorder]:
    recorder = telemetry.Recorder()
    token = telemetry.current.set(recorder)
    try:
        yield recorder
    finally:
        telemetry.current.reset(token)


def test_req_fts_sidecar_018_rolled_back_rewrite_never_reaches_keyword_search(
    raw: RawTurns, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw.turn(_transcript(_user("alpha original wording")))
    message_id = raw.message_id("alpha original wording")
    alpha = _sidecar_fingerprints(raw.sidecar)[0][message_id]

    # The acknowledgement runs last inside the rewrite's transaction; refusing
    # it rolls the rewrite back.
    monkeypatch.setattr(SourceCatalog, "acknowledge", lambda *_args, **_kwargs: False)
    raw.turn(_transcript(_user("omega replacement wording")))

    assert raw.message_id("alpha original wording") == message_id
    assert _keyword_hits(raw.sidecar, "omega") == []
    assert _sidecar_fingerprints(raw.sidecar)[0][message_id] == alpha


def test_req_fts_sidecar_018_one_commit_per_raw_turn(raw: RawTurns) -> None:
    raw.turn(
        _transcript(
            _user("first question about alpha"),
            _shell("call-zero", "echo zero"),
            _shell("call-one", "echo first"),
        )
    )
    before = _change_counter(raw.sidecar)

    # New message, rewritten message, rewritten and new tool calls, removed tool call.
    raw.turn(
        _transcript(
            _user("first question rewritten as omega"),
            _shell("call-one", "printf second"),
            _shell("call-two", "ls newer"),
            _user("second question about gamma"),
        )
    )

    assert _change_counter(raw.sidecar) - before == 1
    _assert_sidecar_matches_duckdb(raw.conn, raw.sidecar)
    assert raw.pending() == []


_CRASH_CHILD = """
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from recall.core.config import AppConfig
from recall.db import connect, fts_sidecar
from recall.parsers.codex import CodexParser
from recall.services.coordinator import (
    capture_path,
    commit_prepared_raw_sources,
    observe_path,
    prepare_raw_sources,
)

kill_point, transcript, root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
publish = fts_sidecar.publish_sidecar


@contextmanager
def publish_then_exit(conn, **kwargs):
    if kill_point == "before_publication":
        os._exit(17)
    with publish(conn, **kwargs) as publication:
        yield publication
    os._exit(17)


fts_sidecar.publish_sidecar = publish_then_exit
config = AppConfig.load()
parser = CodexParser(roots=(root,))
conn = connect(config)
item = observe_path(parser, capture_path(parser, transcript), conn=conn)
commit_prepared_raw_sources(
    prepare_raw_sources((item,), {parser.source.value: parser}), config, conn=conn
)
sys.exit(3)
"""


@pytest.mark.parametrize("kill_point", ["before_publication", "after_sqlite_commit"])
def test_req_fts_sidecar_018_crash_at_the_publication_boundary_heals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kill_point: str
) -> None:
    environment = _raw_environment(tmp_path, monkeypatch)
    config = AppConfig.load()
    _start_keyword_index(config)
    conn = connect(config)
    raw = RawTurns(config, conn, tmp_path / "rollout-keyword.jsonl", CodexParser(roots=(tmp_path,)))
    try:
        raw.turn(_transcript(_user("alpha before the crash")))
    finally:
        conn.close()
    raw.path.write_bytes(_transcript(_user("omega after the crash")))
    child = tmp_path / "crash_child.py"
    child.write_text(_CRASH_CHILD)

    killed = subprocess.run(
        [sys.executable, "-I", str(child), kill_point, str(raw.path), str(tmp_path)],
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert killed.returncode == 17, killed.stderr

    run_startup_fts_sidecar_sync(config)
    with closing(connect(config)) as healed:
        assert healed.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone() == (0,)
        _assert_sidecar_matches_duckdb(healed, raw.sidecar)


_RECOMPUTE_CHILD = """
import os
import sys

from recall.core.config import AppConfig
from recall.db import fts_sidecar
from recall.services.indexer import recompute_context_for_rows


def exit_before_publication(*_args, **_kwargs):
    os._exit(17)


fts_sidecar.publish_sidecar = exit_before_publication
recompute_context_for_rows(config=AppConfig.load(), only_mode="off")
sys.exit(3)
"""


def test_recompute_context_for_rows_crash_after_commit_heals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _raw_environment(tmp_path, monkeypatch)
    config = AppConfig.load()
    _start_keyword_index(config)
    conn = connect(config)
    raw = RawTurns(config, conn, tmp_path / "rollout-keyword.jsonl", CodexParser(roots=(tmp_path,)))
    try:
        raw.turn(_transcript(_user("recompute keeps this wording")))
    finally:
        conn.close()
    child = tmp_path / "recompute_child.py"
    child.write_text(_RECOMPUTE_CHILD)

    killed = subprocess.run(
        [sys.executable, "-I", str(child)],
        env={**os.environ, **environment, "RECALL_CONTEXT_MODE": "template"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert killed.returncode == 17, killed.stderr

    run_startup_fts_sidecar_sync(config)
    with closing(connect(config)) as healed:
        assert healed.execute(
            "SELECT context_mode FROM message_state WHERE content = 'recompute keeps this wording'"
        ).fetchone() == ("template",)
        assert healed.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone() == (0,)
        _assert_sidecar_matches_duckdb(healed, raw.sidecar)


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Lane]:
    lane = build_lane(tmp_path, monkeypatch)
    yield lane
    asyncio.run(lane.server.stop())


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    holder = sqlite3.connect(path, isolation_level=None, timeout=0)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        yield
        holder.execute("ROLLBACK")
    finally:
        holder.close()


def _reconciliation(lane: Lane) -> dict[str, object]:
    status = asyncio.run(lane.server._handle_daemon_status({}, None))
    return status["reconciliation"]


def test_req_fts_sidecar_018_locked_sidecar_commits_raw_turn_and_defers_keyword_work(
    lane: Lane, recorder: telemetry.Recorder
) -> None:
    server = lane.server
    sidecar = sidecar_path(server._config.data_dir)
    conn = server._get_conn()
    assert sidecar.exists()
    assert conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone() == (0,)
    assert _reconciliation(lane).get("keyword_publish_error") is None
    deferrals = _counter(recorder, "keyword_publish_deferrals")

    with _exclusive_lock(sidecar):
        lane.append(lane.watched, "lockedtoken appended message", uuid="locked-append")
        started = time.monotonic()
        asyncio.run(server.index_session_now(lane.watched))
        elapsed = time.monotonic() - started

        assert elapsed < 3.0
        assert conn.execute(
            "SELECT COUNT(*) FROM message_state WHERE content = 'lockedtoken appended message'"
        ).fetchone() == (1,)
        source = SourceCatalog(conn, clock=time.time).get("claude_code", str(lane.watched))
        assert source is not None
        assert source.committed_generation == source.desired_generation
        assert source.last_error is None
        pending = conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone()
        assert pending is not None and pending[0] > 0
        assert _counter(recorder, "keyword_publish_deferrals") == deferrals + 1
        assert _reconciliation(lane)["keyword_publish_error"] is not None

    server._keyword_gate.due_at = 0.0
    server._next_fts_repair_at = 0.0
    asyncio.run(server._repair_keyword_batch())

    with closing(sqlite3.connect(sidecar)) as reader:
        assert [key for key, _ in search_messages_fts(reader, "lockedtoken", [], 10)] != []
    assert conn.execute("SELECT COUNT(*) FROM fts_sidecar_pending").fetchone() == (0,)
    assert _reconciliation(lane)["keyword_publish_error"] is None


def test_req_fts_sidecar_018_publication_wait_is_one_deadline(
    raw: RawTurns, monkeypatch: pytest.MonkeyPatch, recorder: telemetry.Recorder
) -> None:
    monkeypatch.setattr(fts_sidecar, "FTS_SIDECAR_BATCH_SIZE", 2)
    raw.turn(_transcript(_user("seed message")))
    content = _transcript(
        _user("seed message"),
        _user("batch wording one"),
        _user("batch wording two"),
        _user("batch wording three"),
        _user("batch wording four"),
    )
    raw.path.write_bytes(content)
    item = observe_path(raw.parser, capture_path(raw.parser, raw.path), conn=raw.conn)
    prepared = prepare_raw_sources((item,), {raw.parser.source.value: raw.parser})
    deferrals = _counter(recorder, "keyword_publish_deferrals")
    commits = _counter(recorder, "keyword_publish_commits")

    first_reader = sqlite3.connect(raw.sidecar, isolation_level=None, check_same_thread=False)
    first_reader.execute("BEGIN")
    first_reader.execute("SELECT COUNT(*) FROM message_fts_rowid").fetchone()
    second_reader = sqlite3.connect(raw.sidecar, isolation_level=None, check_same_thread=False)
    publish = fts_sidecar.publish_sidecar
    calls = 0

    @contextmanager
    def publish_with_second_reader(conn: sqlite3.Connection, **kwargs: Any) -> Iterator[object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            second_reader.execute("BEGIN")
            second_reader.execute("SELECT COUNT(*) FROM message_fts_rowid").fetchone()
        with publish(conn, **kwargs) as publication:
            yield publication

    monkeypatch.setattr(fts_sidecar, "publish_sidecar", publish_with_second_reader)
    started = time.monotonic()
    release = threading.Timer(0.7, lambda: first_reader.execute("COMMIT"))
    release.start()
    try:
        commit_prepared_raw_sources(prepared, raw.config, conn=raw.conn)
        elapsed = time.monotonic() - started
    finally:
        release.join()
        if second_reader.in_transaction:
            second_reader.execute("ROLLBACK")
        second_reader.close()
        first_reader.close()

    assert elapsed < 1.0 + 0.3
    staged = [raw.message_id(f"batch wording {word}") for word in ("one", "two", "three", "four")]
    published = set(_sidecar_fingerprints(raw.sidecar)[0])
    pending = {entity for _kind, entity in raw.pending()}
    assert len([key for key in staged if key in published]) == 2
    assert all((key in published) != (key in pending) for key in staged)
    assert _counter(recorder, "keyword_publish_deferrals") == deferrals + 1
    assert _counter(recorder, "keyword_publish_commits") == commits + 1


def _changed_context_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sessions: dict[str, int]
) -> tuple[AppConfig, duckdb.DuckDBPyConnection, PreparedEmbedCycle]:
    """An enrichment cycle generated for `sessions` (id -> message count) with changed context."""
    import types

    from recall.core.config import ContextConfig, EmbeddingConfig
    from recall.core.models import Message, Session
    from recall.core.types import Role, Source
    from recall.db.queries import insert_messages, insert_session
    from recall.db.schema import ensure_schema
    from recall.services.embed_phase import (
        EmbedPhaseState,
        generate_prepared_embed_cycle,
        prepare_embed_cycle,
    )
    from recall.services.indexer import SessionContextStats

    from tests.test_services.test_embed_phase import RecordingBackend, _allow_embedding, _app_config

    base = _app_config(tmp_path)
    config = AppConfig(
        data_dir=base.data_dir,
        db_path=base.db_path,
        lock_path=base.lock_path,
        config_path=base.config_path,
        fts=base.fts,
        embedding=EmbeddingConfig(context=ContextConfig(mode="llm-local")),
        daemon=base.daemon,
        cli=base.cli,
    )
    open_sidecar(sidecar_path(config.data_dir)).close()
    conn = duckdb.connect(":memory:")
    ensure_schema(conn)
    for session_id, count in sessions.items():
        session = Session(
            id=session_id,
            source=Source.CODEX,
            source_path=f"/tmp/{session_id}.jsonl",
            file_mtime=1.0,
            file_size=1,
            messages=[
                Message(
                    id=f"{session_id}-m{idx}",
                    session_id=session_id,
                    idx=idx,
                    role=Role.ASSISTANT,
                    content=f"body {idx}",
                )
                for idx in range(count)
            ],
            message_count=count,
        )
        conn.execute("BEGIN")
        insert_session(conn, session)
        insert_messages(conn, session.messages)
        conn.execute("COMMIT")
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())
    monkeypatch.setattr(
        "recall.services.indexer._prepare_context_run",
        lambda _config: types.SimpleNamespace(config=_config, backend=None),
    )

    def contextualize(
        session: Session, _config: ContextConfig, _backend: object, *, targets: list[Message]
    ) -> SessionContextStats:
        for message in targets:
            message.context_text = "[generated] "
            message.context_mode = "llm-local"
        return SessionContextStats(messages=len(targets))

    monkeypatch.setattr("recall.services.indexer._resolve_session_message_contexts", contextualize)
    prepared = generate_prepared_embed_cycle(
        config, state, prepare_embed_cycle(config, state, conn=conn)
    )
    return config, conn, prepared


def test_req_fts_sidecar_006_embed_publication_failure_keeps_intents_and_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services.embed_phase import publish_prepared_embed_cycle

    config, conn, prepared = _changed_context_cycle(tmp_path, monkeypatch, {"locked": 1})

    with _exclusive_lock(sidecar_path(config.data_dir)):
        committed = publish_prepared_embed_cycle(prepared, config, conn=conn)

    assert committed == 1
    assert conn.execute(
        "SELECT context_text FROM message_state WHERE message_id = 'locked-m0'"
    ).fetchone() == ("[generated] ",)
    assert conn.execute("SELECT kind, id, op FROM fts_sidecar_pending").fetchall() == [
        ("message", "locked-m0", "upsert")
    ]
