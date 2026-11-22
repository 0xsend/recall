"""Service, RPC, and owned poll-snapshot tests for PR 3 search filters."""

from __future__ import annotations

import asyncio
import importlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from recall.core.config import (
    DEFAULT_EMBED_MODEL,
    KNOWN_MODEL_DIMENSIONS,
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.poll_reads import SearchFilteredRead, poll_read_request
from recall.core.rpc_types import INVALID_PARAMS
from recall.core.search_filters import has_filtered_search_params, normalize_search_filters
from recall.core.types import Role, SearchMode, Source
from recall.db import connect
from recall.db.fts_sidecar import (
    open_sidecar,
    set_fts_fields_signature,
    sidecar_path,
    upsert_message_fts,
    upsert_tool_call_fts,
)
from recall.db.lease import StandaloneLifetime
from recall.db.queries import create_fts_indexes
from recall.read_budget import ReadBudget
from recall.services.poll_reads import read_poll_snapshot
from recall.services.rpc_server import ClientConnection, RpcServer
from recall.services.search import search

search_module: Any = importlib.import_module("recall.services.search")

_DIM = KNOWN_MODEL_DIMENSIONS[DEFAULT_EMBED_MODEL]


def _vec(index: int, weight: float = 1.0) -> list[float]:
    values = [0.0] * _DIM
    values[index] = weight
    return values


def _make_config(tmp_path: Path, *, backend: str = "sqlite_sidecar") -> AppConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(backend=backend),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )


def _seed_filtered_corpus(
    config: AppConfig,
) -> tuple[datetime, datetime]:
    """Seed multi-session, multi-role, multi-host, multi-project, multi-time corpus."""
    now = datetime(2026, 11, 20, 12, 0, 0)
    old = now - timedelta(days=30)
    recent = now - timedelta(hours=2)

    with connect(config) as conn:
        sessions = [
            # id, source, path, source_sid, repo, host, started, ended, msg_cnt, tool_cnt
            (
                "s1",
                "codex",
                "/p/s1.jsonl",
                "src-1",
                "github.com/org/Recall",
                "aem5",
                old,
                recent,
                4,
                2,
            ),
            (
                "s2",
                "codex",
                "/p/s2.jsonl",
                "src-2",
                "github.com/org/Recall",
                "aem5",
                recent,
                recent,
                2,
                1,
            ),
            (
                "s3",
                "codex",
                "/p/s3.jsonl",
                "src-3",
                "github.com/org/Other",
                "ae-dev",
                recent,
                recent,
                2,
                1,
            ),
            (
                "s4",
                "codex",
                "/p/s4.jsonl",
                "src-4",
                "github.com/org/Recall",
                "aem5",
                old,
                old,
                2,
                1,
            ),
            # Same source_session_id as s1, different host for --session + --host disambiguation
            (
                "s5",
                "codex",
                "/p/s5.jsonl",
                "src-1",
                "github.com/org/Recall",
                "ae-dev",
                recent,
                recent,
                1,
                0,
            ),
        ]
        for sid, src, path, ssid, repo, host, started, ended, mcnt, tcnt in sessions:
            conn.execute(
                "INSERT INTO sessions (id, source, source_path, source_session_id) "
                "VALUES (?, ?, ?, ?)",
                [sid, src, path, ssid],
            )
            conn.execute(
                """
                INSERT INTO session_state (
                    session_id, started_at, ended_at, git_repo, host,
                    message_count, tool_count, is_complete, file_mtime, file_size, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, true, 1.0, 100, ?)
                """,
                [sid, started, ended, repo, host, mcnt, tcnt, ended],
            )

        messages = [
            # s1: mix of roles/timestamps (old message in recent session + recent + NULL ts)
            ("m1", "s1", 0, "user", "deploy pipeline alpha needle", old, _vec(0)),
            ("m2", "s1", 1, "assistant", "deploy pipeline beta needle", recent, _vec(0)),
            ("m3", "s1", 2, "system", "deploy pipeline gamma needle", recent, _vec(0)),
            ("m4", "s1", 3, "user", "deploy pipeline nullts needle", None, _vec(0)),
            # s2: recent Recall on aem5
            ("m5", "s2", 0, "user", "deploy pipeline s2 user needle", recent, _vec(0)),
            ("m6", "s2", 1, "assistant", "deploy pipeline s2 assistant needle", recent, _vec(0)),
            # s3: recent Other on ae-dev
            ("m7", "s3", 0, "user", "deploy pipeline other repo needle", recent, _vec(0)),
            (
                "m8",
                "s3",
                1,
                "assistant",
                "deploy pipeline other repo reply needle",
                recent,
                _vec(0),
            ),
            # s4: old Recall on aem5
            ("m9", "s4", 0, "user", "deploy pipeline old session needle", old, _vec(0)),
            ("m10", "s4", 1, "assistant", "deploy pipeline old reply needle", old, _vec(0)),
            # s5: recent Recall on ae-dev with same source_session_id as s1
            ("m11", "s5", 0, "user", "deploy pipeline ae-dev src1 needle", recent, _vec(0)),
        ]
        for mid, sid, idx, role, content, ts, emb in messages:
            conn.execute(
                "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)", [mid, sid, idx]
            )
            conn.execute(
                """
                INSERT INTO message_state (
                    message_id, role, content, fts_content, fts_thinking, timestamp
                ) VALUES (?, ?, ?, ?, '', ?)
                """,
                [mid, role, content, content, ts],
            )
            conn.execute(
                "INSERT INTO message_embeddings (message_id, content_embedding) VALUES (?, ?)",
                [mid, emb],
            )

        tool_calls = [
            ("tc1", "s1", "m2", 0, "Bash", "deploy needle cmd one", _vec(0)),
            ("tc2", "s1", "m2", 1, "Bash", "deploy needle cmd two", _vec(0)),
            ("tc3", "s2", "m6", 0, "Bash", "deploy needle cmd three", _vec(0)),
            ("tc4", "s3", "m8", 0, "Bash", "deploy needle cmd other", _vec(0)),
            ("tc5", "s4", "m10", 0, "Bash", "deploy needle cmd old", _vec(0)),
        ]
        for tcid, sid, mid, idx, tool_name, cmd, emb in tool_calls:
            conn.execute(
                """
                INSERT INTO tool_calls (id, session_id, message_id, idx, tool_name, bash_command)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [tcid, sid, mid, idx, tool_name, cmd],
            )
            conn.execute(
                "INSERT INTO tool_call_embeddings (tool_call_id, bash_embedding) VALUES (?, ?)",
                [tcid, emb],
            )

        if config.fts.backend == "duckdb":
            create_fts_indexes(conn, config.fts, observer=StandaloneLifetime())
        else:
            sidecar = open_sidecar(sidecar_path(config.data_dir))
            try:
                for mid, _sid, _idx, _role, content, _ts, _emb in messages:
                    upsert_message_fts(sidecar, mid, content, "")
                for tcid, _sid, _mid, _idx, _tool_name, cmd, _emb in tool_calls:
                    upsert_tool_call_fts(sidecar, tcid, cmd)
                set_fts_fields_signature(sidecar, config.fts.fields)
                sidecar.commit()
            finally:
                sidecar.close()

    return old, recent


def test_normalize_search_filters_unit() -> None:
    parsed = normalize_search_filters(
        role=[" USER ", "assistant", "user"],
        project="  recall  ",
        since="7d",
        host=" aem5 ",
        per_session=2,
    )
    assert parsed.roles == (Role.USER, Role.ASSISTANT)
    assert parsed.project == "recall"
    assert parsed.since is not None
    assert parsed.host == "aem5"
    assert parsed.per_session == 2

    with pytest.raises(ValueError, match="unsupported role"):
        normalize_search_filters(role=["tool"])
    with pytest.raises(ValueError, match="--role cannot be combined with --tool"):
        normalize_search_filters(tool="Bash", role=["user"])
    with pytest.raises(ValueError, match="--role cannot be combined with --tool"):
        normalize_search_filters(tool="", role=["user"])
    with pytest.raises(ValueError, match="positive integer"):
        normalize_search_filters(per_session=0)
    with pytest.raises(ValueError, match="positive integer"):
        normalize_search_filters(per_session=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="project must not be empty"):
        normalize_search_filters(project="   ")
    with pytest.raises(ValueError, match="since must not be empty"):
        normalize_search_filters(since="")
    with pytest.raises(ValueError, match="host must not be empty"):
        normalize_search_filters(host="  ")

    assert not has_filtered_search_params(host="aem5", fleet=True)
    assert has_filtered_search_params(host="aem5", fleet=False)
    assert has_filtered_search_params(per_session=1, fleet=True)


@pytest.mark.parametrize("backend", ["sqlite_sidecar", "duckdb"])
@pytest.mark.parametrize("mode", [SearchMode.KEYWORD, SearchMode.VECTOR, SearchMode.HYBRID])
def test_search_filters_all_backends_and_modes(
    tmp_path: Path, backend: str, mode: SearchMode
) -> None:
    config = _make_config(tmp_path / f"{backend}-{mode.value}", backend=backend)
    _old, recent = _seed_filtered_corpus(config)
    cutoff = recent - timedelta(hours=1)
    q_emb = _vec(0) if mode in (SearchMode.VECTOR, SearchMode.HYBRID) else None

    # 1. --role user excludes assistant, system, AND tool_calls
    user_hits = search(
        query="needle",
        source=None,
        tool=None,
        roles=(Role.USER,),
        limit=20,
        mode=mode,
        config=config,
        query_embedding=q_emb,
    )
    assert user_hits
    assert all(r.kind == "message" and r.role == "user" for r in user_hits)

    # 2. Repeatable --role user + assistant excludes system & tool calls
    ua_hits = search(
        query="needle",
        source=None,
        tool=None,
        roles=(Role.USER, Role.ASSISTANT),
        limit=20,
        mode=mode,
        config=config,
        query_embedding=q_emb,
    )
    assert {r.role for r in ua_hits} == {"user", "assistant"}
    assert all(r.kind == "message" for r in ua_hits)

    # 3. --project case-insensitive substring on git_repo
    proj_hits = search(
        query="needle",
        source=None,
        tool=None,
        project="other",
        limit=20,
        mode=mode,
        config=config,
        query_embedding=q_emb,
    )
    assert proj_hits
    assert {r.session_id for r in proj_hits} == {"s3"}

    # 4. --since excludes old m1 even in recent s1, keeps NULL-ts m4 via session fallback
    since_hits = search(
        query="needle",
        source=None,
        tool=None,
        since=cutoff,
        limit=20,
        mode=mode,
        config=config,
        query_embedding=q_emb,
    )
    msg_ids = {r.message_id for r in since_hits if r.kind == "message"}
    tc_ids = {r.tool_call_id for r in since_hits if r.kind == "tool_call"}
    assert "m1" not in msg_ids
    assert "m9" not in msg_ids
    assert "m10" not in msg_ids
    assert "m2" in msg_ids
    assert "m4" in msg_ids  # NULL message timestamp falls back to recent s1.ended_at
    assert "tc5" not in tc_ids  # old session tool call excluded
    assert "tc1" in tc_ids  # recent session tool call included

    # 5. --host filters locally
    host_hits = search(
        query="needle",
        source=None,
        tool=None,
        host="ae-dev",
        limit=20,
        mode=mode,
        config=config,
        query_embedding=q_emb,
    )
    assert host_hits
    assert {r.session_id for r in host_hits} == {"s3", "s5"}

    # 6. --session + --host disambiguates shared source_session_id "src-1"
    disambig = search(
        query="needle",
        source=Source.CODEX,
        tool=None,
        session="src-1",
        host="ae-dev",
        limit=10,
        mode=mode,
        config=config,
        query_embedding=q_emb,
    )
    assert disambig
    assert {r.session_id for r in disambig} == {"s5"}

    # 7. --per-session 1 caps every session and fills exactly up to distinct sessions (5)
    capped = search(
        query="needle",
        source=None,
        tool=None,
        per_session=1,
        limit=4,
        mode=mode,
        config=config,
        query_embedding=q_emb,
    )
    assert len(capped) == 4
    assert len({r.session_id for r in capped}) == 4


@pytest.mark.parametrize("backend", ["sqlite_sidecar", "duckdb"])
def test_search_tool_filters_and_per_session(tmp_path: Path, backend: str) -> None:
    config = _make_config(tmp_path / backend, backend=backend)
    _old, recent = _seed_filtered_corpus(config)
    cutoff = recent - timedelta(hours=1)

    hits = search(
        query="needle",
        source=None,
        tool="Bash",
        project="recall",
        since=cutoff,
        host="aem5",
        per_session=1,
        limit=10,
        mode=SearchMode.KEYWORD,
        config=config,
    )
    assert len(hits) == 2
    assert {r.session_id for r in hits} == {"s1", "s2"}
    assert all(r.kind == "tool_call" for r in hits)


def test_sidecar_selective_per_session_prepares_candidates_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selective filter + per_session paging prepares temp.fts_candidates once, not per page."""
    config = _make_config(tmp_path, backend="sqlite_sidecar")
    with connect(config) as conn:
        # 20 background sessions so project="target" is < 10% selective
        sidecar = open_sidecar(sidecar_path(config.data_dir))
        try:
            for i in range(20):
                sid = f"bg-{i}"
                conn.execute(
                    "INSERT INTO sessions (id, source, source_path) VALUES (?, 'codex', ?)",
                    [sid, f"/bg/{i}.jsonl"],
                )
                conn.execute(
                    "INSERT INTO session_state (session_id, git_repo, host, message_count, "
                    "tool_count, is_complete, file_mtime, file_size) "
                    "VALUES (?, 'other', 'aem5', 20, 0, true, 1.0, 1)",
                    [sid],
                )
            # 2 target sessions: s-a has 12 high-scoring hits, s-b has 1 lower-scoring hit
            for sid, repo, count in (("t-a", "target-repo", 12), ("t-b", "target-repo", 1)):
                conn.execute(
                    "INSERT INTO sessions (id, source, source_path) VALUES (?, 'codex', ?)",
                    [sid, f"/t/{sid}.jsonl"],
                )
                conn.execute(
                    "INSERT INTO session_state (session_id, git_repo, host, message_count, "
                    "tool_count, is_complete, file_mtime, file_size) "
                    "VALUES (?, ?, 'aem5', ?, 0, true, 1.0, 1)",
                    [sid, repo, count],
                )
                for idx in range(count):
                    mid = f"{sid}-m{idx}"
                    # Make t-a messages match strongly, t-b match once
                    body = "needle " * (5 if sid == "t-a" else 1) + f"unique{idx}"
                    conn.execute(
                        "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
                        [mid, sid, idx],
                    )
                    conn.execute(
                        "INSERT INTO message_state "
                        "(message_id, role, content, fts_content, fts_thinking) "
                        "VALUES (?, 'assistant', ?, ?, '')",
                        [mid, body, body],
                    )
                    upsert_message_fts(sidecar, mid, body, "")
            set_fts_fields_signature(sidecar, config.fts.fields)
            sidecar.commit()
        finally:
            sidecar.close()

    # Force small page size so selective + per_session must fetch multiple SQLite pages
    monkeypatch.setattr(search_module, "_SIDECAR_FILTER_BATCH_SIZE", 4)
    prepare_calls = 0
    real_prepare = search_module.prepare_candidate_ids

    def counting_prepare(conn, ids):
        nonlocal prepare_calls
        prepare_calls += 1
        return real_prepare(conn, ids)

    monkeypatch.setattr(search_module, "prepare_candidate_ids", counting_prepare)

    results = search(
        query="needle",
        source=None,
        tool=None,
        roles=(Role.ASSISTANT,),
        project="target-repo",
        per_session=1,
        limit=2,
        mode=SearchMode.KEYWORD,
        config=config,
    )
    assert prepare_calls == 1
    assert {r.session_id for r in results} == {"t-a", "t-b"}
    assert len(results) == 2


def test_poll_snapshot_supports_search_filtered_without_mutating_stores(tmp_path: Path) -> None:
    config = _make_config(tmp_path, backend="sqlite_sidecar")
    _seed_filtered_corpus(config)
    db_before = config.db_path.read_bytes()
    fts_before = sidecar_path(config.data_dir).read_bytes()

    request = poll_read_request(
        "recall.search_filtered",
        {
            "query": "needle",
            "mode": "keyword",
            "role": ["user"],
            "project": "recall",
            "host": "aem5",
            "per_session": 1,
            "limit": 10,
        },
    )
    assert isinstance(request, SearchFilteredRead)
    result = read_poll_snapshot(config, request, ReadBudget.start(30))
    rows = result.data
    assert rows
    assert all(row["kind"] == "message" and row["role"] == "user" for row in rows)
    assert all(row["host"] == "aem5" for row in rows)
    assert len({row["session_id"] for row in rows}) == len(rows)
    assert config.db_path.read_bytes() == db_before
    assert sidecar_path(config.data_dir).read_bytes() == fts_before


class _FakeWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        return None


def test_rpc_server_handles_search_filtered_and_rejects_invalid_params(tmp_path: Path) -> None:
    config = _make_config(tmp_path, backend="sqlite_sidecar")
    _seed_filtered_corpus(config)
    server = RpcServer(config)

    async def call_rpc(method: str, params: dict[str, object]) -> dict[str, Any]:
        writer = _FakeWriter()
        raw = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1}).encode()
        await server._process_request(
            raw,
            cast(asyncio.StreamWriter, writer),
            cast(ClientConnection, None),
        )
        return json.loads(b"".join(writer.chunks))

    async def exercise() -> None:
        ok_resp = await call_rpc(
            "recall.search_filtered",
            {
                "query": "needle",
                "mode": "keyword",
                "role": ["assistant"],
                "project": "recall",
                "host": "aem5",
                "per_session": 1,
                "limit": 10,
            },
        )
        assert "result" in ok_resp, ok_resp
        rows = ok_resp["result"]
        assert rows
        assert all(row["role"] == "assistant" for row in rows)
        assert len({row["session_id"] for row in rows}) == len(rows)

        err_resp = await call_rpc(
            "recall.search_filtered",
            {"query": "needle", "tool": "Bash", "role": ["user"]},
        )
        assert err_resp["error"]["code"] == INVALID_PARAMS

    try:
        asyncio.run(exercise())
    finally:
        server._database.close()


@pytest.mark.parametrize("backend", ["sqlite_sidecar", "duckdb"])
@pytest.mark.parametrize("mode", [SearchMode.KEYWORD, SearchMode.VECTOR, SearchMode.HYBRID])
def test_since_aware_utc_excludes_older_naive_rows_on_non_utc_connection(
    tmp_path: Path, backend: str, mode: SearchMode
) -> None:
    """Finding #2: aware UTC cutoffs must compare as naive UTC even on TZ=America/Chicago."""
    config = _make_config(tmp_path / f"tz-{backend}-{mode.value}", backend=backend)
    with connect(config) as conn:
        conn.execute("SET TimeZone = 'America/Chicago'")
        for sid, ts in (
            ("s-old", datetime(2026, 1, 1, 0, 30, 0)),
            ("s-new", datetime(2026, 1, 1, 4, 0, 0)),
        ):
            conn.execute(
                "INSERT INTO sessions (id, source, source_path) VALUES (?, 'codex', ?)",
                [sid, f"/{sid}.jsonl"],
            )
            conn.execute(
                "INSERT INTO session_state (session_id, started_at, ended_at, host, "
                "message_count, tool_count, is_complete, file_mtime, file_size, indexed_at) "
                "VALUES (?, ?, ?, 'aem5', 1, 1, true, 1.0, 100, ?)",
                [sid, ts, ts, ts],
            )
            mid = f"m-{sid}"
            conn.execute("INSERT INTO messages (id, session_id, idx) VALUES (?, ?, 0)", [mid, sid])
            conn.execute(
                "INSERT INTO message_state (message_id, role, content, fts_content, fts_thinking, "
                "timestamp) VALUES (?, 'assistant', 'tzneedle body', 'tzneedle body', '', ?)",
                [mid, ts],
            )
            conn.execute(
                "INSERT INTO message_embeddings (message_id, content_embedding) VALUES (?, ?)",
                [mid, _vec(0)],
            )
            tcid = f"tc-{sid}"
            conn.execute(
                "INSERT INTO tool_calls (id, session_id, message_id, idx, tool_name, bash_command) "
                "VALUES (?, ?, ?, 0, 'Bash', 'tzneedle cmd')",
                [tcid, sid, mid],
            )
            conn.execute(
                "INSERT INTO tool_call_embeddings (tool_call_id, bash_embedding) VALUES (?, ?)",
                [tcid, _vec(0)],
            )
        if backend == "duckdb":
            create_fts_indexes(conn, config.fts, observer=StandaloneLifetime())
        else:
            sidecar = open_sidecar(sidecar_path(config.data_dir))
            try:
                for sid in ("s-old", "s-new"):
                    upsert_message_fts(sidecar, f"m-{sid}", "tzneedle body", "")
                    upsert_tool_call_fts(sidecar, f"tc-{sid}", "tzneedle cmd")
                set_fts_fields_signature(sidecar, config.fts.fields)
                sidecar.commit()
            finally:
                sidecar.close()

        aware_cutoff = datetime(2026, 1, 1, 3, 0, 0, tzinfo=UTC)
        q_emb = _vec(0) if mode in (SearchMode.VECTOR, SearchMode.HYBRID) else None
        results = search(
            query="tzneedle",
            source=None,
            tool=None,
            since=aware_cutoff,
            limit=10,
            mode=mode,
            config=config,
            conn=conn,
            query_embedding=q_emb,
        )
        assert results
        assert {r.session_id for r in results} == {"s-new"}


@pytest.mark.parametrize("backend", ["sqlite_sidecar", "duckdb"])
@pytest.mark.parametrize("mode", [SearchMode.KEYWORD, SearchMode.VECTOR, SearchMode.HYBRID])
def test_project_filter_matches_literal_substring_not_wildcards(
    tmp_path: Path, backend: str, mode: SearchMode
) -> None:
    """Finding #4: `--project acme_tools` and `%` must match literally, not as LIKE wildcards."""
    config = _make_config(tmp_path / f"proj-{backend}-{mode.value}", backend=backend)
    with connect(config) as conn:
        for sid, repo in (
            ("s-lit", "owner/acme_tools"),
            ("s-wild", "owner/acmeXtools"),
            ("s-pct", "owner/other"),
        ):
            conn.execute(
                "INSERT INTO sessions (id, source, source_path) VALUES (?, 'codex', ?)",
                [sid, f"/{sid}.jsonl"],
            )
            conn.execute(
                "INSERT INTO session_state (session_id, git_repo, host, message_count, "
                "tool_count, is_complete, file_mtime, file_size) "
                "VALUES (?, ?, 'aem5', 1, 0, true, 1.0, 100)",
                [sid, repo],
            )
            mid = f"m-{sid}"
            conn.execute("INSERT INTO messages (id, session_id, idx) VALUES (?, ?, 0)", [mid, sid])
            conn.execute(
                "INSERT INTO message_state (message_id, role, content, fts_content, fts_thinking) "
                "VALUES (?, 'assistant', 'wildneedle body', 'wildneedle body', '')",
                [mid],
            )
            conn.execute(
                "INSERT INTO message_embeddings (message_id, content_embedding) VALUES (?, ?)",
                [mid, _vec(0)],
            )
        if backend == "duckdb":
            create_fts_indexes(conn, config.fts, observer=StandaloneLifetime())
        else:
            sidecar = open_sidecar(sidecar_path(config.data_dir))
            try:
                for sid in ("s-lit", "s-wild", "s-pct"):
                    upsert_message_fts(sidecar, f"m-{sid}", "wildneedle body", "")
                set_fts_fields_signature(sidecar, config.fts.fields)
                sidecar.commit()
            finally:
                sidecar.close()

    q_emb = _vec(0) if mode in (SearchMode.VECTOR, SearchMode.HYBRID) else None
    under_hits = search(
        query="wildneedle",
        source=None,
        tool=None,
        project="ACME_TOOLS",
        limit=10,
        mode=mode,
        config=config,
        query_embedding=q_emb,
    )
    assert {r.session_id for r in under_hits} == {"s-lit"}

    pct_hits = search(
        query="wildneedle",
        source=None,
        tool=None,
        project="%",
        limit=10,
        mode=mode,
        config=config,
        query_embedding=q_emb,
    )
    assert pct_hits == []


def test_sidecar_keyset_paging_skips_long_capped_run_without_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding #3: sidecar paging uses keyset continuation (no OFFSET) across tied runs."""
    config = _make_config(tmp_path, backend="sqlite_sidecar")
    with connect(config) as conn:
        sidecar = open_sidecar(sidecar_path(config.data_dir))
        try:
            for sid, count, repeats in (("s-spam", 25, 4), ("s-late", 1, 1)):
                conn.execute(
                    "INSERT INTO sessions (id, source, source_path) VALUES (?, 'codex', ?)",
                    [sid, f"/{sid}.jsonl"],
                )
                conn.execute(
                    "INSERT INTO session_state (session_id, host, message_count, tool_count, "
                    "is_complete, file_mtime, file_size) VALUES (?, 'aem5', ?, 0, true, 1.0, 1)",
                    [sid, count],
                )
                for idx in range(count):
                    mid = f"{sid}-{idx:03d}"
                    body = ("keysetneedle " * repeats) + "same"
                    conn.execute(
                        "INSERT INTO messages (id, session_id, idx) VALUES (?, ?, ?)",
                        [mid, sid, idx],
                    )
                    conn.execute(
                        "INSERT INTO message_state "
                        "(message_id, role, content, fts_content, fts_thinking) "
                        "VALUES (?, 'assistant', ?, ?, '')",
                        [mid, body, body],
                    )
                    upsert_message_fts(sidecar, mid, body, "")
            set_fts_fields_signature(sidecar, config.fts.fields)
            sidecar.commit()
        finally:
            sidecar.close()

    monkeypatch.setattr(search_module, "_SIDECAR_FILTER_BATCH_SIZE", 5)
    seen_afters: list[tuple[float, str] | None] = []
    seen_offsets: list[int] = []
    real_search_fts = search_module.search_messages_fts

    def spy_search_fts(conn, query, fields, limit, **kwargs):
        seen_afters.append(kwargs.get("after"))
        seen_offsets.append(kwargs.get("offset", 0))
        return real_search_fts(conn, query, fields, limit, **kwargs)

    monkeypatch.setattr(search_module, "search_messages_fts", spy_search_fts)

    results = search(
        query="keysetneedle",
        source=None,
        tool=None,
        per_session=1,
        limit=2,
        mode=SearchMode.KEYWORD,
        config=config,
    )
    assert [r.session_id for r in results] == ["s-spam", "s-late"]
    assert len(seen_afters) >= 5
    assert seen_afters[0] is None
    assert all(cursor is not None for cursor in seen_afters[1:])
    assert all(off == 0 for off in seen_offsets)
