"""Semantic contracts of the owned, read-only poll snapshot (REQ-CLI-028)."""

from __future__ import annotations

import os
from contextlib import closing
from datetime import datetime

import pytest
from recall.core.rpc_types import APP_NOT_FOUND, RpcError
from recall.db import connect
from recall.db.fts_sidecar import (
    open_sidecar,
    set_fts_fields_signature,
    sidecar_path,
    upsert_message_fts,
    upsert_tool_call_fts,
)
from recall.db.ownership import offline_ownership
from recall.services.live import decode_cursor

from tests.test_services.test_database_ownership import config_at
from tests.test_services.test_fts_sidecar_search import _vec
from tests.test_services.test_poll_reads import snapshot

MAIN = "abcdef01000000000000000000000001"
OTHER = "abcdef02000000000000000000000002"
ARCHIVE = "fedcba03000000000000000000000003"


def preserved_snapshot(config, method="recall.stats", params=None):
    """Every semantic read must leave both files intact and release ownership."""
    sidecar = sidecar_path(config.data_dir)
    before = (config.db_path.read_bytes(), sidecar.read_bytes())
    try:
        return snapshot(config, method, params)
    finally:
        assert (config.db_path.read_bytes(), sidecar.read_bytes()) == before
        with offline_ownership(config.db_path):
            pass


@pytest.fixture
def semantic_index(tmp_path, monkeypatch):
    config = config_at(tmp_path)
    monkeypatch.setattr("recall.services.analytics.default_session_host", lambda: "test-host")
    with (
        connect(config) as conn,
        closing(open_sidecar(sidecar_path(config.data_dir))) as sidecar,
        sidecar,
    ):
        for sid, source, source_id, day, repo, host, tokens in [
            (MAIN, "codex", "codex-main", 2, "alpha", "desk", (100, 20, 25)),
            (OTHER, "codex", "shared-source", 3, "beta", "desk", (300, 40, 50)),
            (ARCHIVE, "grok", "shared-source", 1, "alpha", "archive", (50, 10, None)),
        ]:
            path = tmp_path / f"{sid}.jsonl"
            path.write_text("application-owned transcript\n")
            stat = path.stat()
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?)",
                [sid, source, str(path), source_id],
            )
            conn.execute(
                """INSERT INTO session_state (
                    session_id, started_at, ended_at, model, cwd, git_repo, host,
                    message_count, tool_count, input_tokens, output_tokens,
                    cached_input_tokens, file_mtime, file_size, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    sid,
                    datetime(2026, 1, day),
                    datetime(2026, 1, day, 1),
                    "model-g" if source == "grok" else "model-c",
                    f"/projects/{repo}",
                    f"/projects/{repo}",
                    host,
                    4 if sid == MAIN else 1,
                    3 if sid == MAIN else 1,
                    *tokens,
                    stat.st_mtime,
                    stat.st_size,
                    datetime(2026, 1, 4),
                ],
            )
            if sid == MAIN:
                conn.execute(
                    """INSERT INTO source_files (
                        source_key, source, source_path, root_path, session_id,
                        dev, inode, ctime_ns, mtime_ns, size, sidecar_signature,
                        desired_generation, committed_generation, committed_offset,
                        content_epoch, observed_at
                    ) VALUES ('main', 'codex', ?, ?, ?, ?, ?, ?, ?, ?, ?, 3, 3, ?, 0, 0)""",
                    [
                        str(path),
                        str(tmp_path),
                        sid,
                        stat.st_dev,
                        stat.st_ino,
                        stat.st_ctime_ns,
                        stat.st_mtime_ns,
                        stat.st_size,
                        # Codex declares no sidecars: SHA-256 of the empty stream.
                        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                        stat.st_size,
                    ],
                )
        for mid, sid, idx, role, content, embedding in [
            ("a0", MAIN, 0, "user", "needle in the orchard", _vec(0)),
            ("a1", MAIN, 1, "assistant", "semantic neighbor", _vec(1)),
            ("a2", MAIN, 2, "user", "second question", None),
            ("a3", MAIN, 3, "assistant", "final answer", None),
            ("b0", OTHER, 0, "assistant", "needle in the forest", _vec(1)),
            ("c0", ARCHIVE, 0, "assistant", "needle in the garden", _vec(1)),
        ]:
            conn.execute("INSERT INTO messages VALUES (?, ?, ?, NULL)", [mid, sid, idx])
            conn.execute(
                """INSERT INTO message_state (
                    message_id, role, content, thinking, has_thinking,
                    context_text, context_mode, fts_content
                ) VALUES (?, ?, ?, 'private reasoning', true, 'orchard context',
                          'template', ?)""",
                [mid, role, content, content],
            )
            upsert_message_fts(sidecar, mid, content, "private reasoning")
            if embedding is not None:
                conn.execute(
                    "INSERT INTO message_embeddings(message_id,content_embedding) VALUES (?,?)",
                    [mid, embedding],
                )
        for tid, sid, mid, idx, name, command, base, sub, tool_input in [
            ("git1", MAIN, "a1", 0, "bash", "git status needle", "git", "status", None),
            ("git2", MAIN, "a3", 1, "bash", "git status needle", "git", "status", None),
            ("rm", OTHER, "b0", 0, "bash", "rm needle", "rm", None, None),
            ("skill1", MAIN, "a2", 2, "Skill", None, None, None, '{"skill":"orchard"}'),
            ("skill2", ARCHIVE, "c0", 0, "Skill", None, None, None, '{"skill":"garden"}'),
        ]:
            conn.execute(
                """INSERT INTO tool_calls (
                    id, session_id, message_id, idx, tool_name, bash_command,
                    bash_base, bash_sub, tool_input
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [tid, sid, mid, idx, name, command, base, sub, tool_input],
            )
            if command:
                upsert_tool_call_fts(sidecar, tid, command)
        conn.execute(
            """UPDATE runtime_state SET last_successful_at='2026-01-04 12:34:56',
                last_index_total=13, last_index_indexed=7, last_index_skipped=4,
                last_index_failed=2, last_context_messages=6,
                last_context_mode='llm-remote', last_context_input_tokens=1234,
                last_context_output_tokens=321, last_context_model='context-fixture'"""
        )
        set_fts_fields_signature(sidecar, config.fts.fields)
    return config


def test_stats_counts_runtime_context_and_snapshot_provenance(semantic_index):
    result = preserved_snapshot(semantic_index)
    assert result.data == {
        "sessions": 3,
        "messages": 6,
        "tool_calls": 5,
        "bash_calls": 3,
        "last_index_total": 13,
        "last_index_indexed": 7,
        "last_index_skipped": 4,
        "last_index_failed": 2,
        "last_context_messages": 6,
        "last_context_mode": "llm-remote",
        "last_context_input_tokens": 1234,
        "last_context_output_tokens": 321,
        "last_context_model": "context-fixture",
    }
    assert result.last_successful_at == datetime(2026, 1, 4, 12, 34, 56)
    assert result.coverage is not None


@pytest.mark.parametrize(
    ("method", "params", "expected"),
    [
        (
            "recall.stats_tools",
            {},
            [
                {"tool_name": "bash", "count": 3},
                {"tool_name": "Skill", "count": 2},
            ],
        ),
        (
            "recall.stats_bash",
            {},
            [
                {"bash_base": "git", "bash_sub": "status", "count": 2, "is_compound": False},
                {"bash_base": "rm", "bash_sub": None, "count": 1, "is_compound": False},
            ],
        ),
        (
            "recall.stats_bash",
            {"suggest": True},
            {
                "suggestions": [
                    {
                        "pattern": "git status",
                        "count": 2,
                        "confidence": "review",
                        "reason": "Low usage volume",
                    }
                ],
                "skipped": [{"pattern": "rm *", "count": 1, "reason": "Destructive command"}],
            },
        ),
        (
            "recall.stats_tokens",
            {},
            [
                {"repo": "/projects/beta", "input_tokens": 300, "output_tokens": 40},
                {"repo": "/projects/alpha", "input_tokens": 150, "output_tokens": 30},
            ],
        ),
        (
            "recall.stats_usage",
            {},
            [
                {
                    "source": "codex",
                    "model": "model-c",
                    "host": "desk",
                    "input_tokens": 400,
                    "cached_input_tokens": 75,
                    "output_tokens": 60,
                    "fresh_input_tokens": 325,
                    "session_count": 2,
                },
                {
                    "source": "grok",
                    "model": "model-g",
                    "host": "archive",
                    "input_tokens": 50,
                    "cached_input_tokens": 0,
                    "output_tokens": 10,
                    "fresh_input_tokens": None,
                    "session_count": 1,
                },
            ],
        ),
        (
            "recall.stats_usage",
            {"since": "2026-01-03"},
            [
                {
                    "source": "codex",
                    "model": "model-c",
                    "host": "desk",
                    "input_tokens": 300,
                    "cached_input_tokens": 50,
                    "output_tokens": 40,
                    "fresh_input_tokens": 250,
                    "session_count": 1,
                },
            ],
        ),
    ],
)
def test_analytics_methods_return_indexed_answers(semantic_index, method, params, expected):
    assert preserved_snapshot(semantic_index, method, params).data == expected


@pytest.mark.parametrize(
    ("params", "expected_rows", "population", "control"),
    [
        (
            {"source": ["codex", "grok"]},
            [
                {
                    "skill_name": "garden",
                    "source": "grok",
                    "host": "test-host",
                    "invocations": 1,
                    "sessions": 1,
                },
                {
                    "skill_name": "orchard",
                    "source": "codex",
                    "host": "test-host",
                    "invocations": 1,
                    "sessions": 1,
                },
            ],
            (3, 2),
            (3, 2),
        ),
        ({"source": ["codex", "codex"], "since": "2026-01-03"}, [], (1, 0), (2, 1)),
        (
            {"source": ["grok"], "since": "2026-01-01"},
            [
                {
                    "skill_name": "garden",
                    "source": "grok",
                    "host": "test-host",
                    "invocations": 1,
                    "sessions": 1,
                },
            ],
            (1, 1),
            (1, 1),
        ),
    ],
)
def test_skill_stats_filters_and_coverage(
    semantic_index, params, expected_rows, population, control
):
    data = preserved_snapshot(semantic_index, "recall.stats_skills", params).data
    assert data["rows"] == expected_rows
    assert data["coverage"] == {
        "scope": "local",
        "expected_hosts": ["test-host"],
        "successful_hosts": ["test-host"],
        "covered_sources": list(dict.fromkeys(params["source"])),
        "considered_sessions": population[0],
        "attributed_invocations": population[1],
        "unattributed_candidates": 0,
        "control": {
            "considered_sessions": control[0],
            "attributed_invocations": control[1],
            "unattributed_candidates": 0,
        },
    }


@pytest.mark.parametrize(
    ("method", "params", "expected"),
    [
        ("recall.list", {}, [OTHER, MAIN, ARCHIVE]),
        ("recall.list", {"source": "grok"}, [ARCHIVE]),
        ("recall.list", {"since": "2026-01-03"}, [OTHER]),
        ("recall.list", {"project": "ALPHA"}, [MAIN, ARCHIVE]),
        ("recall.list", {"host": "archive"}, [ARCHIVE]),
        ("recall.list", {"session": "codex-main"}, [MAIN]),
        (
            "recall.list",
            {
                "source": "codex",
                "since": "2026-01-02",
                "project": "alpha",
                "host": "desk",
                "session": MAIN,
                "limit": 1,
            },
            [MAIN],
        ),
        ("recall.list", {"source": "grok", "host": "desk"}, []),
        ("recall.list", {"limit": 1}, [OTHER]),
        ("recall.list_by_session", {"session": "codex-main"}, [MAIN]),
        (
            "recall.list_by_session",
            {"session": "abcdef01", "source": "codex", "host": "desk", "project": "alpha"},
            [MAIN],
        ),
        ("recall.list_by_session", {"session": MAIN, "since": "2026-01-03"}, []),
    ],
)
def test_list_filters_and_session_method(semantic_index, method, params, expected):
    rows = preserved_snapshot(semantic_index, method, params).data
    assert [row["id"] for row in rows] == expected
    if MAIN in expected:
        row = next(row for row in rows if row["id"] == MAIN)
        assert (row["source_session_id"], row["message_count"], row["tool_count"], row["host"]) == (
            "codex-main",
            4,
            3,
            "desk",
        )


@pytest.fixture
def query_backend(monkeypatch):
    class Backend:
        query_prefix = "query: "

        def embed(self, texts):
            assert texts == ["query: needle"]
            return [_vec(0)]

    monkeypatch.setattr("recall.services.embeddings.get_backend", lambda _config: Backend())
    monkeypatch.setattr("recall.services.embeddings.any_backend_available", lambda: True)


@pytest.mark.parametrize("mode", ["keyword", "vector", "hybrid", "auto"])
@pytest.mark.parametrize(
    ("filters", "lexical", "vector"),
    [
        ({}, {"a0", "b0", "c0", "git1", "git2", "rm"}, {"a0", "a1", "b0", "c0"}),
        ({"source": "codex"}, {"a0", "b0", "git1", "git2", "rm"}, {"a0", "a1", "b0"}),
        ({"session": "codex-main"}, {"a0", "git1", "git2"}, {"a0", "a1"}),
        ({"source": "codex", "session": "abcdef01"}, {"a0", "git1", "git2"}, {"a0", "a1"}),
        ({"tool": "bash"}, {"git1", "git2", "rm"}, set()),
        ({"tool": "bash", "source": "codex"}, {"git1", "git2", "rm"}, set()),
        ({"tool": "bash", "session": MAIN}, {"git1", "git2"}, set()),
        ({"tool": "bash", "source": "codex", "session": MAIN}, {"git1", "git2"}, set()),
        ({"source": "grok"}, {"c0"}, {"c0"}),
        ({"source": "pi_agent"}, set(), set()),
        ({"tool": "Read"}, set(), set()),
        ({"tool": "bash", "source": "grok", "session": ARCHIVE}, set(), set()),
    ],
)
def test_search_modes_and_every_filter_combination(
    semantic_index, query_backend, mode, filters, lexical, vector
):
    rows = preserved_snapshot(
        semantic_index, "recall.search", {"query": "needle", "mode": mode, **filters}
    ).data
    if filters.get("tool") or mode == "keyword":
        expected = lexical
    elif mode == "vector":
        expected = vector
    else:
        expected = lexical | vector
    assert {
        row["message_id"] if row["kind"] == "message" else row["tool_call_id"] for row in rows
    } == expected
    assert len(rows) == len(expected)
    for row in rows:
        key = row["message_id"] if row["kind"] == "message" else row["tool_call_id"]
        assert row["lexical_match"] is (
            (key in lexical and mode != "vector") or bool(filters.get("tool"))
        )
        assert row["host"] == ("archive" if key == "c0" else "desk")


@pytest.mark.parametrize(
    ("mode", "expected_message", "expected_tool"),
    [
        ("keyword", "a1", "git1"),
        ("vector", "a0", None),
        ("hybrid", "a0", None),
        ("auto", "a0", None),
    ],
)
def test_search_limit_and_ranking(
    semantic_index, query_backend, mode, expected_message, expected_tool
):
    rows = preserved_snapshot(
        semantic_index,
        "recall.search",
        {"query": "needle", "mode": mode, "session": MAIN, "limit": 1},
    ).data
    assert len(rows) == 1
    # The two identical commands tie lexically, so git1 wins their ID tie.
    # Within this session a0 appears in both hybrid legs and outranks either command.
    assert (rows[0]["message_id"], rows[0]["tool_call_id"]) == (expected_message, expected_tool)


@pytest.mark.parametrize("identifier", [MAIN, "codex-main", "abcdef01"])
def test_show_resolves_identifiers_and_preserves_message_fields(semantic_index, identifier):
    result = preserved_snapshot(semantic_index, "recall.show", {"session_id": identifier})
    data = result.data
    assert (data["id"], data["source_session_id"], data["host"]) == (MAIN, "codex-main", "desk")
    assert [(m["id"], m["idx"], m["content"]) for m in data["messages"]] == [
        ("a0", 0, "needle in the orchard"),
        ("a1", 1, "semantic neighbor"),
        ("a2", 2, "second question"),
        ("a3", 3, "final answer"),
    ]
    assert data["messages"][1]["thinking"] == "private reasoning"
    assert data["messages"][1]["context_text"] == "orchard context"
    assert data["messages"][1]["context_mode"] == "template"
    assert all(m["tool_calls"] == [] for m in data["messages"])
    cursor = decode_cursor(data["cursor"])
    assert (cursor.session_id, cursor.last_idx, cursor.content_epoch) == (MAIN, 3, 0)
    assert data["cursor_reset"] is False
    assert data["cursor_reset_reason"] is None
    assert data["freshness"]["current"] is True
    assert result.coverage is None


@pytest.mark.parametrize(
    ("identifier", "kind"),
    [
        ("abcdef", "session id prefix"),
        ("shared-source", "source_session_id"),
    ],
)
def test_show_ambiguity_names_candidates_without_guessing(semantic_index, identifier, kind):
    with pytest.raises(ValueError, match=f"ambiguous {kind}") as error:
        preserved_snapshot(semantic_index, "recall.show", {"session_id": identifier})
    candidates = (MAIN, OTHER) if identifier == "abcdef" else (OTHER, ARCHIVE)
    for candidate in candidates:
        assert candidate in str(error.value)
        assert (
            preserved_snapshot(semantic_index, "recall.show", {"session_id": candidate}).data["id"]
            == candidate
        )


def test_show_head_tail_after_and_window_tools(semantic_index):
    head = preserved_snapshot(
        semantic_index,
        "recall.show",
        {
            "session_id": MAIN,
            "message_limit": 2,
            "tools": True,
        },
    ).data
    assert [m["id"] for m in head["messages"]] == ["a0", "a1"]
    assert [tc["id"] for tc in head["messages"][1]["tool_calls"]] == ["git1"]
    cursor = head["cursor"]
    delta = preserved_snapshot(
        semantic_index,
        "recall.show",
        {
            "session_id": "codex-main",
            "after": cursor,
            "tools": True,
        },
    ).data
    assert [m["id"] for m in delta["messages"]] == ["a2", "a3"]
    assert [[tc["id"] for tc in m["tool_calls"]] for m in delta["messages"]] == [
        ["skill1"],
        ["git2"],
    ]
    assert delta["orphan_tool_calls"] == []
    tail = preserved_snapshot(semantic_index, "recall.show", {"session_id": MAIN, "tail": 2}).data
    assert [m["id"] for m in tail["messages"]] == ["a2", "a3"]
    assert all(m["tool_calls"] == [] for m in tail["messages"])
    combined = preserved_snapshot(
        semantic_index,
        "recall.show",
        {
            "session_id": MAIN,
            "after": cursor,
            "tail": 1,
        },
    ).data
    assert [m["id"] for m in combined["messages"]] == ["a3"]
    empty = preserved_snapshot(
        semantic_index,
        "recall.show",
        {
            "session_id": MAIN,
            "after": delta["cursor"],
        },
    ).data
    assert empty["messages"] == []
    assert empty["cursor"] == delta["cursor"]
    assert empty["cursor_reset"] is False


def test_show_resets_continuation_after_indexed_rewrite(semantic_index):
    old = preserved_snapshot(semantic_index, "recall.show", {"session_id": MAIN, "tail": 1}).data
    with offline_ownership(semantic_index.db_path), connect(semantic_index) as conn:
        conn.execute("UPDATE source_files SET content_epoch=1 WHERE session_id=?", [MAIN])
        conn.execute("UPDATE message_state SET content='rewritten beginning' WHERE message_id='a0'")
    reset = preserved_snapshot(
        semantic_index,
        "recall.show",
        {
            "session_id": MAIN,
            "after": old["cursor"],
        },
    ).data
    assert reset["cursor_reset"] is True
    assert reset["cursor_reset_reason"] == "content_rewritten"
    assert [m["id"] for m in reset["messages"]] == ["a0", "a1", "a2", "a3"]
    assert reset["messages"][0]["content"] == "rewritten beginning"
    cursor = decode_cursor(reset["cursor"])
    assert (cursor.session_id, cursor.last_idx, cursor.content_epoch) == (MAIN, 3, 1)
    tail_reset = preserved_snapshot(
        semantic_index,
        "recall.show",
        {"session_id": MAIN, "after": old["cursor"], "tail": 1},
    ).data
    assert tail_reset["cursor_reset"] is True
    assert tail_reset["cursor_reset_reason"] == "content_rewritten"
    assert [m["id"] for m in tail_reset["messages"]] == ["a3"]
    continued = preserved_snapshot(
        semantic_index,
        "recall.show",
        {
            "session_id": MAIN,
            "after": reset["cursor"],
        },
    ).data
    assert continued["messages"] == []
    assert continued["cursor_reset"] is False


@pytest.mark.parametrize("mode", ["keyword", "vector", "hybrid", "auto"])
def test_search_refuses_session_outside_source_scope(semantic_index, query_backend, mode):
    with pytest.raises(ValueError, match="outside the requested source/host scope"):
        preserved_snapshot(
            semantic_index,
            "recall.search",
            {"query": "needle", "mode": mode, "source": "grok", "session": MAIN},
        )


def test_show_reports_changed_transcript_without_reading_unindexed_content(semantic_index):
    before = preserved_snapshot(semantic_index, "recall.show", {"session_id": MAIN}).data
    assert before["freshness"]["validated"] is True
    assert before["freshness"]["limitations"] == []
    path = semantic_index.data_dir / f"{MAIN}.jsonl"
    indexed_size = path.stat().st_size
    indexed_mtime = path.stat().st_mtime
    addition = "unindexed append\n"
    with path.open("a") as stream:
        stream.write(addition)
    os.utime(path, (indexed_mtime + 10, indexed_mtime + 10))
    stale = preserved_snapshot(semantic_index, "recall.show", {"session_id": MAIN}).data
    assert stale["messages"] == before["messages"]
    assert stale["cursor"] == before["cursor"]
    assert stale["freshness"] == {
        "file_mtime": indexed_mtime + 10,
        "file_size": indexed_size + len(addition),
        "indexed_mtime": indexed_mtime,
        "indexed_size": indexed_size,
        "lag_seconds": 10.0,
        "lag_bytes": len(addition),
        "current": False,
        "validated": False,
        "generation": 3,
        "content_epoch": 0,
        "limitations": ["catalog_signature_mismatch", "catalog_boundary_incomplete"],
    }


@pytest.mark.parametrize(
    ("method", "params", "expected"),
    [
        ("recall.list", {}, []),
        ("recall.search", {"query": "absent", "mode": "keyword"}, []),
        ("recall.stats_tools", {}, []),
        ("recall.stats_tokens", {}, []),
        ("recall.stats_usage", {}, []),
        ("recall.stats_bash", {}, []),
        ("recall.stats_bash", {"suggest": True}, {"suggestions": [], "skipped": []}),
        (
            "recall.stats",
            {},
            {
                "sessions": 0,
                "messages": 0,
                "tool_calls": 0,
                "bash_calls": 0,
                "last_index_total": 0,
                "last_index_indexed": 0,
                "last_index_skipped": 0,
                "last_index_failed": 0,
                "last_context_messages": 0,
                "last_context_mode": "off",
                "last_context_input_tokens": 0,
                "last_context_output_tokens": 0,
                "last_context_model": None,
            },
        ),
    ],
)
def test_empty_methods_have_truthful_defaults(tmp_path, method, params, expected):
    config = config_at(tmp_path)
    with connect(config), closing(open_sidecar(sidecar_path(config.data_dir))) as sidecar, sidecar:
        set_fts_fields_signature(sidecar, config.fts.fields)
    result = preserved_snapshot(config, method, params)
    assert result.data == expected
    assert result.last_successful_at is None


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("recall.show", {"session_id": "absent"}),
        ("recall.list_by_session", {"session": "absent"}),
    ],
)
def test_empty_identifier_reads_refuse_without_creating_rows(tmp_path, method, params):
    config = config_at(tmp_path)
    with connect(config), closing(open_sidecar(sidecar_path(config.data_dir))) as sidecar, sidecar:
        set_fts_fields_signature(sidecar, config.fts.fields)
    if method == "recall.show":
        with pytest.raises(RpcError, match="session not found") as error:
            preserved_snapshot(config, method, params)
        assert error.value.code == APP_NOT_FOUND
    else:
        with pytest.raises(ValueError, match="session not found"):
            preserved_snapshot(config, method, params)


def test_empty_skill_population_is_reported(tmp_path, monkeypatch):
    config = config_at(tmp_path)
    monkeypatch.setattr("recall.services.analytics.default_session_host", lambda: "test-host")
    with connect(config), closing(open_sidecar(sidecar_path(config.data_dir))) as sidecar, sidecar:
        set_fts_fields_signature(sidecar, config.fts.fields)
    result = preserved_snapshot(config, "recall.stats_skills", {"source": ["codex"]})
    assert result.data == {
        "rows": [],
        "coverage": {
            "scope": "local",
            "expected_hosts": ["test-host"],
            "successful_hosts": ["test-host"],
            "covered_sources": ["codex"],
            "considered_sessions": 0,
            "attributed_invocations": 0,
            "unattributed_candidates": 0,
            "control": {
                "considered_sessions": 0,
                "attributed_invocations": 0,
                "unattributed_candidates": 0,
            },
        },
    }
    assert result.last_successful_at is None
