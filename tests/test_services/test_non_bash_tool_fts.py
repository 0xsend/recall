"""Tests for PR 4 non-bash tool FTS derivation, crash-safe rescope, and snippet parity."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import duckdb
import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.models import Message, Session, TailFacts, ToolCall
from recall.core.poll_reads import poll_read_request
from recall.core.snippets import SNIPPET_MAX_CHARS, build_search_snippet
from recall.core.tool_format import (
    derive_tool_call_fts_text,
    parse_tool_input_mapping,
)
from recall.core.types import Role, SearchMode, Source
from recall.db import connect
from recall.db.fts_sidecar import (
    FTS_DERIVATION_VERSION,
    SIDECAR_SCHEMA_VERSION,
    content_fingerprint,
    fts_fields_signature,
    get_fts_fields_signature,
    get_rescope_in_progress,
    open_sidecar,
    rescope_kinds_needed,
    search_tool_calls_fts,
    set_fts_fields_signature,
    set_rescope_in_progress,
    sidecar_path,
    upsert_message_fts,
    upsert_tool_call_fts,
)
from recall.db.lease import StandaloneLifetime
from recall.db.schema import SCHEMA_VERSION, ensure_schema, get_schema_version
from recall.read_budget import ReadBudget
from recall.services.daemon import _rescope_sidecar_fields_if_needed
from recall.services.fts_sidecar_bootstrap import bootstrap_sidecar
from recall.services.fts_sidecar_rebuild import verify_candidate
from recall.services.fts_sidecar_reconcile import (
    _partition_by_byte_budget,
    iter_tool_call_fts_batches,
    iter_tool_call_fts_chunks_for_ids,
    iter_tool_call_inputs_bounded,
    reconcile_sidecar,
    repair_pending_batch,
)
from recall.services.indexer import (
    _load_session_write_snapshot,
    _searchable_rows_changed,
    _write_session,
)
from recall.services.manual_keywords import require_replaceable_sidecar
from recall.services.poll_reads import read_poll_snapshot
from recall.services.search import search

from tests.keyword_turns import without_keyword_index


def _config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(backend="sqlite_sidecar"),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )


def test_derive_tool_call_fts_text_determinism_and_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Non-NULL bash_command preserved verbatim, including whitespace-only historical rows
    assert derive_tool_call_fts_text("Bash", {"file_path": "/a"}, "git status") == "git status"
    assert derive_tool_call_fts_text("Bash", {"file_path": "/a"}, " \t") == " \t"

    # 2. Non-bash harnesses omit tool_name so generic verbs don't pollute general search
    read_text = derive_tool_call_fts_text("Read", {"file_path": "/repo/src/search.py"}, None)
    assert read_text == "/repo/src/search.py"
    assert "read" not in read_text.lower()

    # 3. MCP fallback key order is deterministic between live dict and DuckDB sorted JSON
    live_mcp = {"z": "needle", "y": 2, "x": 3, "a": 4}
    persisted_mcp = json.dumps(live_mcp, sort_keys=True)
    assert derive_tool_call_fts_text("mcp_tool", live_mcp, None) == derive_tool_call_fts_text(
        "mcp_tool", persisted_mcp, None
    )
    assert derive_tool_call_fts_text("mcp_tool", live_mcp, None) == "a=4 x=3 y=2"

    # 4. Case-insensitive key collisions resolve identically between live dict and sorted JSON
    live_collide = {"path": "lower.py", "Path": "upper.py"}
    persisted_collide = json.dumps(live_collide, sort_keys=True)
    assert derive_tool_call_fts_text("Read", live_collide, None) == derive_tool_call_fts_text(
        "Read", persisted_collide, None
    )

    # 5. Deeply nested JSON (RecursionError) and malformed JSON degrade cleanly to ""
    nested = "[" * 2000 + "1" + "]" * 2000
    assert parse_tool_input_mapping(nested) is None
    assert derive_tool_call_fts_text("mcp_tool", nested, None) == ""
    assert derive_tool_call_fts_text("mcp_tool", "{not-json", None) == ""

    # 6. Oversized tool_input (> cap) returns "" identically for live Mapping and persisted JSON
    monkeypatch.setattr("recall.core.tool_format.TOOL_INPUT_MAX_PARSE_BYTES", 256)
    big_live = {"file_path": "src/needle.py", "content": "x" * 512}
    big_json = json.dumps(big_live, sort_keys=True)
    assert derive_tool_call_fts_text("Write", big_live, None) == ""
    assert derive_tool_call_fts_text("Write", big_json, None) == ""

    # 7. Lone UTF-16 surrogates are normalized cleanly for SQLite bindings
    surrogate_text = derive_tool_call_fts_text("Read", {"file_path": "/tmp/bad\ud800.py"}, None)
    surrogate_text.encode("utf-8")  # must not raise UnicodeEncodeError


def test_tool_call_snippet_parity_prefix_bound_and_centering() -> None:
    # 1. apply_patch index and snippet extract identical file target even with OSC escapes
    patch_input = {
        "patch": (
            "*** Begin Patch\n"
            "*** Update File: \x1b]0;evil\x07src/needle.py\n"
            "@@ -1 +1 @@\n-old\n+new\n*** End Patch"
        )
    }
    snippet = build_search_snippet(
        query="needle",
        kind="tool_call",
        content=None,
        thinking=None,
        role=None,
        tool_name="apply_patch",
        bash_command=None,
        tool_input=patch_input,
    )
    assert snippet == "[apply_patch] src/needle.py"
    assert "\x1b" not in snippet
    assert "evil" not in snippet

    # 2. Unconstrained 250-char tool_name cannot exhaust or break the 240-char snippet budget
    long_name = "mcp__" + ("server_" * 40)
    long_name_snippet = build_search_snippet(
        query="needle",
        kind="tool_call",
        content=None,
        thinking=None,
        role=None,
        tool_name=long_name,
        bash_command=None,
        tool_input={"file_path": "/repo/src/needle.py"},
    )
    assert len(long_name_snippet) <= SNIPPET_MAX_CHARS
    assert "needle.py" in long_name_snippet

    # 3. Long detail (400+ chars) centers around late match while keeping [tool_name] prefix
    long_prompt = ("filler words " * 25) + "critical_target_token appears near the end"
    centered = build_search_snippet(
        query="critical_target_token",
        kind="tool_call",
        content=None,
        thinking=None,
        role=None,
        tool_name="Agent",
        bash_command=None,
        tool_input={"subagent_type": "research", "description": long_prompt},
    )
    assert centered.startswith("[Agent] ")
    assert len(centered) <= SNIPPET_MAX_CHARS
    assert "critical_target_token" in centered


def test_v2_to_v3_rescope_skips_messages_preserves_bash_and_indexes_non_bash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    with connect(config) as conn:
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) "
            "VALUES ('s1', 'antigravity_cli', '/tmp/s1.jsonl')"
        )
        conn.execute(
            "INSERT INTO session_state (session_id, message_count, tool_count, is_complete, "
            "file_mtime, file_size) VALUES ('s1', 1, 4, true, 1.0, 100)"
        )
        conn.execute("INSERT INTO messages (id, session_id, idx) VALUES ('m1', 's1', 0)")
        conn.execute(
            "INSERT INTO message_state (message_id, role, content, fts_content, fts_thinking) "
            "VALUES ('m1', 'assistant', 'hello world', 'hello world', '')"
        )
        tool_rows = [
            ("tc-bash", "Bash", None, "git status"),
            ("tc-ws-bash", "Bash", json.dumps({"file_path": "/ignored"}), " \t"),
            ("tc-read", "read_file", json.dumps({"Path": "/repo/search_filters.py"}), None),
            ("tc-empty", "scheduler_list", json.dumps({}), None),
        ]
        for idx, (tcid, tname, tinput, bcmd) in enumerate(tool_rows):
            conn.execute(
                "INSERT INTO tool_calls (id, session_id, message_id, idx, tool_name, "
                "tool_input, bash_command) VALUES (?, 's1', 'm1', ?, ?, ?, ?)",
                [tcid, idx, tname, tinput, bcmd],
            )

        # Seed a |v2 sidecar containing only messages and bash tool calls
        sidecar = open_sidecar(sidecar_path(config.data_dir))
        try:
            upsert_message_fts(sidecar, "m1", "hello world", "")
            upsert_tool_call_fts(sidecar, "tc-bash", "git status")
            upsert_tool_call_fts(sidecar, "tc-ws-bash", " \t")
            with sidecar:
                sidecar.execute(
                    "INSERT OR REPLACE INTO fts_sidecar_meta(key, value) "
                    "VALUES ('fts_fields_signature', 'bash,content,thinking|v2')"
                )
            bash_fp_before = sidecar.execute(
                "SELECT tool_call_id, content_fingerprint FROM tool_calls_fts_rowid ORDER BY 1"
            ).fetchall()

            # Prove message rescope is NOT called during clean |v2 -> |v3 upgrade
            def fail_if_messages_rescoped(*_args, **_kwargs):
                raise AssertionError("_rescope_message_rows should be skipped on clean |v2 -> |v3")

            monkeypatch.setattr(
                "recall.services.fts_sidecar_reconcile._rescope_message_rows",
                fail_if_messages_rescoped,
            )

            with StandaloneLifetime().lease(conn) as lease:
                _rescope_sidecar_fields_if_needed(
                    lease,
                    conn,
                    sidecar,
                    fts_fields=("content", "thinking", "bash"),
                    sidecar_existed=True,
                )

            assert get_schema_version(conn) == SCHEMA_VERSION == 41
            assert FTS_DERIVATION_VERSION == 3
            assert get_fts_fields_signature(sidecar) == fts_fields_signature(
                ("content", "thinking", "bash")
            )
            assert get_rescope_in_progress(sidecar) == frozenset()

            # Existing bash fingerprints untouched (including whitespace-only bash_command)
            after_fps = dict(
                sidecar.execute(
                    "SELECT tool_call_id, content_fingerprint FROM tool_calls_fts_rowid"
                ).fetchall()
            )
            for tcid, fp in bash_fp_before:
                assert after_fps[tcid] == fp
            assert set(after_fps.keys()) == {"tc-bash", "tc-ws-bash", "tc-read", "tc-empty"}
            assert after_fps["tc-empty"] == content_fingerprint(("",))

            # Non-bash read_file is now searchable via sidecar & search service with snippet
            hits = search_tool_calls_fts(sidecar, "search_filters", 10)
            assert [h[0] for h in hits] == ["tc-read"]

            # Offline verify_candidate passes with 100% fingerprint parity
            verify_candidate(conn, sidecar, ("content", "thinking", "bash"))

            # Reconcile finds 0 orphans and 0 ghosts
            reconcile = reconcile_sidecar(conn, sidecar, fts_fields=("content", "thinking", "bash"))
            assert reconcile.orphans_backfilled == {"message": 0, "tool_call": 0}
            assert reconcile.ghosts_deleted == {"message": 0, "tool_call": 0}
        finally:
            sidecar.close()

    # End-to-end search --tool read_file + general search returns clean [read_file] snippet
    # without widening SearchResult schema with raw tool_input.
    results = search(
        query="search_filters",
        source=None,
        tool="read_file",
        limit=5,
        mode=SearchMode.KEYWORD,
        config=config,
    )
    assert len(results) == 1
    assert results[0].tool_call_id == "tc-read"
    assert results[0].snippet == "[read_file] /repo/search_filters.py"
    assert "tool_input" not in asdict(results[0])


def test_interrupted_rescope_refuses_poll_reads_and_forces_union_recovery(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with connect(config) as conn:
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES ('s1', 'codex', '/tmp/s1.jsonl')"
        )
        conn.execute(
            "INSERT INTO session_state (session_id, message_count, tool_count, is_complete, "
            "file_mtime, file_size) VALUES ('s1', 1, 1, true, 1.0, 100)"
        )
        conn.execute("INSERT INTO messages (id, session_id, idx) VALUES ('m1', 's1', 0)")
        conn.execute(
            "INSERT INTO message_state (message_id, role, content, thinking, fts_content, "
            "fts_thinking) "
            "VALUES ('m1', 'assistant', 'body', 'secretthought', 'body', 'secretthought')"
        )
        conn.execute(
            "INSERT INTO tool_calls (id, session_id, message_id, idx, tool_name, tool_input) "
            "VALUES ('tc1', 's1', 'm1', 0, 'Edit', ?)",
            [json.dumps({"file_path": "/repo/edit_target.py"})],
        )

        sidecar = open_sidecar(sidecar_path(config.data_dir))
        try:
            bootstrap_sidecar(conn, sidecar, fts_fields=("content", "thinking", "bash"))
            set_fts_fields_signature(sidecar, ("content", "thinking", "bash"))
            # Simulate crash mid-rescope: marker present even though signature matches current
            set_rescope_in_progress(sidecar, {"tool_call"})
            sidecar.commit()
        finally:
            sidecar.close()

    # Owned poll snapshot MUST refuse while rescope_in_progress is set (REQ-CONC-009)
    req = poll_read_request("recall.search", {"query": "body", "mode": "keyword"})
    assert req is not None
    with pytest.raises(RuntimeError, match="rescope is incomplete"):
        read_poll_snapshot(config, req, ReadBudget.start(30))

    # Union rule: interrupted tool_call + removed thinking -> rescopes BOTH message and tool_call
    assert rescope_kinds_needed(
        "bash,content,thinking|v3",
        ("content", "bash"),
        in_progress=frozenset({"tool_call"}),
    ) == frozenset({"message", "tool_call"})

    with connect(config) as conn:
        sidecar = open_sidecar(sidecar_path(config.data_dir))
        try:
            with StandaloneLifetime().lease(conn) as lease:
                _rescope_sidecar_fields_if_needed(
                    lease,
                    conn,
                    sidecar,
                    fts_fields=("content", "bash"),
                    sidecar_existed=True,
                )
            assert get_rescope_in_progress(sidecar) == frozenset()
            assert get_fts_fields_signature(sidecar) == fts_fields_signature(("content", "bash"))
            # Thinking token was removed because union recovery ran the message pass too
            assert sidecar.execute(
                "SELECT COUNT(*) FROM message_fts WHERE message_fts MATCH 'secretthought'"
            ).fetchone() == (0,)
            verify_candidate(conn, sidecar, ("content", "bash"))
        finally:
            sidecar.close()

    # Owned poll snapshot succeeds once rescope completes cleanly
    poll_cfg = AppConfig(
        data_dir=config.data_dir,
        db_path=config.db_path,
        lock_path=config.lock_path,
        config_path=config.config_path,
        fts=FtsConfig(backend="sqlite_sidecar", fields=("content", "bash")),
        embedding=config.embedding,
        daemon=config.daemon,
        cli=config.cli,
    )
    snap = read_poll_snapshot(poll_cfg, req, ReadBudget.start(30))
    assert len(snap.data) == 1


def test_indexer_searchable_signature_and_pending_repair_for_non_bash_tools(tmp_path: Path) -> None:
    config = _config(tmp_path)
    tc1 = ToolCall(
        id="tc-edit-1",
        session_id="s-idx",
        message_id="m-idx",
        idx=0,
        tool_name="Edit",
        tool_input={"file_path": "/repo/first_path.py", "old_string": "a", "new_string": "b"},
        bash_command=None,
    )
    msg1 = Message(
        id="m-idx",
        session_id="s-idx",
        idx=0,
        role=Role.ASSISTANT,
        content=None,
        thinking=None,
        timestamp=None,
        has_thinking=False,
        tool_calls=[tc1],
    )
    session = Session(
        id="s-idx",
        source=Source.CLAUDE_CODE,
        source_path="/tmp/idx.jsonl",
        source_session_id="src-idx",
        started_at=None,
        ended_at=None,
        duration_seconds=None,
        model=None,
        cwd="/repo",
        git_repo="org/recall",
        git_branch="main",
        message_count=1,
        tool_count=1,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        is_complete=True,
        file_mtime=1.0,
        file_size=100,
        messages=[msg1],
    )

    with connect(config) as conn:
        _write_session(
            conn,
            session,
            turn=without_keyword_index(),
            tail_facts=TailFacts(),
        )
        # Unchanged re-index must report NO searchable change (symmetric persisted/session sigs)
        snapshot = _load_session_write_snapshot(conn, session, host=None)
        assert not _searchable_rows_changed(snapshot.rows, session)

        # Changing the non-bash Edit target path MUST report a searchable change
        tc2 = ToolCall(
            id="tc-edit-1",
            session_id="s-idx",
            message_id="m-idx",
            idx=0,
            tool_name="Edit",
            tool_input={"file_path": "/repo/second_path.py", "old_string": "a", "new_string": "b"},
            bash_command=None,
        )
        changed_session = session.model_copy(
            update={"messages": [msg1.model_copy(update={"tool_calls": [tc2]})]}
        )
        assert _searchable_rows_changed(snapshot.rows, changed_session)

        # Pending repair drains non-bash tool call into sidecar
        sidecar = open_sidecar(sidecar_path(config.data_dir))
        try:
            conn.execute(
                "INSERT INTO fts_sidecar_pending(kind, id, op) "
                "VALUES ('tool_call', 'tc-edit-1', 'upsert')"
            )
            repaired = repair_pending_batch(
                conn, sidecar, fts_fields=("content", "thinking", "bash")
            )
            assert repaired.applied == 1
            assert repaired.remaining == 0
            hits = search_tool_calls_fts(sidecar, "first_path", 10)
            assert [h[0] for h in hits] == ["tc-edit-1"]
        finally:
            sidecar.close()


def test_byte_budget_partition_and_sql_oversize_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = list(_partition_by_byte_budget([("a", 6), ("b", 6), ("c", 4), ("d", 15)], 10))
    assert chunks == [["a"], ["b", "c"], ["d"]]

    monkeypatch.setattr("recall.services.fts_sidecar_reconcile.TOOL_INPUT_MAX_PARSE_BYTES", 128)
    monkeypatch.setattr("recall.core.tool_format.TOOL_INPUT_MAX_PARSE_BYTES", 128)
    db_path = tmp_path / "bounded.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES ('s1', 'codex', '/tmp/b.jsonl')"
        )
        conn.execute(
            "INSERT INTO tool_calls (id, session_id, idx, tool_name, tool_input, bash_command) "
            "VALUES ('tc-small', 's1', 0, 'Read', ?, NULL), ('tc-big', 's1', 1, 'Write', ?, NULL)",
            [
                json.dumps({"file_path": "/small.py"}),
                json.dumps({"file_path": "/big.py", "content": "x" * 300}),
            ],
        )
        sidecar = open_sidecar(tmp_path / "recall.fts.sqlite")
        try:
            bootstrap_sidecar(conn, sidecar, fts_fields=("content", "thinking", "bash"))
            assert sidecar.execute(
                "SELECT value FROM fts_sidecar_meta WHERE key = 'schema_version'"
            ).fetchone() == (str(SIDECAR_SCHEMA_VERSION),)
            # Both rows preserve 1:1 membership; oversized row gets empty string fingerprint
            fps = dict(
                sidecar.execute(
                    "SELECT tool_call_id, content_fingerprint FROM tool_calls_fts_rowid"
                ).fetchall()
            )
            assert fps["tc-small"] == content_fingerprint(("/small.py",))
            assert fps["tc-big"] == content_fingerprint(("",))
            # Bounded snippet hydration skips the oversized JSON in SQL
            hydrated: dict[str, str] = {}
            for chunk in iter_tool_call_inputs_bounded(conn, ["tc-small", "tc-big"]):
                hydrated.update(chunk)
            assert "tc-small" in hydrated
            assert "tc-big" not in hydrated
            verify_candidate(conn, sidecar, ("content", "thinking", "bash"))
        finally:
            sidecar.close()


def test_live_and_persisted_surrogate_pair_fingerprint_parity(tmp_path: Path) -> None:
    """Live JS-decoded UTF-16 surrogate pairs match DuckDB persisted JSON fingerprints."""
    live_input = {"path": "/repo/a\ud83d\ude00b.py"}
    persisted_json = json.dumps(live_input, sort_keys=True)
    live_text = derive_tool_call_fts_text("read_file", live_input, None)
    persisted_text = derive_tool_call_fts_text("read_file", persisted_json, None)
    assert live_text == "/repo/a😀b.py"
    assert live_text == persisted_text
    assert content_fingerprint((live_text,)) == content_fingerprint((persisted_text,))

    db_path = tmp_path / "surrogate.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES ('s1', 'codex', '/tmp/s.jsonl')"
        )
        conn.execute(
            "INSERT INTO tool_calls (id, session_id, idx, tool_name, tool_input, bash_command) "
            "VALUES ('tc-surrogate', 's1', 0, 'read_file', ?, NULL)",
            [persisted_json],
        )
        sidecar = open_sidecar(tmp_path / "recall.fts.sqlite")
        try:
            # Publish using the live-derived text, then verify against DuckDB's persisted JSON
            upsert_tool_call_fts(
                sidecar, "tc-surrogate", live_text, fields=("content", "thinking", "bash")
            )
            verify_candidate(conn, sidecar, ("content", "thinking", "bash"))
        finally:
            sidecar.close()


def test_snippet_shares_field_selection_past_64_keys_and_long_tool_name() -> None:
    """Snippet rendering never drops fields past 64 keys and caps tool_name only in prefix."""
    many_keys: dict[str, object] = {f"a{i:02d}": i for i in range(64)}
    many_keys["path"] = "needle.txt"
    assert derive_tool_call_fts_text("read_file", many_keys, None) == "needle.txt"
    snippet = build_search_snippet(
        query="needle",
        kind="tool_call",
        content=None,
        thinking=None,
        role=None,
        tool_name="read_file",
        bash_command=None,
        tool_input=many_keys,
    )
    assert snippet == "[read_file] needle.txt"

    # Long tool_name prefix is bounded without altering field selection
    long_name = "mcp__very_long_server_prefix__" + "x" * 120
    long_snippet = build_search_snippet(
        query="needle",
        kind="tool_call",
        content=None,
        thinking=None,
        role=None,
        tool_name=long_name,
        bash_command=None,
        tool_input=many_keys,
    )
    assert len(long_snippet) <= SNIPPET_MAX_CHARS
    assert "needle.txt" in long_snippet


def test_streamed_hydration_pending_repair_and_orphan_backfill_respect_chunk_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hydration, pending repair, and orphan backfill consume one byte-bounded chunk at a time."""
    monkeypatch.setattr("recall.services.fts_sidecar_reconcile._TOOL_PAYLOAD_PAGE_MAX_BYTES", 200)
    db_path = tmp_path / "streamed.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES ('s1', 'codex', '/tmp/s.jsonl')"
        )
        for i in range(5):
            payload = json.dumps({"path": f"/repo/chunk_target_{i}.py", "pad": "z" * 100})
            conn.execute(
                "INSERT INTO tool_calls (id, session_id, idx, tool_name, tool_input, bash_command) "
                "VALUES (?, 's1', ?, 'read_file', ?, NULL)",
                [f"tc-{i}", i, payload],
            )
            conn.execute(
                "INSERT INTO fts_sidecar_pending(kind, id, op) VALUES ('tool_call', ?, 'upsert')",
                [f"tc-{i}"],
            )

        # Streaming hydration yields multiple bounded chunks rather than one giant dict
        input_chunks = list(
            iter_tool_call_inputs_bounded(conn, [f"tc-{i}" for i in range(5)], max_bytes=200)
        )
        assert len(input_chunks) >= 3
        assert {k for chunk in input_chunks for k in chunk} == {f"tc-{i}" for i in range(5)}

        # Streaming FTS chunk iterator for arbitrary IDs splits into bounded chunks
        fts_chunks = list(
            iter_tool_call_fts_chunks_for_ids(conn, [f"tc-{i}" for i in range(5)], max_bytes=200)
        )
        assert len(fts_chunks) >= 3

        sidecar = open_sidecar(tmp_path / "recall.fts.sqlite")
        try:
            # repair_pending_batch splits by byte bound and advances cursor until drained
            first = repair_pending_batch(
                conn, sidecar, fts_fields=("content", "thinking", "bash"), limit=256
            )
            assert 1 <= first.attempted < 5
            assert first.applied == first.attempted
            assert first.cursor is not None
            assert first.remaining == 5 - first.attempted

            cursor = first.cursor
            total_applied = first.applied
            while cursor is not None:
                step = repair_pending_batch(
                    conn,
                    sidecar,
                    cursor=cursor,
                    fts_fields=("content", "thinking", "bash"),
                    limit=256,
                )
                total_applied += step.applied
                cursor = step.cursor
            assert total_applied == 5
            verify_candidate(conn, sidecar, ("content", "thinking", "bash"))

            # Reconcile orphan backfill also streams bounded chunks
            sidecar.execute("DELETE FROM tool_calls_fts_rowid")
            sidecar.execute("DELETE FROM tool_calls_fts")
            sidecar.commit()
            stats = reconcile_sidecar(conn, sidecar, fts_fields=("content", "thinking", "bash"))
            assert stats.orphans_backfilled["tool_call"] == 5
            verify_candidate(conn, sidecar, ("content", "thinking", "bash"))
        finally:
            sidecar.close()


def test_oversized_bash_row_isolated_as_singleton_preserving_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversized Bash command is isolated in its own 1-row batch with unchanged fingerprint."""
    monkeypatch.setattr("recall.services.fts_sidecar_reconcile._TOOL_PAYLOAD_PAGE_MAX_BYTES", 64)
    big_cmd = "echo " + ("x" * 200)
    db_path = tmp_path / "oversized_bash.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES ('s1', 'codex', '/tmp/s.jsonl')"
        )
        conn.execute(
            "INSERT INTO tool_calls (id, session_id, idx, tool_name, tool_input, bash_command) "
            "VALUES ('tc-1', 's1', 0, 'Bash', NULL, 'ls'), "
            "('tc-2', 's1', 1, 'Bash', NULL, ?), "
            "('tc-3', 's1', 2, 'Bash', NULL, 'pwd')",
            [big_cmd],
        )
        batches = list(iter_tool_call_fts_batches(conn, batch_size=10))
        assert [[row[0] for row in batch] for batch in batches] == [["tc-1"], ["tc-2"], ["tc-3"]]
        assert batches[1] == [("tc-2", big_cmd)]


def test_manual_keywords_accepts_rescope_in_progress_metadata(tmp_path: Path) -> None:
    """Manual keyword repair recognizes `rescope_in_progress` as valid sidecar metadata."""
    config = _config(tmp_path)
    sidecar = open_sidecar(sidecar_path(config.data_dir))
    try:
        set_fts_fields_signature(sidecar, config.fts.fields)
        set_rescope_in_progress(sidecar, {"tool_call"})
    finally:
        sidecar.close()
    lifetime = StandaloneLifetime()
    require_replaceable_sidecar(config, lifetime=lifetime)


def test_apply_patch_ansi_prefixed_line_selection_matches_indexing() -> None:
    """Sanitization runs after target extraction so ANSI prefixes cannot invent patch targets."""
    patch_text = (
        "*** Begin Patch\n"
        "*** Update File: /tmp/original.py\n"
        "\x1b[31m*** Update File: /tmp/other.py\n"
        "*** End Patch\n"
    )
    indexed = derive_tool_call_fts_text("apply_patch", {"input": patch_text}, None)
    assert indexed == "/tmp/original.py"
    snippet = build_search_snippet(
        query="original",
        kind="tool_call",
        content=None,
        thinking=None,
        role=None,
        tool_name="apply_patch",
        bash_command=None,
        tool_input={"input": patch_text},
    )
    assert snippet == "[apply_patch] /tmp/original.py"


def test_streaming_hydration_releases_prior_chunk_before_next_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Search snippet hydration empties each chunk before the iterator fetches the next."""
    monkeypatch.setattr("recall.services.fts_sidecar_reconcile._TOOL_PAYLOAD_PAGE_MAX_BYTES", 200)
    config = _config(tmp_path)
    with connect(config) as conn:
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES ('s1', 'codex', '/tmp/s.jsonl')"
        )
        for i in range(4):
            payload = json.dumps({"path": f"/repo/release_target_{i}.py", "pad": "z" * 120})
            conn.execute(
                "INSERT INTO tool_calls (id, session_id, idx, tool_name, tool_input, bash_command) "
                "VALUES (?, 's1', ?, 'read_file', ?, NULL)",
                [f"tc-rel-{i}", i, payload],
            )
        sidecar = open_sidecar(sidecar_path(config.data_dir))
        try:
            bootstrap_sidecar(conn, sidecar, fts_fields=("content", "thinking", "bash"))
        finally:
            sidecar.close()

        observed_chunks: list[dict[str, str]] = []
        prior_sizes_on_next: list[int] = []
        orig_iter = iter_tool_call_inputs_bounded

        def spying_iter(duck_conn, ids, *, max_bytes=None):
            for chunk in orig_iter(duck_conn, ids, max_bytes=max_bytes):
                if observed_chunks:
                    prior_sizes_on_next.append(len(observed_chunks[-1]))
                observed_chunks.append(chunk)
                yield chunk

        import sys

        monkeypatch.setattr(
            sys.modules["recall.services.search"], "iter_tool_call_inputs_bounded", spying_iter
        )
        hits = search(
            query="release_target",
            source=None,
            tool="read_file",
            limit=10,
            mode=SearchMode.KEYWORD,
            config=config,
            conn=conn,
        )
        assert len(hits) == 4
        assert len(observed_chunks) >= 2
        assert prior_sizes_on_next and all(size == 0 for size in prior_sizes_on_next)
        assert all(hit.snippet is not None and "release_target_" in hit.snippet for hit in hits)


def test_oversized_tool_name_bounded_in_sql_without_losing_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oversized `tool_name` values are bounded in SQL while preserving special-tool dispatch."""
    monkeypatch.setattr("recall.services.fts_sidecar_reconcile._TOOL_PAYLOAD_PAGE_MAX_BYTES", 256)
    huge_name = "mcp__" + ("x" * 1000)
    db_path = tmp_path / "huge_name.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES ('s1', 'codex', '/tmp/s.jsonl')"
        )
        conn.execute(
            "INSERT INTO tool_calls (id, session_id, idx, tool_name, tool_input, bash_command) "
            "VALUES ('tc-huge-1', 's1', 0, ?, NULL, NULL), "
            "('tc-huge-2', 's1', 1, ?, ?, NULL), "
            "('tc-patch', 's1', 2, '  apply_patch  ', ?, NULL)",
            [
                huge_name,
                huge_name,
                json.dumps({"path": "/repo/from_huge.py"}),
                json.dumps({"input": "*** Update File: /repo/patched.py\n"}),
            ],
        )
        batches = list(iter_tool_call_fts_batches(conn, batch_size=10))
        flat = dict(row for batch in batches for row in batch)
        assert flat == {
            "tc-huge-1": "",
            "tc-huge-2": "/repo/from_huge.py",
            "tc-patch": "/repo/patched.py",
        }
        # SQL projection drops huge tool_name (>64 bytes) so both NULL-input huge-name rows
        # charge 0 bytes and transfer empty string instead of the huge name.
        rows = conn.execute(
            "SELECT CASE WHEN COALESCE(strlen(trim(tool_name, ' \t\n\r\f\v')), 0) > 64 "
            "THEN '' ELSE trim(tool_name, ' \t\n\r\f\v') END "
            "FROM tool_calls WHERE id = 'tc-huge-1'"
        ).fetchone()
        assert rows == ("",)


def test_tab_and_unicode_whitespace_tool_name_parity_and_canonical_rows_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tab-padded and NBSP-padded special names match live vs batch, and _canonical_rows
    releases each batch's payload text before fetching the next batch."""
    from recall.services.fts_sidecar_rebuild import _canonical_rows

    monkeypatch.setattr("recall.services.fts_sidecar_reconcile._TOOL_PAYLOAD_PAGE_MAX_BYTES", 64)
    padding = "\n".join("+padding" for _ in range(100))
    patch_payload = {
        "input": (
            "*** Begin Patch\n"
            "*** Add File: /tmp/first.py\n"
            f"{padding}\n"
            "*** Update File: /tmp/unique_target.py\n"
            "*** End Patch\n"
        )
    }
    tab_name = "\t" * 70 + "apply_patch"
    nbsp_name = "\u00a0" * 40 + "apply_patch"

    db_path = tmp_path / "parity.duckdb"
    with duckdb.connect(str(db_path)) as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) VALUES ('s1', 'codex', '/tmp/s.jsonl')"
        )
        conn.execute(
            "INSERT INTO tool_calls (id, session_id, idx, tool_name, tool_input, bash_command) "
            "VALUES ('tc-tab', 's1', 0, ?, ?, NULL), ('tc-nbsp', 's1', 1, ?, ?, NULL)",
            [tab_name, json.dumps(patch_payload), nbsp_name, json.dumps(patch_payload)],
        )
        batch_map = dict(
            row for batch in iter_tool_call_fts_batches(conn, batch_size=10) for row in batch
        )
        expected = "/tmp/first.py, /tmp/unique_target.py"
        assert batch_map["tc-tab"] == derive_tool_call_fts_text(tab_name, patch_payload, None)
        assert batch_map["tc-tab"] == expected
        assert batch_map["tc-nbsp"] == derive_tool_call_fts_text(nbsp_name, patch_payload, None)
        assert batch_map["tc-nbsp"] == expected

        # Verify _canonical_rows holds no prior oversized Bash payload when fetching the next batch
        conn.execute("DELETE FROM tool_calls")
        cmd_a = "a" * 200
        cmd_b = "b" * 200
        conn.execute(
            "INSERT INTO tool_calls (id, session_id, idx, tool_name, tool_input, bash_command) "
            "VALUES ('a', 's1', 0, 'Bash', NULL, ?), ('b', 's1', 1, 'Bash', NULL, ?)",
            [cmd_a, cmd_b],
        )

        from typing import Any, cast

        saw_second_fetch = False

        class _Probe:
            def execute(self, sql: str, args: object = None):
                nonlocal saw_second_fetch
                if "WHERE id IN" in sql and isinstance(args, list) and args[-1] == "b":
                    saw_second_fetch = True
                    frame = getattr(canonical_iter, "gi_frame", None)
                    assert frame is not None
                    assert "fts_text" not in frame.f_locals
                    assert frame.f_locals.get("batch") == []
                return conn.execute(sql, args) if args is not None else conn.execute(sql)

        canonical_iter = _canonical_rows(cast(Any, _Probe()), "tool_call", ("bash",))
        first = next(canonical_iter)
        second = next(canonical_iter)
        assert saw_second_fetch
        assert first == ("a", content_fingerprint((cmd_a,)))
        assert second == ("b", content_fingerprint((cmd_b,)))
