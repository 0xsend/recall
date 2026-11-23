"""Tests for zero-DuckDB-migration FTS sidecar non-blank derivation & batched rescope."""

from __future__ import annotations

from pathlib import Path

import duckdb
from recall.core.models import Message, Session, TailFacts
from recall.core.types import Role, Source
from recall.db import insert_message_embeddings
from recall.db.fts_sidecar import (
    FTS_DERIVATION_VERSION,
    fts_fields_signature,
    get_fts_fields_signature,
    open_sidecar,
    search_messages_fts,
    upsert_message_fts,
)
from recall.db.lease import StandaloneLifetime
from recall.db.schema import SCHEMA_VERSION, ensure_schema, get_schema_version
from recall.services.daemon import _rescope_sidecar_fields_if_needed
from recall.services.fts_sidecar_bootstrap import bootstrap_sidecar
from recall.services.fts_sidecar_rebuild import _canonical_rows
from recall.services.fts_sidecar_reconcile import repair_pending_batch
from recall.services.indexer import _write_session

from tests.keyword_turns import without_keyword_index


def _seed_v41_db_with_legacy_polluted_fts(db_path: Path) -> None:
    with duckdb.connect(str(db_path)) as conn:
        ensure_schema(conn, embed_dim=8)
        conn.execute(
            "INSERT INTO sessions (id, source, source_path) "
            "VALUES ('s1', 'antigravity_cli', '/tmp/s1.jsonl')"
        )
        conn.execute(
            "INSERT INTO session_state "
            "(session_id, message_count, tool_count, is_complete, file_mtime, file_size) "
            "VALUES ('s1', 4, 0, TRUE, 1.0, 10)"
        )
        rows = [
            # Legacy polluted empty content
            (
                "m-null",
                0,
                "assistant",
                None,
                None,
                "[0xbigboss/recall main] ",
                "[0xbigboss/recall main] ",
            ),
            # Legacy polluted Unicode whitespace body
            ("m-ws", 1, "assistant", " \t\n\u2003", "\t\r\n", "[repo] ", "[repo] "),
            # Mixed: empty content + real thinking
            ("m-mixed", 2, "assistant", "", "real thought", "[repo] ", "[repo] real thought"),
            # Clean row
            ("m-clean", 3, "assistant", "real answer", None, "[repo] real answer", "[repo] "),
        ]
        for msg_id, idx, role, content, thinking, fts_c, fts_t in rows:
            conn.execute(
                "INSERT INTO messages (id, session_id, idx) VALUES (?, 's1', ?)",
                [msg_id, idx],
            )
            conn.execute(
                "INSERT INTO message_state "
                "(message_id, role, content, thinking, has_thinking, context_text, "
                "fts_content, fts_thinking) "
                "VALUES (?, ?, ?, ?, ?, '[0xbigboss/recall main] ', ?, ?)",
                [msg_id, role, content, thinking, thinking is not None, fts_c, fts_t],
            )


def test_sidecar_rescope_cleans_polluted_v41_rows_without_mutating_duckdb(tmp_path: Path) -> None:
    db_path = tmp_path / "recall.duckdb"
    sidecar_file = tmp_path / "recall.fts.sqlite"
    _seed_v41_db_with_legacy_polluted_fts(db_path)

    # Simulate an existing v0.48.0 sidecar populated with polluted tokens & v1 signature
    sidecar = open_sidecar(sidecar_file)
    try:
        upsert_message_fts(
            sidecar, "m-null", "[0xbigboss/recall main] ", "[0xbigboss/recall main] "
        )
        upsert_message_fts(sidecar, "m-ws", "[repo] ", "[repo] ")
        upsert_message_fts(sidecar, "m-mixed", "[repo] ", "[repo] real thought")
        upsert_message_fts(sidecar, "m-clean", "[repo] real answer", "[repo] ")
        with sidecar:
            sidecar.execute(
                "INSERT OR REPLACE INTO fts_sidecar_meta(key, value) "
                "VALUES ('fts_fields_signature', 'bash,content,thinking')"
            )
        # Before rescope, empty m-null matches "recall"
        before_ids = [
            mid for mid, _ in search_messages_fts(sidecar, "recall", ["content", "thinking"], 10)
        ]
        assert "m-null" in before_ids

        with duckdb.connect(str(db_path)) as conn:
            before_ms = conn.execute(
                "SELECT message_id, fts_content, fts_thinking "
                "FROM message_state ORDER BY message_id"
            ).fetchall()
            with StandaloneLifetime().lease(conn) as lease:
                _rescope_sidecar_fields_if_needed(
                    lease,
                    conn,
                    sidecar,
                    fts_fields=("content", "thinking", "bash"),
                    sidecar_existed=True,
                )
            # Zero DuckDB mutation, schema stays at current SCHEMA_VERSION
            assert get_schema_version(conn) == SCHEMA_VERSION
            after_ms = conn.execute(
                "SELECT message_id, fts_content, fts_thinking "
                "FROM message_state ORDER BY message_id"
            ).fetchall()
            assert after_ms == before_ms

        sig = get_fts_fields_signature(sidecar)
        assert sig == fts_fields_signature(("content", "thinking", "bash"))
        assert sig is not None and sig.endswith(f"|v{FTS_DERIVATION_VERSION}")

        # After rescope, blank rows no longer match "recall", while non-blank rows still match
        after_ids = [
            mid for mid, _ in search_messages_fts(sidecar, "recall", ["content", "thinking"], 10)
        ]
        assert "m-null" not in after_ids
        assert "m-ws" not in after_ids
        assert "m-clean" in after_ids
        assert "m-mixed" in after_ids

        # Second call is an idempotent no-op
        with duckdb.connect(str(db_path)) as conn, StandaloneLifetime().lease(conn) as lease:
            _rescope_sidecar_fields_if_needed(
                lease,
                conn,
                sidecar,
                fts_fields=("content", "thinking", "bash"),
                sidecar_existed=True,
            )
    finally:
        sidecar.close()


def test_bootstrap_pending_repair_and_canonical_rows_derive_nonblank_fts(tmp_path: Path) -> None:
    db_path = tmp_path / "recall.duckdb"
    sidecar_file = tmp_path / "recall.fts.sqlite"
    _seed_v41_db_with_legacy_polluted_fts(db_path)

    with duckdb.connect(str(db_path)) as conn:
        sidecar = open_sidecar(sidecar_file)
        try:
            bootstrap_sidecar(conn, sidecar, fts_fields=("content", "thinking"), batch_size=10)
            hits = [
                mid
                for mid, _ in search_messages_fts(sidecar, "recall", ["content", "thinking"], 10)
            ]
            assert "m-null" not in hits
            assert "m-ws" not in hits
            assert set(hits) == {"m-clean", "m-mixed"}

            # Pending repair also derives cleanly
            conn.execute(
                "INSERT INTO fts_sidecar_pending (kind, id, op) "
                "VALUES ('message', 'm-null', 'upsert')"
            )
            repair_pending_batch(conn, sidecar, fts_fields=("content", "thinking"), limit=10)
            hits_after_repair = [
                mid
                for mid, _ in search_messages_fts(sidecar, "recall", ["content", "thinking"], 10)
            ]
            assert "m-null" not in hits_after_repair

            # Canonical fingerprints match the sidecar's derived fingerprints
            canonical = dict(_canonical_rows(conn, "message", ("content", "thinking")))
            actual = dict(
                sidecar.execute(
                    "SELECT message_id, content_fingerprint FROM message_fts_rowid"
                ).fetchall()
            )
            assert actual == canonical
        finally:
            sidecar.close()


def test_indexer_does_not_preserve_stale_blank_body_content_embedding() -> None:
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        original = Session(
            id="s-blank-vec",
            source=Source.CODEX,
            source_path="/owned/blank-vec.jsonl",
            file_mtime=1,
            file_size=1,
            messages=[
                Message(
                    id="m0",
                    session_id="s-blank-vec",
                    idx=0,
                    role=Role.ASSISTANT,
                    content="   ",
                    thinking="real reasoning",
                    has_thinking=True,
                ),
            ],
        )
        _write_session(conn, original, tail_facts=TailFacts(), turn=without_keyword_index())
        insert_message_embeddings(conn, [("m0", [1.0, 0.0], [0.0, 1.0])])

        rewritten = Session(
            id="s-blank-vec",
            source=Source.CODEX,
            source_path="/owned/blank-vec.jsonl",
            file_mtime=2,
            file_size=2,
            messages=[
                Message(
                    id="m-pre",
                    session_id="s-blank-vec",
                    idx=0,
                    role=Role.USER,
                    content="prepended",
                ),
                Message(
                    id="m1",
                    session_id="s-blank-vec",
                    idx=1,
                    role=Role.ASSISTANT,
                    content="   ",
                    thinking="real reasoning",
                    has_thinking=True,
                ),
            ],
        )
        _write_session(conn, rewritten, tail_facts=TailFacts(), turn=without_keyword_index())

        row = conn.execute(
            "SELECT content_embedding, thinking_embedding "
            "FROM message_embeddings WHERE message_id = 'm1'"
        ).fetchone()
        assert row == (None, (0.0, 1.0))


def test_move_carryover_drops_stale_blank_content_vector_and_keeps_thinking_vector() -> None:
    from recall.services.moves import _carry_stored_contexts, _carry_stored_vectors

    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        for sid, mid in (("pred", "p0"), ("succ", "s0")):
            conn.execute(
                "INSERT INTO sessions (id, source, source_path) VALUES (?, 'codex', ?)",
                [sid, f"/tmp/{sid}.jsonl"],
            )
            conn.execute("INSERT INTO messages (id, session_id, idx) VALUES (?, ?, 0)", [mid, sid])
            conn.execute(
                "INSERT INTO message_state "
                "(message_id, role, content, thinking, has_thinking, context_text, context_mode, "
                "fts_content, fts_thinking) "
                "VALUES (?, 'assistant', '\u2003', 'valid thought', TRUE, '[repo] ', 'template', "
                "'', '[repo] valid thought')",
                [mid],
            )
        # Predecessor has legacy stale blank-body content vector + valid thinking vector
        insert_message_embeddings(conn, [("p0", [1.0, 0.0], [0.0, 1.0])])

        _carry_stored_contexts(conn, predecessor_id="pred", successor_id="succ")
        _carry_stored_vectors(conn, predecessor_id="pred", successor_id="succ")

        carried = conn.execute(
            "SELECT content_embedding, thinking_embedding "
            "FROM message_embeddings WHERE message_id = 's0'"
        ).fetchone()
        assert carried == (None, (0.0, 1.0))


def test_same_id_rewrite_does_not_reuse_legacy_blank_body_vector() -> None:
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=2)
        original = Session(
            id="s-same-id",
            source=Source.CODEX,
            source_path="/owned/same-id.jsonl",
            file_mtime=1,
            file_size=1,
            messages=[
                Message(
                    id="m0",
                    session_id="s-same-id",
                    idx=0,
                    role=Role.ASSISTANT,
                    content=" ",
                    context_text="[repo] ",
                    context_mode="template",
                ),
            ],
        )
        _write_session(conn, original, tail_facts=TailFacts(), turn=without_keyword_index())
        insert_message_embeddings(conn, [("m0", [1.0, 0.0], None)])

        # Same ID, context disabled, body equals old concatenation "[repo]  "
        rewritten = Session(
            id="s-same-id",
            source=Source.CODEX,
            source_path="/owned/same-id.jsonl",
            file_mtime=2,
            file_size=2,
            messages=[
                Message(
                    id="m0",
                    session_id="s-same-id",
                    idx=0,
                    role=Role.ASSISTANT,
                    content="[repo]  ",
                    context_text="",
                    context_mode="off",
                ),
            ],
        )
        _write_session(conn, rewritten, tail_facts=TailFacts(), turn=without_keyword_index())

        assert conn.execute(
            "SELECT COUNT(*) FROM message_embeddings WHERE message_id = 'm0'"
        ).fetchone() == (0,)
