"""Stored outcomes and lifecycle recovery of semantic work, with real DuckDB."""

from __future__ import annotations

from array import array
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.models import Message, Session, TailFacts, ToolCall
from recall.core.types import Role, Source
from recall.db.queries import insert_message_embeddings
from recall.db.schema import ensure_schema
from recall.db.semantic_work import SemanticWorkUnavailable, audit, publication
from recall.services.embed_phase import count_pending_split
from recall.services.indexer import _write_session
from recall.services.semantic_work import SemanticLifecycle


def configuration(path: Path) -> AppConfig:
    path.mkdir()
    return AppConfig(
        data_dir=path,
        db_path=path / "recall.duckdb",
        lock_path=path / "recall.lock",
        config_path=path / "config.toml",
        fts=FtsConfig(backend="sqlite_sidecar"),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )


def seed(conn: duckdb.DuckDBPyConnection) -> list[float]:
    ensure_schema(conn)
    vector = [0.0, -0.0, 1e-40] + [0.25] * 381
    session = Session(
        id="s",
        source=Source.CODEX,
        source_path="/fixture/session.jsonl",
        file_mtime=1,
        file_size=1,
        messages=[
            Message(
                id="partial",
                session_id="s",
                idx=0,
                role=Role.ASSISTANT,
                content="copper",
                thinking="silver",
                context_text="stored prefix ",
                context_mode="template",
            ),
            Message(id="empty", session_id="s", idx=1, role=Role.USER),
        ],
        orphan_tool_calls=[
            ToolCall(
                id="bash",
                session_id="s",
                message_id=None,
                idx=0,
                tool_name="Bash",
                bash_command="printf turquoise",
            )
        ],
    )
    _write_session(conn, session, tail_facts=TailFacts())
    insert_message_embeddings(conn, [("partial", vector, None)])
    conn.execute("CHECKPOINT")
    stored = conn.execute(
        "SELECT content_embedding FROM message_embeddings WHERE message_id='partial'"
    ).fetchone()
    assert stored is not None
    return list(stored[0])


def schema32_fixture(conn: duckdb.DuckDBPyConnection) -> list[float]:
    """Offline preimage-equivalent fixture reversal documented by migration 0033."""
    vector = seed(conn)
    conn.execute("BEGIN")
    conn.execute("DROP VIEW semantic_pending")
    for table in ("semantic_work_counts", "semantic_work", "semantic_work_state"):
        conn.execute(f"DROP TABLE {table}")
    conn.execute("DELETE FROM schema_version WHERE version=33")
    conn.execute("INSERT INTO schema_version(version) VALUES (32)")
    conn.execute("COMMIT")
    return vector


def test_pending_upgrade_refuses_without_payload_mutation(tmp_path: Path) -> None:
    config = configuration(tmp_path / "data")
    with duckdb.connect(str(config.db_path)) as conn:
        schema32_fixture(conn)
        with pytest.raises(SemanticWorkUnavailable, match="verified backup"):
            ensure_schema(conn)
        assert conn.execute("SELECT max(version) FROM schema_version").fetchone() == (32,)
        assert conn.execute(
            "SELECT content,thinking FROM message_state WHERE message_id='partial'"
        ).fetchone() == ("copper", "silver")
        assert not list(config.data_dir.glob("snapshots/*"))


def test_initialize_reopen_and_fill_preserve_exact_sibling_and_context(tmp_path: Path) -> None:
    config = configuration(tmp_path / "data")
    with duckdb.connect(str(config.db_path)) as conn:
        vector = schema32_fixture(conn)
        lifecycle = SemanticLifecycle(config)
        lifecycle.run(conn)
        assert lifecycle.admitted, lifecycle.error
        assert conn.execute(
            "SELECT "
            "kind,id,missing_content,missing_thinking,missing_bash,needs_marker "
            "FROM semantic_work ORDER BY kind,id"
        ).fetchall() == [
            ("message", "empty", False, False, False, True),
            ("message", "partial", False, True, False, False),
            ("tool", "bash", False, False, True, True),
        ]
        assert conn.execute(
            "SELECT "
            "pending_messages,pending_tools,missing_content,missing_thinking,missing_bash "
            "FROM semantic_work_counts"
        ).fetchone() == (2, 1, 0, 1, 1)
        assert count_pending_split(conn, 0).eligible == 3
    with duckdb.connect(str(config.db_path)) as conn:
        lifecycle = SemanticLifecycle(config)
        lifecycle.run(conn)
        assert lifecycle.admitted, lifecycle.error
        insert_message_embeddings(conn, [("empty", None, None)])
        from recall.db.enrichment_inputs import GeneratedEmbedSession
        from recall.services.embed_phase import (
            EmbedPhaseState,
            commit_prepared_embed_cycle,
            generate_prepared_embed_cycle,
            prepare_embed_cycle,
        )

        class Backend:
            dimensions = 384
            model_id = "literal"
            query_prefix = ""

            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.5] * 384 for _ in texts]

        state = EmbedPhaseState(_config=config)
        state.backend = Backend()
        prepared = prepare_embed_cycle(config, state, conn=conn)
        generate_prepared_embed_cycle(config, state, prepared)
        assert all(
            isinstance(result, GeneratedEmbedSession) for result in prepared.results.values()
        )
        assert commit_prepared_embed_cycle(prepared, conn=conn) == 2
        audit(conn)
        assert conn.execute("SELECT count(*) FROM semantic_work").fetchone() == (0,)
    with duckdb.connect(str(config.db_path), read_only=True) as conn:
        stored = conn.execute(
            "SELECT content_embedding,thinking_embedding FROM "
            "message_embeddings WHERE message_id='partial'"
        ).fetchone()
        assert stored is not None
        assert array("f", stored[0]).tobytes() == array("f", vector).tobytes()
        assert tuple(stored[1]) == (0.5,) * 384
        assert conn.execute(
            "SELECT context_text,context_mode,content,thinking FROM "
            "message_state WHERE message_id='partial'"
        ).fetchone() == ("stored prefix ", "template", "copper", "silver")
        audit(conn)
        coverage = count_pending_split(conn, 0).semantic_coverage
        assert coverage is not None and coverage["complete"] is True


def test_late_counter_failure_rolls_back_payload_and_latches_invalid(tmp_path: Path) -> None:
    config = configuration(tmp_path / "data")
    with duckdb.connect(str(config.db_path)) as conn:
        seed(conn)
        # Deliberate out-of-band drift makes a late delta impossible.
        conn.execute(
            "UPDATE semantic_work_counts SET "
            "pending_messages=0,pending_tools=3,missing_thinking=0,missing_bash=1"
        )
        before = conn.execute("SELECT * FROM semantic_work ORDER BY kind,id").fetchall()
        with pytest.raises(AssertionError, match="counter preimage"):
            publication(conn).message_vectors((("empty", None, None),))
        assert conn.execute(
            "SELECT count(*) FROM message_embeddings WHERE message_id='empty'"
        ).fetchone() == (0,)
        assert conn.execute("SELECT * FROM semantic_work ORDER BY kind,id").fetchall() == before
        with pytest.raises(SemanticWorkUnavailable):
            count_pending_split(conn, 0)
    with duckdb.connect(str(config.db_path), read_only=True) as conn:
        assert conn.execute(
            "SELECT count(*) FROM message_embeddings WHERE message_id='empty'"
        ).fetchone() == (0,)
        assert conn.execute("SELECT phase FROM semantic_work_state").fetchone() == ("failed",)


def test_drift_repair_stale_plan_and_durable_repeat(tmp_path: Path) -> None:
    config = configuration(tmp_path / "data")
    with duckdb.connect(str(config.db_path)) as conn:
        schema32_fixture(conn)
        lifecycle = SemanticLifecycle(config)
        lifecycle.run(conn)
        assert lifecycle.admitted, lifecycle.error
        conn.execute("UPDATE semantic_work SET id='wrong' WHERE kind='tool'")
        lifecycle = SemanticLifecycle(config)
        lifecycle.run(conn)
        assert not lifecycle.admitted and "drift" in (lifecycle.error or "")
        stale = lifecycle.preview(conn)
    with duckdb.connect(str(config.db_path)) as conn:
        lifecycle = SemanticLifecycle(config)
        lifecycle.run(conn)
        assert not lifecycle.admitted
        before = conn.execute("SELECT * FROM semantic_work_state").fetchall()
        with pytest.raises(SemanticWorkUnavailable, match="conflict"):
            lifecycle.apply(conn, stale.plan_id)
        assert conn.execute("SELECT * FROM semantic_work_state").fetchall() == before
        plan = lifecycle.preview(conn)
        result = lifecycle.apply(conn, plan.plan_id)
        assert lifecycle.admitted and result.outcome == "accepted"
        assert conn.execute("SELECT id FROM semantic_work WHERE kind='tool'").fetchall() == [
            ("bash",)
        ]
    with duckdb.connect(str(config.db_path)) as conn:
        lifecycle = SemanticLifecycle(config)
        lifecycle.run(conn)
        repeated = lifecycle.apply(conn, plan.plan_id)
        assert repeated.outcome == "unchanged" and repeated.build_id == result.build_id
        assert lifecycle.admitted


def test_missing_original_preimage_repairs_current_payload_without_fabricated_rollback(
    tmp_path: Path,
) -> None:
    config = configuration(tmp_path / "data")
    with duckdb.connect(str(config.db_path)) as conn:
        schema32_fixture(conn)
        lifecycle = SemanticLifecycle(config)
        lifecycle.run(conn)
        assert lifecycle.intent is not None and lifecycle.intent.backup_path is not None
        preimage = Path(lifecycle.intent.backup_path) / "recall.duckdb"
        assert preimage.is_relative_to(tmp_path)
        preimage.rename(preimage.with_suffix(".retained-missing-case"))
        conn.execute("UPDATE semantic_work SET id='wrong' WHERE kind='tool'")
        lifecycle = SemanticLifecycle(config)
        lifecycle.run(conn)
        plan = lifecycle.preview(conn)
        assert plan.rollback_scope == "schema33_current"
        lifecycle.apply(conn, plan.plan_id)
        assert lifecycle.admitted
        assert not preimage.exists()
        assert conn.execute(
            "SELECT content,thinking FROM message_state WHERE message_id='partial'"
        ).fetchone() == ("copper", "silver")


def test_dimension_reset_has_a_preimage_and_requeues_raw_fields(tmp_path: Path) -> None:
    config = configuration(tmp_path / "data")
    with duckdb.connect(str(config.db_path)) as conn:
        seed(conn)
        changed = replace(config, embedding=replace(config.embedding, dimensions=256))
        lifecycle = SemanticLifecycle(changed)
        lifecycle.run(conn)
        assert lifecycle.admitted, lifecycle.error
        assert lifecycle.intent is not None and lifecycle.intent.operation == "dimensions"
        assert conn.execute("SELECT embedding_dimensions FROM runtime_state").fetchone() == (256,)
        assert conn.execute("SELECT count(*) FROM message_embeddings").fetchone() == (0,)
        assert conn.execute(
            "SELECT missing_content,missing_thinking,missing_bash FROM semantic_work_counts"
        ).fetchone() == (1, 1, 1)
        assert lifecycle.intent.backup_path is not None
        with duckdb.connect(
            str(Path(lifecycle.intent.backup_path) / "recall.duckdb"), read_only=True
        ) as before:
            assert before.execute("SELECT embedding_dimensions FROM runtime_state").fetchone() == (
                384,
            )
            assert before.execute("SELECT count(*) FROM message_embeddings").fetchone() == (1,)
