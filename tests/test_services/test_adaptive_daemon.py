from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig, CliConfig, DaemonConfig, EmbeddingConfig, FtsConfig
from recall.core.types import DaemonMode
from recall.services.embed_phase import find_pending_embeds
from recall.services.indexer import index_sessions


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


class TestIndexWithoutEmbedding:
    def test_index_produces_no_embeddings_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-ADAPT-001: daemon index phase must not invoke embedding."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("RECALL_FTS_BACKEND", "duckdb")
        codex_dir = tmp_path / ".codex" / "sessions" / "s1"
        codex_dir.mkdir(parents=True)
        fixture = (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "codex"
            / "session1"
            / "rollout.jsonl"
        )
        shutil.copy(fixture, codex_dir / "rollout.jsonl")

        # Index without embedding (simulates daemon behavior)
        result = index_sessions(source=None, full=False, recreate=True, verbose=False)
        assert result.indexed >= 1

        config = _app_config(tmp_path)
        conn = duckdb.connect(str(config.db_path), read_only=True)
        try:
            # Messages exist
            row = conn.execute("SELECT COUNT(*) FROM messages").fetchone()
            assert row is not None
            msg_count = row[0]
            assert msg_count > 0

            # But no embeddings
            row = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()
            assert row is not None
            embed_count = row[0]
            assert embed_count == 0, "daemon index should not produce embeddings"

            # FTS still works (keyword search available immediately)
            row = conn.execute(
                "SELECT COUNT(*) FROM information_schema.schemata "
                "WHERE schema_name LIKE 'fts_main_%'"
            ).fetchone()
            assert row is not None
            fts_exists = row[0]
            assert fts_exists > 0, "FTS indexes should be built"
        finally:
            conn.close()


class TestEmbedPhaseDiscovery:
    def test_embed_phase_finds_unembedded_idle_sessions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-ADAPT-003, REQ-ADAPT-004: embed phase discovers idle un-embedded content."""
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

        # Set mtime to the past (simulate idle session)
        old_time = time.time() - 3600
        os.utime(dest, (old_time, old_time))

        index_sessions(source=None, full=False, recreate=True, verbose=False)

        config = _app_config(tmp_path)
        conn = duckdb.connect(str(config.db_path), read_only=True)
        try:
            # With idle_threshold=600 and file 1 hour old, should find work
            pending = find_pending_embeds(conn, idle_threshold=600)
            assert pending.message_count > 0

            # With idle_threshold=7200, file is too recent
            pending = find_pending_embeds(conn, idle_threshold=7200)
            assert pending.message_count == 0
        finally:
            conn.close()
