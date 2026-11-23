"""Regression tests for PR 6b: compressed-limit parking, readiness, verification, and v2 cursors."""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from recall.cli.daemon import _print_index_health_text
from recall.cli.status_notices import _index_divergence_notice
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
    SourceConfig,
)
from recall.core.types import Source
from recall.db.maintenance import IndexDivergenceReport
from recall.db.schema import ensure_schema
from recall.db.source_files import PARKED_RETRY_AT, CatalogCursor, SourceCatalog, SourceSignature
from recall.parsers.codex import CodexParser
from recall.parsers.compressed import CompressedCapture, CompressedCaptureLimitError, zstd
from recall.services.coordinator import (
    capture_path,
    commit_prepared_raw_sources,
    observe_path,
    prepare_raw_sources,
    reconciliation_status,
)
from recall.services.unsupported_diagnostics import reopen_unsupported_parks, unsupported_summary


def _config(tmp_path: Path, *, codex_root: Path | None = None) -> AppConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        s.value: SourceConfig(roots=(codex_root,) if codex_root and s is Source.CODEX else ())
        for s in Source
    }
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=data_dir / "config.toml",
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(),
        cli=CliConfig(),
        fts=FtsConfig(backend="sqlite_sidecar"),
        sources=sources,
    )


def _write_zst(plain_path: Path, payload: bytes) -> Path:
    zst_path = plain_path.with_suffix(".jsonl.zst")
    zst_path.write_bytes(zstd.compress(payload))
    return zst_path


def test_compressed_limit_parks_as_unsupported_and_excludes_from_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb
    from recall.db.fts_sidecar import open_sidecar, sidecar_path
    from recall.parsers import compressed as compressed_mod

    root = tmp_path / "codex"
    root.mkdir()
    plain = root / "rollout-2026-08-18T09-39-09-01a0154f-d84f-71c2-8b6e-91eba2c8c2e7.jsonl"
    body = b'{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}\n'
    _write_zst(plain, body)

    config = _config(tmp_path, codex_root=root)
    open_sidecar(sidecar_path(config.data_dir)).close()
    conn = duckdb.connect(str(config.db_path))
    try:
        ensure_schema(conn, embed_dim=8)
        parser = CodexParser(roots=(root,))
        catalog = SourceCatalog(conn, clock=lambda: 1000.0)
        catalog.record_scan(
            "codex",
            str(root),
            started_at=999.0,
            finished_at=1000.0,
            discovered_count=1,
            failures=[],
        )

        # First commit the compressed rollout normally so prior history exists.
        item = observe_path(parser, capture_path(parser, plain), conn=conn)
        commit_prepared_raw_sources(
            prepare_raw_sources((item,), {"codex": parser}), config, conn=conn
        )
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)

        # Now grow the compressed rollout past the decoded limit (monkeypatched small for speed).
        big_body = body * 50
        _write_zst(plain, big_body)
        item2 = observe_path(parser, capture_path(parser, plain), conn=conn)
        monkeypatch.setattr(compressed_mod, "MAX_COMPRESSED_CAPTURE_BYTES", len(body) * 2)

        with (
            pytest.raises(CompressedCaptureLimitError),
            CompressedCapture(plain.with_suffix(".jsonl.zst")),
        ):
            pass

        prepared = prepare_raw_sources((item2,), {"codex": parser})
        assert prepared[0].unsupported_limit is not None
        commit_prepared_raw_sources(prepared, config, conn=conn)

        stored = catalog.get("codex", str(plain))
        assert stored is not None
        assert stored.last_error == "unsupported"
        assert stored.retry_count == 0
        assert stored.next_retry_at == PARKED_RETRY_AT
        assert stored.committed_generation < stored.desired_generation
        # Prior history preserved untouched.
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)

        status = reconciliation_status(config, conn=conn)
        assert status["pending"] == 0
        assert status["raw_indexing_ready"] is True
        assert status["keyword_search_ready"] is True
        codex_cov = next(c for c in status["coverage"] if c["source"] == "codex")
        assert codex_cov["pending"] == 0
        assert codex_cov["retry"] == 0
        assert codex_cov["unsupported"] == 1
        assert codex_cov["oldest_pending_age"] is None
        summary = unsupported_summary(conn)
        assert summary["files"] == 1
        groups = summary["groups"]
        assert isinstance(groups, list)
        assert groups[0]["detail"] == "compressed rollout decoded content exceeds 96 MiB"

        # Materializing to plain .jsonl re-queues it: drops out of unsupported, enters pending.
        plain.write_bytes(big_body)
        observe_path(parser, capture_path(parser, plain), conn=conn)
        status_after = reconciliation_status(config, conn=conn)
        assert status_after["pending"] == 1
        assert status_after["raw_indexing_ready"] is False
        # Keyword sidecar has 0 pending intents, so keyword_search_ready stays True!
        assert status_after["keyword_search_ready"] is True
        assert unsupported_summary(conn)["files"] == 0

        # Parser revision reopen also immediately moves an unsupported_limit park back to pending.
        plain.unlink()
        catalog.defer_unsupported(
            "codex",
            str(plain),
            {
                "records": [
                    {
                        "kind": "unsupported_limit",
                        "detail": "compressed rollout decoded content exceeds 96 MiB",
                        "byte_offset": 0,
                    }
                ]
            },
            parser_revision="older-rev",
        )
        assert unsupported_summary(conn)["files"] == 1
        assert reopen_unsupported_parks(conn, 2000.0) == 1
        assert unsupported_summary(conn)["files"] == 0
        assert reconciliation_status(config, conn=conn)["pending"] == 1
    finally:
        conn.close()


def test_materialization_between_prepare_and_commit_prevents_stale_limit_park(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb
    from recall.parsers import compressed as compressed_mod

    root = tmp_path / "codex"
    root.mkdir()
    plain = root / "rollout-2026-08-23T19-15-39-01a0311f-6fa9-7f43-9110-5a47ccf09870.jsonl"
    body = b'{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}\n' * 10
    _write_zst(plain, body)

    config = _config(tmp_path, codex_root=root)
    conn = duckdb.connect(str(config.db_path))
    try:
        ensure_schema(conn, embed_dim=8)
        parser = CodexParser(roots=(root,))
        catalog = SourceCatalog(conn, clock=lambda: 1000.0)
        item = observe_path(parser, capture_path(parser, plain), conn=conn)

        monkeypatch.setattr(compressed_mod, "MAX_COMPRESSED_CAPTURE_BYTES", 32)
        prepared = prepare_raw_sources((item,), {"codex": parser})
        assert prepared[0].unsupported_limit is not None

        # Case A: un-catalogued physical materialization to plain .jsonl BEFORE commit.
        plain.write_bytes(body)
        commit_prepared_raw_sources(prepared, config, conn=conn)
        stored = catalog.get("codex", str(plain))
        assert stored is not None
        assert stored.last_error != "unsupported"
        assert stored.next_retry_at < 1e300
        assert stored.desired_generation > item.desired_generation

        # Case B: catalogued newer generation BEFORE commit.
        plain.unlink()
        item_zst = observe_path(parser, capture_path(parser, plain), conn=conn)
        prepared2 = prepare_raw_sources((item_zst,), {"codex": parser})
        plain.write_bytes(body)
        observe_path(parser, capture_path(parser, plain), conn=conn)
        commit_prepared_raw_sources(prepared2, config, conn=conn)
        stored2 = catalog.get("codex", str(plain))
        assert stored2 is not None
        assert stored2.last_error != "unsupported"
        assert stored2.next_retry_at < 1e300
    finally:
        conn.close()


def test_legacy_unsupported_park_with_unacknowledged_bytes_stays_pending_null_safe(
    tmp_path: Path,
) -> None:
    """REQ-RECON-029 null-safety guard: empty/NULL diagnostics must NOT be exempted from pending."""
    import duckdb

    root = tmp_path / "codex"
    root.mkdir()
    config = _config(tmp_path, codex_root=root)
    conn = duckdb.connect(str(config.db_path))
    try:
        ensure_schema(conn, embed_dim=8)
        catalog = SourceCatalog(conn, clock=lambda: 1000.0)
        catalog.record_scan(
            "codex",
            str(root),
            started_at=999.0,
            finished_at=1000.0,
            discovered_count=1,
            failures=[],
        )
        path = str(root / "rollout-legacy.jsonl")
        catalog.observe("codex", str(root), path, SourceSignature(1, 2, 3, 4, 100))
        catalog.defer_until_changed("codex", path, "unsupported", {})

        status = reconciliation_status(config, conn=conn)
        assert status["pending"] == 1
        assert status["raw_indexing_ready"] is False
    finally:
        conn.close()


def test_status_notice_distinguishes_unprobed_vs_probed_verification_and_disk_full_reset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    unprobed = {
        "runtime_status": {"needs_index_verification": True},
        "index_divergence": None,
        "index_verification_probe_reason": None,
    }
    notice_unprobed = _index_divergence_notice(unprobed)
    assert notice_unprobed is not None
    assert "recall daemon restart" in notice_unprobed

    probed_unverifiable = {
        "runtime_status": {"needs_index_verification": True},
        "index_divergence": {
            "checked_at": "2026-11-22T21:17:36",
            "indexes_probed": 17,
            "samples_checked": 16,
            "samples_unverifiable": 1,  # e.g. overwritten later by live check_indexes!
            "diverged": [],
            "diverged_count": 0,
            "complete": True,
            "elapsed_seconds": 0.5,
        },
        "index_verification_probe_reason": "startup probe left 16 sampled key(s) unverifiable",
    }
    notice_probed = _index_divergence_notice(probed_unverifiable)
    assert notice_probed is not None
    assert "recall db rebuild-indexes" in notice_probed
    # Must report startup's 16 unverifiable keys, NOT live check_indexes's 1!
    assert "16 sampled key(s) unverifiable" in notice_probed
    assert "recall daemon restart" not in notice_probed

    # Live check_indexes without startup assessment must NOT trigger rebuild advice.
    live_only = {**probed_unverifiable, "index_verification_probe_reason": None}
    notice_live = _index_divergence_notice(live_only)
    assert notice_live is not None
    assert "recall daemon restart" in notice_live

    # Text stdout remains facts-only (REQ-CLI-012 / REQ-DAEMON-075).
    _print_index_health_text(probed_unverifiable, probed_unverifiable["runtime_status"])
    out = capsys.readouterr().out
    assert "still needed (startup probe left 16 sampled key(s) unverifiable" in out
    assert "rebuild-indexes" not in out


def test_note_disk_full_and_live_check_indexes_keep_startup_assessment_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services.rpc_server import RpcServer, _startup_verification_reason

    config = _config(tmp_path)
    server = RpcServer(config)
    startup_probe = IndexDivergenceReport(
        checked_at=datetime(2026, 11, 22, 21, 17, 36),
        indexes_probed=17,
        samples_checked=16,
        samples_unverifiable=16,
        diverged=(),
        complete=True,
        elapsed_seconds=0.1,
    )
    live_probe = replace(startup_probe, samples_unverifiable=1)
    server._last_index_probe = startup_probe
    server._index_verification_probe_reason = _startup_verification_reason(startup_probe)
    assert server._index_verification_probe_reason == (
        "startup probe left 16 sampled key(s) unverifiable"
    )

    async def scenario() -> None:
        # Case 1: same-epoch live check_indexes updates _last_index_probe but never mutates
        # the immutable startup verification assessment.
        async def fast_readonly(_fn: Any) -> IndexDivergenceReport:
            return live_probe

        monkeypatch.setattr(server, "_run_readonly", fast_readonly)
        await server._handle_check_indexes({"sample": 1}, None)
        assert server._last_index_probe == live_probe
        assert server._index_verification_probe_reason == (
            "startup probe left 16 sampled key(s) unverifiable"
        )

        # Case 2: disk-full mid-run invalidates both and prevents in-flight check_indexes
        # from repopulating _last_index_probe.
        async def slow_readonly(_fn: Any) -> IndexDivergenceReport:
            server._note_disk_full(OSError(28, "No space left on device"), "checkpoint")
            return live_probe

        monkeypatch.setattr(server, "_run_readonly", slow_readonly)
        await server._handle_check_indexes({"sample": 2}, None)
        assert server._last_index_probe is None
        assert server._index_verification_probe_reason is None

    asyncio.run(scenario())


def test_keyword_search_ready_stays_false_when_sidecar_startup_failed(
    tmp_path: Path,
) -> None:
    from recall.services.daemon import FtsSidecarStartupResult
    from recall.services.rpc_server import RpcServer

    config = _config(tmp_path)
    server = RpcServer(config)
    server._fts_sidecar_startup = FtsSidecarStartupResult(
        enabled=True,
        bootstrap_messages_processed=0,
        bootstrap_tool_calls_processed=0,
        bootstrap_messages_done=False,
        bootstrap_tool_calls_done=False,
        reconcile_pending_drained={},
        reconcile_orphans_backfilled={},
        reconcile_ghosts_deleted={},
        reconcile_pending_remaining={},
        last_run_at=datetime(2026, 11, 22, 21, 17, 36),
        error="incompatible sidecar schema",
    )
    assert server._keyword_startup_failed is True
    server._fts_sidecar_startup = FtsSidecarStartupResult(
        enabled=True,
        bootstrap_messages_processed=0,
        bootstrap_tool_calls_processed=0,
        bootstrap_messages_done=True,
        bootstrap_tool_calls_done=True,
        reconcile_pending_drained={},
        reconcile_orphans_backfilled={},
        reconcile_ghosts_deleted={},
        reconcile_pending_remaining={},
        last_run_at=datetime(2026, 11, 22, 21, 17, 36),
        error=None,
    )
    assert server._keyword_startup_failed is False


def test_reopened_park_growing_past_limit_records_new_signature_and_exits_source_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Specialist Blocking 3: growth before observation updates catalog signature before parking."""
    import duckdb
    from recall.db.fts_sidecar import open_sidecar, sidecar_path
    from recall.parsers import compressed as compressed_mod

    root = tmp_path / "codex"
    root.mkdir()
    plain = root / "rollout-2026-08-27T17-40-35-01a04561-d872-7de1-8f05-d7c10d957711.jsonl"
    body = b'{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}\n'
    _write_zst(plain, body)

    config = _config(tmp_path, codex_root=root)
    open_sidecar(sidecar_path(config.data_dir)).close()
    conn = duckdb.connect(str(config.db_path))
    try:
        ensure_schema(conn, embed_dim=8)
        parser = CodexParser(roots=(root,))
        catalog = SourceCatalog(conn, clock=lambda: 1000.0)
        catalog.record_scan(
            "codex",
            str(root),
            started_at=999.0,
            finished_at=1000.0,
            discovered_count=1,
            failures=[],
        )
        # Fully acknowledge initial compressed file, then park as unsupported_record & reopen.
        item = observe_path(parser, capture_path(parser, plain), conn=conn)
        commit_prepared_raw_sources(
            prepare_raw_sources((item,), {"codex": parser}), config, conn=conn
        )
        catalog.defer_unsupported(
            "codex",
            str(plain),
            {"records": [{"kind": "unsupported_record", "detail": "old", "byte_offset": 0}]},
            parser_revision="older-rev",
        )
        assert reopen_unsupported_parks(conn, 2000.0) == 1

        # File grows on disk past limit BEFORE any catalog observation!
        big_body = body * 50
        _write_zst(plain, big_body)
        reopened_item = catalog.get("codex", str(plain))
        assert reopened_item is not None
        monkeypatch.setattr(compressed_mod, "MAX_COMPRESSED_CAPTURE_BYTES", len(body) * 2)

        prepared = prepare_raw_sources((reopened_item,), {"codex": parser})
        assert prepared[0].unsupported_limit is not None
        commit_prepared_raw_sources(prepared, config, conn=conn)

        parked = catalog.get("codex", str(plain))
        assert parked is not None
        assert parked.current is False
        assert parked.desired_generation > parked.committed_generation
        assert parked.last_error == "unsupported"
        assert parked.next_retry_at == PARKED_RETRY_AT
        assert reconciliation_status(config, conn=conn)["pending"] == 0
        assert unsupported_summary(conn)["files"] == 1

        # Reopening again exposes the over-limit generation as pending (NOT SOURCE_CURRENT)!
        catalog.defer_unsupported(
            "codex",
            str(plain),
            {
                "records": [
                    {
                        "kind": "unsupported_limit",
                        "detail": "compressed rollout decoded content exceeds 96 MiB",
                        "byte_offset": 0,
                    }
                ]
            },
            parser_revision="older-rev-2",
        )
        assert reopen_unsupported_parks(conn, 3000.0) == 1
        assert reconciliation_status(config, conn=conn)["pending"] == 1
        assert reconciliation_status(config, conn=conn)["raw_indexing_ready"] is False
    finally:
        conn.close()


def test_compressed_limit_takes_precedence_over_truncated_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decoded prefix crossing the ceiling raises CompressedCaptureLimitError before EOF."""
    from recall.parsers import compressed as compressed_mod

    zst_path = tmp_path / "rollout-truncated-over-limit.jsonl.zst"
    big = b'{"type":"event_msg","payload":{"type":"user_message","message":"abcdefghij"}}\n' * 10
    # Complete frame whose decoded prefix crosses the 256-byte limit, followed by a truncated frame.
    zst_path.write_bytes(zstd.compress(big) + zstd.compress(b"tail")[:-2])
    monkeypatch.setattr(compressed_mod, "MAX_COMPRESSED_CAPTURE_BYTES", 256)

    with (
        pytest.raises(CompressedCaptureLimitError, match="decoded content exceeds"),
        CompressedCapture(zst_path),
    ):
        pass


def test_v2_cursor_survives_zsh_echo_json_pipeline_and_accepts_legacy() -> None:
    cursor = CatalogCursor(Source.CLAUDE_CODE, "/Users/allen/.claude/projects/a/session.jsonl")
    assert cursor.token.startswith("v2:")
    assert "\x1f" not in cursor.token
    assert CatalogCursor.parse(cursor.token) == cursor
    assert CatalogCursor.parse(cursor.db_key) == cursor

    payload = json.dumps({"reconciliation": {"next_cursor": cursor.token}}, separators=(",", ":"))
    zsh = subprocess.run(
        ["zsh", "-c", 'out=$1; echo "$out"', "--", payload],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = json.loads(zsh.stdout)
    assert parsed["reconciliation"]["next_cursor"] == cursor.token


def test_verified_startup_rebuild_clears_needs_index_verification_with_unverifiable_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verified full rebuild clears needs_index_verification even when samples_unverifiable > 0."""
    from recall.db.connection import connect
    from recall.db.maintenance import DivergedKey
    from recall.services import self_repair
    from recall.services.runtime_state import load_runtime_status, set_needs_index_verification
    from recall.services.self_repair import remember_fatal_failure, run_startup_self_repair

    config = _config(tmp_path)
    conn = connect(config)
    try:
        set_needs_index_verification(conn, True)
    finally:
        conn.close()

    t0 = datetime(2026, 11, 22, 5, 42, 0)
    remember_fatal_failure(
        config.data_dir,
        RuntimeError("No space left on device"),
        site="checkpoint",
        now=t0,
    )
    diverged = IndexDivergenceReport(
        checked_at=t0,
        indexes_probed=17,
        samples_checked=16,
        samples_unverifiable=16,
        diverged=(
            DivergedKey(
                table="session_state",
                column="git_repo",
                key="/Users/dev/code/app",
                index_count=0,
                full_count=1,
            ),
        ),
        complete=True,
        elapsed_seconds=0.1,
    )
    # Post-rebuild probe still has 16 unverifiable samples (compound/low-cardinality indexes)!
    post_rebuild = replace(diverged, diverged=())
    assert post_rebuild.coverage_verified is False
    answers = iter([diverged, post_rebuild])
    monkeypatch.setattr(self_repair, "probe_index_divergence", lambda *a, **k: next(answers))

    outcome = run_startup_self_repair(config, now=t0)
    assert outcome.action == "rebuilt"
    assert load_runtime_status(config).needs_index_verification is False
