"""Unit tests for the RPC server dispatch logic."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import duckdb
import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    CompactionConfig,
    ContextConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.daemon_status import DaemonSchedulerStatus, RuntimeStatus
from recall.core.rpc_types import APP_CONFIRMATION_REQUIRED, config_fingerprint
from recall.core.types import SchedulerKind, Source
from recall.services.daemon import FtsSidecarStartupResult, run_startup_fts_sidecar_sync
from recall.services.rpc_server import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    ClientConnection,
    RpcError,
    RpcServer,
    _config_with_context_mode,
    _PeerStreamReader,
)

from conftest import _can_acquire_duckdb_lock

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, socket_dir: Path) -> None:
    """Keep default-constructed servers and their failure markers inside each test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in tuple(os.environ):
        if key.startswith("RECALL_"):
            monkeypatch.delenv(key)


@pytest.fixture
def server():
    return RpcServer()


def test_default_server_failure_storage_is_owned(server: RpcServer, tmp_path: Path) -> None:
    """Fatal-error tests must never spool synthetic failures into the user's data."""
    from recall.services.self_repair import read_failure_marker

    assert server._config.data_dir.is_relative_to(tmp_path)
    assert server._stop_on_fatal_db(duckdb.FatalException("database has been invalidated"), "owned")
    marker = read_failure_marker(server._config.data_dir)
    assert marker is not None and marker.failure is not None
    assert marker.failure.site == "owned"


def test_note_oom_logs_actionable_memory_hint(
    server: RpcServer, caplog: pytest.LogCaptureFixture
) -> None:
    import duckdb

    with caplog.at_level(logging.ERROR):
        server._note_oom(duckdb.OutOfMemoryException("Out of Memory Error"), "embed snapshot")

    messages = [record.getMessage() for record in caplog.records]
    assert any("memory_limit" in m and "RECALL_DUCKDB_MEMORY_LIMIT" in m for m in messages)


def test_note_oom_ignores_non_oom(server: RpcServer, caplog: pytest.LogCaptureFixture) -> None:
    import duckdb

    with caplog.at_level(logging.ERROR):
        server._note_oom(duckdb.IOException("some unrelated io error"), "label")

    assert not any("memory_limit" in r.getMessage() for r in caplog.records)


def _runtime_status() -> RuntimeStatus:
    return RuntimeStatus(
        last_attempted_at=None,
        last_successful_at=None,
        last_run_kind=None,
        last_index_summary=None,
        last_failure_message=None,
        last_failure_at=None,
        installed_scheduler=None,
    )


def _scheduler_status() -> DaemonSchedulerStatus:
    return DaemonSchedulerStatus(
        configured_scheduler=SchedulerKind.AUTO,
        scheduler=None,
        installed=False,
        command="/tmp/recall daemon --once",
        config_path="/tmp/config.toml",
        artifact_paths=(),
        runtime_status=_runtime_status(),
    )


def _tmp_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    compaction: CompactionConfig,
) -> AppConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", "data")
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.delenv("RECALL_COMPACTION_AUTO", raising=False)
    monkeypatch.delenv("RECALL_COMPACTION_THRESHOLD", raising=False)
    monkeypatch.delenv("RECALL_COMPACTION_INTERVAL_HOURS", raising=False)
    return replace(AppConfig.load(), compaction=compaction)


@asynccontextmanager
async def _running_server(
    server: RpcServer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    watch: bool,
    shutdown_timeout: float,
) -> AsyncIterator[None]:
    """Charge startup to its 30-second readiness budget, before feature checks."""
    ready = asyncio.Event()
    start_embed_phase = server._start_embed_phase

    async def mark_ready() -> asyncio.Task[None] | None:
        result = await start_embed_phase()
        ready.set()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(server, "_start_embed_phase", mark_ready)
        task = asyncio.create_task(server.start(watch=watch))
        readiness = asyncio.create_task(ready.wait())
        try:
            done, _ = await asyncio.wait(
                (task, readiness), timeout=30.0, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                await task
                pytest.fail("server exited before the feature check")
            assert readiness in done, "startup did not complete within the readiness budget"
            yield
        finally:
            readiness.cancel()
            await asyncio.gather(readiness, return_exceptions=True)
            server.request_shutdown()
            await asyncio.wait_for(task, timeout=shutdown_timeout)


class TestMethodRegistry:
    def test_all_handlers_are_coroutines(self, server: RpcServer) -> None:
        for name, handler in server._methods.items():
            assert inspect.iscoroutinefunction(handler), f"{name} handler is not async"


class TestSignalShutdown:
    def test_signal_shutdown_terminates_active_codex_processes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = 0

        def terminate_spy() -> int:
            nonlocal calls
            calls += 1
            return 1

        monkeypatch.setattr(
            "recall.services.context_backends.codex_cli.terminate_active_codex_processes",
            terminate_spy,
        )
        # The server resolves the terminator when it is built, never at shutdown,
        # so the stand-in has to be in place before construction.
        server = RpcServer()

        server._signal_shutdown()

        assert calls == 1
        assert server._shutdown_event.is_set()


def test_rpc_server_start_runs_snapshot_gc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tmp_config(
        tmp_path,
        monkeypatch,
        compaction=CompactionConfig(auto_trigger=False),
    )
    config = replace(
        config,
        daemon=DaemonConfig(embed=False),
        cli=CliConfig(),
        embedding=EmbeddingConfig(),
    )
    snapshots = config.data_dir / "snapshots"
    snapshots.mkdir(parents=True)
    stale = snapshots / "stale"
    stale.write_text("stale")
    stale_time = time.time() - (10 * 86_400)
    os.utime(stale, (stale_time, stale_time))

    async def init_fts_noop(self: RpcServer, *, reason: str) -> None:
        del self

    monkeypatch.setattr(RpcServer, "_init_fts", init_fts_noop)
    server = RpcServer(config=config)

    async def run_until_gc() -> None:
        async with _running_server(server, monkeypatch, watch=False, shutdown_timeout=2):
            for _ in range(100):
                if not stale.exists():
                    break
                await asyncio.sleep(0.01)
            assert not stale.exists()

    asyncio.run(run_until_gc())


def test_rpc_server_runs_fts_sidecar_sync_on_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _tmp_config(
            tmp_path,
            monkeypatch,
            compaction=CompactionConfig(auto_trigger=False),
        ),
        daemon=DaemonConfig(embed=False),
        cli=CliConfig(),
        embedding=EmbeddingConfig(),
        fts=FtsConfig(backend="sqlite_sidecar"),
    )

    async def stop_after_sidecar_sync(self: RpcServer, *, reason: str) -> None:
        self.request_shutdown()

    # FTS initialization follows the real startup sidecar synchronization.
    monkeypatch.setattr(RpcServer, "_init_fts", stop_after_sidecar_sync)
    server = RpcServer(config=config)

    async def run_until_sidecar_sync() -> None:
        try:
            await asyncio.wait_for(server.start(watch=False), timeout=30)
        finally:
            await server.stop()

        assert isinstance(server._fts_sidecar_startup, FtsSidecarStartupResult)
        assert server._fts_sidecar_startup.enabled is True
        assert server._fts_sidecar_startup.error is None

    asyncio.run(run_until_sidecar_sync())


@pytest.mark.parametrize(
    "fts", [FtsConfig(backend="sqlite_sidecar"), FtsConfig(backend="duckdb", fields=())]
)
def test_noop_fts_refresh_does_not_wait_for_the_writer(
    fts: FtsConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config = replace(AppConfig.load(), fts=fts)
    server = RpcServer(config=config)
    server._keyword_dirty = True

    async def scenario() -> None:
        async with server._write_lock:
            for _ in range(3):
                await asyncio.wait_for(server._init_fts(reason="watch debounce"), timeout=0.1)

    with caplog.at_level(logging.DEBUG, logger="recall.rpc_server"):
        asyncio.run(scenario())

    assert not config.db_path.exists()
    assert server._keyword_dirty is False
    records = [r for r in caplog.records if "FTS refresh skipped" in r.getMessage()]
    assert len(records) == 3
    assert all(r.levelno == logging.DEBUG for r in records)
    assert all("watch debounce" in r.getMessage() for r in records)
    assert not any("on startup" in r.getMessage() for r in caplog.records)


def test_rpc_server_status_surfaces_fts_sidecar_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _tmp_config(
            tmp_path,
            monkeypatch,
            compaction=CompactionConfig(auto_trigger=False),
        ),
        daemon=DaemonConfig(embed=False),
        cli=CliConfig(),
        embedding=EmbeddingConfig(),
        fts=FtsConfig(backend="sqlite_sidecar"),
    )
    server = RpcServer(config=config)
    server._fts_sidecar_startup = run_startup_fts_sidecar_sync(config)
    monkeypatch.setattr("recall.services.daemon._read_crontab", lambda: "")

    async def fake_run_readonly(fn):
        from recall.db.schema import ensure_schema

        with duckdb.connect() as conn:
            ensure_schema(conn)
            return fn(conn)

    monkeypatch.setattr(server, "_run_readonly", fake_run_readonly)

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(server._handle_daemon_status({}, None))
    finally:
        loop.close()

    assert result["fts_sidecar_enabled"] is True
    assert result["fts_sidecar_bootstrap_messages_done"] is True
    assert result["fts_sidecar_bootstrap_tool_calls_done"] is True
    assert result["fts_sidecar_reconcile_pending_remaining_messages"] == 0
    assert result["fts_sidecar_reconcile_pending_remaining_tool_calls"] == 0
    assert result["fts_sidecar_last_run_at"] is not None


def test_rpc_server_status_names_the_serving_daemon_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-DAEMON-074: a status served by the daemon names that daemon's own pid."""
    server = RpcServer()
    monkeypatch.setattr(
        "recall.services.daemon.daemon_status",
        lambda **_kwargs: _scheduler_status(),
    )

    async def fake_run_readonly(fn):
        from recall.db.schema import ensure_schema

        with duckdb.connect() as conn:
            ensure_schema(conn)
            return fn(conn)

    monkeypatch.setattr(server, "_run_readonly", fake_run_readonly)

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(server._handle_daemon_status({}, None))
    finally:
        loop.close()

    assert result["daemon_pid"] == os.getpid()


def _package_missing(_name: str) -> str:
    raise importlib.metadata.PackageNotFoundError("recall")


def _package_version(version: str) -> Callable[[str], str]:
    return lambda _name: version


class TestDaemonStatusVersions:
    @pytest.mark.parametrize(
        ("at_start", "at_status", "daemon_version", "binary_version", "drift"),
        [
            pytest.param(
                _package_version("0.10.3"),
                _package_version("0.10.4"),
                "0.10.3",
                "0.10.4",
                True,
                id="binary-changed",
            ),
            pytest.param(
                _package_version("0.10.4"),
                _package_version("0.10.4"),
                "0.10.4",
                "0.10.4",
                False,
                id="equal-versions",
            ),
            # The real-world failure: the daemon started against a populated
            # venv, then `uv tool install --reinstall` rewrote it, so the running
            # daemon's metadata lookup raises PackageNotFoundError. The CLI must
            # see that as drift, or the missing metadata masquerades as healthy.
            pytest.param(
                _package_version("0.10.5"),
                _package_missing,
                "0.10.5",
                None,
                True,
                id="binary-metadata-disappeared",
            ),
            # With no daemon version to compare against, flagging drift would be
            # a phantom warning shown to every pre-metadata install.
            pytest.param(
                _package_missing,
                _package_missing,
                None,
                None,
                False,
                id="both-unknown",
            ),
        ],
    )
    def test_daemon_status_reports_version_drift(
        self,
        monkeypatch: pytest.MonkeyPatch,
        at_start: Callable[[str], str],
        at_status: Callable[[str], str],
        daemon_version: str | None,
        binary_version: str | None,
        drift: bool,
    ) -> None:
        monkeypatch.setattr(importlib.metadata, "version", at_start)
        server = RpcServer()
        assert server._daemon_version == daemon_version

        monkeypatch.setattr(importlib.metadata, "version", at_status)
        monkeypatch.setattr(
            "recall.services.daemon.daemon_status",
            lambda **_kwargs: _scheduler_status(),
        )

        async def fake_run_readonly(fn):
            from recall.db.schema import ensure_schema

            with duckdb.connect() as conn:
                ensure_schema(conn)
                return fn(conn)

        monkeypatch.setattr(server, "_run_readonly", fake_run_readonly)

        result = _run(server._handle_daemon_status({}, None))

        assert result["daemon_version"] == daemon_version
        assert result["binary_version"] == binary_version
        assert result["version_drift"] is drift


class TestRequestProcessing:
    def test_search_requires_query(self, server: RpcServer) -> None:
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(RpcError) as exc_info:
                loop.run_until_complete(server._handle_search({}, None))
            assert "query is required" in exc_info.value.message
        finally:
            loop.close()

    def test_show_requires_session_id(self, server: RpcServer) -> None:
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(RpcError) as exc_info:
                loop.run_until_complete(server._handle_show({}, None))
            assert "session_id is required" in exc_info.value.message
        finally:
            loop.close()

    def test_index_recreate_requires_confirmed(self, server: RpcServer) -> None:
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(RpcError) as exc_info:
                loop.run_until_complete(server._handle_index({"recreate": True}, None))
            assert "confirmed" in exc_info.value.message.lower()
        finally:
            loop.close()

    def test_stats_skills_parses_repeatable_sources(
        self,
        server: RpcServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from recall.services import analytics as analytics_module
        from recall.services.analytics import (
            SkillCoverage,
            SkillPopulation,
            SkillUsageResult,
        )

        captured: dict[str, object] = {}

        def fake_skill_usage(**kwargs: object) -> SkillUsageResult:
            captured.update(kwargs)
            population = SkillPopulation(0, 0, 0)
            return SkillUsageResult(
                rows=(),
                coverage=SkillCoverage(
                    scope="local",
                    expected_hosts=("control",),
                    successful_hosts=("control",),
                    covered_sources=("codex", "grok"),
                    considered_sessions=0,
                    attributed_invocations=0,
                    unattributed_candidates=0,
                    control=population,
                ),
            )

        async def fake_run_readonly(fn: Callable[[object], object]) -> object:
            return fn(None)

        monkeypatch.setattr(analytics_module, "skill_usage", fake_skill_usage)
        monkeypatch.setattr(server, "_run_readonly", fake_run_readonly)

        result = _run(
            server._handle_stats_skills(
                {"source": ["codex", "grok-build"], "since": "7d"},
                None,
            )
        )

        assert captured["sources"] == (Source.CODEX, Source.GROK)
        assert isinstance(captured["since"], datetime)
        assert result["coverage"]["covered_sources"] == ("codex", "grok")

    def test_stats_skills_rejects_non_array_source(self, server: RpcServer) -> None:
        with pytest.raises(RpcError) as exc_info:
            _run(server._handle_stats_skills({"source": "codex"}, None))

        assert exc_info.value.code == INVALID_PARAMS


class TestErrorCodes:
    def test_error_code_constants(self) -> None:
        assert PARSE_ERROR == -32700
        assert INVALID_REQUEST == -32600
        assert METHOD_NOT_FOUND == -32601
        assert INVALID_PARAMS == -32602
        assert INTERNAL_ERROR == -32603
        assert APP_CONFIRMATION_REQUIRED == -32001


class TestConfigWithContextMode:
    """Regression: an RPC --context override must not silently drop ContextConfig fields
    that aren't explicitly enumerated in the helper. Prior to REQ-CTX-015 the helper
    rebuilt the config from a hand-rolled subset; base_url and timeout were therefore
    dropped on every `recall index --context llm-remote` invocation.
    """

    def _build_config_with_full_context(self) -> AppConfig:
        context = ContextConfig(
            mode="template",
            fallback="off",
            model="custom-model",
            max_tokens=64,
            batch_size=4,
            concurrency=3,
            min_chars=80,
            base_url="http://litellm.local:4000",
            timeout=25.0,
        )
        return AppConfig(
            data_dir=Path.home() / "data",
            db_path=Path.home() / "data/recall.duckdb",
            lock_path=Path.home() / "data/recall.lock",
            config_path=Path.home() / "config.toml",
            fts=cast(Any, None),  # not exercised by the helper
            embedding=EmbeddingConfig(context=context),
            daemon=cast(Any, None),  # not exercised by the helper
            cli=cast(Any, None),  # not exercised by the helper
            compaction=CompactionConfig(),
        )

    def test_override_preserves_all_context_fields(self) -> None:
        config = self._build_config_with_full_context()

        result = _config_with_context_mode(config, "llm-remote")

        ctx = result.embedding.context
        assert ctx.mode == "llm-remote"  # the explicit override
        # Every other field must survive the override.
        assert ctx.fallback == "off"
        assert ctx.model == "custom-model"
        assert ctx.max_tokens == 64
        assert ctx.batch_size == 4
        assert ctx.concurrency == 3
        assert ctx.min_chars == 80
        assert ctx.base_url == "http://litellm.local:4000"
        assert ctx.timeout == 25.0

    def test_none_mode_returns_config_unchanged(self) -> None:
        config = self._build_config_with_full_context()

        result = _config_with_context_mode(config, None)

        # No override requested -> return the same object identity.
        assert result is config


class TestSerialization:
    def test_serialize_handles_primitives(self) -> None:
        from recall.services.rpc_server import _serialize

        assert _serialize(None) is None
        assert _serialize(42) == 42
        assert _serialize("hello") == "hello"
        assert _serialize(True) is True
        assert _serialize(3.14) == 3.14

    def test_serialize_handles_datetime(self) -> None:
        from datetime import datetime

        from recall.services.rpc_server import _serialize

        dt = datetime(2024, 1, 15, 10, 30, 0)
        assert _serialize(dt) == "2024-01-15T10:30:00"

    def test_serialize_handles_enum(self) -> None:
        from recall.core.types import Source
        from recall.services.rpc_server import _serialize

        assert _serialize(Source.CLAUDE_CODE) == "claude_code"

    def test_serialize_handles_dataclass(self) -> None:
        from dataclasses import dataclass

        from recall.services.rpc_server import _serialize

        @dataclass(frozen=True)
        class Sample:
            name: str
            count: int

        result = _serialize(Sample(name="test", count=5))
        assert result == {"name": "test", "count": 5}

    def test_serialize_handles_nested_structures(self) -> None:
        from recall.services.rpc_server import _serialize

        data = {"items": [1, "two", {"nested": True}], "count": 3}
        result = _serialize(data)
        assert result == {"items": [1, "two", {"nested": True}], "count": 3}

    def test_serialize_handles_path(self) -> None:
        from pathlib import Path

        from recall.services.rpc_server import _serialize

        assert _serialize(Path("/tmp/test")) == "/tmp/test"


@_requires_duckdb_lock
class TestConcurrentReadServicing:
    """Verify REQ-RPC-012: read RPCs are served concurrently."""

    def test_reads_run_concurrently_not_serially(self, tmp_path, monkeypatch) -> None:
        """Two list calls with 0.5s delay each finish in ~0.5s, not ~1.0s.

        Proves reads execute in parallel via run_in_executor, not
        sequentially on the event loop.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

        from recall.core.config import AppConfig
        from recall.db import connect
        from recall.services import sessions as sessions_mod

        config = AppConfig.load()
        # Ensure schema is created before concurrent reads
        conn = connect(config)
        conn.close()
        server = RpcServer(config=config)

        DELAY = 0.5
        original_list = sessions_mod.list_sessions

        def slow_list(**kwargs):
            time.sleep(DELAY)
            return original_list(**kwargs)

        monkeypatch.setattr(sessions_mod, "list_sessions", slow_list)

        async def run_two():
            t1 = asyncio.create_task(server._handle_list({}, None))
            t2 = asyncio.create_task(server._handle_list({}, None))
            start = time.monotonic()
            await asyncio.gather(t1, t2)
            return time.monotonic() - start

        loop = asyncio.new_event_loop()
        try:
            elapsed = loop.run_until_complete(run_two())
        finally:
            loop.close()

        # If concurrent: elapsed ≈ 0.5s. If serial: elapsed ≈ 1.0s.
        assert elapsed < 0.9, (
            f"reads took {elapsed:.2f}s — expected <0.9s for concurrent, "
            f"got ≥1.0s which means they serialized"
        )

    def test_read_not_blocked_by_write_lock(self, tmp_path, monkeypatch) -> None:
        """A read completes while the write lock is held."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

        from recall.core.config import AppConfig

        config = AppConfig.load()
        server = RpcServer(config=config)

        async def run():
            async with server._write_lock:
                return await server._handle_stats({}, None)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(run())
        finally:
            loop.close()

        assert result is not None

    def test_status_remains_responsive_while_poll_parser_capture_blocks(
        self, tmp_path, monkeypatch
    ) -> None:
        """A slow raw capture runs before the coordinator takes the writer."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))

        from recall.core.config import AppConfig

        server = RpcServer(config=AppConfig.load())
        entered = threading.Event()
        release = threading.Event()
        capture_returned = threading.Event()

        def blocked_capture(_config, *, cancelled):
            entered.set()
            assert release.wait(30.0), "test did not release parser capture"
            capture_returned.set()
            yield from ()

        monkeypatch.setattr("recall.services.coordinator.prepare_raw_cycle", blocked_capture)
        monkeypatch.setattr("recall.services.coordinator.is_paused", lambda _config: True)

        async def run() -> None:
            task = asyncio.create_task(server._run_reconciliation_poll_loop())
            try:
                assert await asyncio.to_thread(entered.wait, 5.0)
                # The capture holds until `release`, which is set only after status
                # returns. A status call that waited on the capture would time out
                # here rather than return, so the budget is generous for loaded hosts.
                status = await asyncio.wait_for(server._handle_daemon_status({}, None), timeout=5.0)
                assert not capture_returned.is_set()
                assert status["reconciliation"]["rpc_ready"] is False
            finally:
                server._shutdown_event.set()
                release.set()
                await asyncio.wait_for(task, timeout=5.0)

        _run(run())


class TestAutoCompactMonitor:
    def _watch_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        auto_trigger: bool,
    ) -> AppConfig:
        return _tmp_config(
            tmp_path,
            monkeypatch,
            compaction=CompactionConfig(
                auto_trigger=auto_trigger,
                check_interval_hours=cast(Any, 0.00001),
            ),
        )

    def _stub_start_dependencies(
        self,
        server: RpcServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_init_fts(*, reason: str) -> None:
            pass

        async def fake_start_embed_phase() -> None:
            return None

        async def fake_start_watch_mode() -> tuple[object, asyncio.Task[None]]:
            async def fake_watch_loop() -> None:
                await server._shutdown_event.wait()

            return object(), asyncio.create_task(fake_watch_loop())

        async def fake_stop_watch_mode(_runtime: object) -> None:
            pass

        monkeypatch.setattr(server, "_init_fts", fake_init_fts)
        monkeypatch.setattr(server, "_start_embed_phase", fake_start_embed_phase)
        monkeypatch.setattr(server, "_start_watch_mode", fake_start_watch_mode)
        monkeypatch.setattr(server, "_stop_watch_mode", fake_stop_watch_mode)

    def test_watch_mode_runs_auto_compact_monitor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = self._watch_config(tmp_path, monkeypatch, auto_trigger=True)
        server = RpcServer(config=config)
        self._stub_start_dependencies(server, monkeypatch)

        async def run() -> list[tuple[RpcServer, AppConfig]]:
            loop = asyncio.get_running_loop()
            invoked = asyncio.Event()
            calls: list[tuple[RpcServer, AppConfig]] = []

            def compact_spy(state: RpcServer, cfg: AppConfig) -> None:
                calls.append((state, cfg))
                loop.call_soon_threadsafe(invoked.set)

            monkeypatch.setattr("recall.services.daemon._maybe_run_auto_compact", compact_spy)

            async with _running_server(server, monkeypatch, watch=True, shutdown_timeout=1.0):
                await asyncio.wait_for(invoked.wait(), timeout=1.0)
            return calls

        loop = asyncio.new_event_loop()
        try:
            calls = loop.run_until_complete(run())
        finally:
            loop.close()

        assert calls == [(server, config)]

    def test_auto_compact_monitor_opt_out_does_not_call_helper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = self._watch_config(tmp_path, monkeypatch, auto_trigger=False)
        server = RpcServer(config=config)
        self._stub_start_dependencies(server, monkeypatch)
        calls = 0

        def compact_spy(_state: RpcServer, _cfg: AppConfig) -> None:
            nonlocal calls
            calls += 1

        monkeypatch.setattr("recall.services.daemon._maybe_run_auto_compact", compact_spy)

        async def run() -> None:
            async with _running_server(server, monkeypatch, watch=True, shutdown_timeout=1.0):
                await asyncio.sleep(0.05)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run())
        finally:
            loop.close()

        assert calls == 0

    def test_daemon_run_still_checks_auto_compaction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = self._watch_config(tmp_path, monkeypatch, auto_trigger=True)
        server = RpcServer(config=config)
        calls: list[tuple[RpcServer, AppConfig]] = []

        monkeypatch.setattr(server, "_get_conn", lambda: object())

        from recall.services.indexer import IndexSummary

        monkeypatch.setattr(
            server,
            "_reconcile_index_request",
            AsyncMock(return_value=IndexSummary(total=0, indexed=0, skipped=0, failed=0)),
        )
        monkeypatch.setattr(
            "recall.services.runtime_state.record_run_attempt",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "recall.services.runtime_state.record_run_success",
            lambda *_args, **_kwargs: None,
        )

        def compact_spy(state: RpcServer, cfg: AppConfig) -> None:
            calls.append((state, cfg))

        monkeypatch.setattr("recall.services.daemon._maybe_run_auto_compact", compact_spy)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                server._handle_daemon_run({"once": True, "embed": False}, None)
            )
        finally:
            loop.close()

        assert calls == [(server, config)]
        assert result.index_summary.total == 0


class TestWatchModeConfigReload:
    def test_index_handler_reloads_runtime_config_before_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base_config = _tmp_config(
            tmp_path,
            monkeypatch,
            compaction=CompactionConfig(auto_trigger=False),
        )
        base_config.config_path.parent.mkdir(parents=True, exist_ok=True)
        base_config.config_path.write_text(
            '[embedding.context]\nmode = "template"\n',
            encoding="utf-8",
        )
        off_config = replace(
            base_config,
            embedding=replace(base_config.embedding, context=ContextConfig(mode="off")),
        )
        template_config = replace(
            base_config,
            embedding=replace(base_config.embedding, context=ContextConfig(mode="template")),
        )
        server = RpcServer(config=off_config)

        from recall.services.indexer import IndexSummary

        captured_modes: list[str] = []
        monkeypatch.setattr(
            "recall.services.rpc_server.AppConfig.load",
            lambda: template_config,
        )

        async def capture_index_session(**kwargs: Any) -> IndexSummary:
            captured_modes.append(kwargs["config"].embedding.context.mode)
            return IndexSummary(total=0, indexed=0, skipped=0, failed=0)

        monkeypatch.setattr(server, "_manual_index_request", capture_index_session)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(server._handle_index({"full": True, "embed": False}, None))
        finally:
            loop.close()

        assert captured_modes == ["template"]

    def test_daemon_run_handler_reloads_runtime_config_before_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base_config = _tmp_config(
            tmp_path,
            monkeypatch,
            compaction=CompactionConfig(auto_trigger=False),
        )
        base_config.config_path.parent.mkdir(parents=True, exist_ok=True)
        base_config.config_path.write_text(
            '[embedding.context]\nmode = "template"\n',
            encoding="utf-8",
        )
        off_config = replace(
            base_config,
            embedding=replace(base_config.embedding, context=ContextConfig(mode="off")),
        )
        template_config = replace(
            base_config,
            embedding=replace(base_config.embedding, context=ContextConfig(mode="template")),
        )
        server = RpcServer(config=off_config)

        from recall.services.indexer import IndexSummary

        captured_modes: list[str] = []
        monkeypatch.setattr(
            "recall.services.rpc_server.AppConfig.load",
            lambda: template_config,
        )
        monkeypatch.setattr(server, "_get_conn", lambda: object())
        monkeypatch.setattr(
            "recall.services.runtime_state.record_run_attempt",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "recall.services.runtime_state.record_run_success",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr("recall.services.daemon._maybe_run_auto_compact", lambda *_args: None)

        async def capture_index_session(**kwargs: Any) -> IndexSummary:
            captured_modes.append(kwargs["config"].embedding.context.mode)
            return IndexSummary(total=0, indexed=0, skipped=0, failed=0)

        monkeypatch.setattr(server, "_reconcile_index_request", capture_index_session)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(server._handle_daemon_run({"once": True, "embed": False}, None))
        finally:
            loop.close()

        assert captured_modes == ["template"]

    def test_runtime_config_reload_updates_cached_config_and_index_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base_config = _tmp_config(
            tmp_path,
            monkeypatch,
            compaction=CompactionConfig(auto_trigger=False),
        )
        base_config.config_path.parent.mkdir(parents=True, exist_ok=True)
        base_config.config_path.write_text(
            '[embedding.context]\nmode = "off"\n',
            encoding="utf-8",
        )
        server = RpcServer(config=AppConfig.load())

        base_config.config_path.write_text(
            '[embedding.context]\nmode = "template"\n',
            encoding="utf-8",
        )
        reloaded = server._load_runtime_config()

        assert reloaded.embedding.context.mode == "template"
        assert server._config.embedding.context.mode == "template"
        assert server._config_fp == config_fingerprint(reloaded)

        from recall.services.indexer import IndexSummary

        captured_modes: list[str] = []

        async def capture_index_session(**kwargs: Any) -> IndexSummary:
            captured_modes.append(kwargs["config"].embedding.context.mode)
            return IndexSummary(total=0, indexed=0, skipped=0, failed=0)

        monkeypatch.setattr(server, "_manual_index_request", capture_index_session)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(server._handle_index({"full": True, "embed": False}, None))
        finally:
            loop.close()

        assert captured_modes == ["template"]


class _FetchOneCursor:
    def __init__(self, row: tuple[int, ...] = (1,)) -> None:
        self.row = row
        self.fetchone_called = False

    def fetchone(self) -> tuple[int, ...]:
        self.fetchone_called = True
        return self.row


class _RecoveryConn:
    def __init__(
        self,
        *,
        on_execute: Callable[[str], Any] | None = None,
        select_row: tuple[int, ...] = (1,),
    ) -> None:
        self.on_execute = on_execute
        self.select_row = select_row
        self.sql: list[str] = []
        self.closed = False
        self.select_cursor: _FetchOneCursor | None = None

    def execute(self, sql: str) -> Any:
        self.sql.append(sql)
        if self.on_execute is not None:
            result = self.on_execute(sql)
            if result is not None:
                return result
        if sql == "SELECT 1":
            self.select_cursor = _FetchOneCursor(self.select_row)
            return self.select_cursor
        return self

    def close(self) -> None:
        self.closed = True

    def fetchall(self) -> list[Any]:
        return []


def _run(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _abort_error(message: str = "Current transaction is aborted, please ROLLBACK") -> Exception:
    return duckdb.TransactionException(message)


class TestSharedConnRecovery:
    def _server_with_conn(self, conn: Any) -> RpcServer:
        server = RpcServer()
        server._conn = conn
        return server

    def test_rollback_after_aborted_txn(self) -> None:
        conn = _RecoveryConn()
        server = self._server_with_conn(conn)

        server._recover_shared_conn(_abort_error(), "unit")

        assert conn.sql == ["ROLLBACK", "SELECT 1"]
        assert conn.select_cursor is not None
        assert conn.select_cursor.fetchone_called is True
        assert server._conn is conn

    def test_rollback_swallows_no_active_txn(self) -> None:
        def on_execute(sql: str) -> None:
            if sql == "ROLLBACK":
                raise duckdb.TransactionException("no transaction is active")

        conn = _RecoveryConn(on_execute=on_execute)
        server = self._server_with_conn(conn)

        server._recover_shared_conn(_abort_error(), "unit")

        assert conn.closed is False
        assert server._conn is conn
        assert conn.sql == ["ROLLBACK", "SELECT 1"]

    def test_close_and_reopen_when_rollback_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def on_execute(sql: str) -> None:
            if sql == "ROLLBACK":
                raise duckdb.TransactionException("connection has been invalidated")

        conn = _RecoveryConn(on_execute=on_execute)
        fresh = _RecoveryConn()
        server = self._server_with_conn(conn)

        monkeypatch.setattr(server, "_open_conn_unlocked", lambda: fresh)

        server._recover_shared_conn(_abort_error(), "unit")

        assert conn.closed is True
        assert server._conn is None
        assert server._get_conn() is fresh

    def test_rollback_then_health_check_failure_triggers_close_clear(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def on_execute(sql: str) -> None:
            if sql == "SELECT 1":
                raise duckdb.Error("connection invalidated")

        conn = _RecoveryConn(on_execute=on_execute)
        fresh = _RecoveryConn()
        server = self._server_with_conn(conn)
        monkeypatch.setattr(server, "_open_conn_unlocked", lambda: fresh)

        server._recover_shared_conn(_abort_error(), "unit")

        assert conn.sql == ["ROLLBACK", "SELECT 1"]
        assert conn.closed is True
        assert server._conn is None
        assert server._get_conn() is fresh

    def test_recovery_log_line(self, caplog: pytest.LogCaptureFixture) -> None:
        good_conn = _RecoveryConn()
        good_server = self._server_with_conn(good_conn)

        def fail_rollback(sql: str) -> None:
            if sql == "ROLLBACK":
                raise duckdb.TransactionException("connection has been invalidated")

        def no_active_txn(sql: str) -> None:
            if sql == "ROLLBACK":
                raise duckdb.TransactionException("no transaction is active")

        bad_conn = _RecoveryConn(on_execute=fail_rollback)
        bad_server = self._server_with_conn(bad_conn)
        no_active_server = self._server_with_conn(_RecoveryConn(on_execute=no_active_txn))

        with caplog.at_level("WARNING", logger="recall.rpc_server"):
            good_server._recover_shared_conn(ValueError("first"), "happy")
            no_active_server._recover_shared_conn(ValueError("benign"), "empty")
            bad_server._recover_shared_conn(ValueError("second"), "closed")

        messages = [record.message for record in caplog.records]
        assert len(messages) == 3
        assert 'label="happy"' in messages[0]
        assert 'origin="ValueError: first"' in messages[0]
        assert 'path="rollback_verified"' in messages[0]
        assert 'label="empty"' in messages[1]
        assert 'origin="ValueError: benign"' in messages[1]
        assert 'path="no_active_txn"' in messages[1]
        assert 'label="closed"' in messages[2]
        assert 'origin="ValueError: second"' in messages[2]
        assert 'path="close_clear"' in messages[2]

    def test_recovery_acquires_write_gate_rollback_branch(self) -> None:
        conn = _RecoveryConn()
        server = self._server_with_conn(conn)
        events: list[str] = []

        gate = cast(Any, server._conn_lifecycle_gate)
        gate.acquire_write = Mock(side_effect=lambda: events.append("acquire"))
        gate.release_write = Mock(side_effect=lambda: events.append("release"))
        conn.on_execute = lambda sql: events.append(sql) or None

        server._recover_shared_conn(_abort_error(), "unit")

        assert events == ["acquire", "ROLLBACK", "SELECT 1", "release"]

    def test_recovery_acquires_write_gate_no_active_txn_branch(self) -> None:
        conn = _RecoveryConn()
        server = self._server_with_conn(conn)
        events: list[str] = []

        gate = cast(Any, server._conn_lifecycle_gate)
        gate.acquire_write = Mock(side_effect=lambda: events.append("acquire"))
        gate.release_write = Mock(side_effect=lambda: events.append("release"))

        def on_execute(sql: str) -> None:
            events.append(sql)
            if sql == "ROLLBACK":
                raise duckdb.TransactionException("no transaction is active")

        conn.on_execute = on_execute

        server._recover_shared_conn(_abort_error(), "unit")

        assert events == ["acquire", "ROLLBACK", "SELECT 1", "release"]

    def test_recovery_acquires_write_gate_close_clear_branch(self) -> None:
        conn = _RecoveryConn()
        server = self._server_with_conn(conn)
        events: list[str] = []

        gate = cast(Any, server._conn_lifecycle_gate)
        gate.acquire_write = Mock(side_effect=lambda: events.append("acquire"))
        gate.release_write = Mock(side_effect=lambda: events.append("release"))

        def on_execute(sql: str) -> None:
            events.append(sql)
            if sql == "ROLLBACK":
                raise duckdb.TransactionException("connection invalidated")

        def close() -> None:
            events.append("close")
            conn.closed = True

        conn.on_execute = on_execute
        cast(Any, conn).close = close

        server._recover_shared_conn(_abort_error(), "unit")

        assert events == ["acquire", "ROLLBACK", "close", "release"]
        assert server._conn is None

    def test_concurrent_read_during_recovery(self) -> None:
        conn = _RecoveryConn()
        server = self._server_with_conn(conn)
        server._conn_lifecycle_gate.acquire_write()
        read_started = False

        def read() -> Any:
            nonlocal read_started
            read_started = True
            return server._get_conn()

        async def run() -> Any:
            task = asyncio.create_task(asyncio.to_thread(read))
            await asyncio.sleep(0.05)
            assert read_started is True
            assert task.done() is False
            server._conn_lifecycle_gate.release_write()
            return await asyncio.wait_for(task, timeout=1.0)

        assert _run(run()) is conn

    def test_search_recovers_on_direct_abort_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server = RpcServer()
        calls = 0
        search_calls = 0
        recover = Mock()

        async def fake_run_readonly(fn: Any) -> Any:
            nonlocal calls
            calls += 1
            return fn("cursor")

        def fake_search(**_kwargs: Any) -> list[dict[str, str]]:
            nonlocal search_calls
            search_calls += 1
            if search_calls == 1:
                raise RuntimeError("Current transaction is aborted (please ROLLBACK)")
            return [{"session_id": "s"}]

        search_mod = importlib.import_module("recall.services.search")
        monkeypatch.setattr(server, "_run_readonly", fake_run_readonly)
        monkeypatch.setattr(server, "_recover_shared_conn", recover)
        monkeypatch.setattr(search_mod, "search", fake_search)

        result = _run(server._handle_search({"query": "x", "mode": "keyword"}, None))

        assert result == [{"session_id": "s"}]
        recover.assert_called_once()
        assert calls == 3

    def test_search_reports_missing_fts_without_rebuilding_from_a_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = RpcServer()
        calls = 0

        async def fake_run_readonly(fn: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 1:
                return fn("cursor")
            raise RuntimeError(
                "FTS indexes missing; please run `recall index` to create FTS indexes"
            )

        init_fts_spy = AsyncMock(side_effect=AssertionError("read attempted an FTS rebuild"))
        monkeypatch.setattr(server, "_run_readonly", fake_run_readonly)
        monkeypatch.setattr(server, "_init_fts", init_fts_spy)

        with pytest.raises(RuntimeError, match="FTS indexes missing"):
            _run(server._handle_search({"query": "x", "mode": "keyword"}, None))

        assert calls == 2
        assert init_fts_spy.await_count == 0
        assert server._keyword_dirty

    def test_watch_task_crash_requests_shutdown(self, caplog: pytest.LogCaptureFixture) -> None:
        server = RpcServer()

        async def boom() -> None:
            raise RuntimeError("discovery failed")

        async def run() -> None:
            task = asyncio.create_task(boom())
            task.add_done_callback(lambda done: server._fail_on_watch_task_exit("discovery", done))
            with caplog.at_level("ERROR", logger="recall.rpc_server"):
                await asyncio.wait_for(task, timeout=1.0)

        with pytest.raises(RuntimeError, match="discovery failed"):
            _run(run())

        assert server._shutdown_event.is_set()
        assert "watch discovery task crashed" in caplog.text


class _FakeStreamWriter:
    """Minimal StreamWriter stand-in for _process_request tests."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        return None


@pytest.mark.parametrize("confirmed", [None, False, True], ids=["omitted", "false", "true"])
@pytest.mark.parametrize(
    ("invalid_params", "message", "before_confirmation"),
    [
        ({"source": "bananas"}, "unsupported source: bananas", True),
        ({"since": "bananas"}, "invalid since value: bananas", True),
        ({"root": "\0"}, "embedded null byte", True),
        (
            {"workers": "bananas"},
            "workers must be a positive integer or 'auto'",
            False,
        ),
        ({"workers": 0}, "workers must be positive", False),
        (
            {"context": "bananas"},
            "invalid context mode: bananas. Expected one of: "
            "llm-codex, llm-local, llm-remote, off, template",
            False,
        ),
        (
            {"project": "repo"},
            "recreate cannot be combined with recompute_context or project",
            False,
        ),
        (
            {"recompute_context": True},
            "recreate cannot be combined with recompute_context or project",
            False,
        ),
        (
            {"recompute_context": True, "only_mode": "bananas"},
            "recreate cannot be combined with recompute_context or project",
            False,
        ),
        (
            {"source": "bananas", "since": "bananas"},
            "unsupported source: bananas",
            True,
        ),
        (
            {"since": "bananas", "root": "\0"},
            "invalid since value: bananas",
            True,
        ),
        ({"root": "\0", "workers": "bananas"}, "embedded null byte", True),
        (
            {"workers": "bananas", "context": "bananas"},
            "workers must be a positive integer or 'auto'",
            False,
        ),
        (
            {"context": "bananas", "project": "repo"},
            "invalid context mode: bananas. Expected one of: "
            "llm-codex, llm-local, llm-remote, off, template",
            False,
        ),
    ],
    ids=[
        "source",
        "since",
        "root",
        "workers",
        "workers-zero",
        "context",
        "project",
        "recompute",
        "only-mode",
        "source-and-since",
        "since-and-root",
        "root-and-workers",
        "workers-and-context",
        "context-and-project",
    ],
)
def test_index_rpc_validation_confirmation_precedence(
    server: RpcServer,
    confirmed: bool | None,
    invalid_params: dict[str, object],
    message: str,
    before_confirmation: bool,
) -> None:
    params = {"recreate": True, **invalid_params}
    if confirmed is not None:
        params["confirmed"] = confirmed
    writer = _FakeStreamWriter()
    request = json.dumps(
        {"jsonrpc": "2.0", "method": "recall.index", "params": params, "id": 1}
    ).encode()

    _run(
        server._process_request(
            request, cast(asyncio.StreamWriter, writer), cast(ClientConnection, None)
        )
    )

    expected_error = {"code": INVALID_PARAMS, "message": message}
    if not before_confirmation and not confirmed:
        expected_error = {
            "code": APP_CONFIRMATION_REQUIRED,
            "message": "recreate requires confirmed: true",
        }
    assert json.loads(b"".join(writer.chunks)) == {
        "jsonrpc": "2.0",
        "id": 1,
        "error": expected_error,
    }
    assert not server._config.db_path.exists()
    assert not server._shutdown_event.is_set()


@pytest.mark.parametrize("query", [7, 1.5, None, ["term"], {"term": "x"}, 0, False, [], {}])
def test_dispatch_rejects_non_string_search_query(server: RpcServer, query: object) -> None:
    writer = _FakeStreamWriter()
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "recall.search",
            "params": {"query": query, "mode": "keyword", "limit": 1},
            "id": 1,
        }
    ).encode()
    _run(
        server._process_request(
            request, cast(asyncio.StreamWriter, writer), cast(ClientConnection, None)
        )
    )
    response = json.loads(b"".join(writer.chunks))
    assert response["error"]["code"] == INVALID_PARAMS
    assert "query" in response["error"]["message"]
    assert not server._shutdown_event.is_set()


@pytest.mark.parametrize(
    ("method", "params", "parameter"),
    [
        ("recall.list", {"since": ["yesterday"]}, "since"),
        ("recall.list", {"limit": []}, "limit"),
        ("recall.list", {"limit": True}, "limit"),
        ("recall.index", {"full": "false"}, "full"),
        ("recall.show", {"session_id": ["session"]}, "session_id"),
        ("recall.stats_bash", {"suggest": "false"}, "suggest"),
        ("recall.stats_usage", {"since": 7}, "since"),
        ("recall.stats_skills", {"source": ["codex", 7]}, "source"),
        ("recall.daemon_status", {"limit": {}}, "limit"),
        ("recall.daemon_run", {"once": "false"}, "once"),
        ("recall._set_installed_scheduler", {"scheduler": []}, "scheduler"),
        ("recall.migrate_storage", {"dry_run": "false"}, "dry_run"),
        ("recall.check_indexes", {"sample": True}, "sample"),
        ("recall.live", {"all": "false"}, "all"),
        ("recall.live_mark", {"session": "s", "pid": True}, "pid"),
        ("recall.show_follow", {"session_id": "s", "timeout": True}, "timeout"),
    ],
)
def test_dispatch_rejects_wrong_parameter_type(
    server: RpcServer, method: str, params: dict[str, object], parameter: str
) -> None:
    writer = _FakeStreamWriter()
    request = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1}).encode()
    _run(
        server._process_request(
            request, cast(asyncio.StreamWriter, writer), cast(ClientConnection, None)
        )
    )
    response = json.loads(b"".join(writer.chunks))
    assert response["error"]["code"] == INVALID_PARAMS
    assert parameter in response["error"]["message"]
    assert not server._shutdown_event.is_set()


class _ConnectionWriter(_FakeStreamWriter):
    def __init__(self) -> None:
        super().__init__()
        self.closing = False

    def is_closing(self) -> bool:
        return self.closing

    def close(self) -> None:
        self.closing = True

    async def wait_closed(self) -> None:
        return None


class TestBoundedReadRequests:
    @pytest.mark.parametrize("ending", ["eof", "reset", "cancel"])
    def test_a_quiet_stream_releases_its_subscription(self, server: RpcServer, ending: str) -> None:
        """Declared streams share the connection-owned departure event."""

        async def scenario() -> None:
            reader = _PeerStreamReader()
            writer = _ConnectionWriter()
            entered = asyncio.Event()

            async def stream(params: dict, client: object) -> None:
                with server._session_indexed.subscribe("quiet-session"):
                    entered.set()
                    await asyncio.Event().wait()

            server._methods["recall.show_follow"] = stream
            request = b'{"jsonrpc":"2.0","method":"recall.show_follow","id":1}'
            task = asyncio.create_task(
                server._process_request(
                    request,
                    cast(asyncio.StreamWriter, writer),
                    cast(ClientConnection, None),
                    reader=reader,
                )
            )
            await asyncio.wait_for(entered.wait(), 1)
            assert server._session_indexed.total_subscriptions == 1
            if ending == "eof":
                reader.feed_eof()
            elif ending == "reset":
                reader.set_exception(ConnectionResetError("peer reset"))
            elif ending == "cancel":
                task.cancel()
            if ending == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
            else:
                await asyncio.wait_for(task, 2)
            assert server._session_indexed.total_subscriptions == 0
            assert writer.chunks == []

        _run(scenario())

    def test_disconnect_cancels_an_ordinary_read_handler(self, server: RpcServer) -> None:
        async def scenario() -> None:
            reader = _PeerStreamReader()
            writer = _ConnectionWriter()
            entered = asyncio.Event()
            cancelled = asyncio.Event()

            async def held_read(params: dict, client: object) -> None:
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            server._methods["recall.show"] = held_read
            request = b'{"jsonrpc":"2.0","method":"recall.show","params":{},"id":1}'
            task = asyncio.create_task(
                server._process_request(
                    request,
                    cast(asyncio.StreamWriter, writer),
                    cast(ClientConnection, None),
                    reader=reader,
                )
            )
            await asyncio.wait_for(entered.wait(), 1)
            reader.set_exception(ConnectionResetError("peer reset"))
            await asyncio.wait_for(task, 1)
            assert cancelled.is_set()
            assert writer.chunks == []

        _run(scenario())

    def test_connected_stream_sends_terminal_result_after_shutdown(self, server: RpcServer) -> None:
        async def scenario() -> None:
            reader = _PeerStreamReader()
            writer = _ConnectionWriter()
            entered = asyncio.Event()

            async def stream(params: dict, client: object) -> dict:
                with server._session_indexed.subscribe("quiet-session"):
                    entered.set()
                    await server._shutdown_event.wait()
                    # Completion can require asynchronous cleanup after shutdown.
                    await asyncio.sleep(0.01)
                return {"event": "closed", "reason": "daemon_stopped"}

            server._methods["recall.show_follow"] = stream
            task = asyncio.create_task(
                server._process_request(
                    b'{"jsonrpc":"2.0","method":"recall.show_follow","id":1}',
                    cast(asyncio.StreamWriter, writer),
                    cast(ClientConnection, None),
                    reader=reader,
                )
            )
            await asyncio.wait_for(entered.wait(), 1)
            server._shutdown_event.set()
            await asyncio.wait_for(task, 2)
            assert server._session_indexed.total_subscriptions == 0
            assert json.loads(writer.chunks[0])["result"] == {
                "event": "closed",
                "reason": "daemon_stopped",
            }

        _run(scenario())

    def test_ordinary_read_timeout_is_reported_after_handler_cleanup(
        self, server: RpcServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rpc_server = importlib.import_module("recall.services.rpc_server")
        monkeypatch.setattr(rpc_server, "_READ_REQUEST_TIMEOUT", 0.05)

        async def scenario() -> None:
            reader = _PeerStreamReader()
            writer = _ConnectionWriter()
            cancelled = asyncio.Event()

            async def held_read(params: dict, client: object) -> None:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            server._methods["recall.list"] = held_read
            request = b'{"jsonrpc":"2.0","method":"recall.list","params":{},"id":2}'
            await asyncio.wait_for(
                server._process_request(
                    request,
                    cast(asyncio.StreamWriter, writer),
                    cast(ClientConnection, None),
                    reader=reader,
                ),
                1,
            )
            assert cancelled.is_set()
            response = json.loads(writer.chunks[0])
            assert response["error"]["code"] == INTERNAL_ERROR
            assert response["error"]["message"] == "recall.list timed out after 0.05 seconds"

        _run(scenario())

    def test_a_write_keeps_running_while_its_client_is_present(self, server: RpcServer) -> None:
        """REQ-RPC-018: a long write owns no deadline — only departure ends it.

        The read deadline must not leak onto `recall.index`: an index that
        outlives `_READ_REQUEST_TIMEOUT` with its client still attached is
        ordinary, and killing it would be the worse bug.
        """

        async def scenario() -> None:
            rpc_server = importlib.import_module("recall.services.rpc_server")
            reader = _PeerStreamReader()
            writer = _ConnectionWriter()
            entered = asyncio.Event()
            release = asyncio.Event()
            cancelled = False

            async def held_write(params: dict, client: object) -> str:
                nonlocal cancelled
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled = True
                    raise
                return "committed"

            server._methods["recall.index"] = held_write
            request = b'{"jsonrpc":"2.0","method":"recall.index","params":{},"id":3}'
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(rpc_server, "_READ_REQUEST_TIMEOUT", 0.05)
                task = asyncio.create_task(
                    server._process_request(
                        request,
                        cast(asyncio.StreamWriter, writer),
                        cast(ClientConnection, None),
                        reader=reader,
                    )
                )
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.sleep(0.2)
                assert not task.done()
                assert not cancelled
                release.set()
                await asyncio.wait_for(task, 1)
            assert json.loads(writer.chunks[0])["result"] == "committed"

        _run(scenario())

    @pytest.mark.parametrize("ending", ["eof", "shutdown"])
    def test_disconnect_ends_an_in_flight_write(self, server: RpcServer, ending: str) -> None:
        """REQ-RPC-018: the write ends when the client that asked for it is gone.

        This inverts the earlier contract, in which a write survived its
        client. Nothing else on that connection observes EOF while the handler
        runs, so an abandoned `recall index` held its index turn for the life
        of the daemon and every later index queued behind it. The work
        is not lost: indexing commits per session and reconciliation owns the
        remainder.
        """

        async def scenario() -> None:
            reader = _PeerStreamReader()
            writer = _ConnectionWriter()
            entered = asyncio.Event()
            cancelled = asyncio.Event()

            async def held_write(params: dict, client: object) -> str:
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                return "committed"

            server._methods["recall.index"] = held_write
            request = b'{"jsonrpc":"2.0","method":"recall.index","params":{},"id":3}'
            task = asyncio.create_task(
                server._process_request(
                    request,
                    cast(asyncio.StreamWriter, writer),
                    cast(ClientConnection, None),
                    reader=reader,
                )
            )
            await asyncio.wait_for(entered.wait(), 1)
            if ending == "eof":
                reader.feed_eof()
            else:
                server._shutdown_event.set()
            await asyncio.wait_for(task, 1)
            assert cancelled.is_set()
            # Nobody is left to answer, and the departure is not an error.
            assert writer.chunks == []

        _run(scenario())

    def test_connection_limit_refuses_work_without_growing_a_waiter_queue(
        self, server: RpcServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rpc_server = importlib.import_module("recall.services.rpc_server")
        monkeypatch.setattr(rpc_server, "_MAX_CONNECTIONS", 1)
        server._active_connections = 1
        writer = _ConnectionWriter()
        reads: list[bool] = []

        class NeverRead:
            async def readline(self) -> bytes:
                reads.append(True)
                return b""

        _run(
            server._handle_connection(
                cast(_PeerStreamReader, NeverRead()), cast(asyncio.StreamWriter, writer)
            )
        )

        assert writer.closing
        assert reads == []
        assert server._active_connections == 1

    def test_pipelined_frames_remain_owned_by_the_connection_loop(self, server: RpcServer) -> None:
        async def echo(params: dict, client: object) -> str:
            await asyncio.sleep(0.01)
            return str(params["value"])

        server._methods["recall.list"] = echo

        async def scenario() -> None:
            reader = _PeerStreamReader()
            reader.feed_data(
                b'{"jsonrpc":"2.0","method":"recall.list","params":{"value":"first"},"id":1}\n'
                b'{"jsonrpc":"2.0","method":"recall.list","params":{"value":"second"},"id":2}\n'
            )
            reader.feed_eof()
            writer = _ConnectionWriter()
            await asyncio.wait_for(
                server._handle_connection(reader, cast(asyncio.StreamWriter, writer)), 1
            )
            responses = [json.loads(chunk) for chunk in writer.chunks]
            assert [(item["id"], item["result"]) for item in responses] == [
                (1, "first"),
                (2, "second"),
            ]

        _run(scenario())

    @_requires_duckdb_lock
    def test_buffered_live_mark_answers_after_half_close(self, server: RpcServer) -> None:
        async def scenario() -> None:
            entered = asyncio.Event()
            live_mark = server._methods["recall.live_mark"]

            async def mark(params: dict[str, Any], client: ClientConnection | None) -> Any:
                entered.set()
                return await live_mark(params, client)

            server._methods["recall.live_mark"] = mark
            reader = _PeerStreamReader()
            reader.feed_data(
                b'{"jsonrpc":"2.0","method":"recall.live_mark",'
                b'"params":{"session":"buffered-session","pid":123},"id":1}\n'
            )
            reader.feed_eof()
            writer = _ConnectionWriter()
            try:
                async with server._write_lock:
                    task = asyncio.create_task(
                        server._handle_connection(reader, cast(asyncio.StreamWriter, writer))
                    )
                    await asyncio.wait_for(entered.wait(), 1)
                await asyncio.wait_for(task, 2)
                responses = [json.loads(chunk) for chunk in writer.chunks]
                assert len(responses) == 1
                assert responses[0]["id"] == 1
                assert responses[0]["result"]["marked"] is True
                assert responses[0]["result"]["source_session_id"] == "buffered-session"
                assert server._get_conn().execute(
                    "SELECT source_session_id, pid FROM live_marks"
                ).fetchall() == [("buffered-session", 123)]
            finally:
                await server.stop()

        _run(scenario())


class TestFatalDbInvalidation:
    """A fatal DuckDB invalidation cannot be healed in-process (reconnecting
    to the same path returns the same cached, still-invalidated instance), so
    the daemon must fail loud and let the scheduler restart it instead of
    serving INTERNAL_ERROR forever — the observed wedged-daemon state."""

    def test_invalidation_message_without_duckdb_type_sets_shutdown(
        self, server: RpcServer
    ) -> None:
        async def _wrapped_fatal_handler(params: dict, client: object) -> None:
            raise RuntimeError(
                "query failed: database has been invalidated because of a previous fatal error"
            )

        server._methods["recall.fatal_test"] = _wrapped_fatal_handler
        writer = _FakeStreamWriter()
        request = b'{"jsonrpc":"2.0","method":"recall.fatal_test","params":{},"id":2}'

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                server._process_request(
                    request, cast(asyncio.StreamWriter, writer), cast(ClientConnection, None)
                )
            )
        finally:
            loop.close()

        assert server._shutdown_event.is_set()

    def test_ordinary_handler_error_does_not_stop_daemon(self, server: RpcServer) -> None:
        async def _boom_handler(params: dict, client: object) -> None:
            raise RuntimeError("boom")

        server._methods["recall.fatal_test"] = _boom_handler
        writer = _FakeStreamWriter()
        request = b'{"jsonrpc":"2.0","method":"recall.fatal_test","params":{},"id":3}'

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                server._process_request(
                    request, cast(asyncio.StreamWriter, writer), cast(ClientConnection, None)
                )
            )
        finally:
            loop.close()

        assert writer.chunks
        assert not server._shutdown_event.is_set()


class _DisconnectedStreamWriter:
    """Writer whose client has gone away: every write raises."""

    def write(self, data: bytes) -> None:
        raise ConnectionResetError(54, "Connection reset by peer")

    async def drain(self) -> None:
        raise ConnectionResetError(54, "Connection reset by peer")


def test_fatal_invalidation_sets_shutdown_even_when_client_gone(
    server: RpcServer,
) -> None:
    """REQ-RESIL-011: the timed-out client that exposes a wedged DB is exactly
    the client likely to have disconnected; a failed error-send must not skip
    the daemon shutdown."""
    import duckdb

    async def _fatal_handler(params: dict, client: object) -> None:
        raise duckdb.FatalException(
            "database has been invalidated because of a previous fatal error"
        )

    server._methods["recall.fatal_test"] = _fatal_handler
    writer = _DisconnectedStreamWriter()
    request = b'{"jsonrpc":"2.0","method":"recall.fatal_test","params":{},"id":9}'

    loop = asyncio.new_event_loop()
    try:
        with suppress(ConnectionResetError):
            loop.run_until_complete(
                server._process_request(
                    request, cast(asyncio.StreamWriter, writer), cast(ClientConnection, None)
                )
            )
    finally:
        loop.close()

    assert server._shutdown_event.is_set()


def test_stop_waits_for_in_flight_shared_conn_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RESIL-021: `stop()` closes the shared connection only after the
    write lock is free, so executor work still using it never observes
    `Connection already closed`."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    server = RpcServer(config=AppConfig.load())
    conn = Mock()
    server._conn = conn

    async def run() -> None:
        async with server._write_lock:
            stop_task = asyncio.create_task(server.stop())
            await asyncio.sleep(0.05)
            assert not stop_task.done()
            assert conn.close.call_count == 0
        await asyncio.wait_for(stop_task, timeout=1.0)
        assert conn.close.call_count == 1
        assert server._conn is None

    _run(run())


def test_simultaneous_first_reads_share_one_initialized_database(tmp_path, monkeypatch) -> None:
    """Concurrent first reads must all observe a complete schema, including cold startup."""
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "config.toml"))
    from recall.core.config import AppConfig
    from recall.db.schema import SCHEMA_VERSION

    server = RpcServer(config=AppConfig.load())
    barrier = threading.Barrier(4, timeout=3)

    def read() -> object:
        barrier.wait()
        return server._run_with_read_cursor_sync(
            lambda conn: conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        )

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(read) for _ in range(4)]
            results = [future.result(timeout=5) for future in futures]
        assert results == [(SCHEMA_VERSION,)] * 4
    finally:
        _run(server.stop())


def test_embed_batch_defers_before_snapshot_when_load_gate_fails(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A load deferral must not pay for the embed snapshot first."""
    from recall.services.embed_phase import EmbedPhaseState
    from recall.services.system_state import EmbedPreconditionResult

    config = server._config
    monkeypatch.setattr("recall.services.coordinator.is_paused", lambda _config: False)
    monkeypatch.setattr(
        "recall.services.system_state.check_load",
        lambda _threshold: EmbedPreconditionResult(ok=False, reason="test load"),
    )

    snapshot_calls: list[str] = []

    def forbidden_snapshot(*_args: Any, **_kwargs: Any) -> None:
        snapshot_calls.append("snapshot")
        raise AssertionError("embed snapshot should not run while the load gate defers")

    def forbidden_generate(*_args: Any, **_kwargs: Any) -> None:
        snapshot_calls.append("generate")
        raise AssertionError("embed generation should not run while the load gate defers")

    monkeypatch.setattr("recall.services.embed_phase.prepare_embed_cycle", forbidden_snapshot)
    monkeypatch.setattr(
        "recall.services.embed_phase.generate_prepared_embed_cycle", forbidden_generate
    )

    state = EmbedPhaseState(_config=config)
    result = asyncio.run(server._run_owned_embed_batch(config, state))

    assert result == -1
    assert state.deferred_reason == "test load"
    assert snapshot_calls == []


def test_writer_call_logs_wait_and_operation_durations(
    server: RpcServer, caplog: pytest.LogCaptureFixture
) -> None:
    """A successful writer call reports lock wait and operation time at INFO."""
    with caplog.at_level(logging.INFO):
        asyncio.run(server._writer_call(lambda: None, "test-writer-label"))

    messages = [record.getMessage() for record in caplog.records]
    assert any("test-writer-label" in message and "wait=" in message for message in messages)


@pytest.mark.parametrize("waiting_for", ["permit", "enrichment"])
def test_memory_admission_wait_uses_read_deadline_and_removes_cancelled_waiter(
    server: RpcServer, monkeypatch: pytest.MonkeyPatch, waiting_for: str
) -> None:
    """REQ-RPC-020: waiting expires through the existing RPC error contract."""
    rpc_module = importlib.import_module("recall.services.rpc_server")
    monkeypatch.setattr(rpc_module, "_READ_REQUEST_TIMEOUT", 0.05)

    async def embedding(_operation: object) -> list[float]:
        return [0.25] * 384

    monkeypatch.setattr(server, "_prepare_query_embedding", embedding)

    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def hold_memory() -> None:
            entered.set()
            await release.wait()

        owner = None
        if waiting_for == "permit":
            owner = asyncio.create_task(server._run_duckdb_scan(hold_memory))
            await entered.wait()
        else:
            server._background_enrichment_idle.clear()
        # Database initialization is outside this scenario's short wait budget.
        await server._run_readonly(lambda conn: conn.execute("SELECT 1").fetchone())
        writer = _ConnectionWriter()
        try:
            await asyncio.wait_for(
                server._process_request(
                    b'{"jsonrpc":"2.0","method":"recall.search",'
                    b'"params":{"query":"test","mode":"vector"},"id":19}',
                    cast(asyncio.StreamWriter, writer),
                    cast(ClientConnection, None),
                    reader=_PeerStreamReader(),
                ),
                2,
            )
            response = json.loads(writer.chunks[0])
            assert response["error"]["code"] == INTERNAL_ERROR
            phase = "scan_admission" if waiting_for == "permit" else "enrichment_wait"
            assert response["error"]["message"] == (
                "recall.search timed out after 0.05 seconds "
                f"(phase: {phase}; largest phase: {phase})"
            )
            assert response["error"]["data"]["phase"] == phase
            release.set()
            server._background_enrichment_idle.set()
            if owner is not None:
                await owner
            # The timed-out request owns no worker/cursor and cannot strand the next search.
            assert (
                await asyncio.wait_for(
                    server._handle_search({"query": "test", "mode": "vector"}, None), 2
                )
                == []
            )
        finally:
            release.set()
            server._background_enrichment_idle.set()
            if owner is not None:
                await owner
            await server.stop()

    asyncio.run(scenario())


def test_memory_admission_search_waits_for_startup_sidecar_scan(
    monkeypatch: pytest.MonkeyPatch, socket_dir: Path
) -> None:
    """RPC can answer before startup reconciliation finishes its DuckDB scans."""
    monkeypatch.setenv("RECALL_DATA_DIR", str(socket_dir / "d"))
    server = RpcServer()
    entered = threading.Event()
    release = threading.Event()
    original = run_startup_fts_sidecar_sync

    def held_sync(config: AppConfig) -> FtsSidecarStartupResult:
        entered.set()
        assert release.wait(5)
        return original(config)

    async def embedding(_operation: object) -> list[float]:
        return [0.25] * 384

    monkeypatch.setattr("recall.services.daemon.run_startup_fts_sidecar_sync", held_sync)
    monkeypatch.setattr(server, "_prepare_query_embedding", embedding)

    async def scenario() -> None:
        startup = asyncio.create_task(server.start())
        search = None
        status = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            search = asyncio.create_task(
                server._handle_search({"query": "test", "mode": "vector"}, None)
            )
            assert (await server._handle_stats({}, None))["messages"] == 0
            status = asyncio.create_task(server._handle_daemon_status({}, None))
            assert (await asyncio.wait_for(status, 2))["daemon_pid"] == os.getpid()
            completed, _ = await asyncio.wait({search}, timeout=0.05)
            assert not completed, "a vector scan must wait for startup reconciliation"
            release.set()
            assert await asyncio.wait_for(search, 2) == []
        finally:
            release.set()
            if search is not None:
                await asyncio.gather(search, return_exceptions=True)
            if status is not None:
                await asyncio.gather(status, return_exceptions=True)
            await server.stop()
            await asyncio.wait_for(startup, 2)

    asyncio.run(scenario())


def test_duckdb_fts_refresh_repairs_missing_search_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from recall.db.schema import ensure_schema

    config = replace(
        _tmp_config(tmp_path, monkeypatch, compaction=CompactionConfig(auto_trigger=False)),
        fts=FtsConfig(backend="duckdb"),
    )
    server = RpcServer(config)
    server._executor = ThreadPoolExecutor(max_workers=1)
    conn = server._get_conn()
    ensure_schema(conn)
    conn.execute(
        "INSERT INTO message_state (message_id, role, content) "
        "VALUES ('keyword-test', 'user', 'kumquat')"
    )

    async def scenario() -> None:
        await server._init_fts(reason="startup")
        conn.execute("DROP SCHEMA fts_main_message_state CASCADE")
        conn.execute(
            "UPDATE message_state SET content = 'pineapple' WHERE message_id = 'keyword-test'"
        )
        server._keyword_dirty = True
        await server._init_fts(reason="keyword dirty")
        assert conn.execute(
            "SELECT message_id FROM message_state "
            "WHERE fts_main_message_state.match_bm25(message_id, 'pineapple') IS NOT NULL"
        ).fetchall() == [("keyword-test",)]
        assert (
            conn.execute(
                "SELECT message_id FROM message_state "
                "WHERE fts_main_message_state.match_bm25(message_id, 'kumquat') IS NOT NULL"
            ).fetchall()
            == []
        )
        assert server._keyword_dirty is False

    try:
        with caplog.at_level(logging.DEBUG, logger="recall.rpc_server"):
            asyncio.run(scenario())
        refreshed = [r for r in caplog.records if "FTS indexes refreshed" in r.getMessage()]
        assert len(refreshed) == 2
        assert "keyword dirty" in refreshed[-1].getMessage()
        assert all(r.levelno == logging.DEBUG for r in refreshed)
        assert not any("on startup" in r.getMessage() for r in caplog.records)
    finally:
        asyncio.run(server.stop())
