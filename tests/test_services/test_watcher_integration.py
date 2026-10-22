from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import recall.services.watcher as watcher_module
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.types import DaemonMode, Source
from recall.db import connect
from recall.services.rpc_server import start_server_blocking
from recall.services.watcher import WatcherLiveSnapshot, get_live_snapshot, reset_live_snapshot
from watchdog.observers.polling import PollingObserver

from conftest import _can_acquire_duckdb_lock


@pytest.fixture(autouse=True)
def _isolate_live_snapshot() -> Iterator[None]:
    reset_live_snapshot()
    yield
    reset_live_snapshot()


def _claude_fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "claude_code" / "session1.jsonl"


def _app_config(data_dir: Path, tmp_path: Path) -> AppConfig:
    data_dir.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / "config.toml",
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(
            source=Source.CLAUDE_CODE,
            mode=DaemonMode.WATCH,
            embed=False,
            debounce=1,
            fts_debounce=1,
            live_idle_threshold=60,
            live_discovery_interval=1,
        ),
        cli=CliConfig(),
    )


def _wait_for_live_snapshot(
    *,
    server_error: list[BaseException],
    timeout_seconds: float,
) -> WatcherLiveSnapshot:
    deadline = time.monotonic() + timeout_seconds
    last_snapshot = get_live_snapshot()
    while time.monotonic() < deadline:
        if server_error:
            raise AssertionError("RPC server exited unexpectedly") from server_error[0]
        snapshot = get_live_snapshot()
        last_snapshot = snapshot
        if (
            snapshot.live_session_count >= 1
            and snapshot.watcher_subscription_count >= 1
            and snapshot.discovery_last_run_at is not None
        ):
            return snapshot
        time.sleep(0.1)
    pytest.fail(f"live snapshot never populated: {last_snapshot}; server_error={server_error!r}")


def _wait_for_message_count(
    config: AppConfig,
    *,
    expected_minimum: int,
    server_error: list[BaseException],
    timeout_seconds: float,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    last_count = -1
    while time.monotonic() < deadline:
        if server_error:
            raise AssertionError("RPC server exited unexpectedly") from server_error[0]
        conn = connect(config)
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(message_count), 0) FROM session_state"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        last_count = int(row[0])
        if last_count >= expected_minimum:
            return last_count
        time.sleep(0.1)
    pytest.fail(f"message_count never reached {expected_minimum}; last_count={last_count}")


def test_rpc_watch_mode_populates_live_snapshot_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, socket_dir: Path
) -> None:
    if not _can_acquire_duckdb_lock():
        pytest.skip("daemon-lock integration requires DuckDB lock acquisition capability")

    monkeypatch.setattr(watcher_module.watchdog.observers, "Observer", PollingObserver)
    claude_root = tmp_path / ".claude" / "projects" / "proj1"
    claude_root.mkdir(parents=True, exist_ok=True)
    session_path = claude_root / "session1.jsonl"
    shutil.copy(_claude_fixture_path(), session_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    runtime_dir = Path(".")
    config = _app_config(runtime_dir / "data", tmp_path)
    shutdown = threading.Event()
    server_error: list[BaseException] = []

    def _run() -> None:
        try:
            start_server_blocking(
                config=config,
                idle_timeout=None,
                watch=True,
                shutdown_event=shutdown,
            )
        except BaseException as err:  # pragma: no cover - surfaced by the test
            server_error.append(err)

    server_thread = threading.Thread(target=_run, name="rpc-watch-test", daemon=True)
    server_thread.start()

    try:
        snapshot = _wait_for_live_snapshot(
            server_error=server_error,
            timeout_seconds=5.0,
        )
        assert snapshot.live_session_count >= 1
        assert snapshot.watcher_subscription_count >= 1
        assert snapshot.discovery_last_run_at is not None

        indexed_count = _wait_for_message_count(
            config,
            expected_minimum=4,
            server_error=server_error,
            timeout_seconds=5.0,
        )
        assert indexed_count >= 4

        appended_line = json.dumps(
            {
                "type": "message",
                "timestamp": "2024-01-15T10:04:00Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Appended from integration test"}],
                },
            }
        )
        with session_path.open("a", encoding="utf-8") as handle:
            handle.write(appended_line + "\n")

        updated_count = _wait_for_message_count(
            config,
            expected_minimum=5,
            server_error=server_error,
            timeout_seconds=5.0,
        )
        assert updated_count >= 5
    finally:
        shutdown.set()
        server_thread.join(timeout=10.0)

    assert not server_thread.is_alive(), "RPC server thread did not exit cleanly"
    assert not server_error, f"RPC server raised: {server_error[0]}"

    shutdown_snapshot = get_live_snapshot()
    assert shutdown_snapshot.live_session_count == 0
    assert shutdown_snapshot.discovery_last_run_at is None


def _wait_for_indexed_path(
    config: AppConfig,
    path: Path,
    *,
    server_error: list[BaseException],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server_error:
            raise AssertionError("RPC server exited unexpectedly") from server_error[0]
        conn = connect(config)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE source_path = ?", [str(path.resolve())]
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        if row[0] == 1:
            return
        time.sleep(0.1)
    pytest.fail(f"{path} was not indexed within {timeout_seconds} s")


def test_new_and_renamed_transcripts_are_indexed_before_any_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, socket_dir: Path
) -> None:
    """A created or renamed transcript is indexed from its own filesystem event.

    The discovery sweep runs once an hour here and the inventory walk every
    30 s, so a path indexed within 10 s was found by neither (REQ-DAEMON-078).
    Uses the platform's own observer.
    """
    if not _can_acquire_duckdb_lock():
        pytest.skip("daemon-lock integration requires DuckDB lock acquisition capability")

    projects = tmp_path / ".claude" / "projects"
    existing = projects / "proj1" / "session1.jsonl"
    existing.parent.mkdir(parents=True)
    shutil.copy(_claude_fixture_path(), existing)
    monkeypatch.setenv("HOME", str(tmp_path))

    runtime_dir = Path(".")
    config = _app_config(runtime_dir / "data", tmp_path)
    config = replace(
        config, daemon=replace(config.daemon, interval=3600, live_discovery_interval=3600)
    )
    shutdown = threading.Event()
    server_error: list[BaseException] = []

    def _run() -> None:
        try:
            start_server_blocking(
                config=config,
                idle_timeout=None,
                watch=True,
                shutdown_event=shutdown,
            )
        except BaseException as err:  # pragma: no cover - surfaced by the test
            server_error.append(err)

    server_thread = threading.Thread(target=_run, name="rpc-watch-test", daemon=True)
    server_thread.start()

    try:
        deadline = time.monotonic() + 10.0
        while get_live_snapshot().runtime_started_at is None:
            assert not server_error, f"RPC server raised: {server_error[0]}"
            assert time.monotonic() < deadline, "watch runtime never started"
            time.sleep(0.1)
        # The startup walk indexes what already exists; everything after it
        # has no sweep to wait for inside the deadlines below.
        _wait_for_indexed_path(config, existing, server_error=server_error, timeout_seconds=10.0)

        created = projects / "proj2" / "session2.jsonl"
        created.parent.mkdir()
        shutil.copy(_claude_fixture_path(), created)
        _wait_for_indexed_path(config, created, server_error=server_error, timeout_seconds=10.0)

        renamed = existing.with_name("session3.jsonl")
        existing.rename(renamed)
        _wait_for_indexed_path(config, renamed, server_error=server_error, timeout_seconds=10.0)
    finally:
        shutdown.set()
        server_thread.join(timeout=10.0)

    assert not server_thread.is_alive(), "RPC server thread did not exit cleanly"
    assert not server_error, f"RPC server raised: {server_error[0]}"
