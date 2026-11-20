"""A daemon that owns the database publishes its repair, and removes it when the repair ends."""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import duckdb
import pytest
from recall.core.repair_progress import PROGRESS_FILENAME, PROGRESS_TEMP_FILENAME, read_progress
from recall.db import ownership
from recall.db.connection import connect
from recall.db.ownership import DatabaseOwnershipError
from recall.db.repair import load_repair_row
from recall.services import repair_progress, repair_protocol
from recall.services.daemon import offline_daemon_status
from recall.services.self_repair import (
    DaemonStartupRefused,
    read_refusal_marker,
    remember_fatal_failure,
)

from tests.test_services.test_rpc_owned_boundary import server_at

WAIT_SECONDS = 30


def _wait_for_phase(data_dir: Path, phase: str) -> None:
    """The writer thread publishes a phase change asynchronously; wait for the file to show it."""
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        record = read_progress(data_dir)
        if record is not None and record.phase == phase:
            return
        time.sleep(0.01)
    raise AssertionError(f"the progress file never showed phase {phase}")


def _fatal(data_dir: Path) -> None:
    remember_fatal_failure(
        data_dir,
        duckdb.FatalException("Failed to delete all rows from index in the original run"),
        site="source_commit",
    )


def _seed(server) -> None:
    with connect(server._config) as conn:
        conn.execute("CREATE TABLE witness(k INT, value TEXT)")
        conn.execute("INSERT INTO witness VALUES (1, 'same'), (1, 'same')")
        conn.execute("CREATE INDEX witness_compound ON witness(k, value)")


class _HeldCapture:
    """Stops the repair inside its reference capture until the test releases it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reached = threading.Event()
        self.release = threading.Event()
        original = repair_protocol.capture_oracle

        def capture(*args, **kwargs):
            self.reached.set()
            assert self.release.wait(WAIT_SECONDS), "the test never released the capture"
            return original(*args, **kwargs)

        monkeypatch.setattr(repair_protocol, "capture_oracle", capture)


class _Daemon(threading.Thread):
    """`RpcServer._owned` around `work`, on its own event loop, as a daemon process owns it."""

    def __init__(self, server, work: Callable[[], Awaitable[object]]) -> None:
        super().__init__(name="daemon-under-test")
        self.errors: list[BaseException] = []
        self._server = server
        self._work = work

    def run(self) -> None:
        async def body() -> None:
            await self._work()

        try:
            asyncio.run(self._server._owned(body))
        except BaseException as error:  # handed to the test thread through `errors`
            self.errors.append(error)


def _run_owned(server, work: Callable[[], Awaitable[object]]) -> _Daemon:
    daemon = _Daemon(server, work)
    daemon.start()
    return daemon


def test_status_answers_within_two_seconds_while_the_capture_is_held_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = server_at(tmp_path)
    _seed(server)
    _fatal(tmp_path)
    held = _HeldCapture(monkeypatch)
    thread = _run_owned(server, server._run_startup_self_repair)
    try:
        assert held.reached.wait(WAIT_SECONDS)
        _wait_for_phase(tmp_path, "capturing_reference")
        started = time.monotonic()
        status, _ = offline_daemon_status(server._config, limit=1, cursor=None)
        assert time.monotonic() - started < 2.0
        assert status.daemon_pid is None
        assert status.recovery is not None and status.recovery["state"] == "running"
        assert status.recovery["phase"] == "capturing_reference"
        record = read_progress(tmp_path)
        assert record is not None
        assert status.recovery["recorded_by_pid"] == record.owner_pid
        assert not (tmp_path / "daemon-refused").exists()
    finally:
        held.release.set()
        thread.join(WAIT_SECONDS)
    assert thread.errors == []
    with duckdb.connect(str(server._config.db_path), read_only=True) as fresh:
        assert load_repair_row(fresh).phase == "verified"


def test_the_record_is_gone_when_the_repair_is_verified_and_before_the_daemon_binds(
    tmp_path: Path,
) -> None:
    server = server_at(tmp_path)
    _seed(server)
    _fatal(tmp_path)
    observed: list[bool] = []

    async def repair_then_look() -> None:
        await server._run_startup_self_repair()
        observed.append((tmp_path / PROGRESS_FILENAME).exists())
        observed.append(server._socket_path.exists())

    thread = _run_owned(server, repair_then_look)
    thread.join(WAIT_SECONDS)
    assert thread.errors == []
    assert observed == [False, False]


def test_a_start_with_nothing_to_repair_never_creates_the_record(tmp_path: Path) -> None:
    server = server_at(tmp_path)
    _seed(server)
    seen: list[list[str]] = []

    async def start_only() -> None:
        await server._run_startup_self_repair()
        seen.append(sorted(path.name for path in tmp_path.iterdir() if "progress" in path.name))

    thread = _run_owned(server, start_only)
    thread.join(WAIT_SECONDS)
    assert thread.errors == []
    assert seen == [[]]


def test_a_refused_start_leaves_its_marker_first_and_then_removes_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = server_at(tmp_path)
    _seed(server)
    _fatal(tmp_path)
    marker_at_removal: list[str | None] = []
    remove = repair_progress.remove_progress

    def remove_after_noting_the_marker(data_dir: Path) -> None:
        if (data_dir / PROGRESS_FILENAME).exists():
            marker_at_removal.append(read_refusal_marker(data_dir))
        remove(data_dir)

    monkeypatch.setattr(repair_progress, "remove_progress", remove_after_noting_the_marker)

    def fail(*args, **kwargs):
        raise OSError("injected capture failure")

    monkeypatch.setattr(repair_protocol, "capture_oracle", fail)
    thread = _run_owned(server, server._run_startup_self_repair)
    thread.join(WAIT_SECONDS)
    errors = thread.errors
    assert len(errors) == 1 and isinstance(errors[0], DaemonStartupRefused)
    assert len(marker_at_removal) == 1
    marker = marker_at_removal[0]
    assert marker is not None and "last repair progress: phase capturing the reference" in marker
    assert sorted(path.name for path in tmp_path.iterdir() if "progress" in path.name) == []


def test_a_file_left_by_a_killed_owner_is_removed_by_the_next_owner(tmp_path: Path) -> None:
    server = server_at(tmp_path)
    (tmp_path / PROGRESS_FILENAME).write_text("{}")
    (tmp_path / PROGRESS_TEMP_FILENAME).write_text("{")
    seen: list[list[str]] = []

    async def look() -> None:
        seen.append(sorted(path.name for path in tmp_path.iterdir() if "progress" in path.name))

    thread = _run_owned(server, look)
    thread.join(WAIT_SECONDS)
    assert thread.errors == []
    assert seen == [[]]


def test_a_second_daemon_cannot_publish_beside_the_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = server_at(tmp_path)
    _seed(owner)
    _fatal(tmp_path)
    held = _HeldCapture(monkeypatch)
    thread = _run_owned(owner, owner._run_startup_self_repair)
    try:
        assert held.reached.wait(WAIT_SECONDS)
        recorded = read_progress(tmp_path)
        assert recorded is not None
        monkeypatch.setattr(ownership, "STARTUP_OWNERSHIP_TIMEOUT_SECONDS", 0.2)
        rival = server_at(tmp_path)

        async def never() -> None:
            raise AssertionError("a rival must not reach the repair")

        rival_thread = _run_owned(rival, never)
        rival_thread.join(WAIT_SECONDS)
        errors = rival_thread.errors
        assert len(errors) == 1 and isinstance(errors[0], DatabaseOwnershipError)
        after = read_progress(tmp_path)
        assert after is not None
        assert (after.owner_pid, after.started_at) == (recorded.owner_pid, recorded.started_at)
    finally:
        held.release.set()
        thread.join(WAIT_SECONDS)
