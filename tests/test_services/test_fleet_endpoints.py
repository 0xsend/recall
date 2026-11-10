"""Bounded endpoint work, stable merge ordering and real socket deadlines."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest
from recall.core.fleet import FleetConfig, FleetHost
from recall.services.fleet_endpoints import FleetExecutor
from recall.services.fleet_query import fleet_list, fleet_live, fleet_search
from recall.services.fleet_transport import FleetRemoteResult

from tests.fleet_support import inventory_executor


def test_fleet_bounds_work_and_ties_ignore_completion_order(monkeypatch):
    lock = threading.Lock()
    admitted = threading.Event()
    active = 0
    peak = 0
    names = tuple(f"edge-{i}" for i in range(8))

    def remote(host, argv, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 6:
                admitted.set()
        assert admitted.wait(3), "six admitted workers did not start"
        row = {"id": host.name, "score": 1, "started_at": "2026-11-10T12:00:00Z"}
        with lock:
            active -= 1
        return FleetRemoteResult(True, json.dumps([row]), "", 0)

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", remote)
    executor = inventory_executor(FleetHost(name, name) for name in names)
    result = fleet_list(executor, limit=8)
    assert [row["id"] for row in result.rows] == list(names)
    assert peak == 6
    assert active == 0


@pytest.mark.parametrize("query", [fleet_list, fleet_search])
def test_fleet_ties_ignore_reversed_completion(monkeypatch, query):
    finished = threading.Event()

    def remote(host, argv, **kwargs):
        if host.name == "first":
            assert finished.wait(2)
        else:
            finished.set()
        return FleetRemoteResult(
            True,
            json.dumps([{"id": host.name, "score": 1, "started_at": "2026-11-10T12:00:00Z"}]),
            "",
            0,
        )

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", remote)
    executor = inventory_executor((FleetHost("first", "first"), FleetHost("second", "second")))
    kwargs = {"query": "fixture"} if query is fleet_search else {}
    result = query(executor, limit=1, **kwargs)
    assert [row["id"] for row in result.rows] == ["first"]


@pytest.mark.parametrize("notifications", [False, True])
def test_fleet_local_deadline_is_absolute(socket_dir: Path, monkeypatch, notifications):
    monkeypatch.setenv("RECALL_DATA_DIR", str(socket_dir))
    monkeypatch.setattr("socket.gethostname", lambda: "caller")
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(socket_dir / "recall.sock"))
    listener.listen(1)
    closed = threading.Event()
    requests = []

    def serve():
        conn, _ = listener.accept()
        with conn:
            requests.append(json.loads(conn.makefile("rb").readline()))
            conn.settimeout(0.02)
            for _ in range(100):
                if notifications:
                    try:
                        conn.sendall(b'{"jsonrpc":"2.0","method":"progress","params":{}}\n')
                    except BrokenPipeError:
                        closed.set()
                        return
                try:
                    if conn.recv(1) == b"":
                        closed.set()
                        return
                except TimeoutError:
                    continue

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        executor = FleetExecutor(FleetConfig((), socket_dir / "empty.toml"))
        result = fleet_list(executor, timeout=0.25)
        assert result.hosts_ok == 0
        assert result.hosts_failed == 1
        assert "deadline expired" in result.errors[0]["error"]
        assert closed.wait(2), "expired client did not close"
        assert len(requests) == 1
    finally:
        listener.close()
        thread.join(3)
        assert not thread.is_alive()


def test_fleet_live_keeps_local_read_budget(monkeypatch, tmp_path):
    monkeypatch.setattr("socket.gethostname", lambda: "caller")
    durations = []

    def read(self, method, params, *, seconds):
        durations.append(seconds)
        return {
            "schema_version": 2,
            "watching": True,
            "sessions": [],
            "coverage": {"complete": True},
            "next_cursor": None,
        }

    monkeypatch.setattr("recall.core.read_scope.AggregateRead.call", read)
    executor = FleetExecutor(FleetConfig((), tmp_path / "empty.toml"))
    result = fleet_live(executor)
    assert result.coverage["complete"] is True
    assert result.coverage["hosts"][0]["watching"] is True
    assert durations == [2.0]


def test_fleet_local_remote_global_limit_and_origin_precedence(monkeypatch, tmp_path):
    monkeypatch.setattr("socket.gethostname", lambda: "caller")

    def read(*args, **kwargs):
        return [
            {"id": "local40", "ended_at": "2026-11-10T00:00:40Z", "host": "imported"},
            {"id": "local20", "ended_at": "2026-11-10T00:00:20Z", "host": "local"},
        ]

    def remote(*args, **kwargs):
        return FleetRemoteResult(
            True,
            json.dumps(
                [
                    {"id": "remote30", "ended_at": "2026-11-10T00:00:30Z", "host": "local"},
                    {"id": "remote10", "ended_at": "2026-11-10T00:00:10Z", "host": "foreign"},
                ]
            ),
            "",
            0,
        )

    monkeypatch.setattr("recall.core.read_scope.AggregateRead.call", read)
    monkeypatch.setattr("recall.services.fleet_transport.run_remote", remote)
    executor = FleetExecutor(FleetConfig((FleetHost("edge", "edge"),), tmp_path / "fleet.toml"))
    result = fleet_list(executor, limit=3)
    assert [row["id"] for row in result.rows] == ["local40", "remote30", "local20"]
    assert [row["host"] for row in result.rows] == ["imported", "edge", "caller"]
    union = fleet_list(executor, limit=4)
    assert len(union.rows) == 4
    assert union.rows[-1]["host"] == "foreign"
