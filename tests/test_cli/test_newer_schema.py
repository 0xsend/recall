from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.fts_sidecar import open_sidecar
from recall.db.ownership import ownership_path
from recall.db.schema import SCHEMA_VERSION, ensure_schema, ensure_schema_lenient

FUTURE_VERSION = 999
MESSAGE = (
    f"Database schema {FUTURE_VERSION} is newer than this build supports "
    f"(schema {SCHEMA_VERSION}). Upgrade recall to a release that supports "
    f"database schema {FUTURE_VERSION}, or restore the pre-upgrade backup. "
    "Automatic downgrade is not supported."
)


@pytest.mark.parametrize("lenient", [False, True])
def test_future_schema_refusal_preserves_database(lenient: bool) -> None:
    ensure = ensure_schema_lenient if lenient else ensure_schema
    with duckdb.connect(":memory:") as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (999)")
        conn.execute("CREATE TABLE retained AS SELECT 'keep me' AS content")
        with pytest.raises(RuntimeError, match="Database schema 999") as caught:
            ensure(conn, embed_dim=2)
        assert str(caught.value) == MESSAGE
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(999,)]
        assert conn.execute("SELECT content FROM retained").fetchall() == [("keep me",)]
        assert conn.execute("SHOW TABLES").fetchall() == [("retained",), ("schema_version",)]


@pytest.fixture(params=[False, True], ids=["checkpointed", "committed-wal"])
def future_database(
    socket_dir: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Path:
    data = socket_dir / "future"
    data.mkdir()
    config = socket_dir / "config.toml"
    config.write_text(
        "\n".join(
            f"[sources.{source}]\nroots = []"
            for source in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        )
        + "\n[compaction]\nauto_trigger = false\n",
    )
    for key, path in {
        "RECALL_DATA_DIR": data,
        "RECALL_DB_PATH": data / "recall.duckdb",
        "RECALL_LOCK_PATH": data / "recall.lock",
        "RECALL_CONFIG_PATH": config,
    }.items():
        monkeypatch.setenv(key, str(path))
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    with duckdb.connect(str(data / "recall.duckdb")) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE retained AS SELECT 'keep me' AS content")
    _mark_future(data, retain_wal=bool(request.param))
    open_sidecar(data / "recall.fts.sqlite").close()
    ownership_path(data / "recall.duckdb").touch()
    return data


def _mark_future(data: Path, *, retain_wal: bool) -> None:
    writer = subprocess.run(
        [
            sys.executable,
            "-c",
            "import duckdb,os,sys; c=duckdb.connect(sys.argv[1]); "
            "c.execute('INSERT INTO schema_version (version) VALUES (999)'); os._exit(0)",
            str(data / "recall.duckdb"),
        ],
        check=False,
        capture_output=True,
    )
    assert writer.returncode == 0, writer.stderr
    assert (data / "recall.duckdb.wal").is_file()
    if not retain_wal:
        duckdb.connect(str(data / "recall.duckdb")).close()


def _hashes(data: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in data.iterdir()
        if path.name.startswith(("recall.duckdb", "recall.fts.sqlite"))
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["daemon", "--mode", "poll", "--no-embed"],
        ["daemon", "--once", "--json"],
        ["daemon", "start", "--background", "--json"],
        ["daemon", "start", "--json"],
        ["compact", "--yes", "--json"],
        ["db", "rebuild-indexes", "--json"],
        ["db", "check-indexes", "--json"],
        ["db", "rebuild-indexes", "--format", "text"],
        ["db", "supersede-moved", "--apply", "--json"],
    ],
)
def test_future_schema_cli_refuses_without_rebuild(
    future_database: Path, arguments: list[str]
) -> None:
    before = _hashes(future_database)
    result = subprocess.run(
        [str(Path(sys.executable).parent / "recall"), *arguments],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert "Traceback" not in result.stdout + result.stderr
    assert "--recreate" not in result.stdout + result.stderr
    if "--json" in arguments:
        assert json.loads(result.stdout) == {"error": {"code": "RUNTIME", "message": MESSAGE}}
        assert result.stderr == ""
    else:
        assert f"error: {MESSAGE}" in result.stderr
        assert result.stdout == ""
    assert _hashes(future_database) == before
    assert not (future_database / "recall.pid").exists()
    assert not (future_database / "recall.sock").exists()
    assert not (future_database / "recall.duckdb.compact").exists()


@pytest.mark.parametrize("operation", ["start", "compact"])
def test_future_schema_refuses_before_lifecycle_effects(
    future_database: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from recall.cli import compact as compact_cli
    from recall.cli import daemon as daemon_cli
    from recall.cli.app import app
    from typer.testing import CliRunner

    marker = future_database / "lifecycle-started"

    def lifecycle(*_args, **_kwargs):
        marker.touch()
        raise RuntimeError("unexpected lifecycle operation")

    if operation == "start":
        monkeypatch.setattr(daemon_cli, "start_daemon_durable", lifecycle)
        arguments = ["daemon", "start", "--json"]
    else:
        monkeypatch.setattr(compact_cli, "_stop_daemon", lifecycle)
        arguments = ["compact", "--yes", "--json"]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout) == {"error": {"code": "RUNTIME", "message": MESSAGE}}
    assert not marker.exists()


@pytest.mark.parametrize("read_only", [False, True])
def test_existing_connection_refuses_future_schema(future_database: Path, read_only: bool) -> None:
    from recall.db.connection import connect_existing

    before = _hashes(future_database)
    with pytest.raises(RuntimeError, match="Database schema 999") as caught:
        connect_existing(AppConfig.load(), read_only=read_only)
    assert str(caught.value) == MESSAGE
    assert _hashes(future_database) == before


@pytest.mark.parametrize("operation", ["compact", "keyword"])
def test_maintenance_refuses_future_schema_before_creating_artifacts(
    future_database: Path, operation: str
) -> None:
    from recall.services.compaction import compact
    from recall.services.fts_sidecar_rebuild import KeywordRebuild

    config = AppConfig.load()
    before = _hashes(future_database)
    with pytest.raises(RuntimeError, match="Database schema 999"):
        if operation == "compact":
            compact(config)
        else:
            KeywordRebuild(config, future_database / "operation", "test-build").begin()
    assert _hashes(future_database) == before
    assert not (future_database / "operation").exists()
    assert not list(future_database.glob("*.compact"))
    assert not list(future_database.glob("*.pre-compact"))
    assert not (future_database / "keyword-maintenance.json").exists()


@pytest.mark.parametrize("unavailable", ["external_writer", "ownership_busy", "unreadable"])
def test_unavailable_schema_snapshot_preserves_service_refusal(
    future_database: Path, monkeypatch: pytest.MonkeyPatch, unavailable: str
) -> None:
    import fcntl

    from recall.cli import daemon as daemon_cli
    from recall.cli import rpc as rpc_cli
    from recall.cli.app import app
    from recall.db.ownership import ownership_path
    from typer.testing import CliRunner

    monkeypatch.setattr(rpc_cli, "_in_process_server", None)
    monkeypatch.setattr(
        daemon_cli,
        "resolve_service_scope",
        lambda config: SimpleNamespace(
            service=SimpleNamespace(scheduler=SimpleNamespace(value="systemd")),
            other_service=None,
        ),
    )
    process = None
    owner = None
    try:
        if unavailable == "external_writer":
            ready = future_database / "writer-ready"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import duckdb,sys; from pathlib import Path; "
                    "conn=duckdb.connect(sys.argv[1]); Path(sys.argv[2]).touch(); "
                    "sys.stdin.read(); conn.close()",
                    str(future_database / "recall.duckdb"),
                    str(ready),
                ],
                stdin=subprocess.PIPE,
            )
            deadline = time.monotonic() + 10
            while not ready.exists():
                assert process.poll() is None, "external writer exited before readiness"
                assert time.monotonic() < deadline, "external writer readiness timed out"
                time.sleep(0.01)
        elif unavailable == "ownership_busy":
            owner = ownership_path(future_database / "recall.duckdb").open("a")
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            (future_database / "recall.duckdb").write_bytes(b"unreadable database")

        result = CliRunner().invoke(app, ["stats", "--json"])
        assert result.exit_code == 1, result.output
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "RUNTIME"
        assert "refusing to auto-fork" in error["message"]
        assert "Traceback" not in result.output
    finally:
        if owner is not None:
            owner.close()
        if process is not None:
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)


@pytest.mark.parametrize("lenient", [False, True])
def test_writer_open_refuses_without_recovering_wal(future_database: Path, lenient: bool) -> None:
    from recall.db.connection import connect

    before = _hashes(future_database)
    with pytest.raises(RuntimeError, match="Database schema 999"):
        connect(AppConfig.load(), lenient_schema=lenient)
    assert _hashes(future_database) == before


def test_storage_transition_refuses_future_schema(future_database: Path) -> None:
    from recall.services.index_migration import begin_storage_migration, perform_storage_transition

    config = AppConfig.load()
    retain_wal = (future_database / "recall.duckdb.wal").exists()
    # Prepare a real eligible legacy-format job and its matching backup.
    with duckdb.connect(str(config.db_path)) as conn:
        conn.execute("DELETE FROM schema_version WHERE version = 999")
        status = begin_storage_migration(conn, config)
        assert status.storage_target == "v1.2.0"
        assert status.backup_path is not None
    _mark_future(future_database, retain_wal=retain_wal)
    before = _hashes(future_database)
    with pytest.raises(RuntimeError, match="Database schema 999"):
        perform_storage_transition(config)
    assert _hashes(future_database) == before


def test_confirmed_future_recreation_retains_readable_backup(
    future_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.cli import rpc as rpc_cli
    from recall.cli.app import app
    from recall.core import rpc_client
    from recall.core.entrypoint import DetachedProcess
    from typer.testing import CliRunner

    monkeypatch.setattr(rpc_cli, "_in_process_server", None)
    children: list[subprocess.Popen[bytes]] = []
    log_path = future_database / "owned-daemon.log"
    with log_path.open("wb") as log:

        def start(data_dir, _args):
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import runpy,sys; from pathlib import Path; "
                    "from recall.services import usage_harvest; "
                    "original=usage_harvest.harvest_grok_unified_log; "
                    "empty=Path(sys.argv[2]); empty.touch(); "
                    "usage_harvest.harvest_grok_unified_log="
                    "lambda c,*a,**k: original(c,empty,**k); "
                    "sys.argv=[sys.argv[1],'daemon','--mode','poll','--no-embed']; "
                    "runpy.run_path(sys.argv[0],run_name='__main__')",
                    str(Path(sys.executable).parent / "recall"),
                    str(future_database / "empty-grok.jsonl"),
                ],
                stdout=log,
                stderr=log,
                env=os.environ.copy(),
            )
            children.append(child)
            return DetachedProcess(child, "owned foreground daemon", data_dir)

        # Only the launcher is replaced, to retain and reap its actual child.
        # CLI parsing, preflight, backup, daemon startup and RPC are real.
        monkeypatch.setattr(rpc_client, "spawn_detached_recall", start)
        try:
            result = CliRunner().invoke(
                app, ["index", "--recreate", "--yes", "--no-embed", "--json"]
            )
            assert result.exit_code == 0, (result.output, log_path.read_text())
            assert json.loads(result.stdout)["failed"] == 0
            backups = list((future_database / "snapshots").glob("recreate-*"))
            assert len(backups) == 1
            assert json.loads(result.stdout)["backup_path"] == str(backups[0])
            retained = []
            for backup in backups:
                manifest = json.loads((backup / "manifest.json").read_text())
                assert manifest["complete"] is True
                assert (backup / "recall.fts.sqlite").is_file()
                with duckdb.connect(str(backup / "recall.duckdb"), read_only=True) as conn:
                    version = conn.execute("SELECT max(version) FROM schema_version").fetchone()
                    if version == (999,):
                        retained = conn.execute("SELECT content FROM retained").fetchall()
            assert retained == [("keep me",)]
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
    with duckdb.connect(str(future_database / "recall.duckdb"), read_only=True) as conn:
        assert conn.execute("SELECT max(version) FROM schema_version").fetchone() == (36,)


@pytest.mark.parametrize(
    "arguments",
    [
        ["index", "--recreate", "--no-embed", "--json"],
        ["index", "--recreate", "--yes", "--dry-run", "--json"],
    ],
)
def test_future_recreation_requires_confirmation_and_execution(
    future_database: Path, arguments: list[str]
) -> None:
    before = _hashes(future_database)
    result = subprocess.run(
        [str(Path(sys.executable).parent / "recall"), *arguments],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == (0 if "--dry-run" in arguments else 2)
    assert _hashes(future_database) == before
    assert not (future_database / "snapshots").exists()
    assert not (future_database / "recall.pid").exists()


def test_recreation_backup_failure_preserves_future_content(
    future_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.cli import rpc as rpc_cli
    from recall.cli.app import app
    from recall.services import recreation
    from typer.testing import CliRunner

    def unavailable(*_args, **_kwargs):
        raise OSError("backup destination unavailable")

    monkeypatch.setattr(rpc_cli, "_in_process_server", None)
    monkeypatch.setattr(recreation, "backup_before_recreate", unavailable)
    result = CliRunner().invoke(app, ["index", "--recreate", "--yes", "--json"])
    assert result.exit_code == 1
    assert "backup destination unavailable" in json.loads(result.stdout)["error"]["message"]
    with duckdb.connect(str(future_database / "recall.duckdb"), read_only=True) as conn:
        assert conn.execute("SELECT max(version) FROM schema_version").fetchone() == (999,)
        assert conn.execute("SELECT content FROM retained").fetchall() == [("keep me",)]
    assert not (future_database / "recall.pid").exists()


def test_writer_handles_share_compatible_database(tmp_path: Path) -> None:
    from recall.db.connection import connect, connect_existing

    config = replace(AppConfig.load(), data_dir=tmp_path, db_path=tmp_path / "recall.duckdb")
    first = connect(config)
    try:
        first.execute("CREATE TABLE retained AS SELECT 17 AS value")
        second = connect_existing(config, read_only=False)
        try:
            assert second.execute("SELECT value FROM retained").fetchall() == [(17,)]
            second.execute("UPDATE retained SET value=18")
        finally:
            second.close()
        third = connect(config)
        try:
            assert third.execute("SELECT value FROM retained").fetchall() == [(18,)]
        finally:
            third.close()
    finally:
        first.close()


def test_paused_future_recreation_refuses_before_writes(
    future_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.cli import rpc as rpc_cli
    from recall.cli.app import app
    from recall.services.coordinator import set_paused
    from typer.testing import CliRunner

    monkeypatch.setattr(rpc_cli, "_in_process_server", None)
    set_paused(AppConfig.load(), True)
    before = _hashes(future_database)
    result = CliRunner().invoke(app, ["index", "--recreate", "--yes", "--json"])
    assert result.exit_code == 1
    assert result.stdout, repr(result.exception)
    assert json.loads(result.stdout) == {
        "error": {
            "code": "LOCKED",
            "message": "reconciliation is paused; resume it with `recall daemon resume`",
        }
    }
    assert _hashes(future_database) == before
    assert not (future_database / "snapshots").exists()
    assert not (future_database / "recall.pid").exists()


def test_recreated_future_backup_survives_daemon_launch_failure(
    future_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.cli import rpc as rpc_cli
    from recall.cli.app import app
    from recall.core import rpc_client
    from typer.testing import CliRunner

    def unavailable(*_args, **_kwargs):
        raise OSError("daemon launch unavailable")

    monkeypatch.setattr(rpc_cli, "_in_process_server", None)
    monkeypatch.setattr(rpc_client, "spawn_detached_recall", unavailable)
    result = CliRunner().invoke(app, ["index", "--recreate", "--yes", "--json"])
    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert error["code"] == "RUNTIME"
    assert "daemon launch unavailable" in error["message"]
    backup = Path(error["details"]["backup_path"])
    assert str(backup) in error["message"]
    assert list((future_database / "snapshots").glob("recreate-*")) == [backup]
    with duckdb.connect(str(backup / "recall.duckdb"), read_only=True) as conn:
        assert conn.execute("SELECT max(version) FROM schema_version").fetchone() == (999,)
        assert conn.execute("SELECT content FROM retained").fetchall() == [("keep me",)]
    assert not (future_database / "recall.pid").exists()


def test_confirmed_future_recreation_respects_managed_service(
    future_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.cli import daemon as daemon_cli
    from recall.cli import rpc as rpc_cli
    from recall.cli.app import app
    from typer.testing import CliRunner

    monkeypatch.setattr(rpc_cli, "_in_process_server", None)
    monkeypatch.setattr(
        daemon_cli,
        "resolve_service_scope",
        lambda _config: SimpleNamespace(
            service=SimpleNamespace(scheduler=SimpleNamespace(value="systemd")),
            other_service=None,
        ),
    )
    before = _hashes(future_database)
    result = CliRunner().invoke(app, ["index", "--recreate", "--yes", "--json"])
    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert error["code"] == "RUNTIME"
    assert "refusing to auto-fork" in error["message"]
    assert _hashes(future_database) == before
    assert not (future_database / "snapshots").exists()
