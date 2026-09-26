from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path
from typing import Any, Protocol, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "recall" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("PYTHONUTF8", "1")

# The database a real daemon on this machine uses, resolved before any test
# changes the environment.  A test that reaches it would hold its lock and
# write to it, so the in-process RPC server refuses to open it.
_LIVE_DB_PATH = Path(
    os.environ.get(
        "RECALL_DB_PATH",
        Path(os.environ.get("RECALL_DATA_DIR", Path.home() / ".local/share/recall"))
        / "recall.duckdb",
    )
).resolve()


class _LazyRpcServer:
    """Create the test RPC server only when a CLI call needs it."""

    def __init__(self) -> None:
        self._server: Any = None
        self.refused_live_db = False

    @property
    def _real(self) -> Any:
        if self._server is None:
            from recall.core.config import AppConfig
            from recall.services.rpc_server import RpcServer

            config = AppConfig.load()
            if config.db_path.resolve() == _LIVE_DB_PATH:
                # RPC dispatch can run on a worker thread that swallows this
                # error, so the fixture also fails the test from the flag.
                self.refused_live_db = True
                raise RuntimeError(
                    f"test RPC server would open the live database {_LIVE_DB_PATH}; "
                    "point RECALL_DATA_DIR or HOME at a temporary directory"
                )
            self._server = RpcServer(config=config)
        return self._server

    @property
    def _methods(self) -> Any:
        return self._real._methods


class _RpcServerWithConnection(Protocol):
    """The minimal server state required for fixture teardown."""

    _conn: Any


class _RpcServerLifecycle:
    """Own every in-process server installed during one test's lifetime."""

    def __init__(self) -> None:
        self._servers: list[object] = []

    def install(self, server: object) -> None:
        """Install a server and retain it for central teardown."""
        if not any(existing is server for existing in self._servers):
            self._servers.append(server)

        from recall.cli.rpc import set_in_process_server as set_production_server

        set_production_server(server)

    def cleanup(self) -> None:
        """Close realized connections and always clear production dispatch state."""
        close_error: Exception | None = None
        try:
            for server in self._servers:
                # Do not ask a lazy proxy for `_real`: unused tests must remain lazy.
                realized = server._server if isinstance(server, _LazyRpcServer) else server
                if realized is None or not hasattr(realized, "_conn"):
                    continue
                managed_server = cast(_RpcServerWithConnection, realized)
                conn = managed_server._conn
                if conn is None:
                    continue
                try:
                    conn.close()
                except Exception as err:
                    if close_error is None:
                        close_error = err
                finally:
                    # A later teardown must not re-close this connection.
                    managed_server._conn = None
        finally:
            self._servers.clear()
            from recall.cli.rpc import set_in_process_server as set_production_server

            set_production_server(None)

        if close_error is not None:
            raise close_error


_rpc_lifecycle: _RpcServerLifecycle | None = None


def set_in_process_server(server: object) -> None:
    """Install a test replacement while assigning its teardown ownership."""
    if _rpc_lifecycle is None:
        raise RuntimeError("in-process RPC lifecycle is not active")
    _rpc_lifecycle.install(server)


@pytest.fixture(autouse=True)
def _rpc_in_process():
    """Route all CLI RPC calls through in-process dispatch during tests.

    Uses a lazy server that creates itself with the current AppConfig when
    first called, so tests that monkeypatch env vars (HOME, RECALL_DATA_DIR)
    get a server pointing at the right database.
    """
    global _rpc_lifecycle
    lifecycle = _RpcServerLifecycle()
    _rpc_lifecycle = lifecycle
    lazy_server = _LazyRpcServer()
    lifecycle.install(lazy_server)
    try:
        yield
    finally:
        try:
            lifecycle.cleanup()
        finally:
            _rpc_lifecycle = None
    if lazy_server.refused_live_db:
        pytest.fail(
            f"test reached the live database {_LIVE_DB_PATH}; "
            "point RECALL_DATA_DIR or HOME at a temporary directory"
        )


@cache
def _can_invoke_crontab() -> bool:
    """Return whether tests may safely exercise the real crontab binary.

    The probe lists the current crontab but never mutates it. A missing crontab
    still proves the binary is callable and the user has sufficient access.
    """
    if shutil.which("crontab") is None:
        return False
    try:
        result = subprocess.run(
            ["crontab", "-l"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode == 0:
        return True
    return result.returncode == 1 and "no crontab" in (result.stderr or "").lower()


@cache
def _can_acquire_duckdb_lock() -> bool:
    """Return whether tests may open a writable DuckDB database in this sandbox."""
    import duckdb

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "probe.duckdb"
        try:
            conn = duckdb.connect(str(db_path))
            conn.close()
        except Exception:
            return False
        return True


def revert_source_files_to_v29(conn: Any) -> Any:
    """Reshape a current-schema database back to schema version 29.

    Version 30 (REQ-MIG-010) dropped `idx_source_files_pending` and
    `source_files.inventory_generation`. Migration tests need a database that
    still carries both; restoring them follows the same DDL recipe 0026 used,
    because DuckDB cannot add a constrained column or alter one while a
    secondary index is present.
    """
    conn.execute(
        "ALTER TABLE source_files ADD COLUMN IF NOT EXISTS inventory_generation BIGINT DEFAULT 0"
    )
    conn.execute("DROP INDEX IF EXISTS idx_source_files_source_path")
    conn.execute("DROP INDEX IF EXISTS idx_source_files_pending")
    conn.execute("ALTER TABLE source_files ALTER COLUMN inventory_generation SET NOT NULL")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_files_source_path "
        "ON source_files(source, source_path)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_files_pending "
        "ON source_files(committed_generation, desired_generation, next_retry_at)"
    )
    conn.execute("DELETE FROM schema_version WHERE version > 29")
    conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (29)")
    conn.execute("DELETE FROM schema_migrations WHERE migration_id LIKE '0030%'")
    return conn


@pytest.fixture
def to_schema_v29() -> Any:
    """Expose `revert_source_files_to_v29` to tests in any test package."""
    return revert_source_files_to_v29


@pytest.fixture
def launchd_without_legacy_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report no job under the legacy launchd label.

    Fakes that answer every `launchctl print` with 0 would otherwise report a
    legacy job on every host, sending each install through the migration path.
    The legacy label keeps its real probe only in tests that model it.
    """
    import recall.services.daemon as daemon_module

    real_probe = daemon_module._launchd_label_is_loaded

    def probe(label: str) -> bool:
        if label == daemon_module.LEGACY_LAUNCHD_LABEL:
            return False
        return real_probe(label)

    monkeypatch.setattr(daemon_module, "_launchd_label_is_loaded", probe)
