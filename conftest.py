"""Isolation and RPC fixtures shared by both repository test trees."""

from __future__ import annotations

import inspect
import os
import pwd
import shutil
import socket
import subprocess
import sys
import tempfile
from functools import cache, wraps
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import pytest

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "packages" / "recall" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("PYTHONUTF8", "1")

_TEST_ROOT = pytest.StashKey[Path]()
_RECALL_PATHS = {
    "RECALL_DATA_DIR": "data",
    "RECALL_CONFIG_PATH": "config/config.toml",
    "RECALL_DB_PATH": "data/recall.duckdb",
    "RECALL_LOCK_PATH": "data/recall.lock",
    "RECALL_FLEET_PATH": "config/fleet.toml",
    "RECALL_DUCKDB_TEMP_DIR": "spill",
}


@pytest.fixture(params=["~no-such-user", "~no-such-user/", "~no-such-user/existing-root"])
def unexpandable_index_root(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Control failed home lookup without depending on the host's account database."""
    root: str = request.param
    selected = Path(root)
    original = Path.expanduser

    def expanduser(path: Path) -> Path:
        if path == selected:
            raise RuntimeError("Could not determine home directory.")
        return original(path)

    monkeypatch.setattr(Path, "expanduser", expanduser)
    return root


@pytest.fixture
def saturated_rpc_listener(socket_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Hold a real Linux Unix backlog full until the caller drains it."""
    if sys.platform != "linux":
        pytest.skip("deterministic kernel admission EAGAIN requires Linux")
    monkeypatch.setenv("RECALL_DATA_DIR", str(socket_dir))
    held: list[socket.socket] = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(socket_dir / "recall.sock"))
        listener.listen(1)
        listener.settimeout(2)
        try:
            for _ in range(16):
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(0.2)
                try:
                    client.connect(str(socket_dir / "recall.sock"))
                except BlockingIOError:
                    client.close()
                    break
                except BaseException:
                    client.close()
                    raise
                held.append(client)
            else:
                pytest.fail("kernel did not refuse admission within 16 connections")
            yield listener
        finally:
            for client in held:
                client.close()


def _require_test_path(root: Path, name: str, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        pytest.fail(f"recall test isolation: {name}={path} escapes session root {root}")
    return resolved


def pytest_configure(config: pytest.Config) -> None:
    """Isolate collection too: session fixtures run after test-module imports."""
    # HOME and TMPDIR may be arbitrarily deep. Unix sockets need a short root.
    work = Path(pwd.getpwuid(os.getuid()).pw_dir) / "Work"
    work.mkdir(exist_ok=True)
    directory = tempfile.TemporaryDirectory(prefix="rt-", dir=work)
    root = Path(directory.name).resolve()
    config.stash[_TEST_ROOT] = root
    config.add_cleanup(directory.cleanup)
    patch = pytest.MonkeyPatch()
    config.add_cleanup(patch.undo)
    home = root / "home"
    home.mkdir()
    for name, path in {
        "HOME": home,
        "XDG_CONFIG_HOME": home / ".config",
        "XDG_DATA_HOME": home / ".local/share",
        "XDG_CACHE_HOME": home / ".cache",
        "XDG_STATE_HOME": home / ".local/state",
        "XDG_RUNTIME_DIR": root / "run",
        "TMPDIR": root / "tmp",
        "HF_HOME": root / "hf",
        "HF_HUB_CACHE": root / "hf/hub",
        "HUGGINGFACE_HUB_CACHE": root / "hf/hub",
        "HF_XET_CACHE": root / "hf/xet",
        "HF_ASSETS_CACHE": root / "hf/assets",
        "TRANSFORMERS_CACHE": root / "hf/hub",
        "TORCH_HOME": root / "torch",
        **{name: root / relative for name, relative in _RECALL_PATHS.items()},
    }.items():
        patch.setenv(name, str(path))
    (root / "tmp").mkdir()
    (root / "run").mkdir(mode=0o700)
    patch.setattr(tempfile, "tempdir", str(root / "tmp"))
    patch.delenv("KIMI_CODE_HOME", raising=False)
    patch.setenv("PATH", os.pathsep.join((str(Path(sys.executable).parent), os.environ["PATH"])))

    # All tmp_path fixtures must be descendants of the guarded session root.
    config.option.basetemp = str(root / "t")

    find_executable = shutil.which

    def scheduler_guard(event: str, args: tuple[Any, ...]) -> None:
        if event == "subprocess.Popen":
            executable = os.fsdecode(args[0])
            if Path(executable).name in {"systemctl", "launchctl", "crontab"}:
                child_env = args[3]
                search_path = child_env.get("PATH", os.defpath) if child_env else os.environ["PATH"]
                resolved = Path(
                    find_executable(executable, path=search_path) or executable
                ).resolve()
                if not resolved.is_relative_to(root):
                    raise RuntimeError(f"test isolation forbids real scheduler: {resolved}")

    sys.addaudithook(scheduler_guard)

    fake_bin = root / "bin"
    fake_bin.mkdir()
    for name in ("systemctl", "launchctl", "crontab"):
        script = fake_bin / name
        script.write_text(
            f"#!{sys.executable}\nimport os, sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from tests.scheduler_fakes import ArtifactScheduler\n"
            "result = ArtifactScheduler().query([os.path.basename(sys.argv[0]), *sys.argv[1:]])\n"
            "sys.stdout.write(result.stdout)\nsys.stderr.write(result.stderr)\nsys.exit(result.returncode)\n"
        )
        script.chmod(0o755)
    patch.setenv("PATH", os.pathsep.join((str(fake_bin), os.environ["PATH"])))

    from recall.core.config import AppConfig, default_data_dir_for

    load_config = AppConfig.load
    init_config = AppConfig.__init__
    init_signature = inspect.signature(init_config)

    def guarded_load(cls: type[AppConfig]) -> AppConfig:
        home = Path.home()
        data = Path(os.environ.get("RECALL_DATA_DIR", default_data_dir_for(home)))
        for name, default in {
            "RECALL_DATA_DIR": data,
            "RECALL_CONFIG_PATH": home / ".config/recall/config.toml",
            "RECALL_DB_PATH": data / "recall.duckdb",
            "RECALL_LOCK_PATH": data / "recall.lock",
        }.items():
            _require_test_path(root, name, Path(os.environ.get(name, default)))
        # Validate before load reads a potentially external config file.
        return load_config()

    @wraps(init_config)
    def guarded_init(self: AppConfig, *args: Any, **kwargs: Any) -> None:
        bound = init_signature.bind(self, *args, **kwargs)
        for name in ("data_dir", "config_path", "db_path", "lock_path"):
            bound.arguments[name] = _require_test_path(root, name, bound.arguments[name])
        init_config(*bound.args, **bound.kwargs)

    patch.setattr(AppConfig, "load", classmethod(guarded_load))
    patch.setattr(AppConfig, "__init__", guarded_init)


@pytest.fixture(scope="session", autouse=True)
def recall_test_root(pytestconfig: pytest.Config) -> Path:
    """Own the environment and resolver guards installed before collection."""
    return pytestconfig.stash[_TEST_ROOT]


@pytest.fixture(autouse=True)
def _isolated_recall_home(
    recall_test_root: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give each test default-path semantics inside its own fake home.

    Collection pins every override. Tests start without those overrides so
    an explicit fake HOME or RECALL_DATA_DIR also controls derived paths.
    The resolver guards remain active, including when a test changes HOME.
    """
    # Keep HOME separate from fixture trees and directories with exact artifact counts.
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    for name, relative in {
        "XDG_CONFIG_HOME": ".config",
        "XDG_DATA_HOME": ".local/share",
        "XDG_CACHE_HOME": ".cache",
        "XDG_STATE_HOME": ".local/state",
    }.items():
        monkeypatch.setenv(name, str(home / relative))
    for name in _RECALL_PATHS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def socket_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Own a short absolute socket directory, including relative-path callers."""
    directory = tmp_path_factory.mktemp("s")
    monkeypatch.chdir(directory)
    return directory


class _LazyRpcServer:
    """Create the test RPC server only when a CLI call needs it."""

    def __init__(self) -> None:
        self._server: Any = None

    @property
    def _real(self) -> Any:
        if self._server is None:
            from recall.core.config import AppConfig
            from recall.services.rpc_server import RpcServer

            config = AppConfig.load()
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
def _rpc_in_process(_isolated_recall_home):
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


@pytest.fixture(autouse=True)
def scheduler_namespace(monkeypatch: pytest.MonkeyPatch, _isolated_recall_home):
    """Default to verified absence; service scenarios supply their own manager records."""
    from recall.services import service_records

    actual_query = service_records.query

    def query(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["systemctl", "--user", "show"]:
            return subprocess.CompletedProcess(args, 0, "LoadState=not-found\n", "")
        if args == ["systemctl", "--user", "show-environment"]:
            return subprocess.CompletedProcess(args, 0, f"HOME={Path.home()}\n", "")
        if args[:2] == ["launchctl", "print"]:
            if args[2].count("/") == 1:
                return subprocess.CompletedProcess(
                    args,
                    0,
                    f"{args[2]} = {{\n environment = {{\n HOME => {Path.home()}\n }}\n}}\n",
                    "",
                )
            label = args[2].rsplit("/", 1)[1]
            return subprocess.CompletedProcess(
                args, 113, "", f'Could not find service "{label}" in domain for user gui'
            )
        if args == ["crontab", "-l"]:
            return subprocess.CompletedProcess(args, 1, "", "no crontab for test-account")
        raise AssertionError(f"scenario must supply a scheduler response: {args}")

    monkeypatch.setattr(service_records, "query", query)
    monkeypatch.setattr(service_records, "account_home", lambda: Path.home())
    return actual_query


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


@pytest.fixture
def scheduler_platform(monkeypatch: pytest.MonkeyPatch):
    """Declare a scheduler scenario's OS in both the parent and CLI child."""
    from tests.scheduler_fakes import SchedulerPlatform

    def model(name: Literal["linux", "darwin"]) -> SchedulerPlatform:
        monkeypatch.setattr(sys, "platform", name)
        return SchedulerPlatform(name)

    return model


@pytest.fixture
def scheduler_records_from_artifacts(monkeypatch, scheduler_namespace):
    from recall.services import service_records

    from tests.scheduler_fakes import ArtifactScheduler

    scheduler = ArtifactScheduler()
    monkeypatch.setattr(service_records, "query", scheduler.query)
    return scheduler
