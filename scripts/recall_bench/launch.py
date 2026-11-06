"""Run the build under test's recall with the bench guards installed.

The supervisor runs this file with the build's own interpreter (`-I -B`), so
it imports the standard library and the build's `recall` package. An
embedding-enabled daemon also loads the adjacent observer-only instrumentation
module with the build's psutil and Pydantic dependencies. Before any recall code runs it

- installs an audit hook that refuses to start a service manager
  (`launchctl`, `systemctl`, `crontab`) or another `recall` process, and to
  send any signal to a process other than itself, and records every
  refusal. A signal's target would come from a PID file or a probe, and
  neither proves the process is one the supervisor owns; only the
  supervisor, which holds its children unreaped, signals them;
- replaces the build's service probe, scheduler query and control entry
  points with stubs that report "not installed" or refuse, recording each
  call;
- points the Grok usage harvester at the injected log.

All modes intercept seam imports before any recall import. Function code
constants are replaced before module execution; original bodies never become
callable, including during initialization and circular imports. Original code
metadata supplies parameter-name validation. `check` and `admission` import
and validate every seam; other modes import only the modules they use. Preloaded
seam modules, reloads, unsupported loaders, execution failures and validation
failures terminate the process with a SEAM_MISSING report and an attempt record.
Failures cannot be caught by importing code or leave threads running. This
protects against accidental drift in trusted Python builds, not hostile code.

In `cli` mode the call must be one of `ALLOWED_CLI`, judged by the build's own
command parser, with no raw `--params`, `--fleet*` or `--host` value. There
is no `daemon stop` of any kind: the supervisor stops the daemon when the
job ends. Anything else exits REFUSED without running.

A supervised launcher waits for one byte on stdin before anything else, so
its supervisor records it before it does any work; `--ungated` skips that for
a guest's own CLI calls.

Usage: python -I -B launch.py --attempts FILE --usage-log FILE [--ungated] MODE [ARGS...]
MODE is `check` (seams and version only), `preflight` (print the resolved
config's paths), `daemon` (the foreground daemon), `cli` (one CLI call),
or `admission` (one fixed offline database operation).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import inspect
import json
import os
import runpy
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import CodeType, ModuleType
from typing import Any, Literal, Never, cast

SEAM_MISSING = 3
UNSUPPORTED_BUILD = 4
REFUSED = 5
NOT_RELEASED = 6
# The first release whose service actions are scoped to scheduler records (#217)
# and that has every seam below.
SUPPORTED_BUILD_FLOOR = (0, 40, 5)
SERVICE_MANAGERS = frozenset({"launchctl", "systemctl", "crontab"})
EXEC_EVENTS = frozenset({"subprocess.Popen", "os.posix_spawn", "os.exec", "os.spawn", "os.system"})
SIGNAL_EVENTS = frozenset({"os.kill", "os.killpg"})


@dataclass(frozen=True)
class Seam:
    """One build entry point the launcher replaces before any recall code runs."""

    module: str
    attribute: str
    # Parameter names its signature must have, so a changed contract is refused.
    parameters: tuple[str, ...]
    # What replaces it: a "not installed" probe, a refusal, or the injected log.
    replacement: Literal["probe", "refuse", "usage_log"]

    @property
    def name(self) -> str:
        return f"{self.module}.{self.attribute}"


SEAMS = (
    Seam("recall.services.daemon", "probe_service", ("config", "installed_scheduler"), "probe"),
    Seam("recall.services.daemon", "_run_command", ("args",), "refuse"),
    Seam("recall.services.daemon", "_run_lifecycle_command", ("args", "action"), "refuse"),
    Seam("recall.services.daemon", "_write_crontab", ("contents",), "refuse"),
    Seam("recall.services.service_records", "query", ("args",), "refuse"),
    Seam("recall.services.service_records", "discover", ("scheduler", "run"), "refuse"),
    Seam("recall.services.usage_harvest", "default_grok_unified_log", ("home",), "usage_log"),
    Seam("recall.core.entrypoint", "spawn_detached_recall", ("data_dir", "args"), "refuse"),
    Seam("recall.core.rpc_client", "spawn_detached_recall", ("data_dir", "args"), "refuse"),
    Seam("recall.cli.daemon", "spawn_detached_recall", ("data_dir", "args"), "refuse"),
)
SERVICE_VIEW_FIELDS = ("installed_scheduler_record", "scheduler", "installed", "measured_at")
ALLOWED_CLI = frozenset({"stats", "live", "show", "search", "list", "daemon status"})
# Options that change what a call reaches rather than what it reports.
REACHING_OPTIONS = ("params", "fleet", "fleet_config", "host")


class Attempts:
    """An append-only record of every suppressed or refused service reach."""

    def __init__(self, path: Path) -> None:
        self._descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)

    def record(self, kind: str, seam: str, detail: object) -> None:
        line = json.dumps(
            {"kind": kind, "seam": seam, "detail": detail, "pid": os.getpid(), "at": time.time()},
            default=str,
        )
        os.write(self._descriptor, (line + "\n").encode())


def _seam_problems() -> list[str]:
    # The guarded loader validates the original definitions as code, never as
    # callable originals. Import failure terminates the disposable process.
    for name in dict.fromkeys(seam.module for seam in SEAMS):
        importlib.import_module(name)
    if not callable(getattr(importlib.import_module("recall.cli"), "main", None)):
        return ["recall.cli.main is missing"]
    return []


def _version() -> tuple[str, tuple[int, ...]]:
    text = importlib.metadata.version("recall")
    numbers: list[int] = []
    for part in text.split(".")[:3]:
        digits = "".join(character for character in part if character.isdigit())
        numbers.append(int(digits) if digits else 0)
    return text, tuple(numbers)


def _program(event: str, arguments: tuple[object, ...]) -> str | None:
    """The executable an exec-family audit event would start."""
    if event == "os.system":
        words = _text(arguments[0]).split()
        return words[0] if words else None
    if event == "subprocess.Popen":
        executable, argv = arguments[0], arguments[1]
        if executable is not None:
            return _text(executable)
        if isinstance(argv, str | bytes):
            words = _text(argv).split()
            return words[0] if words else None
        if isinstance(argv, list | tuple) and argv:
            return _text(argv[0])
        return None
    return _text(arguments[1] if event == "os.spawn" else arguments[0])


def _text(value: object) -> str:
    if isinstance(value, str | bytes | os.PathLike):
        return os.fsdecode(value)
    return str(value)


def _signals_another_process(event: str, arguments: tuple[object, ...]) -> bool:
    """A `kill` or `killpg` that delivers a signal to anything but this process."""
    target, signum = arguments[0], arguments[1]
    if signum == 0:
        # Signal 0 only asks whether the target exists.
        return False
    return not (event == "os.kill" and target == os.getpid())


def _install_guard(attempts: Attempts) -> None:
    def guard(event: str, arguments: tuple[object, ...]) -> None:
        if event in SIGNAL_EVENTS:
            if _signals_another_process(event, arguments):
                target, signum = arguments[0], arguments[1]
                attempts.record("refused_signal", event, {"target": target, "signal": signum})
                raise PermissionError(f"bench launcher refuses to signal {target}")
            return
        if event not in EXEC_EVENTS:
            return
        program = _program(event, arguments)
        name = Path(program).name if program else ""
        if name in SERVICE_MANAGERS or name == "recall":
            attempts.record("refused", event, {"program": program})
            raise PermissionError(f"bench launcher refuses to start {program}")

    sys.addaudithook(guard)


class _SeamGuard:
    """Own interception and fatal failure for this disposable launcher process.

    Builds and their loaders are trusted, but their import order is not. Only
    Python code loaders are supported. Original seam code is inspected as data
    and replaced before execution can create a function referring to it. The
    original code is never executed, including during circular imports or in
    threads started by initializers. This is not a sandbox for hostile Python.
    """

    def __init__(self, attempts: Attempts, usage_log: Path) -> None:
        self.attempts = attempts
        self.usage_log = usage_log
        self.seams = {
            name: tuple(seam for seam in SEAMS if seam.module == name)
            for name in dict.fromkeys(seam.module for seam in SEAMS)
        }
        self.signatures: dict[CodeType, tuple[str, ...]] = {}

    def fail(self, problem: str) -> Never:
        # Exceptions, even SystemExit, can be caught by an importing module.
        # No finalizers or initializer-started threads may continue after failure.
        try:
            self.attempts.record("seam_missing", "import", problem)
            print(json.dumps({"seams": "missing", "problems": [problem]}), flush=True)
        finally:
            os._exit(SEAM_MISSING)

    def replacement(self, seam: Seam, module: ModuleType) -> Callable[..., object]:
        if seam.replacement == "probe":

            def probe(config: object, installed_scheduler: object) -> object:
                self.attempts.record(
                    "suppressed", "probe_service", {"installed_scheduler": installed_scheduler}
                )
                try:
                    service_view: Callable[..., object] = getattr(module, "ServiceView")  # noqa: B009
                    return service_view(
                        installed_scheduler_record=installed_scheduler,
                        scheduler=None,
                        installed=False,
                        service_data_dir=None,
                        measured_at=time.time(),
                    )
                except BaseException as error:
                    self.fail(f"{seam.name}: {error}")

            return probe

        def stub(*args: object, **kwargs: object) -> object:
            if seam.replacement == "usage_log":
                return self.usage_log
            self.attempts.record("suppressed", seam.name, {"args": args, "kwargs": kwargs})
            raise PermissionError(f"bench launcher suppresses {seam.name}")

        return stub

    def protect(self, code: CodeType, seams: tuple[Seam, ...]) -> CodeType:
        """Replace function constants without constructing original callables.

        The loader's code (usually from the build's pyc cache) retains its own
        defaults, annotations and imports. A replacement function accepts their
        calls through the module's injected stubs; captures therefore retain
        protected code too. No bytecode instructions depend on a Python version.
        """
        constants: list[object] = []
        by_name = {seam.attribute: seam for seam in seams}
        for value in code.co_consts:
            if isinstance(value, CodeType):
                seam = by_name.get(value.co_name)
                if seam is None:
                    value = self.protect(value, seams)
                else:
                    count = value.co_argcount + value.co_kwonlyargcount
                    count += bool(value.co_flags & inspect.CO_VARARGS)
                    count += bool(value.co_flags & inspect.CO_VARKEYWORDS)
                    names = value.co_varnames[:count]
                    if value.co_freevars or not set(seam.parameters) <= set(names):
                        self.fail(
                            f"{seam.name}{names} lacks parameters {seam.parameters} "
                            "or closes over variables"
                        )
                    # Compiling a forwarding function gives CPython ownership of
                    # its bytecode layout; no original body enters the result.
                    wrapper = compile(
                        f"def {seam.attribute}(*args, **kwargs):\n"
                        f"    return __recall_bench_stubs__[{seam.attribute!r}](*args, **kwargs)\n",
                        code.co_filename,
                        "exec",
                    )
                    value = next(item for item in wrapper.co_consts if isinstance(item, CodeType))
                    self.signatures[value] = names
            constants.append(value)
        return code.replace(co_consts=tuple(constants))

    def validate(self, module: ModuleType) -> None:
        for seam in self.seams[module.__name__]:
            target = getattr(module, seam.attribute, None)
            names = self.signatures.get(getattr(target, "__code__", None))
            if names is None or not set(seam.parameters) <= set(names):
                self.fail(f"{seam.name} is missing or is not a protected Python function")
            if seam.replacement == "probe":
                fields = getattr(getattr(module, "ServiceView", None), "__dataclass_fields__", {})
                missing = [name for name in SERVICE_VIEW_FIELDS if name not in fields]
                if missing:
                    self.fail(f"{seam.module}.ServiceView lacks fields {missing}")


class _PatchingLoader(importlib.abc.Loader):
    """Execute only code whose seam function constants have been replaced."""

    def __init__(self, loader: importlib.abc.InspectLoader, guard: _SeamGuard) -> None:
        self._loader = loader
        self._guard = guard

    def create_module(self, spec: Any) -> None:
        # A custom create_module may execute arbitrary build code. Use the
        # standard Python module object and keep all execution inside the guard.
        return None

    def exec_module(self, module: ModuleType) -> None:
        name = module.__name__
        try:
            code = self._loader.get_code(name)
            if code is None:
                self._guard.fail(f"{name} has no Python code")
            seams = self._guard.seams[name]
            protected = self._guard.protect(code, seams)
            module.__dict__["__recall_bench_stubs__"] = {
                seam.attribute: self._guard.replacement(seam, module) for seam in seams
            }
            exec(protected, module.__dict__)
            self._guard.validate(module)
        except BaseException as error:
            self._guard.fail(f"{name}: {type(error).__name__}: {error}")


class _SeamFinder(importlib.abc.MetaPathFinder):
    """Retain every seam module, refusing preloaded modules and all reloads."""

    def __init__(self, guard: _SeamGuard) -> None:
        self._guard = guard
        self._finders = tuple(sys.meta_path)

    def find_spec(self, fullname: str, path: Any = None, target: ModuleType | None = None) -> Any:
        if fullname not in self._guard.seams:
            return None
        if target is not None:
            self._guard.fail(f"{fullname}: reload is refused")
        try:
            for finder in self._finders:
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    if isinstance(spec.loader, _PatchingLoader):
                        if spec.loader._guard is not self._guard:
                            self._guard.fail(f"{fullname}: another seam guard owns this loader")
                        return spec
                    if not isinstance(spec.loader, importlib.abc.InspectLoader):
                        self._guard.fail(f"{fullname}: a Python code loader is required")
                    spec.loader = _PatchingLoader(spec.loader, self._guard)
                    return spec
        except BaseException as error:
            self._guard.fail(f"{fullname}: {type(error).__name__}: {error}")
        self._guard.fail(f"{fullname} is missing")


def _install_stubs(attempts: Attempts, usage_log: Path) -> None:
    guard = _SeamGuard(attempts, usage_log)
    preloaded = [name for name in guard.seams if name in sys.modules]
    if preloaded:
        guard.fail(f"seam modules were imported before interception: {preloaded}")
    sys.meta_path.insert(0, _SeamFinder(guard))


def _subcommands(command: object) -> dict[str, object]:
    commands = getattr(command, "commands", None)
    return commands if isinstance(commands, dict) else {}


def _cli_refusal(argv: list[str]) -> str | None:
    """Why the product parser reads `argv` as a call outside ALLOWED_CLI, or None."""
    typer_main = importlib.import_module("typer.main")
    app = importlib.import_module("recall.cli.app").app
    command = typer_main.get_command(app)
    path: list[str] = []
    remaining = list(argv)
    while remaining and remaining[0] in _subcommands(command):
        path.append(remaining[0])
        command = _subcommands(command)[remaining.pop(0)]
    name = " ".join(path)
    if name not in ALLOWED_CLI:
        return f"`recall {name or ' '.join(argv)}` is not an allowed bench call"
    try:
        context = command.make_context(name, remaining)  # ty: ignore[unresolved-attribute]
    except Exception as error:
        # `--help` prints the command's help and exits 0 through the parser; nothing else runs.
        if getattr(error, "exit_code", None) == 0:
            raise SystemExit(0) from None
        return f"`recall {name}` does not parse: {error}"
    params = dict(context.params)
    reaching = [option for option in REACHING_OPTIONS if params.get(option) not in (None, False)]
    if reaching:
        return f"`recall {name}` may not set {reaching}"
    return None


def _config_paths(config: Any = None) -> dict[str, object]:
    if config is None:
        config = importlib.import_module("recall.core.config").AppConfig.load()
    return {
        "cwd": os.getcwd(),
        "data_dir": str(config.data_dir),
        "db_path": str(config.db_path),
        "lock_path": str(config.lock_path),
        "config_path": str(config.config_path),
        "fleet_path": os.environ.get("RECALL_FLEET_PATH", ""),
        "socket": str(config.data_dir / "recall.sock"),
        "pid": str(config.data_dir / "recall.pid"),
        "duckdb_temp_dir": str(getattr(config.duckdb, "temp_directory", None) or ""),
        "sources": {
            name: [str(root) for root in (source.roots or ())]
            for name, source in config.sources.items()
        },
        "scheduler": str(config.daemon.scheduler.value),
        "mode": str(config.daemon.mode.value),
        "embed": bool(config.daemon.embed),
    }


ADMISSION_FORMAT = "recall-bench-admission/1"
ADMISSION_BYTES_MAX = 1024 * 1024
SCHEMA_MAX = 2_147_483_647
AUDIT_ROWS_MAX = 4096
SPILL_BYTES_MAX = 64 * 1024**3
Operation = Literal[
    "inspect", "positive-read", "positive-write", "prepare-newer", "negative-read", "negative-write"
]
State = Literal["inspected", "opened", "prepared", "refused", "error"]
OPERATIONS = frozenset(
    {
        "inspect",
        "positive-read",
        "positive-write",
        "prepare-newer",
        "negative-read",
        "negative-write",
    }
)
VERSIONED_OPERATIONS = frozenset({"prepare-newer", "negative-read", "negative-write"})


def schema_number(text: str) -> int:
    if not text or len(text) > 10 or not text.isascii() or not text.isdecimal():
        raise ValueError("schema must be canonical ASCII decimal")
    number = int(text)
    if str(number) != text or not 1 <= number <= SCHEMA_MAX:
        raise ValueError(f"schema must be canonical decimal within 1..{SCHEMA_MAX}")
    return number


@dataclass(frozen=True)
class AdmissionRequest:
    operation: Operation
    schema: int | None = None

    def __post_init__(self) -> None:
        if self.operation not in OPERATIONS:
            raise ValueError("unknown admission operation")
        if self.operation in VERSIONED_OPERATIONS:
            if type(self.schema) is not int or not 1 <= self.schema <= SCHEMA_MAX:
                raise ValueError("operation requires a bounded schema number")
        elif self.schema is not None:
            raise ValueError("operation takes no operand")

    @classmethod
    def parse(cls, argv: list[str]) -> AdmissionRequest:
        if not argv or argv[0] not in OPERATIONS:
            raise ValueError("unknown admission operation")
        versioned = argv[0] in VERSIONED_OPERATIONS
        if len(argv) != (2 if versioned else 1):
            raise ValueError("wrong admission operand count")
        return cls(cast(Operation, argv[0]), schema_number(argv[1]) if versioned else None)

    def argv(self) -> list[str]:
        return [self.operation, *([] if self.schema is None else [str(self.schema)])]

    def validate_schema(self, supported: int) -> None:
        if self.schema is not None and self.schema <= supported:
            raise ValueError(
                f"negative schema {self.schema} must exceed supported schema {supported}"
            )


def validate_admission_paths(paths: object, root: Path) -> dict[str, object]:
    """One exact config-path contract, used by parent preflight and the child leaf."""
    expected: dict[str, object] = {
        "cwd": os.path.realpath(root),
        "data_dir": str(root),
        "db_path": str(root / "recall.duckdb"),
        "lock_path": str(root / "recall.lock"),
        "config_path": str(root / "config.toml"),
        "fleet_path": str(root / "fleet.toml"),
        "socket": str(root / "recall.sock"),
        "pid": str(root / "recall.pid"),
        "duckdb_temp_dir": str(root / "spill"),
        "sources": {
            name: [str(root / "pool" / name)]
            for name in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        },
        "scheduler": "auto",
        "mode": "watch",
        "embed": False,
    }
    if not isinstance(paths, dict):
        raise ValueError("admission config has no path observations")
    different = [name for name, value in expected.items() if paths.get(name) != value]
    if different:
        raise ValueError(f"admission config differs from the owned run: {different}")
    return paths


def _nonstandard_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant {value}")


@dataclass(frozen=True)
class AdmissionObservation:
    operation: Operation
    state: State
    build: str
    supported_schema: int
    paths: dict[str, object]
    observed_schema: int | None = None
    schema_before: int | None = None
    audit_before: dict[str, object] | None = None
    audit_after: dict[str, object] | None = None
    settings: dict[str, object] | None = None
    probe: dict[str, object] | None = None
    error: dict[str, object] | None = None
    format: str = ADMISSION_FORMAT

    def __post_init__(self) -> None:
        if self.format != ADMISSION_FORMAT or self.operation not in OPERATIONS:
            raise ValueError("unknown admission document format/operation")
        if self.state not in {"inspected", "opened", "prepared", "refused", "error"}:
            raise ValueError("unknown admission state")
        if not isinstance(self.build, str) or not self.build or len(self.build) > 256:
            raise ValueError("invalid build version")
        for value in (self.supported_schema, self.observed_schema, self.schema_before):
            if value is not None and (type(value) is not int or not 0 <= value <= SCHEMA_MAX):
                raise ValueError("invalid schema observation")
        if type(self.supported_schema) is not int:
            raise ValueError("missing supported schema")
        if not isinstance(self.paths, dict):
            raise ValueError("missing path observations")
        for value in (self.audit_before, self.audit_after, self.settings, self.probe, self.error):
            if value is not None and not isinstance(value, dict):
                raise ValueError("invalid admission detail")
        if self.state in {"refused", "error"} and self.error is None:
            raise ValueError("missing error observation")
        if self.error is not None:
            if self.state != "error" and "original_bytes" in self.error:
                raise ValueError("oversized diagnostic cannot be a completed observation")
            for name in ("type", "message"):
                value = self.error.get(name)
                if not isinstance(value, str) or len(value.encode()) > 4096:
                    raise ValueError("invalid or oversized error detail")
            if self.state == "refused" and any(
                type(self.error.get(name)) is not int
                for name in ("database_version", "supported_version")
            ):
                raise ValueError("refusal lacks typed schema versions")
        if self.state in {"opened", "prepared"} and self.observed_schema is None:
            raise ValueError("missing observed schema")

    def to_json(self) -> bytes:
        data = (json.dumps(asdict(self), sort_keys=True, allow_nan=False) + "\n").encode()
        if len(data) > ADMISSION_BYTES_MAX:
            raise ValueError("admission document exceeds 1 MiB")
        return data

    @classmethod
    def parse(cls, data: bytes) -> AdmissionObservation:
        if len(data) > ADMISSION_BYTES_MAX:
            raise ValueError("admission document exceeds 1 MiB")
        try:
            document = json.loads(data, parse_constant=_nonstandard_constant)
        except (UnicodeError, RecursionError) as error:
            raise ValueError("invalid admission JSON encoding/depth") from error
        if not isinstance(document, dict) or "format" not in document:
            raise ValueError("admission document must be an object")
        try:
            return cls(**document)
        except TypeError as error:
            raise ValueError(f"invalid admission fields: {error}") from error


ADMISSION_SEAMS = (
    ("recall.db.connection", "connect", ("config", "recreate", "lenient_schema")),
    ("recall.db.connection", "connect_readonly", ("config",)),
    ("recall.db.connection", "connect_existing", ("config", "read_only")),
    ("recall.db.schema", "get_schema_version", ("conn",)),
    ("recall.db.schema", "set_schema_version_to", ("conn", "version")),
    ("recall.db.maintenance", "probe_index_divergence", ("conn", "sample", "budget_seconds")),
    ("recall.core.schema_version", "SchemaTooNewError", ("database_version", "supported_version")),
)


def _admission_problems() -> list[str]:
    problems: list[str] = []
    for module_name, name, parameters in ADMISSION_SEAMS:
        try:
            module = importlib.import_module(module_name)
            target = getattr(module, name, None)
            if not callable(target) or not set(parameters) <= set(
                inspect.signature(target).parameters
            ):
                problems.append(f"{module_name}.{name} lacks admission signature {parameters}")
        except (ImportError, ValueError, TypeError) as error:
            problems.append(f"{module_name}.{name}: {error}")
    if problems:
        return problems
    schema = importlib.import_module("recall.db.schema")
    supported = getattr(schema, "SCHEMA_VERSION", None)
    if type(supported) is not int or not 0 <= supported <= SCHEMA_MAX:
        problems.append("recall.db.schema.SCHEMA_VERSION is not a bounded integer")
    if not problems:
        error_class = importlib.import_module("recall.core.schema_version").SchemaTooNewError
        if not isinstance(error_class, type) or not issubclass(error_class, Exception):
            problems.append("SchemaTooNewError is not an exception")
        else:
            error = error_class(SCHEMA_MAX, 0)
            if (
                getattr(error, "database_version", None),
                getattr(error, "supported_version", None),
            ) != (SCHEMA_MAX, 0):
                problems.append("SchemaTooNewError lacks version attributes")
    return problems


def _audit(connection: Any) -> dict[str, object]:
    result: dict[str, object] = {}
    for table, order in (("schema_version", "version"), ("schema_migrations", "migration_id")):
        exists = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main' "
            "AND table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            result[table] = None
            continue
        rows = connection.execute(
            f"SELECT * FROM {table} ORDER BY {order} LIMIT {AUDIT_ROWS_MAX + 1}"
        ).fetchall()
        if len(rows) > AUDIT_ROWS_MAX:
            raise ValueError("migration metadata exceeds 4096 rows")
        result[table] = [
            [str(value) if value is not None else None for value in row] for row in rows
        ]
    return result


def _settings(connection: Any) -> dict[str, object]:
    connection.execute("SET max_temp_directory_size = '64GiB'")
    names = ("memory_limit", "threads", "temp_directory", "max_temp_directory_size")
    return {
        name: connection.execute("SELECT current_setting(?)", [name]).fetchone()[0]
        for name in names
    }


def _error_fields(error: Exception) -> dict[str, object]:
    message = str(error)
    if len(message.encode()) > 4096:
        return {
            "type": type(error).__name__,
            "message": "error message exceeds the 4096-byte diagnostic bound",
            "original_bytes": len(message.encode()),
        }
    return {"type": type(error).__name__, "message": message}


def _offline(root: Path, request: AdmissionRequest) -> None:
    if any(os.path.lexists(root / name) for name in ("recall.sock", "recall.pid")):
        raise ValueError("offline admission found a socket or PID file")
    if request.operation != "inspect" and not (root / "recall.duckdb").is_file():
        raise ValueError("offline admission requires an existing database")


def _admission(request: AdmissionRequest, build: str) -> int:
    schema = importlib.import_module("recall.db.schema")
    connection_module = importlib.import_module("recall.db.connection")
    error_class = importlib.import_module("recall.core.schema_version").SchemaTooNewError
    supported = schema.SCHEMA_VERSION
    root = Path(os.environ["RECALL_DATA_DIR"])
    config = importlib.import_module("recall.core.config").AppConfig.load()
    paths = _config_paths(config)
    fields: dict[str, Any] = {
        "operation": request.operation,
        "build": build,
        "supported_schema": supported,
        "paths": paths,
    }
    status = 0
    try:
        validate_admission_paths(paths, root)
        request.validate_schema(supported)
        _offline(root, request)
        if request.operation == "inspect":
            state: State = "inspected"
        else:
            negative = request.operation.startswith("negative-")
            try:
                if request.operation == "prepare-newer":
                    connection = connection_module.connect_existing(config, read_only=False)
                elif request.operation.endswith("read"):
                    connection = connection_module.connect_readonly(config)
                else:
                    if not negative:
                        # Record what strict connect may migrate on the positive copy.
                        before = connection_module.connect_existing(config, read_only=True)
                        try:
                            fields["schema_before"] = schema.get_schema_version(before)
                            fields["audit_before"] = _audit(before)
                        finally:
                            before.close()
                    connection = connection_module.connect(
                        config, recreate=False, lenient_schema=False
                    )
            except error_class as error:
                fields["error"] = {
                    **_error_fields(error),
                    "database_version": error.database_version,
                    "supported_version": error.supported_version,
                }
                exact = negative and (error.database_version, error.supported_version) == (
                    request.schema,
                    supported,
                )
                state = "refused" if exact else "error"
                status = 0 if exact else 1
            else:
                try:
                    fields["settings"] = _settings(connection)
                    fields["schema_before"] = fields.get(
                        "schema_before", schema.get_schema_version(connection)
                    )
                    if "audit_before" not in fields:
                        fields["audit_before"] = _audit(connection)
                    if request.operation == "prepare-newer":
                        if fields["schema_before"] != supported:
                            raise ValueError(
                                "normalize the control to the selected schema before preparation"
                            )
                        connection.execute("PRAGMA disable_checkpoint_on_shutdown")
                        connection.execute("SET wal_autocheckpoint = '1TB'")
                        assert request.schema is not None
                        schema.set_schema_version_to(connection, request.schema)
                    elif request.operation == "positive-read":
                        report = importlib.import_module(
                            "recall.db.maintenance"
                        ).probe_index_divergence(connection, sample=2, budget_seconds=10)
                        fields["probe"] = report.to_payload()
                    fields["observed_schema"] = schema.get_schema_version(connection)
                    fields["audit_after"] = _audit(connection)
                    state = "prepared" if request.operation == "prepare-newer" else "opened"
                finally:
                    connection.close()
                if state == "prepared":
                    if fields["observed_schema"] != request.schema:
                        raise ValueError("prepared schema differs from request")
                    wal = root / "recall.duckdb.wal"
                    if not wal.is_file() or wal.stat().st_size == 0:
                        raise ValueError("preparation did not retain a nonempty WAL")
                if negative:
                    status = 1  # Preserve the unexpected open as an observation.
        _offline(root, request)
    except Exception as error:
        fields["error"] = _error_fields(error)
        state, status = "error", 1
    try:
        observation = AdmissionObservation(state=state, **fields)
        data = observation.to_json()
    except ValueError as error:
        # Never truncate a large report into a passing observation.
        data = AdmissionObservation(
            request.operation,
            "error",
            build,
            supported,
            paths,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "cause": fields.get("error"),
            },
        ).to_json()
        status = 1
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--usage-log", type=Path, required=True)
    parser.add_argument("--ungated", action="store_true")
    parser.add_argument("--scheduling-trace", action="store_true")
    parser.add_argument("mode", choices=("check", "preflight", "daemon", "cli", "admission"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    request = None
    if args.mode == "admission":
        try:
            request = AdmissionRequest.parse(args.arguments)
        except ValueError as error:
            parser.error(str(error))
        if args.ungated:
            return REFUSED
    # The supervisor writes one byte once it has recorded this process.
    if not args.ungated and sys.stdin.buffer.read(1) != b"g":
        return NOT_RELEASED
    attempts = Attempts(args.attempts)
    _install_guard(attempts)
    _install_stubs(attempts, args.usage_log)
    if args.mode in ("check", "admission"):
        problems = _seam_problems()
        if problems:
            print(json.dumps({"seams": "missing", "problems": problems}), flush=True)
            return SEAM_MISSING
    text, version = _version()
    if version < SUPPORTED_BUILD_FLOOR:
        floor = ".".join(map(str, SUPPORTED_BUILD_FLOOR))
        print(json.dumps({"build": text, "unsupported": f"older than {floor}"}), flush=True)
        return UNSUPPORTED_BUILD
    if args.mode == "check":
        print(json.dumps({"seams": "ok", "build": text}), flush=True)
        return 0
    if args.mode == "preflight":
        print(json.dumps({"build": text, "paths": _config_paths()}), flush=True)
        return 0
    if args.mode == "admission":
        assert request is not None
        problems = _admission_problems()
        if problems:
            print(json.dumps({"seams": "missing", "problems": problems}), flush=True)
            return SEAM_MISSING
        return _admission(request, text)
    if args.scheduling_trace:
        if args.mode != "daemon" or args.ungated:
            return REFUSED
        observer = runpy.run_path(str(Path(__file__).with_name("scheduling_trace.py")))
        observer["install"](Path(os.environ["RECALL_DATA_DIR"]))
    argv = ["daemon"] if args.mode == "daemon" else args.arguments
    if args.mode == "cli":
        refusal = _cli_refusal(argv)
        if refusal is not None:
            attempts.record("refused_call", "cli", {"argv": argv, "reason": refusal})
            print(json.dumps({"refused": refusal}), file=sys.stderr, flush=True)
            return REFUSED
    cli = importlib.import_module("recall.cli")
    sys.argv = [str(Path(sys.executable).with_name("recall")), *argv]
    if args.mode == "daemon":
        config = importlib.import_module("recall.core.config").AppConfig.load()
        if config.daemon.embed:
            observer = runpy.run_path(str(Path(__file__).with_name("embedding_trace.py")))
            backend = config.embedding.backend
            if backend not in ("onnx", "mlx"):
                return SEAM_MISSING
            with observer["observe"](config.data_dir, backend):
                result = cli.main()
        else:
            result = cli.main()
    else:
        result = cli.main()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
