"""Run the build under test's recall with the bench guards installed.

The supervisor runs this file with the build's own interpreter (`-I -B`), so
it imports nothing from the tool: only the standard library and the build's
`recall` package. Before any recall code runs it

- installs an audit hook that refuses to start a service manager
  (`launchctl`, `systemctl`, `crontab`) or another `recall` process, and
  records every refusal;
- replaces the build's service probe, scheduler query and control entry
  points with stubs that report "not installed" or refuse, recording each
  call;
- points the Grok usage harvester at the injected log.

Every replaced entry point is checked first. A build that lacks one, or whose
signature differs, exits with SEAM_MISSING before anything is patched or run,
so a renamed seam can never fall back to the host's service manager or usage
log.

In `cli` mode the call must be one of `ALLOWED_CLI`, judged by the build's own
command parser: `daemon stop` only with `--soft`, and no raw `--params`,
`--fleet*` or `--host` value. Anything else exits REFUSED without running.

A supervised launcher waits for one byte on stdin before anything else, so
its supervisor records it before it does any work; `--ungated` skips that for
a guest's own CLI calls.

Usage: python -I -B launch.py --attempts FILE --usage-log FILE [--ungated] MODE [ARGS...]
MODE is `check` (seams and version only), `preflight` (print the resolved
config's paths), `daemon` (the foreground daemon) or `cli` (one CLI call).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SEAM_MISSING = 3
UNSUPPORTED_BUILD = 4
REFUSED = 5
NOT_RELEASED = 6
# The first release whose service actions are scoped to scheduler records (#217)
# and that has every seam below.
SUPPORTED_BUILD_FLOOR = (0, 40, 5)
SERVICE_MANAGERS = frozenset({"launchctl", "systemctl", "crontab"})
EXEC_EVENTS = frozenset({"subprocess.Popen", "os.posix_spawn", "os.exec", "os.spawn", "os.system"})


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
ALLOWED_CLI = frozenset({"stats", "live", "show", "search", "list", "daemon status", "daemon stop"})
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
    problems: list[str] = []
    for seam in SEAMS:
        try:
            module = importlib.import_module(seam.module)
        except ImportError as error:
            problems.append(f"{seam.module}: {error}")
            continue
        target = getattr(module, seam.attribute, None)
        if not callable(target):
            problems.append(f"{seam.name} is missing")
            continue
        names = tuple(inspect.signature(target).parameters)
        if not set(seam.parameters) <= set(names):
            problems.append(f"{seam.name}{names} lacks parameters {seam.parameters}")
    daemon = sys.modules.get("recall.services.daemon")
    fields = getattr(getattr(daemon, "ServiceView", None), "__dataclass_fields__", {})
    missing = [name for name in SERVICE_VIEW_FIELDS if name not in fields]
    if missing:
        problems.append(f"recall.services.daemon.ServiceView lacks fields {missing}")
    if not callable(getattr(importlib.import_module("recall.cli"), "main", None)):
        problems.append("recall.cli.main is missing")
    return problems


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


def _install_exec_guard(attempts: Attempts) -> None:
    def guard(event: str, arguments: tuple[object, ...]) -> None:
        if event not in EXEC_EVENTS:
            return
        program = _program(event, arguments)
        name = Path(program).name if program else ""
        if name in SERVICE_MANAGERS or name == "recall":
            attempts.record("refused", event, {"program": program})
            raise PermissionError(f"bench launcher refuses to start {program}")

    sys.addaudithook(guard)


def _install_stubs(attempts: Attempts, usage_log: Path) -> None:
    # Typed loosely: the launcher runs against any supported build, not this checkout.
    daemon = importlib.import_module("recall.services.daemon")
    service_view: Callable[..., object] = getattr(daemon, "ServiceView")  # noqa: B009

    def probe_service(config: object, installed_scheduler: object) -> object:
        attempts.record("suppressed", "probe_service", {"installed_scheduler": installed_scheduler})
        return service_view(
            installed_scheduler_record=installed_scheduler,
            scheduler=None,
            installed=False,
            service_data_dir=None,
            measured_at=time.time(),
        )

    def refuse(seam: Seam) -> Callable[..., object]:
        def stub(*args: object, **kwargs: object) -> object:
            attempts.record("suppressed", seam.name, {"args": args, "kwargs": kwargs})
            raise PermissionError(f"bench launcher suppresses {seam.name}")

        return stub

    def grok_log(*, home: object = None) -> Path:
        return usage_log

    for seam in SEAMS:
        match seam.replacement:
            case "probe":
                replacement: Callable[..., object] = probe_service
            case "refuse":
                replacement = refuse(seam)
            case "usage_log":
                replacement = grok_log
        setattr(importlib.import_module(seam.module), seam.attribute, replacement)


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
    if name == "daemon stop" and params.get("soft") is not True:
        return "`recall daemon stop` is allowed only with --soft"
    return None


def _config_paths() -> dict[str, object]:
    config_module = importlib.import_module("recall.core.config")
    config = config_module.AppConfig.load()
    return {
        "cwd": os.getcwd(),
        "data_dir": str(config.data_dir),
        "db_path": str(config.db_path),
        "lock_path": str(config.lock_path),
        "config_path": str(config.config_path),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--usage-log", type=Path, required=True)
    parser.add_argument("--ungated", action="store_true")
    parser.add_argument("mode", choices=("check", "preflight", "daemon", "cli"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # The supervisor writes one byte once it has recorded this process.
    if not args.ungated and sys.stdin.buffer.read(1) != b"g":
        return NOT_RELEASED
    attempts = Attempts(args.attempts)
    _install_exec_guard(attempts)
    problems = _seam_problems()
    if problems:
        print(json.dumps({"seams": "missing", "problems": problems}), flush=True)
        return SEAM_MISSING
    text, version = _version()
    if version < SUPPORTED_BUILD_FLOOR:
        floor = ".".join(map(str, SUPPORTED_BUILD_FLOOR))
        print(json.dumps({"build": text, "unsupported": f"older than {floor}"}), flush=True)
        return UNSUPPORTED_BUILD
    _install_stubs(attempts, args.usage_log)
    if args.mode == "check":
        print(json.dumps({"seams": "ok", "build": text}), flush=True)
        return 0
    if args.mode == "preflight":
        print(json.dumps({"build": text, "paths": _config_paths()}), flush=True)
        return 0
    argv = ["daemon"] if args.mode == "daemon" else args.arguments
    if args.mode == "cli":
        refusal = _cli_refusal(argv)
        if refusal is not None:
            attempts.record("refused_call", "cli", {"argv": argv, "reason": refusal})
            print(json.dumps({"refused": refusal}), file=sys.stderr, flush=True)
            return REFUSED
    cli = importlib.import_module("recall.cli")
    sys.argv = [str(Path(sys.executable).with_name("recall")), *argv]
    result = cli.main()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
