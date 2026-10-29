"""Scheduler-owned identity and mutation authority for per-user daemon services."""

from __future__ import annotations

import os
import plistlib
import pwd
import re
import selectors
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.parsers.expat import ExpatError

from recall.core.config import default_data_dir_for
from recall.core.types import DaemonMode, SchedulerKind

LAUNCHD_LABEL = "it.send.recall.daemon"
LEGACY_LAUNCHD_LABEL = "xyz.metalrodeo.recall.daemon"
CRON_BEGIN_MARKER = "# BEGIN recall daemon"
CRON_END_MARKER = "# END recall daemon"
RECORD_BYTES_MAX = 1024 * 1024
QUERY_TIMEOUT_SECONDS = 5


class ServiceLookupError(RuntimeError):
    """The manager did not establish either identity or absence."""

    def __init__(self, scheduler: SchedulerKind, reason: str) -> None:
        self.scheduler = scheduler
        super().__init__(reason)


@dataclass(frozen=True)
class Service:
    scheduler: SchedulerKind
    data_dir: Path
    effective_data_dir: Path
    name: str
    path: Path | None
    command: str | None
    mode: DaemonMode | None
    loaded: bool = False
    pid: int | None = None


@dataclass(frozen=True)
class SchedulerRecord:
    scheduler: SchedulerKind
    services: tuple[Service, ...]
    paths: tuple[Path, ...]
    timer: bool = False
    crontab: str | None = None
    environment: tuple[tuple[str, str], ...] = ()
    needs_reload: bool = False

    @property
    def service(self) -> Service | None:
        return next(
            (service for service in self.services if service.loaded),
            self.services[0] if self.services else None,
        )


@dataclass(frozen=True)
class ArtifactSnapshot:
    path: Path
    content: bytes | None


@dataclass(frozen=True)
class ServiceCapability:
    """A record checked against the selected data dir, carried through recovery."""

    record: SchedulerRecord

    def snapshot_artifacts(self, data_dir: Path) -> tuple[ArtifactSnapshot, ...]:
        if self.record.needs_reload:
            raise RuntimeError(
                "cannot replace service artifacts: the manager requires daemon-reload"
            )
        try:
            snapshots = tuple(ArtifactSnapshot(path, _read(path)) for path in self.record.paths)
            if self.record.scheduler == SchedulerKind.LAUNCHD:
                for label, snapshot in zip(
                    (LAUNCHD_LABEL, LEGACY_LAUNCHD_LABEL), snapshots, strict=True
                ):
                    if snapshot.content is None:
                        if any(
                            service.name == label and service.loaded
                            for service in self.record.services
                        ):
                            raise ValueError(
                                f"loaded job has no recoverable plist: {snapshot.path}"
                            )
                    else:
                        service = _parse_plist_service(
                            snapshot.content, snapshot.path, label, dict(self.record.environment)
                        )
                        if service.effective_data_dir != data_dir.resolve():
                            raise ValueError(
                                f"plist {snapshot.path} runs another data dir: {service.data_dir}"
                            )
            return snapshots
        except (
            OSError,
            ValueError,
            plistlib.InvalidFileException,
            ExpatError,
            RecursionError,
        ) as err:
            raise RuntimeError(f"cannot validate scheduler artifacts: {err}") from err


Query = Callable[[list[str]], subprocess.CompletedProcess[str]]


def query(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Capture bounded scheduler output, terminating and reaping on every failure."""
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
        assert child.stdout is not None and child.stderr is not None
        output = {child.stdout.fileno(): bytearray(), child.stderr.fileno(): bytearray()}
        deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
        try:
            with selectors.DefaultSelector() as selector:
                for pipe in output:
                    selector.register(pipe, selectors.EVENT_READ)
                size = 0
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("scheduler query timed out")
                    for key, _ in selector.select(remaining):
                        pipe = key.fd
                        assert pipe in output
                        block = os.read(pipe, min(65536, RECORD_BYTES_MAX + 1 - size))
                        if not block:
                            selector.unregister(pipe)
                        else:
                            size += len(block)
                            if size > RECORD_BYTES_MAX:
                                raise ValueError("scheduler output exceeds 1 MiB")
                            output[pipe].extend(block)
            child.wait(timeout=max(0.001, deadline - time.monotonic()))
            return subprocess.CompletedProcess(
                args,
                child.returncode,
                output[child.stdout.fileno()].decode("utf-8"),
                output[child.stderr.fileno()].decode("utf-8"),
            )
        except BaseException:
            # This child belongs to this query; no process-table selection is used.
            if child.poll() is None:
                child.kill()
            child.wait()
            raise


def _ask(scheduler: SchedulerKind, args: list[str], run: Query) -> subprocess.CompletedProcess[str]:
    try:
        result = run(args)
    except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired) as err:
        raise ServiceLookupError(scheduler, str(err)) from err
    if len(result.stdout.encode()) + len(result.stderr.encode()) > RECORD_BYTES_MAX:
        raise ServiceLookupError(scheduler, "scheduler output exceeds 1 MiB")
    return result


def _failed(
    scheduler: SchedulerKind, result: subprocess.CompletedProcess[str]
) -> ServiceLookupError:
    return ServiceLookupError(scheduler, result.stderr.strip() or "unreadable scheduler record")


def account_home() -> Path:
    path = Path(pwd.getpwuid(os.getuid()).pw_dir)
    if not path.is_absolute():
        raise ValueError("OS account home is not absolute")
    return path


def _data_dir(environment: dict[str, str], cwd: Path) -> tuple[Path, Path]:
    if "RECALL_DATA_DIR" in environment:
        literal = Path(environment["RECALL_DATA_DIR"])
    else:
        home = environment.get("HOME")
        if not home or not Path(home).is_absolute():
            raise ValueError("effective scheduler HOME is unavailable")
        literal = default_data_dir_for(Path(home))
    # Tilde is an ordinary character in scheduler environment values.
    effective = literal if literal.is_absolute() else cwd / literal
    return literal, effective.resolve()


def _scheduler_words(text: str) -> list[str]:
    """Read systemctl's shell quoting and C-escaped control bytes without expansion."""
    words: list[str] = []
    word = bytearray()
    quote: str | None = None
    started = False
    index = 0
    controls = {"a": 7, "b": 8, "f": 12, "n": 10, "r": 13, "t": 9, "v": 11}
    while index < len(text):
        character = text[index]
        index += 1
        if quote is None and character.isspace():
            if started:
                words.append(word.decode("utf-8"))
                word.clear()
                started = False
            continue
        started = True
        if character in {"'", '"'}:
            if quote is None:
                quote = character
                continue
            if quote == character:
                quote = None
                continue
        if character == "\\" and quote != "'":
            if index >= len(text):
                raise ValueError("truncated scheduler escape")
            escaped = text[index]
            index += 1
            if escaped in controls:
                word.append(controls[escaped])
            elif escaped == "x":
                digits = text[index : index + 2]
                if len(digits) != 2 or any(char not in "0123456789abcdefABCDEF" for char in digits):
                    raise ValueError("invalid scheduler hex escape")
                word.append(int(digits, 16))
                index += 2
            elif escaped in "01234567":
                digits = escaped + text[index : index + 2]
                if len(digits) != 3 or any(char not in "01234567" for char in digits):
                    raise ValueError("invalid scheduler octal escape")
                value = int(digits, 8)
                if value > 255:
                    raise ValueError("scheduler octal escape exceeds one byte")
                word.append(value)
                index += 2
            elif escaped in "\\\"'$` ":
                word.extend(escaped.encode("utf-8"))
            else:
                raise ValueError("unsupported scheduler escape")
        else:
            word.extend(character.encode("utf-8"))
    if quote is not None:
        raise ValueError("unclosed scheduler quote")
    if started:
        words.append(word.decode("utf-8"))
    return words


def _environment_assignments(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for word in _scheduler_words(text):
        key, sep, value = word.partition("=")
        if not sep or not key.isidentifier():
            raise ValueError("malformed scheduler environment")
        result[key] = value
    return result


def _validate_command(command: str) -> None:
    words = shlex.split(command)
    while words and "=" in words[0]:
        words.pop(0)
    if not words or Path(words[0]).name in {"env", "sh", "bash", "zsh", "dash", "fish"}:
        raise ValueError("scheduler command has unresolved environment indirection")


def parse_systemd_properties(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if not sep or key in result:
            raise ValueError("malformed or duplicate scheduler property")
        result[key] = value
    return result


def _absolute(value: str, name: str) -> Path:
    if not value or not Path(value).is_absolute():
        raise ValueError(f"scheduler {name} is not an absolute path")
    return Path(value)


def _read(path: Path) -> bytes | None:
    try:
        with path.open("rb") as stream:
            value = stream.read(RECORD_BYTES_MAX + 1)
    except FileNotFoundError:
        return None
    if len(value) > RECORD_BYTES_MAX:
        raise ValueError(f"scheduler artifact exceeds 1 MiB: {path}")
    return value


def discover(scheduler: SchedulerKind, *, run: Query = query) -> SchedulerRecord:
    try:
        match scheduler:
            case SchedulerKind.SYSTEMD:
                return _systemd(run)
            case SchedulerKind.LAUNCHD:
                return _launchd(run)
            case SchedulerKind.CRON:
                return _cron(run)
            case _:
                raise ValueError(f"unsupported scheduler {scheduler}")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        plistlib.InvalidFileException,
        ExpatError,
        RecursionError,
    ) as err:
        raise ServiceLookupError(scheduler, str(err)) from err


def _systemd(run: Query) -> SchedulerRecord:
    kind = SchedulerKind.SYSTEMD
    fields = [
        "LoadState",
        "FragmentPath",
        "Environment",
        "EnvironmentFiles",
        "UnsetEnvironment",
        "Type",
        "WorkingDirectory",
        "ExecStart",
        "MainPID",
        "ActiveState",
        "NeedDaemonReload",
    ]

    def show(unit: str, properties: list[str]) -> dict[str, str]:
        result = _ask(
            kind,
            [
                "systemctl",
                "--user",
                "show",
                unit,
                *[f"--property={field}" for field in properties],
                "--no-pager",
            ],
            run,
        )
        values = parse_systemd_properties(result.stdout)
        if values.get("LoadState") == "not-found":
            return values
        if result.returncode or values.get("LoadState") != "loaded":
            raise _failed(kind, result)
        for field in properties:
            if field not in values:
                raise ValueError(f"missing systemd property {field}")
        return values

    service = show("recall-daemon.service", fields)
    timer = show(
        "recall-daemon.timer", ["LoadState", "FragmentPath", "Triggers", "NeedDaemonReload"]
    )
    has_timer = timer["LoadState"] == "loaded"
    if has_timer and timer["Triggers"].split() != ["recall-daemon.service"]:
        raise ValueError("recall timer does not target exactly recall-daemon.service")
    result = _ask(kind, ["systemctl", "--user", "show-environment"], run)
    if result.returncode:
        raise _failed(kind, result)
    environment = _environment_assignments(result.stdout)
    home = environment.get("HOME")
    config_home = environment.get("XDG_CONFIG_HOME")
    directory = (
        _absolute(config_home, "XDG_CONFIG_HOME")
        if config_home
        else _absolute(home, "HOME") / ".config"
        if home
        else account_home() / ".config"
    )
    directory /= "systemd/user"
    service_path = (
        _absolute(service["FragmentPath"], "FragmentPath")
        if service["LoadState"] == "loaded"
        else directory / "recall-daemon.service"
    )
    timer_path = (
        _absolute(timer["FragmentPath"], "timer FragmentPath")
        if has_timer
        else directory / "recall-daemon.timer"
    )
    paths = (service_path, timer_path)
    if service["LoadState"] == "not-found":
        if has_timer:
            raise ValueError("recall timer has no identifiable service")
        if any(_read(path) is not None for path in paths):
            raise ValueError("unit artifact is not identified by the manager; reload it first")
        return SchedulerRecord(kind, (), paths)
    if not has_timer and _read(timer_path) is not None:
        raise ValueError("timer artifact is not identified by the manager; reload it first")
    if service["EnvironmentFiles"]:
        raise ValueError("effective EnvironmentFile values cannot be identified")
    environment.update(_environment_assignments(service["Environment"]))
    for assignment in _scheduler_words(service["UnsetEnvironment"]):
        key, sep, value = assignment.partition("=")
        if not sep or environment.get(key) == value:
            environment.pop(key, None)
    cwd = service["WorkingDirectory"]
    working = account_home() if cwd in {"", "~"} else _absolute(cwd, "working directory")
    literal, effective = _data_dir(environment, working)
    mode = DaemonMode.POLL if service["Type"] == "oneshot" else DaemonMode.WATCH
    command_match = re.search(r"argv\[\]=(.*?)(?: ;|$)", service["ExecStart"])
    if command_match is None or service["ExecStart"].count("argv[]=") != 1:
        raise ValueError("systemd record does not identify one executable command")
    command = command_match[1].strip()
    _validate_command(command)
    pid = int(service["MainPID"] or "0")
    record = Service(
        kind,
        literal,
        effective,
        "recall-daemon.service",
        service_path,
        command,
        mode,
        service["ActiveState"] in {"active", "activating"},
        pid or None,
    )
    return SchedulerRecord(
        kind,
        (record,),
        paths,
        timer=has_timer,
        needs_reload=service["NeedDaemonReload"] != "no"
        or (has_timer and timer["NeedDaemonReload"] != "no"),
    )


@dataclass(frozen=True)
class LaunchdPrint:
    fields: dict[str, str]
    blocks: dict[str, tuple[str, ...]]


def _launchd_print(text: str, target: str) -> LaunchdPrint:
    """Parse the named top-level fields without treating nested jobs as this job."""
    if not text.lstrip().startswith(target + " = {\n"):
        raise ValueError("launchctl record does not identify the requested target")
    fields: dict[str, str] = {}
    blocks: dict[str, tuple[str, ...]] = {}
    stack: list[str] = []
    content: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith("{"):
            name = line[:-1].strip().removesuffix("=").strip()
            stack.append(name)
            if len(stack) > 32:
                raise ValueError("launchctl nesting exceeds 32")
            if len(stack) == 2:
                content = []
        elif line == "}":
            if not stack:
                raise ValueError("unbalanced launchctl record")
            if len(stack) == 2:
                name = stack[-1]
                if name in blocks:
                    raise ValueError("duplicate launchctl block")
                blocks[name] = tuple(content)
            stack.pop()
        elif len(stack) == 2:
            content.append(line)
        elif len(stack) == 1:
            key, sep, value = line.partition(" = ")
            if sep:
                if key in fields:
                    raise ValueError("duplicate launchctl field")
                fields[key] = value
    if stack or (not fields and not blocks):
        raise ValueError("unreadable launchctl print record")
    return LaunchdPrint(fields, blocks)


def _launchd_environment(record: LaunchdPrint) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("default environment", "inherited environment", "environment"):
        for line in record.blocks.get(name, ()):
            key, sep, value = line.partition(" => ")
            if not sep or not key.isidentifier():
                raise ValueError("malformed launchctl environment")
            result[key] = value
    return result


def _plist_service(path: Path, label: str, inherited: dict[str, str]) -> Service | None:
    content = _read(path)
    if content is None:
        return None
    return _parse_plist_service(content, path, label, inherited)


def _parse_plist_service(
    content: bytes, path: Path, label: str, inherited: dict[str, str]
) -> Service:
    value = plistlib.loads(content)
    if not isinstance(value, dict) or value.get("Label") != label:
        raise ValueError(f"invalid launchd plist identity: {path}")
    args = value.get("ProgramArguments", [value.get("Program")])
    environment = value.get("EnvironmentVariables", {})
    if (
        not isinstance(args, list)
        or not args
        or not all(isinstance(arg, str) for arg in args)
        or not isinstance(environment, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items())
    ):
        raise ValueError(f"invalid launchd plist program/environment: {path}")
    working = _absolute(value.get("WorkingDirectory", "/"), "WorkingDirectory")
    literal, effective = _data_dir(inherited | environment, working)
    _validate_command(shlex.join(args))
    mode = DaemonMode.WATCH if value.get("KeepAlive") else DaemonMode.POLL
    return Service(SchedulerKind.LAUNCHD, literal, effective, label, path, shlex.join(args), mode)


def _launchd(run: Query) -> SchedulerRecord:
    kind = SchedulerKind.LAUNCHD
    domain = f"gui/{os.getuid()}"
    result = _ask(kind, ["launchctl", "print", domain], run)
    if result.returncode:
        raise _failed(kind, result)
    inherited = _launchd_environment(_launchd_print(result.stdout, domain))
    directory = account_home() / "Library/LaunchAgents"
    paths = tuple(directory / f"{label}.plist" for label in (LAUNCHD_LABEL, LEGACY_LAUNCHD_LABEL))
    services: list[Service] = []
    absent: list[str] = []
    for label in (LAUNCHD_LABEL, LEGACY_LAUNCHD_LABEL):
        result = _ask(kind, ["launchctl", "print", f"{domain}/{label}"], run)
        if result.returncode:
            if f'Could not find service "{label}"' not in result.stderr:
                raise _failed(kind, result)
            absent.append(label)
            continue
        record = _launchd_print(result.stdout, f"{domain}/{label}")
        path = _absolute(record.fields["path"], "launchd path")
        environment = inherited | _launchd_environment(record)
        literal, effective = _data_dir(
            environment, _absolute(record.fields.get("working directory", "/"), "working directory")
        )
        args = record.blocks.get("arguments", ())
        command = shlex.join(args) if args else record.fields.get("program")
        if not command:
            raise ValueError("launchctl record has no program")
        _validate_command(command)
        mode = DaemonMode.WATCH if "--once" not in shlex.split(command) else DaemonMode.POLL
        pid = int(record.fields.get("pid", "0"))
        services.append(
            Service(kind, literal, effective, label, path, command, mode, True, pid or None)
        )
    if len(absent) == 2:
        services = [
            service
            for path, label in zip(paths, absent, strict=True)
            if (service := _plist_service(path, label, inherited)) is not None
        ]
    else:
        for path in paths:
            if path.stem in absent:
                stopped = _plist_service(path, path.stem, inherited)
                if stopped is not None:
                    services.append(stopped)
        services.sort(key=lambda service: service.name != LAUNCHD_LABEL)
    verified_paths: list[Path] = []
    for path in paths:
        service = next((service for service in services if service.name == path.stem), None)
        if service is not None:
            assert service.path is not None
            path = service.path
        verified_paths.append(path)
    return SchedulerRecord(
        kind, tuple(services), tuple(verified_paths), environment=tuple(inherited.items())
    )


def _cron(run: Query) -> SchedulerRecord:
    kind = SchedulerKind.CRON
    result = _ask(kind, ["crontab", "-l"], run)
    if result.returncode:
        if result.returncode == 1 and "no crontab for " in result.stderr.lower():
            return SchedulerRecord(kind, (), (), crontab="")
        raise _failed(kind, result)
    lines = result.stdout.splitlines()
    begin = [i for i, line in enumerate(lines) if line.strip() == CRON_BEGIN_MARKER]
    end = [i for i, line in enumerate(lines) if line.strip() == CRON_END_MARKER]
    if begin or end:
        if len(begin) != 1 or len(end) != 1 or begin[0] >= end[0]:
            raise ValueError("ambiguous recall crontab block")
        entries = [
            line.strip()
            for line in lines[begin[0] + 1 : end[0]]
            if line.strip()
            and not line.strip().startswith("#")
            and _CRON_ENVIRONMENT_LINE.fullmatch(line.strip()) is None
        ]
        if len(entries) > 1:
            raise ValueError("multiple jobs in recall crontab block")
        for entry in entries:
            shlex.split(entry)
    job = cron_job(result.stdout)
    if job is None:
        return SchedulerRecord(kind, (), (), crontab=result.stdout)
    if job.command is not None:
        _validate_command(job.command)
    environment = {"HOME": str(account_home())} | job.environment
    cwd = _absolute(environment["HOME"], "cron HOME")
    literal, effective = _data_dir(environment, cwd)
    service = Service(kind, literal, effective, "recall daemon", None, job.command, DaemonMode.POLL)
    return SchedulerRecord(kind, (service,), (), crontab=result.stdout)


_CRON_ENVIRONMENT_LINE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)")


@dataclass(frozen=True)
class CronJob:
    """The recall block of a crontab, read as cron runs it."""

    environment: dict[str, str]
    """What cron gives the job: earlier `NAME = value` lines, then the entry's own prefix."""
    command: str | None
    """The entry line after its schedule, prefix included; None when the block has no entry."""


def cron_job(crontab: str) -> CronJob | None:
    """The recall block's job; None when the crontab has no recall block.

    Cron applies each `NAME = value` line to the entries after it, so only
    assignments before the job count. The job is the block's first entry line;
    comments and blank lines are skipped. A block with no entry ends at its
    END marker, and later lines do not apply to it.
    """
    environment: dict[str, str] = {}
    inside_block = False
    for line in crontab.splitlines():
        stripped = line.strip()
        if stripped == CRON_BEGIN_MARKER:
            inside_block = True
            continue
        if stripped == CRON_END_MARKER and inside_block:
            return CronJob(environment, None)
        if not stripped or stripped.startswith("#"):
            continue
        assignment = _CRON_ENVIRONMENT_LINE.fullmatch(stripped)
        if assignment is not None:
            environment[assignment[1]] = _unquote_cron_value(assignment[2])
            continue
        if inside_block:
            return _cron_entry(environment, stripped)
    return CronJob(environment, None) if inside_block else None


def _cron_entry(environment: dict[str, str], line: str) -> CronJob:
    # Five schedule fields, or one `@` nickname, then the prefix and the command.
    schedule_fields = 1 if line.startswith("@") else 5
    words = (_shell_words(line) or [])[schedule_fields:]
    command = " ".join(shlex.quote(word) for word in words) if words else None
    return CronJob(environment | _assignments(words), command)


def _unquote_cron_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _shell_words(text: str) -> list[str] | None:
    """`text` split as the shell would; None when its quoting is unbalanced."""
    try:
        return shlex.split(text)
    except ValueError:
        return None


def _assignments(words: list[str]) -> dict[str, str]:
    """The leading `KEY=value` words."""
    environment: dict[str, str] = {}
    for word in words:
        key, separator, value = word.partition("=")
        if separator != "=" or not key.isidentifier():
            break
        environment[key] = value
    return environment
