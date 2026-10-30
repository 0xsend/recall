"""Drive `bench.py daemon-run` with real processes, on the `with_bench_lock.py -- bench.py` shape.

The build under test is this checkout, against a template prepared from a
synthetic backup (a freshly initialized recall database whose runtime state
records an installed launchd or systemd scheduler) and the transcript fixtures.
A second, fake build (a virtualenv holding a stand-in `recall` package) gives
renamed seams, an unsupported version, and a daemon that exits leaving a
descendant behind.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import duckdb
import psutil
import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "scripts/bench.py"
LOCK = ROOT / "scripts/with_bench_lock.py"
FIXTURES = ROOT / "tests/fixtures"
SOURCES = ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
NATIVE = "launchd" if sys.platform == "darwin" else "systemd"
FOREIGN = "systemd" if sys.platform == "darwin" else "launchd"
INJECTED_SESSION = "grok-injected-session"
HOST_SESSION = "grok-host-session"
SERVICE_TOOLS = ("recall", "launchctl", "systemctl", "crontab")
EVENT_SECONDS = 120
RUN_SECONDS = 300


def grok_event(session: str) -> str:
    return json.dumps(
        {
            "msg": "shell.turn.inference_done",
            "sid": session,
            "ts": "2026-10-30T00:00:00Z",
            "ctx": {"prompt_tokens": 11, "completion_tokens": 7, "loop_index": 0},
        }
    )


@dataclass(frozen=True)
class Lane:
    root: Path
    canary_log: Path
    environment: dict[str, str]

    def new(self, name: str) -> Path:
        path = self.root / name
        assert not path.exists()
        return path


@pytest.fixture(scope="module")
def lane(tmp_path_factory: pytest.TempPathFactory) -> Lane:
    """A short write root, and canaries first on the tool's own PATH."""
    root = tmp_path_factory.mktemp("bd")
    canary_bin = root / "canary-bin"
    canary_bin.mkdir()
    log = root / "canary-invocations"
    for name in SERVICE_TOOLS:
        script = canary_bin / name
        script.write_text(f"#!/bin/sh\necho \"$0 $*\" >> '{log}'\nexit 97\n")
        script.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{canary_bin}:{environment['PATH']}"
    environment["BENCH_WRITE_ROOTS"] = str(root)
    # A planted host usage log: an injected harvester must never read it.
    host_log = Path(environment["HOME"]) / ".grok/logs/unified.jsonl"
    host_log.parent.mkdir(parents=True, exist_ok=True)
    host_log.write_text(grok_event(HOST_SESSION) + "\n")
    return Lane(root, log, environment)


def synthetic_backup(directory: Path, scheduler: str) -> Path:
    """A fresh recall database, made by the build, recording an installed scheduler."""
    work, backup = directory / "work", directory / "backup"
    work.mkdir(parents=True)
    backup.mkdir()
    script = textwrap.dedent(
        """
        import sys
        from recall.core.config import AppConfig
        from recall.core.types import SchedulerKind
        from recall.db.connection import connect
        from recall.services.runtime_state import set_installed_scheduler
        connection = connect(AppConfig.load())
        set_installed_scheduler(connection, SchedulerKind(sys.argv[1]))
        connection.execute("CHECKPOINT")
        connection.close()
        """
    )
    environment = dict(os.environ)
    environment.update(
        RECALL_DATA_DIR=str(work),
        RECALL_CONFIG_PATH=str(work / "absent.toml"),
        RECALL_DB_PATH=str(work / "recall.duckdb"),
        RECALL_LOCK_PATH=str(work / "recall.lock"),
    )
    subprocess.run(
        [sys.executable, "-c", script, scheduler], env=environment, check=True, timeout=120
    )
    (work / "recall.duckdb").rename(backup / "recall.duckdb")
    return backup


def bench(lane: Lane, *args: str | Path, env: dict[str, str] | None = None) -> tuple[int, dict]:
    completed = subprocess.run(
        [sys.executable, str(BENCH), *map(str, args)],
        capture_output=True,
        text=True,
        env=lane.environment if env is None else env,
        timeout=RUN_SECONDS,
        check=False,
    )
    lines = completed.stdout.strip().splitlines()
    assert lines, completed.stderr
    return completed.returncode, json.loads(lines[-1])


def prepare(lane: Lane, name: str, scheduler: str) -> Path:
    directory = lane.new(f"{name}-src")
    backup = synthetic_backup(directory, scheduler)
    pool = directory / "pool"
    for source in SOURCES:
        shutil.copytree(FIXTURES / source, pool / source)
    usage = directory / "usage.jsonl"
    usage.write_text(grok_event(INJECTED_SESSION) + "\n")
    template = lane.new(name)
    status, line = bench(
        lane,
        "prepare",
        "--backup", backup,
        "--template", template,
        "--source-pool", pool,
        "--usage-log", usage,
    )  # fmt: skip
    assert (status, line["outcome"]) == (0, "passed"), line
    return template


@pytest.fixture(scope="module")
def native_template(lane: Lane) -> Path:
    return prepare(lane, "tn", NATIVE)


@pytest.fixture(scope="module")
def foreign_template(lane: Lane) -> Path:
    return prepare(lane, "tf", FOREIGN)


class Locked:
    """`with_bench_lock.py -- bench.py daemon-run ...`, read event by event."""

    def __init__(self, lane: Lane, queue: Path, *args: str | Path) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable, str(LOCK), "--lock", str(queue), "bench-test", "--",
                sys.executable, str(BENCH), "daemon-run", *map(str, args),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=lane.environment,
        )  # fmt: skip
        self.events: list[dict] = []

    def until(self, predicate: dict) -> dict:
        """Read stderr until an event containing every field of `predicate`."""
        deadline = time.monotonic() + EVENT_SECONDS
        assert self.process.stderr is not None
        while time.monotonic() < deadline:
            raw = self.process.stderr.readline()
            assert raw, f"stderr ended before {predicate}; saw {self.events[-5:]}"
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            self.events.append(event)
            if all(event.get(key) == value for key, value in predicate.items()):
                return event
        raise AssertionError(f"no {predicate} within {EVENT_SECONDS} s")

    def finish(self) -> tuple[int, dict]:
        stdout, stderr = self.process.communicate(timeout=RUN_SECONDS)
        for raw in stderr.splitlines():
            try:
                self.events.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        lines = stdout.strip().splitlines()
        assert lines, stderr
        return self.process.returncode, json.loads(lines[-1])

    def signal(self, signum: int) -> None:
        # The lock runner is this test's own child.
        self.process.send_signal(signum)

    def bench_pid(self) -> int:
        return next(event["pid"] for event in self.events if event.get("event") == "bench")


@pytest.fixture
def queue(tmp_path: Path) -> Path:
    return tmp_path / "queue"


def registry(output: Path) -> list[dict]:
    return [json.loads(line) for line in (output / "processes.jsonl").read_text().splitlines()]


def assert_every_group_gone(output: Path) -> None:
    lines = registry(output)
    started = {line["pid"]: line for line in lines if line["event"] == "started"}
    stopped = {line["pid"]: line for line in lines if line["event"] == "stopped"}
    assert started, lines
    assert stopped.keys() == started.keys()
    assert all(line["group_gone"] for line in stopped.values()), stopped
    for pid, line in started.items():
        # A reused PID belongs to another process: its start time differs.
        with suppress(psutil.NoSuchProcess):
            assert psutil.Process(pid).create_time() != line["create_time"], line


def assert_lock_released(lane: Lane, queue: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(LOCK), "--lock", str(queue), "--timeout", "10", "probe", "--", "true"],
        capture_output=True,
        text=True,
        env=lane.environment,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def assert_no_service_reach(lane: Lane, result: dict, run: Path) -> None:
    assert not lane.canary_log.exists(), lane.canary_log.read_text()
    assert not (run / "bench-canary.log").exists(), (run / "bench-canary.log").read_text()
    reach = result["scope"]["service_reach"]
    assert (reach["canary_invocations"], reach["refused"], reach["refused_signals"]) == ([], [], [])


def assert_home_untouched(lane: Lane) -> None:
    home = Path(lane.environment["HOME"])
    for relative in (
        ".local/share/recall",
        ".config/recall",
        ".local/state/recall",
        ".cache/recall",
    ):
        assert not (home / relative).exists(), relative


def run_args(template: Path, output: Path, run: Path, *extra: str) -> list[str | Path]:
    return [
        "--template", template, "--build", ROOT, "--output", output, "--run-dir", run,
        "--profile", "idle", *extra,
    ]  # fmt: skip


def snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


def test_an_idle_run_passes_from_the_owned_daemon_with_the_injected_usage_log(
    lane: Lane, native_template: Path, queue: Path
) -> None:
    output, run = lane.new("o-idle"), lane.new("r-idle")
    before = snapshot(native_template)
    locked = Locked(lane, queue, *run_args(native_template, output, run, "--window", "32",
                                           "--usage-log", "template"))  # fmt: skip
    status, result = locked.finish()

    assert (status, result["outcome"]) == (0, "passed"), result
    steps = {step["step"]: step for step in result["steps"]}
    daemon = next(line for line in registry(output) if line.get("name") == "daemon")
    assert steps["readiness"]["daemon_pid"] == daemon["pid"]
    assert steps["readiness"]["installed_scheduler"] == NATIVE
    assert steps["settle"]["pending"] == 0
    assert steps["settle"]["stats"]["sessions"] > 0
    assert steps["final"]["stats"]["sessions"] == steps["settle"]["stats"]["sessions"]
    assert steps["window"]["samples"] >= 2 and steps["window"]["rss_bytes_max"] > 0
    assert json.loads((output / "result.json").read_text()) == result
    # The persisted record made status probe the native service; the stub answered.
    assert result["scope"]["service_reach"]["suppressed"].get("probe_service", 0) > 0
    assert_no_service_reach(lane, result, run)
    assert_every_group_gone(output)
    assert_lock_released(lane, queue)
    with duckdb.connect(str(run / "recall.duckdb"), read_only=True) as database:
        cursors = database.execute("SELECT path FROM usage_log_cursors").fetchall()
        sessions = database.execute("SELECT source_session_id FROM usage_events").fetchall()
    assert cursors == [(str(run / "usage.jsonl"),)]
    assert sessions == [(INJECTED_SESSION,)]
    assert snapshot(native_template) == before
    assert_home_untouched(lane)


def test_term_while_the_daemon_starts_stops_every_group_and_releases_the_lock(
    lane: Lane, native_template: Path, queue: Path
) -> None:
    output, run = lane.new("o-term"), lane.new("r-term")
    locked = Locked(lane, queue, *run_args(native_template, output, run))
    locked.until({"event": "phase", "phase": "readiness"})
    locked.signal(signal.SIGTERM)
    status, result = locked.finish()

    assert (status, result["outcome"]) == (143, "incomplete"), result
    assert result["error"] == {"kind": "cancelled", "signal": signal.SIGTERM}
    assert "window" not in {step["step"] for step in result["steps"]}
    assert json.loads((output / "result.json").read_text()) == result
    assert_every_group_gone(output)
    assert_no_service_reach(lane, result, run)
    assert_lock_released(lane, queue)


def test_a_second_term_during_cleanup_does_not_cut_it_short(
    lane: Lane, native_template: Path, queue: Path
) -> None:
    output, run = lane.new("o-term2"), lane.new("r-term2")
    locked = Locked(lane, queue, *run_args(native_template, output, run))
    locked.until({"event": "phase", "phase": "readiness"})
    locked.signal(signal.SIGTERM)
    locked.until({"event": "cancelled"})
    os.kill(locked.bench_pid(), signal.SIGTERM)
    locked.signal(signal.SIGTERM)
    status, result = locked.finish()

    assert (status, result["outcome"]) == (143, "incomplete"), result
    assert result["error"] == {"kind": "cancelled", "signal": signal.SIGTERM}
    assert_every_group_gone(output)
    assert_lock_released(lane, queue)


def test_a_run_dir_replaced_while_the_daemon_runs_fails_the_next_check_and_stops_it(
    lane: Lane, native_template: Path, queue: Path
) -> None:
    output, run = lane.new("o-swap"), lane.new("r-swap")
    locked = Locked(lane, queue, *run_args(native_template, output, run, "--window", "60"))
    locked.until({"event": "phase", "phase": "window"})
    # A careless cleanup moves the run away and recreates a directory at its name.
    run.rename(run.with_name("r-swap-moved"))
    run.mkdir()
    status, result = locked.finish()

    assert (status, result["outcome"]) == (1, "failed"), result
    assert result["error"]["reason"] == "is no longer the directory created for this run"
    assert_every_group_gone(output)
    assert_lock_released(lane, queue)


def test_a_hold_that_times_out_is_incomplete_and_still_cleans_up(
    lane: Lane, foreign_template: Path, queue: Path
) -> None:
    output, run = lane.new("o-hold"), lane.new("r-hold")
    locked = Locked(lane, queue, *run_args(foreign_template, output, run, "--window", "2",
                                           "--hold", "--hold-timeout", "2"))  # fmt: skip
    status, result = locked.finish()

    assert (status, result["outcome"]) == (2, "incomplete"), result
    assert result["error"]["kind"] == "hold_timeout"
    guest = (output / "guest.env").read_text()
    assert "Never use `pkill`, `killall`," in guest
    assert "never `daemon stop|restart|install|uninstall`" in guest
    assert "`recall daemon stop` is refused here, `--soft` included" in guest
    assert_every_group_gone(output)
    assert_no_service_reach(lane, result, run)


REFUSED_CALLS = {
    "soft stop": ["daemon", "stop", "--soft", "--json"],
    "durable stop": ["daemon", "stop", "--json"],
    "stop with --soft as a field value": ["daemon", "stop", "--fields", "--soft", "--json"],
    "restart": ["daemon", "restart"],
    "install": ["daemon", "install"],
    "uninstall": ["daemon", "uninstall"],
    "start": ["daemon", "start", "--background"],
    "foreground daemon": ["daemon"],
    "raw params": ["stats", "--json", "--params", "{}"],
    "index": ["index"],
}


def test_a_guest_during_a_hold_gets_only_allowed_calls_and_ends_it_with_done(
    lane: Lane, foreign_template: Path, queue: Path
) -> None:
    output, run = lane.new("o-guest"), lane.new("r-guest")
    locked = Locked(lane, queue, *run_args(foreign_template, output, run, "--window", "2",
                                           "--hold", "--hold-timeout", "120"))  # fmt: skip
    hold = locked.until({"event": "hold"})
    wrapper = run / "bin/recall"
    # The guest's shell carries nothing of the run; the wrapper supplies it all.
    shell = {"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]}
    daemon = next(line for line in registry(output) if line.get("name") == "daemon")

    def guest(*argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(wrapper), *argv], capture_output=True, text=True, env=shell, timeout=60
        )

    status = guest("daemon", "status", "--json")
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["daemon_pid"] == daemon["pid"]
    shown = guest("search", "--help")
    assert (shown.returncode, "Usage" in shown.stdout) == (0, True), shown.stderr
    refused = {name: guest(*argv).returncode for name, argv in REFUSED_CALLS.items()}
    assert refused == dict.fromkeys(REFUSED_CALLS, 5)
    assert psutil.Process(daemon["pid"]).status() != psutil.STATUS_ZOMBIE
    Path(hold["done"]).touch()
    exit_status, result = locked.finish()

    assert (exit_status, result["outcome"]) == (0, "passed"), result
    assert {"step": "hold", "ended": "done"} in result["steps"]
    assert_every_group_gone(output)
    assert_no_service_reach(lane, result, run)
    attempts = [
        json.loads(line) for line in (run / "bench-attempts.jsonl").read_text().splitlines()
    ]
    refusals = [item["detail"]["argv"] for item in attempts if item["seam"] == "cli"]
    assert refusals == list(REFUSED_CALLS.values())


def test_a_deadline_is_incomplete(lane: Lane, native_template: Path, queue: Path) -> None:
    output, run = lane.new("o-deadline"), lane.new("r-deadline")
    locked = Locked(lane, queue, *run_args(native_template, output, run, "--deadline", "0.5"))
    status, result = locked.finish()

    assert (status, result["outcome"]) == (2, "incomplete"), result
    assert result["error"]["kind"] == "deadline"
    assert_every_group_gone(output)


def test_a_deadline_bounds_hashing_a_large_template(lane: Lane, tmp_path: Path) -> None:
    backup = tmp_path / "large-backup"
    backup.mkdir()
    with (backup / "recall.duckdb").open("wb") as large:
        large.truncate(1024**3)
    template = lane.new("t-large")
    status, line = bench(lane, "prepare", "--backup", backup, "--template", template)
    assert (status, line["outcome"]) == (0, "passed"), line
    output, run = lane.new("o-large"), lane.new("r-large")
    started = time.monotonic()
    status, result = bench(
        lane, "daemon-run", *run_args(template, output, run, "--deadline", "0.2")
    )
    seconds = time.monotonic() - started

    assert (status, result["outcome"]) == (2, "incomplete"), result
    assert result["error"] == {"kind": "deadline", "message": "the run deadline expired"}
    # Verifying the template alone reads a gigabyte; the deadline stops it partway.
    assert seconds < 10
    assert result["steps"] == []
    assert not run.exists() and not output.exists()


def test_run_dir_refusals_exit_before_any_child(
    lane: Lane, native_template: Path, tmp_path: Path
) -> None:
    existing = lane.new("r-existing")
    existing.mkdir()
    other_lane = tmp_path / "other-lane"
    other_lane.mkdir()
    too_long = lane.root / ("r" * (110 - len(str(lane.root))))
    production = Path(lane.environment["HOME"]) / ".local/share/recall"
    cases = {
        "existing": (existing, "already exists; every run uses a new directory"),
        "other lane": (other_lane / "run", "is outside every write root"),
        "socket too long": (too_long, "the limit is"),
        "production data": (production, "overlaps refused location"),
    }
    for name, (run, reason) in cases.items():
        output = lane.new(f"o-refused-{name.replace(' ', '-')}")
        status, result = bench(lane, "daemon-run", *run_args(native_template, output, run))
        assert (status, result["outcome"]) == (1, "failed"), (name, result)
        assert reason in result["error"]["reason"], (name, result["error"])
        assert not output.exists(), name
    assert list(existing.iterdir()) == []
    assert list(other_lane.iterdir()) == []
    assert not too_long.exists()
    assert not lane.canary_log.exists()
    assert_home_untouched(lane)


FAKE_MODULES = {
    "recall/__init__.py": "",
    "recall/services/__init__.py": "",
    "recall/core/__init__.py": "",
    "recall/services/daemon.py": """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class ServiceView:
            installed_scheduler_record: object
            scheduler: object
            installed: bool
            service_data_dir: object
            measured_at: object

        def probe_service(config, installed_scheduler):
            raise AssertionError("the launcher replaces this")

        def _run_command(args, *, check=False):
            raise AssertionError("the launcher replaces this")

        def _run_lifecycle_command(args, action):
            raise AssertionError("the launcher replaces this")

        def _write_crontab(contents):
            raise AssertionError("the launcher replaces this")
    """,
    "recall/services/service_records.py": """
        def query(args):
            raise AssertionError("the launcher replaces this")

        def discover(scheduler, *, run=None):
            raise AssertionError("the launcher replaces this")
    """,
    "recall/services/usage_harvest.py": """
        def default_grok_unified_log(*, home=None):
            raise AssertionError("the launcher replaces this")
    """,
    "recall/core/entrypoint.py": """
        def spawn_detached_recall(data_dir, args):
            raise AssertionError("the launcher replaces this")
    """,
    "recall/core/rpc_client.py": "from recall.core.entrypoint import spawn_detached_recall\n",
    "recall/core/config.py": """
        import os
        import tomllib
        from dataclasses import dataclass, field
        from enum import Enum
        from pathlib import Path

        class Kind(Enum):
            AUTO = "auto"
            WATCH = "watch"

        @dataclass
        class Daemon:
            scheduler: Kind = Kind.AUTO
            mode: Kind = Kind.WATCH
            embed: bool = False

        @dataclass
        class Source:
            roots: tuple

        @dataclass
        class DuckDB:
            temp_directory: str

        @dataclass
        class AppConfig:
            data_dir: Path
            db_path: Path
            lock_path: Path
            config_path: Path
            duckdb: DuckDB
            sources: dict
            daemon: Daemon = field(default_factory=Daemon)

            @classmethod
            def load(cls):
                config = Path(os.environ["RECALL_CONFIG_PATH"])
                data = tomllib.loads(config.read_text())
                return cls(
                    Path(os.environ["RECALL_DATA_DIR"]),
                    Path(os.environ["RECALL_DB_PATH"]),
                    Path(os.environ["RECALL_LOCK_PATH"]),
                    config,
                    DuckDB(data["duckdb"]["temp_directory"]),
                    {name: Source(tuple(Path(root) for root in section["roots"]))
                     for name, section in data["sources"].items()},
                )
    """,
    "recall/cli/daemon.py": "from recall.core.entrypoint import spawn_detached_recall\n",
}
# A daemon that dies leaving a descendant in its process group.
ORPHANING_MAIN = """
    import os
    import subprocess
    import sys
    from pathlib import Path

    def main():
        if sys.argv[1:] == ["daemon"]:
            descendant = subprocess.Popen(["/bin/sleep", "300"])
            data = Path(os.environ["RECALL_DATA_DIR"])
            (data / "fake-descendant.pid").write_text(str(descendant.pid))
        sys.exit(1)
"""
# A daemon that signals the process a stale PID file names, then exits.
SIGNALLING_MAIN = """
    import os
    import signal
    import sys

    def main():
        try:
            os.kill(TARGET, signal.SIGTERM)
        except PermissionError:
            pass
        sys.exit(1)
"""


def fake_build(
    directory: Path,
    *,
    version: str = "0.40.5",
    rename: str | None = None,
    main: str = ORPHANING_MAIN,
) -> Path:
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(directory / ".venv")],
        check=True,
        timeout=120,
    )
    version_dir = next((directory / ".venv/lib").iterdir()).name
    # The layout uv writes, which the build resolver requires: aliases link to `python`.
    links = {"python": os.path.realpath(sys.executable), "python3": "python", version_dir: "python"}
    for alias, target in links.items():
        link = directory / ".venv/bin" / alias
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    site = directory / ".venv/lib" / version_dir / "site-packages"
    for relative, text in {**FAKE_MODULES, "recall/cli/__init__.py": main}.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        source = textwrap.dedent(text)
        if rename is not None:
            source = source.replace(f"def {rename}(", f"def {rename}_renamed(")
        path.write_text(source)
    metadata = site / f"recall-{version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: recall\nVersion: {version}\n")
    recall = directory / ".venv/bin/recall"
    recall.write_text(f"#!{directory / '.venv/bin/python'}\nfrom recall.cli import main\nmain()\n")
    recall.chmod(0o755)
    return directory


def fake_args(template: Path, build: Path, output: Path, run: Path) -> list[str | Path]:
    return [
        "--template", template, "--build", build, "--output", output, "--run-dir", run,
        "--profile", "idle",
    ]  # fmt: skip


def test_a_daemon_that_exits_leaving_a_descendant_has_its_whole_group_stopped(
    lane: Lane, native_template: Path, queue: Path, tmp_path: Path
) -> None:
    build = fake_build(tmp_path / "fake")
    output, run = lane.new("o-orphan"), lane.new("r-orphan")
    locked = Locked(lane, queue, *fake_args(native_template, build, output, run))
    status, result = locked.finish()

    assert (status, result["outcome"]) == (1, "failed"), result
    assert result["error"]["kind"] == "daemon_exited", result["error"]
    descendant = int((run / "fake-descendant.pid").read_text())
    assert (
        not psutil.pid_exists(descendant)
        or psutil.Process(descendant).status() == psutil.STATUS_ZOMBIE
    )
    daemon = next(line for line in registry(output) if line.get("name") == "daemon")
    stopped = next(
        line
        for line in registry(output)
        if line["event"] == "stopped" and line["pid"] == daemon["pid"]
    )
    assert stopped["signals"] == ["SIGKILL"] and stopped["status"] == 1
    assert_every_group_gone(output)
    assert_lock_released(lane, queue)


SENTINEL = """\
import signal, sys, time
log = open(sys.argv[1], "a", buffering=1)
for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
    signal.signal(signum, lambda received, _frame: log.write(f"{received}\\n"))
print("ready", flush=True)
time.sleep(600)
"""


def test_a_build_that_signals_a_process_it_does_not_own_is_refused(
    lane: Lane, native_template: Path, queue: Path, tmp_path: Path
) -> None:
    log = tmp_path / "sentinel-signals"
    sentinel = subprocess.Popen(
        [sys.executable, "-c", SENTINEL, str(log)],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert sentinel.stdout is not None and sentinel.stdout.readline() == "ready\n"
        main = SIGNALLING_MAIN.replace("TARGET", str(sentinel.pid))
        build = fake_build(tmp_path / "fake", main=main)
        output, run = lane.new("o-signal"), lane.new("r-signal")
        status, result = Locked(
            lane, queue, *fake_args(native_template, build, output, run)
        ).finish()
        alive = sentinel.poll() is None
        received = log.read_text()
    finally:
        sentinel.kill()
        sentinel.wait()
        assert sentinel.stdout is not None
        sentinel.stdout.close()

    assert (status, result["outcome"]) == (1, "failed"), result
    assert (alive, received) == (True, "")
    reach = result["scope"]["service_reach"]
    assert reach["refused_signals"] == [{"target": sentinel.pid, "signal": signal.SIGTERM}]
    assert_every_group_gone(output)
    assert_lock_released(lane, queue)


@pytest.mark.parametrize("seam", ["probe_service", "default_grok_unified_log"])
def test_a_renamed_seam_is_refused_before_any_recall_code_runs(
    lane: Lane, native_template: Path, tmp_path: Path, seam: str
) -> None:
    build = fake_build(tmp_path / "fake", rename=seam)
    output, run = lane.new(f"o-seam-{seam}"), lane.new(f"r-seam-{seam}")
    status, result = bench(lane, "daemon-run", *fake_args(native_template, build, output, run))

    assert (status, result["outcome"]) == (2, "incomplete"), result
    assert result["error"]["kind"] == "seam_missing"
    assert any(seam in problem for problem in result["error"]["report"]["problems"])
    assert [line["name"] for line in registry(output) if line["event"] == "started"] == ["check"]
    assert (
        not (run / "bench-attempts.jsonl").exists()
        or "suppressed" not in (run / "bench-attempts.jsonl").read_text()
    )


def test_a_build_older_than_the_floor_is_refused_by_name(
    lane: Lane, native_template: Path, tmp_path: Path
) -> None:
    build = fake_build(tmp_path / "fake", version="0.40.4")
    output, run = lane.new("o-old"), lane.new("r-old")
    status, result = bench(lane, "daemon-run", *fake_args(native_template, build, output, run))

    assert (status, result["outcome"]) == (1, "failed"), result
    assert result["error"]["kind"] == "unsupported_build"
    assert result["error"]["report"] == {"build": "0.40.4", "unsupported": "older than 0.40.5"}


@pytest.fixture(autouse=True)
def _no_leftover_children() -> Iterator[None]:
    """A test leaves none of the processes it started; other modules' helpers are theirs."""
    before = {child.pid for child in psutil.Process().children()}
    yield
    assert [child for child in psutil.Process().children() if child.pid not in before] == []
