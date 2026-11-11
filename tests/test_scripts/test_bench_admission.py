"""Guarded offline admission through real supervisor/selected-build processes.

Synthetic recall databases verify actual DuckDB WAL refusal. Controlled builds
exercise isolation and error propagation, not product compatibility correctness.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from recall.db.schema import SCHEMA_VERSION

from tests.test_scripts.test_bench_daemon import (
    BENCH,
    ROOT,
    Lane,
    assert_every_group_gone,
    bench,
    synthetic_backup,
)
from tests.test_scripts.test_bench_daemon import lane as lane

LAUNCH = ROOT / "scripts/recall_bench/launch.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module", params=["systemd", "launchd", "cron"])
def template(lane: Lane, request: pytest.FixtureRequest) -> Path:
    name = request.param
    backup = synthetic_backup(lane.new(f"{name}-source"), name)
    target = lane.new(f"{name}-template")
    status, result = bench(lane, "prepare", "--backup", backup, "--template", target, entry=BENCH)
    assert (status, result["outcome"]) == (0, "passed"), result
    return target


def arguments(template: Path, output: Path, run: Path, build: Path = ROOT) -> list[str | Path]:
    return [
        "admission", "--template", template, "--build", build,
        "--output", output, "--run-dir", run,
    ]  # fmt: skip


def test_real_admission_keeps_newer_version_in_wal_and_refuses_each_open(
    lane: Lane, template: Path
) -> None:
    name = template.name
    output, run = lane.new(f"{name}-out"), lane.new(f"{name}-run")
    original = digest(template / "payload/recall.duckdb")
    contaminated = {
        **lane.environment,
        "RECALL_FLEET_PATH": str(lane.root / "host-fleet-sentinel"),
        "RECALL_DUCKDB_MEMORY_LIMIT": "99GB",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/not-a-service-bus",
        "XDG_RUNTIME_DIR": str(lane.root / "host-runtime"),
    }
    status, result = bench(lane, *arguments(template, output, run), env=contaminated, entry=BENCH)
    assert (status, result["outcome"]) == (0, "passed"), result
    assert json.loads((output / "result.json").read_bytes()) == result
    admission = json.loads((output / "admission.json").read_bytes())
    assert admission["requests"][-1] == {
        "case": "negative",
        "operation": "negative-write",
        "schema": SCHEMA_VERSION + 1,
    }
    observations = admission["observations"]
    assert [item["operation"] for item in observations] == [
        "inspect",
        "positive-read",
        "positive-write",
        "prepare-newer",
        "negative-read",
        "negative-write",
    ]
    assert [item["state"] for item in observations] == [
        "inspected",
        "opened",
        "opened",
        "prepared",
        "refused",
        "refused",
    ]
    assert (
        observations[1]["observed_schema"] == observations[2]["observed_schema"] == SCHEMA_VERSION
    )
    assert observations[2]["audit_before"] == observations[2]["audit_after"]
    assert observations[3]["schema_before"] == SCHEMA_VERSION
    assert observations[3]["observed_schema"] == SCHEMA_VERSION + 1
    refusals = admission["refusals"]
    assert len(refusals) == 2
    for refusal in refusals:
        assert refusal["unchanged"] is True
        assert refusal["before"] == refusal["after"]
        assert refusal["before"]["recall.duckdb.wal"]["bytes"] > 0
        assert refusal["observation"]["error"]["database_version"] == SCHEMA_VERSION + 1
        assert refusal["observation"]["error"]["supported_version"] == SCHEMA_VERSION
    # Preparing a WAL did not checkpoint the logical version into the base DB.
    assert admission["control_initial"]["recall.duckdb"] == refusals[0]["before"]["recall.duckdb"]
    assert refusals[0]["after"] == refusals[1]["before"]
    assert digest(template / "payload/recall.duckdb") == original
    assert not lane.canary_log.exists()
    for name in ("inspect", "positive", "negative"):
        assert not (run / name / "recall.sock").exists()
        assert not (run / name / "recall.pid").exists()
        reach = admission["safety"][name]
        assert reach["canary_invocations"] == reach["refused"] == reach["refused_signals"] == []
    assert_every_group_gone(output / "inspect")
    assert_every_group_gone(output / "positive")
    assert_every_group_gone(output / "negative")


@pytest.mark.skipif(sys.platform != "darwin", reason="requires the native /tmp logical alias")
def test_darwin_alias_run_preserves_logical_paths_and_each_refusal_hash(
    lane: Lane, template: Path
) -> None:
    output = lane.new(template.name + "-alias-out")
    with tempfile.TemporaryDirectory(
        prefix="bench-admission-", dir=os.environ.get("BENCH_TEST_ALIAS_ROOT", "/tmp")
    ) as temporary:
        root = Path(temporary)
        run = root / "run"
        assert root != root.resolve()
        environment = {**lane.environment, "BENCH_WRITE_ROOTS": f"{lane.root}:{root}"}
        status, result = bench(
            lane, *arguments(template, output, run), env=environment, entry=BENCH
        )
        assert (status, result["outcome"]) == (0, "passed"), result
        record = json.loads((output / "admission.json").read_bytes())
        assert [item["state"] for item in record["observations"]] == [
            "inspected",
            "opened",
            "opened",
            "prepared",
            "refused",
            "refused",
        ]
        for observation in record["observations"]:
            if observation["operation"] == "inspect":
                case = "inspect"
            elif observation["operation"] in {"positive-read", "positive-write"}:
                case = "positive"
            else:
                case = "negative"
            paths = observation["paths"]
            assert paths["cwd"] == str(run.resolve() / case)
            assert paths["data_dir"] == str(run / case)
            assert paths["db_path"] == str(run / case / "recall.duckdb")
        assert [item["observation"]["operation"] for item in record["refusals"]] == [
            "negative-read",
            "negative-write",
        ]
        for refusal in record["refusals"]:
            assert refusal["unchanged"] is True
            assert refusal["before"] == refusal["after"]
            for name in ("recall.duckdb", "recall.duckdb.wal"):
                fingerprint = refusal["after"][name]
                payload = run / "negative" / name
                assert fingerprint["present"] is True
                assert fingerprint["bytes"] == payload.stat().st_size > 0
                assert fingerprint["sha256"] == digest(payload)
            error = refusal["observation"]["error"]
            assert (error["type"], error["database_version"], error["supported_version"]) == (
                "SchemaTooNewError",
                SCHEMA_VERSION + 1,
                SCHEMA_VERSION,
            )
        for case in ("inspect", "positive", "negative"):
            assert_every_group_gone(output / case)
            reach = record["safety"][case]
            assert reach["canary_invocations"] == reach["refused"] == reach["refused_signals"] == []


@pytest.mark.parametrize(
    "schema", ["0", str(SCHEMA_VERSION), "01", "+37", "-1", "2147483648", "true"]
)
def test_bad_negative_schema_does_not_open_database(
    lane: Lane, template: Path, schema: str
) -> None:
    tag = f"{template.name}-schema-{schema}"
    output, run = lane.new(tag + "-out"), lane.new(tag + "-run")
    status, result = bench(
        lane, *arguments(template, output, run), "--negative-schema", schema, entry=BENCH
    )
    assert (status, result["outcome"]) == (1, "failed"), result
    assert digest(template / "payload/recall.duckdb")
    assert not (run / "positive").exists()
    assert not (run / "negative").exists()
    assert not list(run.glob("**/recall.duckdb"))
    if (output / "admission.json").exists():
        record = json.loads((output / "admission.json").read_bytes())
        assert all(o["operation"] == "inspect" for o in record["observations"])


@pytest.mark.parametrize(
    "operands",
    [
        ["unknown"],
        ["positive-write", "/elsewhere"],
        ["prepare-newer", "01"],
        ["negative-read", "37", "SELECT 1"],
        ["inspect", "exec('x')"],
    ],
)
def test_invalid_leaf_request_runs_no_build_work(tmp_path: Path, operands: list[str]) -> None:
    attempts = tmp_path / "attempts"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(LAUNCH),
            "--attempts",
            str(attempts),
            "--usage-log",
            str(tmp_path / "usage"),
            "admission",
            *operands,
        ],
        input=b"g",
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode != 0
    assert not attempts.exists()


def test_admission_cannot_skip_registration_gate(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts"
    for flags, released in (([], b""), (["--ungated"], b"g")):
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(LAUNCH),
                "--attempts",
                str(attempts),
                "--usage-log",
                str(tmp_path / "usage"),
                *flags,
                "admission",
                "inspect",
            ],
            input=released,
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert completed.returncode == (5 if flags else 6), completed.stderr
        assert not attempts.exists()


def controlled_build(directory: Path, *, version: str = "0.40.9", change: str = "") -> Path:
    """A selected package with real DB leaves and explicit admission faults.

    Its .pth makes dependencies available without importing recall before the
    launcher's audit guard. The copied product files belong to this fixture.
    """
    import shutil

    from tests.test_scripts.test_bench_daemon import fake_build

    build = fake_build(directory, version=version, main="def main(): return 0\n")
    site = next((build / ".venv/lib").glob("python*/site-packages"))
    shutil.rmtree(site / "recall")
    shutil.copytree(
        ROOT / "packages/recall/src/recall",
        site / "recall",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    tool_site = (
        Path(sys.executable).parent.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    (site / "dependencies.pth").write_text(str(tool_site) + "\n")
    if change:
        with (site / "recall/db/connection.py").open("a") as target:
            target.write("\n" + change)
    return build


def test_selected_interpreter_supplies_version_and_supported_schema(
    lane: Lane, template: Path
) -> None:
    name = template.name
    build = controlled_build(
        lane.new(name + "-selected"),
        change="""
import recall.db.schema as selected_schema
selected_schema.SCHEMA_VERSION = 73
""",
    )
    output, run = lane.new(name + "-selected-out"), lane.new(name + "-selected-run")
    status, result = bench(lane, *arguments(template, output, run, build), entry=BENCH)
    assert status == 1, result  # The synthetic current DB cannot migrate to this invented schema.
    assert result["error"]["kind"] == "admission_observation", result
    observations = json.loads((output / "admission.json").read_bytes())["observations"]
    assert observations[0]["operation"] == "inspect"
    assert observations[0]["supported_schema"] == 73
    assert observations[0]["build"] == "0.40.9"
    assert_every_group_gone(output / "positive")


@pytest.mark.parametrize(
    "fault, expected",
    [
        ("wrong-error", "admission_observation"),
        ("wrong-version", "admission_observation"),
        ("unexpected-open", "admission_observation"),
        ("mutate-db", "refusal_mutated"),
        ("delete-wal", "refusal_mutated"),
    ],
)
def test_negative_failure_keeps_independent_after_hash_and_stops(
    lane: Lane, template: Path, fault: str, expected: str
) -> None:
    name = f"{template.name}-{fault}"
    # Fail only read-only opens of the prepared control, after the real product
    # has exercised the positive flow and the preparer has retained its WAL.
    change = f"""
from recall.db.schema import SCHEMA_VERSION

_original_readonly = connect_readonly

def connect_readonly(config):
    wal = Path(str(config.db_path) + ".wal")
    if config.data_dir.name == "negative" and wal.exists():
        fault = {fault!r}
        if fault == "wrong-error":
            raise RuntimeError("controlled wrong error")
        if fault == "wrong-version":
            raise SchemaTooNewError(99, SCHEMA_VERSION)
        if fault == "unexpected-open":
            return duckdb.connect(str(config.db_path))
        if fault == "mutate-db":
            with config.db_path.open("ab") as target:
                target.write(b"controlled mutation")
        if fault == "delete-wal":
            wal.unlink()
        raise SchemaTooNewError(SCHEMA_VERSION + 1, SCHEMA_VERSION)
    return _original_readonly(config)
"""
    build = controlled_build(lane.new(name + "-build"), change=change)
    output, run = (
        lane.new(name + "-out"),
        lane.new("r-" + hashlib.sha256(name.encode()).hexdigest()[:12]),
    )
    status, result = bench(lane, *arguments(template, output, run, build), entry=BENCH)
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", expected), result
    document = json.loads((output / "admission.json").read_bytes())
    assert len(document["refusals"]) == 1
    refusal = document["refusals"][0]
    assert "after" in refusal
    assert not any(o["operation"] == "negative-write" for o in document["observations"])
    if fault in ("mutate-db", "delete-wal"):
        assert refusal["before"] != refusal["after"]
        assert refusal["unchanged"] is False
    else:
        assert refusal["unchanged"] is None
    assert_every_group_gone(output / "positive")
    assert_every_group_gone(output / "negative")


@pytest.mark.parametrize("fault", ["usage", "admission", "old"])
def test_missing_seam_and_old_build_never_fall_back(lane: Lane, template: Path, fault: str) -> None:
    name = f"{template.name}-missing-{fault}"
    build = controlled_build(
        lane.new(name + "-build"), version="0.40.4" if fault == "old" else "0.40.9"
    )
    site = next((build / ".venv/lib").glob("python*/site-packages"))
    if fault == "usage":
        path = site / "recall/services/usage_harvest.py"
        path.write_text(
            path.read_text().replace("def default_grok_unified_log(", "def removed_usage_seam(")
        )
    if fault == "admission":
        path = site / "recall/db/connection.py"
        path.write_text(
            path.read_text().replace("def connect_existing(", "def removed_admission_seam(")
        )
    output, run = (
        lane.new(name + "-out"),
        lane.new("r-" + hashlib.sha256(name.encode()).hexdigest()[:12]),
    )
    status, result = bench(lane, *arguments(template, output, run, build), entry=BENCH)
    assert (status, result["outcome"]) == (2, "incomplete"), result
    assert result["error"]["kind"] == ("unsupported_build" if fault == "old" else "seam_missing")
    assert not (run / "positive").exists()
    assert not (run / "negative").exists()
    assert not list(run.glob("**/recall.duckdb"))
    assert_every_group_gone(output / "inspect")


@pytest.mark.parametrize("fault", ["manifest", "hardlink", "symlink", "missing-db", "overlap"])
def test_unverified_and_overlapping_inputs_start_no_child(
    lane: Lane, template: Path, fault: str
) -> None:
    import shutil

    name = f"{template.name}-path-{fault}"
    source = lane.new(name + "-template")
    shutil.copytree(template, source)
    payload = source / "payload/recall.duckdb"
    if fault == "manifest":
        payload.write_bytes(b"corrupt")
    elif fault == "hardlink":
        os.link(payload, lane.new(name + "-alias"))
    elif fault == "symlink":
        payload.unlink()
        payload.symlink_to(template / "payload/recall.duckdb")
    elif fault == "missing-db":
        payload.unlink()
    output = lane.new(name + "-out")
    run = (
        output
        if fault == "overlap"
        else lane.new("r-" + hashlib.sha256(name.encode()).hexdigest()[:12])
    )
    status, result = bench(lane, *arguments(source, output, run), entry=BENCH)
    assert (status, result["outcome"]) == (1, "failed"), result
    assert not output.exists()
    assert not run.exists()
    assert not lane.canary_log.exists()


@pytest.mark.parametrize("fault", ["malformed", "oversized", "seam-reach", "exec-reach"])
def test_bad_output_and_caught_service_reach_cannot_pass(
    lane: Lane, template: Path, fault: str
) -> None:
    name = f"{template.name}-output-{fault}"
    change = f"""
from recall.db.schema import SCHEMA_VERSION

_original_readonly = connect_readonly

def connect_readonly(config):
    if config.data_dir.name == "positive":
        fault = {fault!r}
        if fault in ("malformed", "oversized"):
            import sys
            sys.stdout.write("not-json" if fault == "malformed" else "x" * (1024 * 1024 + 1))
            sys.stdout.flush()
            os._exit(0)
        try:
            if fault == "seam-reach":
                from recall.services import daemon
                daemon._run_command(["bench-canary"])
            else:
                import subprocess
                subprocess.run(
                    [str(config.data_dir / "canary/systemctl"), "bench-canary"], check=True
                )
        except PermissionError:
            pass
    return _original_readonly(config)
"""
    build = controlled_build(lane.new(name + "-build"), change=change)
    output = lane.new(name + "-out")
    run = lane.new("r-" + hashlib.sha256(name.encode()).hexdigest()[:12])
    status, result = bench(lane, *arguments(template, output, run, build), entry=BENCH)
    assert (status, result["outcome"]) == (1, "failed"), result
    expected = {
        "malformed": "admission_document",
        "oversized": "admission_document",
        "seam-reach": "service_reach_suppressed",
        "exec-reach": "service_reach_refused",
    }
    assert result["error"]["kind"] == expected[fault], result
    assert not (run / "negative").exists()
    assert not lane.canary_log.exists()
    assert_every_group_gone(output / "inspect")
    assert_every_group_gone(output / "positive")


def entrypoint(lane: Lane, name: str, setup: str) -> Path:
    entry = lane.new(name + ".py")
    entry.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
        "import bench\nfrom recall_bench import admission, supervisor\n"
        + setup
        + "\nraise SystemExit(bench.main())\n"
    )
    return entry


def event_from(process: subprocess.Popen[bytes], consumed: list[str]) -> dict:
    """Read one bounded event without a buffered readline hiding a deadline."""
    from tests.bench_lock_events import failure_details, parse_event

    stream = process.stderr
    assert stream is not None
    deadline = time.monotonic() + 60
    raw = bytearray()
    while len(raw) < 1024 * 1024:
        remaining = deadline - time.monotonic()
        assert remaining > 0 and select.select([stream], [], [], remaining)[0], raw
        value = os.read(stream.fileno(), 1)
        assert value, raw
        if value == b"\n":
            line = raw.decode() + "\n"
            consumed.append(line)
            return parse_event(line, lambda: failure_details(process, "".join(consumed)))
        raw.extend(value)
    raise AssertionError("event exceeds 1 MiB")


@pytest.mark.parametrize("signum", [2, 15])
@pytest.mark.parametrize("during", ["operation", "hash"])
def test_cancelled_admission_stops_groups_releases_lock_and_starts_no_later_attempt(
    lane: Lane, template: Path, signum: int, during: str
) -> None:
    import signal

    from tests.test_scripts.test_bench_daemon import LOCK, assert_lock_released

    name = f"{template.name}-cancel-{during}-{signum}"
    change = (
        """
from recall.db.schema import SCHEMA_VERSION

_original_readonly = connect_readonly

def connect_readonly(config):
    if config.data_dir.name == "negative":
        import time
        time.sleep(3600)
    return _original_readonly(config)
"""
        if during == "operation"
        else ""
    )
    build = controlled_build(lane.new(name + "-build"), change=change)
    output = lane.new(name + "-out")
    run = lane.new("r-" + hashlib.sha256(name.encode()).hexdigest()[:12])
    setup = ""
    if during == "hash":
        setup = """
from recall_bench.fs import File
original_hash = File.sha256

def paused_hash(self, cancellation):
    from pathlib import Path
    import time, json
    control = self.path.parent.name == "negative" and self.path.name == "recall.duckdb"
    if control and Path(str(self.path) + ".wal").exists():
        print(json.dumps({"event": "hashing-control"}), file=sys.stderr, flush=True)
        while True:
            cancellation.check()
            time.sleep(0.05)
    return original_hash(self, cancellation)
File.sha256 = paused_hash
"""
    entry = entrypoint(lane, name, setup)
    queue = lane.new(name + "-queue")
    process = subprocess.Popen(
        [
            sys.executable,
            str(LOCK),
            "--lock",
            str(queue),
            "admission-test",
            "--",
            sys.executable,
            str(entry),
            *map(str, arguments(template, output, run, build)),
        ],
        env=lane.environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    events: list[dict] = []
    consumed: list[str] = []
    try:
        assert process.stderr is not None
        for _ in range(200):
            event = event_from(process, consumed)
            events.append(event)
            active = (
                (event.get("name") == "negative-read" and event.get("event") == "started")
                if during == "operation"
                else event.get("event") == "hashing-control"
            )
            if active:
                break
        else:
            raise AssertionError(events)
        import psutil

        bench_pid = next(event["pid"] for event in events if event.get("event") == "bench")
        owned_bench = psutil.Process(bench_pid)
        birth = owned_bench.create_time()
        assert owned_bench.ppid() == process.pid
        # Signal the recorded command itself: the runner deliberately sends TERM
        # to its group when it receives INT, which would not exercise bench INT.
        os.kill(bench_pid, signum)
        for _ in range(200):
            event = event_from(process, consumed)
            events.append(event)
            if event.get("event") == "cancelled":
                assert owned_bench.create_time() == birth and owned_bench.ppid() == process.pid
                os.kill(bench_pid, signal.SIGTERM)
                break
        stdout, stderr = process.communicate(timeout=60)
        result = json.loads(stdout.strip().splitlines()[-1])
        assert (process.returncode, result["outcome"]) == (128 + signum, "incomplete"), (
            result,
            stderr,
        )
        assert result["error"] == {"kind": "cancelled", "signal": signum}
        record = json.loads((output / "admission.json").read_bytes())
        assert not any(o["operation"] == "negative-write" for o in record["observations"])
        assert all(r["unchanged"] is None for r in record["refusals"])
        assert_every_group_gone(output / "positive")
        assert_every_group_gone(output / "negative")
        assert_lock_released(lane, queue)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=60)


def test_stalled_child_deadline_is_incomplete_with_partial_evidence(
    lane: Lane, template: Path
) -> None:
    name = template.name + "-deadline"
    build = controlled_build(
        lane.new(name + "-build"),
        change="""
from recall.db.schema import SCHEMA_VERSION

_original_readonly = connect_readonly

def connect_readonly(config):
    (config.data_dir / "stall-entered").write_text("ready")
    import time
    time.sleep(3600)
    return _original_readonly(config)
""",
    )
    entry = entrypoint(
        lane,
        name,
        """
original_observe = admission.Case.observe

def observe(self, request):
    if request.operation == "positive-read":
        self.deadline = supervisor.Deadline(5.0)
        self.supervisor._cancellation.expire_at(self.deadline.at)
    return original_observe(self, request)

admission.Case.observe = observe
""",
    )
    output = lane.new(name + "-out")
    run = lane.new("r-" + hashlib.sha256(name.encode()).hexdigest()[:12])
    status, result = bench(lane, *arguments(template, output, run, build), entry=entry)
    assert (status, result["outcome"], result["error"]["kind"]) == (2, "incomplete", "deadline"), (
        result
    )
    assert not (run / "negative").exists()
    assert (run / "positive/stall-entered").read_text() == "ready"
    assert json.loads((output / "result.json").read_bytes()) == result
    assert_every_group_gone(output / "inspect")
    assert_every_group_gone(output / "positive")


def test_supported_older_copy_records_real_migration_and_normalizes_separate_control(
    lane: Lane,
) -> None:
    import duckdb
    from recall.db.migrations import get_migrations

    backup = synthetic_backup(lane.new("older-source"), "systemd")
    with duckdb.connect(str(backup / "recall.duckdb")) as connection:
        connection.execute("DELETE FROM schema_version WHERE version >= 36")
        connection.execute("INSERT INTO schema_version(version) VALUES (35)")
        for migration in get_migrations():
            if migration.target_version > 35:
                connection.execute(
                    "DELETE FROM schema_migrations WHERE migration_id = ?", [migration.id]
                )
        connection.execute("DROP TABLE daemon_index_repair")
        connection.execute("ALTER TABLE usage_log_cursors DROP COLUMN file_device")
        connection.execute("ALTER TABLE usage_log_cursors DROP COLUMN file_inode")
    template = lane.new("older-template")
    status, line = bench(lane, "prepare", "--backup", backup, "--template", template, entry=BENCH)
    assert status == 0, line
    output, run = lane.new("older-out"), lane.new("older-run")
    status, result = bench(lane, *arguments(template, output, run), entry=BENCH)
    assert (status, result["outcome"]) == (0, "passed"), result
    record = json.loads((output / "admission.json").read_bytes())
    observations = record["observations"]
    assert [o["operation"] for o in observations].count("positive-write") == 2
    assert observations[1]["observed_schema"] == 35
    assert observations[2]["schema_before"] == 35
    assert observations[2]["observed_schema"] == SCHEMA_VERSION
    assert observations[2]["audit_before"] != observations[2]["audit_after"]
    assert observations[3]["operation"] == "positive-write"
    assert observations[3]["schema_before"] == 35
    assert observations[4]["schema_before"] == SCHEMA_VERSION
    assert_every_group_gone(output / "positive")
    assert_every_group_gone(output / "negative")


@pytest.mark.parametrize("damage", ["delete", "empty"])
def test_preparation_without_retained_wal_cannot_pass(
    lane: Lane, template: Path, damage: str
) -> None:
    name = f"{template.name}-control-{damage}"
    build = controlled_build(
        lane.new(name + "-build"),
        change=f"""
_original_existing = connect_existing

class PreparedConnection:
    def __init__(self, connection, path):
        self.connection = connection
        self.path = path
    def __getattr__(self, name):
        return getattr(self.connection, name)
    def close(self):
        self.connection.close()
        wal = Path(str(self.path) + ".wal")
        if {damage!r} == "delete":
            wal.unlink(missing_ok=True)
        else:
            wal.write_bytes(b"")

def connect_existing(config, *, read_only):
    connection = _original_existing(config, read_only=read_only)
    if not read_only and config.data_dir.name == "negative":
        return PreparedConnection(connection, config.db_path)
    return connection
""",
    )
    output = lane.new(name + "-out")
    run = lane.new("r-" + hashlib.sha256(name.encode()).hexdigest()[:12])
    status, result = bench(lane, *arguments(template, output, run, build), entry=BENCH)
    assert (status, result["outcome"], result["error"]["kind"]) == (
        1,
        "failed",
        "admission_observation",
    ), result
    record = json.loads((output / "admission.json").read_bytes())
    assert record["refusals"] == []
    assert record["observations"][-1]["state"] == "error"
    assert "nonempty WAL" in record["observations"][-1]["error"]["message"]
    assert_every_group_gone(output / "negative")


@pytest.mark.parametrize("which", ["registry", "output"])
def test_terminal_descriptor_failure_cannot_turn_into_success(
    lane: Lane, template: Path, which: str
) -> None:
    name = f"{template.name}-close-{which}"
    setup = f"""
import os
original_close = os.close
faults = set()
original_init = supervisor.Supervisor.__init__
original_spawn = supervisor.Supervisor.spawn

def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if {which!r} == "registry":
        faults.add(self._registry)

def spawn(self, *args, **kwargs):
    child = original_spawn(self, *args, **kwargs)
    if {which!r} == "output":
        faults.add(child.stdout)
    return child

def close(descriptor):
    original_close(descriptor)
    if descriptor in faults:
        faults.remove(descriptor)
        raise OSError(5, "controlled close failure")

supervisor.Supervisor.__init__ = init
supervisor.Supervisor.spawn = spawn
os.close = close
"""
    entry = entrypoint(lane, name, setup)
    output = lane.new(name + "-out")
    run = lane.new("r-" + hashlib.sha256(name.encode()).hexdigest()[:12])
    status, result = bench(
        lane,
        *arguments(template, output, run),
        "--negative-schema",
        str(SCHEMA_VERSION),
        entry=entry,
    )
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "stop_failed"), (
        result
    )
    assert json.loads((output / "result.json").read_bytes()) == result
    if which == "registry":
        assert result["error"]["original"]["kind"] == "negative_schema"
    assert_every_group_gone(output / "inspect")


@pytest.mark.parametrize(
    "vector",
    [
        ("positive-write", 37),
        ("prepare-newer", None),
        ("prepare-newer", True),
        ("negative-read", -1),
        ("negative-write", 2147483648),
        ("not-an-operation", None),
    ],
)
def test_request_construction_rejects_vectors_outside_fixed_grammar(vector: tuple) -> None:
    from scripts.recall_bench.launch import AdmissionRequest

    with pytest.raises(ValueError):
        AdmissionRequest(*vector)


def test_writer_admission_stays_strict_when_selected_defaults_differ(
    lane: Lane, template: Path
) -> None:
    name = template.name + "-strict-writer"
    build = controlled_build(
        lane.new(name + "-build"),
        change="""
_original_connect = connect

def connect(config, *, recreate=True, lenient_schema=True):
    if recreate or lenient_schema:
        raise RuntimeError("writer admission must withhold recreate and lenient authority")
    return _original_connect(config, recreate=False, lenient_schema=False)
""",
    )
    output = lane.new(name + "-out")
    run = lane.new("r-" + hashlib.sha256(name.encode()).hexdigest()[:12])
    status, result = bench(lane, *arguments(template, output, run, build), entry=BENCH)
    assert (status, result["outcome"]) == (0, "passed"), result
    record = json.loads((output / "admission.json").read_bytes())
    assert [o["state"] for o in record["observations"]] == [
        "inspected",
        "opened",
        "opened",
        "prepared",
        "refused",
        "refused",
    ]
    assert all(r["unchanged"] is True for r in record["refusals"])
    assert_every_group_gone(output / "positive")
    assert_every_group_gone(output / "negative")


def test_oversized_refusal_diagnostic_cannot_be_a_passing_observation(
    lane: Lane, template: Path
) -> None:
    name = template.name + "-oversized-refusal"
    build = controlled_build(
        lane.new(name + "-build"),
        change="""
from recall.core.schema_version import SchemaTooNewError
SchemaTooNewError.__str__ = lambda self: "x" * 5000
""",
    )
    output = lane.new(name + "-out")
    run = lane.new("r-" + hashlib.sha256(name.encode()).hexdigest()[:12])
    status, result = bench(lane, *arguments(template, output, run, build), entry=BENCH)
    assert (status, result["outcome"], result.get("error", {}).get("kind")) == (
        1,
        "failed",
        "admission_observation",
    ), result
    record = json.loads((output / "admission.json").read_bytes())
    assert len(record["refusals"]) == 1
    refusal = record["refusals"][0]
    assert refusal["unchanged"] is None
    assert refusal["before"] == refusal["after"]
    observation = refusal["observation"]
    assert observation["state"] == "error"
    assert observation["error"]["cause"]["type"] == "SchemaTooNewError"
    assert observation["error"]["cause"]["original_bytes"] == 5000
    assert not any(o["operation"] == "negative-write" for o in record["observations"])
    assert_every_group_gone(output / "negative")
