"""Real-process compare and FIFO pairs outcomes.

Pairs replaces only daemon-run with an in-process stub. It verifies sequencing,
lock admission and failure propagation, not daemon measurements or supervision.
Compare uses real DuckDB databases, prepare manifests and CLI processes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "scripts/bench.py"


@pytest.fixture
def lane(tmp_path: Path) -> Path:
    root = tmp_path / "lane"
    root.mkdir()
    (root / "runs").mkdir()
    return root


def invoke(lane: Path, *args: str | Path, prefix: list[str] | None = None) -> tuple[int, dict]:
    completed = subprocess.run(
        [sys.executable, *(prefix or [str(BENCH)]), *map(str, args)],
        capture_output=True,
        text=True,
        env={**os.environ, "BENCH_WRITE_ROOTS": str(lane)},
        timeout=60,
        check=False,
    )
    assert completed.stdout.strip(), completed.stderr
    return completed.returncode, json.loads(completed.stdout.splitlines()[-1])


def prepare(lane: Path, name: str, sql: str) -> Path:
    backup = lane / (name + "-backup")
    backup.mkdir()
    with duckdb.connect(str(backup / "recall.duckdb"), config={"threads": 2}) as connection:
        connection.execute(sql)
    target = lane / name
    status, result = invoke(lane, "prepare", "--backup", backup, "--template", target)
    assert (status, result["outcome"]) == (0, "passed"), result
    return target


SQL = """CREATE SCHEMA "odd's schema";
CREATE TABLE "odd's schema"."a""b" (n INTEGER, value VARCHAR);
INSERT INTO "odd's schema"."a""b" VALUES (1, 'one'), (1, 'one'), (NULL, NULL);
CREATE TABLE empty (n BIGINT);"""


def run_compare(lane: Path, before: Path, after: Path, **options: str) -> tuple[int, dict]:
    return invoke(
        lane,
        "compare",
        "--before",
        before,
        "--after",
        after,
        "--output",
        lane / "output",
        *(part for key, value in options.items() for part in ("--" + key, value)),
    )


def snapshot(path: Path) -> dict[str, bytes]:
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_equal_multisets_read_only_inputs_and_output_local_spill(lane: Path) -> None:
    before = prepare(lane, "before", SQL)
    after = prepare(lane, "after", SQL)
    originals = snapshot(before), snapshot(after)
    status, result = run_compare(lane, before, after, **{"memory-limit": str(32 * 1024**2)})
    assert (status, result["outcome"]) == (0, "passed"), result
    comparison = json.loads((lane / "output/comparison.json").read_bytes())
    assert comparison["equal"] is True
    assert len(comparison["tables"]) == 2
    assert all(t["before_only"] == t["after_only"] == 0 for t in comparison["tables"])
    assert (lane / "output/spill").is_dir()
    assert result["scope"]["settings"]["read_only"] is True
    assert originals == (snapshot(before), snapshot(after))
    assert json.loads((lane / "output/result.json").read_bytes()) == result


@pytest.mark.parametrize(
    ("mutation", "before_only", "after_only"),
    [
        (
            'DELETE FROM "odd\'s schema"."a""b" WHERE n = 1; '
            'INSERT INTO "odd\'s schema"."a""b" VALUES (1, \'one\')',
            1,
            0,
        ),
        ('UPDATE "odd\'s schema"."a""b" SET value = \'changed\' WHERE n = 1', 2, 2),
        ('INSERT INTO "odd\'s schema"."a""b" VALUES (NULL, NULL)', 0, 1),
    ],
)
def test_changed_rows_and_duplicate_multiplicity(
    lane: Path, mutation: str, before_only: int, after_only: int
) -> None:
    before = prepare(lane, "before", SQL)
    after = prepare(lane, "after", SQL + mutation)
    status, result = run_compare(lane, before, after)
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "different")
    table = next(t for t in result["steps"][-1]["tables"] if t["table"] == 'a"b')
    assert (table["before_only"], table["after_only"]) == (before_only, after_only)
    assert json.loads((lane / "output/comparison.json").read_bytes())["equal"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "CREATE TABLE added (n INTEGER)",
        "DROP TABLE empty",
        "ALTER TABLE empty ADD COLUMN value VARCHAR",
        "ALTER TABLE empty ALTER COLUMN n TYPE INTEGER",
    ],
)
def test_schema_differences_fail_even_for_empty_tables(lane: Path, mutation: str) -> None:
    before = prepare(lane, "before", SQL)
    after = prepare(lane, "after", SQL + mutation)
    status, result = run_compare(lane, before, after)
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "different")
    assert any(not table["equal"] for table in result["steps"][-1]["tables"])


@pytest.mark.parametrize("which", ["before", "after"])
@pytest.mark.parametrize("damage", ["manifest", "payload", "symlink", "hardlink"])
def test_unverified_or_aliased_inputs_are_refused_before_output(
    lane: Path, which: str, damage: str
) -> None:
    before = prepare(lane, "before", SQL)
    after = prepare(lane, "after", SQL)
    damaged = {"before": before, "after": after}[which]
    if damage == "manifest":
        (damaged / "manifest.json").unlink()
    elif damage == "payload":
        with (damaged / "payload/recall.duckdb").open("ab") as handle:
            handle.write(b"tampered")
    else:
        file = damaged / "payload/recall.duckdb"
        other = lane / "foreign.duckdb"
        file.rename(other)
        if damage == "symlink":
            file.symlink_to(other)
        else:
            os.link(other, file)
    status, result = run_compare(lane, before, after)
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "refused")
    assert not (lane / "output").exists()


# The stub replaces only daemon.run. It is exercised through the real bench
# dispatcher in a fresh interpreter; pairs, the lock and result publication run normally.
DAEMON_STUB = """import sys, os, signal, psutil
sys.path.insert(0, sys.argv.pop(1))
import bench
from recall_bench import daemon
from recall_bench.evidence import JobFailed, JobIncomplete
from recall_bench.fs import Cancelled
from pathlib import Path
calls = Path(sys.argv.pop(1))
mode = sys.argv.pop(1)
fail_at = int(sys.argv.pop(1))
owner_pid = int(sys.argv.pop(1))
owner_birth = float(sys.argv.pop(1))
def run(args, result, cancellation, progress):
    n = len(calls.read_text().splitlines()) if calls.exists() else 0
    with calls.open('a') as f:
        f.write(Path(args.run_dir).name + '\\n')
    if n + 1 == fail_at:
        if mode == 'failed': raise JobFailed('stub', 'deliberate failed daemon-run')
        if mode == 'incomplete': raise JobIncomplete('stub', 'deliberate incomplete daemon-run')
        if mode == 'cancelled': raise Cancelled(15)
        if mode == 'reported-failed':
            result.end(JobFailed('stub','reported failure'))
            result.emit(); raise SystemExit(0)
        if mode == 'reported-incomplete':
            result.end(JobIncomplete('stub','reported incomplete'))
            result.emit(); raise SystemExit(0)
    result.scope = {'profile':'idle', 'build':{'head':'stub'}, 'tool':{'head':'stub'},
                    'fixture':{'payload':{}}}
    result.step('readiness',seconds=0)
    result.step('settle',seconds=0)
    result.step('window',cpu_percent=0)
    result.step('final',floors={'rss_bytes_max':0,'status_seconds_max':0})
    result.end(None)
    if mode == 'cancel-owner' and n + 1 == fail_at:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        result.emit()
        assert os.getppid() == owner_pid
        assert psutil.Process(owner_pid).create_time() == owner_birth
        os.kill(owner_pid, signal.SIGTERM)
        raise SystemExit(0)
daemon.run = run
raise SystemExit(bench.main(sys.argv[1:]))
"""


STUB = """import sys, os, psutil
sys.path.insert(0, sys.argv.pop(1))
import bench
from recall_bench import pairs
scripts, calls, mode, fail_at, worker = sys.argv[1:6]
del sys.argv[1:6]
original = pairs.daemon_command
def command(args):
    return [sys.executable, '-c', worker, scripts, calls, mode, fail_at,
            str(os.getpid()), str(psutil.Process().create_time()), *original(args)[2:]]
pairs.daemon_command = command
raise SystemExit(bench.main(sys.argv[1:]))
"""


def pair_args(lane: Path, template: Path) -> list[str | Path]:
    return [
        "pairs",
        "--lock",
        lane / "queue",
        "--baseline",
        ROOT,
        "--candidate",
        ROOT,
        "--template",
        template,
        "--run-parent",
        lane / "runs",
        "--output",
        lane / "batch",
        "--count",
        "3",
        "--",
        "--profile",
        "idle",
        "--window",
        "0.1",
    ]


@pytest.mark.parametrize(
    ("mode", "status", "outcome"),
    [
        ("passed", 0, "passed"),
        ("failed", 1, "failed"),
        ("incomplete", 2, "incomplete"),
        ("cancelled", 143, "incomplete"),
        ("cancel-owner", 143, "incomplete"),
        ("reported-failed", 1, "failed"),
        ("reported-incomplete", 2, "incomplete"),
    ],
)
@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_pairs_stops_at_first_nonpassed_daemon_run_stub(
    lane: Path, mode: str, status: int, outcome: str, fail_at: int
) -> None:
    template = prepare(lane, "template", SQL)
    calls = lane / "calls"
    code, result = invoke(
        lane,
        *pair_args(lane, template),
        prefix=[
            "-c",
            STUB,
            str(ROOT / "scripts"),
            str(ROOT / "scripts"),
            str(calls),
            mode,
            str(fail_at),
            DAEMON_STUB,
        ],
    )
    assert (code, result["outcome"]) == (status, outcome), result
    order = ["1-baseline", "1-candidate", "2-baseline", "2-candidate", "3-baseline", "3-candidate"]
    expected = order if mode == "passed" else order[:fail_at]
    assert calls.read_text().splitlines() == expected
    assert [step["variant"] for step in result["steps"]] == [s.split("-")[1] for s in expected]
    child_status = 0 if mode.startswith("reported-") or mode == "cancel-owner" else status
    assert list(result["exit_codes"].values()).count(0) == len(expected) - (child_status != 0)
    assert result["exit_codes"][expected[-1]] == child_status
    if mode == "cancel-owner":
        saved = json.loads((lane / "batch" / (expected[-1] + ".json")).read_bytes())
        assert saved["outcome"] == "passed"
        assert result["steps"][-1]["outcome"] == "passed"
    assert json.loads((lane / "batch/result.json").read_bytes()) == result
    # A new process acquires the exact queue after pairs, proving admission was released.
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/with_bench_lock.py"),
            "--lock",
            str(lane / "queue"),
            "--timeout",
            "1",
            "reacquire",
            sys.executable,
            "-c",
            "pass",
        ],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("extra", [["--build", "/foreign"], ["--hold"], ["--unknown"]])
def test_pairs_refuses_paths_or_hold_in_daemon_options(lane: Path, extra: list[str]) -> None:
    template = prepare(lane, "template", SQL)
    status, result = invoke(lane, *pair_args(lane, template), *extra)
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "usage")
    assert not (lane / "batch").exists()


def test_wal_rows_participate_without_changing_inputs(lane: Path) -> None:
    before = prepare(lane, "before", "CREATE TABLE items (n INTEGER); INSERT INTO items VALUES (1)")
    backup = lane / "wal-backup"
    backup.mkdir()
    path = backup / "recall.duckdb"
    with duckdb.connect(str(path), config={"threads": 2}) as connection:
        connection.execute("CREATE TABLE items (n INTEGER); INSERT INTO items VALUES (1)")
    # Abrupt exit preserves a real, recoverable WAL; it launches no recall code.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import duckdb,os,sys; "
            "c=duckdb.connect(sys.argv[1],config={'threads':2}); "
            "c.execute('INSERT INTO items VALUES (2)'); os._exit(0)",
            str(path),
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert path.with_suffix(".duckdb.wal").stat().st_size > 0
    after = lane / "after"
    status, result = invoke(lane, "prepare", "--backup", backup, "--template", after)
    assert (status, result["outcome"]) == (0, "passed"), result
    originals = snapshot(before), snapshot(after)
    status, result = run_compare(lane, before, after)
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "different")
    assert (
        result["steps"][-1]["tables"][0]["before_only"],
        result["steps"][-1]["tables"][0]["after_only"],
    ) == (0, 1)
    assert originals == (snapshot(before), snapshot(after))


@pytest.mark.parametrize("input_name", ["before", "after"])
def test_compare_output_cannot_be_inside_an_input(lane: Path, input_name: str) -> None:
    before = prepare(lane, "before", SQL)
    after = prepare(lane, "after", SQL)
    original = snapshot(before), snapshot(after)
    status, result = invoke(
        lane,
        "compare",
        "--before",
        before,
        "--after",
        after,
        "--output",
        lane / input_name / "output",
    )
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "refused")
    assert original == (snapshot(before), snapshot(after))


def test_existing_compare_output_is_never_overwritten(lane: Path) -> None:
    template = prepare(lane, "template", SQL)
    output = lane / "output"
    output.mkdir()
    (output / "keep").write_bytes(b"keep")
    status, result = run_compare(lane, template, template)
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "refused")
    assert snapshot(output) == {"keep": b"keep"}


@pytest.mark.parametrize("destination", ["lock", "run-parent", "output"])
def test_pairs_destinations_cannot_mutate_the_template(lane: Path, destination: str) -> None:
    template = prepare(lane, "template", SQL)
    original = snapshot(template)
    args = pair_args(lane, template)
    args[args.index("--" + destination) + 1] = (
        template if destination != "output" else template / "output"
    )
    status, result = invoke(lane, *args)
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "refused")
    assert snapshot(template) == original


def test_pairs_refuses_an_aliased_queue_mutex(lane: Path) -> None:
    template = prepare(lane, "template", SQL)
    queue = lane / "queue"
    queue.mkdir()
    sentinel = lane / "sentinel"
    sentinel.write_bytes(b"do not touch")
    (queue / "queue").symlink_to(sentinel)
    status, result = invoke(lane, *pair_args(lane, template))
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "refused")
    assert sentinel.read_bytes() == b"do not touch"
    assert not (lane / "batch").exists()


@pytest.mark.parametrize("which", ["before", "after"])
def test_compare_requires_database_before_output(lane: Path, which: str) -> None:
    valid = prepare(lane, "valid", SQL)
    backup = lane / "without-database"
    backup.mkdir()
    (backup / "other.txt").write_bytes(b"a valid backup payload, but not a database")
    missing = lane / "missing"
    status, result = invoke(lane, "prepare", "--backup", backup, "--template", missing)
    assert (status, result["outcome"]) == (0, "passed"), result
    before, after = (missing, valid) if which == "before" else (valid, missing)
    status, result = run_compare(lane, before, after)
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "refused")
    assert result["error"]["argument"] == "--" + which
    assert not (lane / "output").exists()


@pytest.mark.parametrize(
    "overlap", ["first-baseline", "first-candidate", "lock-output", "lock-runs"]
)
def test_pairs_paths_use_disjoint_namespaces_before_mutation(lane: Path, overlap: str) -> None:
    template = prepare(lane, "template", SQL)
    args = pair_args(lane, template)
    if overlap.startswith("first-"):
        args[args.index("--output") + 1] = lane / "runs" / ("1-" + overlap.removeprefix("first-"))
    elif overlap == "lock-output":
        args[args.index("--lock") + 1] = lane / "batch"
    else:
        args[args.index("--lock") + 1] = lane / "runs/queue"
    calls = lane / "calls"
    status, result = invoke(
        lane,
        *args,
        prefix=[
            "-c",
            STUB,
            str(ROOT / "scripts"),
            str(ROOT / "scripts"),
            str(calls),
            "passed",
            "1",
            DAEMON_STUB,
        ],
    )
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "refused")
    assert not calls.exists()
    assert list((lane / "runs").iterdir()) == []
    assert not (lane / "queue").exists()
    assert not (lane / "batch").exists()


def test_pairs_requires_database_before_any_arm_or_output(lane: Path) -> None:
    backup = lane / "without-database"
    backup.mkdir()
    (backup / "other.txt").write_bytes(b"no database")
    template = lane / "template"
    status, result = invoke(lane, "prepare", "--backup", backup, "--template", template)
    assert (status, result["outcome"]) == (0, "passed"), result
    calls = lane / "calls"
    status, result = invoke(
        lane,
        *pair_args(lane, template),
        prefix=[
            "-c",
            STUB,
            str(ROOT / "scripts"),
            str(ROOT / "scripts"),
            str(calls),
            "passed",
            "1",
            DAEMON_STUB,
        ],
    )
    assert (status, result["outcome"], result["error"]["kind"]) == (1, "failed", "refused")
    assert result["error"]["argument"] == "--template"
    assert not calls.exists()
    assert not (lane / "queue").exists()
    assert not (lane / "batch").exists()
