"""Drive bench isolation with real processes and files: refusals, `prepare`, run dirs.

The inode-reuse tests run only where the directory they use is on ext4, which
gives a freed inode number to the next entry, and skip elsewhere with the
reason. On ext4 a missed reuse fails them. A qualification run sets
`BENCH_REQUIRE_INODE_REUSE=1` so that a temporary directory on any other
filesystem fails these tests instead of skipping them.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import duckdb
import pytest
import typer.main
from recall.cli.app import app

from scripts.recall_bench import cli, fs, isolation
from scripts.recall_bench.fs import Cancellation, Cancelled, Identity, Refused, identity
from scripts.recall_bench.isolation import (
    SOURCES,
    Build,
    Mounts,
    Policy,
    RunDir,
    Template,
    UsageLog,
    WriteRoots,
    child_environment,
)
from scripts.recall_bench.prepare import build_template
from scripts.recall_bench.supervisor import Completed, Deadline, Supervisor

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "scripts/bench.py"
DUCK_SHA256 = "2d2370db2447ff8cf4f3accd68c85aa119a9c893effd200a9b69176e9fc5eb98"
FTS_SHA256 = "dd8f6640e1c144ff8cb56e681948ad4690c9333e62121f8fb8395feb04769ad0"
SESSION_SHA256 = "e9d6e33fc2fea3757c6a1e121276a412ca43028a7a1efc9b6a68b1182bd55c8e"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
POOL_REASON = "files must be under one of claude_code, codex, pi_agent, grok, kimi_code"
ALIAS_REASON = "multiply linked file; it may alias another file"
SOCKET_LIMIT = {"darwin": 104, "linux": 108}[sys.platform]
NEVER = Cancellation()


@pytest.fixture
def lane(tmp_path: Path) -> Path:
    """This test's write root, as a lane launcher would export it."""
    directory = tmp_path / "lane"
    directory.mkdir()
    return directory


@pytest.fixture
def canaries(tmp_path: Path) -> Path:
    """recall/launchctl/systemctl first on PATH that record any invocation."""
    directory = tmp_path / "canary-bin"
    directory.mkdir()
    marker = tmp_path / "canary-invocations"
    for name in ("recall", "launchctl", "systemctl"):
        script = directory / name
        script.write_text(
            f"#!{sys.executable}\nimport sys\n"
            f"open({str(marker)!r}, 'a').write(' '.join(sys.argv) + '\\n')\n"
        )
        script.chmod(0o755)
    return marker


def bench_environment(canaries: Path, lane: Path, **overrides: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PATH"] = f"{canaries.parent / 'canary-bin'}:{environment['PATH']}"
    environment["BENCH_WRITE_ROOTS"] = str(lane)
    environment.update(overrides)
    return environment


def bench(
    *args: str | Path, canaries: Path, lane: Path, env: dict[str, str] | None = None
) -> tuple[int, dict]:
    completed = subprocess.run(
        [sys.executable, str(BENCH), *map(str, args)],
        capture_output=True,
        text=True,
        env=bench_environment(canaries, lane) if env is None else env,
        timeout=60,
        check=False,
    )
    lines = completed.stdout.strip().splitlines()
    assert lines, completed.stderr
    return completed.returncode, json.loads(lines[-1])


def snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


@contextmanager
def write_roots(*roots: Path) -> Iterator[WriteRoots]:
    policy = Policy.for_host()
    with policy.open_write_roots({"BENCH_WRITE_ROOTS": ":".join(map(str, roots))}) as opened:
        yield opened


@pytest.fixture
def backup(tmp_path: Path) -> Path:
    directory = tmp_path / "backup"
    directory.mkdir()
    (directory / "recall.duckdb").write_bytes(b"duck")
    (directory / "recall.fts.sqlite").write_bytes(b"fts")
    return directory


@pytest.fixture
def pool(tmp_path: Path) -> Path:
    directory = tmp_path / "pool"
    (directory / "claude_code/project").mkdir(parents=True)
    (directory / "claude_code/project/s.jsonl").write_bytes(b"session\n")
    (directory / "codex/empty").mkdir(parents=True)
    return directory


@pytest.fixture
def template(tmp_path: Path, lane: Path, backup: Path, pool: Path, canaries: Path) -> Path:
    usage = tmp_path / "usage.jsonl"
    usage.write_bytes(b"")
    path = lane / "template"
    status, line = bench(
        "prepare",
        "--backup", backup,
        "--template", path,
        "--source-pool", pool,
        "--usage-log", usage,
        canaries=canaries,
        lane=lane,
    )  # fmt: skip
    assert (status, line["outcome"]) == (0, "passed"), line
    return path


def own_build() -> Build:
    """This checkout as the build under test."""
    return Build.resolve(Policy.for_host(), "--build", ROOT)


@contextmanager
def opened(template: Path) -> Iterator[Template]:
    with Template.open(Policy.for_host(), "--t", template, NEVER) as held:
        yield held


def test_prepare_writes_a_verified_manifest_and_touches_nothing_else(
    tmp_path: Path, backup: Path, pool: Path, template: Path, canaries: Path
) -> None:
    manifest = json.loads((template / "manifest.json").read_bytes())
    assert manifest["format"] == "recall-bench-template/2"
    assert manifest["directories"] == [
        "payload",
        "pool",
        "pool/claude_code",
        "pool/claude_code/project",
        "pool/codex",
        "pool/codex/empty",
    ]
    assert manifest["files"] == [
        {"path": "payload/recall.duckdb", "bytes": 4, "sha256": DUCK_SHA256},
        {"path": "payload/recall.fts.sqlite", "bytes": 3, "sha256": FTS_SHA256},
        {"path": "pool/claude_code/project/s.jsonl", "bytes": 8, "sha256": SESSION_SHA256},
        {"path": "usage.jsonl", "bytes": 0, "sha256": EMPTY_SHA256},
    ]
    assert manifest["provenance"]["backup"] == str(backup)
    assert (template / "payload/recall.duckdb").read_bytes() == b"duck"
    assert snapshot(backup) == {"recall.duckdb": b"duck", "recall.fts.sqlite": b"fts"}
    assert snapshot(pool) == {
        "claude_code": None,
        "claude_code/project": None,
        "claude_code/project/s.jsonl": b"session\n",
        "codex": None,
        "codex/empty": None,
    }
    assert list(Path.home().iterdir()) == []
    assert not canaries.exists()
    with opened(template) as held:
        assert held.path == template


def test_prepare_refuses_an_existing_template_and_leaves_it_unchanged(
    lane: Path, backup: Path, canaries: Path
) -> None:
    existing = lane / "existing"
    existing.mkdir()
    (existing / "keep").write_bytes(b"keep")
    status, line = bench(
        "prepare", "--backup", backup, "--template", existing, canaries=canaries, lane=lane
    )
    assert (status, line["outcome"]) == (1, "failed")
    assert line["error"]["reason"] == "already exists; every run uses a new directory"
    assert snapshot(existing) == {"keep": b"keep"}


def test_prepare_needs_write_roots_from_its_launcher(
    lane: Path, backup: Path, canaries: Path
) -> None:
    environment = bench_environment(canaries, lane)
    del environment["BENCH_WRITE_ROOTS"]
    status, line = bench(
        "prepare",
        "--backup", backup,
        "--template", lane / "t",
        canaries=canaries,
        lane=lane,
        env=environment,
    )  # fmt: skip
    assert (status, line["error"]["argument"]) == (1, "BENCH_WRITE_ROOTS")
    assert line["error"]["reason"] == (
        "is not set; the lane launcher names the directories bench jobs may write in"
    )
    assert list(lane.iterdir()) == []


def test_a_copied_command_cannot_write_into_another_lane(
    tmp_path: Path, lane: Path, backup: Path, canaries: Path
) -> None:
    other = tmp_path / "other-lane"
    (other / "scratch").mkdir(parents=True)
    (other / "scratch/report.md").write_bytes(b"theirs")
    status, line = bench(
        "prepare",
        "--backup", backup,
        "--template", other / "scratch/template",
        canaries=canaries,
        lane=lane,
    )  # fmt: skip
    assert (status, line["outcome"]) == (1, "failed")
    assert line["error"]["reason"] == f"is outside every write root ({lane})"
    assert snapshot(other) == {"scratch": None, "scratch/report.md": b"theirs"}


def test_a_write_root_may_not_hold_production_state(
    tmp_path: Path, backup: Path, canaries: Path
) -> None:
    status, line = bench(
        "prepare",
        "--backup", backup,
        "--template", Path.home() / ".local/t",
        canaries=canaries,
        lane=Path.home(),
    )  # fmt: skip
    assert (status, line["error"]["argument"]) == (1, "BENCH_WRITE_ROOTS")
    assert line["error"]["reason"] == f"is or contains the account home {Path.home()}"


REFUSED_LOCATIONS: dict[str, Callable[[Path], Path]] = {
    "data dir": lambda home: home / ".local/share/recall",
    "config dir": lambda home: home / ".config/recall",
    "xdg state dir": lambda home: Path(os.environ["XDG_STATE_HOME"]) / "recall",
    "stable tool install": lambda home: home / ".local/share/uv/tools/recall",
    "stable bin": lambda home: home / ".local/bin",
    "launch agents": lambda home: home / "Library/LaunchAgents",
    "systemd user units": lambda home: home / ".config/systemd/user",
    "tool checkout": lambda _home: ROOT,
}


@pytest.fixture
def refused_tree(request: pytest.FixtureRequest) -> Path:
    location = REFUSED_LOCATIONS[request.param](Path.home())
    if location != ROOT:
        location.mkdir(parents=True)
        (location / "state").write_bytes(b"production")
    return location


def _argument_path(argument: str, location: Path) -> Path:
    if argument == "--template":
        return location / "refused-template"
    if argument == "--usage-log":
        return location / ("pyproject.toml" if location == ROOT else "state")
    return location


def _prepare_arguments(lane: Path, backup: Path, argument: str, path: Path) -> list[Path | str]:
    arguments: dict[str, Path] = {"--backup": backup, "--template": lane / "template"}
    arguments[argument] = path
    return [part for pair in arguments.items() for part in pair]


@pytest.mark.parametrize("argument", ["--backup", "--template", "--source-pool", "--usage-log"])
@pytest.mark.parametrize("refused_tree", sorted(REFUSED_LOCATIONS), indirect=True)
def test_every_prepare_argument_refuses_every_refused_location(
    lane: Path, backup: Path, canaries: Path, argument: str, refused_tree: Path
) -> None:
    before = snapshot(refused_tree) if refused_tree != ROOT else None

    status, line = bench(
        "prepare",
        *_prepare_arguments(lane, backup, argument, _argument_path(argument, refused_tree)),
        canaries=canaries,
        lane=lane,
    )

    assert (status, line["outcome"]) == (1, "failed"), line
    assert line["error"]["argument"] == argument
    assert line["error"]["reason"] == f"overlaps refused location {refused_tree}"
    assert list(lane.iterdir()) == []
    assert not (refused_tree / "refused-template").exists()
    if before is not None:
        assert snapshot(refused_tree) == before
    assert not canaries.exists()


ENVIRONMENT_LOCATIONS = {
    "XDG_CONFIG_HOME systemd units": ("XDG_CONFIG_HOME", "xdg-config", "systemd/user"),
    "XDG_DATA_HOME uv tool": ("XDG_DATA_HOME", "xdg/share", "uv/tools/recall"),
    "XDG_DATA_HOME uv bin": ("XDG_DATA_HOME", "xdg/share", "../bin"),
    "UV_TOOL_DIR": ("UV_TOOL_DIR", "uv-tools", "recall"),
    "UV_TOOL_BIN_DIR": ("UV_TOOL_BIN_DIR", "uv-bin", "."),
    "XDG_BIN_HOME": ("XDG_BIN_HOME", "xdg-bin", "."),
}


@pytest.mark.parametrize("argument", ["--backup", "--template"])
@pytest.mark.parametrize("location", sorted(ENVIRONMENT_LOCATIONS))
def test_service_and_tool_locations_follow_the_environment(
    tmp_path: Path, lane: Path, backup: Path, canaries: Path, argument: str, location: str
) -> None:
    variable, base, relative = ENVIRONMENT_LOCATIONS[location]
    protected = Path(os.path.normpath(tmp_path / base / relative))
    protected.mkdir(parents=True)
    (protected / "recall-daemon.service").write_bytes(b"unit")
    path = protected / "refused-template" if argument == "--template" else protected
    status, line = bench(
        "prepare",
        *_prepare_arguments(lane, backup, argument, path),
        canaries=canaries,
        lane=lane,
        env=bench_environment(canaries, lane, **{variable: str(tmp_path / base)}),
    )
    assert (status, line["error"]["argument"]) == (1, argument)
    assert line["error"]["reason"] == f"overlaps refused location {protected}"
    assert snapshot(protected) == {"recall-daemon.service": b"unit"}


@pytest.mark.parametrize("argument", ["--backup", "--template", "--source-pool", "--usage-log"])
def test_every_prepare_argument_refuses_the_account_home(
    lane: Path, backup: Path, canaries: Path, argument: str
) -> None:
    status, line = bench(
        "prepare",
        *_prepare_arguments(lane, backup, argument, Path.home()),
        canaries=canaries,
        lane=lane,
    )
    assert (status, line["outcome"]) == (1, "failed"), line
    assert line["error"]["reason"] == f"is or contains the account home {Path.home()}"
    assert list(Path.home().iterdir()) == []


def test_a_directory_containing_production_state_is_refused(lane: Path, canaries: Path) -> None:
    data = Path.home() / ".local/share/recall"
    data.mkdir(parents=True)
    status, line = bench(
        "prepare",
        "--backup", Path.home() / ".local/share",
        "--template", lane / "template",
        canaries=canaries,
        lane=lane,
    )  # fmt: skip
    assert (status, line["error"]["reason"]) == (1, f"overlaps refused location {data}")


def test_an_unreadable_protected_location_is_compared_by_path(tmp_path: Path) -> None:
    # A mount namespace maps the account to root, whose passwd home this user cannot read.
    cache = Path.home() / ".cache"
    cache.mkdir(mode=0o000)
    try:
        with pytest.raises(Refused) as refused:
            Policy.for_host().check_spelling("--backup", cache / "recall/backup")
    finally:
        cache.chmod(0o755)
    assert refused.value.reason == f"overlaps refused location {cache / 'recall'}"


def test_another_spelling_of_production_state_is_refused(lane: Path, canaries: Path) -> None:
    data = Path.home() / ".local/share/recall"
    data.mkdir(parents=True)
    respelled = Path.home() / ".LOCAL/share/recall"
    if not respelled.exists():
        pytest.skip("volume is case-sensitive; there is no second spelling")
    status, line = bench(
        "prepare", "--backup", respelled, "--template", lane / "t", canaries=canaries, lane=lane
    )
    assert (status, line["error"]["reason"]) == (1, f"overlaps refused location {data}")


def _symlinked_component(tmp_path: Path, backup: Path) -> Path:
    (tmp_path / "link").symlink_to(backup)
    return tmp_path / "link"


def _relative(tmp_path: Path, backup: Path) -> Path:
    return Path(os.path.relpath(backup))


def _symlinked_payload(tmp_path: Path, backup: Path) -> Path:
    (backup / "recall.fts.sqlite").unlink()
    (backup / "recall.fts.sqlite").symlink_to(backup / "recall.duckdb")
    return backup


def _hardlinked_payload(tmp_path: Path, backup: Path) -> Path:
    # A second name for production bytes, made before the bench ever ran.
    os.link(backup / "recall.duckdb", tmp_path / "production-recall.duckdb")
    return backup


def _fifo_payload(tmp_path: Path, backup: Path) -> Path:
    os.mkfifo(backup / "recall.duckdb.wal")
    return backup


def _nested_payload(tmp_path: Path, backup: Path) -> Path:
    (backup / "nested").mkdir()
    (backup / "nested/file").write_bytes(b"x")
    return backup


def _empty_nested_directory(tmp_path: Path, backup: Path) -> Path:
    (backup / "nested").mkdir()
    return backup


def _reserved_payload(tmp_path: Path, backup: Path) -> Path:
    (backup / "config.toml").write_bytes(b"")
    return backup


def _empty_backup(tmp_path: Path, backup: Path) -> Path:
    empty = tmp_path / "empty"
    empty.mkdir()
    return empty


@pytest.mark.parametrize(
    ("make_backup", "reason"),
    [
        (_relative, "path must be absolute"),
        (_symlinked_component, "symlinked component {tmp}/link"),
        (_symlinked_payload, "symlinked payload file"),
        (_hardlinked_payload, ALIAS_REASON),
        (_fifo_payload, "not a regular file"),
        (_nested_payload, "backup payload must be flat files"),
        (_empty_nested_directory, "backup payload must be flat files"),
        (_reserved_payload, "name is reserved in a run directory"),
        (_empty_backup, "holds no files"),
    ],
)
def test_prepare_refuses_backups_it_cannot_copy_faithfully(
    tmp_path: Path,
    lane: Path,
    backup: Path,
    canaries: Path,
    make_backup: Callable[[Path, Path], Path],
    reason: str,
) -> None:
    source = make_backup(tmp_path, backup)
    status, line = bench(
        "prepare", "--backup", source, "--template", lane / "template", canaries=canaries, lane=lane
    )
    assert (status, line["outcome"]) == (1, "failed"), line
    assert line["error"]["reason"] == reason.format(tmp=tmp_path)
    assert list(lane.iterdir()) == []


@pytest.mark.parametrize(
    ("layout", "reason"),
    [
        ("symlink", "symlinked payload file"),
        ("hardlink", ALIAS_REASON),
        ("unknown source", POOL_REASON),
        ("top-level file", POOL_REASON),
    ],
)
def test_prepare_refuses_source_pools_it_cannot_pin(
    tmp_path: Path,
    lane: Path,
    backup: Path,
    pool: Path,
    canaries: Path,
    layout: str,
    reason: str,
) -> None:
    session = pool / "claude_code/project/s.jsonl"
    if layout == "symlink":
        (pool / "claude_code/project/link.jsonl").symlink_to(session)
    elif layout == "hardlink":
        os.link(session, tmp_path / "live-session.jsonl")
    elif layout == "unknown source":
        (pool / "claude").mkdir()
        (pool / "claude/s.jsonl").write_bytes(b"")
    else:
        (pool / "notes.jsonl").write_bytes(b"")
    status, line = bench(
        "prepare",
        "--backup", backup,
        "--template", lane / "template",
        "--source-pool", pool,
        canaries=canaries,
        lane=lane,
    )  # fmt: skip
    assert (status, line["error"]["reason"]) == (1, reason)
    assert list(lane.iterdir()) == []


@pytest.mark.parametrize(
    ("make_log", "reason"),
    [
        ("symlink", "symlinked component {log}"),
        ("hardlink", ALIAS_REASON),
    ],
)
def test_prepare_refuses_a_usage_log_with_another_name(
    tmp_path: Path, lane: Path, backup: Path, canaries: Path, make_log: str, reason: str
) -> None:
    real = tmp_path / "unified.jsonl"
    real.write_bytes(b"{}\n")
    log = tmp_path / "usage.jsonl"
    if make_log == "symlink":
        log.symlink_to(real)
    else:
        os.link(real, log)
    status, line = bench(
        "prepare",
        "--backup", backup,
        "--template", lane / "template",
        "--usage-log", log,
        canaries=canaries,
        lane=lane,
    )  # fmt: skip
    assert (status, line["error"]["reason"]) == (1, reason.format(log=log))
    assert list(lane.iterdir()) == []


def test_an_invalid_invocation_fails_rather_than_reporting_incomplete(
    lane: Path, canaries: Path
) -> None:
    status, line = bench("prepare", "--backup", canaries=canaries, lane=lane)
    assert (status, line["outcome"], line["error"]["kind"]) == (1, "failed", "usage")


def _flip_byte(template: Path) -> str:
    (template / "payload/recall.duckdb").write_bytes(b"dusk")
    return "manifest mismatch: sha256 differs"


def _add_file(template: Path) -> str:
    (template / "payload/recall.duckdb.wal").write_bytes(b"")
    return "manifest mismatch: missing [], unexpected ['payload/recall.duckdb.wal']"


def _add_empty_directory(template: Path) -> str:
    (template / "pool/grok").mkdir()
    return "manifest mismatch: missing directories [], unexpected ['pool/grok']"


def _remove_file(template: Path) -> str:
    (template / "usage.jsonl").unlink()
    return "manifest mismatch: missing ['usage.jsonl'], unexpected []"


def _symlink_file(template: Path) -> str:
    (template / "payload/recall.fts.sqlite").unlink()
    (template / "payload/recall.fts.sqlite").symlink_to(template / "payload/recall.duckdb")
    return "symlinked payload file"


def _hardlink_file(template: Path) -> str:
    os.link(template / "payload/recall.duckdb", template.parent / "alias")
    return ALIAS_REASON


def _edit_manifest(template: Path) -> str:
    manifest = template / "manifest.json"
    manifest.write_text(manifest.read_text().replace(DUCK_SHA256, FTS_SHA256))
    return "manifest mismatch: sha256 differs"


def _collide_in_run_dir(template: Path) -> str:
    manifest = template / "manifest.json"
    document = json.loads(manifest.read_bytes())
    document["files"].append({"path": "payload/usage.jsonl", "bytes": 0, "sha256": EMPTY_SHA256})
    manifest.write_text(json.dumps(document))
    return (
        "invalid manifest: manifest file payload/usage.jsonl: name is reserved in a run directory"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        _flip_byte,
        _add_file,
        _add_empty_directory,
        _remove_file,
        _symlink_file,
        _hardlink_file,
        _edit_manifest,
        _collide_in_run_dir,
    ],
)
def test_a_template_that_no_longer_matches_its_manifest_cannot_be_opened(
    template: Path, mutate: Callable[[Path], str]
) -> None:
    reason = mutate(template)
    with pytest.raises(Refused) as refused, opened(template):
        pass
    assert refused.value.reason == reason


def test_a_run_dir_is_a_verified_clone_with_every_path_pinned_inside(
    template: Path, lane: Path
) -> None:
    with write_roots(lane) as roots, opened(template) as held:
        run = RunDir.create(
            roots, "--run-dir", lane / "r", held, own_build(), UsageLog.EMPTY, NEVER
        )

    assert (run.path / "recall.duckdb").read_bytes() == b"duck"
    assert (run.path / "pool/claude_code/project/s.jsonl").read_bytes() == b"session\n"
    assert (run.path / "pool/codex/empty").is_dir()
    pool = lane / "r/pool"
    assert run.config_path.read_text() == (
        '[daemon]\nmode = "watch"\nembed = false\n\n'
        f'[duckdb]\ntemp_directory = "{lane}/r/spill"\n\n'
        f'[sources.claude_code]\nroots = ["{pool}/claude_code"]\n\n'
        f'[sources.codex]\nroots = ["{pool}/codex"]\n\n'
        f'[sources.pi_agent]\nroots = ["{pool}/pi_agent"]\n\n'
        f'[sources.grok]\nroots = ["{pool}/grok"]\n\n'
        f'[sources.kimi_code]\nroots = ["{pool}/kimi_code"]\n'
    )
    assert all((pool / source).is_dir() for source in ("pi_agent", "grok", "kimi_code"))
    (run.path / "recall.duckdb").write_bytes(b"used")
    with opened(template) as again:
        assert again.path == template


def test_a_run_dir_is_single_use(template: Path, lane: Path) -> None:
    with write_roots(lane) as roots, opened(template) as held:
        RunDir.create(roots, "--run-dir", lane / "r", held, own_build(), UsageLog.EMPTY, NEVER)
        with pytest.raises(Refused) as refused:
            RunDir.create(roots, "--run-dir", lane / "r", held, own_build(), UsageLog.EMPTY, NEVER)
    assert refused.value.reason == "already exists; every run uses a new directory"


def _run_dir_with_socket_bytes(parent: Path, total: int) -> Path:
    """A run dir whose socket path, with its NUL, takes exactly `total` bytes."""
    # Two-byte characters show the limit counts encoded bytes, not characters.
    fixed = len(os.fsencode(parent)) + len("/") + len("/recall.sock") + 1
    remaining = total - fixed
    assert remaining > 2, "test root is too deep for this socket check"
    return parent / ("é" * (remaining // 2) + "x" * (remaining % 2))


def test_socket_path_length_counts_encoded_bytes_and_nul(template: Path, lane: Path) -> None:
    fits = _run_dir_with_socket_bytes(lane, SOCKET_LIMIT)
    too_long = _run_dir_with_socket_bytes(lane / "x", SOCKET_LIMIT + 1)
    (lane / "x").mkdir()
    with write_roots(lane) as roots, opened(template) as held:
        assert (
            RunDir.create(
                roots, "--run-dir", fits, held, own_build(), UsageLog.EMPTY, NEVER
            ).socket_path.exists()
            is False
        )
        with pytest.raises(Refused) as refused:
            RunDir.create(roots, "--run-dir", too_long, held, own_build(), UsageLog.EMPTY, NEVER)
    assert refused.value.reason == (
        f"socket path {too_long}/recall.sock needs {SOCKET_LIMIT + 1} bytes with NUL; "
        f"the limit is {SOCKET_LIMIT}"
    )
    assert not too_long.exists()


def test_a_parent_replaced_after_resolution_is_not_followed(lane: Path) -> None:
    production = Path.home() / ".local/share/recall"
    production.mkdir(parents=True)
    (lane / "runs").mkdir()
    with write_roots(lane) as roots, roots.new_directory("--run-dir", lane / "runs/r") as target:
        # Another lane's cleanup swaps the checked parent for a link into production.
        (lane / "runs").rename(lane / "runs-moved")
        (lane / "runs").symlink_to(production)
        with target.create() as tree:
            tree.write_new(tree.root, PurePosixPath("evidence"), b"run")
            tree.commit(NEVER)
    assert list(production.iterdir()) == []
    assert (lane / "runs-moved/r/evidence").read_bytes() == b"run"


def test_publication_never_replaces_an_existing_name(lane: Path) -> None:
    with (
        write_roots(lane) as roots,
        roots.new_directory("--output", lane / "out") as target,
        target.create() as tree,
    ):
        tree.publish(PurePosixPath("result.json"), b"first")
        with pytest.raises(FileExistsError):
            tree.publish(PurePosixPath("result.json"), b"second")
        (lane / "out/planted.json").symlink_to(lane / "elsewhere")
        with pytest.raises(FileExistsError):
            tree.publish(PurePosixPath("planted.json"), b"third")
        tree.commit(NEVER)
    assert (lane / "out/result.json").read_bytes() == b"first"
    assert not (lane / "elsewhere").exists()
    assert sorted(path.name for path in (lane / "out").iterdir()) == ["planted.json", "result.json"]


def test_an_uncommitted_tree_removes_only_what_it_created(lane: Path) -> None:
    with write_roots(lane) as roots:
        with roots.new_directory("--output", lane / "out") as target, target.create() as tree:
            directory = tree.make_directory(tree.root, PurePosixPath("logs"))
            tree.write_new(directory, PurePosixPath("logs/a"), b"a")
            directory.close()
            (lane / "out/logs/theirs").write_bytes(b"theirs")
        leftovers = roots.leftovers
    assert snapshot(lane) == {"out": None, "out/logs": None, "out/logs/theirs": b"theirs"}
    assert leftovers == [
        {"path": str(lane / "out/logs"), "reason": os.strerror(errno.ENOTEMPTY)},
        {"path": str(lane / "out"), "reason": os.strerror(errno.ENOTEMPTY)},
    ]


def test_cleanup_leaves_entries_under_a_replaced_directory(lane: Path) -> None:
    with write_roots(lane) as roots:
        with roots.new_directory("--output", lane / "out") as target, target.create() as tree:
            directory = tree.make_directory(tree.root, PurePosixPath("logs"))
            tree.write_new(directory, PurePosixPath("logs/a"), b"ours")
            directory.close()
            # Another operation moves the directory aside and puts its own in place.
            (lane / "out/logs").rename(lane / "out/logs-moved")
            (lane / "out/logs").mkdir()
            (lane / "out/logs/a").write_bytes(b"theirs")
        leftovers = roots.leftovers
    assert snapshot(lane) == {
        "out": None,
        "out/logs": None,
        "out/logs/a": b"theirs",
        "out/logs-moved": None,
        "out/logs-moved/a": b"ours",
    }
    replaced = "replaced since this job created it; left in place"
    assert leftovers == [
        {
            "path": str(lane / "out/logs/a"),
            "reason": "its directory was replaced since this job created it",
        },
        {"path": str(lane / "out/logs"), "reason": replaced},
        {"path": str(lane / "out"), "reason": os.strerror(errno.ENOTEMPTY)},
    ]


def test_cleanup_does_not_remove_a_replaced_root(lane: Path) -> None:
    with write_roots(lane) as roots:
        with roots.new_directory("--output", lane / "out") as target, target.create():
            (lane / "out").rename(lane / "ours")
            (lane / "out").mkdir()
        leftovers = roots.leftovers
    assert snapshot(lane) == {"ours": None, "out": None}
    assert leftovers == [
        {"path": str(lane / "out"), "reason": "replaced since this job created it; left in place"}
    ]


REUSE_ATTEMPTS = 20
REQUIRE_INODE_REUSE = "BENCH_REQUIRE_INODE_REUSE"
# linux/magic.h: ext2, ext3 and ext4 share EXT4_SUPER_MAGIC.
EXT4_SUPER_MAGIC = 0xEF53
TMPFS_MAGIC = 0x01021994


def _linux_filesystem_magic(directory: Path) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    buffer = ctypes.create_string_buffer(4096)
    if libc.statfs(os.fsencode(directory), buffer) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(directory))
    # `f_type` is the first field, a native long.
    return ctypes.c_long.from_buffer(buffer).value


def _why_inode_numbers_are_not_reused(directory: Path) -> str | None:
    if sys.platform != "linux":
        # macOS's supported filesystem, APFS, numbers inodes from a counter.
        return f"{directory} is not on Linux ext4"
    magic = _linux_filesystem_magic(directory)
    if magic == EXT4_SUPER_MAGIC:
        return None
    if magic == TMPFS_MAGIC:
        return f"{directory} is on tmpfs, which numbers inodes from a counter"
    return f"{directory} is on filesystem {magic:#x}, which is not known to reuse inode numbers"


def _require_inode_reuse(directory: Path) -> None:
    """Skip unless `directory` is on ext4; fail instead under `BENCH_REQUIRE_INODE_REUSE=1`."""
    reason = _why_inode_numbers_are_not_reused(directory)
    if reason is None:
        return
    if os.environ.get(REQUIRE_INODE_REUSE) == "1":
        pytest.fail(f"{REQUIRE_INODE_REUSE}=1, but {reason}")
    pytest.skip(reason)


def _make_again_on_its_inode(
    path: Path, remove: Callable[[], object], make: Callable[[], object]
) -> None:
    """Remove `path` and make another entry there until it has the freed `(st_dev, st_ino)`."""
    _require_inode_reuse(path.parent)
    freed = identity(path.stat())
    for _ in range(REUSE_ATTEMPTS):
        remove()
        make()
        if identity(path.stat()) == freed:
            return
    pytest.fail(f"no new entry at {path} got inode {freed} in {REUSE_ATTEMPTS} tries")


def _recreate_on_its_inode_number(path: Path) -> None:
    """Remove an empty directory and make another at its name on the inode it freed."""
    _make_again_on_its_inode(path, path.rmdir, path.mkdir)


def _rewrite_on_its_inode_number(path: Path, data: bytes) -> None:
    """Remove a file and write another at its name on the inode it freed."""
    _make_again_on_its_inode(path, path.unlink, lambda: path.write_bytes(data))


def test_cleanup_leaves_a_directory_made_on_a_removed_one_s_inode_number(lane: Path) -> None:
    with write_roots(lane) as roots:
        with roots.new_directory("--output", lane / "out") as target, target.create() as tree:
            tree.make_directory(tree.root, PurePosixPath("logs")).close()
            # A careless cleanup removes the empty directory, and another
            # operation makes its own at the name, on the same inode number.
            _recreate_on_its_inode_number(lane / "out/logs")
        leftovers = roots.leftovers
    assert snapshot(lane) == {"out": None, "out/logs": None}
    replaced = "replaced since this job created it; left in place"
    assert leftovers == [
        {"path": str(lane / "out/logs"), "reason": replaced},
        {"path": str(lane / "out"), "reason": os.strerror(errno.ENOTEMPTY)},
    ]


def test_cleanup_leaves_a_file_written_on_a_removed_one_s_inode_number(lane: Path) -> None:
    with write_roots(lane) as roots:
        with roots.new_directory("--output", lane / "out") as target, target.create() as tree:
            tree.write_new(tree.root, PurePosixPath("result.json"), b"ours")
            _rewrite_on_its_inode_number(lane / "out/result.json", b"theirs")
        leftovers = roots.leftovers
    assert snapshot(lane) == {"out": None, "out/result.json": b"theirs"}
    replaced = "replaced since this job created it; left in place"
    assert leftovers == [
        {"path": str(lane / "out/result.json"), "reason": replaced},
        {"path": str(lane / "out"), "reason": os.strerror(errno.ENOTEMPTY)},
    ]


def test_a_write_root_that_cannot_tell_a_re_created_entry_apart_is_refused(
    lane: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_generation(*_args: object) -> None:
        raise OSError(errno.ENOTTY, os.strerror(errno.ENOTTY))

    # A filesystem with no inode generation that may reuse inode numbers.
    monkeypatch.setattr(fs.fcntl, "ioctl", no_generation)
    monkeypatch.setattr(fs, "_numbers_inodes_from_a_counter", lambda _descriptor: False)
    with pytest.raises(Refused) as refused, write_roots(lane):
        pass
    assert (refused.value.argument, refused.value.reason) == (
        "BENCH_WRITE_ROOTS",
        fs.UNIDENTIFIABLE_REASON,
    )
    assert list(lane.iterdir()) == []


def test_a_bind_mount_is_refused_by_its_mount_root(tmp_path: Path) -> None:
    mounts = Mounts({1: "/", 7: "/home/user/.local/share/recall/sub"})
    with pytest.raises(Refused) as refused:
        mounts.check("--backup", tmp_path / "bound", 7)
    assert refused.value.reason == "is inside a bind mount of /home/user/.local/share/recall/sub"
    with pytest.raises(Refused) as unknown:
        mounts.check("--backup", tmp_path / "late", 9)
    assert unknown.value.reason == "is on a mount that appeared after the job started"
    mounts.check("--backup", tmp_path, 1)


def _mount_namespace() -> list[str]:
    """A prefix that runs its command in a new user and mount namespace."""
    if sys.platform != "linux":
        pytest.skip("bind mounts are a Linux mechanism")
    probe = subprocess.run(["unshare", "-Urm", "true"], capture_output=True, check=False)
    if probe.returncode != 0:
        pytest.skip(f"no unprivileged mount namespace: {probe.stderr!r}")
    return ["unshare", "-Urm"]


def _unshare() -> list[str]:
    """`_mount_namespace` that first binds its first argument over its second."""
    return [
        *_mount_namespace(),
        *("sh", "-c", 'mount --bind "$1" "$2" && shift 2 && exec "$@"', "sh"),
    ]


@pytest.mark.parametrize("where", ["backup root", "pool directory"])
def test_a_bind_mounted_protected_tree_is_refused(
    tmp_path: Path, lane: Path, backup: Path, pool: Path, canaries: Path, where: str
) -> None:
    protected = Path.home() / ".local/share/recall/snapshots"
    protected.mkdir(parents=True)
    (protected / "recall.duckdb").write_bytes(b"production")
    if where == "backup root":
        mount_point, arguments = tmp_path / "bound", ["--backup", tmp_path / "bound"]
        reason = f"is inside a bind mount of {protected}"
    else:
        mount_point = pool / "claude_code/bound"
        arguments = ["--backup", backup, "--source-pool", pool]
        reason = "crosses a mount point"
    mount_point.mkdir()
    completed = subprocess.run(
        [
            *_unshare(),
            str(protected),
            str(mount_point),
            sys.executable,
            str(BENCH),
            "prepare",
            *map(str, arguments),
            "--template",
            str(lane / "template"),
        ],
        capture_output=True,
        text=True,
        env=bench_environment(canaries, lane),
        timeout=60,
        check=False,
    )
    line = json.loads(completed.stdout.strip().splitlines()[-1])
    assert (completed.returncode, line["error"]["reason"]) == (1, reason), completed.stderr
    assert list(lane.iterdir()) == []
    assert snapshot(protected) == {"recall.duckdb": b"production"}


def test_a_link_planted_at_a_copy_destination_is_not_followed(
    tmp_path: Path, lane: Path, backup: Path
) -> None:
    outside = tmp_path / "outside"
    with (
        write_roots(lane) as roots,
        Policy.for_host().open_tree("--backup", backup) as source,
        roots.new_directory("--template", lane / "t") as target,
        target.create() as tree,
        source.open_file("--backup", "recall.duckdb") as file,
    ):
        (lane / "t/recall.duckdb").symlink_to(outside)
        with pytest.raises(FileExistsError):
            tree.clone(tree.root, PurePosixPath("recall.duckdb"), file, NEVER)
        assert sorted(path.name for path in (lane / "t").iterdir()) == ["recall.duckdb"]
    assert not outside.exists()


def test_only_the_macos_private_links_are_followed_at_the_root() -> None:
    policy = Policy.for_host()
    with pytest.raises(Refused) as refused:
        policy.open_file("--usage-log", "/etc/localtime")
    assert refused.value.reason == "symlinked component /etc/localtime"
    if sys.platform != "darwin":
        return
    with (
        policy.open_write_roots({"BENCH_WRITE_ROOTS": "/tmp"}) as roots,
        policy.open_tree("--backup", "/private/tmp") as real,
    ):
        assert [(root.path, root.identity) for root in roots.held] == [
            (Path("/tmp"), real.identity)
        ]


def _prepare_in_process(
    lane: Path,
    backup: Path,
    on_event: Callable[[dict[str, object]], None],
    cancellation: Cancellation = NEVER,
) -> None:
    policy = Policy.for_host()
    with ExitStack() as held:
        roots = held.enter_context(write_roots(lane))
        source = held.enter_context(policy.open_tree("--backup", backup))
        target = held.enter_context(roots.new_directory("--template", lane / "template"))
        build_template(target, source, None, None, cancellation, on_event, provenance={})


def test_a_file_added_while_copying_fails_prepare(lane: Path, backup: Path) -> None:
    def add(event: dict[str, object]) -> None:
        if event == {"event": "clone", "file": "payload/recall.duckdb"}:
            (backup / "recall.duckdb.wal").write_bytes(b"late")

    with pytest.raises(Refused) as refused:
        _prepare_in_process(lane, backup, add)
    assert refused.value.reason == "source changed while copying: ['recall.duckdb.wal']"
    assert list(lane.iterdir()) == []


def test_a_same_size_rewrite_with_restored_mtime_fails_prepare(lane: Path, backup: Path) -> None:
    copied = backup / "recall.duckdb"
    status = copied.stat()

    def rewrite(event: dict[str, object]) -> None:
        if event == {"event": "clone", "file": "payload/recall.fts.sqlite"}:
            copied.write_bytes(b"dusk")
            os.utime(copied, ns=(status.st_atime_ns, status.st_mtime_ns))

    with pytest.raises(Refused) as refused:
        _prepare_in_process(lane, backup, rewrite)
    assert refused.value.reason == "source changed while copying: ['recall.duckdb']"
    assert list(lane.iterdir()) == []


@pytest.mark.parametrize(
    ("event", "signals", "status"),
    [
        ("hash", (signal.SIGTERM,), 143),
        ("hash", (signal.SIGINT,), 130),
        ("hash", (signal.SIGTERM, signal.SIGINT), 143),
        ("hash", (signal.SIGINT, signal.SIGTERM), 130),
        ("clone", (signal.SIGTERM,), 143),
        ("clone", (signal.SIGINT,), 130),
    ],
)
def test_a_signal_during_prepare_removes_the_template(
    lane: Path,
    backup: Path,
    canaries: Path,
    event: str,
    signals: tuple[int, ...],
    status: int,
) -> None:
    # Large enough that hashing outlasts signal delivery; sparse, so it costs no disk.
    with (backup / "recall.duckdb").open("r+b") as handle:
        handle.truncate(512 * 1024 * 1024)
    process = subprocess.Popen(
        [sys.executable, str(BENCH), "prepare", "--backup", backup, "--template", lane / "t"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=bench_environment(canaries, lane),
    )
    try:
        assert process.stderr is not None
        first, *later = signals
        for line in process.stderr:
            if json.loads(line) == {"event": event, "file": "payload/recall.duckdb"}:
                process.send_signal(first)
            if json.loads(line) == {"event": "cancelled", "signal": first}:
                # Cleanup is starting; these must change neither it nor the status.
                for signum in later:
                    process.send_signal(signum)
                break
        stdout, _ = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    result = json.loads(stdout.strip().splitlines()[-1])
    assert (process.returncode, result["outcome"]) == (status, "incomplete")
    assert result["error"] == {"kind": "cancelled", "signal": signals[0]}
    assert result["leftovers"] == []
    assert list(lane.iterdir()) == []


def make_venv(checkout: Path, python: Path, shebang: Path | None = None) -> None:
    """The layout uv writes: a real environment whose entry point names its own python."""
    (checkout / ".venv/bin").mkdir(parents=True)
    (checkout / ".venv/lib/python3.12/site-packages").mkdir(parents=True)
    (checkout / ".venv/pyvenv.cfg").write_text(f"home = {Path(sys.executable).parent}\n")
    (checkout / ".venv/bin/python").symlink_to(python)
    (checkout / ".venv/bin/python3").symlink_to("python")
    recall = checkout / ".venv/bin/recall"
    recall.write_text(f"#!{shebang or checkout / '.venv/bin/python'}\n")
    recall.chmod(0o755)


@pytest.fixture
def build(tmp_path: Path) -> tuple[Build, Path]:
    """A build whose recall records each invocation and its environment."""
    checkout = tmp_path / "build"
    make_venv(checkout, Path(sys.executable))
    marker = tmp_path / "recall-invocations"
    recall = checkout / ".venv/bin/recall"
    recall.write_text(
        f"#!{checkout}/.venv/bin/python\nimport json, os, sys\n"
        f"open({str(marker)!r}, 'a').write("
        "json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'env': dict(os.environ)})"
        " + '\\n')\n"
    )
    recall.chmod(0o755)
    return Build.resolve(Policy.for_host(), "--build", checkout), marker


@pytest.fixture
def run_dir(template: Path, lane: Path) -> RunDir:
    with write_roots(lane) as roots, opened(template) as held:
        return RunDir.create(
            roots, "--run-dir", lane / "r", held, own_build(), UsageLog.EMPTY, NEVER
        )


def _subcommands(command: object) -> dict[str, Any]:
    commands = getattr(command, "commands", {})
    assert isinstance(commands, dict)
    return commands


def _parsed(argv: tuple[str, ...]) -> tuple[str, dict[str, object]]:
    """What the product's own parser makes of an argument vector, without invoking it."""
    command: Any = typer.main.get_command(app)
    path: list[str] = []
    remaining = list(argv)
    while remaining and remaining[0] in _subcommands(command):
        path.append(remaining[0])
        command = _subcommands(command)[remaining.pop(0)]
    context = command.make_context(" ".join(path), remaining)
    return " ".join(path), dict(context.params)


def _defaults(path: str) -> dict[str, object]:
    command: Any = typer.main.get_command(app)
    for part in path.split():
        command = _subcommands(command)[part]
    return {parameter.name: parameter.default for parameter in command.params if parameter.name}


ALLOWED = {
    "stats": (cli.Command.stats(), "stats", {}),
    "daemon status": (cli.Command.daemon_status(), "daemon status", {}),
    "live": (cli.Command.live(5), "live", {"limit": 5}),
    "show": (cli.Command.show("abc123"), "show", {"session_id": "abc123"}),
    "search": (cli.Command.search("needle hay", 3), "search", {"query": "needle hay", "limit": 3}),
}


@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_every_allowed_command_means_what_it_says_to_the_product_parser(name: str) -> None:
    command, path, expected = ALLOWED[name]
    parsed_path, params = _parsed(command.argv)
    defaults = _defaults(path)
    assert parsed_path == path
    assert params == {**defaults, **expected, "json_output": True}


def test_option_shaped_values_cannot_reach_the_product_parser() -> None:
    # Round 1 accepted this vector; the product reads --soft as the --fields value.
    _path, params = _parsed(("daemon", "stop", "--fields", "--soft", "--json"))
    assert (params["soft"], params["fields"]) == (False, "--soft")
    for make in (
        lambda: cli.Command.show("--soft"),
        lambda: cli.Command.search("-x", 1),
        lambda: cli.Command.search("needle", 0),
        lambda: cli.Command.show(""),
    ):
        with pytest.raises(Refused):
            make()


@contextmanager
def supervised(lane: Path, run: RunDir, build: Build) -> Iterator[Supervisor]:
    """A job's supervisor over `run` and `build`, writing its evidence in the lane."""
    with (
        write_roots(lane) as roots,
        roots.new_directory("--output", lane / "out") as target,
        target.create() as tree,
    ):
        tree.commit(NEVER)
        with Supervisor(tree, run, build, NEVER, lambda _event: None) as supervisor:
            yield supervisor


def call(supervisor: Supervisor, build: Build, run: RunDir, command: cli.Command) -> Completed:
    return cli.run_cli(supervisor, build, run, command, timeout_seconds=60, deadline=Deadline(120))


OBSERVE = "import json, os; print(json.dumps({'cwd': os.getcwd(), 'env': dict(os.environ)}))"


def test_a_recall_child_sees_only_the_run_dir_and_its_own_tools_first(
    run_dir: RunDir, lane: Path
) -> None:
    run_root = lane / "r"
    build = own_build()
    with supervised(lane, run_dir, build) as supervisor:
        completed = supervisor.run(
            "observe",
            [str(build.python), "-I", "-c", OBSERVE],
            timeout_seconds=60,
            deadline=Deadline(120),
        )
    assert completed.status == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed["cwd"] == os.path.realpath(run_root)
    env = observed["env"]
    assert env == {**child_environment(build, run_dir), **_platform_additions(env)}
    assert {name: value for name, value in env.items() if name.startswith("RECALL_")} == {
        "RECALL_DATA_DIR": str(run_root),
        "RECALL_CONFIG_PATH": f"{run_root}/config.toml",
        "RECALL_DB_PATH": f"{run_root}/recall.duckdb",
        "RECALL_LOCK_PATH": f"{run_root}/recall.lock",
        "RECALL_FLEET_PATH": f"{run_root}/fleet.toml",
    }
    assert env["PATH"] == f"{run_root}/bin:{run_root}/canary:{ROOT}/.venv/bin:/usr/bin:/bin"
    assert env["TMPDIR"] == f"{run_root}/tmp"
    assert env["HF_HOME"] == f"{run_root}/hf"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "XDG_RUNTIME_DIR" not in env
    assert "DBUS_SESSION_BUS_ADDRESS" not in env


def _platform_additions(env: dict[str, str]) -> dict[str, str]:
    # macOS adds its text-encoding variable to every process it starts.
    return {name: value for name, value in env.items() if name == "__CF_USER_TEXT_ENCODING"}


def test_every_service_manager_found_by_name_is_the_run_s_recording_canary(
    run_dir: RunDir, lane: Path
) -> None:
    env = child_environment(own_build(), run_dir)
    for name in ("launchctl", "systemctl", "crontab"):
        completed = subprocess.run(
            [name, "print", "gui/1"], env=env, capture_output=True, timeout=10, check=False
        )
        assert completed.returncode == 97
    assert (lane / "r/bench-canary.log").read_text().splitlines() == [
        f"{lane}/r/canary/launchctl print gui/1",
        f"{lane}/r/canary/systemctl print gui/1",
        f"{lane}/r/canary/crontab print gui/1",
    ]


def _canary_is_replaced(run: Path) -> str:
    canary = run / "canary/launchctl"
    canary.unlink()
    canary.write_text('#!/bin/sh\nexec /bin/launchctl "$@"\n')
    return "changed since the run was created"


def test_a_replaced_entry_point_is_not_executed(
    run_dir: RunDir, lane: Path, build: tuple[Build, Path]
) -> None:
    executable, _marker = build
    replacement = executable.recall.with_name("recall.new")
    replacement.write_bytes(executable.recall.read_bytes())
    replacement.chmod(0o755)
    replacement.rename(executable.recall)
    with supervised(lane, run_dir, executable) as supervisor, pytest.raises(Refused) as refused:
        call(supervisor, executable, run_dir, cli.Command.stats())
    assert refused.value.reason == "the recall entry point changed since resolution"
    assert supervisor.children == []


@pytest.fixture
def stable_install() -> Path:
    """A stand-in for the stable uv tool environment under the fake HOME."""
    stable = Path.home() / ".local/share/uv/tools/recall"
    (stable / "bin").mkdir(parents=True)
    (stable / "lib/python3.12/site-packages").mkdir(parents=True)
    (stable / "bin/python").symlink_to(sys.executable)
    (stable / "bin/recall").write_text(f"#!{stable}/bin/python\n")
    (stable / "bin/recall").chmod(0o755)
    return stable


def _venv_is_the_stable_install(checkout: Path, stable: Path) -> str:
    (checkout / ".venv").symlink_to(stable)
    return f"symlinked component {checkout / '.venv'}"


def _entry_point_is_the_stable_binary(checkout: Path, stable: Path) -> str:
    make_venv(checkout, Path(sys.executable))
    (checkout / ".venv/bin/recall").unlink()
    (checkout / ".venv/bin/recall").symlink_to(stable / "bin/recall")
    return f"symlinked component {checkout / '.venv/bin/recall'}"


def _entry_point_runs_another_interpreter(checkout: Path, stable: Path) -> str:
    make_venv(checkout, Path(sys.executable), shebang=stable / "bin/python")
    return f"{checkout / '.venv/bin/recall'} does not run {checkout / '.venv/bin/python'}"


def _interpreter_links_into_the_stable_install(checkout: Path, stable: Path) -> str:
    make_venv(checkout, stable / "bin/python")
    return f"overlaps refused location {stable}"


def _version_alias_is_the_stable_python(checkout: Path, stable: Path) -> str:
    make_venv(checkout, Path(sys.executable))
    (checkout / ".venv/bin/python3.12").symlink_to(stable / "bin/python")
    return "must be absent or a link to python"


def _lib64_is_the_stable_install(checkout: Path, stable: Path) -> str:
    make_venv(checkout, Path(sys.executable))
    (checkout / ".venv/lib64").symlink_to(stable / "lib")
    return "must be absent or a link to lib"


def _site_packages_is_the_stable_install(checkout: Path, stable: Path) -> str:
    make_venv(checkout, Path(sys.executable))
    site = checkout / ".venv/lib/python3.12/site-packages"
    site.rmdir()
    site.symlink_to(stable / "lib/python3.12/site-packages")
    return f"symlinked component {site}"


def _environment_home_is_the_stable_install(checkout: Path, stable: Path) -> str:
    make_venv(checkout, Path(sys.executable))
    (checkout / ".venv/pyvenv.cfg").write_text(f"home = {stable / 'bin'}\n")
    return f"overlaps refused location {stable}"


@pytest.mark.parametrize(
    "make_build",
    [
        _venv_is_the_stable_install,
        _entry_point_is_the_stable_binary,
        _entry_point_runs_another_interpreter,
        _interpreter_links_into_the_stable_install,
        _site_packages_is_the_stable_install,
        _environment_home_is_the_stable_install,
        _version_alias_is_the_stable_python,
        _lib64_is_the_stable_install,
    ],
)
def test_a_build_that_reaches_the_stable_install_is_refused(
    tmp_path: Path, stable_install: Path, make_build: Callable[[Path, Path], str]
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    reason = make_build(checkout, stable_install)
    with pytest.raises(Refused) as refused:
        Build.resolve(Policy.for_host(), "--build", checkout)
    assert refused.value.reason == reason


def test_a_build_inside_the_stable_install_is_refused(stable_install: Path) -> None:
    with pytest.raises(Refused) as refused:
        Build.resolve(Policy.for_host(), "--build", stable_install)
    assert refused.value.reason == f"overlaps refused location {stable_install}"


def test_a_checkout_without_a_virtualenv_is_not_a_build(tmp_path: Path) -> None:
    with pytest.raises(Refused) as refused:
        Build.resolve(Policy.for_host(), "--build", tmp_path)
    assert refused.value.reason == "does not exist"


def test_leftovers_appear_in_the_outcome_line(lane: Path, backup: Path, canaries: Path) -> None:
    with (backup / "recall.duckdb").open("r+b") as handle:
        handle.truncate(512 * 1024 * 1024)
    process = subprocess.Popen(
        [sys.executable, str(BENCH), "prepare", "--backup", backup, "--template", lane / "t"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=bench_environment(canaries, lane),
    )
    try:
        assert process.stderr is not None
        for line in process.stderr:
            if json.loads(line) == {"event": "hash", "file": "payload/recall.duckdb"}:
                # Someone else's file inside the template the job is building.
                (lane / "t/payload/theirs").write_bytes(b"theirs")
                break
        stdout, _ = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    result = json.loads(stdout.strip().splitlines()[-1])
    assert (process.returncode, result["outcome"]) == (1, "failed")
    assert result["error"]["reason"] == (
        "manifest mismatch: missing [], unexpected ['payload/theirs']"
    )
    not_empty = os.strerror(errno.ENOTEMPTY)
    assert result["leftovers"] == [
        {"path": str(lane / "t/payload"), "reason": not_empty},
        {"path": str(lane / "t"), "reason": not_empty},
    ]
    assert snapshot(lane) == {"t": None, "t/payload": None, "t/payload/theirs": b"theirs"}


@pytest.fixture
def cancellation() -> Iterator[Cancellation]:
    """A cancellation recording real TERM and INT, as `bench.py` installs it."""
    previous = {signum: signal.getsignal(signum) for signum in Cancellation.SIGNALS}
    token = Cancellation()
    token.install()
    try:
        yield token
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _signal_after_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliver TERM once a tree has been verified, after its last per-chunk check."""
    verify = isolation.verify_tree

    def verified_then_signalled(*args: Any) -> None:
        verify(*args)
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(isolation, "verify_tree", verified_then_signalled)


def test_a_signal_after_the_last_hash_check_removes_the_template(
    lane: Path, backup: Path, cancellation: Cancellation, monkeypatch: pytest.MonkeyPatch
) -> None:
    _signal_after_verification(monkeypatch)
    with pytest.raises(Cancelled) as cancelled:
        _prepare_in_process(lane, backup, lambda _event: None, cancellation)
    assert cancelled.value.signum == signal.SIGTERM
    assert list(lane.iterdir()) == []


def test_a_signal_after_the_last_hash_check_removes_the_run_dir(
    template: Path, lane: Path, cancellation: Cancellation, monkeypatch: pytest.MonkeyPatch
) -> None:
    with write_roots(lane) as roots, opened(template) as held:
        _signal_after_verification(monkeypatch)
        with pytest.raises(Cancelled):
            RunDir.create(
                roots, "--run-dir", lane / "r", held, own_build(), UsageLog.EMPTY, cancellation
            )
        assert roots.leftovers == []
    assert sorted(path.name for path in lane.iterdir()) == ["template"]


def test_a_signal_held_pending_at_commit_removes_the_tree(
    lane: Path, cancellation: Cancellation
) -> None:
    with write_roots(lane) as roots:
        with (
            roots.new_directory("--output", lane / "out") as target,
            target.create() as tree,
            pytest.raises(Cancelled),
            Cancellation.deferred(),
        ):
            # TERM arrives while signals are held, so no handler has recorded it yet.
            signal.raise_signal(signal.SIGTERM)
            tree.commit(cancellation)
        assert roots.leftovers == []
    assert cancellation.signum == signal.SIGTERM
    assert list(lane.iterdir()) == []


# Runs bench.py with TERM sent from inside the job: at the end of the template's last
# digest read during final verification, or just after the template is committed.
SIGNALLING_BENCH = """
import os, runpy, signal, sys
when, target, bench, *arguments = sys.argv[1:]
sys.path.insert(0, os.path.dirname(bench))
from recall_bench import fs
pread, ends = os.pread, []

def pread_then_signal(descriptor, size, offset):
    data = pread(descriptor, size, offset)
    if not data and os.path.exists(target):
        if os.path.samestat(os.fstat(descriptor), os.stat(target)):
            ends.append(offset)
            # The first end is the hash right after copying; the second is verification's.
            if len(ends) == 2:
                os.kill(os.getpid(), signal.SIGTERM)
    return data

commit = fs.NewTree.commit

def commit_then_signal(tree, cancellation):
    commit(tree, cancellation)
    os.kill(os.getpid(), signal.SIGTERM)

if when == "last-digest":
    os.pread = pread_then_signal
else:
    fs.NewTree.commit = commit_then_signal
sys.argv = [bench, *arguments]
runpy.run_path(bench, run_name="__main__")
"""


@pytest.mark.parametrize(
    ("when", "status", "outcome", "kept"),
    [("last-digest", 143, "incomplete", False), ("after-commit", 0, "passed", True)],
)
def test_a_signal_at_the_end_of_prepare_is_decided_by_the_commit(
    lane: Path,
    backup: Path,
    canaries: Path,
    when: str,
    status: int,
    outcome: str,
    kept: bool,
) -> None:
    template = lane / "t"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            SIGNALLING_BENCH,
            when,
            str(template / "payload/recall.fts.sqlite"),
            str(BENCH),
            "prepare",
            "--backup",
            str(backup),
            "--template",
            str(template),
        ],
        capture_output=True,
        text=True,
        env=bench_environment(canaries, lane),
        timeout=60,
        check=False,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert (completed.returncode, result["outcome"]) == (status, outcome), completed.stderr
    assert result["leftovers"] == []
    if kept:
        with opened(template) as held:
            assert held.path == template
    else:
        assert result["error"] == {"kind": "cancelled", "signal": signal.SIGTERM}
        assert list(lane.iterdir()) == []


def test_a_signal_after_a_run_dir_is_committed_keeps_it_and_stops_the_job(
    template: Path, lane: Path, cancellation: Cancellation, monkeypatch: pytest.MonkeyPatch
) -> None:
    check_held = Cancellation.check_held

    def check_then_signal(token: Cancellation) -> None:
        check_held(token)
        # Held until the commit's mask is restored, then recorded.
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(Cancellation, "check_held", check_then_signal)
    with write_roots(lane) as roots, opened(template) as held:
        run = RunDir.create(
            roots, "--run-dir", lane / "r", held, own_build(), UsageLog.EMPTY, cancellation
        )
    # The run directory stays; the job's next step observes the signal and stops.
    with pytest.raises(Cancelled):
        cancellation.check()
    with run.held() as root:
        assert root.creation == run.identity


def test_a_signal_during_a_chunked_copy_stops_it_and_removes_the_template(
    lane: Path, backup: Path, cancellation: Cancellation, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (backup / "recall.duckdb").open("r+b") as handle:
        handle.truncate(4 * fs.CHUNK_BYTES)
    # The fallback a volume without clones or reflinks takes.
    monkeypatch.setattr(fs, "_clone_darwin", lambda *_args: None)
    monkeypatch.setattr(fs, "FICLONE", 0)
    copy_chunk = fs._copy_chunk
    offsets: list[int] = []

    def copy_then_signal(source: int, destination: int, offset: int) -> int:
        offsets.append(offset)
        copied = copy_chunk(source, destination, offset)
        signal.raise_signal(signal.SIGTERM)
        return copied

    monkeypatch.setattr(fs, "_copy_chunk", copy_then_signal)
    with pytest.raises(Cancelled):
        _prepare_in_process(lane, backup, lambda _event: None, cancellation)
    assert offsets == [0]
    assert list(lane.iterdir()) == []


@pytest.mark.parametrize(
    "argv",
    [
        ("daemon", "stop", "--json"),
        ("daemon", "stop", "--soft", "--json"),
        ("daemon", "stop", "--fields", "--soft", "--json"),
        ("daemon", "stop", "--soft", "--json", "--fields", "x"),
        ("daemon", "restart", "--json"),
        ("daemon", "start"),
        ("stats",),
        ("stats", "--json", "--"),
        ("live", "--json", "--limit", "05"),
        ("live", "--json", "--limit", "-1"),
        ("live", "--json", "--limit", "10001"),
        ("show", "--json", "--soft"),
        ("search", "--json", "--limit", "3", "needle", "--fields"),
        ["stats", "--json"],
    ],
)
def test_a_command_outside_the_grammar_cannot_be_constructed(
    argv: Any, run_dir: RunDir, lane: Path, build: tuple[Build, Path]
) -> None:
    executable, _marker = build
    with supervised(lane, run_dir, executable) as supervisor, pytest.raises(Refused):
        call(supervisor, executable, run_dir, cli.Command(argv))
    assert supervisor.children == []


def _production_recall() -> Path:
    production = Path.home() / ".local/share/recall"
    production.mkdir(parents=True, exist_ok=True)
    return production


def _run_dir_is_a_link_to_production(run: Path) -> str:
    run.rename(run.with_name("r-moved"))
    run.symlink_to(_production_recall())
    return f"symlinked component {run}"


def _run_dir_is_another_directory(run: Path) -> str:
    run.rename(run.with_name("r-moved"))
    run.mkdir()
    return "is no longer the directory created for this run"


def _source_root_is_replaced(run: Path) -> str:
    (run / "pool/claude_code").rename(run / "claude_code-moved")
    (run / "pool/claude_code").mkdir()
    return "was replaced since the run was created"


def _config_is_edited_in_place(run: Path) -> str:
    with (run / "config.toml").open("a") as config:
        config.write('[sources.codex]\nroots = ["/"]\n')
    return "changed since the run was created"


def _database_is_a_link_to_production(run: Path) -> str:
    (run / "recall.duckdb").unlink()
    (run / "recall.duckdb").symlink_to(_production_recall() / "recall.duckdb")
    return "symlinked entry in the run directory"


def _database_has_another_name(run: Path) -> str:
    os.link(run / "recall.duckdb", _production_recall() / "recall.duckdb")
    return ALIAS_REASON


@pytest.mark.parametrize(
    "replace",
    [
        _run_dir_is_a_link_to_production,
        _run_dir_is_another_directory,
        _source_root_is_replaced,
        _config_is_edited_in_place,
        _database_is_a_link_to_production,
        _database_has_another_name,
        _canary_is_replaced,
    ],
)
def test_a_run_dir_changed_after_creation_starts_no_child(
    run_dir: RunDir, lane: Path, build: tuple[Build, Path], replace: Callable[[Path], str]
) -> None:
    executable, _marker = build
    reason = replace(run_dir.path)
    with supervised(lane, run_dir, executable) as supervisor, pytest.raises(Refused) as refused:
        call(supervisor, executable, run_dir, cli.Command.stats())
    assert refused.value.reason == reason
    assert supervisor.children == []


def test_a_database_the_product_replaced_by_rename_is_still_the_run(
    run_dir: RunDir, lane: Path
) -> None:
    build = own_build()
    # Compaction writes a new database beside the old one and renames it into place.
    duckdb.connect(str(run_dir.path / "recall.duckdb.compacted")).close()
    (run_dir.path / "recall.duckdb.compacted").rename(run_dir.path / "recall.duckdb")
    with supervised(lane, run_dir, build) as supervisor:
        completed = call(supervisor, build, run_dir, cli.Command.daemon_status())
    assert completed.status == 0, completed.stderr
    status = json.loads(completed.stdout)
    # The build read the run's config and opened the renamed database.
    assert status["config_path"] == str(run_dir.config_path)
    assert status["watched_dirs"] == [str(run_dir.path / "pool" / source) for source in SOURCES]
    assert (status["installed"], status["daemon_pid"]) == (False, None)


SENTINEL = """\
import signal, sys, time
log = open(sys.argv[1], "a", buffering=1)
for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
    signal.signal(signum, lambda received, _frame: log.write(f"{received}\\n"))
print("ready", flush=True)
time.sleep(600)
"""


@contextmanager
def sentinel(tmp_path: Path) -> Iterator[tuple[subprocess.Popen[str], Path]]:
    """An unrelated process that records every signal it receives; this test's own child."""
    log = tmp_path / "sentinel-signals"
    process = subprocess.Popen(
        [sys.executable, "-c", SENTINEL, str(log)],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert process.stdout is not None and process.stdout.readline() == "ready\n"
        yield process, log
    finally:
        process.kill()
        process.wait()
        assert process.stdout is not None
        process.stdout.close()


def test_a_stale_pid_file_cannot_make_the_run_s_recall_signal_another_process(
    run_dir: RunDir, tmp_path: Path
) -> None:
    with sentinel(tmp_path) as (process, log):
        # A run that ended early left its PID file, and the PID now names another process.
        (run_dir.path / "recall.pid").write_text(f"{process.pid}\n")
        completed = subprocess.run(
            [str(run_dir.path / "bin/recall"), "daemon", "stop", "--soft", "--json"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]},
            timeout=60,
            check=False,
        )
        alive = process.poll() is None
        received = log.read_text()
    assert completed.returncode == 5, completed.stderr
    assert "`recall daemon stop` is not an allowed bench call" in completed.stderr
    assert (alive, received) == (True, "")


def test_a_run_directory_made_again_on_its_inode_number_starts_no_child(
    run_dir: RunDir, lane: Path
) -> None:
    build = own_build()
    _recreate_on_its_inode_number(run_dir.path / "spill")
    with supervised(lane, run_dir, build) as supervisor, pytest.raises(Refused) as refused:
        call(supervisor, build, run_dir, cli.Command.stats())
    assert refused.value.reason == "was replaced since the run was created"
    assert supervisor.children == []


STALL_AFTER_REPLACING_THE_RUN = """\
import os, sys, time
run = sys.argv[1]
os.rename(run, run + "-moved")
os.mkdir(run)
time.sleep(600)
"""


def test_a_run_replaced_under_a_stalled_child_is_refused_promptly_and_its_group_stopped(
    run_dir: RunDir, lane: Path
) -> None:
    build = own_build()
    argv = [str(build.python), "-I", "-c", STALL_AFTER_REPLACING_THE_RUN, str(run_dir.path)]
    with supervised(lane, run_dir, build) as supervisor:
        started = time.monotonic()
        with pytest.raises(Refused) as refused:
            supervisor.run("stall", argv, timeout_seconds=60, deadline=Deadline(120))
        seconds = time.monotonic() - started
        [child] = supervisor.children
    assert refused.value.reason == "is no longer the directory created for this run"
    # Well inside the call's own 60 s timeout.
    assert seconds < 10, seconds
    assert child.stopped is not None
    assert (child.stopped.signals[0], child.stopped.group_gone) == ("SIGTERM", True)


BIND_OVER_RUN_ENTRY = """\
import json, subprocess, sys
from pathlib import Path
from scripts.recall_bench import cli
from scripts.recall_bench.fs import Cancellation, Refused
from scripts.recall_bench.isolation import Build, Policy, RunDir, Template, UsageLog
from scripts.recall_bench.supervisor import Deadline, Supervisor
production, template, lane, checkout, entry = map(Path, sys.argv[1:6])
unlink_source = sys.argv[6:] == ["unlink"]
policy = Policy.for_host()
never = Cancellation()
build = Build.resolve(policy, "--build", checkout)
with policy.open_write_roots({"BENCH_WRITE_ROOTS": str(lane)}) as roots:
    with Template.open(policy, "--t", template, never) as held:
        run = RunDir.create(roots, "--run-dir", lane / "r", held, build, UsageLog.EMPTY, never)
    if production.is_dir():
        (run.path / entry).mkdir()
    subprocess.run(["mount", "--bind", str(production), str(run.path / entry)], check=True)
    if unlink_source:
        # The mount keeps the file reachable in the run with a link count of 0.
        production.unlink()
    with (
        roots.new_directory("--output", lane / "out") as target,
        target.create() as tree,
    ):
        tree.commit(never)
        with Supervisor(tree, run, build, never, lambda _event: None) as supervisor:
            try:
                cli.run_cli(
                    supervisor, build, run, cli.Command.stats(),
                    timeout_seconds=10, deadline=Deadline(60),
                )
                reason = None
            except Refused as error:
                reason = error.reason
            started = len(supervisor.children)
print(json.dumps({"reason": reason, "started": started}))
"""


@pytest.mark.parametrize(("entry", "kind"), [("recall.duckdb", "file"), ("models", "directory")])
def test_a_production_entry_bind_mounted_into_a_run_dir_starts_no_child(
    template: Path, lane: Path, build: tuple[Build, Path], entry: str, kind: str
) -> None:
    executable, marker = build
    # A bind on the same filesystem keeps st_dev; only the mount id tells it apart.
    production = _production_recall() / f"bound-{kind}"
    if kind == "file":
        production.write_bytes(b"production")
    else:
        production.mkdir()
        (production / "recall.duckdb").write_bytes(b"production")
    before = production.stat(), snapshot(production.parent)
    completed = subprocess.run(
        [
            *_mount_namespace(),
            sys.executable,
            "-c",
            BIND_OVER_RUN_ENTRY,
            str(production),
            str(template),
            str(lane),
            str(executable.path),
            entry,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"reason": "crosses a mount point", "started": 0}
    assert not marker.exists()
    assert (production.stat(), snapshot(production.parent)) == before


def test_a_bind_mounted_file_whose_source_was_unlinked_starts_no_child(
    template: Path, lane: Path, build: tuple[Build, Path]
) -> None:
    executable, marker = build
    # Unlinking the source name leaves the bound file a link count of 0, the
    # same count as a SQLite journal deleted since its open.
    production = _production_recall() / "bound-file"
    production.write_bytes(b"production")
    completed = subprocess.run(
        [
            *_mount_namespace(),
            sys.executable,
            "-c",
            BIND_OVER_RUN_ENTRY,
            str(production),
            str(template),
            str(lane),
            str(executable.path),
            "recall.duckdb",
            "unlink",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"reason": "crosses a mount point", "started": 0}
    assert not marker.exists()


def _interpreter_into_the_stable_install(build: Build, stable: Path) -> str:
    python = build.path / ".venv/bin/python"
    python.unlink()
    python.symlink_to(stable / "bin/python")
    return f"overlaps refused location {stable}"


def _interpreter_to_another_python(build: Build, _stable: Path) -> str:
    other = build.path.parent / "other-python"
    other.write_bytes(Path(sys.executable).read_bytes())
    other.chmod(0o755)
    python = build.path / ".venv/bin/python"
    python.unlink()
    python.symlink_to(other)
    return "the interpreter changed since resolution"


@pytest.mark.parametrize(
    "replace", [_interpreter_into_the_stable_install, _interpreter_to_another_python]
)
def test_an_interpreter_replaced_after_resolution_is_not_executed(
    stable_install: Path,
    run_dir: RunDir,
    lane: Path,
    build: tuple[Build, Path],
    replace: Callable[[Build, Path], str],
) -> None:
    executable, _marker = build
    entry_point = identity(executable.recall.stat(follow_symlinks=False))
    reason = replace(executable, stable_install)
    assert identity(executable.recall.stat(follow_symlinks=False)) == entry_point
    with supervised(lane, run_dir, executable) as supervisor, pytest.raises(Refused) as refused:
        call(supervisor, executable, run_dir, cli.Command.stats())
    assert refused.value.reason == reason
    assert supervisor.children == []


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> set[Identity]:
    """The identity of every file whose bytes the tool reads or clones."""
    seen: set[Identity] = set()
    pread, clone = os.pread, fs._clone_darwin

    def recording_pread(descriptor: int, size: int, offset: int) -> bytes:
        seen.add(identity(os.fstat(descriptor)))
        return pread(descriptor, size, offset)

    def recording_clone(source: fs.File, *args: Any) -> fs.Creation | None:
        # Every copy starts here, on Linux too, before any byte moves.
        seen.add(identity(source.status))
        return clone(source, *args)

    monkeypatch.setattr(os, "pread", recording_pread)
    monkeypatch.setattr(fs, "_clone_darwin", recording_clone)
    return seen


def test_a_hardlinked_manifest_is_refused_before_it_is_read(
    template: Path, reads: set[Identity]
) -> None:
    manifest = template / "manifest.json"
    production = _production_recall() / "manifest.json"
    production.write_bytes(manifest.read_bytes())
    manifest.unlink()
    os.link(production, manifest)
    with pytest.raises(Refused) as refused, opened(template):
        pass
    assert refused.value.reason == ALIAS_REASON
    assert identity(production.stat()) not in reads


def test_a_file_unlinked_after_its_open_is_absent_not_an_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "recall.fts.sqlite-journal").write_bytes(b"journal")

    class UnlinkedOnOpen(fs.File):
        # The product deletes its journal between this open and the link check.
        def __init__(self, path: Path, descriptor: int) -> None:
            path.unlink()
            super().__init__(path, descriptor)

    monkeypatch.setattr(fs, "File", UnlinkedOnOpen)
    with Policy.for_host().open_tree("--run-dir", tmp_path) as directory:
        assert directory.open_present_file("--run-dir", "recall.fts.sqlite-journal") is None
        (tmp_path / "recall.fts.sqlite-journal").write_bytes(b"journal")
        with pytest.raises(Refused) as refused:
            directory.open_file("--run-dir", "recall.fts.sqlite-journal")
    assert refused.value.reason == "does not exist"


def test_a_payload_hardlinked_after_verification_is_not_copied(
    template: Path, lane: Path, reads: set[Identity]
) -> None:
    production = _production_recall() / "recall.duckdb"
    production.write_bytes(b"duck")
    with write_roots(lane) as roots, opened(template) as held:
        payload = template / "payload/recall.duckdb"
        payload.unlink()
        os.link(production, payload)
        with pytest.raises(Refused) as refused:
            RunDir.create(roots, "--run-dir", lane / "r", held, own_build(), UsageLog.EMPTY, NEVER)
    assert refused.value.reason == ALIAS_REASON
    assert identity(production.stat()) not in reads
    assert not (lane / "r").exists()


def test_a_bind_mounted_manifest_is_refused_before_it_is_read(template: Path) -> None:
    production = _production_recall() / "manifest.json"
    production.write_bytes((template / "manifest.json").read_bytes())
    child = (
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "from scripts.recall_bench.fs import Cancellation, Refused, identity\n"
        "from scripts.recall_bench.isolation import Policy, Template\n"
        "production, template = map(Path, sys.argv[1:3])\n"
        "seen, pread = set(), os.pread\n"
        "def recording(fd, size, offset):\n"
        "    seen.add(identity(os.fstat(fd)))\n"
        "    return pread(fd, size, offset)\n"
        "os.pread = recording\n"
        "try:\n"
        "    Template.open(Policy.for_host(), '--t', template, Cancellation()).close()\n"
        "    reason = None\n"
        "except Refused as error:\n"
        "    reason = error.reason\n"
        "print(json.dumps({'reason': reason, 'read': identity(production.stat()) in seen}))\n"
    )
    completed = subprocess.run(
        [
            *_unshare(),
            str(production),
            str(template / "manifest.json"),
            sys.executable,
            "-c",
            child,
            str(production),
            str(template),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"reason": "crosses a mount point", "read": False}


def test_a_zero_status_child_cannot_hide_a_stop_denial(
    run_dir: RunDir,
    lane: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import errno

    from scripts.recall_bench.supervisor import ChildStopFailed

    build = own_build()
    send = os.killpg

    def denied_term(pgid: int, signum: int) -> None:
        if signum == signal.SIGTERM:
            raise PermissionError(errno.EACCES, "short child cleanup denied")
        send(pgid, signum)

    monkeypatch.setattr(os, "killpg", denied_term)
    with supervised(lane, run_dir, build) as supervisor, pytest.raises(ChildStopFailed) as caught:
        supervisor.run(
            "zero",
            [str(build.python), "-I", "-c", "raise SystemExit(0)"],
            timeout_seconds=60,
            deadline=Deadline(120),
        )
    assert caught.value.stopped.status == 0
    assert not caught.value.stopped.clean
    assert caught.value.stopped.group_gone
    assert caught.value.stopped.errors[0].errno == errno.EACCES
