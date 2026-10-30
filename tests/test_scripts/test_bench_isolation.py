"""Drive bench isolation with real processes and files: refusals, `prepare`, run dirs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import typer.main
from recall.cli.app import app

from scripts.recall_bench import cli
from scripts.recall_bench.fs import Cancellation, Refused
from scripts.recall_bench.isolation import (
    Build,
    Mounts,
    Policy,
    RunDir,
    Template,
    WriteRoots,
)
from scripts.recall_bench.prepare import build_template

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
        run = RunDir.create(roots, "--run-dir", lane / "r", held, NEVER)

    assert (run.path / "recall.duckdb").read_bytes() == b"duck"
    assert (run.path / "pool/claude_code/project/s.jsonl").read_bytes() == b"session\n"
    assert (run.path / "pool/codex/empty").is_dir()
    pool = lane / "r/pool"
    assert run.config_path.read_text() == (
        '[daemon]\nscheduler = "pid"\n\n'
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
        RunDir.create(roots, "--run-dir", lane / "r", held, NEVER)
        with pytest.raises(Refused) as refused:
            RunDir.create(roots, "--run-dir", lane / "r", held, NEVER)
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
        assert RunDir.create(roots, "--run-dir", fits, held, NEVER).socket_path.exists() is False
        with pytest.raises(Refused) as refused:
            RunDir.create(roots, "--run-dir", too_long, held, NEVER)
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
            tree.keep()
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
        tree.keep()
    assert (lane / "out/result.json").read_bytes() == b"first"
    assert not (lane / "elsewhere").exists()
    assert sorted(path.name for path in (lane / "out").iterdir()) == ["planted.json", "result.json"]


def test_an_unkept_tree_removes_only_what_it_created(lane: Path) -> None:
    with (
        write_roots(lane) as roots,
        roots.new_directory("--output", lane / "out") as target,
        target.create() as tree,
    ):
        directory = tree.make_directory(tree.root, PurePosixPath("logs"))
        tree.write_new(directory, PurePosixPath("logs/a"), b"a")
        directory.close()
        (lane / "out/logs/theirs").write_bytes(b"theirs")
    assert snapshot(lane) == {"out": None, "out/logs": None, "out/logs/theirs": b"theirs"}
    assert tree.leftovers == ["logs", "."]


def test_a_bind_mount_is_refused_by_its_mount_root(tmp_path: Path) -> None:
    mounts = Mounts({1: "/", 7: "/home/user/.local/share/recall/sub"})
    with pytest.raises(Refused) as refused:
        mounts.check("--backup", tmp_path / "bound", 7)
    assert refused.value.reason == "is inside a bind mount of /home/user/.local/share/recall/sub"
    with pytest.raises(Refused) as unknown:
        mounts.check("--backup", tmp_path / "late", 9)
    assert unknown.value.reason == "is on a mount that appeared after the job started"
    mounts.check("--backup", tmp_path, 1)


def _unshare() -> list[str]:
    if sys.platform != "linux":
        pytest.skip("bind mounts are a Linux mechanism")
    probe = subprocess.run(["unshare", "-Urm", "true"], capture_output=True, check=False)
    if probe.returncode != 0:
        pytest.skip(f"no unprivileged mount namespace: {probe.stderr!r}")
    return ["unshare", "-Urm", "sh", "-c", 'mount --bind "$1" "$2" && shift 2 && exec "$@"', "sh"]


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
    lane: Path, backup: Path, on_event: Callable[[dict[str, object]], None]
) -> None:
    policy = Policy.for_host()
    with ExitStack() as held:
        roots = held.enter_context(write_roots(lane))
        source = held.enter_context(policy.open_tree("--backup", backup))
        target = held.enter_context(roots.new_directory("--template", lane / "template"))
        build_template(target, source, None, None, NEVER, on_event, provenance={})


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
    ("signals", "status"),
    [
        ((signal.SIGTERM,), 143),
        ((signal.SIGINT,), 130),
        ((signal.SIGTERM, signal.SIGINT), 143),
        ((signal.SIGINT, signal.SIGTERM), 130),
    ],
)
def test_a_signal_during_prepare_removes_the_template(
    lane: Path, backup: Path, canaries: Path, signals: tuple[int, ...], status: int
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
            if json.loads(line) == {"event": "hash", "file": "payload/recall.duckdb"}:
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
        "json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')\n"
    )
    recall.chmod(0o755)
    return Build.resolve(Policy.for_host(), "--build", checkout), marker


@pytest.fixture
def run_dir(template: Path, lane: Path) -> RunDir:
    with write_roots(lane) as roots, opened(template) as held:
        return RunDir.create(roots, "--run-dir", lane / "r", held, NEVER)


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
    "soft stop": (cli.Command.daemon_stop_soft(), "daemon stop", {"soft": True}),
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


def test_allowed_cli_commands_reach_only_the_run_dir(
    run_dir: RunDir, lane: Path, tmp_path: Path, build: tuple[Build, Path]
) -> None:
    executable, marker = build
    for command in (cli.Command.daemon_status(), cli.Command.daemon_stop_soft()):
        assert cli.run_cli(executable, run_dir, command, timeout_seconds=10).returncode == 0

    calls = [json.loads(line) for line in marker.read_text().splitlines()]
    assert [call["argv"] for call in calls] == [
        ["daemon", "status", "--json"],
        ["daemon", "stop", "--soft", "--json"],
    ]
    env = calls[0]["env"]
    run_root = lane / "r"
    assert {name: value for name, value in env.items() if name.startswith("RECALL_")} == {
        "RECALL_DATA_DIR": str(run_root),
        "RECALL_CONFIG_PATH": f"{run_root}/config.toml",
        "RECALL_DB_PATH": f"{run_root}/recall.duckdb",
        "RECALL_LOCK_PATH": f"{run_root}/recall.lock",
        "RECALL_FLEET_PATH": f"{run_root}/fleet.toml",
        "RECALL_DUCKDB_TEMP_DIR": f"{run_root}/spill",
    }
    assert env["PATH"] == f"{tmp_path}/build/.venv/bin:/usr/bin:/bin"
    assert env["TMPDIR"] == f"{run_root}/tmp"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "XDG_RUNTIME_DIR" not in env
    assert "DBUS_SESSION_BUS_ADDRESS" not in env


def test_a_replaced_entry_point_is_not_executed(run_dir: RunDir, build: tuple[Build, Path]) -> None:
    executable, marker = build
    replacement = executable.recall.with_name("recall.new")
    replacement.write_bytes(executable.recall.read_bytes())
    replacement.chmod(0o755)
    replacement.rename(executable.recall)
    with pytest.raises(Refused) as refused:
        cli.run_cli(executable, run_dir, cli.Command.stats(), timeout_seconds=10)
    assert refused.value.reason == "the recall entry point changed since resolution"
    assert not marker.exists()


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
