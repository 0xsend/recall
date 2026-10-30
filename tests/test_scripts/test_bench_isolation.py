"""Drive bench isolation with real processes and files: refusals, `prepare`, run dirs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.recall_bench import cli
from scripts.recall_bench.isolation import Build, Policy, Refused, RunDir, Template

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "scripts/bench.py"
DUCK_SHA256 = "2d2370db2447ff8cf4f3accd68c85aa119a9c893effd200a9b69176e9fc5eb98"
FTS_SHA256 = "dd8f6640e1c144ff8cb56e681948ad4690c9333e62121f8fb8395feb04769ad0"
SESSION_SHA256 = "e9d6e33fc2fea3757c6a1e121276a412ca43028a7a1efc9b6a68b1182bd55c8e"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
POOL_REASON = "files must be under one of claude_code, codex, pi_agent, grok, kimi_code"
SOCKET_LIMIT = {"darwin": 104, "linux": 108}[sys.platform]


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


def bench(*args: str | Path, canaries: Path, env: dict[str, str] | None = None) -> tuple[int, dict]:
    environment = dict(os.environ if env is None else env)
    environment["PATH"] = f"{canaries.parent / 'canary-bin'}:{environment['PATH']}"
    completed = subprocess.run(
        [sys.executable, str(BENCH), *map(str, args)],
        capture_output=True,
        text=True,
        env=environment,
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
    return directory


@pytest.fixture
def template(tmp_path: Path, backup: Path, pool: Path, canaries: Path) -> Path:
    usage = tmp_path / "usage.jsonl"
    usage.write_bytes(b"")
    path = tmp_path / "template"
    status, line = bench(
        "prepare",
        "--backup", backup,
        "--template", path,
        "--source-pool", pool,
        "--usage-log", usage,
        canaries=canaries,
    )  # fmt: skip
    assert (status, line["outcome"]) == (0, "passed"), line
    return path


def test_prepare_writes_a_verified_manifest_and_touches_nothing_else(
    tmp_path: Path, backup: Path, pool: Path, template: Path, canaries: Path
) -> None:
    manifest = json.loads((template / "manifest.json").read_bytes())
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
    }
    assert list(Path.home().iterdir()) == []
    assert not canaries.exists()
    assert Template.open(Policy.for_host(), "--template", template).path == template


def test_prepare_refuses_an_existing_template_and_leaves_it_unchanged(
    tmp_path: Path, backup: Path, canaries: Path
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_bytes(b"keep")
    status, line = bench("prepare", "--backup", backup, "--template", existing, canaries=canaries)
    assert (status, line["outcome"]) == (1, "failed")
    assert line["error"]["reason"] == "already exists; every run uses a new directory"
    assert snapshot(existing) == {"keep": b"keep"}


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


@pytest.mark.parametrize("argument", ["--backup", "--template", "--source-pool", "--usage-log"])
@pytest.mark.parametrize("refused_tree", sorted(REFUSED_LOCATIONS), indirect=True)
def test_every_prepare_argument_refuses_every_refused_location(
    tmp_path: Path, backup: Path, canaries: Path, argument: str, refused_tree: Path
) -> None:
    arguments = {"--backup": backup, "--template": tmp_path / "template"}
    arguments[argument] = _argument_path(argument, refused_tree)
    before = snapshot(refused_tree) if refused_tree != ROOT else None

    status, line = bench(
        "prepare", *[part for pair in arguments.items() for part in pair], canaries=canaries
    )

    assert (status, line["outcome"]) == (1, "failed"), line
    assert line["error"]["argument"] == argument
    assert line["error"]["reason"] == f"overlaps refused location {refused_tree}"
    assert not (tmp_path / "template").exists()
    assert not (refused_tree / "refused-template").exists()
    if before is not None:
        assert snapshot(refused_tree) == before
    assert not canaries.exists()


@pytest.mark.parametrize("argument", ["--backup", "--template", "--source-pool", "--usage-log"])
def test_every_prepare_argument_refuses_the_account_home(
    tmp_path: Path, backup: Path, canaries: Path, argument: str
) -> None:
    arguments = {"--backup": backup, "--template": tmp_path / "template"}
    arguments[argument] = Path.home()
    status, line = bench(
        "prepare", *[part for pair in arguments.items() for part in pair], canaries=canaries
    )
    assert (status, line["outcome"]) == (1, "failed"), line
    assert line["error"]["reason"] == f"is or contains the account home {Path.home()}"
    assert list(Path.home().iterdir()) == []


def test_a_directory_containing_production_state_is_refused(tmp_path: Path, canaries: Path) -> None:
    data = Path.home() / ".local/share/recall"
    data.mkdir(parents=True)
    status, line = bench(
        "prepare",
        "--backup", Path.home() / ".local/share",
        "--template", tmp_path / "template",
        canaries=canaries,
    )  # fmt: skip
    assert (status, line["error"]["reason"]) == (1, f"overlaps refused location {data}")


def test_another_spelling_of_production_state_is_refused(tmp_path: Path, canaries: Path) -> None:
    data = Path.home() / ".local/share/recall"
    data.mkdir(parents=True)
    respelled = Path.home() / ".LOCAL/share/recall"
    if not respelled.exists():
        pytest.skip("volume is case-sensitive; there is no second spelling")
    status, line = bench(
        "prepare", "--backup", respelled, "--template", tmp_path / "t", canaries=canaries
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


def _fifo_payload(tmp_path: Path, backup: Path) -> Path:
    os.mkfifo(backup / "recall.duckdb.wal")
    return backup


def _nested_payload(tmp_path: Path, backup: Path) -> Path:
    (backup / "nested").mkdir()
    (backup / "nested/file").write_bytes(b"x")
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
        (_fifo_payload, "not a regular file"),
        (_nested_payload, "backup payload must be flat files"),
        (_reserved_payload, "name is reserved in a run directory"),
        (_empty_backup, "holds no files"),
    ],
)
def test_prepare_refuses_backups_it_cannot_copy_faithfully(
    tmp_path: Path,
    backup: Path,
    canaries: Path,
    make_backup: Callable[[Path, Path], Path],
    reason: str,
) -> None:
    source = make_backup(tmp_path, backup)
    status, line = bench(
        "prepare", "--backup", source, "--template", tmp_path / "template", canaries=canaries
    )
    assert (status, line["outcome"]) == (1, "failed"), line
    assert line["error"]["reason"] == reason.format(tmp=tmp_path)
    assert not (tmp_path / "template").exists()


@pytest.mark.parametrize(
    ("layout", "reason"),
    [
        ("symlink", "symlinked payload file"),
        ("unknown source", POOL_REASON),
        ("top-level file", POOL_REASON),
    ],
)
def test_prepare_refuses_source_pools_it_cannot_pin(
    tmp_path: Path, backup: Path, pool: Path, canaries: Path, layout: str, reason: str
) -> None:
    if layout == "symlink":
        (pool / "claude_code/project/link.jsonl").symlink_to(pool / "claude_code/project/s.jsonl")
    elif layout == "unknown source":
        (pool / "claude").mkdir()
        (pool / "claude/s.jsonl").write_bytes(b"")
    else:
        (pool / "notes.jsonl").write_bytes(b"")
    status, line = bench(
        "prepare",
        "--backup", backup,
        "--template", tmp_path / "template",
        "--source-pool", pool,
        canaries=canaries,
    )  # fmt: skip
    assert (status, line["error"]["reason"]) == (1, reason)
    assert not (tmp_path / "template").exists()


def test_prepare_refuses_a_symlinked_usage_log(
    tmp_path: Path, backup: Path, canaries: Path
) -> None:
    (tmp_path / "real.jsonl").write_bytes(b"")
    (tmp_path / "usage.jsonl").symlink_to(tmp_path / "real.jsonl")
    status, line = bench(
        "prepare",
        "--backup", backup,
        "--template", tmp_path / "template",
        "--usage-log", tmp_path / "usage.jsonl",
        canaries=canaries,
    )  # fmt: skip
    assert (status, line["error"]["reason"]) == (1, f"symlinked component {tmp_path}/usage.jsonl")


def test_an_invalid_invocation_fails_rather_than_reporting_incomplete(canaries: Path) -> None:
    status, line = bench("prepare", "--backup", canaries=canaries)
    assert (status, line["outcome"], line["error"]["kind"]) == (1, "failed", "usage")


def _flip_byte(template: Path) -> str:
    (template / "payload/recall.duckdb").write_bytes(b"dusk")
    return "manifest mismatch: sha256 differs"


def _add_file(template: Path) -> str:
    (template / "payload/recall.duckdb.wal").write_bytes(b"")
    return "manifest mismatch: missing [], unexpected ['payload/recall.duckdb.wal']"


def _remove_file(template: Path) -> str:
    (template / "usage.jsonl").unlink()
    return "manifest mismatch: missing ['usage.jsonl'], unexpected []"


def _symlink_file(template: Path) -> str:
    (template / "payload/recall.fts.sqlite").unlink()
    (template / "payload/recall.fts.sqlite").symlink_to(template / "payload/recall.duckdb")
    return "symlinked payload file"


def _edit_manifest(template: Path) -> str:
    manifest = template / "manifest.json"
    manifest.write_text(manifest.read_text().replace(DUCK_SHA256, FTS_SHA256))
    return "manifest mismatch: sha256 differs"


@pytest.mark.parametrize(
    "mutate", [_flip_byte, _add_file, _remove_file, _symlink_file, _edit_manifest]
)
def test_a_template_that_no_longer_matches_its_manifest_cannot_be_opened(
    template: Path, tmp_path: Path, mutate: Callable[[Path], str]
) -> None:
    reason = mutate(template)
    policy = Policy.for_host()
    with pytest.raises(Refused) as refused:
        RunDir.create(policy, "--run-dir", tmp_path / "r", Template.open(policy, "--t", template))
    assert refused.value.reason == reason
    assert not (tmp_path / "r").exists()


def test_a_run_dir_is_a_verified_clone_with_every_path_pinned_inside(
    template: Path, tmp_path: Path
) -> None:
    policy = Policy.for_host()
    run = RunDir.create(policy, "--run-dir", tmp_path / "r", Template.open(policy, "--t", template))

    assert (run.path / "recall.duckdb").read_bytes() == b"duck"
    assert (run.path / "pool/claude_code/project/s.jsonl").read_bytes() == b"session\n"
    pool = tmp_path / "r/pool"
    assert run.config_path.read_text() == (
        '[daemon]\nscheduler = "pid"\n\n'
        f'[sources.claude_code]\nroots = ["{pool}/claude_code"]\n\n'
        f'[sources.codex]\nroots = ["{pool}/codex"]\n\n'
        f'[sources.pi_agent]\nroots = ["{pool}/pi_agent"]\n\n'
        f'[sources.grok]\nroots = ["{pool}/grok"]\n\n'
        f'[sources.kimi_code]\nroots = ["{pool}/kimi_code"]\n'
    )
    assert all((pool / source).is_dir() for source in ("codex", "pi_agent", "grok", "kimi_code"))
    (run.path / "recall.duckdb").write_bytes(b"used")
    assert Template.open(policy, "--t", template).path == template


def test_a_run_dir_is_single_use(template: Path, tmp_path: Path) -> None:
    policy = Policy.for_host()
    opened = Template.open(policy, "--t", template)
    RunDir.create(policy, "--run-dir", tmp_path / "r", opened)
    with pytest.raises(Refused) as refused:
        RunDir.create(policy, "--run-dir", tmp_path / "r", opened)
    assert refused.value.reason == "already exists; every run uses a new directory"


def _run_dir_with_socket_bytes(parent: Path, total: int) -> Path:
    """A run dir whose socket path, with its NUL, takes exactly `total` bytes."""
    # Two-byte characters show the limit counts encoded bytes, not characters.
    fixed = len(os.fsencode(parent)) + len("/") + len("/recall.sock") + 1
    remaining = total - fixed
    assert remaining > 2, "test root is too deep for this socket check"
    return parent / ("é" * (remaining // 2) + "x" * (remaining % 2))


def test_socket_path_length_counts_encoded_bytes_and_nul(template: Path, tmp_path: Path) -> None:
    policy = Policy.for_host()
    opened = Template.open(policy, "--t", template)
    fits = _run_dir_with_socket_bytes(tmp_path, SOCKET_LIMIT)
    assert RunDir.create(policy, "--run-dir", fits, opened).socket_path.exists() is False

    too_long = _run_dir_with_socket_bytes(tmp_path / "x", SOCKET_LIMIT + 1)
    (tmp_path / "x").mkdir()
    with pytest.raises(Refused) as refused:
        RunDir.create(policy, "--run-dir", too_long, opened)
    assert refused.value.reason == (
        f"socket path {too_long}/recall.sock needs {SOCKET_LIMIT + 1} bytes with NUL; "
        f"the limit is {SOCKET_LIMIT}"
    )
    assert not too_long.exists()


@pytest.fixture
def build(tmp_path: Path) -> tuple[Build, Path]:
    """A build whose recall records each invocation and its environment."""
    checkout = tmp_path / "build"
    (checkout / ".venv/bin").mkdir(parents=True)
    (checkout / ".venv/bin/python").symlink_to(sys.executable)
    marker = tmp_path / "recall-invocations"
    recall = checkout / ".venv/bin/recall"
    recall.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"open({str(marker)!r}, 'a').write("
        "json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')\n"
    )
    recall.chmod(0o755)
    return Build.resolve(Policy.for_host(), "--build", checkout), marker


@pytest.mark.parametrize(
    "command",
    [
        ["daemon", "stop"],
        ["daemon", "stop", "--json"],
        ["daemon", "restart"],
        ["daemon", "install"],
        ["daemon", "uninstall"],
        ["daemon", "start"],
        ["daemon", "start", "--background"],
        ["daemon"],
        ["fleet", "status"],
        ["--help"],
        [],
    ],
)
def test_forbidden_cli_commands_never_start_recall(
    template: Path, tmp_path: Path, build: tuple[Build, Path], command: list[str]
) -> None:
    executable, marker = build
    policy = Policy.for_host().with_checkout(executable.path)
    run = RunDir.create(policy, "--run-dir", tmp_path / "r", Template.open(policy, "--t", template))
    with pytest.raises(Refused) as refused:
        cli.run_cli(executable, run, command, timeout_seconds=10)
    assert refused.value.reason == "command is not on the bench allow-list"
    assert not marker.exists()


def test_allowed_cli_commands_reach_only_the_run_dir(
    template: Path, tmp_path: Path, build: tuple[Build, Path]
) -> None:
    executable, marker = build
    policy = Policy.for_host().with_checkout(executable.path)
    run = RunDir.create(policy, "--run-dir", tmp_path / "r", Template.open(policy, "--t", template))
    for command in (["status", "--json"], ["daemon", "status"], ["daemon", "stop", "--soft"]):
        assert cli.run_cli(executable, run, command, timeout_seconds=10).returncode == 0

    calls = [json.loads(line) for line in marker.read_text().splitlines()]
    assert [call["argv"] for call in calls] == [
        ["status", "--json"],
        ["daemon", "status"],
        ["daemon", "stop", "--soft"],
    ]
    env = calls[0]["env"]
    run_root = tmp_path / "r"
    assert {name: value for name, value in env.items() if name.startswith("RECALL_")} == {
        "RECALL_DATA_DIR": str(run_root),
        "RECALL_CONFIG_PATH": f"{run_root}/config.toml",
        "RECALL_DB_PATH": f"{run_root}/recall.duckdb",
        "RECALL_LOCK_PATH": f"{run_root}/recall.lock",
        "RECALL_FLEET_PATH": f"{run_root}/fleet.toml",
        "RECALL_DUCKDB_TEMP_DIR": f"{run_root}/spill",
    }
    assert env["PATH"] == f"{tmp_path}/build/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    assert env["TMPDIR"] == f"{run_root}/tmp"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert "XDG_RUNTIME_DIR" not in env
    assert "DBUS_SESSION_BUS_ADDRESS" not in env


def test_a_build_inside_the_stable_install_is_refused(tmp_path: Path) -> None:
    stable = Path.home() / ".local/share/uv/tools/recall"
    (stable / ".venv/bin").mkdir(parents=True)
    with pytest.raises(Refused) as refused:
        Build.resolve(Policy.for_host(), "--build", stable)
    assert refused.value.reason == f"overlaps refused location {stable}"


def test_a_checkout_without_a_virtualenv_is_not_a_build(tmp_path: Path) -> None:
    with pytest.raises(Refused) as refused:
        Build.resolve(Policy.for_host(), "--build", tmp_path)
    assert refused.value.reason == f"{tmp_path}/.venv/bin/python is not an executable file"
