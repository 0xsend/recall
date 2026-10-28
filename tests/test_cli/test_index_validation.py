from __future__ import annotations

import json
from pathlib import Path

import pytest
from recall.cli.app import app
from typer.testing import CliRunner


@pytest.mark.parametrize("dry_run", [False, True], ids=["execute", "dry-run"])
@pytest.mark.parametrize("via_params", [False, True], ids=["flags", "params"])
@pytest.mark.parametrize("format_name", ["json", "text"])
@pytest.mark.parametrize("explicit_host", [False, True])
def test_index_rejects_unexpandable_root_before_rpc(
    unexpandable_index_root: str,
    monkeypatch: pytest.MonkeyPatch,
    dry_run: bool,
    via_params: bool,
    format_name: str,
    explicit_host: bool,
) -> None:
    def unexpected_rpc(*args, **kwargs):
        pytest.fail("invalid index root must not contact the daemon")

    monkeypatch.setattr("recall.cli.index.rpc_call_or_error", unexpected_rpc)
    options = {"root": unexpandable_index_root}
    if explicit_host:
        options["host"] = "remote-host"
    args = ["index", "--format", format_name, "--no-embed", "--no-progress"]
    if via_params:
        args.extend(["--params", json.dumps(options)])
    else:
        for name, value in options.items():
            args.extend(["--" + name, value])
    if dry_run:
        args.append("--dry-run")

    result = CliRunner().invoke(app, args)

    message = f"cannot expand root: {unexpandable_index_root}: Could not determine home directory."
    assert result.exit_code == 2, result.output
    if format_name == "json":
        assert (
            result.stdout
            == json.dumps(
                {"error": {"code": "VALIDATION", "message": message}}, separators=(",", ":")
            )
            + "\n"
        )
        assert result.stderr == ""
    else:
        assert result.stdout == ""
        assert result.stderr == f"error: {message}\n"


@pytest.mark.parametrize("dry_run", [False, True], ids=["execute", "dry-run"])
@pytest.mark.parametrize("via_params", [False, True], ids=["flags", "params"])
@pytest.mark.parametrize("format_name", ["json", "text"])
@pytest.mark.parametrize("explicit_host", [False, True])
@pytest.mark.parametrize("dangling_link", [False, True], ids=["missing", "dangling-link"])
def test_index_rejects_missing_root_before_rpc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dry_run: bool,
    via_params: bool,
    format_name: str,
    explicit_host: bool,
    dangling_link: bool,
) -> None:
    root = tmp_path / "missing-root"
    if dangling_link:
        root.symlink_to(tmp_path / "absent-target", target_is_directory=True)

    def unexpected_rpc(*args, **kwargs):
        pytest.fail("invalid index root must not contact the daemon")

    monkeypatch.setattr("recall.cli.index.rpc_call_or_error", unexpected_rpc)
    options = {"root": str(root)}
    if explicit_host:
        options["host"] = "remote-host"
    args = ["index", "--format", format_name, "--no-embed", "--no-progress"]
    if via_params:
        args.extend(["--params", json.dumps(options)])
    else:
        for name, value in options.items():
            args.extend(["--" + name, value])
    if dry_run:
        args.append("--dry-run")

    result = CliRunner().invoke(app, args)

    message = f"root does not exist: {root}"
    assert result.exit_code == 2, result.output
    if format_name == "json":
        assert (
            result.stdout
            == json.dumps(
                {"error": {"code": "VALIDATION", "message": message}}, separators=(",", ":")
            )
            + "\n"
        )
        assert result.stderr == ""
    else:
        assert result.stdout == ""
        assert result.stderr == f"error: {message}\n"


@pytest.mark.parametrize("dry_run", [False, True], ids=["execute", "dry-run"])
@pytest.mark.parametrize("spelling", ["absolute", "relative", "tilde", "symlink", "file"])
def test_index_preserves_existing_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool, spelling: str
) -> None:
    root = tmp_path / "existing-root"
    if spelling == "file":
        root.touch()
    else:
        root.mkdir()
    expected_host = root.name
    if spelling == "relative":
        monkeypatch.chdir(tmp_path)
        value = root.name
    elif spelling == "tilde":
        monkeypatch.setenv("HOME", str(tmp_path))
        value = "~/existing-root"
    elif spelling == "symlink":
        link = tmp_path / "root-link"
        link.symlink_to(root, target_is_directory=True)
        value = str(link)
    else:
        value = str(root)

    result = CliRunner().invoke(
        app,
        ["index", "--root", value, "--no-embed", "--no-progress", "--json"]
        + (["--dry-run"] if dry_run else []),
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    if dry_run:
        assert payload["request"]["root"] == str(root)
        assert payload["request"]["host"] == expected_host
    else:
        assert payload["total"] == payload["indexed"] == payload["failed"] == 0


@pytest.mark.parametrize("dry_run", [False, True], ids=["execute", "dry-run"])
@pytest.mark.parametrize("via_params", [False, True], ids=["flags", "params"])
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"workers": "bananas"}, "workers must be a positive integer or 'auto'"),
        ({"workers": "0"}, "workers must be positive"),
        ({"workers": "-1"}, "workers must be positive"),
        ({"since": "bananas"}, "invalid since value: bananas"),
        (
            {"context": "bananas"},
            "invalid context mode: bananas. Expected one of: "
            "llm-codex, llm-local, llm-remote, off, template",
        ),
        (
            {"recompute_context": True, "only_mode": "bananas"},
            "unsupported context mode: bananas",
        ),
        (
            {"recreate": True, "yes": True, "project": "repo"},
            "recreate cannot be combined with recompute_context or project",
        ),
        (
            {"recreate": True, "yes": True, "recompute_context": True},
            "recreate cannot be combined with recompute_context or project",
        ),
    ],
)
def test_index_rejects_invalid_request(
    options: dict[str, str | bool], message: str, via_params: bool, dry_run: bool
) -> None:
    args = ["index", "--json", "--no-embed", "--no-progress"]
    if via_params:
        args.extend(["--params", json.dumps(options)])
    else:
        for name, value in options.items():
            args.append("--" + name.replace("_", "-"))
            if not isinstance(value, bool):
                args.append(value)
    if dry_run:
        args.append("--dry-run")

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout) == {"error": {"code": "VALIDATION", "message": message}}


@pytest.mark.parametrize("dry_run", [False, True], ids=["execute", "dry-run"])
@pytest.mark.parametrize(
    "options",
    [
        ["--workers", "1"],
        ["--workers", " +2 "],
        ["--workers", " AUTO "],
        ["--context", " template "],
        ["--since", ""],
        ["--since", "30d"],
        ["--since", "2024-01-01T00:00:00Z"],
        ["--only-mode", "ignored-without-recompute"],
        ["--root", "", "--host", ""],
        ["--full", "--project", "", "--source", "codex", "--verbose"],
    ],
)
def test_index_preserves_accepted_arguments(options: list[str], dry_run: bool) -> None:
    result = CliRunner().invoke(
        app,
        ["index", "--no-embed", "--no-progress", "--json", *options]
        + (["--dry-run"] if dry_run else []),
    )

    assert result.exit_code == 0, result.output


def test_index_preview_validates_after_flag_precedence_without_rpc(monkeypatch, tmp_path) -> None:
    def unexpected_rpc(*args, **kwargs):
        pytest.fail("index dry-run must not contact the daemon")

    monkeypatch.setattr("recall.cli.index.rpc_call_or_error", unexpected_rpc)
    result = CliRunner().invoke(
        app,
        [
            "index",
            "--params",
            json.dumps(
                {
                    "workers": "bananas",
                    "context": "bananas",
                    "since": "bananas",
                    "root": str(tmp_path / "missing-root"),
                }
            ),
            "--workers",
            "2",
            "--context",
            "template",
            "--since",
            "30d",
            "--root",
            str(tmp_path),
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    request = json.loads(result.stdout)["request"]
    assert request["workers"] == "2"
    assert request["context"] == "template"
    assert request["since"] == "30d"
    assert request["root"] == str(tmp_path)
