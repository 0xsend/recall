from __future__ import annotations

import json

import pytest
from recall.cli.app import app
from typer.testing import CliRunner


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


def test_index_preview_validates_after_flag_precedence_without_rpc(monkeypatch) -> None:
    def unexpected_rpc(*args, **kwargs):
        pytest.fail("index dry-run must not contact the daemon")

    monkeypatch.setattr("recall.cli.index.rpc_call_or_error", unexpected_rpc)
    result = CliRunner().invoke(
        app,
        [
            "index",
            "--params",
            json.dumps({"workers": "bananas", "context": "bananas", "since": "bananas"}),
            "--workers",
            "2",
            "--context",
            "template",
            "--since",
            "30d",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    request = json.loads(result.stdout)["request"]
    assert request["workers"] == "2"
    assert request["context"] == "template"
    assert request["since"] == "30d"
