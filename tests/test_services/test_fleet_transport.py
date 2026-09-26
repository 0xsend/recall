"""SSH fleet transport argv and status probe (REQ-FLEET-SSH / CMD-001)."""

from __future__ import annotations

import shlex

import pytest
from recall.core.fleet import FleetHost
from recall.services.fleet_transport import (
    MAX_REMOTE_ERROR_CHARS,
    FleetRemoteResult,
    build_ssh_argv,
    extract_remote_error,
    fleet_status,
    parse_daemon_status_text,
)


def test_build_ssh_argv_includes_required_options() -> None:
    argv = build_ssh_argv(
        "devbox.example",
        ["recall", "--version"],
        connect_timeout=12,
    )
    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert "RemoteCommand=none" in argv
    assert "ControlPath=none" in argv
    assert "ConnectTimeout=12" in argv
    assert "devbox.example" in argv
    target_idx = argv.index("devbox.example")
    assert argv[target_idx + 1] == "--"
    assert argv[target_idx + 2 :] == ["recall", "--version"]


def _remote_command(argv: list[str], target: str) -> str:
    """The single string ssh actually hands to the remote login shell."""
    target_idx = argv.index(target)
    assert argv[target_idx + 1] == "--"
    return " ".join(argv[target_idx + 2 :])


@pytest.mark.parametrize(
    ("case_id", "remote_argv"),
    [
        ("HP-01", ["recall", "--version"]),
        ("BV-01", ["recall", "search", "send connect evm", "--json", "--limit", "5"]),
        ("BV-02", ["recall", "list", "--json", "--project", "/home/dev/my projects/app"]),
        ("EDGE-01", ["recall", "search", 'it\'s a "quoted" $HOME `sub`', "--json"]),
        ("EDGE-02", ["recall", "search", "a;b|c&d>e", "--json"]),
        ("EDGE-03", ["recall", "search", "", "--json"]),
        ("EDGE-04", ["recall", "show", "abc123", "--json"]),
    ],
)
def test_build_ssh_argv_survives_remote_shell_word_splitting(
    case_id: str,
    remote_argv: list[str],
) -> None:
    """ssh has no remote argv: it joins trailing args with spaces and the remote
    login shell re-splits the result. One round of shell word-splitting MUST
    reproduce remote_argv exactly (REQ-FLEET-SSH-002)."""
    argv = build_ssh_argv("devbox.example", remote_argv)
    assert shlex.split(_remote_command(argv, "devbox.example")) == remote_argv, case_id


_PTY_NOTICE = "Pseudo-terminal will not be allocated because stdin is not a terminal."
_KNOWN_HOSTS_NOTICE = "Warning: Permanently added 'devbox' (ED25519) to the list of known hosts."


@pytest.mark.parametrize(
    ("case_id", "stdout", "stderr", "expected"),
    [
        ("ERR-01", "", "recall: no such option --nope\n", "recall: no such option --nope"),
        ("ERR-02", "remote db is locked\n", f"{_PTY_NOTICE}\n", "remote db is locked"),
        (
            "ERR-03",
            "",
            f"{_PTY_NOTICE}\nError: Got unexpected extra argument(s) (connect evm)\n",
            "Error: Got unexpected extra argument(s) (connect evm)",
        ),
        ("ERR-04", "", f"{_KNOWN_HOSTS_NOTICE}\n", "exit 2"),
        ("ERR-05", "", "", "exit 2"),
        ("ERR-06", "  \n\n", "   \n", "exit 2"),
        ("ERR-07", "", "Usage: recall\n\nTry --help.\n", "Usage: recall; Try --help."),
    ],
)
def test_extract_remote_error_drops_ssh_transport_noise(
    case_id: str,
    stdout: str,
    stderr: str,
    expected: str,
) -> None:
    """SSH client notices are informational and MUST NOT shadow the remote error
    (REQ-FLEET-SSH-005). Per-host skip lines are single-line, so the result never
    contains a newline."""
    error = extract_remote_error(stdout, stderr, 2)
    assert error == expected, case_id
    assert "\n" not in error, case_id


def test_extract_remote_error_is_bounded() -> None:
    """Per-host error strings are bounded (a remote traceback must not flood stderr)."""
    error = extract_remote_error("", "x" * (MAX_REMOTE_ERROR_CHARS * 3), 1)
    assert len(error) <= MAX_REMOTE_ERROR_CHARS
    assert error.endswith("...")


def test_parse_daemon_status_text() -> None:
    text = """
installed: true
daemon_version: 0.25.7
binary_version: 0.25.7
version_drift: false
"""
    parsed = parse_daemon_status_text(text)
    assert parsed["daemon_version"] == "0.25.7"
    assert parsed["binary_version"] == "0.25.7"
    assert parsed["version_drift"] is False


def test_fleet_status_skips_failed_host(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts = (
        FleetHost(name="ok-host", ssh="ok.example"),
        FleetHost(name="bad-host", ssh="bad.example"),
    )

    def fake_run(host: FleetHost, remote_argv: list[str], **_kwargs: object) -> FleetRemoteResult:
        if host.name == "ok-host":
            if remote_argv == ["recall", "--version"]:
                return FleetRemoteResult(ok=True, stdout="0.25.7\n", stderr="", returncode=0)
            return FleetRemoteResult(
                ok=True,
                stdout="daemon_version: 0.25.7\nbinary_version: 0.25.7\nversion_drift: false\n",
                stderr="",
                returncode=0,
            )
        return FleetRemoteResult(
            ok=False,
            stdout="",
            stderr="ssh: connect timed out",
            returncode=255,
            error="ssh: connect timed out",
        )

    monkeypatch.setattr("recall.services.fleet_transport.run_remote", fake_run)
    rows = fleet_status(hosts)
    assert len(rows) == 2
    ok = next(r for r in rows if r.name == "ok-host")
    bad = next(r for r in rows if r.name == "bad-host")
    assert ok.ok is True
    assert ok.binary_version == "0.25.7"
    assert ok.daemon_version == "0.25.7"
    assert ok.version_drift is False
    assert bad.ok is False
    assert bad.error
