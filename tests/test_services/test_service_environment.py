"""Service identity read from environment files and process environments (REQ-DAEMON-084).

Expected values follow systemd v255 `load_env_file` (src/basic/env-file.c) and
systemd.exec(5). Every vector carries a synthetic secret that must never leave
the reader.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from recall.services import service_environment
from recall.services.service_environment import (
    EnvironmentFile,
    IdentityKey,
    parse_environment_file_entry,
    read_environment_file,
    read_process_identity,
)

SECRET = "sk-sentinel-7f3a9c"
DATA = IdentityKey.RECALL_DATA_DIR


def _read(tmp_path: Path, content: bytes, *, ignore_errors: bool = False):
    path = tmp_path / "secrets.env"
    path.write_bytes(content)
    return read_environment_file(EnvironmentFile(path, ignore_errors))


def _refusal(tmp_path: Path, content: bytes, caplog) -> str:
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError) as raised:
        _read(tmp_path, content)
    message = str(raised.value)
    assert SECRET not in message
    assert SECRET not in caplog.text
    return message


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (f"API_KEY={SECRET}\n".encode(), ()),
        (f"API_KEY={SECRET}\nRECALL_DATA_DIR=/data\n".encode(), ((DATA, "/data"),)),
        (f"# RECALL_DATA_DIR=/x\n; {SECRET}\nRECALL_DATA_DIR=/d\n".encode(), ((DATA, "/d"),)),
        (b"  RECALL_DATA_DIR  =   /d   \n", ((DATA, "/d"),)),
        (b"RECALL_DATA_DIR=/with space\n", ((DATA, "/with space"),)),
        (b"RECALL_DATA_DIR='/single'\n", ((DATA, "/single"),)),
        (b'RECALL_DATA_DIR="/double"  \n', ((DATA, "/double"),)),
        # Quoted values span lines: identity-looking text inside one is not an assignment.
        (f'API_KEY="{SECRET}\nRECALL_DATA_DIR=/evil"\n'.encode(), ()),
        (f"API_KEY='{SECRET}\nRECALL_DATA_DIR=/evil'\n".encode(), ()),
        # A continuation line belongs to the value before it.
        (f"API_KEY={SECRET}\\\nRECALL_DATA_DIR=/evil\n".encode(), ()),
        (b"export RECALL_DATA_DIR=/exported\n", ()),
        (b"RECALL-DATA-DIR=/invalid\n", ()),
        (b"RECALL_DATA_DIR=/first\nRECALL_DATA_DIR=/last\n", ((DATA, "/first"), (DATA, "/last"))),
        (f"API_KEY={SECRET}\r\nRECALL_DATA_DIR=/crlf\r\n".encode(), ((DATA, "/crlf"),)),
        (b"RECALL_DATA_DIR=/no-newline", ((DATA, "/no-newline"),)),
        (
            b"HOME=/h\nRECALL_CONFIG_PATH=/c\n",
            ((IdentityKey.HOME, "/h"), (IdentityKey.RECALL_CONFIG_PATH, "/c")),
        ),
        # systemd strips a key's trailing whitespace, however long.
        (b"RECALL_DATA_DIR" + b" " * 200 + b"=/spaced\n", ((DATA, "/spaced"),)),
        (b"RECALL_DATA_DIR" + b" " * 200 + b"X=/other\n", ()),
        (b"RECALL_DATA_DIRX=/longer\n", ()),
        (f"# trailing escape \\ {SECRET}\nRECALL_DATA_DIR=/d\n".encode(), ((DATA, "/d"),)),
    ],
)
def test_reads_identity_assignments_as_systemd_does(tmp_path, caplog, content, expected) -> None:
    with caplog.at_level(logging.DEBUG):
        result = _read(tmp_path, content)
    assert result == expected
    assert SECRET not in repr(result)
    assert SECRET not in caplog.text


@pytest.mark.parametrize(
    ("content", "cause"),
    [
        (f"API_KEY={SECRET}\0\n".encode(), "contains a NUL byte"),
        (b"\xef\xbb\xbfRECALL_DATA_DIR=/d\n" + SECRET.encode(), "contains a non-ASCII byte"),
        (f"API_KEY={SECRET}".encode() + b"\xff\n", "contains a non-ASCII byte"),
        (f"# {SECRET} \\\nRECALL_DATA_DIR=/d\n".encode(), "comment line 1 ends with a backslash"),
        (f"A={SECRET}\nRECALL_DATA_DIR=/a\\b\n".encode(), "RECALL_DATA_DIR in EnvironmentFile"),
        (f"A={SECRET}\nRECALL_DATA_DIR='/a''/b'\n".encode(), "line 2 uses quoting or escapes"),
        (f"A={SECRET}\nRECALL_DATA_DIR='/a'/b\n".encode(), "uses quoting or escapes"),
        (f"A={SECRET}\nRECALL_DATA_DIR=/a'b'\n".encode(), "uses quoting or escapes"),
        (f'A={SECRET}\nRECALL_DATA_DIR="/a\\"b"\n'.encode(), "uses quoting or escapes"),
        (f"A={SECRET}\nRECALL_DATA_DIR='/a\n/b'\n".encode(), "uses quoting or escapes"),
        (f"A={SECRET}\nRECALL_DATA_DIR='/unterminated\n".encode(), "uses quoting or escapes"),
        (f"A={SECRET}\nRECALL_DATA_DIR=/a\\\n/b\n".encode(), "uses quoting or escapes"),
        (f"A={SECRET}\nHOME=/a\x01b\n".encode(), "contains a control character"),
        (f"A={SECRET}\nRECALL_DATA_DIR=/".encode() + b"x" * 4096 + b"\n", "longer than 4096"),
    ],
)
def test_refuses_what_it_cannot_prove(tmp_path, caplog, content, cause) -> None:
    assert cause in _refusal(tmp_path, content, caplog)


# systemd 255 rejects these code points (unichar_is_valid) and drops the whole
# file, so recall must not read past them; every non-ASCII byte refuses.
NON_ASCII_FILES = {
    "noncharacter-value": f"BAD=\ufdd0{SECRET}\nRECALL_DATA_DIR=/selected\n",
    "noncharacter-range-end": f"BAD={SECRET}\ufdef\nRECALL_DATA_DIR=/selected\n",
    "plane-0-fffe": f"BAD={SECRET}\ufffe\nRECALL_DATA_DIR=/selected\n",
    "plane-1-ffff": f"BAD={SECRET}\U0001ffff\nRECALL_DATA_DIR=/selected\n",
    "noncharacter-key": f"B\ufdd0D={SECRET}\nRECALL_DATA_DIR=/selected\n",
    "noncharacter-comment": f"# \ufdd0 {SECRET}\nRECALL_DATA_DIR=/selected\n",
    "noncharacter-identity": f"A={SECRET}\nRECALL_DATA_DIR=/sel\ufdd0ected\n",
    "letter-in-comment": f"# caf\u00e9 {SECRET}\nRECALL_DATA_DIR=/selected\n",
    "letter-in-identity": f"A={SECRET}\nRECALL_DATA_DIR=/caf\u00e9\n",
}


@pytest.mark.parametrize("ignore_errors", [False, True])
@pytest.mark.parametrize("content", NON_ASCII_FILES.values(), ids=NON_ASCII_FILES.keys())
def test_non_ascii_file_refuses_even_when_optional(
    tmp_path, caplog, content, ignore_errors
) -> None:
    path = tmp_path / "secrets.env"
    path.write_text(content, encoding="utf-8")
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError) as raised:
        read_environment_file(EnvironmentFile(path, ignore_errors))
    assert str(raised.value) == f"EnvironmentFile {path} contains a non-ASCII byte"
    assert SECRET not in caplog.text


def test_empty_identity_value_refuses(tmp_path, caplog) -> None:
    assert "is empty" in _refusal(tmp_path, f"A={SECRET}\nRECALL_DATA_DIR=\n".encode(), caplog)


@pytest.mark.parametrize("ignore_errors", [False, True])
def test_missing_file_is_skipped_only_when_optional(tmp_path, ignore_errors) -> None:
    entry = EnvironmentFile(tmp_path / "absent.env", ignore_errors)
    if ignore_errors:
        assert read_environment_file(entry) == ()
    else:
        with pytest.raises(ValueError, match="is missing and required"):
            read_environment_file(entry)


def test_dangling_symlink_is_missing(tmp_path) -> None:
    link = tmp_path / "link.env"
    link.symlink_to(tmp_path / "absent.env")
    assert read_environment_file(EnvironmentFile(link, True)) == ()


@pytest.mark.parametrize("ignore_errors", [False, True])
@pytest.mark.parametrize("kind", ["unreadable", "directory", "fifo", "oversize"])
def test_unreadable_file_refuses_even_when_optional(tmp_path, ignore_errors, kind) -> None:
    path = tmp_path / "secrets.env"
    match kind:
        case "unreadable":
            if os.geteuid() == 0:
                pytest.skip("root reads mode-000 files")
            path.write_text(f"RECALL_DATA_DIR=/d\nAPI_KEY={SECRET}\n")
            path.chmod(0)
            cause = "is unreadable by recall"
        case "directory":
            path.mkdir()
            cause = "is not a regular file"
        case "fifo":
            # A FIFO with no writer must not block discovery.
            os.mkfifo(path)
            cause = "is not a regular file"
        case _:
            path.write_bytes(b"A=" + b"x" * (1024 * 1024))
            cause = "exceeds 1 MiB"
    with pytest.raises(ValueError, match=cause) as raised:
        read_environment_file(EnvironmentFile(path, ignore_errors))
    assert SECRET not in str(raised.value)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/etc/recall/secrets.env (ignore_errors=yes)", ("/etc/recall/secrets.env", True)),
        ("/path with spaces/x.env (ignore_errors=no)", ("/path with spaces/x.env", False)),
    ],
)
def test_reads_manager_environment_file_entries(line, expected) -> None:
    entry = parse_environment_file_entry(line)
    assert (str(entry.path), entry.ignore_errors) == expected


@pytest.mark.parametrize(
    ("line", "cause"),
    [
        ("relative.env (ignore_errors=no)", "unreadable systemd EnvironmentFiles entry"),
        ("/etc/x.env", "unreadable systemd EnvironmentFiles entry"),
        ("/etc/x.env (ignore_errors=maybe)", "unreadable systemd EnvironmentFiles entry"),
        ("/etc/*.env (ignore_errors=yes)", "glob pattern"),
        ("/etc/x?.env (ignore_errors=no)", "glob pattern"),
        ("/etc/[ab].env (ignore_errors=no)", "glob pattern"),
    ],
)
def test_refuses_unreadable_manager_entries(line, cause) -> None:
    with pytest.raises(ValueError, match=cause):
        parse_environment_file_entry(line)


UNIT_CGROUP = "/user.slice/user-1000.slice/user@1000.service/app.slice/recall-daemon.service"


def _proc(tmp_path: Path, monkeypatch, environ: bytes, cgroup: str = f"0::{UNIT_CGROUP}\n") -> int:
    pid = 4242
    root = tmp_path / "proc"
    (root / str(pid)).mkdir(parents=True)
    (root / str(pid) / "environ").write_bytes(environ)
    (root / str(pid) / "cgroup").write_text(cgroup)
    (root / str(pid) / "cwd").symlink_to(tmp_path)
    monkeypatch.setattr(service_environment, "PROC_ROOT", root)
    return pid


def test_process_identity_reads_only_identity_entries(tmp_path, monkeypatch, caplog) -> None:
    environ = (
        f"API_KEY={SECRET}\0".encode()
        + b"RECALL_DATA_DIR=/running\0HOME=/home/u\0BINARY=\xff\xfe\0RECALL_DATA_DIRX=/no\0"
    )
    pid = _proc(tmp_path, monkeypatch, environ)
    with caplog.at_level(logging.DEBUG):
        identity = read_process_identity(pid, UNIT_CGROUP)
    assert identity.environment == {"RECALL_DATA_DIR": "/running", "HOME": "/home/u"}
    assert identity.working_directory == tmp_path
    assert SECRET not in repr(identity)
    assert SECRET not in caplog.text


@pytest.mark.parametrize(
    ("environ", "cause"),
    [
        (b"", "is empty"),
        (f"API_KEY={SECRET}".encode(), "is not NUL-terminated"),
        (f"API_KEY={SECRET}\0RECALL_DATA_DIR\0".encode(), "RECALL_DATA_DIR without a value"),
        (b"HOME=/a\0HOME=/b\0" + SECRET.encode() + b"\0", "sets HOME twice"),
        (f"A={SECRET}\0".encode() + b"RECALL_DATA_DIR=/\xff\0", "is not valid UTF-8"),
        (b"HOME=/" + b"x" * 4096 + b"\0", "longer than 4096"),
    ],
)
def test_process_identity_refuses_malformed_blocks(tmp_path, monkeypatch, environ, cause) -> None:
    pid = _proc(tmp_path, monkeypatch, environ)
    with pytest.raises(ValueError, match=cause) as raised:
        read_process_identity(pid, UNIT_CGROUP)
    assert SECRET not in str(raised.value)


@pytest.mark.parametrize(
    "cgroup",
    [
        f"0::{UNIT_CGROUP}\n",
        # cgroup v1: systemd tracks units in its named hierarchy.
        f"3:memory:/user.slice\n1:name=systemd:{UNIT_CGROUP}\n",
        # Hybrid: both systemd hierarchies name the unit.
        f"1:name=systemd:{UNIT_CGROUP}\n0::{UNIT_CGROUP}\n",
    ],
)
def test_process_identity_accepts_a_process_in_the_units_control_group(
    tmp_path, monkeypatch, cgroup
) -> None:
    pid = _proc(tmp_path, monkeypatch, b"HOME=/h\0", cgroup)
    assert read_process_identity(pid, UNIT_CGROUP).environment == {"HOME": "/h"}


@pytest.mark.parametrize(
    ("cgroup", "unit_cgroup", "cause"),
    [
        ("0::/user.slice/user-1000.slice/session-2.scope\n", UNIT_CGROUP, "control group"),
        (f"0::{UNIT_CGROUP}/child\n", UNIT_CGROUP, "control group"),
        (f"1:name=systemd:/other.scope\n0::{UNIT_CGROUP}\n", UNIT_CGROUP, "control group"),
        ("3:memory:/user.slice\n", UNIT_CGROUP, "names no systemd control group"),
        ("", UNIT_CGROUP, "names no systemd control group"),
        (f"0::{UNIT_CGROUP}\n", "", "reports no control group"),
        ("0::/\n", "/", "reports no control group"),
    ],
)
def test_process_identity_refuses_a_process_outside_the_unit(
    tmp_path, monkeypatch, cgroup, unit_cgroup, cause
) -> None:
    pid = _proc(tmp_path, monkeypatch, f"API_KEY={SECRET}\0HOME=/h\0".encode(), cgroup)
    with pytest.raises(ValueError, match=cause) as raised:
        read_process_identity(pid, unit_cgroup)
    assert "; retry" in str(raised.value)
    assert SECRET not in str(raised.value)


def test_process_identity_refuses_an_unreadable_process(tmp_path, monkeypatch) -> None:
    if os.geteuid() == 0:
        pytest.skip("root reads mode-000 files")
    pid = _proc(tmp_path, monkeypatch, f"API_KEY={SECRET}\0".encode())
    (tmp_path / "proc" / str(pid) / "environ").chmod(0)
    with pytest.raises(ValueError, match="cannot be read during identification") as raised:
        read_process_identity(pid, UNIT_CGROUP)
    assert SECRET not in str(raised.value)


def test_process_identity_refuses_a_vanished_process(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(service_environment, "PROC_ROOT", tmp_path)
    with pytest.raises(ValueError, match="changed or cannot be read during identification"):
        read_process_identity(4242, UNIT_CGROUP)


def _control_group(pid: int) -> str:
    lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    return next(line.removeprefix("0::") for line in lines if line.startswith("0::"))


def _child(environment: dict[str, str], cwd: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        env=environment,
        cwd=cwd,
        stdin=subprocess.PIPE,
    )


def _stop(child: subprocess.Popen[bytes]) -> None:
    assert child.stdin is not None
    child.stdin.close()
    child.wait(timeout=10)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs is Linux-only")
def test_process_identity_reads_a_real_process(tmp_path) -> None:
    env = {"RECALL_DATA_DIR": "relative/data", "HOME": str(tmp_path), "API_KEY": SECRET}
    child = _child(env, tmp_path)
    try:
        identity = read_process_identity(child.pid, _control_group(child.pid))
        assert identity.environment == {"RECALL_DATA_DIR": "relative/data", "HOME": str(tmp_path)}
        assert identity.working_directory == tmp_path.resolve()
    finally:
        _stop(child)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="procfs is Linux-only")
def test_process_identity_never_mixes_two_process_instances(tmp_path, monkeypatch) -> None:
    """The process exits and its number is reused between the environ and cwd reads."""
    started, reused_in = tmp_path / "started", tmp_path / "reused"
    started.mkdir()
    reused_in.mkdir()
    original = _child({"RECALL_DATA_DIR": "relative", "API_KEY": SECRET}, started)
    reuser = _child({"RECALL_DATA_DIR": "relative"}, reused_in)
    proc = tmp_path / "proc"
    proc.mkdir()
    entry = proc / str(original.pid)
    entry.symlink_to(f"/proc/{original.pid}")
    monkeypatch.setattr(service_environment, "PROC_ROOT", proc)
    readlink = os.readlink

    def replace_then_readlink(path, *args, **kwargs):
        if os.fspath(path).endswith("cwd") and entry.readlink() != Path(f"/proc/{reuser.pid}"):
            _stop(original)
            entry.unlink()
            entry.symlink_to(f"/proc/{reuser.pid}")
        return readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", replace_then_readlink)
    try:
        with pytest.raises(ValueError, match="changed or cannot be read during identification"):
            read_process_identity(original.pid, _control_group(original.pid))
        # A reader that reopened the number would now see the other process.
        assert readlink(entry / "cwd") == str(reused_in)
    finally:
        _stop(original)
        _stop(reuser)
