from __future__ import annotations

import json
import logging
import os
import stat
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
from recall.cli.daemon import (
    _rotate_daemon_log_if_needed,
    _start_foreground_server,
)


@pytest.fixture(autouse=True)
def _restore_process_logging() -> Iterator[None]:
    """Starting the foreground server configures this process's logging for real."""
    root = logging.getLogger()
    recall_logger = logging.getLogger("recall")
    handlers = root.handlers[:]
    root_level = root.level
    recall_level = recall_logger.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(root_level)
        recall_logger.setLevel(recall_level)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _run_probe(probe: str, *args: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(probe), *(str(arg) for arg in args)],
        capture_output=True,
        check=False,
        env=os.environ.copy(),
        text=True,
    )


def test_progress_bars_disabled_on_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TQDM_DISABLE", raising=False)

    _start_foreground_server(_exit_before_bind=True)

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TQDM_DISABLE"] == "1"


@pytest.mark.parametrize(
    ("user_set", "other"),
    [
        ("HF_HUB_DISABLE_PROGRESS_BARS", "TQDM_DISABLE"),
        ("TQDM_DISABLE", "HF_HUB_DISABLE_PROGRESS_BARS"),
    ],
)
def test_does_not_override_user_value(
    monkeypatch: pytest.MonkeyPatch, user_set: str, other: str
) -> None:
    monkeypatch.setenv(user_set, "0")
    monkeypatch.delenv(other, raising=False)

    _start_foreground_server(_exit_before_bind=True)

    assert os.environ[user_set] == "0"


def test_no_hf_imports_before_setdefault() -> None:
    probe = textwrap.dedent(
        """
        import json
        import sys

        from recall.cli.daemon import _start_foreground_server

        _start_foreground_server(
            verbose=False,
            mode_override=None,
            embed_override=False,
            source_override=None,
            batch_size_override=None,
            _exit_before_bind=True,
        )
        deny = ("huggingface_hub", "transformers", "sentence_transformers", "tokenizers")
        loaded = sorted(k for k in sys.modules if any(needle in k for needle in deny))
        print(json.dumps(loaded))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        check=False,
        env=os.environ.copy(),
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == []


def test_rotate_when_oversized(tmp_path: Path) -> None:
    path = tmp_path / "daemon.log"
    payload = b"before rotation"
    path.write_bytes(payload)

    _rotate_daemon_log_if_needed(path, max_bytes=4, target_fd=None)

    assert (tmp_path / "daemon.log.1").read_bytes() == payload
    assert path.read_bytes() == b""


def test_no_rotate_when_under(tmp_path: Path) -> None:
    path = tmp_path / "daemon.log"
    payload = b"small"
    path.write_bytes(payload)

    _rotate_daemon_log_if_needed(path, max_bytes=1024, target_fd=None)

    assert path.read_bytes() == payload
    assert not (tmp_path / "daemon.log.1").exists()


def test_under_threshold_log_perms_tightened(tmp_path: Path) -> None:
    path = tmp_path / "daemon.log"
    payload = b"small"
    path.write_bytes(payload)
    os.chmod(path, 0o644)

    _rotate_daemon_log_if_needed(path, max_bytes=10_000_000, target_fd=None)

    assert _mode(path) == 0o600
    assert path.read_bytes() == payload
    assert not (tmp_path / "daemon.log.1").exists()


def test_under_threshold_chmod_failure_does_not_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "daemon.log"
    path.write_bytes(b"small")

    def fail_chmod(chmod_path: Path, _mode: int) -> None:
        if chmod_path == path:
            raise OSError("permission denied")

    monkeypatch.setattr(os, "chmod", fail_chmod)
    caplog.set_level(logging.WARNING, logger="recall.cli.daemon")

    _rotate_daemon_log_if_needed(path, max_bytes=10_000_000, target_fd=None)

    assert "chmod 0600 on under-threshold" in caplog.text


def test_no_rotate_when_disabled(tmp_path: Path) -> None:
    path = tmp_path / "daemon.log"
    payload = b"oversized but disabled"
    path.write_bytes(payload)

    _rotate_daemon_log_if_needed(path, max_bytes=0, target_fd=None)

    assert path.read_bytes() == payload
    assert not (tmp_path / "daemon.log.1").exists()


def test_overwrites_prior_generation(tmp_path: Path) -> None:
    path = tmp_path / "daemon.log"
    rotated = tmp_path / "daemon.log.1"
    path.write_bytes(b"new generation")
    rotated.write_bytes(b"old generation")

    _rotate_daemon_log_if_needed(path, max_bytes=4, target_fd=None)

    assert rotated.read_bytes() == b"new generation"
    assert path.read_bytes() == b""


def test_both_streams_rotated_when_both_oversized(tmp_path: Path) -> None:
    log_path = tmp_path / "daemon.log"
    err_log_path = tmp_path / "daemon.err.log"
    log_path.write_bytes(b"stdout before")
    err_log_path.write_bytes(b"stderr before")
    probe = """
        import os
        import sys
        from pathlib import Path

        from recall.cli.daemon import _rotate_daemon_log_if_needed

        log_path = Path(sys.argv[1])
        err_log_path = Path(sys.argv[2])
        fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND)
        try:
            os.dup2(fd, 1)
        finally:
            os.close(fd)
        err_fd = os.open(str(err_log_path), os.O_WRONLY | os.O_APPEND)
        try:
            os.dup2(err_fd, 2)
        finally:
            os.close(err_fd)

        _rotate_daemon_log_if_needed(log_path, 4, 1)
        _rotate_daemon_log_if_needed(err_log_path, 4, 2)
        os.write(1, b"STDOUT_AFTER\\n")
        os.write(2, b"STDERR_AFTER\\n")
    """

    result = _run_probe(probe, log_path, err_log_path)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "daemon.log.1").read_bytes() == b"stdout before"
    assert (tmp_path / "daemon.err.log.1").read_bytes() == b"stderr before"
    assert log_path.read_bytes() == b"STDOUT_AFTER\n"
    assert err_log_path.read_bytes() == b"STDERR_AFTER\n"


def test_dup2_failure_does_not_blackhole(tmp_path: Path) -> None:
    path = tmp_path / "daemon.log"
    warning_path = tmp_path / "warnings.log"
    path.write_bytes(b"stdout before")
    probe = """
        import logging
        import os
        import sys
        from pathlib import Path

        import recall.cli.daemon as daemon

        path = Path(sys.argv[1])
        warning_path = Path(sys.argv[2])

        def fail_dup2(_new_fd: int, _target_fd: int) -> None:
            raise OSError("EBADF")

        handler = logging.FileHandler(warning_path)
        daemon.logger.addHandler(handler)
        daemon.logger.setLevel(logging.WARNING)
        daemon.os.dup2 = fail_dup2
        daemon._rotate_daemon_log_if_needed(path, 4, 99)
        handler.flush()
        handler.close()
        if "dup2" not in warning_path.read_text(encoding="utf-8"):
            raise SystemExit(2)
    """

    result = _run_probe(probe, path, warning_path)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "daemon.log.1").read_bytes() == b"stdout before"
    assert path.read_bytes() == b""


def test_active_log_perms_are_0600(tmp_path: Path) -> None:
    path = tmp_path / "daemon.log"
    path.write_bytes(b"stdout before")

    _rotate_daemon_log_if_needed(path, max_bytes=4, target_fd=None)

    assert _mode(path) == 0o600


def test_rotated_log_perms_are_0600(tmp_path: Path) -> None:
    path = tmp_path / "daemon.log"
    path.write_bytes(b"stdout before")
    os.chmod(path, 0o644)

    _rotate_daemon_log_if_needed(path, max_bytes=4, target_fd=None)

    assert _mode(tmp_path / "daemon.log.1") == 0o600


def test_rotated_log_chmod_failure_logs_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "daemon.log"
    path.write_bytes(b"stdout before")

    def fail_chmod(_path: Path, _mode: int) -> None:
        raise OSError("EPERM")

    monkeypatch.setattr(os, "chmod", fail_chmod)
    caplog.set_level(logging.WARNING, logger="recall.cli.daemon")

    _rotate_daemon_log_if_needed(path, max_bytes=4, target_fd=None)

    assert (tmp_path / "daemon.log.1").exists()
    assert "chmod 0600 on rotated" in caplog.text


def test_rotation_handles_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "daemon.log"
    payload = b"stdout before"
    path.write_bytes(payload)

    def fail_replace(_src: Path, _dst: Path) -> None:
        raise OSError("EXDEV")

    monkeypatch.setattr(os, "replace", fail_replace)
    caplog.set_level(logging.WARNING, logger="recall.cli.daemon")

    _rotate_daemon_log_if_needed(path, max_bytes=4, target_fd=None)

    assert path.read_bytes() == payload
    assert not (tmp_path / "daemon.log.1").exists()
    assert "replace" in caplog.text


def test_rotation_handles_open_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "daemon.log"
    rotated = tmp_path / "daemon.log.1"
    payload = b"stdout before"
    path.write_bytes(payload)
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(src: Path, dst: Path) -> None:
        replace_calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    def fail_open(_path: str, _flags: int, _mode: int = 0o777) -> int:
        raise OSError("EMFILE")

    monkeypatch.setattr(os, "replace", recording_replace)
    monkeypatch.setattr(os, "open", fail_open)
    caplog.set_level(logging.WARNING, logger="recall.cli.daemon")

    _rotate_daemon_log_if_needed(path, max_bytes=4, target_fd=None)

    assert replace_calls == [(path, rotated), (rotated, path)]
    assert path.read_bytes() == payload
    assert not rotated.exists()
    assert "open of fresh" in caplog.text


def test_start_foreground_server_skips_rotation_when_fd_not_managed_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import recall.cli.daemon as daemon

    calls: list[tuple[Path, int, int | None]] = []
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "missing.toml"))
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)
    monkeypatch.setattr(daemon, "_fd_is_managed_log", lambda _fd, _path: False)
    monkeypatch.setattr(
        daemon,
        "_rotate_daemon_log_if_needed",
        lambda path, max_bytes, target_fd: calls.append((path, max_bytes, target_fd)),
    )

    _start_foreground_server(_exit_before_bind=True)

    assert calls == []

    monkeypatch.setattr(daemon, "_fd_is_managed_log", lambda _fd, _path: True)

    _start_foreground_server(_exit_before_bind=True)

    assert calls == [
        (tmp_path / "logs" / "daemon.log", 52_428_800, 1),
        (tmp_path / "logs" / "daemon.err.log", 52_428_800, 2),
    ]


def test_fd_is_managed_log_inode_match(tmp_path: Path) -> None:
    log_path = tmp_path / "daemon.log"
    other_path = tmp_path / "other.log"
    result_path = tmp_path / "result.json"
    log_path.write_bytes(b"log")
    other_path.write_bytes(b"other")
    probe = """
        import json
        import os
        import pty
        import sys
        from pathlib import Path

        from recall.cli.daemon import _fd_is_managed_log

        log_path = Path(sys.argv[1])
        other_path = Path(sys.argv[2])
        result_path = Path(sys.argv[3])
        pipe_match = _fd_is_managed_log(1, log_path)

        fd = os.open(str(log_path), os.O_WRONLY | os.O_APPEND)
        try:
            os.dup2(fd, 1)
        finally:
            os.close(fd)
        inode_match = _fd_is_managed_log(1, log_path)
        different_file_match = _fd_is_managed_log(1, other_path)

        master_fd, slave_fd = pty.openpty()
        try:
            os.dup2(slave_fd, 1)
            tty_match = _fd_is_managed_log(1, log_path)
        finally:
            os.close(slave_fd)
            os.close(master_fd)

        result_path.write_text(
            json.dumps(
                {
                    "pipe_match": pipe_match,
                    "inode_match": inode_match,
                    "different_file_match": different_file_match,
                    "tty_match": tty_match,
                }
            ),
            encoding="utf-8",
        )
    """

    result = _run_probe(probe, log_path, other_path, result_path)

    assert result.returncode == 0, result.stderr
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "pipe_match": False,
        "inode_match": True,
        "different_file_match": False,
        "tty_match": False,
    }


def test_rotation_runs_once_per_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import recall.cli.daemon as daemon

    calls: list[tuple[Path, int, int | None]] = []
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / "missing.toml"))
    monkeypatch.setattr("recall.core.config.embedding_backend_available", lambda _backend: False)
    monkeypatch.setattr(daemon, "_fd_is_managed_log", lambda _fd, _path: True)
    monkeypatch.setattr(
        daemon,
        "_rotate_daemon_log_if_needed",
        lambda path, max_bytes, target_fd: calls.append((path, max_bytes, target_fd)),
    )

    _start_foreground_server(_exit_before_bind=True)

    assert [call[2] for call in calls] == [1, 2]

    _start_foreground_server(_exit_before_bind=True)

    assert len(calls) == 4
    assert [call[2] for call in calls] == [1, 2, 1, 2]
    repo_root = Path(__file__).parents[2]
    rpc_server_source = (repo_root / "packages/recall/src/recall/services/rpc_server.py").read_text(
        encoding="utf-8"
    )
    assert "_rotate_daemon_log_if_needed" not in rpc_server_source


def test_refused_start_exits_3_instead_of_0(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-RESIL-016: a start refused by the self-repair step exits non-zero so the
    scheduler is not handed a clean exit into an identical restart."""
    import typer
    from recall.services import rpc_server as rpc_server_module
    from recall.services.self_repair import DaemonStartupRefused

    def refuse(**_kwargs: object) -> None:
        raise DaemonStartupRefused("index divergence recurred after rebuild")

    monkeypatch.setattr(rpc_server_module, "start_server_blocking", refuse)
    monkeypatch.setattr("recall.cli.daemon._fd_is_managed_log", lambda _fd, _path: False)

    with pytest.raises(typer.Exit) as excinfo:
        _start_foreground_server()

    assert excinfo.value.exit_code == 3
