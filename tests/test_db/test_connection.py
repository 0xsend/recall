from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from recall.db import RecallLockError, advisory_lock


def _probe_advisory_lock(lock_path: Path) -> subprocess.CompletedProcess[str]:
    script = """
import sys
from pathlib import Path

from recall.db import RecallLockError, advisory_lock

try:
    with advisory_lock(Path(sys.argv[1])):
        pass
except RecallLockError:
    raise SystemExit(23)
"""
    return subprocess.run(
        [sys.executable, "-c", script, str(lock_path)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_advisory_lock_reentrant_same_thread(tmp_path: Path) -> None:
    lock_path = tmp_path / "recall.lock"

    with advisory_lock(lock_path), advisory_lock(lock_path):
        pass

    result = _probe_advisory_lock(lock_path)

    assert result.returncode == 0, result.stderr


def test_advisory_lock_blocks_other_process(tmp_path: Path) -> None:
    lock_path = tmp_path / "recall.lock"

    with advisory_lock(lock_path):
        result = _probe_advisory_lock(lock_path)

    assert result.returncode == 23, result.stderr


def test_advisory_lock_blocks_cross_thread_in_same_process(tmp_path: Path) -> None:
    lock_path = tmp_path / "recall.lock"
    first_holding_lock = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    second_acquired_lock: list[bool] = []
    thread_errors: list[BaseException] = []

    def hold_lock() -> None:
        try:
            with advisory_lock(lock_path):
                first_holding_lock.set()
                if not release_first.wait(timeout=5):
                    raise TimeoutError("timed out waiting to release advisory lock")
        except BaseException as err:
            thread_errors.append(err)

    def contend_for_lock() -> None:
        try:
            if not first_holding_lock.wait(timeout=5):
                raise TimeoutError("timed out waiting for first advisory lock holder")
            with advisory_lock(lock_path):
                second_acquired_lock.append(True)
        except RecallLockError:
            pass
        except BaseException as err:
            thread_errors.append(err)
        finally:
            second_done.set()

    first = threading.Thread(target=hold_lock)
    second = threading.Thread(target=contend_for_lock)
    first.start()
    try:
        assert first_holding_lock.wait(timeout=5)
        second.start()
        assert second_done.wait(timeout=5)
    finally:
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert thread_errors == []
    assert second_acquired_lock == []
