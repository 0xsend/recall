"""Scheduling counters report run-queue waiting and the OS priority clamp."""

import logging
import os
import selectors
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from recall.services import scheduling
from recall.services.scheduling import (
    Scheduling,
    _read_linux,
    demote_current_thread,
    demoted_threads,
    parse_schedstat,
    read,
)

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="QoS clamp is macOS")

# A hang guard, not a performance requirement: utility work yields to host load.
PROBE_HANG_SECONDS = 120


def test_schedstat_run_delay_is_the_second_field_in_nanoseconds() -> None:
    assert parse_schedstat("812345678 2500000000 41\n") == 2.5
    assert parse_schedstat("0 0 0") == 0.0


def test_malformed_schedstat_is_rejected() -> None:
    with pytest.raises(ValueError, match="2 fields"):
        parse_schedstat("812345678 2500000000")


def test_linux_waiting_sums_live_threads(tmp_path: Path) -> None:
    for tid, line in (("101", "10 1500000000 3\n"), ("102", "20 500000000 9\n")):
        (tmp_path / tid).mkdir()
        (tmp_path / tid / "schedstat").write_text(line)
    (tmp_path / "103").mkdir()  # exited between listing and reading

    result = _read_linux(tmp_path)

    assert result == Scheduling(
        runnable_waiting_seconds=2.0, priority_ceiling=None, qos_cpu_seconds=None
    )


def test_linux_without_task_directory_is_unsupported(tmp_path: Path) -> None:
    assert _read_linux(tmp_path / "missing") is None


def test_linux_without_schedstats_is_unsupported_not_zero(tmp_path: Path) -> None:
    for tid in ("101", "102"):
        (tmp_path / tid).mkdir()

    assert _read_linux(tmp_path) is None


def _child_output(code: str, *prefix: str) -> str:
    """Collect a native probe only after it signals that its work is complete."""
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb") as ready, os.fdopen(write_fd, "wb") as signal:
        code += f"\nimport os\nos.write({write_fd}, b'R')\n"
        with subprocess.Popen(
            [*prefix, sys.executable, "-c", code],
            pass_fds=(write_fd,),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as child:
            signal.close()
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(ready, selectors.EVENT_READ)
                    assert selector.select(PROBE_HANG_SECONDS), (
                        f"probe readiness hang guard expired after {PROBE_HANG_SECONDS}s; "
                        "this is not a scheduling performance deadline"
                    )
                notification = ready.read(1)
                output, error = child.communicate(timeout=PROBE_HANG_SECONDS)
            except subprocess.TimeoutExpired:
                pytest.fail(
                    f"probe exit hang guard expired after {PROBE_HANG_SECONDS}s following readiness"
                )
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()
    assert child.returncode == 0, error
    assert notification == b"R", "probe exited without its readiness signal"
    return output


def _child_scheduling(*prefix: str) -> Scheduling:
    code = (
        "from recall.services.scheduling import read\n"
        "x = 0\n"
        "for i in range(3_000_000): x += i\n"
        "print(read().model_dump_json())\n"
    )
    return Scheduling.model_validate_json(_child_output(code, *prefix))


@darwin_only
def test_utility_clamp_is_visible_as_priority_ceiling_and_qos_billing() -> None:
    # Children inherit a task clamp, and thread QoS cannot lift it. No fake
    # scheduler can create an unclamped task under a clamped macOS runner.
    parent = read()
    assert parent is not None and parent.priority_ceiling is not None
    if parent.priority_ceiling <= 20:
        pytest.skip("native macOS task QoS comparison requires an unclamped parent process")
    clamped = _child_scheduling("/usr/sbin/taskpolicy", "-c", "utility")
    unclamped = _child_scheduling()

    assert clamped.priority_ceiling == 20
    assert unclamped.priority_ceiling is not None and unclamped.priority_ceiling > 20
    assert clamped.qos_cpu_seconds is not None and unclamped.qos_cpu_seconds is not None
    clamped_cpu = sum(clamped.qos_cpu_seconds.values())
    assert clamped.qos_cpu_seconds["utility"] > 0.9 * clamped_cpu
    assert unclamped.qos_cpu_seconds["utility"] < 0.1 * sum(unclamped.qos_cpu_seconds.values())


@pytest.mark.skipif(
    sys.platform != "darwin" and not sys.platform.startswith("linux"),
    reason="scheduling counters are read on macOS and Linux",
)
def test_supported_platforms_report_nonnegative_waiting() -> None:
    result = read()

    assert result is not None
    assert result.runnable_waiting_seconds >= 0


def _demote_on_new_thread(name: str) -> str:
    outcomes: list[str] = []
    thread = threading.Thread(target=lambda: outcomes.append(demote_current_thread()), name=name)
    thread.start()
    thread.join(timeout=PROBE_HANG_SECONDS)
    assert not thread.is_alive(), (
        f"thread probe hang guard expired after {PROBE_HANG_SECONDS}s; "
        "this is not a scheduling performance deadline"
    )
    return outcomes[0]


@pytest.mark.skipif(sys.platform == "darwin", reason="thread QoS exists on macOS")
def test_demotion_is_a_recorded_no_op_off_macos() -> None:
    assert _demote_on_new_thread("demotion-off-macos") == "unsupported"
    assert demoted_threads()["demotion-off-macos"] == "unsupported"


@pytest.mark.skipif(sys.platform == "darwin", reason="needs a libc without thread QoS")
def test_demotion_failure_is_logged_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # This libc has no pthread_set_qos_class_self_np, as an old or unusual
    # macOS runtime might not: the thread keeps running at default QoS.
    monkeypatch.setattr(scheduling.sys, "platform", "darwin")

    with caplog.at_level(logging.WARNING, logger="recall.services.scheduling"):
        outcome = _demote_on_new_thread("demotion-unavailable")

    assert outcome == "failed"
    assert demoted_threads()["demotion-unavailable"] == "failed"
    assert "thread QoS demotion unavailable" in caplog.text


@darwin_only
def test_demotion_lowers_an_unclamped_thread_to_utility() -> None:
    code = (
        "import ctypes, threading\n"
        "from recall.services.scheduling import demote_current_thread\n"
        "libc = ctypes.CDLL(None)\n"
        "libc.pthread_self.restype = ctypes.c_void_p\n"
        "libc.pthread_get_qos_class_np.argtypes = (\n"
        "    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p\n"
        ")\n"
        "qos = ctypes.c_uint()\n"
        "def probe():\n"
        "    print(demote_current_thread())\n"
        "    libc.pthread_get_qos_class_np(libc.pthread_self(), ctypes.byref(qos), None)\n"
        "    print(hex(qos.value))\n"
        "t = threading.Thread(target=probe); t.start(); t.join()\n"
    )
    output = _child_output(code)

    assert output.split() == ["applied", "0x11"]
