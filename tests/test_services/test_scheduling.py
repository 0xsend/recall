"""Scheduling counters report run-queue waiting and the OS priority clamp."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from recall.services.scheduling import Scheduling, _read_linux, parse_schedstat, read

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="QoS clamp is macOS")


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


def _child_scheduling(*prefix: str) -> Scheduling:
    code = (
        "from recall.services.scheduling import read\n"
        "x = 0\n"
        "for i in range(3_000_000): x += i\n"
        "print(read().model_dump_json())\n"
    )
    out = subprocess.run(
        [*prefix, sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return Scheduling.model_validate(json.loads(out.stdout))


@darwin_only
def test_utility_clamp_is_visible_as_priority_ceiling_and_qos_billing() -> None:
    clamped = _child_scheduling("taskpolicy", "-c", "utility")
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
