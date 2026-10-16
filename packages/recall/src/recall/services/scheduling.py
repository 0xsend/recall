"""Process CPU-scheduling counters for daemon diagnostics.

Wall time that is neither CPU nor blocking is time a runnable thread waited for
a core. Under host contention that wait dominates when the OS runs the daemon
at a low priority (launchd applies a QoS clamp to jobs without `ProcessType`),
so diagnostics record it beside CPU time.
"""

from __future__ import annotations

import ctypes
import itertools
import os
import sys
import threading
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel

QosClass = Literal[
    "default",
    "maintenance",
    "background",
    "utility",
    "legacy",
    "user_initiated",
    "user_interactive",
]

# Thread directories summed per Linux sample; a daemon runs about a hundred.
LINUX_TASKS_MAX = 4096


class Scheduling(BaseModel):
    """Cumulative for the process lifetime unless stated otherwise."""

    # Thread-seconds runnable but not on a core. macOS: the difference of two
    # kernel counters for the whole process, exited threads included. Linux:
    # summed `run_delay` of the threads alive at sampling time.
    runnable_waiting_seconds: float
    # macOS: the sampling thread's maximum scheduling priority. A QoS clamp
    # lowers it for every thread (utility clamp: 20; unclamped: 63).
    priority_ceiling: int | None
    # macOS: CPU time billed per QoS class.
    qos_cpu_seconds: dict[QosClass, float] | None


def read() -> Scheduling | None:
    """Return this process's scheduling counters, or None where unsupported."""
    if sys.platform == "darwin":
        return _read_darwin()
    if sys.platform.startswith("linux"):
        return _read_linux(Path("/proc/self/task"))
    return None


def parse_schedstat(text: str) -> float:
    """Seconds of `run_delay` from one `/proc/<pid>/task/<tid>/schedstat` line.

    The line holds time on CPU (ns), time waiting on a runqueue (ns) and the
    number of timeslices run on this CPU.
    """
    fields = text.split()
    if len(fields) != 3:
        raise ValueError(f"schedstat has {len(fields)} fields, expected 3")
    return int(fields[1]) / 1e9


def _read_linux(tasks: Path) -> Scheduling | None:
    try:
        names = os.listdir(tasks)
    except FileNotFoundError:
        return None
    waiting = 0.0
    for name in itertools.islice(names, LINUX_TASKS_MAX):
        try:
            waiting += parse_schedstat((tasks / name / "schedstat").read_text())
        except FileNotFoundError:
            continue  # the thread exited after listing
    return Scheduling(runnable_waiting_seconds=waiting, priority_ceiling=None, qos_cpu_seconds=None)


_U64 = ctypes.c_uint64
_RUSAGE_INFO_V6 = 6
_PROC_PIDTHREADID64INFO = 15
_RUSAGE_FIELDS = (
    "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
    "wired_size", "resident_size", "phys_footprint", "proc_start_abstime",
    "proc_exit_abstime", "child_user_time", "child_system_time",
    "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins",
    "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
    "cpu_time_qos_default", "cpu_time_qos_maintenance", "cpu_time_qos_background",
    "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated",
    "cpu_time_qos_user_interactive", "billed_system_time", "serviced_system_time",
    "logical_writes", "lifetime_max_phys_footprint", "instructions", "cycles",
    "billed_energy", "serviced_energy", "interval_max_phys_footprint",
    "runnable_time", "flags", "user_ptime", "system_ptime", "pinstructions",
    "pcycles", "energy_nj", "penergy_nj", "secure_time_in_system",
    "secure_ptime_in_system", "neural_footprint", "lifetime_max_neural_footprint",
    "interval_max_neural_footprint",
)  # fmt: skip


class _RusageInfoV6(ctypes.Structure):
    """`struct rusage_info_v6` from <sys/resource.h>; times are Mach absolute ticks."""

    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        *((name, _U64) for name in _RUSAGE_FIELDS),
        ("reserved", _U64 * 9),
    ]


class _ThreadInfo(ctypes.Structure):
    """`struct proc_threadinfo` from <sys/proc_info.h>."""

    _fields_ = [
        ("user_time", _U64),
        ("system_time", _U64),
        ("cpu_usage", ctypes.c_int32),
        ("policy", ctypes.c_int32),
        ("run_state", ctypes.c_int32),
        ("flags", ctypes.c_int32),
        ("sleep_time", ctypes.c_int32),
        ("curpri", ctypes.c_int32),
        ("priority", ctypes.c_int32),
        ("maxpriority", ctypes.c_int32),
        ("name", ctypes.c_char * 64),
    ]


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


_darwin: tuple[ctypes.CDLL, float] | None = None


def _darwin_libraries() -> tuple[ctypes.CDLL, float]:
    global _darwin
    if _darwin is None:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        timebase = _Timebase()
        ctypes.CDLL(None).mach_timebase_info(ctypes.byref(timebase))
        _darwin = (libproc, timebase.numer / timebase.denom / 1e9)
    return _darwin


def _read_darwin() -> Scheduling:
    libproc, seconds_per_tick = _darwin_libraries()
    usage = _RusageInfoV6()
    if libproc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V6, ctypes.byref(usage)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"proc_pid_rusage: {os.strerror(error)}")
    cpu = usage.user_time + usage.system_time
    thread = _ThreadInfo()
    size = libproc.proc_pidinfo(
        os.getpid(),
        _PROC_PIDTHREADID64INFO,
        _U64(threading.get_native_id()),
        ctypes.byref(thread),
        ctypes.sizeof(thread),
    )
    if size != ctypes.sizeof(thread):
        error = ctypes.get_errno()
        raise OSError(error, f"proc_pidinfo thread info: {os.strerror(error)}")
    qos: dict[QosClass, float] = {
        name: getattr(usage, f"cpu_time_qos_{name}") * seconds_per_tick
        for name in get_args(QosClass)
    }
    # Runnable time includes time on a core; the remainder is queueing. The two
    # counters are not read atomically: CPU the calling thread accrues between
    # them can make the difference a few ticks negative, which is not a wait.
    waiting_ticks = max(0, usage.runnable_time - cpu)
    return Scheduling(
        runnable_waiting_seconds=waiting_ticks * seconds_per_tick,
        priority_ceiling=thread.maxpriority,
        qos_cpu_seconds=qos,
    )
