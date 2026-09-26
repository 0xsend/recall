from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbedPreconditionResult:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class LoadThreshold:
    """The per-CPU load ceiling in force, and the power state that selected it.

    Two configured fractions can be in force and they differ by an order of
    magnitude, so a deferral naming a bare "threshold" leaves an operator unable
    to check the daemon's verdict against `pmset` (`REQ-ADAPT-006`).
    """

    fraction: float
    on_ac: bool
    power_detail: str = ""

    @property
    def name(self) -> str:
        """The configuration field this ceiling came from, as an operator reads it."""
        return "load threshold" if self.on_ac else "battery threshold"

    @property
    def power(self) -> str:
        """The power state that selected the ceiling."""
        if self.on_ac:
            return "AC power"
        return self.power_detail or "on battery"


def resolve_load_threshold(load_threshold: float, battery_threshold: float) -> LoadThreshold:
    """Read the power state and return the ceiling it puts in force."""
    power = check_power()
    if power.ok:
        return LoadThreshold(fraction=load_threshold, on_ac=True)
    return LoadThreshold(fraction=battery_threshold, on_ac=False, power_detail=power.reason)


def check_load(threshold: LoadThreshold) -> EmbedPreconditionResult:
    """Check if system load is below the in-force ceiling times cpu_count."""
    try:
        load_1m = os.getloadavg()[0]
    except (AttributeError, OSError):
        return EmbedPreconditionResult(ok=True)
    cpu_count = os.cpu_count() or 1
    max_load = threshold.fraction * cpu_count
    if load_1m > max_load:
        return EmbedPreconditionResult(
            ok=False,
            reason=f"load {load_1m:.1f} > {threshold.name} {max_load:.1f} ({threshold.power})",
        )
    return EmbedPreconditionResult(ok=True)


def check_power() -> EmbedPreconditionResult:
    """Check if on AC power (macOS only). Returns ok=True on other platforms."""
    if sys.platform != "darwin":
        return EmbedPreconditionResult(ok=True)
    return _check_macos_power()


def _check_macos_power() -> EmbedPreconditionResult:
    """Use IOKit to check AC power on macOS."""
    try:
        import ctypes
        import ctypes.util

        iokit_path = ctypes.util.find_library("IOKit")
        cf_path = ctypes.util.find_library("CoreFoundation")
        if iokit_path is None or cf_path is None:
            return EmbedPreconditionResult(ok=True, reason="IOKit/CoreFoundation not found")
        iokit = ctypes.cdll.LoadLibrary(iokit_path)
        cf = ctypes.cdll.LoadLibrary(cf_path)

        cf.CFRelease.argtypes = [ctypes.c_void_p]
        iokit.IOPSCopyPowerSourcesInfo.restype = ctypes.c_void_p
        iokit.IOPSGetProvidingPowerSourceType.restype = ctypes.c_void_p
        cf.CFStringGetCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_uint32,
        ]
        cf.CFStringGetCString.restype = ctypes.c_bool

        info = iokit.IOPSCopyPowerSourcesInfo()
        if not info:
            return EmbedPreconditionResult(ok=True)
        try:
            source_type = iokit.IOPSGetProvidingPowerSourceType(info)
            if not source_type:
                return EmbedPreconditionResult(ok=True)
            buf = ctypes.create_string_buffer(256)
            if cf.CFStringGetCString(source_type, buf, 256, 0x08000100):
                power_source = buf.value.decode("utf-8")
                if power_source == "AC Power":
                    return EmbedPreconditionResult(ok=True)
                return EmbedPreconditionResult(
                    ok=False,
                    reason=f"on battery: {power_source}",
                )
        finally:
            cf.CFRelease(info)
    except Exception:
        # If IOKit check fails, assume AC power (don't block embedding)
        return EmbedPreconditionResult(ok=True)
    return EmbedPreconditionResult(ok=True)
