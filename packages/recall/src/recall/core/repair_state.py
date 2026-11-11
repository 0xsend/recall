"""Admission decisions for a fenced database's durable recovery state.

Decisions describe effects. Execution additionally requires the live owner and
attempt capability; a persisted phase never grants execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never


class Phase(StrEnum):
    IDLE = "idle"
    REQUIRED = "required"
    RUNNING = "running"
    VERIFIED = "verified"
    REFUSED = "refused"
    INVALID = "invalid"


class Marker(StrEnum):
    NONE = "none"
    INDEX = "index"
    DISK = "disk"
    OTHER = "other"
    PROBE = "probe"
    REPLAY = "replay"
    MIXED = "mixed"
    INVALID = "invalid"


class Refusal(StrEnum):
    ABSENT = "absent"
    PRESENT = "present"
    UNREADABLE = "unreadable"


class Budget(StrEnum):
    AVAILABLE = "available"
    SPENT = "spent"
    INVALID = "invalid"


class Event(StrEnum):
    START_NEGATIVE = "start_negative"
    START_POSITIVE = "start_positive"
    FOLD = "fold"
    ARM = "arm"
    STEP = "step"
    VERIFY = "verify"
    RETIRE = "retire"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    KILL = "kill"
    PERSIST_FAIL = "persist_fail"
    MANUAL_REBUILD = "manual_rebuild"
    MANUAL_COMPACT = "manual_compact"
    MANUAL_ARM = "manual_arm"
    MANUAL_STEP = "manual_step"
    MANUAL_RELEASE = "manual_release"
    AUTO_COMPACT = "auto_compact"
    OLD_START = "old_start"
    OLD_RESET = "old_reset"
    INVALID = "invalid_event"


class Action(StrEnum):
    HOLD = "HOLD"
    NORMAL = "NORMAL"
    REQUIRE_INDEX = "REQUIRE_INDEX"
    REQUIRE_PROBE = "REQUIRE_PROBE"
    RESUME_REQUIRED = "RESUME_REQUIRED"
    DIAGNOSE = "DIAGNOSE"
    FOLD = "FOLD"
    ARM = "ARM"
    STEP = "STEP"
    VERIFY = "VERIFY"
    RETIRE = "RETIRE"
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"
    UNCHANGED = "UNCHANGED"
    MANUAL = "MANUAL"
    MANUAL_ARM = "MANUAL_ARM"
    MANUAL_STEP = "MANUAL_STEP"
    MANUAL_RELEASE = "MANUAL_RELEASE"
    PRESERVE = "PRESERVE"
    OLD_FENCE = "OLD_FENCE"
    OLD_RESET = "OLD_RESET"
    REFUSE = "REFUSE"


@dataclass(frozen=True)
class Admission:
    phase: Phase
    marker: Marker
    refusal: Refusal
    budget: Budget


def decide(state: Admission, event: Event) -> Action:
    phase, marker, refusal, budget = state.phase, state.marker, state.refusal, state.budget
    match event:
        case Event.OLD_RESET:
            return Action.OLD_RESET
        case Event.OLD_START:
            return Action.OLD_FENCE
        case Event.KILL:
            return Action.UNCHANGED
        case Event.PERSIST_FAIL:
            return Action.ERROR
        case _:
            pass

    match phase:
        case Phase.INVALID:
            return Action.HOLD
        case Phase.IDLE | Phase.REQUIRED | Phase.RUNNING | Phase.VERIFIED | Phase.REFUSED:
            pass
        case _:
            assert_never(phase)
    match marker:
        case Marker.INVALID | Marker.MIXED:
            return Action.HOLD
        case Marker.NONE | Marker.INDEX | Marker.DISK | Marker.OTHER | Marker.PROBE | Marker.REPLAY:
            pass
        case _:
            assert_never(marker)
    match refusal:
        case Refusal.UNREADABLE:
            return Action.HOLD
        case Refusal.ABSENT | Refusal.PRESENT:
            pass
        case _:
            assert_never(refusal)
    match budget:
        case Budget.INVALID:
            return Action.HOLD
        case Budget.AVAILABLE | Budget.SPENT:
            pass
        case _:
            assert_never(budget)
    # An inconsistent running row grants no manual effect, including release.
    if (
        phase == Phase.RUNNING
        and budget != Budget.SPENT
        and event
        in {
            Event.MANUAL_REBUILD,
            Event.MANUAL_COMPACT,
            Event.MANUAL_ARM,
            Event.MANUAL_STEP,
            Event.MANUAL_RELEASE,
        }
    ):
        return Action.HOLD

    match event:
        case Event.MANUAL_REBUILD | Event.MANUAL_COMPACT:
            return Action.MANUAL
        case Event.MANUAL_ARM:
            return Action.MANUAL_ARM if phase in {Phase.REQUIRED, Phase.RUNNING} else Action.HOLD
        case Event.MANUAL_STEP:
            return Action.MANUAL_STEP if phase == Phase.RUNNING else Action.HOLD
        case Event.MANUAL_RELEASE:
            if phase == Phase.RUNNING:
                return Action.MANUAL_RELEASE
            if phase == Phase.VERIFIED and marker in {Marker.NONE, Marker.REPLAY}:
                return Action.MANUAL_RELEASE
            return Action.HOLD
        case _:
            pass

    if refusal != Refusal.ABSENT:
        return Action.HOLD
    if event == Event.FAILURE:
        return Action.ERROR
    if phase == Phase.REFUSED:
        return Action.HOLD
    if phase == Phase.RUNNING:
        if budget != Budget.SPENT:
            return Action.HOLD
        match event:
            case Event.STEP:
                return Action.STEP
            case Event.VERIFY:
                return Action.VERIFY
            case _:
                return Action.HOLD
    if phase == Phase.REQUIRED:
        if budget != Budget.AVAILABLE:
            return Action.HOLD
        match event:
            case Event.START_NEGATIVE | Event.START_POSITIVE:
                return Action.RESUME_REQUIRED
            case Event.ARM:
                return Action.ARM
            case Event.FOLD:
                return Action.FOLD
            case _:
                return Action.HOLD

    assert phase in {Phase.IDLE, Phase.VERIFIED}
    match event:
        case Event.START_NEGATIVE | Event.START_POSITIVE:
            actionable = marker in {Marker.INDEX, Marker.PROBE} or event == Event.START_POSITIVE
            if actionable:
                if budget == Budget.SPENT:
                    return Action.REFUSE if phase == Phase.VERIFIED else Action.HOLD
                return Action.REQUIRE_INDEX if marker == Marker.INDEX else Action.REQUIRE_PROBE
            return Action.DIAGNOSE if marker in {Marker.DISK, Marker.OTHER} else Action.NORMAL
        case Event.FOLD:
            return Action.FOLD if marker not in {Marker.NONE, Marker.REPLAY} else Action.HOLD
        case Event.RETIRE:
            return Action.RETIRE if marker in {Marker.NONE, Marker.REPLAY} else Action.HOLD
        case Event.SUCCESS:
            return Action.SUCCESS if marker == Marker.NONE else Action.HOLD
        case Event.PARTIAL:
            return Action.PARTIAL
        case Event.AUTO_COMPACT:
            return Action.PRESERVE if marker in {Marker.NONE, Marker.REPLAY} else Action.HOLD
        case Event.ARM | Event.STEP | Event.VERIFY | Event.INVALID:
            return Action.HOLD
        case (
            Event.OLD_RESET
            | Event.OLD_START
            | Event.KILL
            | Event.PERSIST_FAIL
            | Event.MANUAL_REBUILD
            | Event.MANUAL_COMPACT
            | Event.MANUAL_ARM
            | Event.MANUAL_STEP
            | Event.MANUAL_RELEASE
            | Event.FAILURE
        ):
            raise AssertionError("event already dispatched")
        case _:
            assert_never(event)
