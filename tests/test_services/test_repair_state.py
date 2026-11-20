"""Literal recovery contract vectors; execution authority is tested by the owner."""

from itertools import product

import pytest
from recall.core.repair_state import Action, Admission, Budget, Event, Marker, Phase, Refusal


@pytest.mark.parametrize(
    ("state", "event", "expected"),
    [
        ("idle/none/absent/available", "start_negative", "NORMAL"),
        ("idle/none/absent/spent", "start_negative", "NORMAL"),
        ("verified/none/absent/spent", "start_negative", "NORMAL"),
        ("idle/none/absent/available", "success", "SUCCESS"),
        ("verified/none/absent/spent", "success", "SUCCESS"),
        ("idle/index/absent/available", "start_negative", "REQUIRE_INDEX"),
        ("idle/index/absent/available", "start_positive", "REQUIRE_INDEX"),
        ("idle/disk/absent/available", "start_positive", "REQUIRE_PROBE"),
        ("idle/other/absent/available", "start_positive", "REQUIRE_PROBE"),
        ("required/none/absent/available", "start_negative", "RESUME_REQUIRED"),
        ("required/none/absent/available", "arm", "ARM"),
        ("running/none/absent/spent", "step", "STEP"),
        ("running/none/absent/spent", "verify", "VERIFY"),
        ("running/none/present/spent", "manual_step", "MANUAL_STEP"),
        ("verified/replay/absent/spent", "retire", "RETIRE"),
        ("verified/none/present/spent", "manual_release", "MANUAL_RELEASE"),
        ("verified/index/absent/spent", "start_negative", "REFUSE"),
        ("verified/probe/absent/spent", "start_negative", "REFUSE"),
        ("verified/other/absent/spent", "start_positive", "REFUSE"),
        ("verified/index/present/spent", "manual_release", "HOLD"),
        ("verified/probe/absent/spent", "manual_release", "HOLD"),
        ("verified/none/absent/spent", "manual_arm", "HOLD"),
        ("refused/none/present/spent", "manual_arm", "HOLD"),
        ("required/none/present/spent", "manual_arm", "MANUAL_ARM"),
        ("idle/mixed/absent/available", "start_negative", "HOLD"),
        ("idle/index/present/available", "start_negative", "HOLD"),
        ("running/none/absent/spent", "start_negative", "HOLD"),
        ("running/none/absent/spent", "failure", "ERROR"),
        ("refused/none/absent/spent", "kill", "UNCHANGED"),
        ("invalid/invalid/unreadable/invalid", "persist_fail", "ERROR"),
    ],
)
def test_admission_contract(state: str, event: str, expected: str) -> None:
    from recall.core.repair_state import decide

    phase, marker, refusal, budget = state.split("/")
    observed = decide(
        Admission(Phase(phase), Marker(marker), Refusal(refusal), Budget(budget)), Event(event)
    )
    assert observed == Action(expected)


@pytest.mark.parametrize(
    "marker,refusal,event",
    tuple(
        product(
            Marker,
            Refusal,
            [
                Event.MANUAL_REBUILD,
                Event.MANUAL_COMPACT,
                Event.MANUAL_ARM,
                Event.MANUAL_STEP,
                Event.MANUAL_RELEASE,
            ],
        )
    ),
)
def test_inconsistent_running_budget_cannot_grant_manual_authority(
    marker: Marker, refusal: Refusal, event: Event
) -> None:
    from recall.core.repair_state import decide

    assert decide(Admission(Phase.RUNNING, marker, refusal, Budget.AVAILABLE), event) == Action.HOLD
