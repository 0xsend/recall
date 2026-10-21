from __future__ import annotations

from dataclasses import dataclass
from threading import Event

import pytest
from recall.core.read_cancellation import (
    ReadCancelled,
    check_read_cancelled,
    read_cancellation,
    read_cancelled,
)
from recall.core.rpc_types import INVALID_PARAMS, RpcError


@dataclass(frozen=True)
class FrozenError(Exception):
    code: int


@pytest.mark.parametrize(
    ("error", "code"), [(RpcError(INVALID_PARAMS, "invalid cursor"), -32602), (FrozenError(42), 42)]
)
def test_body_exception_propagates_unchanged(error: RpcError | FrozenError, code: int) -> None:
    event = Event()
    with pytest.raises(type(error)) as caught, read_cancellation(event):
        event.set()
        raise error

    assert caught.value is error
    assert caught.value.code == code
    assert not read_cancelled()


def test_nested_scope_restores_outer_cancellation() -> None:
    outer = Event()
    with read_cancellation(outer):
        outer.set()
        with read_cancellation(Event()):
            assert not read_cancelled()
        with pytest.raises(ReadCancelled, match="read cancelled"):
            check_read_cancelled()
    assert not read_cancelled()


def test_cancelled_entry_restores_outer_scope() -> None:
    cancelled = Event()
    cancelled.set()
    with read_cancellation(Event()):
        with pytest.raises(ReadCancelled, match="read cancelled"), read_cancellation(cancelled):
            pytest.fail("a cancelled read must not enter its body")
        assert not read_cancelled()
    assert not read_cancelled()
