"""Windowed reads at the end of a session: `--tail` and `--after` (REQ-LIVE-004).

A monitor loop watching a running agent reads the same session over and over.
It must be able to say "only what is new" and get back a cursor it can hand to
the next call, and an empty delta has to be an ordinary success — the common
case is that nothing happened since the last look.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from lane_harness import UNWATCHED_SESSION, WATCHED_SESSION, Lane, build_lane
from recall.core.rpc_types import INVALID_PARAMS, RpcError
from recall.services.live import decode_cursor, encode_cursor


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lane:
    return build_lane(tmp_path, monkeypatch)


def _run(coro):
    return asyncio.run(coro)


def _show(lane: Lane, **params: object) -> dict:
    return _run(lane.server._handle_show(params, None))


def _idxs(payload: dict) -> list[int]:
    return [message["idx"] for message in payload["messages"]]


class TestTail:
    def test_tail_returns_the_last_n_messages_in_reading_order(self, lane: Lane) -> None:
        """The window is at the end, but it still reads oldest-first."""
        payload = _show(lane, session_id=UNWATCHED_SESSION, tail=2)

        assert _idxs(payload) == [2, 3]

    def test_tail_larger_than_the_session_returns_all_of_it(self, lane: Lane) -> None:
        payload = _show(lane, session_id=UNWATCHED_SESSION, tail=99)

        assert _idxs(payload) == [0, 1, 2, 3]

    def test_a_non_positive_tail_is_rejected(self, lane: Lane) -> None:
        with pytest.raises(RpcError) as err:
            _show(lane, session_id=UNWATCHED_SESSION, tail=0)

        assert err.value.code == INVALID_PARAMS

    def test_tail_and_message_limit_cannot_both_bound_the_read(self, lane: Lane) -> None:
        """One is head-anchored and one is tail-anchored; together they mean nothing."""
        with pytest.raises(RpcError) as err:
            _show(lane, session_id=UNWATCHED_SESSION, tail=2, message_limit=2)

        assert err.value.code == INVALID_PARAMS


class TestCursor:
    def test_a_read_carries_a_cursor_naming_its_newest_message(self, lane: Lane) -> None:
        payload = _show(lane, session_id=UNWATCHED_SESSION)

        assert decode_cursor(payload["cursor"]) == (payload["id"], 3)

    def test_after_returns_only_what_landed_since(self, lane: Lane) -> None:
        first = _show(lane, session_id=WATCHED_SESSION)
        lane.append(lane.watched, "one more thing", uuid="mid-u2")

        delta = _show(lane, session_id=WATCHED_SESSION, after=first["cursor"], fresh=True)

        assert _idxs(delta) == [2]
        assert delta["messages"][0]["content"] == "one more thing"

    def test_asking_twice_with_the_same_cursor_is_an_empty_success(self, lane: Lane) -> None:
        """Nothing happened is the common answer, not an error."""
        first = _show(lane, session_id=WATCHED_SESSION)

        again = _show(lane, session_id=WATCHED_SESSION, after=first["cursor"])

        assert again["messages"] == []
        assert again["cursor"] == first["cursor"]

    def test_the_cursor_advances_across_a_polling_loop(self, lane: Lane) -> None:
        """Two appends read one at a time, the way a follower consumes them."""
        cursor = _show(lane, session_id=WATCHED_SESSION)["cursor"]

        lane.append(lane.watched, "first", uuid="mid-u2")
        step = _show(lane, session_id=WATCHED_SESSION, after=cursor, fresh=True)
        assert _idxs(step) == [2]

        lane.append(lane.watched, "second", uuid="mid-u3")
        step = _show(lane, session_id=WATCHED_SESSION, after=step["cursor"], fresh=True)
        assert _idxs(step) == [3]
        assert step["messages"][0]["content"] == "second"

    def test_tail_bounds_the_delta_a_cursor_opens(self, lane: Lane) -> None:
        """A follower that fell far behind still gets a bounded read."""
        beginning = encode_cursor(_show(lane, session_id=UNWATCHED_SESSION)["id"], -1)

        payload = _show(lane, session_id=UNWATCHED_SESSION, after=beginning, tail=2)

        assert _idxs(payload) == [2, 3]

    def test_a_cursor_from_another_session_is_rejected(self, lane: Lane) -> None:
        """Silently reading someone else's offset would hand back a plausible lie."""
        foreign = _show(lane, session_id=UNWATCHED_SESSION)["cursor"]

        with pytest.raises(RpcError) as err:
            _show(lane, session_id=WATCHED_SESSION, after=foreign)

        assert err.value.code == INVALID_PARAMS
        assert "another session" in err.value.message

    def test_a_cursor_recall_did_not_write_is_rejected(self, lane: Lane) -> None:
        with pytest.raises(RpcError) as err:
            _show(lane, session_id=WATCHED_SESSION, after="not-a-cursor")

        assert err.value.code == INVALID_PARAMS

    def test_after_and_message_limit_cannot_both_bound_the_read(self, lane: Lane) -> None:
        cursor = _show(lane, session_id=WATCHED_SESSION)["cursor"]

        with pytest.raises(RpcError) as err:
            _show(lane, session_id=WATCHED_SESSION, after=cursor, message_limit=2)

        assert err.value.code == INVALID_PARAMS

    def test_a_windowed_read_still_reports_freshness(self, lane: Lane) -> None:
        """REQ-LIVE-002 does not stop applying because the read was bounded."""
        appended = lane.append(lane.watched, "one more thing", uuid="mid-u2")

        payload = _show(lane, session_id=WATCHED_SESSION, tail=1)

        assert payload["freshness"].lag_bytes == appended
        assert payload["freshness"].current is False
