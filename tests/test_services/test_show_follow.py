"""`show --follow`: stream a session's messages as they land (REQ-LIVE-004/011).

A follower is the one caller that cannot poll — it holds a connection open and
expects the daemon to push. That makes two things load-bearing here that
`--after` does not have to worry about: the stream reads from its OWN cursor
rather than trusting the event it was woken by, and every exit path releases
the subscription, because a leaked one is a slot no other follower can have.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import pytest
from lane_harness import WATCHED_SESSION, Lane, build_lane
from recall.core.rpc_types import INVALID_PARAMS, RpcError
from recall.services.live import decode_cursor, encode_cursor
from recall.services.live_events import SessionIndexed
from recall.services.rpc_server import ClientConnection, ClientDisconnected


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lane:
    return build_lane(tmp_path, monkeypatch)


class RecordingWriter:
    """A `StreamWriter` that keeps the bytes instead of sending them."""

    def __init__(self, *, close_after: int | None = None) -> None:
        self.lines: list[str] = []
        self._close_after = close_after

    def is_closing(self) -> bool:
        return self._close_after is not None and len(self.lines) >= self._close_after

    def write(self, data: bytes) -> None:
        self.lines.append(data.decode("utf-8").rstrip("\n"))

    async def drain(self) -> None:
        return None


class RecordingClient(ClientConnection):
    """A real `ClientConnection` over a writer that records.

    Deliberately the production class rather than a stand-in: the frames it
    writes are the streaming handler's observable output, and only the real
    `send_notification` proves the payload actually serialises to JSON.
    """

    def __init__(self, *, disconnect_after: int | None = None) -> None:
        self._recorder = RecordingWriter(close_after=disconnect_after)
        super().__init__(cast(asyncio.StreamWriter, self._recorder))

    @property
    def frames(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self._recorder.lines]

    @property
    def deltas(self) -> list[dict[str, Any]]:
        return [f["params"] for f in self.frames if f["method"] == "live.delta"]


def _follow(lane: Lane, client: ClientConnection, **params: object) -> Any:
    return lane.server._handle_show_follow(params, client)


class TestFollowStream:
    def test_a_follower_that_starts_behind_is_caught_up_before_it_waits(self, lane: Lane) -> None:
        """Subscribing is not a reason to miss what already landed."""

        async def scenario() -> RecordingClient:
            start = (await lane.server._handle_show({"session_id": WATCHED_SESSION}, None))[
                "cursor"
            ]
            lane.append(lane.watched, "landed before the subscribe", uuid="mid-u2")
            await lane.server._handle_show({"session_id": WATCHED_SESSION, "fresh": True}, None)

            client = RecordingClient()
            await _follow(lane, client, session_id=WATCHED_SESSION, after=start, timeout=0.05)
            return client

        client = asyncio.run(scenario())

        assert len(client.deltas) == 1
        assert [m["content"] for m in client.deltas[0]["messages"]] == [
            "landed before the subscribe"
        ]

    def test_a_message_that_lands_while_following_is_streamed(self, lane: Lane) -> None:
        async def scenario() -> RecordingClient:
            client = RecordingClient()
            follow = asyncio.ensure_future(
                _follow(lane, client, session_id=WATCHED_SESSION, timeout=5.0)
            )
            await asyncio.sleep(0.05)

            lane.append(lane.watched, "arrived mid-stream", uuid="mid-u2")
            await lane.server._handle_show({"session_id": WATCHED_SESSION, "fresh": True}, None)
            await asyncio.sleep(0.05)
            follow.cancel()
            with suppress(asyncio.CancelledError):
                await follow
            return client

        client = asyncio.run(scenario())

        assert [m["content"] for delta in client.deltas for m in delta["messages"]] == [
            "arrived mid-stream"
        ]

    def test_two_messages_landing_together_arrive_in_one_delta(self, lane: Lane) -> None:
        """The slot is depth one, so the cursor -- not the event -- decides the window.

        Two writes indexed in one pass wake the follower once. A follower that
        treated the event's `high_water_idx` as its read bound would deliver one
        message and silently drop the other.
        """

        async def scenario() -> RecordingClient:
            client = RecordingClient()
            follow = asyncio.ensure_future(
                _follow(lane, client, session_id=WATCHED_SESSION, timeout=5.0)
            )
            await asyncio.sleep(0.05)

            lane.append(lane.watched, "first", uuid="mid-u2")
            lane.append(lane.watched, "second", uuid="mid-u3")
            await lane.server._handle_show({"session_id": WATCHED_SESSION, "fresh": True}, None)
            await asyncio.sleep(0.05)
            follow.cancel()
            with suppress(asyncio.CancelledError):
                await follow
            return client

        client = asyncio.run(scenario())

        assert len(client.deltas) == 1
        assert [m["content"] for m in client.deltas[0]["messages"]] == ["first", "second"]

    def test_a_stale_high_water_hint_does_not_bound_the_read(self, lane: Lane) -> None:
        """The event is a hint; two publishers can deliver a lower idx after a higher one."""

        async def scenario() -> RecordingClient:
            client = RecordingClient()
            follow = asyncio.ensure_future(
                _follow(lane, client, session_id=WATCHED_SESSION, timeout=5.0)
            )
            await asyncio.sleep(0.05)

            lane.append(lane.watched, "first", uuid="mid-u2")
            lane.append(lane.watched, "second", uuid="mid-u3")
            await lane.server._handle_show({"session_id": WATCHED_SESSION, "fresh": True}, None)
            # A hint naming an idx behind what is actually readable.
            lane.server._session_indexed.publish(
                SessionIndexed(path=str(lane.watched), session_id=None, high_water_idx=0)
            )
            await asyncio.sleep(0.05)
            follow.cancel()
            with suppress(asyncio.CancelledError):
                await follow
            return client

        client = asyncio.run(scenario())

        streamed = [m["content"] for delta in client.deltas for m in delta["messages"]]
        assert streamed == ["first", "second"]

    def test_the_cursor_advances_across_deltas(self, lane: Lane) -> None:
        async def scenario() -> tuple[RecordingClient, dict[str, Any]]:
            client = RecordingClient()
            lane.append(lane.watched, "one more", uuid="mid-u2")
            await lane.server._handle_show({"session_id": WATCHED_SESSION, "fresh": True}, None)
            start = decode_cursor(
                (await lane.server._handle_show({"session_id": WATCHED_SESSION}, None))["cursor"]
            )
            closed = await _follow(
                lane,
                client,
                session_id=WATCHED_SESSION,
                after=encode_cursor(start[0], 1),
                timeout=0.05,
            )
            return client, closed

        client, closed = asyncio.run(scenario())

        assert decode_cursor(client.deltas[0]["cursor"])[1] == 2
        assert closed["cursor"] == client.deltas[0]["cursor"]


class TestFollowClose:
    def test_a_quiet_stream_closes_on_its_deadline(self, lane: Lane) -> None:
        async def scenario() -> dict[str, Any]:
            client = RecordingClient()
            return await _follow(lane, client, session_id=WATCHED_SESSION, timeout=0.05)

        closed = asyncio.run(scenario())

        assert closed["event"] == "closed"
        assert closed["reason"] == "timeout"

    def test_a_quiet_stream_emits_no_deltas(self, lane: Lane) -> None:
        """Nothing happened is silence, not an empty frame."""

        async def scenario() -> RecordingClient:
            client = RecordingClient()
            await _follow(lane, client, session_id=WATCHED_SESSION, timeout=0.05)
            return client

        assert asyncio.run(scenario()).deltas == []

    def test_a_stopping_daemon_says_so_rather_than_timing_out(self, lane: Lane) -> None:
        async def scenario() -> dict[str, Any]:
            client = RecordingClient()
            lane.server._shutdown_event.set()
            return await _follow(lane, client, session_id=WATCHED_SESSION, timeout=30.0)

        closed = asyncio.run(scenario())

        assert closed["reason"] == "daemon_stopped"

    def test_an_unwatched_daemon_says_nobody_is_looking(self, lane: Lane) -> None:
        """Poll follow reports accelerator state independently of committed updates."""

        async def scenario() -> dict[str, Any]:
            client = RecordingClient()
            return await _follow(lane, client, session_id=WATCHED_SESSION, timeout=0.05)

        assert asyncio.run(scenario())["watching"] is False


class TestFollowSubscription:
    def test_a_finished_stream_leaves_no_subscription(self, lane: Lane) -> None:
        async def scenario() -> int:
            client = RecordingClient()
            await _follow(lane, client, session_id=WATCHED_SESSION, timeout=0.05)
            return lane.server._session_indexed.total_subscriptions

        assert asyncio.run(scenario()) == 0

    def test_a_disconnected_client_leaves_no_subscription(self, lane: Lane) -> None:
        """The slot a vanished follower held has to come back."""

        async def scenario() -> int:
            client = RecordingClient(disconnect_after=0)
            lane.append(lane.watched, "one more", uuid="mid-u2")
            await lane.server._handle_show({"session_id": WATCHED_SESSION, "fresh": True}, None)
            start = encode_cursor(
                (await lane.server._handle_show({"session_id": WATCHED_SESSION}, None))["id"], 1
            )
            with pytest.raises(ClientDisconnected):
                await _follow(lane, client, session_id=WATCHED_SESSION, after=start, timeout=5.0)
            return lane.server._session_indexed.total_subscriptions

        assert asyncio.run(scenario()) == 0

    def test_a_cancelled_stream_leaves_no_subscription(self, lane: Lane) -> None:
        async def scenario() -> int:
            client = RecordingClient()
            follow = asyncio.ensure_future(
                _follow(lane, client, session_id=WATCHED_SESSION, timeout=30.0)
            )
            await asyncio.sleep(0.05)
            follow.cancel()
            with suppress(asyncio.CancelledError):
                await follow
            return lane.server._session_indexed.total_subscriptions

        assert asyncio.run(scenario()) == 0


class TestFollowValidation:
    def test_a_cursor_from_another_session_is_rejected_before_subscribing(self, lane: Lane) -> None:
        async def scenario() -> None:
            client = RecordingClient()
            foreign = encode_cursor("0" * 32, 3)
            with pytest.raises(RpcError) as err:
                await _follow(lane, client, session_id=WATCHED_SESSION, after=foreign, timeout=5.0)
            assert err.value.code == INVALID_PARAMS
            assert lane.server._session_indexed.total_subscriptions == 0

        asyncio.run(scenario())

    def test_a_non_positive_timeout_is_rejected(self, lane: Lane) -> None:
        async def scenario() -> None:
            client = RecordingClient()
            with pytest.raises(RpcError) as err:
                await _follow(lane, client, session_id=WATCHED_SESSION, timeout=0)
            assert err.value.code == INVALID_PARAMS

        asyncio.run(scenario())

    def test_following_without_a_client_is_refused(self, lane: Lane) -> None:
        """There is nowhere to stream to; answering as if there were would hang."""

        async def scenario() -> None:
            with pytest.raises(RpcError) as err:
                await lane.server._handle_show_follow(
                    {"session_id": WATCHED_SESSION, "timeout": 5.0}, None
                )
            assert err.value.code == INVALID_PARAMS

        asyncio.run(scenario())


def test_a_daemon_stopping_mid_stream_closes_promptly(lane: Lane) -> None:
    """A follower must not hold shutdown open for the rest of its deadline."""

    async def scenario() -> tuple[dict[str, Any], float]:
        client = RecordingClient()
        loop = asyncio.get_running_loop()
        started = loop.time()
        follow = asyncio.ensure_future(
            lane.server._handle_show_follow(
                {"session_id": WATCHED_SESSION, "timeout": 30.0}, client
            )
        )
        await asyncio.sleep(0.05)
        lane.server._shutdown_event.set()
        closed = await asyncio.wait_for(follow, timeout=2.0)
        return closed, loop.time() - started

    closed, elapsed = asyncio.run(scenario())

    assert closed["reason"] == "daemon_stopped"
    assert elapsed < 2.0


def test_daemon_status_reports_how_many_followers_are_attached(lane: Lane) -> None:
    """A leaked subscription is a slot no other follower can have (REQ-LIVE-011).

    The unit tests prove each exit path releases it; this is the surface an
    operator can actually check on a running daemon.
    """

    async def scenario() -> tuple[int, int]:
        client = RecordingClient()
        follow = asyncio.ensure_future(
            _follow(lane, client, session_id=WATCHED_SESSION, timeout=30.0)
        )
        await asyncio.sleep(0.05)
        during = (await lane.server._handle_daemon_status({}, None))["follow_subscriptions"]
        follow.cancel()
        with suppress(asyncio.CancelledError):
            await follow
        after = (await lane.server._handle_daemon_status({}, None))["follow_subscriptions"]
        return during, after

    during, after = asyncio.run(scenario())

    assert during == 1
    assert after == 0
