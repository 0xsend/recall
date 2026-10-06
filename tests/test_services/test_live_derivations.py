"""Liveness, freshness and turn state derived at read time (REQ-LIVE-001/003/005).

Every one of these is a *derivation*: recall never stores "this session is
active", it works it out from the daemon's live set, a `stat` taken now, and
the indexed tail. These tests pin the derivations as pure functions so the
rules are readable in one place, and the integration tests above them only
have to prove the inputs arrive.
"""

from __future__ import annotations

import base64
import os
from datetime import datetime, timedelta

import pytest
from recall.core.models import Message, ToolCall
from recall.core.types import Role
from recall.parsers import ClaudeCodeParser
from recall.services.live import (
    CatalogProgress,
    Liveness,
    OpenToolCall,
    TurnPhase,
    decode_cursor,
    derive_freshness,
    derive_liveness,
    derive_turn_state,
    encode_cursor,
)
from recall.services.reconciler import capture_sidecars

NOW = datetime(2026, 9, 8, 12, 0, 0)


def _message(
    idx: int,
    role: Role,
    *,
    content: str = "",
    minutes_ago: float = 0.0,
    agent_id: str | None = None,
) -> Message:
    return Message(
        id=f"m{idx}",
        session_id="s1",
        idx=idx,
        role=role,
        content=content,
        timestamp=NOW - timedelta(minutes=minutes_ago),
        agent_id=agent_id,
    )


def _tool_call(
    idx: int, name: str, *, message_id: str = "m1", agent_id: str | None = None
) -> ToolCall:
    return ToolCall(
        id=f"t{idx}",
        session_id="s1",
        message_id=message_id,
        idx=idx,
        tool_name=name,
        tool_input={"command": "pytest -q"},
        agent_id=agent_id,
    )


class TestDeriveLiveness:
    def test_a_watched_transcript_is_active(self) -> None:
        assert (
            derive_liveness(
                watched=True,
                session_ended=False,
                marked_pid_alive=None,
                last_activity_at=NOW,
                now=NOW,
                idle_window_seconds=86400.0,
            )
            is Liveness.ACTIVE
        )

    def test_a_dead_marked_pid_ends_a_watched_transcript(self) -> None:
        """A pid probed dead *now* outranks a write seen up to the idle threshold ago.

        Reversed after the U18 bug bash: a SIGKILLed agent read `active` for the
        whole 300 s live-set idle window, which is the wrong-liveness answer this
        requirement exists to forbid. Live-set membership is an inference from a
        write that may be minutes old; `kill(pid, 0)` is an observation made at
        read time about the process that claimed this session.
        """
        assert (
            derive_liveness(
                watched=True,
                session_ended=False,
                marked_pid_alive=False,
                last_activity_at=NOW,
                now=NOW,
                idle_window_seconds=86400.0,
            )
            is Liveness.ENDED
        )

    def test_a_live_marked_pid_leaves_a_watched_transcript_active(self) -> None:
        assert (
            derive_liveness(
                watched=True,
                session_ended=False,
                marked_pid_alive=True,
                last_activity_at=NOW,
                now=NOW,
                idle_window_seconds=86400.0,
            )
            is Liveness.ACTIVE
        )

    def test_a_harness_end_marker_does_not_outrank_a_watched_transcript(self) -> None:
        """A marker is a record of the past; the live set says bytes are landing now.

        This is the case the old `watched` precedence was written for, and it
        keeps that precedence: a harness that restarts into the same transcript
        is live again whatever an earlier marker said. Only the pid probe, which
        is re-observed on every read, outranks it.
        """
        assert (
            derive_liveness(
                watched=True,
                session_ended=True,
                marked_pid_alive=None,
                last_activity_at=NOW,
                now=NOW,
                idle_window_seconds=86400.0,
            )
            is Liveness.ACTIVE
        )

    def test_a_harness_end_marker_ends_an_unwatched_session(self) -> None:
        assert (
            derive_liveness(
                watched=False,
                session_ended=True,
                marked_pid_alive=None,
                last_activity_at=NOW,
                now=NOW,
                idle_window_seconds=86400.0,
            )
            is Liveness.ENDED
        )

    def test_a_marked_pid_that_is_gone_ends_the_session(self) -> None:
        assert (
            derive_liveness(
                watched=False,
                session_ended=False,
                marked_pid_alive=False,
                last_activity_at=NOW,
                now=NOW,
                idle_window_seconds=86400.0,
            )
            is Liveness.ENDED
        )

    def test_a_marked_pid_that_is_alive_does_not_end_the_session(self) -> None:
        assert (
            derive_liveness(
                watched=False,
                session_ended=False,
                marked_pid_alive=True,
                last_activity_at=NOW - timedelta(hours=1),
                now=NOW,
                idle_window_seconds=86400.0,
            )
            is Liveness.IDLE
        )

    def test_recent_activity_without_a_watcher_is_idle(self) -> None:
        assert (
            derive_liveness(
                watched=False,
                session_ended=False,
                marked_pid_alive=None,
                last_activity_at=NOW - timedelta(hours=3),
                now=NOW,
                idle_window_seconds=86400.0,
            )
            is Liveness.IDLE
        )

    def test_activity_older_than_the_idle_window_is_unknown(self) -> None:
        """Not `ended`: nothing observed an end, so claiming one would be a guess."""
        assert (
            derive_liveness(
                watched=False,
                session_ended=False,
                marked_pid_alive=None,
                last_activity_at=NOW - timedelta(hours=25),
                now=NOW,
                idle_window_seconds=86400.0,
            )
            is Liveness.UNKNOWN
        )

    def test_a_session_with_no_activity_timestamp_is_unknown(self) -> None:
        assert (
            derive_liveness(
                watched=False,
                session_ended=False,
                marked_pid_alive=None,
                last_activity_at=None,
                now=NOW,
                idle_window_seconds=86400.0,
            )
            is Liveness.UNKNOWN
        )


class TestDeriveFreshness:
    def test_matching_byte_counts_without_catalog_validation_are_not_current(
        self, tmp_path
    ) -> None:
        path = tmp_path / "t.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        os.utime(path, (1_700_000_000.0, 1_700_000_000.0))

        freshness = derive_freshness(
            str(path), indexed_mtime=1_700_000_000.0, indexed_size=path.stat().st_size
        )

        assert freshness.current is False
        assert freshness.validated is False
        assert freshness.limitations == ("catalog_progress_unavailable",)
        assert freshness.lag_bytes == 0
        assert freshness.lag_seconds == 0.0

    def test_validated_catalog_signature_and_complete_boundary_are_current(self, tmp_path) -> None:
        path = tmp_path / "t.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        stat = path.stat()
        freshness = derive_freshness(
            str(path),
            indexed_mtime=stat.st_mtime,
            indexed_size=stat.st_size,
            catalog=CatalogProgress(
                desired_generation=4,
                committed_generation=4,
                committed_offset=stat.st_size,
                content_epoch=2,
                signature_dev=stat.st_dev,
                signature_inode=stat.st_ino,
                signature_ctime_ns=stat.st_ctime_ns,
                signature_mtime_ns=stat.st_mtime_ns,
                signature_size=stat.st_size,
                source="claude_code",
                sidecar_signature=capture_sidecars(ClaudeCodeParser(), path)[1],
            ),
        )

        assert freshness.current is True
        assert freshness.validated is True
        assert freshness.content_epoch == 2

    def test_same_size_file_with_a_different_catalog_signature_is_not_current(
        self, tmp_path
    ) -> None:
        path = tmp_path / "t.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        stat = path.stat()
        freshness = derive_freshness(
            str(path),
            indexed_mtime=stat.st_mtime,
            indexed_size=stat.st_size,
            catalog=CatalogProgress(
                desired_generation=1,
                committed_generation=1,
                committed_offset=stat.st_size,
                content_epoch=0,
                signature_dev=stat.st_dev,
                signature_inode=stat.st_ino,
                signature_ctime_ns=stat.st_ctime_ns - 1,
                signature_mtime_ns=stat.st_mtime_ns,
                signature_size=stat.st_size,
                source="claude_code",
                sidecar_signature=capture_sidecars(ClaudeCodeParser(), path)[1],
            ),
        )

        assert freshness.lag_bytes == 0
        assert freshness.current is False
        assert freshness.limitations == ("catalog_signature_mismatch",)

    def test_bytes_appended_after_indexing_show_as_lag(self, tmp_path) -> None:
        path = tmp_path / "t.jsonl"
        path.write_text("{}\n" * 4, encoding="utf-8")
        os.utime(path, (1_700_000_060.0, 1_700_000_060.0))

        freshness = derive_freshness(str(path), indexed_mtime=1_700_000_000.0, indexed_size=3)

        assert freshness.current is False
        assert freshness.lag_bytes == 9
        assert freshness.lag_seconds == 60.0
        assert freshness.file_size == 12

    def test_a_never_indexed_transcript_reports_the_file_and_no_lag(self, tmp_path) -> None:
        """Lag against nothing is not zero — zero would read as 'current'."""
        path = tmp_path / "t.jsonl"
        path.write_text("{}\n", encoding="utf-8")

        freshness = derive_freshness(str(path), indexed_mtime=None, indexed_size=None)

        assert freshness.current is False
        assert freshness.lag_bytes is None
        assert freshness.lag_seconds is None
        assert freshness.file_size == 3
        assert freshness.limitations == ("not_yet_indexed",)

    def test_zero_committed_generation_is_not_yet_indexed(self, tmp_path) -> None:
        """A discovered source with no commit is not current, even if sizes match."""
        path = tmp_path / "t.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        stat = path.stat()
        freshness = derive_freshness(
            str(path),
            indexed_mtime=stat.st_mtime,
            indexed_size=stat.st_size,
            catalog=CatalogProgress(
                desired_generation=1,
                committed_generation=0,
                committed_offset=stat.st_size,
                content_epoch=0,
                signature_dev=stat.st_dev,
                signature_inode=stat.st_ino,
                signature_ctime_ns=stat.st_ctime_ns,
                signature_mtime_ns=stat.st_mtime_ns,
                signature_size=stat.st_size,
                source="claude_code",
                sidecar_signature=capture_sidecars(ClaudeCodeParser(), path)[1],
            ),
        )

        assert freshness.current is False
        assert freshness.generation == 0
        assert "not_yet_indexed" in freshness.limitations

    def test_a_transcript_that_is_gone_reports_no_file_and_is_not_current(self, tmp_path) -> None:
        freshness = derive_freshness(
            str(tmp_path / "gone.jsonl"), indexed_mtime=1_700_000_000.0, indexed_size=3
        )

        assert freshness.current is False
        assert freshness.file_mtime is None
        assert freshness.file_size is None
        assert freshness.indexed_size == 3

    def test_a_rewritten_shorter_transcript_reports_negative_lag_not_current(
        self, tmp_path
    ) -> None:
        """The index is ahead of the file; reporting 0 would claim it is current."""
        path = tmp_path / "t.jsonl"
        path.write_text("{}\n", encoding="utf-8")

        freshness = derive_freshness(str(path), indexed_mtime=1_700_000_000.0, indexed_size=999)

        assert freshness.lag_bytes == 3 - 999
        assert freshness.current is False


class TestDeriveTurnState:
    def test_an_open_tool_call_is_working_and_names_the_tool(self) -> None:
        messages = [
            _message(0, Role.USER, content="run the tests", minutes_ago=2),
            _message(1, Role.ASSISTANT, content="on it", minutes_ago=1),
        ]
        open_call = OpenToolCall(
            tool_call=_tool_call(0, "Bash", message_id="m1"),
            message_idx=1,
            started_at=NOW - timedelta(minutes=1),
        )

        turn = derive_turn_state(
            messages,
            open_tool_calls=[open_call],
            last_stop_reason="tool_use",
            last_stop_ends_turn=False,
            session_ended=False,
        )

        assert turn.state is TurnPhase.WORKING
        assert turn.running_tool is not None
        assert turn.running_tool.name == "Bash"
        assert turn.running_tool.started_at == NOW - timedelta(minutes=1)
        assert turn.stop_reason == "tool_use"

    def test_an_end_of_turn_stop_with_no_later_user_record_awaits_input(self) -> None:
        messages = [
            _message(0, Role.USER, content="what changed?", minutes_ago=3),
            _message(1, Role.ASSISTANT, content="three files", minutes_ago=1),
        ]

        turn = derive_turn_state(
            messages,
            open_tool_calls=[],
            last_stop_reason="end_turn",
            last_stop_ends_turn=True,
            session_ended=False,
        )

        assert turn.state is TurnPhase.AWAITING_INPUT
        assert turn.last_assistant_text == "three files"
        assert turn.last_user_text == "what changed?"
        assert turn.last_assistant_at == NOW - timedelta(minutes=1)
        assert turn.last_user_at == NOW - timedelta(minutes=3)

    def test_a_user_record_after_the_end_of_turn_stop_is_not_awaiting_input(self) -> None:
        """The human already replied; the agent owes the next move, not the human."""
        messages = [
            _message(0, Role.USER, content="what changed?", minutes_ago=3),
            _message(1, Role.ASSISTANT, content="three files", minutes_ago=2),
            _message(2, Role.USER, content="and now?", minutes_ago=1),
        ]

        turn = derive_turn_state(
            messages,
            open_tool_calls=[],
            last_stop_reason="end_turn",
            last_stop_ends_turn=True,
            session_ended=False,
        )

        assert turn.state is TurnPhase.UNKNOWN

    def test_subagents_mid_tool_outrank_the_generic_working_state(self) -> None:
        """A subagent waiting on its own tool is the specific answer for `working`."""
        messages = [
            _message(0, Role.USER, content="fan out", minutes_ago=5),
            _message(1, Role.ASSISTANT, content="dispatching", minutes_ago=4),
            _message(2, Role.ASSISTANT, content="searching", minutes_ago=1, agent_id="a1"),
            _message(3, Role.ASSISTANT, content="reading", minutes_ago=1, agent_id="a2"),
        ]
        open_calls = [
            OpenToolCall(
                tool_call=_tool_call(0, "Grep", message_id="m2", agent_id="a1"),
                message_idx=2,
                started_at=NOW - timedelta(minutes=1),
            ),
            OpenToolCall(
                tool_call=_tool_call(0, "Read", message_id="m3", agent_id="a2"),
                message_idx=3,
                started_at=NOW - timedelta(minutes=1),
            ),
        ]

        turn = derive_turn_state(
            messages,
            open_tool_calls=open_calls,
            last_stop_reason="tool_use",
            last_stop_ends_turn=False,
            session_ended=False,
        )

        assert turn.state is TurnPhase.SUBAGENTS_RUNNING
        assert turn.subagents_active == 2
        assert turn.running_tool is not None
        assert turn.running_tool.name == "Read"

    def test_subagent_messages_alone_do_not_mean_a_subagent_is_running(self) -> None:
        """A finished subagent's messages stay in the transcript forever."""
        messages = [
            _message(0, Role.USER, content="fan out", minutes_ago=5),
            _message(1, Role.ASSISTANT, content="dispatching", minutes_ago=4),
            _message(2, Role.ASSISTANT, content="searching", minutes_ago=3, agent_id="a1"),
        ]

        turn = derive_turn_state(
            messages,
            open_tool_calls=[],
            last_stop_reason="end_turn",
            last_stop_ends_turn=True,
            session_ended=False,
        )

        assert turn.state is TurnPhase.AWAITING_INPUT
        assert turn.subagents_active == 0

    def test_the_parents_open_agent_call_alone_is_working_not_subagents_running(self) -> None:
        """recall saw the dispatch but no subagent activity; claiming one would be a guess."""
        messages = [
            _message(0, Role.USER, content="fan out", minutes_ago=5),
            _message(1, Role.ASSISTANT, content="dispatching", minutes_ago=4),
        ]
        open_call = OpenToolCall(
            tool_call=_tool_call(0, "Agent", message_id="m1"),
            message_idx=1,
            started_at=NOW - timedelta(minutes=4),
        )

        turn = derive_turn_state(
            messages,
            open_tool_calls=[open_call],
            last_stop_reason="tool_use",
            last_stop_ends_turn=False,
            session_ended=False,
        )

        assert turn.state is TurnPhase.WORKING
        assert turn.subagents_active == 0

    def test_a_session_end_marker_wins_over_every_other_signal(self) -> None:
        messages = [_message(0, Role.USER, content="bye", minutes_ago=1)]
        open_call = OpenToolCall(
            tool_call=_tool_call(0, "Bash", message_id="m0"), message_idx=0, started_at=NOW
        )

        turn = derive_turn_state(
            messages,
            open_tool_calls=[open_call],
            last_stop_reason="tool_use",
            last_stop_ends_turn=False,
            session_ended=True,
        )

        assert turn.state is TurnPhase.ENDED

    def test_a_mid_turn_stop_is_working_after_the_tool_result_lands(self) -> None:
        """The harness said the turn is still open; paired results do not close it.

        Pi Agent writes toolUse and its toolResult in the same burst, so the
        open-call window is milliseconds. Without this, a live tool loop reads
        `unknown` for almost the entire turn.
        """
        messages = [
            _message(0, Role.USER, content="inspect the tree", minutes_ago=2),
            _message(1, Role.ASSISTANT, content="checking", minutes_ago=1),
        ]

        turn = derive_turn_state(
            messages,
            open_tool_calls=[],
            last_stop_reason="toolUse",
            last_stop_ends_turn=False,
            session_ended=False,
        )

        assert turn.state is TurnPhase.WORKING
        assert turn.running_tool is None
        assert turn.stop_reason == "toolUse"

    def test_a_harness_that_emits_no_markers_resolves_unknown(self) -> None:
        messages = [
            _message(0, Role.USER, content="hi", minutes_ago=2),
            _message(1, Role.ASSISTANT, content="hello", minutes_ago=1),
        ]

        turn = derive_turn_state(
            messages,
            open_tool_calls=[],
            last_stop_reason=None,
            last_stop_ends_turn=False,
            session_ended=False,
        )

        assert turn.state is TurnPhase.UNKNOWN
        assert turn.stop_reason is None
        assert turn.running_tool is None

    def test_an_empty_tail_resolves_unknown_without_raising(self) -> None:
        turn = derive_turn_state(
            [],
            open_tool_calls=[],
            last_stop_reason=None,
            last_stop_ends_turn=False,
            session_ended=False,
        )

        assert turn.state is TurnPhase.UNKNOWN
        assert turn.last_user_at is None
        assert turn.last_assistant_at is None

    def test_message_text_is_truncated_to_the_configured_budget(self) -> None:
        messages = [
            _message(0, Role.USER, content="u" * 900, minutes_ago=2),
            _message(1, Role.ASSISTANT, content="a" * 900, minutes_ago=1),
        ]

        turn = derive_turn_state(
            messages,
            open_tool_calls=[],
            last_stop_reason=None,
            last_stop_ends_turn=False,
            session_ended=False,
            text_budget=400,
        )

        assert turn.last_user_text is not None
        assert turn.last_assistant_text is not None
        assert len(turn.last_user_text) == 400
        assert len(turn.last_assistant_text) == 400

    def test_the_newest_open_tool_call_is_the_running_one(self) -> None:
        messages = [_message(0, Role.ASSISTANT, content="working", minutes_ago=1)]
        older = OpenToolCall(
            tool_call=_tool_call(0, "Read", message_id="m0"),
            message_idx=0,
            started_at=NOW - timedelta(minutes=9),
        )
        newer = OpenToolCall(
            tool_call=_tool_call(1, "Bash", message_id="m0"),
            message_idx=0,
            started_at=NOW - timedelta(minutes=1),
        )

        turn = derive_turn_state(
            messages,
            open_tool_calls=[older, newer],
            last_stop_reason="tool_use",
            last_stop_ends_turn=False,
            session_ended=False,
        )

        assert turn.running_tool is not None
        assert turn.running_tool.name == "Bash"


def test_turn_state_rejects_a_non_positive_text_budget() -> None:
    with pytest.raises(ValueError, match="text_budget"):
        derive_turn_state(
            [],
            open_tool_calls=[],
            last_stop_reason=None,
            last_stop_ends_turn=False,
            session_ended=False,
            text_budget=0,
        )


class TestCursorCodec:
    def test_a_cursor_round_trips_its_session_and_index(self) -> None:
        assert decode_cursor(encode_cursor("abc123", 41)) == ("abc123", 41)

    def test_a_cursor_carries_the_content_epoch(self) -> None:
        cursor = decode_cursor(encode_cursor("abc123", 41, content_epoch=3))
        assert cursor.content_epoch == 3

    def test_legacy_cursor_is_distinct_from_epoch_zero(self) -> None:
        legacy = base64.urlsafe_b64encode(b"v1:abc123:41").decode().rstrip("=")
        assert decode_cursor(legacy).content_epoch is None

    def test_a_cursor_for_an_empty_session_round_trips(self) -> None:
        assert decode_cursor(encode_cursor("abc123", -1)) == ("abc123", -1)

    def test_a_session_id_containing_the_separator_still_round_trips(self) -> None:
        assert decode_cursor(encode_cursor("a:b:c", 7)) == ("a:b:c", 7)

    def test_a_cursor_is_opaque_rather_than_the_ids_in_plain_text(self) -> None:
        """Callers must not learn to build one by hand; only recall writes cursors."""
        assert "abc123" not in encode_cursor("abc123", 41)

    @pytest.mark.parametrize(
        "cursor",
        ["", "not-base64!!", "YWJj", "djI6YWJjOjE"],
        ids=["empty", "not-base64", "wrong-payload", "wrong-version"],
    )
    def test_anything_recall_did_not_write_is_a_validation_error(self, cursor: str) -> None:
        with pytest.raises(ValueError, match="malformed cursor"):
            decode_cursor(cursor)
