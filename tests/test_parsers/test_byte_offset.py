"""Tests for byte-offset incremental parsing (REQ-INDEX-010 through REQ-INDEX-012)."""

from __future__ import annotations

import json
from pathlib import Path

from recall.parsers.claude_code import ClaudeCodeParser
from recall.parsers.codex import CodexParser
from recall.parsers.kimi_code import KimiCodeParser
from recall.parsers.pi_agent import PiAgentParser


class TestClaudeCodeByteOffset:
    def test_offset_parse_returns_only_new_messages(self, tmp_path: Path) -> None:
        """Append new content after initial parse; incremental parse sees only new lines."""
        session_file = tmp_path / "session.jsonl"

        # Write initial content
        initial_lines = [
            json.dumps({"role": "user", "content": [{"type": "text", "text": "hello"}]}),
            json.dumps({"role": "assistant", "content": [{"type": "text", "text": "hi there"}]}),
        ]
        session_file.write_text("\n".join(initial_lines) + "\n", encoding="utf-8")

        parser = ClaudeCodeParser()

        # Full parse
        full_result = parser.parse(session_file)
        assert full_result.is_full_parse is True
        assert full_result.session.message_count == 2
        first_offset = full_result.next_byte_offset

        # Append new content
        with session_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"role": "user", "content": [{"type": "text", "text": "followup"}]})
                + "\n"
            )

        # Incremental parse from stored offset
        incr_result = parser.parse(session_file, offset=first_offset)
        assert incr_result.is_full_parse is False
        assert incr_result.session.message_count == 1
        assert incr_result.session.messages[0].content == "followup"
        assert incr_result.next_byte_offset == session_file.stat().st_size

    def test_offset_parse_does_not_set_started_at(self, tmp_path: Path) -> None:
        """Incremental parse should not set started_at (caller merges from DB)."""
        session_file = tmp_path / "session.jsonl"

        lines = [
            json.dumps(
                {
                    "timestamp": "2026-01-01T10:00:00Z",
                    "role": "user",
                    "content": [{"type": "text", "text": "first"}],
                }
            ),
        ]
        session_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        parser = ClaudeCodeParser()
        full = parser.parse(session_file)
        assert full.session.started_at is not None

        # Append with later timestamp
        with session_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": "2026-01-01T11:00:00Z",
                        "role": "user",
                        "content": [{"type": "text", "text": "second"}],
                    }
                )
                + "\n"
            )

        incr = parser.parse(session_file, offset=full.next_byte_offset)
        assert incr.session.started_at is None
        assert incr.session.ended_at is not None

    def test_offset_at_eof_returns_empty(self, tmp_path: Path) -> None:
        """Parsing from EOF returns zero messages."""
        session_file = tmp_path / "session.jsonl"
        session_file.write_text(
            json.dumps({"role": "user", "content": [{"type": "text", "text": "only"}]}) + "\n",
            encoding="utf-8",
        )
        parser = ClaudeCodeParser()
        full = parser.parse(session_file)

        # Parse again from the end — nothing new
        empty = parser.parse(session_file, offset=full.next_byte_offset)
        assert empty.session.message_count == 0
        assert empty.is_full_parse is False


class TestCodexByteOffset:
    def test_offset_parse_returns_only_new_messages(self, tmp_path: Path) -> None:
        session_file = tmp_path / "rollout.jsonl"

        initial = [
            json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-01-01T10:00:00Z",
                    "payload": {"type": "user_message", "message": "hello"},
                }
            ),
        ]
        session_file.write_text("\n".join(initial) + "\n", encoding="utf-8")

        parser = CodexParser()
        full = parser.parse(session_file)
        assert full.session.message_count == 1
        offset = full.next_byte_offset

        # Append
        with session_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": "2026-01-01T10:01:00Z",
                        "payload": {"type": "agent_message", "message": "response"},
                    }
                )
                + "\n"
            )

        incr = parser.parse(session_file, offset=offset)
        assert incr.is_full_parse is False
        assert incr.session.message_count == 1
        assert incr.session.messages[0].content == "response"

    def test_offset_parse_returns_absolute_codex_token_usage(self, tmp_path: Path) -> None:
        """Codex offset parses report absolute cumulative totals; REQ-INDEX-014 merges them."""
        session_file = tmp_path / "rollout.jsonl"

        initial = [
            json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-01-01T10:00:00Z",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 100,
                                "cached_input_tokens": 40,
                                "output_tokens": 10,
                                "reasoning_output_tokens": 3,
                                "total_tokens": 110,
                            },
                            "last_token_usage": {
                                "input_tokens": 100,
                                "cached_input_tokens": 40,
                                "output_tokens": 10,
                                "reasoning_output_tokens": 3,
                                "total_tokens": 110,
                            },
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-01-01T10:00:01Z",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 100,
                                "cached_input_tokens": 40,
                                "output_tokens": 10,
                                "reasoning_output_tokens": 3,
                                "total_tokens": 110,
                            },
                            "last_token_usage": {
                                "input_tokens": 100,
                                "cached_input_tokens": 40,
                                "output_tokens": 10,
                                "reasoning_output_tokens": 3,
                                "total_tokens": 110,
                            },
                        },
                    },
                }
            ),
        ]
        session_file.write_text("\n".join(initial) + "\n", encoding="utf-8")

        parser = CodexParser()
        full = parser.parse(session_file)
        assert full.session.input_tokens == 100
        assert full.session.output_tokens == 10
        offset = full.next_byte_offset

        with session_file.open("a", encoding="utf-8") as f:
            duplicate = json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": "2026-01-01T10:00:02Z",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": 100,
                                "cached_input_tokens": 40,
                                "output_tokens": 10,
                                "reasoning_output_tokens": 3,
                                "total_tokens": 110,
                            },
                            "last_token_usage": {
                                "input_tokens": 100,
                                "cached_input_tokens": 40,
                                "output_tokens": 10,
                                "reasoning_output_tokens": 3,
                                "total_tokens": 110,
                            },
                        },
                    },
                }
            )
            f.write(duplicate + "\n")

        duplicate_incr = parser.parse(session_file, offset=offset)
        assert duplicate_incr.is_full_parse is False
        assert duplicate_incr.session.input_tokens == 100
        assert duplicate_incr.session.output_tokens == 10
        duplicate_offset = duplicate_incr.next_byte_offset

        with session_file.open("a", encoding="utf-8") as f:
            appended = [
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": "2026-01-01T10:01:00Z",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 150,
                                    "cached_input_tokens": 50,
                                    "output_tokens": 15,
                                    "reasoning_output_tokens": 4,
                                    "total_tokens": 165,
                                },
                                "last_token_usage": {
                                    "input_tokens": 50,
                                    "cached_input_tokens": 10,
                                    "output_tokens": 5,
                                    "reasoning_output_tokens": 1,
                                    "total_tokens": 55,
                                },
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": "2026-01-01T10:01:01Z",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 150,
                                    "cached_input_tokens": 50,
                                    "output_tokens": 15,
                                    "reasoning_output_tokens": 4,
                                    "total_tokens": 165,
                                },
                                "last_token_usage": {
                                    "input_tokens": 50,
                                    "cached_input_tokens": 10,
                                    "output_tokens": 5,
                                    "reasoning_output_tokens": 1,
                                    "total_tokens": 55,
                                },
                            },
                        },
                    }
                ),
            ]
            f.write("\n".join(appended) + "\n")

        incr = parser.parse(session_file, offset=duplicate_offset)
        assert incr.is_full_parse is False
        assert incr.session.input_tokens == 150
        assert incr.session.output_tokens == 15


class TestPiAgentByteOffset:
    def test_offset_parse_returns_only_new_messages(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session.jsonl"

        initial = [
            json.dumps(
                {
                    "type": "message",
                    "timestamp": "2026-01-01T10:00:00Z",
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello pi"}],
                    },
                }
            ),
        ]
        session_file.write_text("\n".join(initial) + "\n", encoding="utf-8")

        parser = PiAgentParser()
        full = parser.parse(session_file)
        assert full.session.message_count == 1
        offset = full.next_byte_offset

        with session_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "message",
                        "timestamp": "2026-01-01T10:01:00Z",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "hi from pi"}],
                        },
                    }
                )
                + "\n"
            )

        incr = parser.parse(session_file, offset=offset, message_idx_base=1)
        assert incr.is_full_parse is False
        assert incr.session.message_count == 1
        assert incr.session.messages[0].content == "hi from pi"
        assert incr.session.messages[0].idx == 1


class TestIncrementalTokenPreservation:
    """Regression: NULL tokens must not become 0 after incremental append (ISSUE-1)."""

    def test_null_tokens_stay_null_after_incremental_append(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session.jsonl"

        # Write a line with no token info
        session_file.write_text(
            json.dumps({"role": "user", "content": [{"type": "text", "text": "hello"}]}) + "\n",
            encoding="utf-8",
        )

        parser = ClaudeCodeParser()
        full = parser.parse(session_file)
        assert full.session.input_tokens is None
        assert full.session.output_tokens is None

        # Append another line with no token info
        with session_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"role": "assistant", "content": [{"type": "text", "text": "reply"}]})
                + "\n"
            )

        incr = parser.parse(session_file, offset=full.next_byte_offset, message_idx_base=1)
        # Incremental parse also has no tokens — they should remain NULL
        assert incr.session.input_tokens is None
        assert incr.session.output_tokens is None


class TestIncompleteRecordPreservation:
    """A malformed complete record remains visible and unacknowledged."""

    def test_is_complete_false_preserved_after_valid_append(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session.jsonl"

        # Write one valid line + one invalid line
        session_file.write_text(
            json.dumps({"role": "user", "content": [{"type": "text", "text": "hello"}]})
            + "\n"
            + "not valid json\n",
            encoding="utf-8",
        )

        parser = ClaudeCodeParser()
        full = parser.parse(session_file)
        assert full.session.is_complete is False

        # A later valid record cannot make the earlier malformed record clean.
        with session_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"role": "assistant", "content": [{"type": "text", "text": "reply"}]})
                + "\n"
            )

        incr = parser.parse(session_file, offset=full.next_byte_offset, message_idx_base=1)
        assert incr.session.is_complete is False
        assert incr.next_byte_offset == full.next_byte_offset


class TestKimiCodeByteOffset:
    def _wire_line(self, entry: dict) -> str:
        return json.dumps(entry) + "\n"

    def test_offset_parse_returns_only_new_messages(self, tmp_path: Path) -> None:
        """Append a new step after initial parse; incremental parse sees only new lines."""
        wire = tmp_path / "wire.jsonl"
        initial = [
            {"type": "metadata", "protocol_version": "1.4", "created_at": 1784300000000},
            {
                "type": "context.append_message",
                "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                "time": 1784300001000,
            },
        ]
        wire.write_text("".join(self._wire_line(e) for e in initial), encoding="utf-8")

        parser = KimiCodeParser()
        full_result = parser.parse(wire)
        assert full_result.is_full_parse is True
        assert full_result.session.message_count == 1
        assert full_result.session.started_at is not None

        appended = [
            {
                "type": "context.append_loop_event",
                "event": {
                    "type": "content.part",
                    "uuid": "p1",
                    "turnId": "0",
                    "step": 1,
                    "part": {"type": "text", "text": "hi there"},
                },
                "time": 1784300002000,
            },
        ]
        with wire.open("a", encoding="utf-8") as f:
            f.write("".join(self._wire_line(e) for e in appended))

        incr_result = parser.parse(wire, offset=full_result.next_byte_offset, message_idx_base=1)
        assert incr_result.is_full_parse is False
        assert incr_result.session.message_count == 1
        assert incr_result.session.messages[0].content == "hi there"
        # started_at belongs to the full parse; incremental chunks must not set it
        assert incr_result.session.started_at is None
        assert incr_result.session.ended_at is not None
        assert incr_result.next_byte_offset == wire.stat().st_size

    def test_offset_at_eof_returns_empty(self, tmp_path: Path) -> None:
        wire = tmp_path / "wire.jsonl"
        wire.write_text(
            self._wire_line(
                {
                    "type": "context.append_message",
                    "message": {"role": "user", "content": [{"type": "text", "text": "only"}]},
                    "time": 1784300001000,
                }
            ),
            encoding="utf-8",
        )
        parser = KimiCodeParser()
        full = parser.parse(wire)

        empty = parser.parse(wire, offset=full.next_byte_offset, message_idx_base=1)
        assert empty.session.message_count == 0
        assert empty.is_full_parse is False
