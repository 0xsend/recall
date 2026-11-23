"""Response usage snapshots, including appends through durable checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from recall.core.config import AppConfig
from recall.core.models import (
    CLAUDE_USAGE_COUNTER_MAX,
    CLAUDE_USAGE_INT64_MAX,
    ClaudeAnonymousCounters,
    ClaudeCounters,
    NormalizationCheckpoint,
)
from recall.parsers.claude_code import ClaudeCodeParser

from tests.test_services.test_checkpoint_resume_fallbacks import _first_wins_harness

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/claude_code/session_v2.jsonl"


def records(kind: str) -> list[dict]:
    template = json.loads(FIXTURE.read_text().splitlines()[1])
    result = []
    for index, output in enumerate([1, 50, 40]):
        entry = json.loads(json.dumps(template))
        entry["uuid"] = f"block-{index}"
        entry["message"]["content"] = [{"type": "text", "text": f"block {index}"}]
        entry["message"]["usage"]["output_tokens"] = output
        if kind == "anonymous":
            del entry["message"]["id"]
        elif kind == "sidechain":
            entry["isSidechain"] = True
        elif kind == "subagent":
            entry = {
                "type": "progress",
                "timestamp": entry["timestamp"],
                "data": {"type": "agent_progress", "agentId": "agent-1", "message": entry},
            }
        result.append(entry)
    return result


def encoded(items: list[dict]) -> bytes:
    return b"".join((json.dumps(item) + "\n").encode() for item in items)


@pytest.mark.parametrize("kind", ["main", "sidechain", "subagent", "anonymous"])
def test_response_usage_counts_once_without_dropping_blocks(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "session.jsonl"
    path.write_bytes(encoded(records(kind)))
    session = ClaudeCodeParser().parse(path).session
    assert (session.input_tokens, session.output_tokens) == (
        (2400, 91) if kind == "anonymous" else (800, 40)
    )
    assert [message.content for message in session.messages] == ["block 0", "block 1", "block 2"]


@pytest.mark.parametrize("kind", ["main", "sidechain", "subagent", "anonymous"])
@pytest.mark.parametrize("split", [1, 2])
def test_response_usage_across_a_persisted_append(tmp_path: Path, kind: str, split: int) -> None:
    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    items = records(kind)
    try:
        harness.path.write_bytes(encoded(items[:split]))
        assert harness.reconcile() is False
        checkpoint = NormalizationCheckpoint.decode(harness.stored_checkpoint())
        assert checkpoint is not None
        # Force a JSON round trip, as on a daemon restart.
        harness.set_checkpoint(checkpoint.encode())
        harness.path.write_bytes(encoded(items))
        assert harness.reconcile() is True
        row = harness.conn.execute(
            "SELECT input_tokens, output_tokens, message_count FROM session_state"
        ).fetchone()
        assert row == ((2400, 91, 3) if kind == "anonymous" else (800, 40, 3))
        harness.reconcile()
        assert (
            harness.conn.execute(
                "SELECT input_tokens, output_tokens, message_count FROM session_state"
            ).fetchone()
            == row
        )
        harness.reconcile(full=True)
        assert (
            harness.conn.execute(
                "SELECT input_tokens, output_tokens, message_count FROM session_state"
            ).fetchone()
            == row
        )
    finally:
        harness.conn.close()


def test_response_ids_are_local_to_each_session_file(tmp_path: Path) -> None:
    parser = ClaudeCodeParser()
    totals = []
    for name in ["original", "resumed", "forked"]:
        path = tmp_path / f"{name}.jsonl"
        path.write_bytes(encoded(records("main")))
        result = parser.parse(path)
        totals.append((result.session.input_tokens, result.session.output_tokens))
    assert totals == [(800, 40)] * 3


def test_response_updates_are_independent_and_preserve_missing_counters(tmp_path: Path) -> None:
    items = records("main")
    # Two responses interleave. A final snapshot supplies a corrected input,
    # no cache-creation update, and an explicit zero cache-read update.
    items[1]["message"]["id"] = "msg_other"
    items[1]["message"]["usage"] = {"input_tokens": 7, "output_tokens": 3}
    items[2]["message"]["usage"] = {
        "input_tokens": 90,
        "cache_read_input_tokens": 0,
        "output_tokens": 40,
    }
    path = tmp_path / "session.jsonl"
    path.write_bytes(encoded(items))
    session = ClaudeCodeParser().parse(path).session
    assert (session.input_tokens, session.output_tokens) == (597, 43)


def test_repeated_subagent_snapshot_and_sidechain_share_response_identity(tmp_path: Path) -> None:
    items = records("main")[:1]
    items.extend(records("subagent")[1:2])
    items.extend(records("sidechain")[2:])
    path = tmp_path / "session.jsonl"
    path.write_bytes(encoded(items))
    session = ClaudeCodeParser().parse(path).session
    assert (session.input_tokens, session.output_tokens) == (800, 40)
    assert session.message_count == 3


@pytest.mark.parametrize(
    "carried", [[], {"": [1, 2, 3, 4]}, {"msg": [1]}, {"msg": [1, 2, 3, True]}]
)
def test_invalid_response_checkpoint_is_refused(tmp_path: Path, carried: object) -> None:
    from recall.parsers.checkpoint import UnsupportedResumeState

    path = tmp_path / "session.jsonl"
    path.write_bytes(encoded(records("main")))
    with pytest.raises(UnsupportedResumeState):
        ClaudeCodeParser().parse(path, offset=1, resume_state={"response_usage": carried})


def test_oversized_claude_checkpoint_falls_back_without_changing_totals(tmp_path: Path) -> None:
    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    template = records("main")[0]
    items = []
    for index in range(4096):
        entry = json.loads(json.dumps(template))
        entry["message"]["id"] = "msg_" + hashlib.sha256(str(index).encode()).hexdigest()
        entry["message"]["usage"] = {"input_tokens": 100, "output_tokens": 10}
        items.append(entry)
    try:
        harness.path.write_bytes(encoded(items[:1]))
        assert harness.reconcile() is False
        assert harness.stored_checkpoint() is not None
        # Response count does not prevent resumable chunk boundaries.
        harness.path.write_bytes(encoded(items))
        assert harness.reconcile() is True
        for _ in range(32):
            if harness.item.current:
                break
            assert harness.drain() is True
        else:
            pytest.fail("Claude chunks did not reach EOF")
        assert harness.stored_checkpoint() is not None
        assert harness.conn.execute(
            "SELECT input_tokens, output_tokens, message_count FROM session_state"
        ).fetchone() == (409600, 40960, 4096)
        # Revisit the oldest response through the committed selective ledger.
        # Both the old and appended content remain indexed.
        correction = json.loads(json.dumps(items[0]))
        correction["message"]["usage"] = {"input_tokens": 90, "output_tokens": 7}
        items.append(correction)
        harness.path.write_bytes(encoded(items))
        assert harness.reconcile() is True
        row = harness.conn.execute(
            "SELECT input_tokens, output_tokens, message_count FROM session_state"
        ).fetchone()
        assert row == (409590, 40957, 4097)
        assert harness.stored_checkpoint() is not None
        harness.reconcile(full=True)
        assert (
            harness.conn.execute(
                "SELECT input_tokens, output_tokens, message_count FROM session_state"
            ).fetchone()
            == row
        )
    finally:
        harness.conn.close()


@pytest.mark.parametrize(
    "bad_raw_json",
    [
        "-1",
        "true",
        "false",
        "12.5",
        '"99"',
        str(CLAUDE_USAGE_COUNTER_MAX + 1),
        str(1 << 63),
        "9" * 5000,
    ],
)
def test_invalid_and_oversized_counters_leave_valid_values_intact(
    tmp_path: Path, bad_raw_json: str
) -> None:
    """Malformed, negative, boolean, fractional, and 5,000-digit counters are unsupplied."""
    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    first_line = (
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "valid-counters",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {
                    "id": "msg_valid",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "first block"}],
                    "usage": {"input_tokens": 100, "output_tokens": 12},
                },
            }
        )
        + "\n"
    ).encode()
    second_line = (
        '{"type":"assistant","sessionId":"valid-counters","timestamp":"2026-01-01T00:00:01Z",'
        '"message":{"id":"msg_valid","role":"assistant",'
        '"content":[{"type":"text","text":"second block"}],'
        f'"usage":{{"input_tokens":{bad_raw_json},"output_tokens":{bad_raw_json}}}}}}}\n'
    ).encode()
    try:
        # Full parse preserves valid counters AND indexes both blocks.
        full_path = tmp_path / "full.jsonl"
        full_path.write_bytes(first_line + second_line)
        parsed = harness.parser.parse(full_path)
        assert (parsed.session.input_tokens, parsed.session.output_tokens) == (100, 12)
        assert [m.content for m in parsed.session.messages] == ["first block", "second block"]

        # Resumed suffix parse behaves identically across a persisted checkpoint.
        harness.path.write_bytes(first_line)
        assert harness.reconcile() is False
        harness.path.write_bytes(first_line + second_line)
        assert harness.reconcile() is True
        assert harness.conn.execute(
            "SELECT input_tokens, output_tokens, message_count FROM session_state"
        ).fetchone() == (100, 12, 2)
        harness.reconcile(full=True)
        assert harness.conn.execute(
            "SELECT input_tokens, output_tokens, message_count FROM session_state"
        ).fetchone() == (100, 12, 2)
    finally:
        harness.conn.close()


def test_oversized_unquoted_id_does_not_collide_with_quoted_id(tmp_path: Path) -> None:
    """A 5,000-digit unquoted JSON integer ID decodes to None (anonymous), not a quoted string."""
    digits = "9" * 5000
    quoted_line = (
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "id-collision",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {
                    "id": digits,
                    "role": "assistant",
                    "content": [{"type": "text", "text": "quoted id"}],
                    "usage": {"input_tokens": 10, "output_tokens": 7},
                },
            }
        )
        + "\n"
    ).encode()
    unquoted_line = (
        '{"type":"assistant","sessionId":"id-collision","timestamp":"2026-01-01T00:00:01Z",'
        f'"message":{{"id":{digits},"role":"assistant",'
        '"content":[{"type":"text","text":"unquoted int id"}],'
        '"usage":{"input_tokens":20,"output_tokens":11}}}\n'
    ).encode()

    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    try:
        # Full parse keeps both separate: quoted ID (7) + anonymous unquoted (11) = 18.
        full_path = tmp_path / "full.jsonl"
        full_path.write_bytes(quoted_line + unquoted_line)
        parsed = harness.parser.parse(full_path)
        assert (parsed.session.input_tokens, parsed.session.output_tokens) == (30, 18)

        # Resumed suffix across the boundary matches full parse exactly.
        harness.path.write_bytes(quoted_line)
        assert harness.reconcile() is False
        harness.path.write_bytes(quoted_line + unquoted_line)
        assert harness.reconcile() is True
        assert harness.conn.execute(
            "SELECT input_tokens, output_tokens, message_count FROM session_state"
        ).fetchone() == (30, 18, 2)
        harness.reconcile(full=True)
        assert harness.conn.execute(
            "SELECT input_tokens, output_tokens, message_count FROM session_state"
        ).fetchone() == (30, 18, 2)
    finally:
        harness.conn.close()


def test_claude_counter_and_aggregate_overflow_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counter invariants and int64 aggregate overflow checks fire cleanly in full and suffix."""
    import recall.core.models as models_mod
    import recall.parsers.claude_code as claude_mod

    with pytest.raises(ValueError, match="invalid Claude usage counter"):
        ClaudeCounters(input_tokens=-1)
    with pytest.raises(ValueError, match="invalid Claude usage counter"):
        ClaudeCounters(output_tokens=CLAUDE_USAGE_COUNTER_MAX + 1)
    with pytest.raises(ValueError, match="exceeds int64 range"):
        ClaudeAnonymousCounters(input_tokens=CLAUDE_USAGE_INT64_MAX + 1)

    # Lower INT64 cap in a scoped test to trigger aggregate overflow in 2 records.
    monkeypatch.setattr(models_mod, "CLAUDE_USAGE_INT64_MAX", 150)
    monkeypatch.setattr(claude_mod, "check_claude_aggregate", models_mod.check_claude_aggregate)

    line_a = (
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "overflow",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {
                    "id": "r1",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "one"}],
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                },
            }
        )
        + "\n"
    ).encode()
    line_b = (
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "overflow",
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "id": "r2",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "two"}],
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                },
            }
        )
        + "\n"
    ).encode()

    full_path = tmp_path / "overflow.jsonl"
    full_path.write_bytes(line_a + line_b)
    with pytest.raises(ValueError, match="exceeds int64 range"):
        ClaudeCodeParser().parse(full_path)

    # Suffix overflow fails the turn cleanly and rolls back without corrupting prior totals.
    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    try:
        harness.path.write_bytes(line_a)
        assert harness.reconcile() is False
        assert harness.conn.execute(
            "SELECT input_tokens, output_tokens FROM session_state"
        ).fetchone() == (100, 10)
        harness.path.write_bytes(line_a + line_b)
        harness.reconcile()
        assert harness.conn.execute(
            "SELECT input_tokens, output_tokens FROM session_state"
        ).fetchone() == (100, 10)
        assert harness.item.last_error is not None
        assert "int64 range" in harness.item.last_error
    finally:
        harness.conn.close()


def test_surrogate_and_backslash_response_ids_publish_independently(tmp_path: Path) -> None:
    """A lone UTF-16 surrogate response ID publishes without UTF-8 error and stays distinct."""
    surrogate_line = (
        b'{"type":"assistant","sessionId":"surrogate-ids","timestamp":"2026-01-01T00:00:00Z",'
        b'"message":{"id":"\\ud800","role":"assistant",'
        b'"content":[{"type":"text","text":"surrogate block"}],'
        b'"usage":{"input_tokens":10,"output_tokens":7}}}\n'
    )
    backslash_line = (
        b'{"type":"assistant","sessionId":"surrogate-ids","timestamp":"2026-01-01T00:00:01Z",'
        b'"message":{"id":"\\\\ud800","role":"assistant",'
        b'"content":[{"type":"text","text":"backslash block"}],'
        b'"usage":{"input_tokens":20,"output_tokens":11}}}\n'
    )
    surrogate_update = (
        b'{"type":"assistant","sessionId":"surrogate-ids","timestamp":"2026-01-01T00:00:02Z",'
        b'"message":{"id":"\\ud800","role":"assistant",'
        b'"content":[{"type":"text","text":"surrogate update"}],'
        b'"usage":{"input_tokens":10,"output_tokens":9}}}\n'
    )

    harness = _first_wins_harness("claude-code", "session.jsonl", AppConfig.load(), tmp_path)
    try:
        # Full parse keeps surrogate and literal-backslash IDs distinct (10+20=30, 9+11=20).
        harness.path.write_bytes(surrogate_line + backslash_line)
        assert harness.reconcile() is False
        assert harness.conn.execute(
            "SELECT input_tokens, output_tokens, message_count FROM session_state"
        ).fetchone() == (30, 18, 2)
        assert harness.conn.execute("SELECT count(*) FROM claude_response_usage").fetchone() == (2,)

        # Resumed suffix updates only the surrogate ID, leaving the backslash ID intact.
        harness.path.write_bytes(surrogate_line + backslash_line + surrogate_update)
        assert harness.reconcile() is True
        assert harness.conn.execute(
            "SELECT input_tokens, output_tokens, message_count FROM session_state"
        ).fetchone() == (30, 20, 3)
        assert harness.conn.execute("SELECT count(*) FROM claude_response_usage").fetchone() == (2,)

        harness.reconcile(full=True)
        assert harness.conn.execute(
            "SELECT input_tokens, output_tokens, message_count FROM session_state"
        ).fetchone() == (30, 20, 3)
    finally:
        harness.conn.close()
