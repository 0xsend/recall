"""Response usage snapshots, including appends through durable checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from recall.core.config import AppConfig
from recall.core.models import NormalizationCheckpoint
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
    import hashlib

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
