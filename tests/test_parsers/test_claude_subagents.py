"""Subagent transcripts: lineage from path and meta (REQ-PARSE-046), fork copies (REQ-PARSE-047)."""

from __future__ import annotations

import json
from pathlib import Path

from recall.core.config import AppConfig
from recall.core.models import NormalizationCheckpoint, SessionLineage
from recall.parsers.claude_code import ClaudeCodeParser

from tests.test_services.test_checkpoint_resume_fallbacks import _first_wins_harness

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/claude_code/session_v2.jsonl"
ROOT_SID = "34221246-b285-48af-9b2a-fe4c083f3208"


def response(uuid: str, message_id: str, output: int) -> dict:
    entry = json.loads(FIXTURE.read_text().splitlines()[1])
    entry["uuid"] = uuid
    entry["sessionId"] = ROOT_SID
    entry["message"]["id"] = message_id
    entry["message"]["content"] = [{"type": "text", "text": uuid}]
    entry["message"]["usage"]["output_tokens"] = output
    return entry


def encoded(items: list[dict]) -> bytes:
    return b"".join((json.dumps(item) + "\n").encode() for item in items)


def fork_records() -> list[dict]:
    """A fork: the context ref, the parent's spawning response, then its own work."""
    return [
        {"type": "fork-context-ref", "sessionId": ROOT_SID, "uuid": "ref"},
        response("copied", "msg_parent_spawn", 7),
        response("own-1", "msg_fork_1", 11),
        response("own-2", "msg_fork_2", 13),
    ]


def subagent_path(root: Path, agent_id: str = "a1b2c3") -> Path:
    path = root / "-Users-me-repo" / ROOT_SID / "subagents" / f"agent-{agent_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_subagent_transcript_carries_lineage_from_its_path_and_meta(tmp_path: Path) -> None:
    path = subagent_path(tmp_path)
    path.write_bytes(encoded([response("one", "msg_1", 5)]))
    path.with_name("agent-a1b2c3.meta.json").write_text(
        json.dumps({"agentType": "Explore", "description": "Find the parser", "spawnDepth": 1})
    )
    parser = ClaudeCodeParser(roots=(tmp_path,))

    session = parser.parse(path).session

    assert session.lineage == SessionLineage(
        root_source_session_id=ROOT_SID,
        agent_id="a1b2c3",
        agent_type="Explore",
        agent_description="Find the parser",
    )
    assert parser.sidecar_paths(path) == [path.with_name("agent-a1b2c3.meta.json")]


def test_subagent_without_meta_keeps_its_path_lineage(tmp_path: Path) -> None:
    path = subagent_path(tmp_path)
    path.write_bytes(encoded([response("one", "msg_1", 5)]))
    oversized = path.with_name("agent-a1b2c3.meta.json")
    parser = ClaudeCodeParser(roots=(tmp_path,))
    expected = SessionLineage(root_source_session_id=ROOT_SID, agent_id="a1b2c3")

    nested = "[" * 10_000 + "]" * 10_000
    for meta in (None, "not json", json.dumps(["list"]), " " * (64 * 1024 + 1), nested):
        if meta is None:
            oversized.unlink(missing_ok=True)
        else:
            oversized.write_text(meta)
        result = parser.parse(path)
        assert result.session.lineage == expected
        assert not result.diagnostics


def test_root_transcript_has_no_lineage(tmp_path: Path) -> None:
    path = tmp_path / "-Users-me-repo" / f"{ROOT_SID}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded([response("one", "msg_1", 5)]))
    parser = ClaudeCodeParser(roots=(tmp_path,))

    assert parser.parse(path).session.lineage is None
    assert parser.sidecar_paths(path) == []


def test_fork_copy_of_parent_response_is_not_counted(tmp_path: Path) -> None:
    path = subagent_path(tmp_path)
    # A later content block of the copied response repeats its id.
    path.write_bytes(encoded([*fork_records(), response("copied-block", "msg_parent_spawn", 9)]))

    session = ClaudeCodeParser(roots=(tmp_path,)).parse(path).session

    assert session.output_tokens == 24
    assert [message.content for message in session.messages] == [
        "copied",
        "own-1",
        "own-2",
        "copied-block",
    ]


def test_fork_rule_applies_only_to_a_subagent_that_opens_with_the_marker(tmp_path: Path) -> None:
    parser = ClaudeCodeParser(roots=(tmp_path,))
    root = tmp_path / "-Users-me-repo" / f"{ROOT_SID}.jsonl"
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_bytes(encoded(fork_records()))
    late = subagent_path(tmp_path, "late")
    late.write_bytes(encoded([response("first", "msg_first", 3), *fork_records()]))

    assert parser.parse(root).session.output_tokens == 31
    assert parser.parse(late).session.output_tokens == 34


def test_fork_copy_across_a_resumed_boundary(tmp_path: Path) -> None:
    """A checkpoint between the ref and the copy still keeps the copy uncounted."""
    harness = _first_wins_harness(
        "claude-code",
        f"-Users-me-repo/{ROOT_SID}/subagents/agent-f1.jsonl",
        AppConfig.load(),
        tmp_path,
    )
    items = fork_records()
    try:
        harness.path.write_bytes(encoded(items[:1]))
        harness.reconcile()
        checkpoint = NormalizationCheckpoint.decode(harness.stored_checkpoint())
        assert checkpoint is not None
        harness.set_checkpoint(checkpoint.encode())
        harness.path.write_bytes(encoded(items))
        assert harness.reconcile() is True

        query = "SELECT output_tokens, message_count FROM session_state"
        resumed = harness.conn.execute(query).fetchone()
        harness.reconcile(full=True)
        assert resumed == harness.conn.execute(query).fetchone() == (24, 3)
    finally:
        harness.conn.close()
