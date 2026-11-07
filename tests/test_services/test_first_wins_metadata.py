"""First metadata elections survive durable append boundaries (#323)."""

import json
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.schema import ensure_schema
from recall.parsers.claude_code import ClaudeCodeParser
from recall.parsers.kimi_code import KimiCodeParser
from recall.parsers.protocol import SessionParser

from tests.test_services.test_checkpoint_resume_fallbacks import Harness


@pytest.mark.parametrize("source", ["claude", "kimi"])
@pytest.mark.parametrize("split", [1, 2])
def test_first_wins_metadata_matches_full_parse_after_restart(
    source: str, split: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    parser: SessionParser
    if source == "claude":
        parser = ClaudeCodeParser(roots=(tmp_path,))
        records = [
            {"type": "user", "message": {"role": "user", "content": "hello"}},
            {"type": "system", "model": "first", "cwd": "/first", "gitBranch": "first"},
            {
                "type": "system",
                "model": "second",
                "cwd": "/second",
                "gitBranch": "second",
                "git_root": "/explicit",
            },
            {"type": "system", "git_root": "/ignored"},
        ]
        expected = ("first", "/first", "/explicit", "first")
    else:
        parser = KimiCodeParser(roots=(tmp_path,))
        records = [
            {"type": "context.append_message", "message": {"role": "user", "content": "hello"}},
            {"type": "llm.request", "modelAlias": "first"},
            {"type": "profile.bind", "model": "second"},
            {"type": "llm.request", "modelAlias": "third"},
        ]
        expected = ("first", None, None, None)
    path = tmp_path / "wire.jsonl"
    encoded = [(json.dumps(record) + "\n").encode() for record in records]
    db_path = tmp_path / "resume.duckdb"
    query = "SELECT model, cwd, git_repo, git_branch FROM session_state"
    for end in [split, 3, 4]:
        path.write_bytes(b"".join(encoded[:end]))
        with duckdb.connect(str(db_path)) as conn:
            ensure_schema(conn, embed_dim=config.embedding.dimensions)
            harness = Harness(config, conn, parser, path)
            assert harness.reconcile() is (end != split)
            assert harness.stored_checkpoint() is not None
    with duckdb.connect(str(db_path)) as conn:
        resumed = conn.execute(query).fetchall()
    with duckdb.connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=config.embedding.dimensions)
        assert Harness(config, conn, parser, path).reconcile() is False
        reference = conn.execute(query).fetchall()
    assert reference == [expected]
    assert resumed == reference


@pytest.mark.parametrize("source", ["claude", "kimi"])
@pytest.mark.parametrize("flag", [None, 0, "yes"])
def test_missing_or_invalid_election_state_refuses_resume(
    source: str, flag: object, tmp_path: Path
) -> None:
    from recall.parsers.checkpoint import UnsupportedResumeState

    parser: SessionParser = ClaudeCodeParser() if source == "claude" else KimiCodeParser()
    path = tmp_path / "session.jsonl"
    path.write_text('{"type":"system"}\n' if source == "claude" else '{"type":"metadata"}\n')
    checkpoint = parser.parse(path).normalization_checkpoint
    assert checkpoint is not None
    state = dict(checkpoint.adapter_state)
    if flag is None:
        del state["model_elected"]
    else:
        state["model_elected"] = flag
    with pytest.raises(UnsupportedResumeState):
        parser.parse(path, offset=checkpoint.offset, resume_state=state)


def test_late_claude_cwd_does_not_replace_explicit_repository(tmp_path: Path) -> None:
    parser = ClaudeCodeParser()
    path = tmp_path / "session.jsonl"
    path.write_text('{"type":"system","git_root":"/explicit"}\n')
    checkpoint = parser.parse(path).normalization_checkpoint
    assert checkpoint is not None
    with path.open("a") as stream:
        stream.write('{"type":"system","cwd":"/fallback"}\n')
    resumed = parser.parse(path, offset=checkpoint.offset, resume_state=checkpoint.adapter_state)
    assert resumed.session.cwd == "/fallback"
    assert resumed.session.git_repo is None
    assert parser.parse(path).session.git_repo == "/explicit"
