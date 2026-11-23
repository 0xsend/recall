"""File-local last-supplied counters survive complete-record splits and reopen."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.fts_sidecar import ensure_sidecar
from recall.db.schema import ensure_schema
from recall.parsers.claude_code import ClaudeCodeParser

from tests.test_services.test_checkpoint_resume_fallbacks import Harness


def _block(response: str | None, text: str, usage: dict[str, int]) -> dict:
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "usage": usage,
    }
    if response is not None:
        message["id"] = response
    return {
        "type": "assistant",
        "sessionId": "boundary-session",
        "timestamp": "2026-01-01T00:00:00Z",
        "message": message,
    }


@pytest.mark.parametrize("reverse", [False, True], ids=["A-then-B", "B-then-A"])
@pytest.mark.parametrize("split", [1, 2, 3, 4])
def test_counter_fields_survive_interleaving_and_reopen(
    tmp_path: Path, reverse: bool, split: int
) -> None:
    config = AppConfig.load()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    ensure_sidecar(config)
    parser = ClaudeCodeParser(roots=(tmp_path,))
    path = tmp_path / "session.jsonl"
    database = tmp_path / "index.duckdb"
    first = _block(
        "A",
        "first A",
        {
            "input_tokens": 100,
            "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 200,
            "output_tokens": 50,
        },
    )
    second = _block("B", "independent B", {"input_tokens": 7, "output_tokens": 3})
    initial = [second, first] if reverse else [first, second]
    nested = {
        "type": "progress",
        "timestamp": "2026-01-01T00:00:00Z",
        "data": {
            "type": "agent_progress",
            "agentId": "child",
            "message": _block("A", "nested correction", {"output_tokens": 0}),
        },
    }
    lower = _block("A", "sidechain correction", {"input_tokens": 90, "cache_read_input_tokens": 0})
    lower["isSidechain"] = True
    entries = [
        *initial,
        nested,
        lower,
        _block(None, "anonymous", {"input_tokens": 2, "output_tokens": 1}),
    ]
    lines = [(json.dumps(entry) + "\n").encode() for entry in entries]
    path.write_bytes(b"".join(lines[:split]))
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn, embed_dim=config.embedding.dimensions)
        harness = Harness(config, conn, parser, path)
        assert harness.reconcile() is False
        assert harness.stored_checkpoint() is not None
    path.write_bytes(b"".join(lines))
    with duckdb.connect(str(database)) as conn:
        harness = Harness(config, conn, parser, path)
        assert harness.reconcile() is True
        assert conn.execute(
            "SELECT input_tokens, output_tokens, message_count FROM session_state"
        ).fetchone() == (599, 4, 5)
        reference = parser.parse(path).session
        assert (reference.input_tokens, reference.output_tokens, reference.message_count) == (
            599,
            4,
            5,
        )
        assert len(harness.rows()["messages"]) == 5
        before = harness.rows()
        harness.reconcile()
        assert harness.rows() == before
        harness.reconcile(full=True)
        assert harness.rows() == before
