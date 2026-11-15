"""A persisted checkpoint composes native metadata and external overrides (REQ-INDEX-026)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb as duckdb
import pytest
from recall import _native
from recall.core.config import AppConfig
from recall.db.schema import ensure_schema
from recall.parsers.claude_code import ClaudeCodeParser
from recall.parsers.codex import CodexParser
from recall.parsers.grok import GrokParser
from recall.parsers.kimi_code import KimiCodeParser
from recall.parsers.pi_agent import PiAgentParser

from tests.test_services.test_checkpoint_resume_fallbacks import Harness


def records(*entries: dict[str, object]) -> bytes:
    return b"".join((json.dumps(entry) + "\n").encode() for entry in entries)


@dataclass(frozen=True)
class MetadataCase:
    parser_type: (
        type[ClaudeCodeParser]
        | type[CodexParser]
        | type[PiAgentParser]
        | type[GrokParser]
        | type[KimiCodeParser]
    )
    relative_path: str
    boundary: bytes
    first: bytes
    repeated: bytes
    expected_model: str


CASES = (
    MetadataCase(
        ClaudeCodeParser,
        "claude/session.jsonl",
        records({"type": "user", "message": {"role": "user", "content": "boundary"}}),
        records(
            {
                "type": "assistant",
                "model": "first",
                "cwd": "/first",
                "repo": "/explicit",
                "gitBranch": "main",
                "sessionId": "native-first",
                "message": {"role": "assistant", "content": "first"},
            }
        ),
        records(
            {
                "type": "assistant",
                "model": "second",
                "cwd": "/second",
                "repo": "/other",
                "gitBranch": "topic",
                "sessionId": "native-second",
                "message": {"role": "assistant", "content": "second"},
            }
        ),
        "first",
    ),
    MetadataCase(
        CodexParser,
        "codex/rollout.jsonl",
        records({"type": "event_msg", "payload": {"type": "user_message", "message": "boundary"}}),
        records(
            {
                "type": "session_meta",
                "payload": {
                    "id": "child",
                    "cwd": "/child",
                    "git": {"branch": "child", "root": "/child"},
                },
            },
            {"type": "turn_context", "payload": {"model": "first"}},
        ),
        records(
            {
                "type": "session_meta",
                "payload": {
                    "id": "parent",
                    "cwd": "/parent",
                    "git": {"branch": "parent", "root": "/parent"},
                },
            },
            {"type": "turn_context", "payload": {"model": "second"}},
        ),
        "second",
    ),
    MetadataCase(
        PiAgentParser,
        "pi/session.jsonl",
        records({"type": "message", "message": {"role": "user", "content": "boundary"}}),
        records(
            {"type": "session", "id": "first", "cwd": "/first"},
            {"type": "model_change", "modelId": "first"},
        ),
        records(
            {"type": "session", "id": "second", "cwd": "/second"},
            {"type": "model_change", "modelId": "second"},
        ),
        "second",
    ),
    MetadataCase(
        GrokParser,
        "grok/%2Fnative/session/chat_history.jsonl",
        records({"type": "user", "content": "boundary"}),
        records(
            {"type": "assistant", "model_id": "first", "content": "first"},
            {"type": "user", "content": "closed"},
        ),
        records(
            {"type": "assistant", "model_id": "second", "content": "second"},
            {"type": "user", "content": "closed again"},
        ),
        "first",
    ),
    MetadataCase(
        KimiCodeParser,
        "kimi/workdir/session/agents/main/wire.jsonl",
        records(
            {"type": "context.append_message", "message": {"role": "user", "content": "boundary"}}
        ),
        records({"type": "profile.bind", "modelAlias": "first"}),
        records({"type": "llm.request", "modelAlias": "second"}),
        "first",
    ),
)


def persisted(harness: Harness) -> dict[str, list[tuple[object, ...]]]:
    return {
        **harness.rows(),
        "metadata": harness.conn.execute(
            """SELECT s.source_session_id, ss.model, ss.cwd, ss.git_repo, ss.git_branch,
                      ss.started_at, ss.ended_at, ss.duration_seconds
               FROM sessions s JOIN session_state ss ON ss.session_id = s.id ORDER BY s.id"""
        ).fetchall(),
    }


def reference(harness: Harness) -> dict[str, list[tuple[object, ...]]]:
    with _native.duckdb_connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=harness.config.embedding.dimensions)
        oracle = Harness(harness.config, conn, harness.parser, harness.path)
        assert oracle.reconcile() is False
        return persisted(oracle)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.parser_type.source.value)
@pytest.mark.parametrize("after_first", [False, True], ids=["before-first", "after-first"])
@pytest.mark.parametrize("chunked", [False, True], ids=["unchunked", "chunked"])
def test_checkpoint_metadata_equals_full_reference_after_restart(
    case: MetadataCase,
    after_first: bool,
    chunked: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if chunked:
        from recall.parsers.checkpoint import ChunkTarget

        monkeypatch.setattr("recall.services.coordinator.RESUMED_APPEND_TARGET", ChunkTarget(1, 1))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    path = tmp_path / case.relative_path
    path.parent.mkdir(parents=True)
    database = tmp_path / "index.duckdb"
    content = case.boundary + (case.first if after_first else b"")
    path.write_bytes(content)
    with _native.duckdb_connect(str(database)) as conn:
        ensure_schema(conn, embed_dim=config.embedding.dimensions)
        harness = Harness(config, conn, case.parser_type(roots=(tmp_path,)), path)
        assert harness.reconcile() is False
        assert harness.stored_checkpoint() is not None
    for suffix in (
        (b"" if after_first else case.first) + case.repeated + case.boundary,
        case.repeated + case.boundary,
    ):
        content += suffix
        path.write_bytes(content)
        for turn in range(100):
            # A fresh adapter and connection must recover exclusively from the durable proof.
            with _native.duckdb_connect(str(database)) as conn:
                harness = Harness(config, conn, case.parser_type(roots=(tmp_path,)), path)
                assert harness.reconcile() if turn == 0 else harness.drain()
                if harness.item.current:
                    assert persisted(harness) == reference(harness)
                    assert conn.execute("SELECT model FROM session_state").fetchone() == (
                        case.expected_model,
                    )
                    break
        else:
            pytest.fail("metadata append did not finish")


@pytest.mark.parametrize("native", [False, True], ids=["no-native", "native-first"])
@pytest.mark.parametrize("chunked", [False, True], ids=["unchunked", "chunked"])
def test_grok_checkpoint_keeps_native_metadata_separate_from_sidecar(
    native: bool,
    chunked: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = CASES[3]
    if chunked:
        from recall.parsers.checkpoint import ChunkTarget

        monkeypatch.setattr("recall.services.coordinator.RESUMED_APPEND_TARGET", ChunkTarget(1, 1))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    path = tmp_path / case.relative_path
    path.parent.mkdir(parents=True)
    sidecar = path.parent / "summary.json"
    database = tmp_path / "index.duckdb"
    content = case.boundary + (case.first if native else b"")
    path.write_bytes(content)
    with _native.duckdb_connect(str(database)) as conn:
        ensure_schema(conn, embed_dim=config.embedding.dimensions)
        harness = Harness(config, conn, GrokParser(roots=(tmp_path,)), path)
        harness.reconcile()
        assert harness.stored_checkpoint() is not None
    for model in ("override-added", "override-changed", None):
        if model is None:
            sidecar.unlink()
        else:
            sidecar.write_text(
                json.dumps(
                    {
                        "current_model_id": model,
                        "info": {"cwd": f"/{model}"},
                        "git_root_dir": f"/{model}/repo",
                        "head_branch": model,
                        "created_at": "2026-01-01T00:00:00Z"
                        if model == "override-added"
                        else "2026-01-02T00:00:00Z",
                        "last_active_at": "2026-01-03T00:00:00Z",
                    }
                )
            )
        content += (case.repeated if native else b"") + case.boundary * 3
        path.write_bytes(content)
        for turn in range(100):
            with _native.duckdb_connect(str(database)) as conn:
                harness = Harness(config, conn, GrokParser(roots=(tmp_path,)), path)
                assert harness.reconcile() if turn == 0 else harness.drain()
                if harness.item.current:
                    assert persisted(harness) == reference(harness)
                    assert conn.execute("SELECT model FROM session_state").fetchone() == (
                        model or ("first" if native else None),
                    )
                    break
        else:
            pytest.fail("sidecar append did not finish")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.parser_type.source.value)
@pytest.mark.parametrize(
    "invalid", ["absent", "missing-field", "unknown-field", "wrong-type", "wrong-timestamp"]
)
def test_unsupported_metadata_checkpoint_falls_back_to_full_reference(
    case: MetadataCase,
    invalid: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from recall.core.models import NormalizationCheckpoint

    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    path = tmp_path / case.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(case.boundary + case.first)
    with _native.duckdb_connect(":memory:") as conn:
        ensure_schema(conn, embed_dim=config.embedding.dimensions)
        harness = Harness(config, conn, case.parser_type(roots=(tmp_path,)), path)
        assert harness.reconcile() is False
        checkpoint = NormalizationCheckpoint.decode(harness.stored_checkpoint())
        assert checkpoint is not None
        state = dict(checkpoint.adapter_state)
        metadata = dict(state["metadata"])
        if invalid == "absent":
            state.pop("metadata")
        elif invalid == "missing-field":
            metadata.pop("model")
        elif invalid == "unknown-field":
            metadata["future_election"] = "unknown"
        elif invalid == "wrong-type":
            metadata["model"] = True
        else:
            metadata["started_at"] = "not-a-timestamp"
        if invalid != "absent":
            state["metadata"] = metadata
        harness.set_checkpoint(replace(checkpoint, adapter_state=state).encode())
        path.write_bytes(path.read_bytes() + case.repeated + case.boundary)
        assert harness.reconcile() is False
        assert persisted(harness) == reference(harness)
