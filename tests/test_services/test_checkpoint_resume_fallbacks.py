"""Raw reconciliation resume decision and its fallbacks (REQ-INDEX-025).

``prepare_raw_sources`` runs outside the writer with no connection, so the only
thing it can trust is what the ``SourceFile`` carries.  It may parse a suffix
only when the stored checkpoint was committed with the acknowledgement it names
and the bytes it describes are still on disk unchanged.  Every other outcome is
a full reference parse.

The assertions are the committed rows, not the calls: a fallback is only
correct if what lands in the database is exactly what a from-scratch
reconciliation of the same file would have landed.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog, SourceFile, source_key
from recall.parsers.codex import CodexParser
from recall.parsers.protocol import SessionParser
from recall.services.coordinator import (
    capture_path,
    commit_prepared_raw_sources,
    observe_path,
    prepare_raw_sources,
)

from tests.test_parsers.test_normalization_checkpoint_adapters import CASES, AdapterCase


def _event(payload: dict[str, object], *, at: str) -> bytes:
    return (json.dumps({"type": "event_msg", "timestamp": at, "payload": payload}) + "\n").encode()


HEAD = (
    json.dumps(
        {
            "type": "session_meta",
            "payload": {
                "id": "codex-resume",
                "timestamp": "2026-02-01T09:00:00Z",
                "cwd": "/repo",
                "git": {"branch": "main", "root": "/repo"},
            },
        }
    )
    + "\n"
).encode()

PREFIX = (
    HEAD
    + _event({"type": "task_started"}, at="2026-02-01T09:00:01Z")
    + _event({"type": "user_message", "message": "list the repo"}, at="2026-02-01T09:00:02Z")
    + _event({"type": "agent_message", "message": "Working on it."}, at="2026-02-01T09:00:03Z")
)

APPEND = (
    _event(
        {"type": "agent_message", "message": "The repo has a README."}, at="2026-02-01T09:00:06Z"
    )
    + _event({"type": "agent_message", "message": "Anything else?"}, at="2026-02-01T09:00:07Z")
    + _event({"type": "task_complete"}, at="2026-02-01T09:00:08Z")
)

FULL = PREFIX + APPEND


@dataclass(frozen=True)
class Harness:
    config: AppConfig
    conn: duckdb.DuckDBPyConnection
    parser: SessionParser
    path: Path

    @property
    def catalog(self) -> SourceCatalog:
        import time

        return SourceCatalog(self.conn, clock=time.time)

    @property
    def source(self) -> str:
        return self.parser.source.value

    @property
    def key_path(self) -> str:
        """The catalog spells every source with its resolved path."""
        return str(self.path.resolve())

    @property
    def item(self) -> SourceFile:
        found = self.catalog.get(self.source, self.key_path)
        assert found is not None
        return found

    def reconcile(self, *, full: bool = False) -> bool:
        """One full observe/prepare/commit turn; returns whether a suffix was parsed."""
        item = observe_path(self.parser, capture_path(self.parser, self.path), conn=self.conn)
        prepared = prepare_raw_sources((item,), {self.source: self.parser}, full=full)
        resumed = prepared[0].result is not None and not prepared[0].result.is_full_parse
        commit_prepared_raw_sources(prepared, self.config, conn=self.conn)
        return resumed

    def drain(self) -> bool:
        """Serve whatever the catalog already holds, carrying no request object.

        This is the shared drain: a turn that never saw the request that made
        the source pending, which is what a backlog handoff or a daemon restart
        leaves behind.
        """
        item = self.item
        prepared = prepare_raw_sources((item,), {self.source: self.parser})
        resumed = prepared[0].result is not None and not prepared[0].result.is_full_parse
        commit_prepared_raw_sources(prepared, self.config, conn=self.conn)
        return resumed

    def stored_checkpoint(self) -> str | None:
        row = self.conn.execute(
            "SELECT normalization_checkpoint FROM source_files WHERE source_key = ?",
            [source_key(self.source, self.key_path)],
        ).fetchone()
        assert row is not None
        return None if row[0] is None else str(row[0])

    def set_checkpoint(self, value: str | None) -> None:
        self.conn.execute(
            "UPDATE source_files SET normalization_checkpoint = ? WHERE source_key = ?",
            [value, source_key(self.source, self.key_path)],
        )

    def rows(self) -> dict[str, list[tuple[object, ...]]]:
        session = self.conn.execute(
            "SELECT id, source, source_path FROM sessions ORDER BY id"
        ).fetchall()
        messages = self.conn.execute(
            """SELECT m.session_id, m.idx, m.id, m.agent_id, s.role, s.content, s.thinking,
                      s.timestamp, s.context_mode, s.context_text
               FROM messages m JOIN message_state s ON s.message_id = m.id
               ORDER BY m.session_id, m.idx"""
        ).fetchall()
        tool_calls = self.conn.execute(
            """SELECT session_id, message_id, idx, id, tool_name, CAST(tool_input AS VARCHAR),
                      bash_command
               FROM tool_calls ORDER BY session_id, message_id NULLS LAST, idx"""
        ).fetchall()
        results = self.conn.execute(
            "SELECT tool_call_id, result_summary, is_error FROM tool_results ORDER BY tool_call_id"
        ).fetchall()
        markers = self.conn.execute(
            """SELECT session_id, message_idx, reason, ends_turn
               FROM session_stop_markers ORDER BY session_id, message_idx"""
        ).fetchall()
        # Every column `_incremental_write_session` merges. A narrower
        # projection is what let a resumed append disagree with the reference
        # about first-wins metadata while the suite stayed green.
        state = self.conn.execute(
            """SELECT session_id, started_at, ended_at, duration_seconds, model, cwd, git_repo,
                      git_branch, message_count, tool_count, input_tokens, output_tokens,
                      is_complete, file_size, sidecar_mtime, last_byte_offset, host
               FROM session_state ORDER BY session_id"""
        ).fetchall()
        return {
            "sessions": session,
            "messages": messages,
            "tool_calls": tool_calls,
            "tool_results": results,
            "stop_markers": markers,
            "session_state": state,
        }


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    return AppConfig.load()


@pytest.fixture
def harness(tmp_path: Path, config: AppConfig) -> Iterator[Harness]:
    root = tmp_path / "sessions"
    root.mkdir()
    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=config.embedding.dimensions)
    try:
        yield Harness(config, conn, CodexParser(roots=(root,)), root / "rollout-resume.jsonl")
    finally:
        conn.close()


def _reference(harness: Harness, content: bytes) -> dict[str, list[tuple[object, ...]]]:
    """What a from-scratch reconciliation of ``content`` commits.

    A separate database on the same path keeps session ids and every derived
    identity identical, so the comparison is about normalization and nothing
    else.
    """
    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=harness.config.embedding.dimensions)
    try:
        reference = Harness(harness.config, conn, harness.parser, harness.path)
        original = harness.path.read_bytes() if harness.path.exists() else None
        harness.path.write_bytes(content)
        assert reference.reconcile() is False
        rows = reference.rows()
        if original is not None:
            harness.path.write_bytes(original)
        return rows
    finally:
        conn.close()


def _seed_resumable_checkpoint(harness: Harness) -> str:
    harness.path.write_bytes(PREFIX)
    assert harness.reconcile() is False
    checkpoint = harness.stored_checkpoint()
    assert checkpoint is not None, "a committed prefix must leave a resume proof"
    return checkpoint


def _replace_with_distinct_identity(path: Path, content: bytes) -> None:
    """Replace an existing file without allowing its inode to be reused."""
    replacement = path.with_name(f".{path.name}.replacement")
    assert not replacement.exists()
    original = path.stat()
    replacement.write_bytes(content)
    candidate = replacement.stat()
    assert (candidate.st_dev, candidate.st_ino) != (original.st_dev, original.st_ino)
    replacement.replace(path)


def test_a_plain_append_resumes_from_the_committed_checkpoint(harness: Harness) -> None:
    _seed_resumable_checkpoint(harness)
    expected = _reference(harness, FULL)

    harness.path.write_bytes(FULL)
    assert harness.reconcile() is True

    assert harness.rows() == expected
    item = harness.item
    assert item.current
    assert item.committed_offset == len(FULL)
    assert item.content_epoch == 0


def test_a_resumed_append_advances_the_checkpoint_to_the_new_tail(harness: Harness) -> None:
    first = _seed_resumable_checkpoint(harness)

    harness.path.write_bytes(FULL)
    assert harness.reconcile() is True

    advanced = harness.stored_checkpoint()
    assert advanced is not None
    assert advanced != first
    # The proof left by a resumed turn must be the one a full parse would have
    # left, or the next append resumes from a weaker boundary than a reparse.
    reference_conn = duckdb.connect(":memory:")
    try:
        ensure_schema(reference_conn, embed_dim=harness.config.embedding.dimensions)
        reference = Harness(harness.config, reference_conn, harness.parser, harness.path)
        assert reference.reconcile() is False
        assert reference.stored_checkpoint() == advanced
    finally:
        reference_conn.close()


Mutation = Callable[[Harness], bytes]


def _legacy_checkpoint(harness: Harness) -> bytes:
    harness.set_checkpoint(None)
    return FULL


def _malformed_checkpoint(harness: Harness) -> bytes:
    harness.set_checkpoint("{not json")
    return FULL


def _unknown_envelope_version(harness: Harness) -> bytes:
    stored = harness.stored_checkpoint()
    assert stored is not None
    envelope = json.loads(stored)
    envelope["version"] = int(envelope["version"]) + 1
    harness.set_checkpoint(json.dumps(envelope))
    return FULL


def _parser_revision_mismatch(harness: Harness) -> bytes:
    stored = harness.stored_checkpoint()
    assert stored is not None
    envelope = json.loads(stored)
    envelope["parser_revision"] = "0" * 64
    harness.set_checkpoint(json.dumps(envelope))
    return FULL


def _unsupported_adapter_state(harness: Harness) -> bytes:
    """State this build's adapter cannot read, in an otherwise sound envelope.

    REQ-INDEX-025 names unsupported adapter state as its own invalidation
    condition, and it is the one that does not look like damage: the offset and
    digest still describe the acknowledged prefix, so the envelope passes every
    catalog check and only the adapter can refuse it. A downgrade meets this,
    and it must reparse rather than fail the source.
    """
    stored = harness.stored_checkpoint()
    assert stored is not None
    envelope = json.loads(stored)
    assert envelope["adapter_state"], "the seeded boundary must carry adapter state"
    envelope["adapter_state"] = dict(envelope["adapter_state"]) | {"pending_turn_depth": 3}
    harness.set_checkpoint(json.dumps(envelope))
    return FULL


def _checkpoint_from_an_older_generation(harness: Harness) -> bytes:
    """A checkpoint whose offset no longer matches the acknowledged prefix.

    This is what a worker that prepared against an earlier generation would
    leave behind; the envelope is well formed and the revision matches, but it
    does not describe the bytes the catalog says are committed.
    """
    stored = harness.stored_checkpoint()
    assert stored is not None
    envelope = json.loads(stored)
    envelope["offset"] = int(envelope["offset"]) - 1
    harness.set_checkpoint(json.dumps(envelope))
    return FULL


def _prefix_rewritten_and_grown(harness: Harness) -> bytes:
    return PREFIX.replace(b"list the repo", b"list the tree") + APPEND


def _same_size_rewrite(harness: Harness) -> bytes:
    rewritten = PREFIX.replace(b"Working on it.", b"Working on it!")
    assert len(rewritten) == len(PREFIX)
    return rewritten


def _shrink_below_the_committed_offset(harness: Harness) -> bytes:
    """Truncation back past every committed message.

    The reference row set for this one is empty of messages and markers, so it
    also pins that a full-parse fallback clears the stop markers the previous
    tail left behind rather than stranding them on indices that no longer
    exist.
    """
    return HEAD


def _replaced_file(harness: Harness) -> bytes:
    harness.path.unlink()
    return (
        HEAD
        + _event({"type": "task_started"}, at="2026-02-01T10:00:01Z")
        + _event(
            {"type": "user_message", "message": "different session"}, at="2026-02-01T10:00:02Z"
        )
        + _event(
            {"type": "agent_message", "message": "Different answer."}, at="2026-02-01T10:00:03Z"
        )
        + _event({"type": "task_complete"}, at="2026-02-01T10:00:04Z")
    )


FALLBACKS = (
    pytest.param(_legacy_checkpoint, id="legacy-null-checkpoint"),
    pytest.param(_malformed_checkpoint, id="malformed-checkpoint"),
    pytest.param(_unknown_envelope_version, id="unknown-envelope-version"),
    pytest.param(_parser_revision_mismatch, id="parser-revision-mismatch"),
    pytest.param(_unsupported_adapter_state, id="unsupported-adapter-state"),
    pytest.param(_checkpoint_from_an_older_generation, id="stale-generation"),
    pytest.param(_prefix_rewritten_and_grown, id="prefix-rewrite-plus-growth"),
    pytest.param(_same_size_rewrite, id="same-size-rewrite"),
    pytest.param(_shrink_below_the_committed_offset, id="shrink"),
    pytest.param(_replaced_file, id="replacement"),
)


@pytest.mark.parametrize("mutate", FALLBACKS)
def test_an_invalid_checkpoint_falls_back_to_the_full_reference_parse(
    harness: Harness, mutate: Mutation
) -> None:
    _seed_resumable_checkpoint(harness)
    content = mutate(harness)
    expected = _reference(harness, content)

    harness.path.write_bytes(content)
    assert harness.reconcile() is False

    assert harness.rows() == expected


@pytest.mark.parametrize("mutate", FALLBACKS)
def test_a_fallback_leaves_a_checkpoint_the_next_append_can_use(
    harness: Harness, mutate: Mutation
) -> None:
    """A fallback is not a downgrade: the reference parse re-proves the tail."""
    _seed_resumable_checkpoint(harness)
    content = mutate(harness)

    harness.path.write_bytes(content)
    assert harness.reconcile() is False

    checkpoint = harness.stored_checkpoint()
    assert checkpoint is not None
    envelope = json.loads(checkpoint)
    item = harness.item
    assert item.current
    assert envelope["offset"] == item.committed_offset
    assert envelope["prefix_sha256"] == item.committed_prefix_sha256


def test_a_rewrite_is_never_mistaken_for_an_append(harness: Harness) -> None:
    """Growth alone must not authorize a suffix parse.

    A prefix edit plus an append is the failure that silently corrupts a
    session: the bytes past the old offset really are new, so only the prefix
    digest can tell the two apart.
    """
    _seed_resumable_checkpoint(harness)
    rewritten = PREFIX.replace(b"list the repo", b"list the tree") + APPEND
    assert len(rewritten) > len(PREFIX)

    harness.path.write_bytes(rewritten)
    assert harness.reconcile() is False

    contents = [row[5] for row in harness.rows()["messages"]]
    assert "list the tree" in contents
    assert "list the repo" not in contents
    item = harness.item
    # A semantic rewrite bumps the epoch; an append must not.
    assert item.content_epoch == 1


def test_a_replacement_with_an_identical_prefix_falls_back(harness: Harness) -> None:
    """A new file identity is a replacement even when its bytes begin alike."""
    _seed_resumable_checkpoint(harness)
    assert harness.item.signature is not None
    committed_identity = (harness.item.signature.dev, harness.item.signature.inode)
    expected = _reference(harness, FULL)

    _replace_with_distinct_identity(harness.path, FULL)
    current = harness.path.stat()
    assert (current.st_dev, current.st_ino) != committed_identity

    assert harness.reconcile() is False
    assert harness.rows() == expected


def test_a_prefix_replaced_between_verification_and_parse_falls_back(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parser's own handle must start with the prefix that was verified."""
    _seed_resumable_checkpoint(harness)
    rewritten = FULL.replace(b"list the repo", b"show the tree")
    assert len(rewritten) == len(FULL)
    expected = _reference(harness, rewritten)
    harness.path.write_bytes(FULL)

    import recall.services.coordinator as coordinator

    original_verify = coordinator._on_disk_prefix_sha256

    def verify_then_replace(path: Path, size: int) -> str | None:
        digest = original_verify(path, size)
        path.write_bytes(rewritten)
        return digest

    monkeypatch.setattr(coordinator, "_on_disk_prefix_sha256", verify_then_replace)

    assert harness.reconcile() is False
    assert harness.rows() == expected
    assert harness.item.content_epoch == 1


TORN = FULL + b'{"type": "event_msg"'
AFTER_TORN = FULL + _event({"type": "task_started"}, at="2026-02-01T09:00:09Z")


def test_a_replacement_between_verification_and_parse_falls_back(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parser-handle identity closes the window after the catalog check."""
    _seed_resumable_checkpoint(harness)
    assert harness.item.signature is not None
    committed_identity = (harness.item.signature.dev, harness.item.signature.inode)
    expected = _reference(harness, FULL)
    harness.path.write_bytes(FULL)

    import recall.services.coordinator as coordinator

    original_verify = coordinator._on_disk_prefix_sha256

    def verify_then_replace(path: Path, size: int) -> str | None:
        digest = original_verify(path, size)
        _replace_with_distinct_identity(path, FULL)
        current = path.stat()
        assert (current.st_dev, current.st_ino) != committed_identity
        return digest

    monkeypatch.setattr(coordinator, "_on_disk_prefix_sha256", verify_then_replace)

    assert harness.reconcile() is False
    assert harness.rows() == expected


def test_a_torn_tail_costs_one_reparse_and_not_the_turn_after_it(harness: Harness) -> None:
    """A half-written line is the steady state, not damage.

    The suffix carrying the diagnostic is still discarded -- only a parse that
    saw the whole file can say what the source holds. But the reference parse
    that replaces it stopped at the same complete boundary a clean parse would
    have, so it has a proof to leave, and the next clean append resumes from it
    instead of paying for a second full reparse.
    """
    _seed_resumable_checkpoint(harness)
    harness.path.write_bytes(FULL)
    assert harness.reconcile() is True

    harness.path.write_bytes(TORN)
    assert harness.reconcile() is False
    torn_checkpoint = harness.stored_checkpoint()
    assert torn_checkpoint is not None
    assert json.loads(torn_checkpoint)["offset"] == harness.item.committed_offset

    expected = _reference(harness, AFTER_TORN)
    harness.path.write_bytes(AFTER_TORN)
    assert harness.reconcile() is True

    assert harness.rows() == expected
    assert harness.item.current


def test_a_malformed_tail_leaves_no_proof_past_the_committed_offset(harness: Harness) -> None:
    """Damage is still unresumable, and it never moves the stored proof.

    A stored checkpoint beside a committed offset it does not describe is the
    condition REQ-INDEX-025 forbids outright, so whether the turn defers or
    acknowledges, the two must still agree -- and the committed history the
    prefix left must be untouched.
    """
    _seed_resumable_checkpoint(harness)
    before = harness.rows()

    harness.path.write_bytes(FULL + b"{not json}\n")
    harness.reconcile()

    stored = harness.stored_checkpoint()
    assert stored is None or json.loads(stored)["offset"] == harness.item.committed_offset
    assert harness.rows() == before


def test_a_missing_source_leaves_the_prior_checkpoint_and_history_intact(
    harness: Harness, tmp_path: Path
) -> None:
    """A vanished file is not a truncation; nothing about the row may move."""
    _seed_resumable_checkpoint(harness)
    before_rows = harness.rows()
    before_checkpoint = harness.stored_checkpoint()

    backup = tmp_path / "backup.jsonl"
    item = harness.item
    shutil.move(str(harness.path), backup)
    prepared = prepare_raw_sources((item,), {harness.parser.source.value: harness.parser})
    commit_prepared_raw_sources(prepared, harness.config, conn=harness.conn)

    assert harness.rows() == before_rows
    assert harness.stored_checkpoint() == before_checkpoint


# --- Committed-row equivalence for every adapter (REQ-INDEX-026) -------------
#
# The fallback matrix above is Codex-only on purpose: it exercises catalog and
# coordinator decisions, which are adapter-independent.  What is *not* adapter
# independent is what an adapter's suffix says about the session row, so the
# append path below runs the whole observe/prepare/commit turn for every
# supported adapter and compares the committed rows against a from-scratch
# reconciliation of the same bytes.


def _case_bytes(case: AdapterCase, count: int) -> bytes:
    """Exactly what ``AdapterCase.write`` lays down for ``count`` records."""
    return "".join(f"{record}\n" for record in case.records[:count]).encode("utf-8")


def _adapter_harness(case: AdapterCase, config: AppConfig, root: Path, count: int) -> Harness:
    path = case.write(root, count)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=config.embedding.dimensions)
    return Harness(config, conn, case.parser(root), path)


ADAPTER_CASES = [pytest.param(case, id=case.name) for case in CASES]


@pytest.mark.parametrize("case", ADAPTER_CASES)
def test_a_resumed_append_commits_the_rows_a_full_parse_commits(
    case: AdapterCase, config: AppConfig, tmp_path: Path
) -> None:
    root = tmp_path / "roots"
    root.mkdir()
    harness = _adapter_harness(case, config, root, case.closed_split)
    try:
        assert harness.reconcile() is False
        assert harness.stored_checkpoint() is not None, "a closed boundary must leave a proof"
        full = _case_bytes(case, len(case.records))
        expected = _reference(harness, full)

        harness.path.write_bytes(full)
        assert harness.reconcile() is True

        assert harness.rows() == expected
        assert harness.item.current
        assert harness.item.committed_offset == len(full)
    finally:
        harness.conn.close()


# --- First-wins session metadata across a resume boundary (REQ-INDEX-026) ----
#
# Claude Code and Kimi Code take the *first* value they see for the session's
# model (and, for Claude Code, cwd / git_repo / git_branch).  A suffix sees a
# different "first" than the whole file does, and the incremental merge is
# ``COALESCE(?, stored)``, so a suffix that carries any value overwrites the
# committed one.  Both transcripts below change every such field after the
# checkpoint, which is what `/model` and `git checkout` do mid-session.


def _claude_line(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload) + "\n").encode("utf-8")


CLAUDE_PREFIX = _claude_line(
    {
        "type": "user",
        "uuid": "u1",
        "timestamp": "2026-01-16T04:00:00Z",
        "cwd": "/repo",
        "sessionId": "cc-metadata",
        "gitBranch": "main",
        "git_root": "/repo",
        "message": {"role": "user", "content": [{"type": "text", "text": "run the tests"}]},
    }
) + _claude_line(
    {
        "type": "assistant",
        "uuid": "a1",
        "parentUuid": "u1",
        "timestamp": "2026-01-16T04:01:00Z",
        "cwd": "/repo",
        "gitBranch": "main",
        "git_root": "/repo",
        "message": {
            "role": "assistant",
            "model": "claude-opus-4",
            "content": [{"type": "text", "text": "All green."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        },
    }
)

# `/model` plus `git checkout` after the checkpoint: every first-wins field
# carries a different value here than the committed prefix already holds.
CLAUDE_APPEND = _claude_line(
    {
        "type": "user",
        "uuid": "u2",
        "parentUuid": "a1",
        "timestamp": "2026-01-16T04:02:00Z",
        "cwd": "/worktree",
        "sessionId": "cc-metadata",
        "gitBranch": "feature",
        "git_root": "/worktree",
        "message": {"role": "user", "content": [{"type": "text", "text": "now lint"}]},
    }
) + _claude_line(
    {
        "type": "assistant",
        "uuid": "a2",
        "parentUuid": "u2",
        "timestamp": "2026-01-16T04:03:00Z",
        "cwd": "/worktree",
        "gitBranch": "feature",
        "git_root": "/worktree",
        "message": {
            "role": "assistant",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "Lint is clean."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        },
    }
)


def _kimi_line(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload) + "\n").encode("utf-8")


def _kimi_turn(step: int, *, model: str, text: str, at: int) -> bytes:
    """One closed Kimi step, preceded by the model binding that turn used."""
    return (
        _kimi_line({"type": "llm.request", "modelAlias": model, "time": at})
        + _kimi_line(
            {
                "type": "context.append_message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                    "toolCalls": [],
                    "origin": {"kind": "user"},
                },
                "time": at + 10,
            }
        )
        + _kimi_line(
            {
                "type": "context.append_loop_event",
                "event": {
                    "type": "step.begin",
                    "uuid": f"step-{step}",
                    "turnId": "0",
                    "step": step,
                },
                "time": at + 20,
            }
        )
        + _kimi_line(
            {
                "type": "context.append_loop_event",
                "event": {
                    "type": "content.part",
                    "uuid": f"part-{step}",
                    "turnId": "0",
                    "step": step,
                    "stepUuid": f"step-{step}",
                    "part": {"type": "text", "text": f"Answering {text}"},
                },
                "time": at + 30,
            }
        )
        + _kimi_line(
            {
                "type": "context.append_loop_event",
                "event": {
                    "type": "step.end",
                    "uuid": f"step-{step}",
                    "turnId": "0",
                    "step": step,
                    "finishReason": "stop",
                },
                "time": at + 40,
            }
        )
    )


KIMI_PREFIX = _kimi_turn(1, model="kimi-code/k3", text="list the files", at=1784300000000)
KIMI_APPEND = _kimi_turn(2, model="kimi-code/k4", text="now lint", at=1784300001000)

FIRST_WINS_METADATA = (
    pytest.param(
        "claude-code",
        "projects/repo/session.jsonl",
        CLAUDE_PREFIX,
        CLAUDE_APPEND,
        id="claude-code",
    ),
    pytest.param(
        "kimi-code",
        "workdir-key/sess-1/agents/main/wire.jsonl",
        KIMI_PREFIX,
        KIMI_APPEND,
        id="kimi-code",
    ),
)


def _first_wins_harness(source: str, relative_path: str, config: AppConfig, root: Path) -> Harness:
    """A harness under template context, which reads the session's own location.

    Template context renders `[<repo> <branch>] ` from the session metadata, so
    it is the cheapest surface on which a suffix that has forgotten the
    session's repo and branch shows up in committed rows.
    """
    from recall.parsers.claude_code import ClaudeCodeParser
    from recall.parsers.kimi_code import KimiCodeParser

    parser = {"claude-code": ClaudeCodeParser, "kimi-code": KimiCodeParser}[source](roots=(root,))
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=config.embedding.dimensions)
    templated = replace(
        config,
        embedding=replace(
            config.embedding, context=replace(config.embedding.context, mode="template")
        ),
    )
    return Harness(templated, conn, parser, path)


@pytest.mark.parametrize(("source", "relative_path", "prefix", "append"), FIRST_WINS_METADATA)
def test_a_resumed_append_keeps_the_first_wins_metadata_a_full_parse_keeps(
    source: str,
    relative_path: str,
    prefix: bytes,
    append: bytes,
    config: AppConfig,
    tmp_path: Path,
) -> None:
    """A suffix must not re-elect the session's model, cwd, repo or branch.

    The stored value has to be the *file's* first, not the suffix's first, or
    the committed row depends on where the daemon happened to checkpoint.
    """
    root = tmp_path / "roots"
    root.mkdir()
    harness = _first_wins_harness(source, relative_path, config, root)
    try:
        harness.path.write_bytes(prefix)
        assert harness.reconcile() is False
        assert harness.stored_checkpoint() is not None

        full = prefix + append
        expected = _reference(harness, full)

        harness.path.write_bytes(full)
        assert harness.reconcile() is True

        assert harness.rows()["session_state"] == expected["session_state"]
        assert harness.rows() == expected
    finally:
        harness.conn.close()


# --- Durable explicit-reparse intent (REQ-INDEX-025) ------------------------


def test_an_explicit_reparse_clears_the_proof_without_moving_the_prefix(
    harness: Harness,
) -> None:
    """The reparse statement may clear the proof, never create or advance one.

    `desired_generation + 1` is all an explicit reparse leaves behind, so the
    committed offset and digest must survive it untouched -- a later turn still
    needs them to tell an append from a rewrite.
    """
    _seed_resumable_checkpoint(harness)
    before = harness.item

    harness.catalog.force_reconcile(harness.source, harness.key_path)

    after = harness.item
    assert harness.stored_checkpoint() is None
    assert after.committed_offset == before.committed_offset
    assert after.committed_prefix_sha256 == before.committed_prefix_sha256
    assert after.desired_generation == before.desired_generation + 1


def test_an_explicit_reparse_outlives_the_request_that_recorded_it(harness: Harness) -> None:
    """A drain turn carrying no request object must still rebuild the history.

    `--full --since 30d` hands its backlog to the shared drain, which pops the
    in-memory request before the source is served, and a daemon restart loses
    the whole dict. Only the catalog can carry the intent across either, and a
    source that also grew raises exactly the desired generation an ordinary
    append raises.
    """
    _seed_resumable_checkpoint(harness)
    harness.path.write_bytes(FULL)
    expected = _reference(harness, FULL)
    harness.catalog.force_reconcile(harness.source, harness.key_path)

    assert harness.drain() is False

    assert harness.rows() == expected
    assert harness.item.current
    assert harness.item.committed_offset == len(FULL)


def test_an_inflight_append_cannot_restore_a_proof_a_reparse_revoked(harness: Harness) -> None:
    """A suffix prepared before `--full` cannot erase the durable reparse intent."""
    _seed_resumable_checkpoint(harness)
    harness.path.write_bytes(FULL)
    item = observe_path(
        harness.parser, capture_path(harness.parser, harness.path), conn=harness.conn
    )
    prepared = prepare_raw_sources((item,), {harness.source: harness.parser})
    assert prepared[0].result is not None and not prepared[0].result.is_full_parse

    harness.catalog.force_reconcile(harness.source, harness.key_path)
    assert harness.stored_checkpoint() is None
    commit_prepared_raw_sources(prepared, harness.config, conn=harness.conn)
    assert harness.stored_checkpoint() is None
    assert not harness.item.current

    final = FULL + _event(
        {"type": "agent_message", "message": "A later append."}, at="2026-02-01T09:00:09Z"
    )
    expected = _reference(harness, final)
    harness.path.write_bytes(final)

    assert harness.drain() is False
    assert harness.rows() == expected
    assert harness.item.current


def test_an_append_with_no_reparse_request_still_resumes(harness: Harness) -> None:
    """The durable clear is scoped to the reparse, not to every pending source."""
    _seed_resumable_checkpoint(harness)
    harness.path.write_bytes(FULL)

    assert harness.drain() is True


# --- A suffix with nothing to append to (REQ-INDEX-025) ---------------------


def _purge_session_rows(harness: Harness) -> None:
    """Lose the session while the catalog keeps its row and its proof.

    A restored backup, a partial repair or a manual purge leaves exactly this:
    `source_files` still says the prefix is committed and still carries a valid
    checkpoint, but nothing holds the rows that checkpoint describes.
    """
    for table in (
        "message_state",
        "tool_results",
        "tool_calls",
        "session_stop_markers",
        "messages",
        "session_state",
        "sessions",
    ):
        harness.conn.execute(f"DELETE FROM {table}")


def test_a_resume_onto_a_lost_session_rebuilds_it(harness: Harness) -> None:
    """Losing the session must cost a reparse, not the prefix.

    Before raw reconciliation could resume, this condition healed itself: the
    full parse found no rows and inserted the whole session. A checkpoint must
    not take that away -- the append would land the suffix under a session that
    does not exist and acknowledge the offset, so no later pass ever rebuilds
    the missing history.
    """
    _seed_resumable_checkpoint(harness)
    _purge_session_rows(harness)
    expected = _reference(harness, FULL)

    harness.path.write_bytes(FULL)
    harness.reconcile()

    # The committed rows are the contract: a suffix appended onto nothing would
    # hold only the tail, with the prefix unreachable behind the new offset.
    assert harness.rows() == expected
    assert harness.item.current
    assert harness.item.committed_offset == len(FULL)


def test_the_incremental_write_refuses_a_session_it_cannot_find(harness: Harness) -> None:
    """Defence in depth for every caller of the append path.

    The merge is `UPDATE session_state ... WHERE session_id = ?`. A zero-row
    match is not an append: it means the row the suffix belongs to is gone, so
    the turn has to fail and leave the catalog unacknowledged rather than
    insert messages under a parent that is not there.
    """
    from recall.core.models import NormalizationCheckpoint
    from recall.services.indexer import _incremental_write_session

    stored = _seed_resumable_checkpoint(harness)
    checkpoint = NormalizationCheckpoint.decode(stored)
    assert checkpoint is not None
    harness.path.write_bytes(FULL)
    suffix = harness.parser.parse(
        harness.path,
        offset=checkpoint.offset,
        message_idx_base=checkpoint.message_idx_base,
        orphan_tool_call_idx_base=checkpoint.orphan_tool_call_idx_base,
        resume_state=checkpoint.adapter_state,
    )
    assert suffix.session.messages, "the suffix must carry rows the write could strand"
    harness.conn.execute("DELETE FROM session_state")
    before = harness.conn.execute("SELECT COUNT(*) FROM messages").fetchone()

    with pytest.raises(RuntimeError, match="session"):
        _incremental_write_session(
            harness.conn,
            suffix.session,
            last_byte_offset=suffix.next_byte_offset,
            fts_fields=harness.config.fts.fields,
            tail_facts=suffix.tail_facts,
        )

    assert harness.conn.execute("SELECT COUNT(*) FROM messages").fetchone() == before
