"""Input-equivalence and preparation floors for REQ-RECON-004/006/008."""

from __future__ import annotations

import json
from pathlib import Path

from recall.core.models import Message, Session, TailFacts
from recall.core.types import Role, Source
from recall.services import indexer
from recall.services.watcher import index_single_session
from test_source_reconciliation import _message_rows, _runtime


def test_positional_insertion_preserves_equivalent_content_embedding(tmp_path: Path) -> None:
    path, config, conn, parser = _runtime(tmp_path)
    try:
        message_id = _message_rows(conn)[0][0]
        conn.execute(
            "INSERT INTO message_embeddings(message_id, content_embedding) VALUES (?, ?)",
            [message_id, [0.25] * 384],
        )
        lines = path.read_text().splitlines()
        inserted = {"type": "event_msg", "payload": {"type": "user_message", "message": "Earlier"}}
        path.write_text("\n".join([lines[0], json.dumps(inserted), *lines[1:]]) + "\n")

        index_single_session(path, parser, config, conn=conn)

        row = conn.execute(
            "SELECT m.idx, me.content_embedding[1] FROM messages m "
            "JOIN message_state ms ON ms.message_id=m.id "
            "LEFT JOIN message_embeddings me ON me.message_id=m.id WHERE ms.content='Hi'"
        ).fetchone()
        assert row == (1, 0.25)
        assert (
            conn.execute(
                "SELECT me.content_embedding FROM message_embeddings me "
                "JOIN message_state ms ON ms.message_id=me.message_id WHERE ms.content='Earlier'"
            ).fetchone()
            is None
        )
    finally:
        conn.close()


def test_changed_tool_input_invalidates_legacy_context_for_full_document(tmp_path: Path) -> None:
    path, _config, conn, parser = _runtime(tmp_path)
    try:
        session = parser.parse(path).session
        session.source = Source.CLAUDE_CODE
        session.messages[0].context_text = "summary includes old tool arguments"
        session.messages[0].context_mode = "llm-codex"
        indexer._write_session(conn, session, tail_facts=parser.parse(path).tail_facts)
        rewritten = session.model_copy(deep=True)
        rewritten.messages[0].context_text = ""
        rewritten.messages[0].context_mode = "off"
        rewritten.messages[-1].tool_calls[0].tool_input = {"command": "different arguments"}

        assert indexer._apply_stored_contexts(conn, rewritten) == 0
        assert rewritten.messages[0].context_text == ""
    finally:
        conn.close()


def test_complete_append_prefix_advances_even_while_next_record_is_torn(tmp_path: Path) -> None:
    path, config, conn, parser = _runtime(tmp_path)
    try:
        complete = {"type": "event_msg", "payload": {"type": "user_message", "message": "Next"}}
        with path.open("ab") as stream:
            stream.write(json.dumps(complete).encode() + b"\n")
            boundary = stream.tell()
            stream.write(b'{"type":')

        index_single_session(path, parser, config, conn=conn)

        assert [row[2] for row in _message_rows(conn)] == ["Hi", "Hello", "From legacy", "Next"]
        assert conn.execute("SELECT last_byte_offset FROM session_state").fetchone() == (boundary,)
    finally:
        conn.close()


def test_a_full_reparse_of_an_appended_session_keeps_stored_llm_context(tmp_path: Path) -> None:
    """Appending is not a changed conversation (REQ-CTX-004, REQ-INDEX-027).

    The suffix path never touches stored rows, and a moved transcript carries
    its context when the old conversation is a prefix of the new one. A full
    reparse of the same appended bytes -- a declined checkpoint, a parser
    revision, a restored database -- must keep it too, not re-render it.
    """
    path, _config, conn, parser = _runtime(tmp_path)
    try:
        session = parser.parse(path).session
        for message in session.messages:
            message.context_text = f"[summary {message.idx}] "
            message.context_mode = "llm-codex"
        indexer._write_session(conn, session, tail_facts=parser.parse(path).tail_facts)
        appended = {"type": "event_msg", "payload": {"type": "user_message", "message": "Later"}}
        with path.open("a") as stream:
            stream.write(json.dumps(appended) + "\n")
        reparsed = parser.parse(path).session

        assert indexer._apply_stored_contexts(conn, reparsed) == len(session.messages)
        assert [
            (m.context_mode, m.context_text) for m in reparsed.messages[: len(session.messages)]
        ] == [("llm-codex", f"[summary {m.idx}] ") for m in session.messages]
        assert reparsed.messages[-1].context_text == ""
    finally:
        conn.close()


def test_a_reparse_that_completes_the_last_stored_message_keeps_earlier_context(
    tmp_path: Path,
) -> None:
    """An open step stored mid-stream is amended in place by the records after it.

    A Kimi assistant step whose thinking part was captured before its text part
    is stored with thinking only; the next parse fills in the text at the same
    index. That completes the conversation the stored summaries describe rather
    than changing it, so every earlier summary is kept and only the amended
    message is rendered again.
    """
    _path, _config, conn, _parser = _runtime(tmp_path)
    try:
        stored = Session(
            id="open-step",
            source=Source.KIMI_CODE,
            source_path="/k/wire.jsonl",
            file_mtime=1,
            file_size=1,
            messages=[
                Message(id="m0", session_id="open-step", idx=0, role=Role.USER, content="question"),
                Message(
                    id="m1", session_id="open-step", idx=1, role=Role.ASSISTANT, content="answer"
                ),
                Message(
                    id="m2", session_id="open-step", idx=2, role=Role.ASSISTANT, thinking="planning"
                ),
            ],
        )
        for message in stored.messages:
            message.context_text = f"[summary {message.idx}] "
            message.context_mode = "llm-codex"
        indexer._write_session(conn, stored, tail_facts=TailFacts())
        reparsed = stored.model_copy(deep=True)
        for message in reparsed.messages:
            message.context_text, message.context_mode = "", "off"
        reparsed.messages[2].content = "the finished text"
        reparsed.messages.append(
            Message(id="m3", session_id="open-step", idx=3, role=Role.USER, content="thanks")
        )

        assert indexer._apply_stored_contexts(conn, reparsed) == 2
        assert [(m.context_mode, m.context_text) for m in reparsed.messages] == [
            ("llm-codex", "[summary 0] "),
            ("llm-codex", "[summary 1] "),
            ("off", ""),
            ("off", ""),
        ]
    finally:
        conn.close()


def test_a_reparse_that_changes_an_earlier_message_keeps_no_context(tmp_path: Path) -> None:
    """Only the trailing stored message may differ; an earlier change keeps nothing."""
    _path, _config, conn, _parser = _runtime(tmp_path)
    try:
        stored = Session(
            id="edited",
            source=Source.KIMI_CODE,
            source_path="/k/edited.jsonl",
            file_mtime=1,
            file_size=1,
            messages=[
                Message(id=f"e{i}", session_id="edited", idx=i, role=Role.USER, content=f"m{i}")
                for i in range(3)
            ],
        )
        for message in stored.messages:
            message.context_text, message.context_mode = "[s] ", "llm-codex"
        indexer._write_session(conn, stored, tail_facts=TailFacts())
        reparsed = stored.model_copy(deep=True)
        for message in reparsed.messages:
            message.context_text, message.context_mode = "", "off"
        reparsed.messages[1].content = "edited"

        assert indexer._apply_stored_contexts(conn, reparsed) == 0
    finally:
        conn.close()
