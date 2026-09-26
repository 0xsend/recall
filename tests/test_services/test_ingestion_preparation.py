"""Input-equivalence and preparation floors for REQ-RECON-004/006/008."""

from __future__ import annotations

import json
from pathlib import Path

from recall.core.types import Source
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
