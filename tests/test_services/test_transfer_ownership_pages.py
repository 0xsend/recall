"""Ownership application consumes a bounded page, regardless of session size."""

from __future__ import annotations

import json
from typing import cast

import duckdb
import pyarrow as pa
import pytest
from recall.core.ids import message_id, tool_call_id
from recall.services.transfers import (
    PAGE_ROWS,
    OwnershipCursor,
    TransferDriver,
)

from tests.test_services.test_duplicate_transfers import _advance
from tests.test_services.test_duplicate_transfers import pair as pair


class ObservedConnection:
    def __init__(self, conn, profile):
        self.conn = conn
        self.profile = profile
        self.rows_read = 0
        self.application_rows = 0
        self.widest_operation = None

    def __getattr__(self, name):
        return getattr(self.conn, name)

    def record(self):
        if self.profile.exists():
            data = json.loads(self.profile.read_text())
            pending = list(data.get("children", []))
            while pending:
                operator = pending.pop()
                width = max(operator["operator_cardinality"], operator["operator_rows_scanned"])
                if width > self.application_rows:
                    self.application_rows = width
                    self.widest_operation = (
                        operator["operator_type"],
                        operator["extra_info"],
                        data["query_name"][:120],
                    )
                pending.extend(operator["children"])

    def execute(self, *args, **kwargs):
        result = self.conn.execute(*args, **kwargs)
        self.record()
        return ObservedRows(result, self)


class ObservedRows:
    def __init__(self, result, observation):
        self.result = result
        self.observation = observation

    def fetchall(self):
        rows = self.result.fetchall()
        self.observation.record()
        self.observation.rows_read += len(rows)
        return rows

    def fetchone(self):
        row = self.result.fetchone()
        self.observation.record()
        self.observation.rows_read += int(row is not None)
        return row


@pytest.mark.parametrize("size", [PAGE_ROWS, PAGE_ROWS * 4])
@pytest.mark.parametrize("family", ["messages", "attached", "orphans"])
def test_one_ownership_acquisition_consumes_only_its_page(pair, tmp_path, size, family):
    conn, first, second, predecessor, survivor, _ = pair
    second.reconcile()
    conn.begin()
    for sid in (predecessor, survivor):
        messages = [(message_id(sid, index), sid, index) for index in range(4, size)]
        conn.register(
            "fixture_messages",
            pa.table(
                {
                    "id": [row[0] for row in messages],
                    "session_id": [row[1] for row in messages],
                    "idx": [row[2] for row in messages],
                }
            ),
        )
        conn.execute("INSERT INTO messages (id,session_id,idx) SELECT * FROM fixture_messages")
        conn.execute(
            "INSERT INTO message_state (message_id,role,content) "
            "SELECT id,'user','page' FROM fixture_messages"
        )
        conn.unregister("fixture_messages")
        tools: list[tuple[str, str, str | None, int]] = [
            (tool_call_id(parent, 0), sid, parent, 0) for parent, _, _ in messages
        ]
        # Many tools on the first message also expose unnecessary tool payload reads.
        tools.extend(
            (tool_call_id(message_id(sid, 0), index), sid, message_id(sid, 0), index)
            for index in range(size)
        )
        if family == "orphans":
            tools.extend(
                (tool_call_id(None, index, sid), sid, None, index) for index in range(size)
            )
        conn.register(
            "fixture_tools",
            pa.table(
                {
                    "id": [row[0] for row in tools],
                    "session_id": [row[1] for row in tools],
                    "message_id": [row[2] for row in tools],
                    "idx": [row[3] for row in tools],
                    "embedding": pa.array(
                        [[1.0 if sid == survivor else 2.0] * 384 for _ in tools],
                        type=pa.list_(pa.float32()),
                    ),
                }
            ),
        )
        conn.execute(
            "INSERT INTO tool_calls (id,session_id,message_id,idx,tool_name) "
            "SELECT id,session_id,message_id,idx,'Bash' FROM fixture_tools"
        )
        conn.execute("INSERT INTO tool_call_embeddings SELECT id,embedding FROM fixture_tools")
        conn.unregister("fixture_tools")
        conn.execute(
            "UPDATE session_state SET message_count=?,tool_count=? WHERE session_id=?",
            [size, 1 + len(tools), sid],
        )
    conn.execute(
        "UPDATE duplicate_transfers SET survivor_message_count=?,survivor_tool_count=?,"
        "context_message_bound=?",
        [size, 1 + len(tools), size],
    )
    conn.commit()
    _advance(conn, first.config, "owning")
    driver = TransferDriver(transfer_id=survivor, cursor=OwnershipCursor(family=family))
    driver.prepare(conn)
    profile = tmp_path / "ownership-profile.json"
    conn.execute("PRAGMA enable_profiling='json'")
    conn.execute("SET profiling_output=?", [str(profile)])
    observed = ObservedConnection(conn, profile)
    try:
        assert driver.turn(cast(duckdb.DuckDBPyConnection, observed), first.config)
        cursor = driver.cursor
        assert cursor is not None
    finally:
        conn.execute("PRAGMA disable_profiling")
    # Read results and every materialized application relation stay page-sized.
    # These measure executed work, rather than statement spelling or elapsed time.
    assert observed.rows_read <= PAGE_ROWS * 6 + 32
    assert observed.application_rows <= PAGE_ROWS, observed.widest_operation
    if family == "messages":
        assert cursor.messages == PAGE_ROWS
    else:
        assert (cursor.attached if family == "attached" else cursor.orphans) == PAGE_ROWS
        carried = conn.execute(
            "SELECT count(*) FROM tool_call_embeddings e JOIN tool_calls t ON t.id=e.tool_call_id "
            "WHERE t.session_id=? AND e.bash_embedding[1]=1.0",
            [predecessor],
        ).fetchone()
        assert carried == (PAGE_ROWS,)


@pytest.mark.parametrize("family", ["messages", "attached", "orphans"])
def test_ownership_uses_stored_identities_and_retires_source(pair, family):
    conn, first, second, predecessor, survivor, _ = pair
    second.reconcile()
    identities = {}
    for sid in (survivor, predecessor):
        if family == "messages":
            row = conn.execute(
                "SELECT id FROM messages WHERE session_id=? AND idx=0", [sid]
            ).fetchone()
            assert row is not None
            old_id = row[0]
            stored_id = f"historical-message-{sid}"
            conn.execute("UPDATE messages SET id=? WHERE id=?", [stored_id, old_id])
            conn.execute(
                "UPDATE message_state SET message_id=? WHERE message_id=?", [stored_id, old_id]
            )
            conn.execute(
                "UPDATE tool_calls SET message_id=? WHERE message_id=?", [stored_id, old_id]
            )
        elif family == "attached":
            row = conn.execute("SELECT id FROM tool_calls WHERE session_id=?", [sid]).fetchone()
            assert row is not None
            old_id = row[0]
            stored_id = f"historical-attached-{sid}"
            conn.execute("UPDATE tool_calls SET id=? WHERE id=?", [stored_id, old_id])
            for table in ("tool_results", "tool_use_ids"):
                conn.execute(
                    f"UPDATE {table} SET tool_call_id=? WHERE tool_call_id=?", [stored_id, old_id]
                )
        else:
            stored_id = f"historical-orphan-{sid}"
            conn.execute(
                "INSERT INTO tool_calls (id,session_id,idx,tool_name,tool_input) "
                "VALUES (?,?,0,'Bash','{\"command\":\"echo orphan\"}')",
                [stored_id, sid],
            )
            conn.execute(
                "UPDATE session_state SET tool_count=tool_count+1 WHERE session_id=?", [sid]
            )
        identities[sid] = stored_id
    if family == "orphans":
        conn.execute("UPDATE duplicate_transfers SET survivor_tool_count=survivor_tool_count+1")
    companion_table, key, vector_column, owner_table = (
        ("message_embeddings", "message_id", "content_embedding", "messages")
        if family == "messages"
        else ("tool_call_embeddings", "tool_call_id", "bash_embedding", "tool_calls")
    )
    conn.execute(
        f"INSERT INTO {companion_table} ({key},{vector_column}) VALUES (?,?)",
        [identities[survivor], [0.25] * 384],
    )
    _advance(conn, first.config, "owning")
    driver = TransferDriver(transfer_id=survivor, cursor=OwnershipCursor(family=family))
    driver.prepare(conn)
    assert driver.turn(conn, first.config)
    assert conn.execute(
        f"SELECT {vector_column}[1] FROM {companion_table} WHERE {key}=?",
        [identities[predecessor]],
    ).fetchone() == (0.25,)
    assert (
        conn.execute(
            f"SELECT e.{key} FROM {companion_table} e ANTI JOIN {owner_table} o ON o.id=e.{key}"
        ).fetchall()
        == []
    )
    # Restart the in-memory pass, replay ownership and drive every shrink/finish turn.
    _advance(conn, first.config)
    assert conn.execute("SELECT id FROM sessions WHERE id=?", [survivor]).fetchone() is None
    assert conn.execute(f"SELECT {key},{vector_column}[1] FROM {companion_table}").fetchall() == [
        (identities[predecessor], 0.25)
    ]
    assert conn.execute(
        f"SELECT id FROM {owner_table} WHERE id=?", [identities[predecessor]]
    ).fetchone() == (identities[predecessor],)
