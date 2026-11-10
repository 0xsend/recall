"""Durable copy-then-delete transfer outcomes (REQ-INDEX-027)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.db.schema import ensure_schema
from recall.parsers.claude_code import ClaudeCodeParser
from recall.services.coordinator import capture_path, observe_path

from tests.test_services.test_checkpoint_resume_fallbacks import Harness
from tests.test_services.test_moves import LINES, _walk


def _transfers(conn):
    from recall.services.transfers import transfer_catalog

    result = transfer_catalog(conn)
    assert result is not None
    return result


def _fresh_process(database, action, path):
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "tests.transfer_crash_worker", str(database), action, str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _kill_worker(database, action, when, log):
    import json
    import selectors
    import subprocess
    import sys

    with log.open("w") as error:
        process = subprocess.Popen(
            [sys.executable, "-m", "tests.transfer_crash_worker", str(database), action, when],
            stdout=subprocess.PIPE,
            stderr=error,
            text=True,
        )
        assert process.stdout is not None
        try:
            with selectors.DefaultSelector() as events:
                events.register(process.stdout, selectors.EVENT_READ)
                assert events.select(20), "transaction boundary not reached"
                line = process.stdout.readline()
            assert line and json.loads(line) == {"boundary": when}, log.read_text()
            process.kill()
            assert process.wait(timeout=10) == -9
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            process.stdout.close()


def _repair_keywords(conn, config):
    from contextlib import closing

    from recall.db.fts_sidecar import open_sidecar, sidecar_path
    from recall.services.fts_sidecar_reconcile import has_pending, repair_pending_batch

    with closing(open_sidecar(sidecar_path(config.data_dir))) as sidecar:
        for _ in range(100):
            if not has_pending(conn):
                break
            result = repair_pending_batch(conn, sidecar, fts_fields=config.fts.fields)
            assert result.error is None
        else:
            pytest.fail("keyword intents did not drain")
        expected = {row[0] for row in conn.execute("SELECT id FROM messages").fetchall()}
        assert {
            row[0] for row in sidecar.execute("SELECT message_id FROM message_fts_rowid")
        } == expected


@pytest.fixture
def pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    from recall.db.fts_sidecar import ensure_sidecar

    ensure_sidecar(config)
    database = tmp_path / "transfer.duckdb"
    conn = duckdb.connect(str(database))
    try:
        ensure_schema(conn)
        parser = ClaudeCodeParser(roots=(tmp_path,))
        original = tmp_path / "original.jsonl"
        moved = tmp_path / "moved.jsonl"
        original.write_text("".join(LINES[:3]))
        first = Harness(config, conn, parser, original)
        first.reconcile()
        row = conn.execute("SELECT id FROM sessions").fetchone()
        assert row is not None
        predecessor = row[0]
        moved.write_text("".join(LINES))
        second = Harness(config, conn, parser, moved)
        second.reconcile()
        row = conn.execute("SELECT id FROM sessions WHERE source_path = ?", [str(moved)]).fetchone()
        assert row is not None
        survivor = row[0]
        original.unlink()
        _walk(conn, tmp_path, parser=parser, drain_transfers=False)
        # The surviving copy must stay readable until coverage and bounded shrink finish.
        assert conn.execute("SELECT id FROM sessions ORDER BY id").fetchall() == sorted(
            [(predecessor,), (survivor,)]
        )
        yield conn, first, second, predecessor, survivor, database
    finally:
        conn.close()


def test_rebind_preserves_both_histories_and_parks_survivor(pair):
    conn, _, second, predecessor, survivor, _ = pair
    assert conn.execute("SELECT source_path FROM sessions WHERE id = ?", [survivor]).fetchone() == (
        f"recall-transfer:{survivor}:{second.path}",
    )
    assert conn.execute(
        "SELECT source_path FROM sessions WHERE id = ?", [predecessor]
    ).fetchone() == (str(second.path),)
    assert conn.execute("SELECT phase FROM duplicate_transfers").fetchone() == ("catch_up",)


@pytest.mark.parametrize("restart", [False, True])
def test_clean_truncation_after_finish_deletes_rows(pair, restart):
    from recall.services.transfers import TransferDriver

    conn, first, second, predecessor, survivor, database = pair
    driver = TransferDriver()
    second.reconcile()
    driver.prepare(conn)
    for _ in range(100):
        driver.prepare(conn)
        driver.turn(conn, first.config)
        if conn.execute("SELECT phase FROM duplicate_transfers").fetchone() == ("shrinking",):
            break
    else:
        pytest.fail("transfer never reached shrink")
    second.path.write_text("".join(LINES[:1]))
    second.reconcile(full=True)
    before = conn.execute(
        "SELECT message_count FROM session_state WHERE session_id = ?", [predecessor]
    ).fetchone()[0]
    assert before > 1
    assert "transfer" in second.item.last_error.lower()
    for _ in range(100):
        driver.turn(conn, first.config)
        if conn.execute("SELECT 1 FROM duplicate_transfers").fetchone() is None:
            break
    else:
        pytest.fail("refused rewrite stopped shrink")
    assert conn.execute("SELECT id FROM sessions WHERE id = ?", [survivor]).fetchone() is None
    if restart:
        conn.close()
        result = _fresh_process(database, "ordinary_retry", second.path)
        assert result == {"message_counts": [1], "phases": []}
        with duckdb.connect(str(database)) as reopened:
            row = reopened.execute(
                "SELECT message_count FROM session_state WHERE session_id = ?", [predecessor]
            ).fetchone()
            assert row is not None
            after = row[0]
    else:
        second.reconcile(full=True)
        after = conn.execute(
            "SELECT message_count FROM session_state WHERE session_id = ?", [predecessor]
        ).fetchone()[0]
    assert after == 1
    assert before - after > 0


@pytest.mark.parametrize("mismatch", [False, True])
def test_publication_discards_stale_coverage_success_and_mismatch(pair, mismatch):
    from recall.services.transfers import record_coverage, verify_pass

    conn, _, second, _, _, _ = pair
    second.reconcile()
    catalog = _transfers(conn)
    assert catalog is not None
    transfer = catalog.load()[0]
    if mismatch:
        conn.execute(
            "UPDATE tool_calls SET skill_name = 'drift' WHERE session_id = ?",
            [transfer.predecessor_id],
        )
    result = verify_pass(conn, transfer)
    second.catalog.force_reconcile(second.source, str(second.path))
    second.reconcile(full=True)
    assert not record_coverage(conn, result)
    assert catalog.load()[0].phase == "catch_up"
    assert catalog.load()[0].stuck_reason is None


@pytest.mark.parametrize("pages", [1, 2, 4])
@pytest.mark.parametrize("mutation", ["truncate", "protected_rewrite", "tail_rewrite"])
def test_rewrite_gate_through_partial_shrink(tmp_path, monkeypatch, pages, mutation):
    import json

    from recall.core.types import default_session_host
    from recall.db.fts_sidecar import ensure_sidecar
    from recall.services.moves import plan_moved_duplicates
    from recall.services.transfers import TransferDriver, rebind

    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    ensure_sidecar(config)
    parser = ClaudeCodeParser(roots=(tmp_path,))
    original, moved = tmp_path / "original.jsonl", tmp_path / "moved.jsonl"
    records = [
        json.dumps(
            {
                "type": "user",
                "uuid": f"u{idx}",
                "sessionId": "shrink-fixture",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": f"message {idx}"},
            }
        )
        + "\n"
        for idx in range(2050)
    ]
    original.write_text(records[0])
    moved.write_text("".join(records[:2049]))
    database = tmp_path / "transfer.duckdb"
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        Harness(config, conn, parser, original).reconcile()
        harness = Harness(config, conn, parser, moved)
        harness.reconcile()
        original.unlink()
        plan = plan_moved_duplicates(conn, host=default_session_host())[0]
        rebind(conn, plan)
        moved.write_text("".join(records))
        for _ in range(30):
            harness.reconcile()
            if harness.item.current:
                break
        else:
            pytest.fail("catch-up did not finish")
        _advance(conn, config, "shrinking")
        driver = TransferDriver()
        for _ in range(pages):
            driver.turn(conn, config)
        assert conn.execute(
            "SELECT message_count FROM session_state WHERE session_id=?", [plan.successor_id]
        ).fetchone() == (2049 - pages * 512,)
        transfer_before = conn.execute("SELECT * FROM duplicate_transfers").fetchall()
        if mutation == "truncate":
            moved.write_text(records[0])
        else:
            idx = 17 if mutation == "protected_rewrite" else 2049
            amended = records.copy()
            amended[idx] = amended[idx].replace(f"message {idx}", "amended")
            moved.write_text("".join(amended))
        harness.reconcile(full=True)
        if mutation != "tail_rewrite":
            assert "transfer" in (harness.item.last_error or "").lower()
            assert conn.execute("SELECT * FROM duplicate_transfers").fetchall() == transfer_before
        else:
            assert harness.item.last_error is None
        contents = [
            row[0]
            for row in conn.execute(
                """SELECT ms.content FROM messages m JOIN message_state ms ON ms.message_id=m.id
                    WHERE m.session_id=? ORDER BY m.idx""",
                [plan.predecessor_id],
            ).fetchall()
        ]
        expected = [f"message {idx}" for idx in range(2050)]
        if mutation == "tail_rewrite":
            expected[2049] = "amended"
        assert contents == expected
    result = _fresh_process(database, "ordinary_retry", moved)
    assert result["phases"] == ["shrinking"]
    with duckdb.connect(str(database)) as conn:
        _advance(conn, config)
        assert conn.execute("SELECT id FROM sessions").fetchall() == [(plan.predecessor_id,)]
        assert conn.execute("SELECT message_count FROM session_state").fetchone() == (2050,)
    result = _fresh_process(database, "ordinary_retry", moved)
    assert result == {"message_counts": [1 if mutation == "truncate" else 2050], "phases": []}
    with duckdb.connect(str(database)) as conn:
        final = [
            row[0]
            for row in conn.execute(
                "SELECT content FROM messages m JOIN message_state ms ON ms.message_id=m.id "
                "ORDER BY idx"
            ).fetchall()
        ]
        if mutation == "truncate":
            assert final == ["message 0"]
        else:
            expected[17 if mutation == "protected_rewrite" else 2049] = "amended"
            assert final == expected


@pytest.mark.parametrize("boundary", ["first", "witness", "after_witness"])
@pytest.mark.parametrize("when", ["before", "after"])
def test_witness_survives_chunk_crashes_and_fresh_processes(tmp_path, monkeypatch, boundary, when):
    import json
    import time

    from recall.core.types import default_session_host
    from recall.db.fts_sidecar import ensure_sidecar
    from recall.db.source_files import SourceCatalog
    from recall.services.moves import plan_moved_duplicates
    from recall.services.transfers import rebind

    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    ensure_sidecar(config)
    parser = ClaudeCodeParser(roots=(tmp_path,))
    original, moved = tmp_path / "original.jsonl", tmp_path / "moved.jsonl"
    records = [
        json.dumps(
            {
                "type": "user",
                "uuid": f"u{idx}",
                "sessionId": "witness-fixture",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user", "content": f"message {idx}"},
            }
        )
        + "\n"
        for idx in range(2561)
    ]
    original.write_text(records[0])
    moved.write_text("".join(records[:1025]))
    database = tmp_path / "transfer.duckdb"
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        Harness(config, conn, parser, original).reconcile()
        harness = Harness(config, conn, parser, moved)
        harness.reconcile()
        original.unlink()
        plan = plan_moved_duplicates(conn, host=default_session_host())[0]
        rebind(conn, plan)
        moved.write_text("".join(records))
        for _ in range({"first": 0, "witness": 1, "after_witness": 2}[boundary]):
            assert harness.reconcile()
        transfer = _transfers(conn).load()[0]
        epoch = transfer.witness_epoch
        assert (epoch is not None) == (boundary == "after_witness")
        seq = transfer.predecessor_publish_seq
        offset = harness.item.committed_offset
    _kill_worker(database, "raw", when, tmp_path / "raw-boundary.log")
    with duckdb.connect(str(database)) as conn:
        transfer = _transfers(conn).load()[0]
        assert transfer.predecessor_publish_seq == seq + (when == "after")
        item = SourceCatalog(conn, clock=time.time).get("claude_code", str(moved))
        assert item is not None
        assert (item.committed_offset > offset) == (when == "after")
        assert conn.execute(
            "SELECT message_count FROM session_state WHERE session_id=?", [plan.successor_id]
        ).fetchone() == (1025,)
    for _ in range(8):
        with duckdb.connect(str(database)) as conn:
            transfer = _transfers(conn).load()[0]
            if epoch is not None:
                assert transfer.witness_epoch == epoch
            elif transfer.witness_epoch is not None:
                epoch = transfer.witness_epoch
            item = SourceCatalog(conn, clock=time.time).get("claude_code", str(moved))
            assert item is not None
            if item.current:
                break
        _fresh_process(database, "raw_chunk", moved)
    else:
        pytest.fail("fresh-process raw chunks did not finish")
    assert epoch is not None
    with duckdb.connect(str(database)) as conn:
        _advance(conn, config)
        assert conn.execute("SELECT id FROM sessions").fetchall() == [(plan.predecessor_id,)]
        assert conn.execute("SELECT message_count FROM session_state").fetchone() == (2561,)


def test_one_refusal_and_restart_polls_do_not_mark_stuck(pair):
    conn, _first, second, _, _, database = pair
    # Unsupported history drops the assistant's tool, while staying longer than the witness.
    second.path.write_text(LINES[0] + '{"type":"future_record","padding":"' + "x" * 20000 + '"}\n')
    second.reconcile(full=True)
    assert (
        conn.execute("SELECT predecessor_refusal_kind FROM duplicate_transfers").fetchone()[0]
        is not None
    )
    conn.close()
    assert _fresh_process(database, "poll", second.path)["phases"] == ["catch_up"]
    with duckdb.connect(str(database)) as reopened:
        assert reopened.execute("SELECT phase FROM duplicate_transfers").fetchone() == ("catch_up",)


@pytest.mark.parametrize(
    "column,value",
    [
        ("tool_name", "different"),
        ("tool_input", '{"command":"different"}'),
        ("bash_command", "false"),
        ("bash_base", "false"),
        ("bash_sub", "different"),
        ("is_compound", True),
        ("agent_id", "different"),
        ("subagent_type", "different"),
        ("subagent_description", "different"),
        ("subagent_model", "different"),
        ("skill_name", "different"),
    ],
)
def test_same_position_orphan_payload_mismatch_never_deletes(pair, column, value):
    from recall.services.transfers import record_coverage, verify_pass

    conn, _, second, predecessor, survivor, _ = pair
    second.reconcile()
    for sid in (predecessor, survivor):
        conn.execute(
            "INSERT INTO tool_calls(id, session_id, idx, tool_name) VALUES (?, ?, 0, 'Bash')",
            [sid + "-orphan", sid],
        )
        conn.execute(
            "UPDATE session_state SET tool_count = tool_count + 1 WHERE session_id = ?", [sid]
        )
    conn.execute("UPDATE duplicate_transfers SET survivor_tool_count = survivor_tool_count + 1")
    conn.execute(f"UPDATE tool_calls SET {column} = ? WHERE id = ?", [value, survivor + "-orphan"])
    catalog = _transfers(conn)
    assert catalog is not None
    result = verify_pass(conn, catalog.load()[0])
    record_coverage(conn, result)
    assert catalog.load()[0].stuck_reason == "normalized_mismatch"
    assert conn.execute(
        "SELECT COUNT(*) FROM tool_calls WHERE id IN (?, ?)",
        [predecessor + "-orphan", survivor + "-orphan"],
    ).fetchone() == (2,)


def _advance(conn, config, phase=None):
    from recall.services.transfers import TransferDriver

    driver = TransferDriver()
    for _ in range(100):
        rows = _transfers(conn).load()
        if not rows or (phase is not None and rows[0].phase == phase):
            return
        driver.prepare(conn)
        driver.turn(conn, config)
    pytest.fail("transfer did not converge")


@pytest.mark.parametrize("phase", ["catch_up", "owning", "shrinking"])
def test_restored_original_has_separate_identity(pair, phase):
    conn, first, second, predecessor, survivor, _ = pair
    if phase != "catch_up":
        second.reconcile()
        _advance(conn, first.config, phase)
    first.path.write_text("".join(LINES[:3]))
    first.reconcile(full=True)
    restored = conn.execute(
        "SELECT id FROM sessions WHERE source_path=?", [str(first.path)]
    ).fetchone()[0]
    assert restored not in (predecessor, survivor)
    assert conn.execute(
        "SELECT source_path FROM sessions WHERE id=?", [predecessor]
    ).fetchone() == (str(second.path),)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("missing", "moved_file_missing"),
        ("truncated", "moved_file_truncated"),
        ("rewritten", "witness_mismatch"),
    ],
)
def test_unproven_input_never_deletes_and_apply_ends(pair, mutation, reason):
    from recall.db.source_files import source_key
    from recall.services.transfers import TransferDriver, apply_transfers

    conn, first, second, predecessor, survivor, _ = pair
    before = conn.execute("SELECT id FROM messages ORDER BY id").fetchall()
    if mutation == "missing":
        second.path.unlink()
        second.catalog.mark_missing([source_key(second.source, str(second.path))])
    elif mutation == "truncated":
        second.path.write_text(LINES[0])
        observe_path(second.parser, capture_path(second.parser, second.path), conn=conn)
    else:
        second.path.write_text("".join(LINES).replace("2026-09-07T", "2026-09-08T"))
        second.reconcile(full=True)
    TransferDriver().turn(conn, first.config)
    assert _transfers(conn).load()[0].stuck_reason == reason
    # A raw rewrite may change P; no transfer step ever deletes S.
    assert conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=?", [survivor]
    ).fetchone() == (4,)
    if mutation != "rewritten":
        assert conn.execute("SELECT id FROM messages ORDER BY id").fetchall() == before
    assert apply_transfers(conn, first.config, []) == 0
    assert apply_transfers(conn, first.config, []) == 0
    assert sorted(row[0] for row in conn.execute("SELECT id FROM sessions").fetchall()) == sorted(
        [predecessor, survivor]
    )


def test_second_real_refusal_after_restart_is_resolvable(pair):
    from recall.services.transfers import apply_transfers, transfer_report

    conn, first, second, predecessor, survivor, database = pair
    second.path.write_text(LINES[0] + '{"type":"future_record","padding":"' + "x" * 20000 + '"}\n')
    before = conn.execute("SELECT id FROM messages ORDER BY id").fetchall()
    second.reconcile(full=True)
    conn.close()
    assert _fresh_process(database, "refusal_retry", second.path)["phases"] == ["stuck"]
    with duckdb.connect(str(database)) as reopened:
        report = transfer_report(reopened)[0]
        assert report["stuck_reason"] == "predecessor_refused"
        assert report["resolution"] == "apply_ends"
        assert apply_transfers(reopened, first.config, []) == 0
        assert apply_transfers(reopened, first.config, []) == 0
        assert reopened.execute("SELECT id FROM messages ORDER BY id").fetchall() == before
        assert reopened.execute(
            "SELECT COUNT(*) FROM sessions WHERE id IN (?,?)", [predecessor, survivor]
        ).fetchone() == (2,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "drift"),
        ("role", "user"),
        ("agent_id", "other"),
        ("timestamp", "2025-01-01"),
        ("tool_input", '{"command":"false"}'),
        ("bash_command", "false"),
        ("skill_name", "drift"),
    ],
)
def test_normalized_drift_keeps_both_histories(pair, field, value):
    from recall.services.transfers import record_coverage, verify_pass

    conn, _, second, predecessor, survivor, _ = pair
    second.reconcile()
    if field in ("content",):
        conn.execute(
            f"""UPDATE message_state SET {field}=? WHERE message_id=(
                SELECT id FROM messages WHERE session_id=? AND idx=0)""",
            [value, survivor],
        )
    elif field in ("role", "agent_id", "timestamp"):
        conn.execute(
            f"UPDATE messages SET {field}=? WHERE session_id=? AND idx=1", [value, survivor]
        ) if field == "agent_id" else conn.execute(
            f"""UPDATE message_state SET {field}=? WHERE message_id=(
                SELECT id FROM messages WHERE session_id=? AND idx=1)""",
            [value, survivor],
        )
    else:
        conn.execute(f"UPDATE tool_calls SET {field}=? WHERE session_id=?", [value, survivor])
    counts = conn.execute(
        "SELECT session_id,message_count,tool_count FROM session_state ORDER BY session_id"
    ).fetchall()
    catalog = _transfers(conn)
    record_coverage(conn, verify_pass(conn, catalog.load()[0]))
    assert catalog.load()[0].stuck_reason == "normalized_mismatch"
    assert (
        conn.execute(
            "SELECT session_id,message_count,tool_count FROM session_state ORDER BY session_id"
        ).fetchall()
        == counts
    )
    assert conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE id IN (?,?)", [predecessor, survivor]
    ).fetchone() == (2,)


@pytest.mark.parametrize("predecessor_companion", [False, True])
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("initial_orphans", [0, 512])
@pytest.mark.parametrize("append_when", ["catch_up", "owning_prepared", "unchanged"])
def test_appended_orphan_invalidates_last_context_through_transfer(
    tmp_path, monkeypatch, predecessor_companion, restart, initial_orphans, append_when
):
    import json
    from dataclasses import replace

    from recall.core.types import default_session_host
    from recall.db.fts_sidecar import ensure_sidecar
    from recall.parsers.codex import CodexParser
    from recall.services.embed_phase import (
        EmbedPhaseState,
        commit_prepared_embed_cycle,
        find_pending_embeds,
        generate_prepared_embed_cycle,
        prepare_embed_cycle,
    )
    from recall.services.enrichment_inputs import select_pending_targets
    from recall.services.moves import plan_moved_duplicates
    from recall.services.transfers import TransferDriver, rebind

    from tests.test_services.test_embed_phase import RecordingBackend, _allow_embedding

    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    ensure_sidecar(config)
    parser = CodexParser(roots=(tmp_path,))
    original, moved = tmp_path / "rollout-original.jsonl", tmp_path / "rollout-moved.jsonl"

    def event(payload, second):
        return (
            json.dumps(
                {
                    "type": "event_msg",
                    "timestamp": f"2026-01-01T00:00:0{second}Z",
                    "payload": payload,
                }
            )
            + "\n"
        )

    head = event({"type": "user_message", "message": "start"}, 0)
    body = event({"type": "agent_message", "message": "body"}, 1)
    orphan = event({"type": "function_call", "name": "Bash", "parameters": {"command": "pwd"}}, 2)
    original.write_text(head)
    moved.write_text(head + body + orphan * initial_orphans)
    database = tmp_path / "transfer.duckdb"
    conn = duckdb.connect(str(database))
    try:
        ensure_schema(conn)
        Harness(config, conn, parser, original).reconcile()
        Harness(config, conn, parser, moved).reconcile()
        original.unlink()
        plan = plan_moved_duplicates(conn, host=default_session_host())[0]
        survivor, predecessor = plan.successor_id, plan.predecessor_id
        old_row = conn.execute(
            "SELECT id FROM messages WHERE session_id=? AND idx=1", [survivor]
        ).fetchone()
        assert old_row is not None
        old_id = old_row[0]
        conn.execute(
            "UPDATE message_state SET context_text='survivor: ',context_mode='llm-local' "
            "WHERE message_id=?",
            [old_id],
        )
        conn.execute("INSERT INTO message_embeddings VALUES (?,?,NULL)", [old_id, [0.5] * 384])
        rebind(conn, plan)
        harness = Harness(config, conn, parser, moved)
        driver = TransferDriver()
        if append_when == "owning_prepared":
            harness.reconcile()
            for _ in range(10):
                if harness.item.current:
                    break
                harness.reconcile()
            else:
                pytest.fail(f"initial catch-up failed: {harness.item}")
            _advance(conn, config, "owning")
            driver.prepare(conn)
        # Only P observes the orphan, beyond the frozen S byte boundary.
        if append_when != "unchanged":
            with moved.open("a") as stream:
                stream.write(orphan)
        harness.reconcile()
        for _ in range(10):
            if harness.item.current:
                break
            harness.reconcile()
        else:
            pytest.fail(f"raw catch-up did not reach the appended orphan: {harness.item}")
        new_row = conn.execute(
            "SELECT id FROM messages WHERE session_id=? AND idx=1", [predecessor]
        ).fetchone()
        assert new_row is not None
        new_id = new_row[0]
        orphans = conn.execute(
            "SELECT idx FROM tool_calls WHERE session_id=? AND message_id IS NULL ORDER BY idx",
            [predecessor],
        ).fetchall()
        assert orphans == [(idx,) for idx in range(initial_orphans + (append_when != "unchanged"))]
        if predecessor_companion:
            conn.execute(
                "UPDATE message_state SET context_text='predecessor: ',context_mode='llm-local' "
                "WHERE message_id=?",
                [new_id],
            )
            conn.execute("INSERT INTO message_embeddings VALUES (?,?,NULL)", [new_id, [0.25] * 384])
        conn.execute(
            "INSERT INTO message_embeddings SELECT id,NULL,NULL FROM messages "
            "WHERE session_id=? AND idx=0",
            [predecessor],
        )
        conn.execute(
            "INSERT INTO tool_call_embeddings SELECT id,NULL FROM tool_calls WHERE session_id=?",
            [predecessor],
        )
        _advance(conn, config, "owning")
        if append_when == "owning_prepared":
            # A publication between proof and ownership must discard the proof.
            assert not driver.turn(conn, config)
            assert conn.execute(
                "SELECT context_text FROM message_state WHERE message_id=?", [new_id]
            ).fetchone() == (("predecessor: ",) if predecessor_companion else ("",))
        if restart:
            conn.close()
            conn = duckdb.connect(str(database))
        _advance(conn, config)
        assert conn.execute("SELECT id FROM sessions").fetchall() == [(predecessor,)]
        unchanged = append_when == "unchanged"
        expected_context = (
            ("survivor: ", "llm-local")
            if unchanged
            else ("predecessor: ", "llm-local")
            if predecessor_companion
            else ("", "off")
        )
        assert (
            conn.execute(
                "SELECT context_text,context_mode FROM message_state WHERE message_id=?", [new_id]
            ).fetchone()
            == expected_context
        )
        companion = conn.execute(
            "SELECT content_embedding,thinking_embedding FROM message_embeddings "
            "WHERE message_id=?",
            [new_id],
        ).fetchall()
        assert companion == (
            [((0.5,) * 384, None)]
            if unchanged
            else [((0.25,) * 384, None)]
            if predecessor_companion
            else []
        ), companion
        pending_expected = not unchanged and not predecessor_companion
        pending = select_pending_targets(conn, predecessor).targets.message_ids
        assert pending == ((new_id,) if pending_expected else ())
        config = replace(config, daemon=replace(config.daemon, embed_idle_session=0))
        backend = RecordingBackend()
        state = EmbedPhaseState(_config=config)
        _allow_embedding(monkeypatch, state, backend)
        prepared = prepare_embed_cycle(config, state, conn=conn)
        generate_prepared_embed_cycle(config, state, prepared)
        assert commit_prepared_embed_cycle(prepared, conn=conn) == int(pending_expected)
        assert backend.calls == ([["body"]] if pending_expected else [])
        assert find_pending_embeds(conn, idle_threshold=0).total == 0
    finally:
        conn.close()


@pytest.mark.parametrize("component", ["content", "thinking"])
@pytest.mark.parametrize("vectors", ["vectors", "null"])
def test_completion_keeps_whole_companion_pending(pair, component, vectors, monkeypatch):
    from recall.services.embed_phase import find_pending_embeds
    from recall.services.enrichment_inputs import select_pending_targets
    from recall.services.transfers import TransferDriver

    conn, first, second, predecessor, survivor, _ = pair
    second.reconcile()
    idx = 3
    old_id = conn.execute(
        "SELECT id FROM messages WHERE session_id=? AND idx=?", [survivor, idx]
    ).fetchone()[0]
    new_id = conn.execute(
        "SELECT id FROM messages WHERE session_id=? AND idx=?", [predecessor, idx]
    ).fetchone()[0]
    conn.execute(
        """UPDATE message_state SET content='old',thinking='old thought',
            context_text='paid: ',context_mode='llm-local' WHERE message_id=?""",
        [old_id],
    )
    conn.execute(
        "UPDATE message_state SET content=?, thinking=? WHERE message_id=?",
        [
            "old extended" if component == "content" else "old",
            "old thought extended" if component == "thinking" else "old thought",
            new_id,
        ],
    )
    vector = [0.5] * 384 if vectors == "vectors" else None
    conn.execute("INSERT INTO message_embeddings VALUES (?,?,?)", [old_id, vector, vector])
    conn.execute(
        """INSERT INTO message_embeddings SELECT id,NULL,NULL FROM messages
        WHERE session_id=? AND id<>?""",
        [predecessor, new_id],
    )
    conn.execute(
        """INSERT INTO tool_call_embeddings SELECT id,NULL FROM tool_calls
        WHERE session_id=?""",
        [predecessor],
    )
    # The covered relation admits completion of S's last message.
    driver = TransferDriver()
    driver.prepare(conn)
    driver.turn(conn, first.config)
    assert conn.execute("SELECT phase FROM duplicate_transfers").fetchone() == ("owning",)
    _advance(conn, first.config)
    assert (
        conn.execute("SELECT * FROM message_embeddings WHERE message_id=?", [new_id]).fetchall()
        == []
    )
    targets = select_pending_targets(conn, predecessor)
    assert new_id in targets.targets.message_ids
    assert predecessor in find_pending_embeds(conn, idle_threshold=0).session_ids

    from dataclasses import replace

    from recall.services.embed_phase import (
        EmbedPhaseState,
        commit_prepared_embed_cycle,
        generate_prepared_embed_cycle,
        prepare_embed_cycle,
    )

    from tests.test_services.test_embed_phase import RecordingBackend, _allow_embedding

    config = replace(first.config, daemon=replace(first.config.daemon, embed_idle_session=0))
    backend = RecordingBackend()
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, backend)
    assert find_pending_embeds(conn, idle_threshold=0).total == 1
    prepared = prepare_embed_cycle(config, state, conn=conn)
    generate_prepared_embed_cycle(config, state, prepared)
    assert commit_prepared_embed_cycle(prepared, conn=conn) == 1
    assert sorted(text for batch in backend.calls for text in batch) == sorted(
        [
            "old extended" if component == "content" else "old",
            "old thought extended" if component == "thinking" else "old thought",
        ]
    )
    assert find_pending_embeds(conn, idle_threshold=0).total == 0


@pytest.mark.parametrize("component", ["content", "thinking"])
@pytest.mark.parametrize("vectors", [False, True])
def test_covered_full_completion_reopens_whole_companion(tmp_path, monkeypatch, component, vectors):
    import json

    from recall.core.types import default_session_host
    from recall.db.fts_sidecar import ensure_sidecar
    from recall.services.moves import plan_moved_duplicates
    from recall.services.transfers import TransferDriver, rebind

    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    ensure_sidecar(config)
    parser = ClaudeCodeParser(roots=(tmp_path,))
    original, moved = tmp_path / "original.jsonl", tmp_path / "moved.jsonl"
    head = (
        json.dumps(
            {
                "type": "user",
                "message": {"role": "user", "content": "start"},
                "uuid": "u",
                "timestamp": "2026-01-01T00:00:00Z",
                "sessionId": "fixture",
            }
        )
        + "\n"
    )

    def body(extended):
        return (
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "a",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "body extended"
                                if extended and component == "content"
                                else "body",
                            },
                            {
                                "type": "thinking",
                                "thinking": "thought extended"
                                if extended and component == "thinking"
                                else "thought",
                            },
                        ],
                    },
                }
            )
            + "\n"
        )

    original.write_text(head)
    moved.write_text(head + body(False))
    with duckdb.connect(str(tmp_path / "transfer.duckdb")) as conn:
        ensure_schema(conn)
        Harness(config, conn, parser, original).reconcile()
        harness = Harness(config, conn, parser, moved)
        harness.reconcile()
        original.unlink()
        plan = plan_moved_duplicates(conn, host=default_session_host())[0]
        old_row = conn.execute(
            "SELECT id FROM messages WHERE session_id=? AND idx=1", [plan.successor_id]
        ).fetchone()
        assert old_row is not None
        old_id = old_row[0]
        vector = [0.5] * 384 if vectors else None
        conn.execute("INSERT INTO message_embeddings VALUES (?,?,?)", [old_id, vector, vector])
        rebind(conn, plan)
        harness.reconcile()
        driver = TransferDriver()
        for _ in range(30):
            driver.prepare(conn)
            driver.turn(conn, config)
            if _transfers(conn).load()[0].phase == "shrinking":
                break
        else:
            pytest.fail("ownership did not finish")
        new_row = conn.execute(
            "SELECT id FROM messages WHERE session_id=? AND idx=1", [plan.predecessor_id]
        ).fetchone()
        assert new_row is not None
        new_id = new_row[0]
        assert conn.execute(
            "SELECT count(*) FROM message_embeddings WHERE message_id=?", [new_id]
        ).fetchone() == (1,)
        moved.write_text(head + body(True))
        harness.reconcile(full=True)
        assert (
            conn.execute("SELECT * FROM message_embeddings WHERE message_id=?", [new_id]).fetchall()
            == []
        )


def test_publication_and_ownership_are_restart_idempotent(pair):
    from recall.services.transfers import TransferDriver

    conn, first, second, predecessor, _survivor, database = pair
    second.reconcile()
    for _ in range(30):
        conn.close()
        conn = duckdb.connect(str(database))
        driver = TransferDriver()
        driver.prepare(conn)
        driver.turn(conn, first.config)
        # Restarting every ownership page restarts the pass, so preserve the
        # in-memory cursor for a full pass before the next restart.
        while _transfers(conn).load() and _transfers(conn).load()[0].phase == "owning":
            driver.prepare(conn)
            driver.turn(conn, first.config)
        if not _transfers(conn).load():
            break
    else:
        pytest.fail("restart did not converge")
    assert conn.execute("SELECT id FROM sessions").fetchall() == [(predecessor,)]
    assert conn.execute("SELECT message_count,tool_count FROM session_state").fetchone() == (4, 1)
    assert not TransferDriver().turn(conn, first.config)
    conn.close()


@pytest.mark.parametrize("surface", ["library", "daemon"])
@pytest.mark.parametrize("late", [False, True])
def test_mixed_recompute_counts_only_eligible_model_calls(pair, monkeypatch, surface, late):
    import asyncio
    from dataclasses import replace

    from recall.core.config import ContextConfig
    from recall.core.ids import message_id
    from recall.services.context_backends import ContextResult
    from recall.services.embed_phase import EmbedPhaseState
    from recall.services.indexer import ContextRun, recompute_context_for_rows
    from recall.services.rpc_server import RpcServer

    from tests.test_services.test_embed_phase import RecordingBackend, _allow_embedding

    conn, first, second, predecessor, _survivor, _ = pair
    second.reconcile()
    # One eligible tail message shares the protected conversation as dependency.
    identifier = message_id(predecessor, 4)
    conn.execute(
        "INSERT INTO messages(id,session_id,idx) VALUES (?,?,4)", [identifier, predecessor]
    )
    conn.execute(
        "INSERT INTO message_state(message_id,role,content) VALUES (?,'user','eligible tail')",
        [identifier],
    )
    conn.execute(
        "UPDATE session_state SET message_count=message_count+1 WHERE session_id=?", [predecessor]
    )
    config = replace(
        first.config,
        embedding=replace(
            first.config.embedding, context=ContextConfig(mode="llm-local", min_chars=0)
        ),
    )
    observed = []

    class Model:
        def is_available(self):
            return True

        def generate_prefix(self, session, message):
            observed.append(message.id)
            return ContextResult(prefix="[generated] ", mode="llm-local")

    monkeypatch.setattr(
        "recall.services.indexer._prepare_context_run",
        lambda _: ContextRun(config=config.embedding.context, backend=Model(), mode="llm-local"),
    )
    backend = RecordingBackend()
    protected = conn.execute(
        "SELECT message_id,context_text FROM message_state WHERE message_id<>? ORDER BY message_id",
        [identifier],
    ).fetchall()
    if surface == "library":
        summary = recompute_context_for_rows(
            config=config, conn=conn, embed=True, get_backend=lambda _: backend
        )
    else:
        server = RpcServer(config)
        server._conn = conn
        state = EmbedPhaseState(_config=config)
        server._embed_state = state
        _allow_embedding(monkeypatch, state, backend)
        if late:
            writer = server._enrichment_writer_call
            # Freeze with a temporarily stuck row, then acquire live custody
            # before the preparation callback admits any target.
            conn.execute(
                "UPDATE duplicate_transfers SET phase='stuck',stuck_reason='normalized_mismatch'"
            )

            async def acquire(operation, label, **kwargs):
                result = await writer(operation, label, **kwargs)
                if label == "recompute target selection":
                    conn.execute(
                        "UPDATE duplicate_transfers SET phase='catch_up',stuck_reason=NULL"
                    )
                return result

            monkeypatch.setattr(server, "_enrichment_writer_call", acquire)
        summary = asyncio.run(
            server._recompute_context_request(
                config=config, since=None, only_mode=None, embed=True, client=None
            )
        )
    assert observed == [identifier]
    assert backend.calls == [["[generated] eligible tail"]]
    assert summary.changed == 1
    assert (
        conn.execute(
            "SELECT message_id,context_text FROM message_state "
            "WHERE message_id<>? ORDER BY message_id",
            [identifier],
        ).fetchall()
        == protected
    )
    assert conn.execute(
        "SELECT context_text FROM message_state WHERE message_id=?", [identifier]
    ).fetchone() == ("[generated] ",)


@pytest.mark.parametrize("stage", ["catch_up", "owning", "shrinking"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("suffix", [False, True])
def test_real_grok_metadata_add_change_remove_finishes(
    tmp_path, monkeypatch, stage, native, suffix
):
    import json
    from datetime import UTC, datetime

    from recall.db.fts_sidecar import ensure_sidecar
    from recall.parsers.grok import GrokParser
    from recall.services.transfers import (
        record_coverage,
        verify_pass,
    )

    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    ensure_sidecar(config)
    parser = GrokParser(roots=(tmp_path,))
    original = tmp_path / "old/%2Fnative/session/chat_history.jsonl"
    moved = tmp_path / "new/%2Fnative/session/chat_history.jsonl"
    original.parent.mkdir(parents=True)
    moved.parent.mkdir(parents=True)
    record = b'{"type":"user","content":"boundary"}\n'
    content = record + (
        b'{"type":"assistant","model_id":"native-model","content":"first"}\n' + record
        if native
        else b""
    )
    original.write_bytes(content)
    moved.write_bytes(content + record)
    database = tmp_path / "grok.duckdb"
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        first = Harness(config, conn, parser, original)
        first.reconcile()
        row = conn.execute("SELECT id FROM sessions").fetchone()
        assert row is not None
        p = row[0]
        second = Harness(config, conn, parser, moved)
        second.reconcile()
        row = conn.execute("SELECT id FROM sessions WHERE source_path=?", [str(moved)]).fetchone()
        assert row is not None
        s = row[0]
        original.unlink()
        _walk(conn, tmp_path, parser=parser, drain_transfers=False)
        second.reconcile()
        if stage != "catch_up":
            _advance(conn, config, stage)
    for override in ("added", "changed", None):
        summary = moved.parent / "summary.json"
        if override is None:
            summary.unlink()
        else:
            summary.write_text(
                json.dumps(
                    {
                        "created_at": "2026-01-01T00:00:00Z",
                        "info": {"cwd": "/override"},
                        "git_root_dir": "/override/repo",
                        "head_branch": override,
                        "current_model_id": override,
                    }
                )
            )
        if suffix:
            with moved.open("ab") as stream:
                stream.write(record)
        with duckdb.connect(str(database)) as conn:
            harness = Harness(config, conn, parser, moved)
            assert harness.reconcile() == suffix
            expected = (
                (None, "/native", None, None, "native-model" if native else None)
                if override is None
                else (
                    datetime(2026, 1, 1, tzinfo=UTC).astimezone().replace(tzinfo=None),
                    "/override",
                    "/override/repo",
                    override,
                    override,
                )
            )
            actual = conn.execute(
                """SELECT started_at,cwd,git_repo,git_branch,model
                    FROM session_state WHERE session_id=?""",
                [p],
            ).fetchone()
            assert actual == expected
            parsed = parser.parse(moved).session
            assert (
                parsed.started_at,
                parsed.cwd,
                parsed.git_repo,
                parsed.git_branch,
                parsed.model,
            ) == (
                None if override is None else datetime(2026, 1, 1, tzinfo=UTC),
                *expected[1:],
            )
            row = _transfers(conn).load()[0]
            assert row.phase == stage
            assert row.predecessor_publish_seq >= 2
            if stage == "catch_up":
                result = verify_pass(conn, row)
                harness.catalog.force_reconcile(harness.source, str(harness.path))
                harness.reconcile(full=True)
                assert not record_coverage(conn, result)
    with duckdb.connect(str(database)) as conn:
        _advance(conn, config)
        assert conn.execute("SELECT id FROM sessions").fetchall() == [(p,)]
        assert conn.execute("SELECT id FROM sessions WHERE id=?", [s]).fetchone() is None
        assert conn.execute(
            "SELECT cwd,git_repo,git_branch,model FROM session_state"
        ).fetchone() == ("/native", None, None, "native-model" if native else None)


@pytest.mark.parametrize(
    "action",
    [
        "raw",
        "coverage",
        "ownership",
        "transition",
        "tools",
        "markers",
        "messages",
        "usage",
        "finish",
    ],
)
@pytest.mark.parametrize("when", ["before", "after"])
def test_killed_transaction_resumes_to_same_history(pair, action, when, tmp_path):
    import json
    import os
    import selectors
    import subprocess
    import sys

    from recall.services.transfers import (
        OwnershipCursor,
        TransferDriver,
        ownership_page,
        prepare_ownership_page,
        prove_ownership_context,
    )

    from tests.keyword_turns import without_keyword_index

    conn, first, second, predecessor, survivor, database = pair
    if action != "raw":
        second.reconcile()
    if action not in ("raw", "coverage"):
        _advance(conn, first.config, "owning")
    if action == "transition":
        cursor = OwnershipCursor()
        proof = prove_ownership_context(conn, _transfers(conn).load()[0])
        while cursor.family != "complete":
            transfer = _transfers(conn).load()[0]
            page = prepare_ownership_page(conn, transfer, cursor, proof)
            assert page is not None
            cursor = ownership_page(conn, transfer, cursor, without_keyword_index(), page)
    if action in ("tools", "markers", "messages", "usage", "finish"):
        _advance(conn, first.config, "shrinking")
        conn.execute(
            """INSERT INTO usage_events(id,source,source_session_id,session_id,harvested_at)
                VALUES ('transfer-usage','claude_code','fixture',?,now())""",
            [survivor],
        )
        families = ("tool_calls", "session_stop_markers", "messages", "usage_events")
        desired = {"tools": 0, "markers": 1, "messages": 2, "usage": 3, "finish": 4}[action]
        driver = TransferDriver()
        for family in families[:desired]:
            while conn.execute(
                f"SELECT 1 FROM {family} WHERE session_id=? LIMIT 1", [survivor]
            ).fetchone():
                driver.turn(conn, first.config)
    expected = conn.execute(
        """SELECT idx,role,content,thinking FROM messages m
            JOIN message_state ms ON ms.message_id=m.id WHERE m.session_id=? ORDER BY idx""",
        [predecessor],
    ).fetchall()
    if action == "raw":
        expected = [
            (0, "user", "run the test suite", None),
            (1, "assistant", "Running it now.", None),
            (2, "user", None, None),
            (3, "assistant", "All 312 tests pass.", None),
        ]
    conn.close()
    log = tmp_path / "worker.log"
    with log.open("w") as error:
        process = subprocess.Popen(
            [sys.executable, "-m", "tests.transfer_crash_worker", str(database), action, when],
            stdout=subprocess.PIPE,
            stderr=error,
            env=os.environ.copy(),
            text=True,
        )
        assert process.stdout is not None
        try:
            with selectors.DefaultSelector() as events:
                events.register(process.stdout, selectors.EVENT_READ)
                assert events.select(20), "worker failed to reach the transaction boundary"
                line = process.stdout.readline()
            assert line and json.loads(line) == {"boundary": when}, log.read_text()
            process.kill()
            assert process.wait(timeout=10) == -9
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            process.stdout.close()
    with duckdb.connect(str(database)) as reopened:
        second = Harness(first.config, reopened, second.parser, second.path)
        second.reconcile()
        _advance(reopened, first.config)
        assert reopened.execute("SELECT id FROM sessions").fetchall() == [(predecessor,)]
        actual = reopened.execute(
            """SELECT idx,role,content,thinking FROM messages m
                JOIN message_state ms ON ms.message_id=m.id WHERE m.session_id=? ORDER BY idx""",
            [predecessor],
        ).fetchall()
        assert actual == expected
        assert reopened.execute(
            "SELECT message_count,tool_count FROM session_state"
        ).fetchone() == (4, 1)
        assert not TransferDriver().turn(reopened, first.config)


@pytest.fixture
def unbound_pair(tmp_path, monkeypatch):
    from recall.core.types import default_session_host
    from recall.db.fts_sidecar import ensure_sidecar
    from recall.services.moves import plan_moved_duplicates

    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RECALL_CONTEXT_MODE", "off")
    config = AppConfig.load()
    ensure_sidecar(config)
    with duckdb.connect(str(tmp_path / "unbound.duckdb")) as conn:
        ensure_schema(conn)
        parser = ClaudeCodeParser(roots=(tmp_path,))
        original = tmp_path / "original.jsonl"
        moved = tmp_path / "moved.jsonl"
        original.write_text("".join(LINES[:3]))
        moved.write_text("".join(LINES))
        Harness(config, conn, parser, original).reconcile()
        Harness(config, conn, parser, moved).reconcile()
        original.unlink()
        plans = plan_moved_duplicates(conn, host=default_session_host())
        assert len(plans) == 1
        yield conn, config, parser, plans[0]


@pytest.mark.parametrize("side", ["predecessor", "successor"])
@pytest.mark.parametrize(
    "column",
    [
        "committed_offset",
        "committed_prefix_sha256",
        "normalization_checkpoint",
        "desired_generation",
        "committed_generation",
        "content_epoch",
        "dev",
        "inode",
        "message_count",
        "tool_count",
    ],
)
def test_rebind_vetoes_changed_snapshot_without_mutating(unbound_pair, side, column):
    from recall.services.transfers import rebind

    conn, _, _, plan = unbound_pair
    sid = plan.predecessor_id if side == "predecessor" else plan.successor_id
    if column in ("message_count", "tool_count"):
        conn.execute(f"UPDATE session_state SET {column}={column}+1 WHERE session_id=?", [sid])
    elif column in ("normalization_checkpoint", "committed_prefix_sha256"):
        conn.execute(f"UPDATE source_files SET {column}=? WHERE session_id=?", ["changed", sid])
    else:
        conn.execute(
            f"UPDATE source_files SET {column}=COALESCE({column},0)+1 WHERE session_id=?", [sid]
        )
    before = conn.execute("SELECT id,source_path FROM sessions ORDER BY id").fetchall()
    with pytest.raises(RuntimeError, match="snapshot changed"):
        rebind(conn, plan)
    assert conn.execute("SELECT id,source_path FROM sessions ORDER BY id").fetchall() == before
    assert conn.execute("SELECT * FROM duplicate_transfers").fetchall() == []


def test_rebind_uses_no_content_table(unbound_pair):
    from recall.services.transfers import rebind

    conn, _, _, plan = unbound_pair

    class CatalogOnly:
        def execute(self, sql, *args, **kwargs):
            # This seam makes content unavailable during this operation only.
            if any(
                name in sql.lower().split()
                for name in (
                    "messages",
                    "message_state",
                    "tool_calls",
                    "message_embeddings",
                    "tool_call_embeddings",
                )
            ):
                raise RuntimeError("content unavailable")
            return conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(conn, name)

    rebind(cast(duckdb.DuckDBPyConnection, CatalogOnly()), plan)
    assert conn.execute(
        "SELECT phase,survivor_message_count,survivor_tool_count FROM duplicate_transfers"
    ).fetchone() == ("catch_up", 4, 1)
    assert conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=?", [plan.successor_id]
    ).fetchone() == (4,)


def test_schema36_report_and_apply_do_not_migrate(unbound_pair, monkeypatch):
    import hashlib
    import json

    from recall.cli.app import app
    from recall.db.schema import get_schema_version, set_schema_version_to
    from typer.testing import CliRunner

    conn, _config, _, _ = unbound_pair
    database = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    conn.execute("DROP TABLE duplicate_transfers")
    conn.execute("DELETE FROM schema_version")
    set_schema_version_to(conn, 36)
    conn.execute("DELETE FROM schema_migrations WHERE migration_id='0038_duplicate_transfers'")
    conn.close()
    monkeypatch.setenv("RECALL_DB_PATH", str(database))
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    runner = CliRunner()
    report = runner.invoke(app, ["db", "supersede-moved", "--json"])
    assert report.exit_code == 0, report.output
    payload = json.loads(report.stdout)
    assert set(payload) == {"applied", "superseded", "transfers", "census", "pairs", "moves"}
    assert payload["transfers"] == []
    applied = runner.invoke(app, ["db", "supersede-moved", "--apply", "--json"])
    assert applied.exit_code == 2
    assert "schema 38" in applied.output
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    with duckdb.connect(str(database), read_only=True) as check:
        assert get_schema_version(check) == 36


@pytest.mark.parametrize("surface", ["watcher", "library"])
def test_each_full_publication_fences_same_epoch_even_without_search_change(pair, surface):
    from dataclasses import replace

    from recall.core.config import SourceConfig
    from recall.core.types import Source
    from recall.services.indexer import index_sessions
    from recall.services.transfers import record_coverage, verify_pass
    from recall.services.watcher import index_single_session

    from tests.keyword_turns import without_keyword_index

    conn, first, second, _, _, _ = pair
    second.reconcile()
    catalog = _transfers(conn)
    before = catalog.load()[0]
    stale = verify_pass(conn, before)
    if surface == "watcher":
        index_single_session(
            second.path,
            second.parser,
            first.config,
            conn=conn,
            turn=without_keyword_index(),
            prepared_result=second.parser.parse(second.path, witness_offset=before.survivor_offset),
        )
    else:
        config = replace(
            first.config, sources={"claude_code": SourceConfig(roots=(second.path.parent,))}
        )
        summary = index_sessions(
            source=Source.CLAUDE_CODE,
            full=True,
            recreate=False,
            verbose=False,
            config=config,
            conn=conn,
            embed=False,
            workers=1,
        )
        assert summary.failed == 0
    after = catalog.load()[0]
    assert after.predecessor_publish_seq == before.predecessor_publish_seq + 1
    assert after.witness_epoch == before.witness_epoch
    assert not record_coverage(conn, stale)


def test_rolled_back_raw_publication_keeps_counter_and_catalog(pair):
    from recall.services.indexer import (
        _load_session_write_snapshot,
        _require_committable_capture,
        _write_session,
    )

    from tests.keyword_turns import without_keyword_index

    conn, _, second, _, _, _ = pair
    second.reconcile()
    catalog = _transfers(conn)
    before = catalog.load()[0]
    result = second.parser.parse(second.path, witness_offset=before.survivor_offset)
    snapshot = _load_session_write_snapshot(conn, result.session, host=None, capture=result)
    publication = _require_committable_capture(
        second.path,
        result,
        has_indexed_history=True,
        conn=conn,
        snapshot=snapshot,
        parser=second.parser,
    )
    rows = conn.execute("SELECT * FROM source_files ORDER BY source_key").fetchall()

    def fail():
        raise RuntimeError("injected raw transaction failure")

    with pytest.raises(RuntimeError, match="injected"):
        _write_session(
            conn,
            result.session,
            turn=without_keyword_index(),
            last_byte_offset=result.next_byte_offset,
            tail_facts=result.tail_facts,
            snapshot=snapshot,
            publication=publication,
            on_commit=fail,
        )
    assert catalog.load()[0] == before
    assert conn.execute("SELECT * FROM source_files ORDER BY source_key").fetchall() == rows


def test_custody_rejects_result_prepared_before_rebind(unbound_pair, monkeypatch):
    from recall.services.embed_phase import (
        EmbedPhaseState,
        commit_prepared_embed_cycle,
        generate_prepared_embed_cycle,
        prepare_context_recompute,
    )
    from recall.services.transfers import rebind

    from tests.test_services.test_embed_phase import RecordingBackend, _allow_embedding

    conn, config, _, plan = unbound_pair
    ids = tuple(
        row[0]
        for row in conn.execute(
            "SELECT id FROM messages WHERE session_id=? ORDER BY idx", [plan.successor_id]
        ).fetchall()
    )
    prepared = prepare_context_recompute(
        plan.successor_id, ids, conn=conn, config=config, embed=True
    )
    backend = RecordingBackend()
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, backend)
    generate_prepared_embed_cycle(config, state, prepared)
    rebind(conn, plan)
    before = conn.execute("SELECT * FROM message_state ORDER BY message_id").fetchall()
    assert commit_prepared_embed_cycle(prepared, conn=conn) == 0
    assert conn.execute("SELECT * FROM message_state ORDER BY message_id").fetchall() == before
    assert conn.execute("SELECT * FROM message_embeddings").fetchall() == []


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("fact", ["native", "result"])
def test_missing_native_or_changed_result_refuses_coverage(pair, attached, fact):
    from recall.services.transfers import record_coverage, verify_pass

    conn, _, second, predecessor, survivor, _ = pair
    second.reconcile()
    if attached:
        old = conn.execute("SELECT id FROM tool_calls WHERE session_id=?", [survivor]).fetchone()[0]
        new = conn.execute(
            "SELECT id FROM tool_calls WHERE session_id=?", [predecessor]
        ).fetchone()[0]
    else:
        old, new = survivor + "-orphan", predecessor + "-orphan"
        for sid, identifier in ((survivor, old), (predecessor, new)):
            conn.execute(
                "INSERT INTO tool_calls(id,session_id,idx,tool_name) VALUES (?,?,0,'Bash')",
                [identifier, sid],
            )
            conn.execute(
                "UPDATE session_state SET tool_count=tool_count+1 WHERE session_id=?", [sid]
            )
            conn.execute(
                "INSERT INTO tool_use_ids(tool_call_id,session_id,tool_use_id) "
                "VALUES (?,?,'orphan-native')",
                [identifier, sid],
            )
        conn.execute("UPDATE duplicate_transfers SET survivor_tool_count=survivor_tool_count+1")
    if fact == "native":
        conn.execute("DELETE FROM tool_use_ids WHERE tool_call_id=?", [new])
    else:
        conn.execute("DELETE FROM tool_results WHERE tool_call_id IN (?,?)", [old, new])
        conn.execute("INSERT INTO tool_results VALUES (?,'stored result',false,NULL)", [old])
        conn.execute("INSERT INTO tool_results VALUES (?,'different result',false,NULL)", [new])
    before = conn.execute("SELECT id FROM tool_calls ORDER BY id").fetchall()
    catalog = _transfers(conn)
    record_coverage(conn, verify_pass(conn, catalog.load()[0]))
    assert catalog.load()[0].stuck_reason == "normalized_mismatch"
    assert conn.execute("SELECT id FROM tool_calls ORDER BY id").fetchall() == before


def test_marker_positions_cover_later_events_and_reject_missing(pair):
    from recall.services.transfers import verify_pass

    conn, _, second, predecessor, _survivor, _ = pair
    second.reconcile()
    catalog = _transfers(conn)
    # Same position may carry a later lifecycle value on P.
    conn.execute(
        "UPDATE session_stop_markers SET reason='task_complete' WHERE session_id=?", [predecessor]
    )
    assert verify_pass(conn, catalog.load()[0]).mismatch is None
    conn.execute("DELETE FROM session_stop_markers WHERE session_id=?", [predecessor])
    assert verify_pass(conn, catalog.load()[0]).mismatch == "stop marker position"


@pytest.mark.parametrize(
    "field",
    [
        "dev",
        "inode",
        "size",
        "mtime_ns",
        "generation",
        "ctime_ns",
        "sidecar_sha256",
        "parser_revision",
    ],
)
def test_changed_refusal_observation_cannot_be_reused_after_restart(pair, field):
    from dataclasses import replace

    from recall.services.transfers import (
        RefusalObservation,
        ValidatedRefusal,
        finish_predecessor_refusal,
    )

    conn, _, second, _, _, _ = pair
    second.path.write_text(LINES[0] + '{"type":"future_record","padding":"' + "x" * 20000 + '"}\n')
    second.reconcile(full=True)
    transfer = _transfers(conn).load()[0]
    assert transfer.predecessor_refusal_observation is not None
    assert transfer.predecessor_refusal_kind is not None
    observation = RefusalObservation.decode(transfer.predecessor_refusal_observation)
    old = getattr(observation, field)
    changed = replace(
        observation,
        **{
            field: ("b" * 64 if field == "sidecar_sha256" else "other-parser")
            if isinstance(old, str)
            else (old or 0) + 1
        },
    )
    event = ValidatedRefusal(transfer.survivor_id, transfer.predecessor_refusal_kind, changed)
    finish_predecessor_refusal(conn, event, second.parser)
    assert _transfers(conn).load()[0].phase == "catch_up"


@pytest.mark.parametrize("stale_mismatch", [False, True])
def test_same_byte_preserved_derived_field_publication_invalidates_pass(pair, stale_mismatch):
    from recall.services.indexer import (
        IndexedHistory,
        _compare_indexed_history,
        _load_session_write_snapshot,
        _require_committable_capture,
        _write_session,
    )
    from recall.services.transfers import record_coverage, verify_pass

    from tests.keyword_turns import without_keyword_index

    conn, _, second, predecessor, survivor, _ = pair
    second.reconcile()
    catalog = _transfers(conn)
    before = catalog.load()[0]
    if stale_mismatch:
        conn.execute(
            "UPDATE tool_calls SET skill_name='survivor-only' WHERE session_id=?", [survivor]
        )
    stale = verify_pass(conn, before)
    result = second.parser.parse(second.path, witness_offset=before.survivor_offset)
    snapshot = _load_session_write_snapshot(conn, result.session, host=None, capture=result)
    call = next(call for message in result.session.messages for call in message.tool_calls)
    call.skill_name = "parser-derived-change"
    assert (
        _compare_indexed_history(conn, result.session, snapshot=snapshot)
        is IndexedHistory.PRESERVED
    )
    publication = _require_committable_capture(
        second.path,
        result,
        has_indexed_history=True,
        conn=conn,
        snapshot=snapshot,
        parser=second.parser,
    )
    _write_session(
        conn,
        result.session,
        turn=without_keyword_index(),
        last_byte_offset=result.next_byte_offset,
        tail_facts=result.tail_facts,
        snapshot=snapshot,
        publication=publication,
    )
    after = catalog.load()[0]
    assert after.witness_epoch == before.witness_epoch
    assert after.predecessor_publish_seq == before.predecessor_publish_seq + 1
    assert not record_coverage(conn, stale)
    assert catalog.load()[0].phase == "catch_up"
    assert conn.execute(
        "SELECT skill_name FROM tool_calls WHERE session_id=?", [predecessor]
    ).fetchone() == ("parser-derived-change",)


def test_delayed_result_publication_fences_coverage(tmp_path, monkeypatch):
    from recall.db.fts_sidecar import ensure_sidecar
    from recall.services.transfers import record_coverage, verify_pass

    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    config = AppConfig.load()
    ensure_sidecar(config)
    parser = ClaudeCodeParser(roots=(tmp_path,))
    original = tmp_path / "original.jsonl"
    moved = tmp_path / "moved.jsonl"
    original.write_text(LINES[0])
    moved.write_text("".join(LINES[:2]))
    with duckdb.connect() as conn:
        ensure_schema(conn)
        Harness(config, conn, parser, original).reconcile()
        h = Harness(config, conn, parser, moved)
        h.reconcile()
        original.unlink()
        _walk(conn, tmp_path, parser=parser, drain_transfers=False)
        h.reconcile()
        catalog = _transfers(conn)
        before = catalog.load()[0]
        stale = verify_pass(conn, before)
        moved.write_text("".join(LINES))
        assert h.reconcile()
        after = catalog.load()[0]
        assert after.predecessor_publish_seq == before.predecessor_publish_seq + 1
        assert not record_coverage(conn, stale)
        assert verify_pass(conn, after).mismatch is None
        assert conn.execute(
            """SELECT r.result_summary FROM tool_results r
                JOIN tool_calls t ON t.id=r.tool_call_id WHERE t.session_id=?""",
            [after.predecessor_id],
        ).fetchone() == ("312 passed in 41.02s",)


@pytest.mark.parametrize("version", [36, 38])
def test_compaction_preserves_transfer_capability(unbound_pair, version):
    from dataclasses import replace

    from recall.db.schema import get_schema_version, set_schema_version_to
    from recall.services.compaction import compact
    from recall.services.transfers import rebind

    conn, config, _, plan = unbound_pair
    database = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    if version == 38:
        rebind(conn, plan)
        before = conn.execute("SELECT * FROM duplicate_transfers").fetchall()
    else:
        conn.execute("DROP TABLE duplicate_transfers")
        conn.execute("DELETE FROM schema_version")
        set_schema_version_to(conn, 36)
        before = []
    messages = conn.execute("SELECT * FROM messages ORDER BY id").fetchall()
    conn.close()
    config = replace(config, db_path=database)
    result = compact(config)
    assert result.replaced
    with duckdb.connect(str(database), read_only=True) as reopened:
        assert get_schema_version(reopened) == version
        assert reopened.execute("SELECT * FROM messages ORDER BY id").fetchall() == messages
        if version == 38:
            assert reopened.execute("SELECT * FROM duplicate_transfers").fetchall() == before
        else:
            assert (
                reopened.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name='duplicate_transfers'"
                ).fetchone()
                is None
            )


def test_final_image_equals_single_transaction_oracle(unbound_pair, tmp_path):
    import shutil

    from recall.services.transfers import apply_transfers

    from tests.duplicate_transfer_oracle import supersede_moved_duplicate

    conn, config, _, plan = unbound_pair
    conn.execute(
        """UPDATE message_state SET context_text='paid context: ',context_mode='llm-local',
        fts_content='paid context: '||COALESCE(content,''),
        fts_thinking='paid context: '||COALESCE(thinking,'')
        WHERE message_id IN (SELECT id FROM messages WHERE session_id=? AND idx=0)""",
        [plan.successor_id],
    )
    conn.execute("CHECKPOINT")
    database = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    oracle_path = tmp_path / "oracle.duckdb"
    shutil.copyfile(database, oracle_path)
    with duckdb.connect(str(oracle_path)) as oracle:
        supersede_moved_duplicate(oracle, plan, queue_sidecar_deletes=False)
        assert apply_transfers(conn, config, [plan]) == 1
        for table, key in (
            ("sessions", "id"),
            ("messages", "id"),
            ("message_state", "message_id"),
            ("message_embeddings", "message_id"),
            ("tool_calls", "id"),
            ("tool_call_embeddings", "tool_call_id"),
            ("tool_results", "tool_call_id"),
            ("tool_use_ids", "tool_call_id"),
            ("session_stop_markers", "session_id,message_idx"),
            ("usage_events", "id"),
        ):
            assert (
                conn.execute(f"SELECT * FROM {table} ORDER BY {key}").fetchall()
                == oracle.execute(f"SELECT * FROM {table} ORDER BY {key}").fetchall()
            ), table
        columns = (
            "started_at,ended_at,duration_seconds,model,cwd,git_repo,git_branch,"
            "message_count,tool_count,input_tokens,output_tokens,cached_input_tokens,"
            "is_complete,file_size,last_byte_offset"
        )
        assert (
            conn.execute(f"SELECT {columns} FROM session_state").fetchall()
            == oracle.execute(f"SELECT {columns} FROM session_state").fetchall()
        )


@pytest.mark.parametrize(
    "old_companion,new_companion,old_context,new_context,expected",
    [
        ("null", "vector", "paid: ", "paid: ", "null"),
        ("vector", "null", "paid: ", "paid: ", "vector"),
        (None, "null", "paid: ", "paid: ", "null"),
        (None, "vector", "paid: ", "other: ", None),
        ("null", "vector", "paid: ", "other: ", "null"),
    ],
)
def test_whole_processed_null_companions_follow_final_text(
    old_companion, new_companion, old_context, new_context, expected
):
    from recall.core.models import Message
    from recall.core.types import Role
    from recall.services.transfers import (
        EmbeddingCompanion,
        OwnershipMessage,
        OwnershipRow,
        ownership_decisions,
    )

    old = Message(
        id="s0",
        session_id="s",
        idx=0,
        role=Role.USER,
        content="body",
        thinking="thought",
        has_thinking=True,
        context_text=old_context,
        context_mode="llm-local",
    )
    new = Message(
        id="p0",
        session_id="p",
        idx=0,
        role=Role.USER,
        content="body",
        thinking="thought",
        has_thinking=True,
        context_text=new_context,
        context_mode="llm-local",
    )
    companions = {
        None: None,
        "null": EmbeddingCompanion(None, None),
        "vector": EmbeddingCompanion((0.5,) * 384, (0.25,) * 384),
    }
    decision = ownership_decisions(
        OwnershipRow(OwnershipMessage.from_message(old), companions[old_companion]),
        OwnershipRow(OwnershipMessage.from_message(new), companions[new_companion]),
        context_eligible=True,
    )
    assert decision.context_text == "paid: "
    assert decision.companion == companions[expected]


def test_parked_freshness_reports_missing_history_without_timeout(pair):
    from dataclasses import asdict

    from recall.cli.utils import stale_after_fresh_note
    from recall.services.live import derive_freshness

    conn, _, _, _, survivor, _ = pair
    path = conn.execute("SELECT source_path FROM sessions WHERE id=?", [survivor]).fetchone()[0]
    freshness = derive_freshness(path, indexed_mtime=1, indexed_size=100)
    assert not freshness.current
    assert freshness.file_size is None and freshness.file_mtime is None
    assert "source_unreadable" in freshness.limitations
    note = stale_after_fresh_note([{"freshness": asdict(freshness)}])
    assert (
        note == "note: 1 of 1 session(s) have no readable transcript file; showing indexed history"
    )


@pytest.mark.parametrize(
    "action",
    [
        "rebind",
        "first_refusal",
        "stable_refusal",
        "covered_refusal",
        "accepted_rewrite",
        "verification",
        "keyword_retirement",
    ],
)
@pytest.mark.parametrize("when", ["before", "after"])
def test_remaining_crash_boundaries_preserve_authority(unbound_pair, action, when, tmp_path):
    import json

    from recall.core.types import default_session_host
    from recall.services.moves import plan_moved_duplicates
    from recall.services.transfers import TransferDriver, apply_transfers, rebind

    conn, config, parser, plan = unbound_pair
    database = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    moved = Path(plan.successor_path)
    h = Harness(config, conn, parser, moved)
    if action != "rebind":
        rebind(conn, plan)
    if action in ("first_refusal", "stable_refusal"):
        moved.write_text(LINES[0] + '{"type":"future_record","padding":"' + "x" * 20000 + '"}\n')
        if action == "stable_refusal":
            h.reconcile(full=True)
    elif action not in ("rebind",):
        h.reconcile()
        if action == "accepted_rewrite":
            tail = {
                "type": "user",
                "message": {"role": "user", "content": "initial tail"},
                "uuid": "tail",
                "timestamp": "2026-09-07T18:02:00Z",
            }
            with moved.open("a") as stream:
                stream.write(json.dumps(tail) + "\n")
            assert h.reconcile()
        if action in ("covered_refusal", "accepted_rewrite"):
            _advance(conn, config, "shrinking")
            TransferDriver().turn(conn, config)
            if action == "covered_refusal":
                moved.write_text(LINES[0])
            else:
                moved.write_text(moved.read_text().replace("initial tail", "amended tail"))
        elif action == "keyword_retirement":
            _advance(conn, config, "owning")
            conn.execute(
                """UPDATE message_state SET context_text='paid crash: ',context_mode='llm-local',
                fts_content='paid crash: '||COALESCE(content,''),
                fts_thinking='paid crash: '||COALESCE(thinking,'')
                WHERE message_id IN (SELECT id FROM messages WHERE session_id=?)""",
                [plan.successor_id],
            )
    before = conn.execute("SELECT id FROM messages ORDER BY id").fetchall()
    conn.close()
    _kill_worker(database, action, when, tmp_path / "boundary-worker.log")
    with duckdb.connect(str(database)) as reopened:
        if action in ("first_refusal", "stable_refusal"):
            # A lost stable transition requires another actual retry; a poll
            # never manufactures the second event from durable refusal state.
            before_phase = _transfers(reopened).load()[0].phase
            TransferDriver().turn(reopened, config)
            assert _transfers(reopened).load()[0].phase == before_phase
            for _ in range(2):
                Harness(config, reopened, parser, moved).reconcile(full=True)
                if _transfers(reopened).load()[0].phase == "stuck":
                    break
            assert _transfers(reopened).load()[0].stuck_reason == "predecessor_refused"
            assert apply_transfers(reopened, config, []) == 0
            assert reopened.execute("SELECT id FROM messages ORDER BY id").fetchall() == before
            assert reopened.execute("SELECT COUNT(*) FROM sessions").fetchone() == (2,)
        else:
            if action == "rebind":
                plans = plan_moved_duplicates(
                    reopened, host=default_session_host(), transfers=_transfers(reopened)
                )
                assert apply_transfers(reopened, config, plans) == 1
            else:
                if action == "accepted_rewrite":
                    Harness(config, reopened, parser, moved).reconcile(full=True)
                _advance(reopened, config)
            assert reopened.execute("SELECT id FROM sessions").fetchall() == [
                (plan.predecessor_id,)
            ]
            if action == "covered_refusal":
                assert reopened.execute("SELECT message_count FROM session_state").fetchone() == (
                    4,
                )
            else:
                expected = 5 if action == "accepted_rewrite" else 4
                assert reopened.execute("SELECT message_count FROM session_state").fetchone() == (
                    expected,
                )
            if action == "accepted_rewrite":
                assert reopened.execute(
                    """SELECT content FROM message_state ms
                        JOIN messages m ON m.id=ms.message_id WHERE m.idx=4"""
                ).fetchone() == ("amended tail",)
            _repair_keywords(reopened, config)
    if action == "covered_refusal":
        assert _fresh_process(database, "ordinary_retry", moved) == {
            "message_counts": [1],
            "phases": [],
        }


@pytest.mark.parametrize("distinct_native", [False, True])
@pytest.mark.parametrize("survivor_sorts_first", [False, True])
def test_harvest_never_refills_survivor_usage_pages(
    tmp_path, monkeypatch, distinct_native, survivor_sorts_first
):
    import json

    from recall.core.ids import session_id
    from recall.core.types import default_session_host
    from recall.db.fts_sidecar import ensure_sidecar
    from recall.parsers.grok import GrokParser
    from recall.services.moves import plan_moved_duplicates
    from recall.services.transfers import TransferDriver, rebind
    from recall.services.usage_harvest import rollup_grok_sessions

    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    config = AppConfig.load()
    ensure_sidecar(config)
    parser = GrokParser(roots=(tmp_path,))
    old_sid = "old-sid" if distinct_native else "shared-sid"
    new_sid = "new-sid" if distinct_native else "shared-sid"
    for n in range(100):
        original = tmp_path / f"old-{n}/%2Fnative" / old_sid / "chat_history.jsonl"
        moved = tmp_path / f"new-{n}/%2Fnative" / new_sid / "chat_history.jsonl"
        if (
            session_id("grok", str(moved)) < session_id("grok", str(original))
        ) == survivor_sorts_first:
            break
    else:
        pytest.fail("no deterministic path pair satisfied the fixture order")
    for path in (original, moved):
        path.parent.mkdir(parents=True)
        path.write_text('{"type":"user","content":"body"}\n')
        (path.parent / "summary.json").write_text(
            json.dumps({"created_at": "2026-01-01T00:00:00Z"})
        )
    with duckdb.connect() as conn:
        ensure_schema(conn)
        Harness(config, conn, parser, original).reconcile()
        h = Harness(config, conn, parser, moved)
        h.reconcile()
        original.unlink()
        plan = plan_moved_duplicates(conn, host=default_session_host())[0]
        rebind(conn, plan)
        conn.executemany(
            """INSERT INTO usage_events(id,source,source_session_id,session_id,
            prompt_tokens,completion_tokens,harvested_at) VALUES (?,'grok',?,?,1,0,now())""",
            [(f"old-{n}", new_sid, plan.successor_id) for n in range(513)],
        )
        conn.execute(
            """INSERT INTO usage_events(id,source,source_session_id,prompt_tokens,
            completion_tokens,harvested_at) VALUES ('new-event','grok',?,7,3,now())""",
            [new_sid],
        )
        rollup_grok_sessions(conn)
        linked = conn.execute("SELECT session_id FROM usage_events WHERE id='new-event'").fetchone()
        assert linked == ((None,) if distinct_native else (plan.predecessor_id,))
        assert conn.execute(
            "SELECT COUNT(*) FROM usage_events WHERE session_id=?", [plan.successor_id]
        ).fetchone() == (513,)
        h.reconcile()
        driver = TransferDriver()
        ticks = 0
        for tick in range(100):
            count = conn.execute(
                "SELECT COUNT(*) FROM usage_events WHERE session_id=?", [plan.successor_id]
            ).fetchone()
            assert count is not None
            remaining = count[0]
            conn.execute(
                """INSERT INTO usage_events(id,source,source_session_id,prompt_tokens,
                completion_tokens,harvested_at) VALUES (?,'grok',?,1,0,now())""",
                [f"tick-{tick}", new_sid],
            )
            ticks += 1
            rollup_grok_sessions(conn)
            assert conn.execute(
                "SELECT COUNT(*) FROM usage_events WHERE session_id=?", [plan.successor_id]
            ).fetchone() == (remaining,)
            driver.prepare(conn)
            driver.turn(conn, config)
            if not _transfers(conn).load():
                break
        else:
            pytest.fail("harvest prevented transfer completion")
        rollup_grok_sessions(conn)
        assert conn.execute("SELECT id FROM sessions").fetchall() == [(plan.predecessor_id,)]
        assert conn.execute("SELECT input_tokens,output_tokens FROM session_state").fetchone() == (
            520 + ticks,
            3,
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM usage_events WHERE session_id=?", [plan.predecessor_id]
        ).fetchone() == (514 + ticks,)


@pytest.mark.parametrize("when", ["before", "after"])
def test_custody_rejection_crash_never_publishes_inference(
    unbound_pair, monkeypatch, tmp_path, when
):
    import pickle

    from recall.services.embed_phase import (
        EmbedPhaseState,
        generate_prepared_embed_cycle,
        prepare_context_recompute,
    )
    from recall.services.transfers import rebind

    from tests.test_services.test_embed_phase import RecordingBackend, _allow_embedding

    conn, config, parser, plan = unbound_pair
    database = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    ids = tuple(
        row[0]
        for row in conn.execute(
            "SELECT id FROM messages WHERE session_id=? ORDER BY idx", [plan.successor_id]
        ).fetchall()
    )
    prepared = prepare_context_recompute(
        plan.successor_id, ids, conn=conn, embed=True, config=config
    )
    backend = RecordingBackend()
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, backend)
    generate_prepared_embed_cycle(config, state, prepared)
    assert prepared.generated and backend.calls
    with (config.data_dir / "prepared-cycle.pkl").open("wb") as stream:
        pickle.dump(prepared, stream)
    rebind(conn, plan)
    before = conn.execute("SELECT * FROM message_state ORDER BY message_id").fetchall()
    conn.close()
    _kill_worker(database, "embed_rejected", when, tmp_path / "rejected-worker.log")
    with duckdb.connect(str(database)) as reopened:
        assert reopened.execute("SELECT * FROM message_embeddings").fetchall() == []
        assert (
            reopened.execute("SELECT * FROM message_state ORDER BY message_id").fetchall() == before
        )
        Harness(config, reopened, parser, Path(plan.successor_path)).reconcile()
        _advance(reopened, config)
        assert reopened.execute("SELECT id FROM sessions").fetchall() == [(plan.predecessor_id,)]
        assert reopened.execute("SELECT * FROM message_embeddings").fetchall() == []
