"""Persisted instructions cannot grant authority through coercion or stale identity."""

import duckdb
import pytest
from pydantic import ValidationError
from recall.db.repair import RepairRow, load_repair_row
from recall.db.schema import ensure_schema


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt_no", True),
        ("success_since_attempt", "true"),
        ("success_since_failure", 1),
        ("oracle_hash", "unchecked"),
        ("oracle_ref", "../outside"),
        ("oracle_ref", "repair-oracles/../outside"),
        (
            "acknowledged_events",
            [
                {
                    "event_id": "00000000-0000-4000-8000-000000000001",
                    "content_hash": "unchecked",
                    "disposition": "verified",
                }
            ],
        ),
        ("release_certificate", {}),
    ],
)
def test_malformed_durable_state_cannot_be_interpreted_as_permission(
    field: str, value: object
) -> None:
    with duckdb.connect() as conn:
        ensure_schema(conn)
        payload = load_repair_row(conn).model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        RepairRow.model_validate(payload)


def test_a_phase_change_without_its_singleton_is_not_a_committed_change() -> None:
    from recall.db.repair import save_repair_row

    with duckdb.connect() as conn:
        ensure_schema(conn)
        row = load_repair_row(conn)
        save_repair_row(conn, row)
        conn.execute("DELETE FROM daemon_index_repair")
        with pytest.raises(ValueError, match="recovery singleton"):
            save_repair_row(conn, row)
        assert conn.execute("SELECT count(*) FROM daemon_index_repair").fetchone() == (0,)
