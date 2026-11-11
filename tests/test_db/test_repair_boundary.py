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
