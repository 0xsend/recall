"""Durable bytes have one strict identity before any database write."""

import json
from pathlib import Path

import pytest
from recall.services.repair_events import EventStorageError, read_events


@pytest.mark.parametrize(
    "payload",
    [
        b'{"version":1,"version":1,"failure":null,"needs_index_verification":true}',
        b'{"version":1,"failure":null,"needs_index_verification":"true"}',
        b'{"version":1,"failure":null,"needs_index_verification":false}',
        b'{"version":true,"failure":null,"needs_index_verification":true}',
        json.dumps(
            {
                "version": 2,
                "event_id": "00000000-0000-4000-8000-000000000001",
                "failure": None,
                "needs_index_verification": False,
            }
        ).encode(),
    ],
)
def test_ambiguous_or_coerced_record_holds(tmp_path: Path, payload: bytes) -> None:
    (tmp_path / "daemon-failure.json").write_bytes(payload)
    with pytest.raises(EventStorageError):
        read_events(tmp_path)


@pytest.mark.parametrize("secondary", [False, True])
def test_legacy_instruction_converts_once_after_fence(tmp_path: Path, secondary: bool) -> None:
    import duckdb
    from recall.db.repair import require_schema_fence
    from recall.db.schema import ensure_schema
    from recall.services.repair_events import RecoveryEvent, import_legacy_events

    directory = tmp_path / "fatal-events" if secondary else tmp_path
    directory.mkdir(exist_ok=True)
    filename = "00000000-0000-4000-8000-000000000001.json" if secondary else "daemon-failure.json"
    (directory / filename).write_text(
        '{"version":1,"failure":null,"needs_index_verification":true}'
    )
    with duckdb.connect() as conn:
        ensure_schema(conn)
        fence = require_schema_fence(conn)
        first = import_legacy_events(tmp_path, fence)
        original = first.secondary[0][1] if secondary else first.primary
        assert isinstance(original, RecoveryEvent)
        assert original.failure is None
        assert original.needs_index_verification
        content = (directory / filename).read_bytes()
        second = import_legacy_events(tmp_path, fence)
        assert second == first
        assert (directory / filename).read_bytes() == content
