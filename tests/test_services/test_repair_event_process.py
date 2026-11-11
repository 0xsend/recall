"""Only the event bytes surviving a child can guide the next owner's admission."""

import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.db.repair import SchemaFence
from recall.db.schema import ensure_schema
from recall.services.repair_events import (
    EventFailure,
    RecoveryEvent,
    import_legacy_events,
    publish_event,
    read_events,
)

_STAGES = (
    "before_file_sync",
    "after_file_sync",
    "before_link",
    "after_link",
    "before_directory_sync",
    "after_directory_sync",
)


def _environment(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(
        "\n".join(
            f"[sources.{name}]\nroots = []"
            for name in ("claude_code", "codex", "pi_agent", "grok", "kimi_code")
        )
    )
    database = data / "recall.duckdb"
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
    return data, {
        **os.environ,
        "RECALL_CONFIG_PATH": str(config),
        "RECALL_DATA_DIR": str(data),
        "RECALL_DB_PATH": str(database),
        "RECALL_LOCK_PATH": str(data / "recall.lock"),
    }


def _child(environment, operation, stage, fault, record="unused"):
    return subprocess.run(
        [sys.executable, "-m", "tests.event_process", operation, stage, fault, str(record)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _event(probe: bool, identity: int = 2):
    return RecoveryEvent(
        event_id=UUID(int=identity),
        failure=None
        if probe
        else EventFailure(
            site="selected_origin",
            exception_type="FatalException",
            message="Failed to delete all rows from index",
            signature="origin",
            failure_class="index-divergence",
            at=datetime(2026, 11, 1),
        ),
        needs_index_verification=probe,
        probe=probe,
    )


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("fault", ["kill", "EIO", "ENOSPC"])
@pytest.mark.parametrize("stage", _STAGES)
def test_primary_and_secondary_publication_under_process_loss_and_io_failure(
    tmp_path: Path, retained: bool, probe: bool, fault: str, stage: str
):
    data, environment = _environment(tmp_path)
    old = None
    if retained:
        publish_event(data, _event(False, 1))
        old = (data / "daemon-failure.json").read_bytes()
    event = _event(probe)
    record = tmp_path / "input.json"
    record.write_text(event.model_dump_json())
    result = _child(environment, "publish", stage, fault, record)
    assert f"checkpoint:{stage}" in result.stdout, result.stderr
    assert result.returncode == (-signal.SIGKILL if fault == "kill" else 1)
    batch = read_events(data)
    if retained:
        assert (data / "daemon-failure.json").read_bytes() == old
    durable = stage in {"after_link", "before_directory_sync", "after_directory_sync"}
    events = batch.identified()
    assert any(item.event_id == event.event_id for item in events) == durable
    if durable:
        assert next(item for item in events if item.event_id == event.event_id) == event
    inspected = _child(environment, "inspect", "unused", "unused")
    assert inspected.returncode == 0, inspected.stderr
    expected = "index" if retained or (durable and not probe) else "probe" if durable else "none"
    assert json.loads(inspected.stdout)["marker"] == expected


@pytest.mark.parametrize("secondary", [False, True])
@pytest.mark.parametrize("verification", [False, True])
@pytest.mark.parametrize("fault", ["kill", "EIO", "ENOSPC"])
@pytest.mark.parametrize("stage", tuple(item.replace("link", "replace") for item in _STAGES))
def test_import_interruption_keeps_complete_instruction_and_stable_identity(
    tmp_path: Path, secondary: bool, verification: bool, fault: str, stage: str
):
    data, environment = _environment(tmp_path)
    directory = data / "fatal-events" if secondary else data
    directory.mkdir(exist_ok=True)
    path = directory / (f"{UUID(int=1)}.json" if secondary else "daemon-failure.json")
    payload = {
        "version": 1,
        "failure": None
        if verification
        else {
            "site": "original",
            "exception_type": "FatalException",
            "message": "Failed to delete all rows from index",
            "signature": "original",
            "failure_class": "index-divergence",
            "at": "2026-11-01T03:00:00+03:00",
        },
        "needs_index_verification": verification,
    }
    path.write_text(json.dumps(payload))
    result = _child(environment, "import", stage, fault)
    assert f"checkpoint:{stage}" in result.stdout, result.stderr
    assert result.returncode == (-signal.SIGKILL if fault == "kill" else 1)
    survived = read_events(data)
    current = survived.secondary[0][1] if secondary else survived.primary
    assert current is not None and current.needs_index_verification == verification
    if verification:
        assert current.failure is None
    else:
        assert current.failure is not None
        assert (
            current.failure.site,
            current.failure.message,
            current.failure.signature,
            current.failure.at,
        ) == ("original", "Failed to delete all rows from index", "original", datetime(2026, 11, 1))
    first = import_legacy_events(data, SchemaFence(41))
    content = path.read_bytes()
    second = import_legacy_events(data, SchemaFence(41))
    assert second == first and path.read_bytes() == content
    converted = first.secondary[0][1] if secondary else first.primary
    assert isinstance(converted, RecoveryEvent)
    if isinstance(current, RecoveryEvent):
        assert current.event_id == converted.event_id
    inspected = _child(environment, "inspect", "unused", "unused")
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout)["marker"] == ("disk" if verification else "index")
