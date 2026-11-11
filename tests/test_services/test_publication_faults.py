"""A fresh process admits only the event evidence publication actually retained."""

import errno
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

import pytest
from recall.core.repair_state import PublicationRole
from recall.db.connection import connect
from recall.db.repair import SchemaFence
from recall.services.repair_events import (
    EventFailure,
    LegacyEvent,
    RecoveryEvent,
    import_legacy_events,
    publish_event,
    read_events,
)

from tests.test_services.test_rpc_owned_boundary import server_at

_ADMIT = """
import json,sys
from dataclasses import replace
from pathlib import Path
import duckdb
from recall.core.config import AppConfig
from recall.services.self_repair import run_startup_self_repair
root=Path(sys.argv[1])
config=replace(AppConfig.load(),data_dir=root,db_path=root/'recall.duckdb',lock_path=root/'recall.lock')
first=run_startup_self_repair(config)
second=run_startup_self_repair(config)
with duckdb.connect(str(config.db_path),read_only=True) as fresh:
    row=fresh.execute(
        'SELECT phase,attempt_no,success_since_attempt,success_since_failure '
        'FROM daemon_index_repair'
    ).fetchone()
    repeats=fresh.execute('SELECT fatal_repeat_count FROM runtime_state').fetchone()[0]
print(json.dumps({'first':first.action,'second':second.action,'row':row,'repeats':repeats}))
"""


def _failure(kind: Literal["index-divergence", "other"]) -> EventFailure:
    return EventFailure(
        site="source_commit",
        exception_type="FatalException",
        message="Failed to delete all rows from index"
        if kind == "index-divergence"
        else "ordinary fatal",
        signature=kind,
        failure_class=kind,
        at=datetime(2026, 11, 1),
    )


def _fresh_admission(root: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", _ADMIT, str(root)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("error_number", [errno.EIO, errno.ENOSPC])
@pytest.mark.parametrize("later", [False, True])
@pytest.mark.parametrize(
    ("stage", "retained"),
    [
        ("before_file_sync", False),
        ("after_file_sync", False),
        ("before_link", False),
        ("after_link", True),
        ("before_directory_sync", True),
        ("after_directory_sync", True),
    ],
)
def test_failed_publication_admits_only_retained_evidence(
    tmp_path: Path, error_number: int, later: bool, stage: str, retained: bool
) -> None:
    config = server_at(tmp_path)._config
    with connect(config):
        pass
    primary = tmp_path / "daemon-failure.json"
    original = None
    if later:
        publish_event(
            tmp_path,
            RecoveryEvent(
                event_id=UUID(int=1), failure=_failure("other"), needs_index_verification=False
            ),
        )
        original = primary.read_bytes()

    def fail(boundary: str) -> None:
        if boundary == stage:
            raise OSError(error_number, "injected storage publication failure")

    with pytest.raises(OSError) as error:
        publish_event(
            tmp_path,
            RecoveryEvent(
                event_id=UUID(int=2),
                failure=_failure("index-divergence"),
                needs_index_verification=False,
            ),
            role=PublicationRole.SECONDARY if later else PublicationRole.PRIMARY,
            barrier=fail,
        )
    assert error.value.errno == error_number
    if later:
        assert primary.read_bytes() == original
    batch = read_events(tmp_path)
    actionable = [
        event
        for event in batch.identified()
        if event.failure and event.failure.failure_class == "index-divergence"
    ]
    assert bool(actionable) is retained
    assert not (tmp_path / "daemon-refused").exists()
    admitted = _fresh_admission(tmp_path)
    assert admitted["first"] == ("rebuilt" if retained else "recorded" if later else "none")
    assert admitted["second"] == "none"
    assert admitted["row"] == (
        ["verified", 1, False, False] if retained else ["idle", 0, True, not later]
    )
    assert admitted["repeats"] == (1 if retained or later else 0)


@pytest.mark.parametrize("error_number", [errno.EIO, errno.ENOSPC])
@pytest.mark.parametrize(
    ("stage", "converted"),
    [
        ("before_file_sync", False),
        ("after_file_sync", False),
        ("before_replace", False),
        ("after_replace", True),
        ("before_directory_sync", True),
        ("after_directory_sync", True),
    ],
)
def test_failed_fenced_import_keeps_a_complete_stable_instruction(
    tmp_path: Path, error_number: int, stage: str, converted: bool
) -> None:
    config = server_at(tmp_path)._config
    with connect(config):
        pass
    legacy = LegacyEvent(failure=_failure("index-divergence"), needs_index_verification=False)
    publish_event(tmp_path, legacy)
    primary = tmp_path / "daemon-failure.json"
    original = primary.read_bytes()

    def fail(boundary: str) -> None:
        if boundary == stage:
            raise OSError(error_number, "injected import publication failure")

    with pytest.raises(OSError):
        import_legacy_events(tmp_path, SchemaFence(41), barrier=fail)
    event = read_events(tmp_path).primary
    assert event is not None
    assert isinstance(event, RecoveryEvent if converted else LegacyEvent)
    assert event.failure == legacy.failure
    if not converted:
        assert primary.read_bytes() == original
    admitted = _fresh_admission(tmp_path)
    assert admitted == {
        "first": "rebuilt",
        "second": "none",
        "row": ["verified", 1, False, False],
        "repeats": 1,
    }
