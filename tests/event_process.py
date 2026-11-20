"""Disposable child for immutable event publication and legacy import."""

import errno
import json
import os
import signal
import sys
from pathlib import Path

import duckdb
from recall.core.config import AppConfig
from recall.core.repair_state import PublicationRole
from recall.db.connection import advisory_lock
from recall.db.ownership import offline_ownership
from recall.db.repair import load_repair_row, require_schema_fence
from recall.services.repair_events import (
    RecoveryEvent,
    import_legacy_events,
    publish_event,
    read_events,
)


def main() -> None:
    operation, selected, fault, record = sys.argv[1:]
    config = AppConfig.load()

    def barrier(stage):
        if stage == selected:
            print(f"checkpoint:{stage}", flush=True)
            if fault == "kill":
                os.kill(os.getpid(), signal.SIGKILL)
            raise OSError(errno.EIO if fault == "EIO" else errno.ENOSPC, f"injected {stage}")

    with offline_ownership(config.db_path), advisory_lock(config.lock_path):
        batch = read_events(config.data_dir)
        if operation == "publish":
            event = RecoveryEvent.model_validate_json(Path(record).read_bytes())
            publish_event(
                config.data_dir,
                event,
                role=PublicationRole.PRIMARY,
                barrier=barrier,
            )
        else:
            with duckdb.connect(str(config.db_path), read_only=True) as conn:
                fence = require_schema_fence(conn)
                row = load_repair_row(conn)
            if operation == "import":
                import_legacy_events(config.data_dir, fence, barrier=barrier)
            else:
                assert operation == "inspect"
                print(json.dumps({"marker": batch.classify(row).value}))
                return
    raise AssertionError("selected durability boundary was not reached")


if __name__ == "__main__":
    main()
