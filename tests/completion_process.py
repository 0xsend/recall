"""Native process boundary for the shared completion transaction and real publication."""

import json
import sys
from datetime import datetime
from unittest.mock import patch

from recall.core.config import AppConfig
from recall.core.daemon_status import IndexRunCounts
from recall.core.types import RunKind
from recall.db.fts_sidecar_stage import KeywordTurn
from recall.db.lease import owned_connection
from recall.services.recovery_admission import offline_file_lifetime
from recall.services.run_completion import (
    Cancelled,
    Failed,
    RunRecord,
    RunScope,
    record_completion_on,
)

from tests.repair_process import checkpoint


def main():
    kind, selected, fault = sys.argv[1:]
    config = AppConfig.load()
    with (
        offline_file_lifetime(config) as (owner, admission),
        owned_connection(owner, "completion_source", admission.open_preserving_upgrade) as lease,
    ):
        row = lease.call(
            "completion_fixture:message",
            lambda conn: conn.execute(
                "SELECT message_id,fts_content,fts_thinking FROM message_state ORDER BY ALL LIMIT 1"
            ).fetchone(),
        )
        assert row is not None
        turn = KeywordTurn.for_config(config)
        for _ in range(2):
            with turn.transaction_on(lease) as stage:
                stage.messages((row,))
        publication = turn.publish_on(lease.observer, attempt=True)
        assert publication.outcome == "published"
        assert dict(publication.applied) == {("message", row[0]): 2}
        pending = lease.call(
            "completion_fixture:pending",
            lambda conn: conn.execute(
                "SELECT kind,id,op,queued_at FROM fts_sidecar_pending ORDER BY ALL"
            ).fetchall(),
        )
        (config.data_dir / "published.json").write_text(json.dumps(pending, default=str))
        record = RunRecord(
            RunKind.INDEX,
            datetime(2026, 11, 2),
            IndexRunCounts(total=1, indexed=0, skipped=1, failed=0, changed=0, total_seconds=0),
        )
        scope = RunScope()
        scope.inventory_completed()
        scope.keyword_acknowledged()
        if kind == "qualified":
            outcome = scope.prepare(record, publication)
        elif kind == "partial":
            scope.incomplete("incomplete inventory")
            outcome = scope.prepare(record, publication)
        elif kind == "failed":
            outcome = Failed(record, "producer failed")
        else:
            assert kind == "cancelled"
            outcome = Cancelled(record, "producer cancelled")
        native_call = owner.call_db
        native_cleanup = owner.cleanup_db

        def call(operation, work):
            if operation == "run_completion:commit":
                checkpoint("before_native_commit", selected, fault)
            result = native_call(operation, work)
            if operation == "run_completion:commit":
                checkpoint("after_native_commit", selected, fault)
            return result

        def cleanup(operation, work):
            if operation == "run_completion:rollback":
                checkpoint("before_native_rollback", selected, fault)
            result = native_cleanup(operation, work)
            if operation == "run_completion:rollback":
                checkpoint("after_native_rollback", selected, fault)
            return result

        def barrier(stage):
            if "rollback" in selected and stage == "before_completion_commit":
                raise OSError("producer interrupted before completion")
            checkpoint(stage, selected, fault)

        with patch.object(owner, "call_db", call), patch.object(owner, "cleanup_db", cleanup):
            result = record_completion_on(
                lease,
                lease.completion_permit(),
                outcome,
                data_dir=config.data_dir,
                completed_at=datetime(2026, 11, 3),
                barrier=barrier,
            )
        assert selected == "none", "selected completion boundary was not reached"
        print(result)


if __name__ == "__main__":
    main()
