"""Fresh-process transaction interruption worker for durable transfer tests."""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path
from typing import cast

import duckdb
from recall.core.config import AppConfig
from recall.services.transfers import (
    OwnershipCursor,
    TransferDriver,
    begin_shrink,
    transfer_catalog,
)


class InterruptedConnection:
    def __init__(self, conn, when, *, commit_number=1, boundary_statement="COMMIT"):
        self.conn = conn
        self.when = when
        self.boundary_statement = boundary_statement
        self.commit_number = commit_number
        self.commits = 0

    def execute(self, statement, *args, **kwargs):
        commit = statement.strip().upper() == self.boundary_statement
        if commit:
            self.commits += 1
        commit = commit and self.commits == self.commit_number
        if commit and self.when == "before":
            self.pause()
        result = self.conn.execute(statement, *args, **kwargs)
        if commit and self.when == "after":
            self.pause()
        return result

    def pause(self):
        print(json.dumps({"boundary": self.when}), flush=True)
        signal.pause()
        raise AssertionError("interruption worker resumed without termination")

    def __getattr__(self, name):
        return getattr(self.conn, name)


def main():
    database, action, when = sys.argv[1:]
    config = AppConfig.load()
    with duckdb.connect(database) as conn:
        if action in ("ordinary_retry", "refusal_retry", "poll", "raw_chunk"):
            if action == "poll":
                for _ in range(3):
                    TransferDriver().turn(conn, config)
            else:
                from recall.parsers.claude_code import ClaudeCodeParser

                from tests.test_services.test_checkpoint_resume_fallbacks import Harness

                path = Path(when)
                Harness(config, conn, ClaudeCodeParser(roots=(path.parent,)), path).reconcile(
                    full=action != "raw_chunk"
                )
            rows = conn.execute(
                "SELECT message_count FROM session_state ORDER BY session_id"
            ).fetchall()
            catalog = transfer_catalog(conn)
            assert catalog is not None
            print(
                json.dumps(
                    {
                        "message_counts": [row[0] for row in rows],
                        "phases": [row.phase for row in catalog.load()],
                    }
                )
            )
            return
        if action == "embed_rejected":
            import pickle

            from recall.services.embed_phase import commit_prepared_embed_cycle

            with (config.data_dir / "prepared-cycle.pkl").open("rb") as stream:
                prepared = pickle.load(stream)
            wrapped = cast(
                duckdb.DuckDBPyConnection,
                InterruptedConnection(conn, when, boundary_statement="ROLLBACK"),
            )
            commit_prepared_embed_cycle(prepared, conn=wrapped)
            raise AssertionError("custody rejection boundary not reached")
        if action == "rebind":
            from recall.core.types import default_session_host
            from recall.services.moves import plan_moved_duplicates
            from recall.services.transfers import rebind

            plan = plan_moved_duplicates(conn, host=default_session_host())[0]
            wrapped = cast(duckdb.DuckDBPyConnection, InterruptedConnection(conn, when))
            rebind(wrapped, plan)
            raise AssertionError("rebind did not reach its boundary")
        catalog = transfer_catalog(conn)
        assert catalog is not None
        row = catalog.load()[0]
        wrapped = cast(
            duckdb.DuckDBPyConnection,
            InterruptedConnection(
                conn,
                when,
                commit_number=2 if action in ("stable_refusal", "keyword_retirement") else 1,
            ),
        )
        if action == "transition":
            assert row.survivor_orphan_count is not None
            cursor = OwnershipCursor(
                family="complete",
                messages=row.survivor_message_count,
                attached=row.survivor_tool_count - row.survivor_orphan_count,
                orphans=row.survivor_orphan_count,
            )
            begin_shrink(wrapped, row, cursor)
        elif action in (
            "raw",
            "first_refusal",
            "stable_refusal",
            "covered_refusal",
            "accepted_rewrite",
        ):
            from recall.parsers.claude_code import ClaudeCodeParser

            from tests.test_services.test_checkpoint_resume_fallbacks import Harness

            Harness(
                config,
                wrapped,
                ClaudeCodeParser(roots=(Path(row.moved_path).parent,)),
                Path(row.moved_path),
            ).reconcile(full=action != "raw")
        elif action == "verification":
            from recall.services import transfers

            original = transfers._message_page
            barrier = InterruptedConnection(conn, when)

            def page(
                conn: duckdb.DuckDBPyConnection, session_id: str, indices: tuple[int, ...]
            ) -> transfers.FactPage:
                if session_id == row.predecessor_id and when == "before":
                    barrier.pause()
                result = original(conn, session_id, indices)
                if session_id == row.predecessor_id and when == "after":
                    barrier.pause()
                return result

            from unittest.mock import patch

            with patch.object(transfers, "_message_page", page):
                TransferDriver().prepare(conn)
        else:
            driver = TransferDriver()
            if action == "coverage":
                driver.prepare(conn)
            driver.turn(wrapped, config)
    raise AssertionError("worker never reached its transaction boundary")


if __name__ == "__main__":
    main()
