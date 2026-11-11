"""A run's persisted success requires a complete, published and settled scope."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import duckdb
import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    ContextConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
    SourceConfig,
)
from recall.core.types import RunKind, Source
from recall.db.fts_sidecar_stage import Publication
from recall.db.schema import ensure_schema
from recall.parsers.codex import CodexParser
from recall.services.indexer import index_sessions
from recall.services.runtime_state import record_run_failure, record_run_success


def _config(tmp_path: Path, root: Path | None) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path,
        db_path=tmp_path / "recall.duckdb",
        lock_path=tmp_path / "recall.lock",
        config_path=tmp_path / "config.toml",
        sources={
            name: SourceConfig(roots=(root,) if name == "codex" and root is not None else ())
            for name in ("codex", "claude_code", "pi_agent", "kimi_code", "grok")
        },
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(context=ContextConfig(mode="off")),
        daemon=DaemonConfig(),
        cli=CliConfig(),
    )


def test_a_returned_partial_index_does_not_replace_the_previous_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "rollout-failed.jsonl").write_text("{}\n")
    config = _config(tmp_path, root)
    previous = datetime(2026, 11, 1)
    with duckdb.connect(str(config.db_path)) as conn:
        ensure_schema(conn)
        record_run_success(conn, run_kind=RunKind.INDEX, successful_at=previous)
        record_run_failure(conn, run_kind=RunKind.INDEX, message="prior failure")

    def failed_parse(*args, **kwargs):
        raise ValueError("source capture failed")

    monkeypatch.setattr(CodexParser, "parse", failed_parse)
    summary = index_sessions(
        source=Source.CODEX, full=True, recreate=False, verbose=False, workers=1, config=config
    )
    assert summary.failed == 1
    with duckdb.connect(str(config.db_path), read_only=True) as fresh:
        row = fresh.execute(
            "SELECT last_successful_at, last_index_failed, last_failure_message FROM runtime_state"
        ).fetchone()
        assert row is not None
        successful, failed, message = row
        assert successful == previous
        assert failed == 1
        assert message


def _spent_database(root: Path) -> Path:
    from uuid import UUID

    from recall.services.owned_jobs import DaemonJobs
    from recall.services.repair_events import RecoveryEvent, publish_event, read_events
    from recall.services.repair_protocol import RepairProtocol

    publish_event(
        root,
        RecoveryEvent(
            event_id=UUID(int=1), failure=None, needs_index_verification=True, probe=True
        ),
    )
    path = root / "recall.duckdb"
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    protocol = RepairProtocol(root)
    with duckdb.connect(str(path)) as conn:
        ensure_schema(conn)
        with jobs.lease(conn) as lease:
            protocol.prepare(lease, read_events(root))
            rebuilt = protocol.rebuild(lease, protocol.arm(lease))
    with duckdb.connect(str(path)) as fresh:
        with jobs.lease(fresh) as lease:
            protocol.verify(lease, rebuilt)
            protocol.retire(lease)
        record_run_success(fresh, run_kind=RunKind.INDEX, successful_at=datetime(2026, 11, 1))
        fresh.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) VALUES "
            "('message','ack','upsert'),('message','ack','upsert')"
        )
    return path


def test_real_library_noop_releases_only_after_inventory_and_startup_sync(tmp_path: Path) -> None:
    path = _spent_database(tmp_path)
    config = _config(tmp_path, None)
    summary = index_sessions(source=None, full=False, recreate=False, verbose=False, config=config)
    assert (summary.total, summary.failed, summary.changed) == (0, 0, 0)
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (True, True)
        row = fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone()
        assert row is not None and row[0] > datetime(2026, 11, 1)


def test_a_late_probe_instruction_prevents_noop_budget_release(tmp_path: Path) -> None:
    from uuid import UUID

    from recall.services.repair_events import RecoveryEvent, publish_event

    path = _spent_database(tmp_path)
    original: bytes | None = None

    def late_instruction(progress) -> None:
        nonlocal original
        if progress.status == "start":
            publish_event(
                tmp_path,
                RecoveryEvent(
                    event_id=UUID(int=2), failure=None, needs_index_verification=True, probe=True
                ),
            )
            original = (tmp_path / "daemon-failure.json").read_bytes()

    summary = index_sessions(
        source=None,
        full=False,
        recreate=False,
        verbose=False,
        config=_config(tmp_path, None),
        progress_callback=late_instruction,
    )
    assert summary.total == 0
    assert original is not None and (tmp_path / "daemon-failure.json").read_bytes() == original
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone() == (
            datetime(2026, 11, 1),
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


def test_failed_discovery_does_not_qualify_as_a_zero_item_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.parsers.discovery import DiscoveryScope

    path = _spent_database(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "rollout-denied.jsonl").write_text("{}\n")
    original = DiscoveryScope._entries

    def denied(scope, directory, owner):
        if directory == root:
            scope.fail(owner, PermissionError("capture unavailable"))
            return None
        return original(scope, directory, owner)

    monkeypatch.setattr(DiscoveryScope, "_entries", denied)
    summary = index_sessions(
        source=Source.CODEX,
        full=False,
        recreate=False,
        verbose=False,
        config=_config(tmp_path, root),
    )
    assert summary.total == 0
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone() == (
            datetime(2026, 11, 1),
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


def _completion(publication: Publication | None = None):
    from recall.core.daemon_status import IndexRunCounts
    from recall.services.run_completion import RunContext, RunRecord, RunScope

    scope = RunScope()
    scope.inventory_completed()
    scope.keyword_acknowledged()
    return scope.prepare(
        RunRecord(
            RunKind.INDEX,
            datetime(2026, 11, 2),
            IndexRunCounts(total=1, indexed=1, skipped=0, failed=0, changed=1, total_seconds=0.25),
            RunContext(messages=1, mode="template"),
        ),
        publication or Publication("nothing"),
    )


@pytest.mark.parametrize(
    "stage",
    [
        "after_completion_seed",
        "after_completion_retirement",
        "after_completion_history",
        "after_completion_counts",
        "after_completion_flags",
        "before_completion_commit",
    ],
)
@pytest.mark.parametrize("fatal", [False, True])
def test_completion_error_keeps_old_history_flags_and_duplicate_intents(
    tmp_path: Path, stage: str, fatal: bool
) -> None:
    from recall.services.owned_jobs import DaemonJobs
    from recall.services.run_completion import record_completion_on

    path = _spent_database(tmp_path)
    error = (
        duckdb.FatalException("Failed to delete all rows from index")
        if fatal
        else OSError("injected completion failure")
    )
    observed: list[BaseException] = []
    jobs = DaemonJobs(lambda error, site, role: observed.append(error), lambda: None)

    def fail(at: str) -> None:
        if at == stage:
            raise error

    with duckdb.connect(str(path)) as conn, jobs.lease(conn) as lease:
        permit = lease.completion_permit()
        with pytest.raises(type(error)) as caught:
            record_completion_on(
                lease,
                permit,
                _completion(Publication("published", {("message", "ack"): 2})),
                barrier=fail,
                data_dir=tmp_path,
            )
        assert caught.value is error
        if not fatal:
            with pytest.raises(AssertionError, match="consumed"):
                record_completion_on(lease, permit, _completion(), data_dir=tmp_path)
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT last_successful_at,last_index_total FROM runtime_state"
        ).fetchone() == (datetime(2026, 11, 1), None)
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)
        assert fresh.execute(
            "SELECT count(*) FROM fts_sidecar_pending WHERE id='ack'"
        ).fetchone() == (2,)
    assert observed == ([error] if fatal else [])


def test_acknowledgement_failure_is_not_retried_and_fresh_read_resolves_commit(
    tmp_path: Path,
) -> None:
    from recall.services.owned_jobs import DaemonJobs
    from recall.services.run_completion import CompletionCommitUncertain, record_completion_on

    path = _spent_database(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    error = OSError("lost acknowledgement")

    def fail(stage: str) -> None:
        if stage == "after_completion_commit":
            raise error

    with duckdb.connect(str(path)) as conn, jobs.lease(conn) as lease:
        permit = lease.completion_permit()
        with pytest.raises(CompletionCommitUncertain) as caught:
            record_completion_on(
                lease,
                permit,
                _completion(Publication("published", {("message", "ack"): 2})),
                completed_at=datetime(2026, 11, 3),
                barrier=fail,
                data_dir=tmp_path,
            )
        assert caught.value.__cause__ is error
        with pytest.raises(AssertionError, match="consumed"):
            record_completion_on(lease, permit, _completion(), data_dir=tmp_path)
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT last_successful_at,last_index_total,last_context_messages FROM runtime_state"
        ).fetchone() == (datetime(2026, 11, 3), 1, 1)
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (True, True)
        assert fresh.execute(
            "SELECT count(*) FROM fts_sidecar_pending WHERE id='ack'"
        ).fetchone() == (0,)


def test_unretired_older_duplicate_prevents_qualified_release(tmp_path: Path) -> None:
    from recall.services.owned_jobs import DaemonJobs
    from recall.services.run_completion import record_completion_on

    path = _spent_database(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    with duckdb.connect(str(path)) as conn, jobs.lease(conn) as lease:
        record_completion_on(
            lease,
            lease.completion_permit(),
            _completion(Publication("published", {("message", "ack"): 1})),
            data_dir=tmp_path,
        )
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone() == (
            datetime(2026, 11, 1),
            "published scope still has unresolved keyword intents",
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)
        assert fresh.execute(
            "SELECT count(*) FROM fts_sidecar_pending WHERE id='ack'"
        ).fetchone() == (2,)


def test_scope_evidence_observed_after_preparation_cannot_be_lost(tmp_path: Path) -> None:
    from recall.services.owned_jobs import DaemonJobs
    from recall.services.run_completion import RunRecord, RunScope, record_completion_on

    path = _spent_database(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    scope = RunScope()
    scope.inventory_completed()
    scope.keyword_acknowledged()
    prepared = scope.prepare(
        RunRecord(RunKind.INDEX, datetime(2026, 11, 2)), Publication("nothing")
    )
    scope.incomplete("the client departed before completion")
    with duckdb.connect(str(path)) as conn, jobs.lease(conn) as lease:
        record_completion_on(lease, lease.completion_permit(), prepared, data_dir=tmp_path)
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone() == (datetime(2026, 11, 1), "the client departed before completion")
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


@pytest.mark.parametrize("cancel_first", [False, True])
def test_departure_and_completion_have_one_scope_order(tmp_path: Path, cancel_first: bool) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from recall.services.owned_jobs import DaemonJobs
    from recall.services.run_completion import RunRecord, RunScope, record_completion_on

    path = _spent_database(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    scope = RunScope()
    scope.inventory_completed()
    scope.keyword_acknowledged()
    prepared = scope.prepare(
        RunRecord(RunKind.INDEX, datetime(2026, 11, 2)), Publication("nothing")
    )
    entered = Event()
    departing = Event()

    def barrier(stage: str) -> None:
        if stage == "after_completion_history":
            entered.set()
            assert departing.wait(5)

    def cancel() -> bool:
        assert entered.wait(5)
        departing.set()
        return scope.cancel("cancelled: the client departed")

    with duckdb.connect(str(path)) as conn, jobs.lease(conn) as lease:
        if cancel_first:
            assert scope.cancel("cancelled: the client departed")
            record_completion_on(lease, lease.completion_permit(), prepared, data_dir=tmp_path)
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                worker = executor.submit(cancel)
                record_completion_on(
                    lease, lease.completion_permit(), prepared, barrier=barrier, data_dir=tmp_path
                )
                assert not worker.result(timeout=5)
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (not cancel_first, not cancel_first)
        row = fresh.execute("SELECT last_failure_message FROM runtime_state").fetchone()
        assert row == ("cancelled: the client departed" if cancel_first else None,)


@pytest.mark.parametrize("complete", [False, True])
def test_zero_scope_qualifies_only_after_inventory_and_keyword_acknowledgement(
    tmp_path: Path, complete: bool
) -> None:
    from recall.services.owned_jobs import DaemonJobs
    from recall.services.run_completion import RunRecord, RunScope, record_completion_on

    path = _spent_database(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    scope = RunScope()
    if complete:
        scope.inventory_completed()
        scope.keyword_acknowledged()
    with duckdb.connect(str(path)) as conn, jobs.lease(conn) as lease:
        record_completion_on(
            lease,
            lease.completion_permit(),
            scope.prepare(RunRecord(RunKind.INDEX, datetime(2026, 11, 2)), Publication("nothing")),
            completed_at=datetime(2026, 11, 3),
            data_dir=tmp_path,
        )
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone() == (
            datetime(2026, 11, 3) if complete else datetime(2026, 11, 1),
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (complete, complete)


@pytest.mark.parametrize("fatal_first", [False, True])
def test_fatal_and_completion_have_one_arbiter_order(tmp_path: Path, fatal_first: bool) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from recall.services.owned_jobs import DaemonJobs, WorkNotAdmitted
    from recall.services.run_completion import record_completion_on

    path = _spent_database(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: None, lambda: None)
    original = duckdb.FatalException("Failed to delete all rows from index")
    entered = Event()
    fatal_started = Event()

    def barrier(stage: str) -> None:
        if stage == "after_completion_history":
            entered.set()
            assert fatal_started.wait(5)

    def fatal() -> None:
        assert entered.wait(5)
        fatal_started.set()
        jobs.observe(original, "other_worker")

    with duckdb.connect(str(path)) as conn, jobs.lease(conn) as lease:
        permit = lease.completion_permit()
        if fatal_first:
            jobs.observe(original, "other_worker")
            with pytest.raises(WorkNotAdmitted):
                record_completion_on(lease, permit, _completion(), data_dir=tmp_path)
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                worker = executor.submit(fatal)
                record_completion_on(
                    lease, permit, _completion(), barrier=barrier, data_dir=tmp_path
                )
                worker.result(timeout=5)
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (not fatal_first, not fatal_first)


@pytest.mark.parametrize("kind", ["probe", "index-divergence", "disk-full", "other", "replay"])
def test_retained_instruction_allows_partial_history_but_no_budget_release(
    tmp_path: Path, kind: Literal["probe", "index-divergence", "disk-full", "other", "replay"]
) -> None:
    from uuid import UUID

    from recall.services.owned_jobs import DaemonJobs
    from recall.services.repair_events import EventFailure, RecoveryEvent, publish_event
    from recall.services.run_completion import record_completion_on

    path = _spent_database(tmp_path)
    event = RecoveryEvent(
        event_id=UUID(int=1 if kind == "replay" else 2),
        failure=(
            None
            if kind in {"probe", "replay"}
            else EventFailure(
                site="run",
                exception_type="Error",
                message="retained observation",
                signature="signature",
                failure_class=kind,
                at=datetime(2026, 11, 2),
            )
        ),
        needs_index_verification=True,
        probe=kind in {"probe", "replay"},
    )
    published = publish_event(tmp_path, event)
    original = published.read_bytes()
    shutdowns: list[bool] = []
    jobs = DaemonJobs(
        lambda error, site, role: pytest.fail("no fabricated fatal"), lambda: shutdowns.append(True)
    )
    with duckdb.connect(str(path)) as conn, jobs.lease(conn) as lease:
        record_completion_on(lease, lease.completion_permit(), _completion(), data_dir=tmp_path)
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT last_successful_at,last_attempted_at,last_index_total FROM runtime_state"
        ).fetchone() == (datetime(2026, 11, 1), datetime(2026, 11, 2), 1)
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)
    assert published.read_bytes() == original
    assert not shutdowns
    assert not (tmp_path / "daemon-refused").exists()


@pytest.mark.parametrize(
    "invalid", ["refusal", "unreadable-refusal", "record", "count", "bytes", "counter"]
)
def test_invalid_completion_admission_preserves_all_effects_and_closes_owner(
    tmp_path: Path, invalid: str
) -> None:
    from uuid import UUID

    from recall.services.owned_jobs import DaemonJobs, WorkNotAdmitted
    from recall.services.repair_events import RecoveryEvent
    from recall.services.run_completion import record_completion_on
    from recall.services.self_repair import DaemonStartupRefused

    path = _spent_database(tmp_path)
    shutdowns: list[bool] = []
    jobs = DaemonJobs(
        lambda error, site, role: pytest.fail("no fabricated fatal"), lambda: shutdowns.append(True)
    )
    with duckdb.connect(str(path)) as conn, jobs.lease(conn) as lease:
        if invalid == "refusal":
            (tmp_path / "daemon-refused").write_text("manual repair required\n")
        elif invalid == "unreadable-refusal":
            (tmp_path / "daemon-refused").mkdir()
        elif invalid == "record":
            (tmp_path / "daemon-failure.json").write_bytes(b'{"version":2,')
        elif invalid in {"count", "bytes"}:
            spool = tmp_path / "fatal-events"
            spool.mkdir(exist_ok=True)
            for ordinal in range(64 if invalid == "count" else 2):
                event = RecoveryEvent(
                    event_id=UUID(int=100 + ordinal),
                    failure=None,
                    needs_index_verification=True,
                    probe=True,
                )
                # Valid individual JSON records with whitespace count toward bytes.
                payload = event.model_dump_json().encode()
                if invalid == "bytes":
                    payload += b" " * 600_000
                (spool / f"{event.event_id}.json").write_bytes(payload)
        else:
            conn.execute("UPDATE daemon_index_repair SET attempt_no = 18446744073709551615")
        before_runtime = conn.execute("SELECT * FROM runtime_state").fetchall()
        before_repair = conn.execute("SELECT * FROM daemon_index_repair").fetchall()
        before_pending = conn.execute("SELECT * FROM fts_sidecar_pending ORDER BY ALL").fetchall()
        with pytest.raises(DaemonStartupRefused):
            record_completion_on(
                lease,
                lease.completion_permit(),
                _completion(Publication("published", {("message", "ack"): 2})),
                data_dir=tmp_path,
            )
        with pytest.raises(WorkNotAdmitted):
            lease.call("later-write", lambda db: db.execute("DELETE FROM runtime_state"))
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT * FROM runtime_state").fetchall() == before_runtime
        assert fresh.execute("SELECT * FROM daemon_index_repair").fetchall() == before_repair
        assert (
            fresh.execute("SELECT * FROM fts_sidecar_pending ORDER BY ALL").fetchall()
            == before_pending
        )
    assert shutdowns == [True]
    assert (tmp_path / "daemon-refused").exists()


@pytest.mark.parametrize("refusal", [False, True])
def test_instruction_arriving_before_commit_rolls_back_tentative_release(
    tmp_path: Path, refusal: bool
) -> None:
    from uuid import UUID

    from recall.services.owned_jobs import DaemonJobs
    from recall.services.repair_events import RecoveryEvent, publish_event
    from recall.services.run_completion import CompletionAdmissionChanged, record_completion_on
    from recall.services.self_repair import DaemonStartupRefused

    path = _spent_database(tmp_path)
    jobs = DaemonJobs(lambda error, site, role: pytest.fail("no fabricated fatal"), lambda: None)

    def arrive(stage: str) -> None:
        if stage == "before_completion_commit":
            if refusal:
                (tmp_path / "daemon-refused").write_text("manual repair required\n")
            else:
                publish_event(
                    tmp_path,
                    RecoveryEvent(
                        event_id=UUID(int=2),
                        failure=None,
                        needs_index_verification=True,
                        probe=True,
                    ),
                )

    with (
        duckdb.connect(str(path)) as conn,
        jobs.lease(conn) as lease,
        pytest.raises(DaemonStartupRefused if refusal else CompletionAdmissionChanged),
    ):
        record_completion_on(
            lease,
            lease.completion_permit(),
            _completion(Publication("published", {("message", "ack"): 2})),
            data_dir=tmp_path,
            barrier=arrive,
        )
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT last_successful_at,last_index_total FROM runtime_state"
        ).fetchone() == (datetime(2026, 11, 1), None)
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)
        assert fresh.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (2,)


@pytest.mark.parametrize("borrowed", [False, True])
def test_standalone_completion_failure_requires_a_closed_fresh_native_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    borrowed: bool,
) -> None:
    from recall.db.lease import CompletionReadRequired
    from recall.services import indexer
    from recall.services.recovery_admission import RecoveryAdmission

    path = _spent_database(tmp_path)
    config = _config(tmp_path, None)
    record = indexer.record_completion_on
    resolve = RecoveryAdmission.resolve_completion
    parents: list[duckdb.DuckDBPyConnection] = []
    resolutions = 0
    origin = OSError("library completion failed")

    def failed(lease, *args, **kwargs):
        parents.append(lease._connection)

        def barrier(stage):
            if stage == "before_completion_commit":
                raise origin

        return record(lease, *args, **kwargs, barrier=barrier)

    def resolved(admission, failure):
        nonlocal resolutions
        with pytest.raises(duckdb.ConnectionException, match="closed"):
            parents[0].execute("SELECT 1")
        resolve(admission, failure)
        resolutions += 1

    monkeypatch.setattr(indexer, "record_completion_on", failed)
    monkeypatch.setattr(RecoveryAdmission, "resolve_completion", resolved)
    if borrowed:
        with duckdb.connect(str(path)) as parent:
            with pytest.raises(OSError) as caught:
                index_sessions(
                    source=None,
                    full=False,
                    recreate=False,
                    verbose=False,
                    config=config,
                    conn=parent,
                )
            assert caught.value is origin
            with pytest.raises(CompletionReadRequired):
                index_sessions(
                    source=None,
                    full=False,
                    recreate=False,
                    verbose=False,
                    config=config,
                    conn=parent,
                )
            assert resolutions == 0
    else:
        with pytest.raises(OSError) as caught:
            index_sessions(source=None, full=False, recreate=False, verbose=False, config=config)
        assert caught.value is origin
        assert resolutions == 1
    assert len(parents) == 1
    monkeypatch.setattr(indexer, "record_completion_on", record)
    result = index_sessions(source=None, full=False, recreate=False, verbose=False, config=config)
    assert result.total == 0
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (True, True)


def test_owned_standalone_publishes_its_selected_fatal_before_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recall.db.repair import load_repair_row
    from recall.services import indexer
    from recall.services.repair_events import RecoveryEvent, read_events

    path = _spent_database(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "rollout-source.jsonl").write_text(
        '{"type":"session_meta","payload":{"id":"source"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":"hello"}}\n'
    )
    with duckdb.connect(str(path), read_only=True) as fresh:
        before = load_repair_row(fresh)
    origin = duckdb.FatalException("Failed to delete all rows from index")

    def fail(*args, **kwargs):
        raise origin

    monkeypatch.setattr(indexer, "insert_session", fail)
    with pytest.raises(duckdb.FatalException) as caught:
        index_sessions(
            source=Source.CODEX,
            full=True,
            recreate=False,
            verbose=False,
            config=_config(tmp_path, root),
            workers=1,
        )
    assert caught.value is origin
    event = read_events(tmp_path).primary
    assert isinstance(event, RecoveryEvent) and event.failure is not None
    assert event.failure.message == "Failed to delete all rows from index"
    assert event.failure.exception_type == "FatalException" and event.failure.site
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert load_repair_row(fresh) == before
        assert fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone() == (
            datetime(2026, 11, 1),
        )


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_library_reporting_error_cannot_replace_the_producer_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recompute: bool,
    diagnostics: bool,
) -> None:
    from recall.services import indexer, run_completion

    path = _spent_database(tmp_path)
    origin = ValueError("selected inventory callback failed")

    def progress(_update):
        raise origin

    record = indexer.record_completion_on
    reports = diagnostic_attempts = 0

    def diagnostic_unavailable(*args, **kwargs):
        nonlocal diagnostic_attempts
        diagnostic_attempts += 1
        raise OSError("outcome diagnostic unavailable")

    if diagnostics:
        monkeypatch.setattr(run_completion.logger, "error", diagnostic_unavailable)

    def unavailable(lease, *args, **kwargs):
        nonlocal reports
        reports += 1

        def fail(stage):
            if stage == "after_completion_history":
                lease.call(
                    "injected_reporting_error",
                    lambda conn: conn.execute(
                        "SELECT CAST('reporting unavailable' AS INTEGER)"
                    ).fetchone(),
                )

        return record(lease, *args, **kwargs, barrier=fail)

    monkeypatch.setattr(indexer, "record_completion_on", unavailable)
    if recompute:
        monkeypatch.setattr(
            indexer, "_load_context_recompute_groups", lambda *args, **kwargs: progress(None)
        )
    with pytest.raises(ValueError) as caught:
        if recompute:
            indexer.recompute_context_for_rows(config=_config(tmp_path, None))
        else:
            index_sessions(
                source=None,
                full=False,
                recreate=False,
                verbose=False,
                config=_config(tmp_path, None),
                progress_callback=progress,
            )
    assert caught.value is origin
    assert reports == 1
    assert diagnostic_attempts == int(diagnostics)
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone() == (
            datetime(2026, 11, 1),
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


@pytest.mark.parametrize("recompute", [False, True])
def test_actual_library_cancellation_records_a_cancelled_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recompute: bool,
) -> None:
    import asyncio

    from recall.services import indexer

    path = _spent_database(tmp_path)
    origin = asyncio.CancelledError("caller departed")

    def depart(*args, **kwargs):
        raise origin

    if recompute:
        monkeypatch.setattr(indexer, "_load_context_recompute_groups", depart)
    with pytest.raises(asyncio.CancelledError) as caught:
        if recompute:
            indexer.recompute_context_for_rows(config=_config(tmp_path, None))
        else:
            index_sessions(
                source=None,
                full=False,
                recreate=False,
                verbose=False,
                config=_config(tmp_path, None),
                progress_callback=depart,
            )
    assert caught.value is origin
    with duckdb.connect(str(path), read_only=True) as fresh:
        row = fresh.execute(
            "SELECT last_successful_at,last_failure_message FROM runtime_state"
        ).fetchone()
        assert row is not None and row[0] == datetime(2026, 11, 1)
        assert row[1] is not None and row[1].startswith("cancelled:")
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


@pytest.mark.parametrize("recompute", [False, True])
def test_library_failed_outcome_preserves_success_and_records_its_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recompute: bool,
) -> None:
    from recall.services import indexer

    path = _spent_database(tmp_path)
    origin = ValueError("selected check failed")

    def failed(*args, **kwargs):
        raise origin

    if recompute:
        monkeypatch.setattr(indexer, "_load_context_recompute_groups", failed)
    with pytest.raises(ValueError) as caught:
        if recompute:
            indexer.recompute_context_for_rows(config=_config(tmp_path, None))
        else:
            index_sessions(
                source=None,
                full=False,
                recreate=False,
                verbose=False,
                config=_config(tmp_path, None),
                progress_callback=failed,
            )
    assert caught.value is origin
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute(
            "SELECT last_successful_at,last_failure_message,last_index_total FROM runtime_state"
        ).fetchone() == (datetime(2026, 11, 1), "selected check failed", None)
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)


def test_owned_file_cleanup_keeps_the_primary_and_spools_the_later_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recall.db import lease as leases
    from recall.services import indexer
    from recall.services.repair_events import RecoveryEvent, read_events

    _spent_database(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "rollout-source.jsonl").write_text(
        '{"type":"session_meta","payload":{"id":"source"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":"hello"}}\n'
    )
    origin = duckdb.FatalException("Failed to delete all rows from index")
    cleanup = duckdb.FatalException("database has been invalidated during close")
    close = leases._OwnedConnection.close
    before_close: bytes | None = None

    def failed(*args, **kwargs):
        raise origin

    def close_failed(owned):
        nonlocal before_close
        before_close = (tmp_path / "daemon-failure.json").read_bytes()
        close(owned)
        raise cleanup

    monkeypatch.setattr(indexer, "insert_session", failed)
    monkeypatch.setattr(leases._OwnedConnection, "close", close_failed)
    with pytest.raises(BaseExceptionGroup) as caught:
        index_sessions(
            source=Source.CODEX,
            full=True,
            recreate=False,
            verbose=False,
            config=_config(tmp_path, root),
            workers=1,
        )
    assert caught.value.subgroup(lambda error: error is origin) is not None
    assert caught.value.subgroup(lambda error: error is cleanup) is not None
    assert (
        before_close is not None and (tmp_path / "daemon-failure.json").read_bytes() == before_close
    )
    events = read_events(tmp_path)
    assert isinstance(events.primary, RecoveryEvent) and events.primary.failure is not None
    assert events.primary.failure.message == "Failed to delete all rows from index"
    assert len(events.secondary) == 1
    later = events.secondary[0][1]
    assert isinstance(later, RecoveryEvent) and later.failure is not None
    assert later.failure.message == "database has been invalidated during close"


def test_owned_file_disk_observation_never_replaces_a_retained_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import errno
    from uuid import UUID

    from recall.services import indexer
    from recall.services.repair_events import RecoveryEvent, publish_event, read_events

    path = _spent_database(tmp_path)
    root = tmp_path / "sessions"
    root.mkdir()
    (root / "rollout-source.jsonl").write_text(
        '{"type":"session_meta","payload":{"id":"source"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"user","content":"hello"}}\n'
    )
    original: bytes | None = None

    def full(*args, **kwargs):
        nonlocal original
        publish_event(
            tmp_path,
            RecoveryEvent(
                event_id=UUID(int=2), failure=None, needs_index_verification=True, probe=True
            ),
        )
        original = (tmp_path / "daemon-failure.json").read_bytes()
        raise OSError(errno.ENOSPC, "injected disk capacity failure")

    monkeypatch.setattr(indexer, "insert_session", full)
    summary = index_sessions(
        source=Source.CODEX,
        full=True,
        recreate=False,
        verbose=False,
        config=_config(tmp_path, root),
        workers=1,
    )
    assert summary.failed == 1
    assert original is not None and (tmp_path / "daemon-failure.json").read_bytes() == original
    events = read_events(tmp_path)
    assert len(events.secondary) == 1
    disk = events.secondary[0][1]
    assert isinstance(disk, RecoveryEvent) and disk.failure is None
    assert disk.needs_index_verification and not disk.probe
    with duckdb.connect(str(path), read_only=True) as fresh:
        assert fresh.execute("SELECT last_successful_at FROM runtime_state").fetchone() == (
            datetime(2026, 11, 1),
        )
        assert fresh.execute(
            "SELECT success_since_attempt,success_since_failure FROM daemon_index_repair"
        ).fetchone() == (False, False)
