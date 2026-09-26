"""Failure-signature memory and startup self-repair (REQ-RESIL-014..016, REQ-RESIL-019).

The daemon crash-looped 1,679 times on one persistent DuckDB fault because the
invalidated instance could not record its own failure and nothing compared one
run's death with the next. These tests pin the memory (marker file outside
DuckDB, folded into runtime_state on the next start), the repair it drives, and
the refusal that stops an identical restart.
"""

from __future__ import annotations

import errno
import json
import logging
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest
from conftest import _can_acquire_duckdb_lock
from recall.core.config import AppConfig
from recall.core.types import RunKind
from recall.db import connect
from recall.db.maintenance import DivergedKey, IndexDivergenceReport
from recall.services import self_repair
from recall.services.runtime_state import (
    load_runtime_status,
    record_run_success,
    set_needs_index_verification,
)
from recall.services.self_repair import (
    DaemonStartupRefused,
    FailureMarker,
    build_failure_record,
    classify_failure,
    clear_failure_marker,
    clear_fatal_memory,
    marker_path,
    normalize_failure_signature,
    read_failure_marker,
    remember_disk_full,
    remember_fatal_failure,
    run_startup_self_repair,
    write_failure_marker,
)

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)

INDEX_FATAL_A = (
    "FATAL Error: Invalid Input Error: Failed to delete all rows from index. "
    "Only deleted 0 out of 1 rows.\n"
    "Chunk: Chunk - [20 Columns] - FLAT VARCHAR: 1 = [ eae5a5d3f0c14b2e9a7d6c5b4a392817]"
)
INDEX_FATAL_B = (
    "FATAL Error: Invalid Input Error: Failed to delete all rows from index. "
    "Only deleted 0 out of 3 rows.\n"
    "Chunk: Chunk - [20 Columns] - FLAT VARCHAR: 3 = [ 46101c8a9b8c7d6e5f4a3b2c1d0e9f8a]"
)
ENOSPC_CHECKPOINT = (
    "FATAL Error: Failed to create checkpoint because of error: Could not fsync file "
    '"/Users/dev/.local/share/recall/recall.duckdb": No space left on device'
)
ENOSPC_WAL = (
    "TransactionContext Error: Failed to commit: Could not write file "
    '"/home/dev/.local/share/recall/recall.duckdb.wal": No space left on device'
)
T0 = datetime(2026, 8, 30, 12, 0, 0)


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / ".local/share/recall"))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / ".config/recall/config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    monkeypatch.setenv("RECALL_EMBED_BACKEND", "onnx")
    config = AppConfig.load()
    connect(config).close()
    return config


def _fatal(message: str) -> duckdb.FatalException:
    return duckdb.FatalException(message)


class TestSignature:
    def test_two_runs_dying_on_different_rows_share_a_signature(self) -> None:
        sig_a = normalize_failure_signature("usage_harvest", _fatal(INDEX_FATAL_A))
        sig_b = normalize_failure_signature("usage_harvest", _fatal(INDEX_FATAL_B))

        assert sig_a == sig_b
        assert sig_a.startswith("usage_harvest:FatalException:")
        assert "eae5a5d3" not in sig_a
        assert "Chunk" not in sig_a, "only the first line of the message is signed"

    def test_site_and_exception_class_distinguish_signatures(self) -> None:
        by_site = normalize_failure_signature("catch_up", _fatal(INDEX_FATAL_A))
        by_class = normalize_failure_signature("usage_harvest", RuntimeError(INDEX_FATAL_A))

        assert by_site != normalize_failure_signature("usage_harvest", _fatal(INDEX_FATAL_A))
        assert by_class.startswith("usage_harvest:RuntimeError:")

    def test_quoted_paths_and_numbers_are_placeholders(self) -> None:
        sig = normalize_failure_signature("checkpoint", _fatal(ENOSPC_CHECKPOINT))

        assert "/Users/dev" not in sig
        assert "No space left on device" in sig


class TestClassify:
    def test_index_divergence(self) -> None:
        assert classify_failure(_fatal(INDEX_FATAL_A)) == "index-divergence"

    def test_disk_full_from_message(self) -> None:
        assert classify_failure(_fatal(ENOSPC_CHECKPOINT)) == "disk-full"
        assert classify_failure(duckdb.TransactionException(ENOSPC_WAL)) == "disk-full"

    def test_disk_full_from_errno(self) -> None:
        assert classify_failure(OSError(errno.ENOSPC, "boom")) == "disk-full"

    def test_other(self) -> None:
        msg = "database has been invalidated because of a previous fatal error"
        assert classify_failure(_fatal(msg)) == "other"
        assert classify_failure(RuntimeError("boom")) == "other"


class TestMarkerFile:
    def test_round_trip_is_atomic_and_leaves_no_temp_file(self, tmp_path: Path) -> None:
        record = build_failure_record(_fatal(INDEX_FATAL_A), site="usage_harvest", now=T0)
        marker = FailureMarker(failure=record, needs_index_verification=False)

        assert write_failure_marker(tmp_path, marker) is True

        assert read_failure_marker(tmp_path) == marker
        assert marker_path(tmp_path).name == "daemon-failure.json"
        assert [p.name for p in tmp_path.iterdir()] == ["daemon-failure.json"]
        payload = json.loads(marker_path(tmp_path).read_text(encoding="utf-8"))
        assert payload["failure"]["failure_class"] == "index-divergence"
        assert payload["failure"]["site"] == "usage_harvest"
        assert payload["failure"]["at"] == "2026-08-30T12:00:00"

    def test_message_is_bounded_to_first_line(self, tmp_path: Path) -> None:
        record = build_failure_record(_fatal("x" * 900 + "\nsecond line"), site="s", now=T0)

        assert len(record.message) <= 500
        assert "second line" not in record.message

    def test_write_failure_is_reported_not_raised(self, tmp_path: Path) -> None:
        blocked = tmp_path / "not-a-dir"
        blocked.write_text("file where a directory is expected", encoding="utf-8")
        record = build_failure_record(_fatal(INDEX_FATAL_A), site="s", now=T0)

        assert write_failure_marker(blocked, FailureMarker(record, False)) is False

    def test_corrupt_marker_reads_as_absent(self, tmp_path: Path) -> None:
        marker_path(tmp_path).write_text("{not json", encoding="utf-8")

        assert read_failure_marker(tmp_path) is None

    def test_clear_is_idempotent(self, tmp_path: Path) -> None:
        clear_failure_marker(tmp_path)
        write_failure_marker(tmp_path, FailureMarker(None, True))
        clear_failure_marker(tmp_path)
        clear_failure_marker(tmp_path)

        assert read_failure_marker(tmp_path) is None


class TestRemember:
    def test_remember_fatal_failure_writes_marker_and_returns_record(self, tmp_path: Path) -> None:
        record = remember_fatal_failure(tmp_path, _fatal(INDEX_FATAL_A), site="catch_up", now=T0)

        assert record is not None
        marker = read_failure_marker(tmp_path)
        assert marker is not None
        assert marker.failure == record
        assert marker.needs_index_verification is False

    def test_disk_full_fatal_also_flags_verification(self, tmp_path: Path) -> None:
        remember_fatal_failure(tmp_path, _fatal(ENOSPC_CHECKPOINT), site="checkpoint", now=T0)

        marker = read_failure_marker(tmp_path)
        assert marker is not None
        assert marker.failure is not None
        assert marker.failure.failure_class == "disk-full"
        assert marker.needs_index_verification is True

    def test_remember_fatal_failure_never_raises(self, tmp_path: Path) -> None:
        blocked = tmp_path / "blocked"
        blocked.write_text("", encoding="utf-8")

        assert remember_fatal_failure(blocked, _fatal(INDEX_FATAL_A), site="s", now=T0) is None

    def test_remember_disk_full_preserves_existing_failure(self, tmp_path: Path) -> None:
        remember_fatal_failure(tmp_path, _fatal(INDEX_FATAL_A), site="catch_up", now=T0)

        assert remember_disk_full(tmp_path) is True

        marker = read_failure_marker(tmp_path)
        assert marker is not None
        assert marker.failure is not None
        assert marker.failure.site == "catch_up"
        assert marker.needs_index_verification is True

    @_requires_duckdb_lock
    def test_remember_disk_full_sets_runtime_state_flag_when_conn_alive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, monkeypatch)
        conn = connect(config)
        try:
            remember_disk_full(config.data_dir, conn=conn)
            assert load_runtime_status(config, conn=conn).needs_index_verification is True
        finally:
            conn.close()


@_requires_duckdb_lock
class TestStartupSelfRepair:
    def test_no_marker_and_no_flag_probes_and_records_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, monkeypatch)

        outcome = run_startup_self_repair(config, now=T0)

        assert outcome.action == "none"
        assert outcome.rebuild is None
        assert outcome.probe is not None and outcome.probe.diverged == ()
        status = load_runtime_status(config)
        assert status.last_failure_message is None
        assert status.last_fatal_signature is None

    def test_index_divergence_marker_rebuilds_records_and_consumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, monkeypatch)
        record = remember_fatal_failure(
            config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0
        )
        assert record is not None

        outcome = run_startup_self_repair(config, now=T0 + timedelta(seconds=30))

        assert outcome.action == "rebuilt"
        assert outcome.rebuild is not None and outcome.rebuild.dropped > 0
        assert outcome.record == record
        status = load_runtime_status(config)
        assert status.last_failure_message is not None
        assert "Failed to delete all rows from index" in status.last_failure_message
        assert "usage_harvest" in status.last_failure_message
        assert status.last_failure_at == T0
        assert status.last_fatal_signature == record.signature
        assert status.fatal_repeat_count == 1
        assert status.last_fatal_at == T0, "the signature is dated, so a stale one reads as stale"
        assert status.last_index_repair_at == T0 + timedelta(seconds=30)
        assert status.last_index_repair_signature == record.signature
        assert read_failure_marker(config.data_dir) is None

    def test_failure_record_is_persisted_before_the_rebuild_can_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-014: the record is the first DB write. A rebuild that raises
        must not leave `last_failure_message` null -- that is the invisible crash
        loop the requirement exists to end."""
        config = _config(tmp_path, monkeypatch)
        record = remember_fatal_failure(
            config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0
        )
        assert record is not None

        def explode(_conn: object) -> None:
            raise RuntimeError("rebuild exploded")

        monkeypatch.setattr(self_repair, "rebuild_indexes", explode)

        with pytest.raises(RuntimeError, match="rebuild exploded"):
            run_startup_self_repair(config, now=T0 + timedelta(seconds=30))

        status = load_runtime_status(config)
        assert status.last_failure_message is not None
        assert "usage_harvest" in status.last_failure_message
        assert status.last_failure_at == T0
        assert status.last_fatal_signature == record.signature
        assert status.fatal_repeat_count == 1
        assert read_failure_marker(config.data_dir) is None, "persisted, so the marker is released"

    def test_rebuild_that_raises_arms_the_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-016: a rebuild that raises is a repair that did not hold.
        The next start on the same signature must refuse, not rebuild again,
        and the refusal leaves the marker the scheduler keys on."""
        config = _config(tmp_path, monkeypatch)
        record = remember_fatal_failure(
            config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0
        )
        assert record is not None
        rebuild_calls = 0

        def explode(_conn: object) -> None:
            nonlocal rebuild_calls
            rebuild_calls += 1
            raise RuntimeError("rebuild exploded")

        monkeypatch.setattr(self_repair, "rebuild_indexes", explode)
        with pytest.raises(RuntimeError, match="rebuild exploded"):
            run_startup_self_repair(config, now=T0 + timedelta(seconds=30))

        status = load_runtime_status(config)
        assert status.last_index_repair_signature == record.signature
        assert status.last_index_repair_at == T0 + timedelta(seconds=30)

        # The un-repaired daemon died again on the same fault.
        remember_fatal_failure(
            config.data_dir,
            _fatal(INDEX_FATAL_B),
            site="usage_harvest",
            now=T0 + timedelta(minutes=2),
        )
        with pytest.raises(DaemonStartupRefused):
            run_startup_self_repair(config, now=T0 + timedelta(minutes=3))

        assert rebuild_calls == 1, "a refusal never retries the rebuild"
        assert self_repair.refusal_marker_path(config.data_dir).exists()

    def test_repair_that_cannot_be_armed_refuses_instead_of_rebuilding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-016: an unrecorded repair attempt would be retried on every
        start, so when the arming write fails the rebuild is not run and the
        start refuses with the manual fix."""
        config = _config(tmp_path, monkeypatch)
        remember_fatal_failure(config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0)
        rebuild_calls = 0

        def counting_rebuild(_conn: object) -> None:
            nonlocal rebuild_calls
            rebuild_calls += 1
            raise AssertionError("rebuild must not run unarmed")

        def refuse_arming(*_args: object, **_kwargs: object) -> None:
            raise duckdb.IOException("Could not write file: No space left on device")

        monkeypatch.setattr(self_repair, "rebuild_indexes", counting_rebuild)
        monkeypatch.setattr(self_repair, "record_index_repair", refuse_arming)

        with pytest.raises(DaemonStartupRefused) as excinfo:
            run_startup_self_repair(config, now=T0 + timedelta(seconds=30))

        assert rebuild_calls == 0
        assert excinfo.value.exit_code == 3
        assert "recall db rebuild-indexes" in str(excinfo.value)
        assert self_repair.refusal_marker_path(config.data_dir).exists()

    def test_stale_schema_rebuilds_unarmed_instead_of_refusing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """REQ-RESIL-016: a runtime_state that predates migration 0023 cannot arm
        a repair at all. The lenient schema opened it so the operator can
        recreate, and recreate needs the daemon listening, so the start rebuilds
        unarmed and serves with a warning naming the fix instead of refusing."""
        config = _config(tmp_path, monkeypatch)
        remember_fatal_failure(config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0)
        conn = connect(config)
        try:
            for column in (
                "last_fatal_signature",
                "fatal_repeat_count",
                "last_index_repair_at",
                "last_index_repair_signature",
                "needs_index_verification",
            ):
                conn.execute(f"ALTER TABLE runtime_state DROP COLUMN {column}")
        finally:
            conn.close()

        with caplog.at_level(logging.WARNING):
            outcome = run_startup_self_repair(config, now=T0 + timedelta(seconds=30))

        assert outcome.action == "rebuilt"
        assert outcome.rebuild is not None and outcome.rebuild.dropped > 0
        assert not self_repair.refusal_marker_path(config.data_dir).exists()
        assert any(
            "recall index --recreate --yes" in record.getMessage() for record in caplog.records
        ), "the warning must name the fix the refusal can no longer name"

    def test_clean_start_releases_an_orphaned_refusal_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-024: a start that completes the repair step is serving, so
        the refusal is over; a leftover marker must not hold launchd down."""
        config = _config(tmp_path, monkeypatch)
        self_repair.write_refusal_marker(config.data_dir, "refusing to start: stale")

        outcome = run_startup_self_repair(config, now=T0)

        assert outcome.action == "none"
        assert not self_repair.refusal_marker_path(config.data_dir).exists()

    def test_fatal_while_arming_propagates_instead_of_refusing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-015/016: a dead instance is not a failed arming write. The
        fatal must reach the funnel and the relaunch, not be relabelled a
        refusal that holds the scheduler down."""
        config = _config(tmp_path, monkeypatch)
        remember_fatal_failure(config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0)
        rebuild_calls = 0

        def counting_rebuild(_conn: object) -> None:
            nonlocal rebuild_calls
            rebuild_calls += 1

        def dead_instance(*_args: object, **_kwargs: object) -> None:
            raise duckdb.FatalException(
                "database has been invalidated because of a previous fatal error"
            )

        monkeypatch.setattr(self_repair, "rebuild_indexes", counting_rebuild)
        monkeypatch.setattr(self_repair, "record_index_repair", dead_instance)

        with pytest.raises(duckdb.FatalException):
            run_startup_self_repair(config, now=T0 + timedelta(seconds=30))

        assert rebuild_calls == 0
        assert not self_repair.refusal_marker_path(config.data_dir).exists()

    def test_fatal_while_persisting_the_record_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """INV-RESIL-004: the runtime_state writes tolerate a stale schema, never
        a dead instance -- swallowing it would run the rebuild on a corpse."""
        config = _config(tmp_path, monkeypatch)
        remember_fatal_failure(config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0)

        def dead_instance(*_args: object, **_kwargs: object) -> None:
            raise duckdb.FatalException(
                "database has been invalidated because of a previous fatal error"
            )

        monkeypatch.setattr(self_repair, "record_fatal_failure", dead_instance)

        with pytest.raises(duckdb.FatalException):
            run_startup_self_repair(config, now=T0 + timedelta(seconds=30))

        assert read_failure_marker(config.data_dir) is not None, "the marker stays for the relaunch"

    def test_daemon_status_surfaces_the_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-024: a refused start is exactly when the daemon is down, so
        the status the operator can still run must say why."""
        from recall.services.daemon import daemon_status

        config = _config(tmp_path, monkeypatch)
        assert daemon_status(config).startup_refusal is None

        self_repair.write_refusal_marker(
            config.data_dir, "refusing to start: recurred. Manual fix: x"
        )

        assert daemon_status(config).startup_refusal == "refusing to start: recurred. Manual fix: x"

    def test_clearing_the_failure_marker_also_clears_the_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-024: the refusal marker lives and dies with the failure
        marker, so `recall db rebuild-indexes` / `recall compact` let the
        scheduler relaunch the daemon."""
        config = _config(tmp_path, monkeypatch)
        remember_fatal_failure(config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0)
        run_startup_self_repair(config, now=T0 + timedelta(seconds=30))
        remember_fatal_failure(
            config.data_dir,
            _fatal(INDEX_FATAL_B),
            site="usage_harvest",
            now=T0 + timedelta(minutes=2),
        )
        with pytest.raises(DaemonStartupRefused):
            run_startup_self_repair(config, now=T0 + timedelta(minutes=3))
        assert self_repair.refusal_marker_path(config.data_dir).exists()

        clear_failure_marker(config.data_dir)

        assert not self_repair.refusal_marker_path(config.data_dir).exists()
        assert read_failure_marker(config.data_dir) is None

    def test_marker_survives_when_persisting_the_record_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-RESIL-014: the marker is released only once the record committed;
        a failed write leaves it in place for the next start."""
        config = _config(tmp_path, monkeypatch)
        record = remember_fatal_failure(
            config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0
        )
        assert record is not None

        def refuse_write(*_args: object, **_kwargs: object) -> None:
            raise duckdb.CatalogException("runtime_state has no column last_fatal_signature")

        monkeypatch.setattr(self_repair, "record_fatal_failure", refuse_write)

        outcome = run_startup_self_repair(config, now=T0 + timedelta(seconds=30))

        assert outcome.action == "rebuilt"
        marker = read_failure_marker(config.data_dir)
        assert marker is not None, "an unpersisted record keeps its marker"
        assert marker.failure == record

    def test_same_signature_after_repair_refuses_with_manual_fix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, monkeypatch)
        remember_fatal_failure(config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0)
        run_startup_self_repair(config, now=T0 + timedelta(seconds=30))
        # The repaired daemon died again on the same fault (different row).
        remember_fatal_failure(
            config.data_dir,
            _fatal(INDEX_FATAL_B),
            site="usage_harvest",
            now=T0 + timedelta(minutes=2),
        )

        with pytest.raises(DaemonStartupRefused) as excinfo:
            run_startup_self_repair(config, now=T0 + timedelta(minutes=3))

        assert excinfo.value.exit_code == 3
        assert "recall db rebuild-indexes" in str(excinfo.value)
        status = load_runtime_status(config)
        assert status.last_failure_message is not None
        assert "recall daemon stop" in status.last_failure_message
        assert "recall db rebuild-indexes" in status.last_failure_message
        assert status.fatal_repeat_count == 2
        assert read_failure_marker(config.data_dir) is not None, "refusal keeps the marker"

    def test_refusal_repeats_cheaply_until_memory_is_cleared(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, monkeypatch)
        remember_fatal_failure(config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0)
        run_startup_self_repair(config, now=T0 + timedelta(seconds=30))
        remember_fatal_failure(
            config.data_dir,
            _fatal(INDEX_FATAL_B),
            site="usage_harvest",
            now=T0 + timedelta(minutes=2),
        )
        with pytest.raises(DaemonStartupRefused):
            run_startup_self_repair(config, now=T0 + timedelta(minutes=3))
        with pytest.raises(DaemonStartupRefused):
            run_startup_self_repair(config, now=T0 + timedelta(minutes=4))

        conn = connect(config)
        try:
            clear_fatal_memory(config, conn)
        finally:
            conn.close()

        outcome = run_startup_self_repair(config, now=T0 + timedelta(minutes=5))
        assert outcome.action == "none"
        status = load_runtime_status(config)
        assert status.last_failure_message is None
        assert status.fatal_repeat_count == 0
        assert status.last_fatal_at is None
        assert status.last_index_repair_signature is None

    def test_success_since_repair_allows_repairing_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, monkeypatch)
        remember_fatal_failure(config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0)
        run_startup_self_repair(config, now=T0 + timedelta(seconds=30))
        conn = connect(config)
        try:
            record_run_success(
                conn, run_kind=RunKind.DAEMON_WATCH, successful_at=T0 + timedelta(hours=1)
            )
        finally:
            conn.close()
        remember_fatal_failure(
            config.data_dir, _fatal(INDEX_FATAL_A), site="usage_harvest", now=T0 + timedelta(days=1)
        )

        outcome = run_startup_self_repair(config, now=T0 + timedelta(days=1, seconds=5))

        assert outcome.action == "rebuilt"
        assert load_runtime_status(config).fatal_repeat_count == 1

    def test_other_class_is_recorded_without_repair(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, monkeypatch)
        remember_fatal_failure(
            config.data_dir,
            _fatal("database has been invalidated because of a previous fatal error"),
            site="embed_loop",
            now=T0,
        )

        outcome = run_startup_self_repair(config, now=T0 + timedelta(seconds=10))

        assert outcome.action == "recorded"
        assert outcome.rebuild is None
        status = load_runtime_status(config)
        assert status.last_failure_message is not None
        assert "embed_loop" in status.last_failure_message
        assert status.last_index_repair_at is None
        assert read_failure_marker(config.data_dir) is None

    def test_needs_index_verification_flag_runs_probe_and_clears_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, monkeypatch)
        conn = connect(config)
        try:
            set_needs_index_verification(conn, True)
        finally:
            conn.close()
        write_failure_marker(config.data_dir, FailureMarker(None, True))

        outcome = run_startup_self_repair(config, now=T0)

        assert outcome.action == "none"
        assert outcome.probe is not None
        assert load_runtime_status(config).needs_index_verification is False
        assert read_failure_marker(config.data_dir) is None

    def test_disk_full_marker_rebuilds_when_probe_reports_divergence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, monkeypatch)
        remember_fatal_failure(
            config.data_dir, _fatal(ENOSPC_CHECKPOINT), site="checkpoint", now=T0
        )
        diverged = IndexDivergenceReport(
            checked_at=T0,
            indexes_probed=1,
            samples_checked=1,
            samples_unverifiable=0,
            diverged=(
                DivergedKey(
                    table="session_state",
                    column="git_repo",
                    key="/Users/dev/code/app",
                    index_count=1790,
                    full_count=1797,
                ),
            ),
            complete=True,
            elapsed_seconds=0.01,
        )
        clean = replace(diverged, diverged=())
        answers = iter([diverged, clean])
        monkeypatch.setattr(self_repair, "probe_index_divergence", lambda *a, **k: next(answers))

        outcome = run_startup_self_repair(config, now=T0 + timedelta(seconds=5))

        assert outcome.action == "rebuilt"
        assert outcome.probe == clean, "the cached report is the post-repair probe"
        status = load_runtime_status(config)
        assert status.needs_index_verification is False
        assert status.last_index_repair_at == T0 + timedelta(seconds=5)

    def test_disk_full_marker_without_divergence_does_not_rebuild(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, monkeypatch)
        remember_fatal_failure(
            config.data_dir, _fatal(ENOSPC_CHECKPOINT), site="checkpoint", now=T0
        )

        outcome = run_startup_self_repair(config, now=T0 + timedelta(seconds=5))

        assert outcome.action == "recorded"
        status = load_runtime_status(config)
        assert status.last_failure_message is not None
        assert "No space left on device" in status.last_failure_message
        assert status.needs_index_verification is False
