"""Regressions from the reference host incident.

A 35k-session host could not complete `recall index --full`: six consecutive
runs aborted the daemon with an uncaught DuckDB ``FatalException``. The cause
was update churn -- the database was 52% dead space -- and compacting it
(23.68 GiB -> 11.90 GiB) made the identical run succeed.

Three defects fall out of that, one per section below.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import pytest
from recall.core.config import DEFAULT_COMPACTION_BLOAT_THRESHOLD

# --- the auto-compaction threshold sat above the failure point --------

# Measured on the failing host immediately before compaction.
OBSERVED_FAILING_RATIO_REPORTED = 1.906
OBSERVED_FAILING_RATIO_TRUE = 23.68 / 11.90  # == 1.990


def test_default_bloat_threshold_is_below_the_observed_failure_point() -> None:
    """The host that could not re-parse read 1.906 against a 2.0 default, so
    auto-compaction never fired. Compaction is preventive maintenance; a
    trigger set at the level where the database is *already* broken can never
    prevent anything."""
    assert DEFAULT_COMPACTION_BLOAT_THRESHOLD < OBSERVED_FAILING_RATIO_REPORTED
    assert DEFAULT_COMPACTION_BLOAT_THRESHOLD < OBSERVED_FAILING_RATIO_TRUE


def test_default_bloat_threshold_leaves_a_healthy_database_alone() -> None:
    """A freshly compacted database read 1.150. The threshold must sit above
    that, or every daemon start would compact."""
    assert DEFAULT_COMPACTION_BLOAT_THRESHOLD > 1.15


# --- the pre-compact backup must not need a second full copy ----------


def test_clone_or_copy_reproduces_bytes_exactly(tmp_path: Path) -> None:
    from recall.services.compaction import _clone_or_copy

    src = tmp_path / "src.bin"
    payload = bytes(range(256)) * 4096
    src.write_bytes(payload)
    dst = tmp_path / "dst.bin"

    _clone_or_copy(src, dst)

    assert dst.read_bytes() == payload


def test_clone_or_copy_falls_back_when_cloning_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cloning is filesystem-dependent (APFS/btrfs/XFS reflink). Where it is
    unavailable the copy must still happen rather than raise."""
    import recall.services.compaction as compaction_module

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("clonefile unsupported on this filesystem")

    monkeypatch.setattr(compaction_module, "_clone_file", refuse)

    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dst = tmp_path / "dst.bin"

    compaction_module._clone_or_copy(src, dst)

    assert dst.read_bytes() == b"payload"


def test_compact_backup_uses_clone_not_byte_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`shutil.copy2` of the pre-compact backup is what pushed peak demand to
    ~3x the database size and produced ENOSPC on a full disk. The backup must
    go through the clone-aware helper."""
    import recall.services.compaction as compaction_module

    source = inspect.getsource(compaction_module._compact_locked)
    assert "shutil.copy2(db_path, backup_path)" not in source, (
        "pre-compact backup still uses a full byte copy"
    )
    assert "_clone_or_copy" in source


# --- a failed write must not destroy the original exception ----------


def test_write_session_logs_the_original_exception_before_rollback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The abort happened inside DuckDB's own commit-time revert, which
    destroyed the exception that caused it. Seven hypotheses were investigated
    before the cause was found, purely because nothing logged it. Whatever the
    rollback then does, the cause must already be in the log."""
    from recall.services.indexer import _log_failed_transaction

    boom = ValueError("constraint violation on message_state")
    with caplog.at_level(logging.ERROR, logger="recall.services.indexer"):
        _log_failed_transaction(boom, session_id="6cd08165", site="_sync_existing_session")

    text = caplog.text
    assert "6cd08165" in text
    assert "ValueError" in text
    assert "constraint violation on message_state" in text
    assert "_sync_existing_session" in text


def test_small_databases_are_not_auto_compacted() -> None:
    """A few-MB database reads ~1.65 with nothing dead, purely because
    partially-filled 256 KiB blocks round up. Without a size floor the lowered
    threshold would compact healthy small databases forever, reclaiming
    nothing."""
    from recall.core.config import (
        DEFAULT_COMPACTION_BLOAT_THRESHOLD,
        DEFAULT_COMPACTION_MIN_BYTES,
    )

    observed_fresh_small_ratio = 1.6462053571428572
    assert observed_fresh_small_ratio > DEFAULT_COMPACTION_BLOAT_THRESHOLD
    # ... so the floor, not the ratio, is what protects it.
    assert DEFAULT_COMPACTION_MIN_BYTES >= 64 * 1024 * 1024


def test_auto_compact_skips_a_file_below_min_bytes(monkeypatch) -> None:
    """The floor must be enforced in the daemon, not merely configured."""
    import inspect

    import recall.services.daemon as daemon_module

    source = inspect.getsource(daemon_module._maybe_run_auto_compact)
    assert "min_bytes" in source
    assert source.index("min_bytes") < source.index("bloat_ratio_threshold"), (
        "size floor must be checked before the ratio comparison"
    )
