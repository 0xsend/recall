from __future__ import annotations

import logging
from pathlib import Path

import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.types import DaemonMode
from recall.services.daemon import daemon_status, fts_rebuild_backoff_status
from recall.services.watcher import FtsRebuildDebouncer


def _app_config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(mode=DaemonMode.POLL),
        cli=CliConfig(),
    )


class _FakeObserver:
    def schedule(self, handler: object, path: str, recursive: bool = False) -> object:
        _ = (handler, path, recursive)
        return object()

    def unschedule(self, watch: object) -> None:
        _ = watch
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def join(self, timeout: float) -> None:
        _ = timeout
        return None


def test_mark_oom_uses_exponential_backoff_capped_at_one_hour() -> None:
    debouncer = FtsRebuildDebouncer(fts_debounce=10.0)

    backoffs = [
        debouncer.mark_oom("oom", now_mono=float(at), now_wall=1_700_000_000.0 + at)
        for at in range(8)
    ]

    assert backoffs == [60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3600.0, 3600.0]
    assert debouncer.oom_count == 8


@pytest.mark.parametrize(("now", "ready"), [(79.999, False), (80.0, True)])
def test_ready_waits_out_the_oom_backoff_window(now: float, ready: bool) -> None:
    debouncer = FtsRebuildDebouncer(fts_debounce=10.0)
    debouncer.mark_dirty(now=0.0)
    debouncer.mark_oom("oom", now_mono=20.0, now_wall=1_700_000_000.0)

    assert debouncer.ready(now=now) is ready


def test_mark_rebuilt_resets_active_backoff_but_preserves_last_oom_status() -> None:
    debouncer = FtsRebuildDebouncer(fts_debounce=10.0)
    debouncer.mark_dirty(now=0.0)
    debouncer.mark_oom("FTS rebuild exhausted DuckDB memory_limit", now_mono=20.0, now_wall=1000.0)

    debouncer.mark_rebuilt(now=90.0)

    assert debouncer.oom_count == 0
    assert debouncer.next_retry_at_mono is None
    assert debouncer.last_oom_at_wall == 1000.0
    assert debouncer.last_oom_reason == "FTS rebuild exhausted DuckDB memory_limit"


def test_ready_does_not_log_while_backoff_window_is_active(
    caplog: pytest.LogCaptureFixture,
) -> None:
    debouncer = FtsRebuildDebouncer(fts_debounce=10.0)
    debouncer.mark_dirty(now=0.0)
    debouncer.mark_oom("oom", now_mono=20.0, now_wall=1_700_000_000.0)

    caplog.set_level(logging.WARNING, logger="recall.watcher")
    caplog.clear()

    assert debouncer.ready(now=30.0) is False
    assert debouncer.ready(now=40.0) is False
    assert debouncer.ready(now=50.0) is False
    assert caplog.records == []


def test_daemon_status_defaults_fts_backoff_fields_to_null_and_zero(tmp_path: Path) -> None:
    status = daemon_status(config=_app_config(tmp_path))

    assert status.last_fts_rebuild_failure_at is None
    assert status.last_fts_rebuild_failure_reason is None
    assert status.fts_rebuild_consecutive_failures == 0
    assert status.fts_rebuild_next_retry_at is None


def test_fts_rebuild_backoff_status_reports_last_failure_and_active_retry() -> None:
    debouncer = FtsRebuildDebouncer(fts_debounce=10.0)
    debouncer.mark_dirty(now=0.0)
    debouncer.mark_oom("oom", now_mono=20.0, now_wall=1_700_000_000.0)

    status = fts_rebuild_backoff_status(debouncer, now_mono=30.0)

    assert status["last_fts_rebuild_failure_at"] is not None
    assert status["last_fts_rebuild_failure_reason"] == "oom"
    assert status["fts_rebuild_consecutive_failures"] == 1
    assert status["fts_rebuild_next_retry_at"] is not None


def test_fts_rebuild_backoff_status_hides_expired_retry() -> None:
    debouncer = FtsRebuildDebouncer(fts_debounce=10.0)
    debouncer.mark_dirty(now=0.0)
    debouncer.mark_oom("oom", now_mono=20.0, now_wall=1_700_000_000.0)

    status = fts_rebuild_backoff_status(debouncer, now_mono=80.0)

    assert status["fts_rebuild_consecutive_failures"] == 1
    assert status["fts_rebuild_next_retry_at"] is None
