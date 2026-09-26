"""The daemon process owns its logging; no client request may change it.

Nothing configured logging at daemon startup, so the
process ran on Python's lastResort handler — bare `WARNING`+ text, no timestamp
— until whichever index request first reached `logging.basicConfig` inside the
daemon fixed the level for the rest of the process lifetime, and a request
carrying `verbose` re-levelled the `recall` logger for every other session too.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from lane_harness import Lane, build_lane
from recall.cli.daemon import _DaemonLogHandler, _start_foreground_server
from recall.services.indexer import _bootstrap_logging_for_bare_caller


@pytest.fixture(autouse=True)
def _restore_process_logging() -> Iterator[None]:
    """Hand the process's logging back exactly as it was found.

    These tests configure a real daemon's root logger, which is shared with the
    rest of the suite.
    """
    root = logging.getLogger()
    recall_logger = logging.getLogger("recall")
    handlers = root.handlers[:]
    root_level = root.level
    recall_level = recall_logger.level
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(root_level)
        recall_logger.setLevel(recall_level)


@pytest.fixture(autouse=True)
def _scratch_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in tuple(os.environ):
        if key.startswith("RECALL_"):
            monkeypatch.delenv(key)


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Lane]:
    built = build_lane(tmp_path, monkeypatch)
    yield built
    asyncio.run(built.server.stop())


def _daemon_handler() -> logging.Handler:
    installed = [h for h in logging.getLogger().handlers if isinstance(h, _DaemonLogHandler)]
    assert len(installed) == 1, f"expected exactly one daemon handler, got {installed}"
    return installed[0]


def test_startup_renders_records_with_an_iso_timestamp_and_logger_name() -> None:
    _start_foreground_server(_exit_before_bind=True)

    formatter = _daemon_handler().formatter
    assert formatter is not None
    line = formatter.format(
        logging.LogRecord(
            name="recall.embed_phase",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="embed check: 3 pending across 2 sessions (1 idle)",
            args=(),
            exc_info=None,
        )
    )

    stamp, rendered = line.split(" ", 1)
    assert rendered == "INFO recall.embed_phase: embed check: 3 pending across 2 sessions (1 idle)"
    assert datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z").tzinfo is not None


def test_startup_defaults_to_emitting_recall_info_records() -> None:
    _start_foreground_server(_exit_before_bind=True)

    assert logging.getLogger("recall.embed_phase").isEnabledFor(logging.INFO)


def test_startup_leaves_third_party_loggers_at_their_own_level() -> None:
    _start_foreground_server(_exit_before_bind=True)

    assert not logging.getLogger("some_vendor_library").isEnabledFor(logging.INFO)


def test_verbose_startup_emits_recall_debug_records() -> None:
    _start_foreground_server(verbose=True, _exit_before_bind=True)

    assert logging.getLogger("recall.embed_phase").isEnabledFor(logging.DEBUG)


def test_configured_log_level_silences_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECALL_DAEMON_LOG_LEVEL", "warning")

    _start_foreground_server(_exit_before_bind=True)

    embed_logger = logging.getLogger("recall.embed_phase")
    assert not embed_logger.isEnabledFor(logging.INFO)
    assert embed_logger.isEnabledFor(logging.WARNING)


def test_repeated_startup_configuration_keeps_one_handler() -> None:
    _start_foreground_server(_exit_before_bind=True)
    _start_foreground_server(_exit_before_bind=True)

    _daemon_handler()


def test_unknown_log_level_is_rejected_at_config_load(monkeypatch: pytest.MonkeyPatch) -> None:
    from recall.core.config import AppConfig

    monkeypatch.setenv("RECALL_DAEMON_LOG_LEVEL", "chatty")

    with pytest.raises(ValueError, match="log_level"):
        AppConfig.load()


def test_verbose_daemon_run_leaves_the_daemon_at_its_own_level(lane: Lane) -> None:
    """A client's `-v` is its own; the daemon's log level is the daemon's."""
    root = logging.getLogger()
    root.handlers[:] = [logging.StreamHandler()]
    logging.getLogger("recall").setLevel(logging.WARNING)
    handlers_at_startup = root.handlers[:]

    asyncio.run(
        lane.server._handle_daemon_run({"once": True, "embed": False, "verbose": True}, None)
    )

    assert logging.getLogger("recall").level == logging.WARNING
    assert root.handlers == handlers_at_startup


def test_library_index_call_leaves_a_configured_process_alone() -> None:
    root = logging.getLogger()
    root.handlers[:] = [logging.StreamHandler()]
    root.setLevel(logging.ERROR)

    _bootstrap_logging_for_bare_caller(verbose=True)

    assert root.level == logging.ERROR


def test_library_index_call_bootstraps_a_bare_process() -> None:
    root = logging.getLogger()
    root.handlers[:] = []

    _bootstrap_logging_for_bare_caller(verbose=False)

    assert root.handlers != []
    assert root.level == logging.WARNING
