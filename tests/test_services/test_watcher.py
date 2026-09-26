from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from conftest import _can_acquire_duckdb_lock
from recall.core.config import (
    AppConfig,
    CliConfig,
    ContextConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.models import Message, ParseResult, Session
from recall.core.types import DaemonMode, Role, Source
from recall.services.context_backends import ContextResult
from recall.services.watcher import (
    DebouncedIndexQueue,
    FtsRebuildDebouncer,
    SessionFileHandler,
    index_single_session,
    resolve_daemon_mode,
)

_requires_duckdb_lock = pytest.mark.skipif(
    not _can_acquire_duckdb_lock(),
    reason="DuckDB lock acquisition denied in this environment",
)


def _copy_codex_fixture(tmp_path: Path, name: str = "rollout.jsonl") -> Path:
    target = tmp_path / ".codex" / "sessions" / "s1"
    target.mkdir(parents=True, exist_ok=True)
    fixture = (
        Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "session1" / "rollout.jsonl"
    )
    destination = target / name
    shutil.copy(fixture, destination)
    return destination


def _app_config(tmp_path: Path) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        # Watcher tests exercise scheduling and debounce behavior, not DuckDB FTS
        # installation. Keep FTS disabled here so background-thread daemon tests stay
        # deterministic across hosts.
        fts=FtsConfig(fields=()),
        embedding=EmbeddingConfig(),
        daemon=DaemonConfig(mode=DaemonMode.WATCH, debounce=1, fts_debounce=1),
        cli=CliConfig(),
    )


@dataclass(frozen=True)
class _FakeWatch:
    path: str


class _FakeObserver:
    # The daemon schedules from its own thread while tests read from theirs.
    def __init__(self) -> None:
        self.scheduled: dict[str, tuple[_FakeWatch, object]] = {}
        self._lock = threading.Lock()

    def schedule(self, handler: object, path: str, recursive: bool = False) -> _FakeWatch:
        watch = _FakeWatch(path=path)
        with self._lock:
            self.scheduled[path] = (watch, handler)
        return watch

    def unschedule(self, watch: object) -> None:
        if isinstance(watch, _FakeWatch):
            with self._lock:
                self.scheduled.pop(watch.path, None)

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def join(self, timeout: float) -> None:
        return None

    def _handlers_for(self, path: str) -> list[object]:
        with self._lock:
            return [
                handler
                for watch_path, (_, handler) in self.scheduled.items()
                if path == watch_path or path.startswith(f"{watch_path}{os.sep}")
            ]

    def emit_modified(self, path: str) -> None:
        event = SimpleNamespace(is_directory=False, src_path=path)
        for handler in self._handlers_for(path):
            cast(SessionFileHandler, handler).on_modified(event)

    def wait_for_subscription(self, path: str, timeout: float = 15.0) -> None:
        """Block until an emit on ``path`` would reach a handler.

        The daemon subscribes from its own thread, so emitting before that lands
        silently drops the event and the test observes pre-event state. A fixed
        startup sleep made this a race that only lost on slow CI runners.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._handlers_for(path):
                return
            time.sleep(0.01)
        raise AssertionError(f"no watch covering {path} was subscribed within {timeout}s")


class RecordingContextBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def is_available(self) -> bool:
        return True

    def generate_prefix(self, session: Session, message: Message) -> ContextResult:
        self.calls.append(
            (
                message.id,
                [message.content or message.thinking or "" for message in session.messages],
            )
        )
        return ContextResult(prefix=f"[ctx {message.id}] ", mode="llm-local")


# -- DebouncedIndexQueue tests --


class TestDebouncedIndexQueue:
    def test_not_ready_before_debounce(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        queue.mark("/a.jsonl", now=100.0)
        assert queue.ready(now=103.0) == []

    def test_ready_after_debounce(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        queue.mark("/a.jsonl", now=100.0)
        result = queue.ready(now=105.0)
        assert result == ["/a.jsonl"]

    def test_reset_on_re_event(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        queue.mark("/a.jsonl", now=100.0)
        queue.mark("/a.jsonl", now=103.0)
        assert queue.ready(now=105.0) == []
        assert queue.ready(now=108.0) == ["/a.jsonl"]

    def test_multiple_files(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        queue.mark("/a.jsonl", now=100.0)
        queue.mark("/b.jsonl", now=102.0)
        result = queue.ready(now=106.0)
        assert result == ["/a.jsonl"]
        result = queue.ready(now=108.0)
        assert result == ["/b.jsonl"]

    def test_empty_drain(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        assert queue.ready(now=100.0) == []

    def test_flush_all(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        queue.mark("/a.jsonl", now=100.0)
        queue.mark("/b.jsonl", now=101.0)
        result = queue.flush_all()
        assert sorted(result) == ["/a.jsonl", "/b.jsonl"]
        assert queue.pending_count() == 0

    def test_flush_takes_only_the_named_path_off_the_queue(self) -> None:
        """`index_session_now` must not leave the drain loop a duplicate (REQ-LIVE-011)."""
        queue = DebouncedIndexQueue(debounce=5.0)
        queue.mark("/a.jsonl", now=100.0)
        queue.mark("/b.jsonl", now=100.0)

        assert queue.flush("/a.jsonl") is True
        assert queue.pending_count() == 1
        assert queue.ready(now=106.0) == ["/b.jsonl"]

    def test_flush_of_a_path_that_was_never_queued_reports_nothing_pending(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)

        assert queue.flush("/a.jsonl") is False

    def test_pending_count(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        assert queue.pending_count() == 0
        queue.mark("/a.jsonl", now=100.0)
        assert queue.pending_count() == 1
        queue.mark("/b.jsonl", now=101.0)
        assert queue.pending_count() == 2


# -- FtsRebuildDebouncer tests --


class TestFtsRebuildDebouncer:
    def test_not_ready_when_clean(self) -> None:
        debouncer = FtsRebuildDebouncer(fts_debounce=10.0)
        assert debouncer.ready(now=100.0) is False

    def test_not_ready_before_debounce(self) -> None:
        debouncer = FtsRebuildDebouncer(fts_debounce=10.0)
        debouncer.mark_dirty(now=100.0)
        assert debouncer.ready(now=108.0) is False

    def test_ready_after_debounce(self) -> None:
        debouncer = FtsRebuildDebouncer(fts_debounce=10.0)
        debouncer.mark_dirty(now=100.0)
        assert debouncer.ready(now=110.0) is True

    def test_not_ready_after_rebuild(self) -> None:
        debouncer = FtsRebuildDebouncer(fts_debounce=10.0)
        debouncer.mark_dirty(now=100.0)
        debouncer.mark_rebuilt(now=111.0)
        assert debouncer.ready(now=112.0) is False

    def test_reset_on_dirty(self) -> None:
        debouncer = FtsRebuildDebouncer(fts_debounce=10.0)
        debouncer.mark_dirty(now=100.0)
        debouncer.mark_dirty(now=108.0)
        assert debouncer.ready(now=112.0) is False
        assert debouncer.ready(now=118.0) is True


# -- SessionFileHandler tests --


class TestSessionFileHandler:
    def _make_codex_handler(self, queue: DebouncedIndexQueue) -> SessionFileHandler:
        from unittest.mock import patch as mock_patch

        from recall.parsers.codex import CodexParser

        # Patch watch_roots so it returns the path even when the directory
        # doesn't exist (e.g. on CI runners) — `default_watch_roots` drops
        # roots that are absent.
        with mock_patch.object(
            CodexParser, "watch_roots", return_value=[Path.home() / ".codex" / "sessions"]
        ):
            return SessionFileHandler(queue, [CodexParser()])

    def test_accepts_jsonl(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        handler = self._make_codex_handler(queue)

        event = MagicMock(is_directory=False)
        event.src_path = str(Path.home() / ".codex/sessions/s1/rollout1.jsonl")
        handler.on_modified(event)
        assert queue.pending_count() == 1

    def test_ignores_non_jsonl(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        handler = self._make_codex_handler(queue)

        event = MagicMock(is_directory=False)
        event.src_path = str(Path.home() / ".codex/sessions/s1/data.json")
        handler.on_modified(event)
        assert queue.pending_count() == 0

    def test_ignores_directories(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        handler = self._make_codex_handler(queue)

        event = MagicMock(is_directory=True)
        event.src_path = str(Path.home() / ".codex/sessions/s1/")
        handler.on_modified(event)
        assert queue.pending_count() == 0

    def test_rejects_non_codex_jsonl_in_codex_root(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        handler = self._make_codex_handler(queue)

        event = MagicMock(is_directory=False)
        event.src_path = str(Path.home() / ".codex/sessions/s1/other.jsonl")
        handler.on_modified(event)
        assert queue.pending_count() == 0

    def test_on_created_also_triggers(self) -> None:
        queue = DebouncedIndexQueue(debounce=5.0)
        handler = self._make_codex_handler(queue)

        event = MagicMock(is_directory=False)
        event.src_path = str(Path.home() / ".codex/sessions/s1/rollout2.jsonl")
        handler.on_created(event)
        assert queue.pending_count() == 1


# -- inotify watch limit error handling --


class TestInotifyWatchLimitError:
    def test_enospc_raises_descriptive_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify ENOSPC from observer.start() produces a helpful inotify message."""
        import errno

        monkeypatch.setenv("HOME", str(tmp_path))
        config = _app_config(tmp_path)

        codex_dir = tmp_path / ".codex" / "sessions" / "s1"
        codex_dir.mkdir(parents=True, exist_ok=True)

        mock_observer = MagicMock()
        mock_observer.start.side_effect = OSError(errno.ENOSPC, "No space left on device")

        with (
            patch("watchdog.observers.Observer", return_value=mock_observer),
            pytest.raises(OSError, match="inotify watch limit exceeded"),
        ):
            from recall.services.watcher import build_live_watch_runtime, start_live_watch_runtime

            start_live_watch_runtime(build_live_watch_runtime(config=config))

    def test_non_enospc_oserror_propagates_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Other OSErrors from observer.start() propagate without modification."""
        import errno

        monkeypatch.setenv("HOME", str(tmp_path))
        config = _app_config(tmp_path)

        codex_dir = tmp_path / ".codex" / "sessions" / "s1"
        codex_dir.mkdir(parents=True, exist_ok=True)

        mock_observer = MagicMock()
        mock_observer.start.side_effect = OSError(errno.EACCES, "Permission denied")

        with (
            patch("watchdog.observers.Observer", return_value=mock_observer),
            pytest.raises(OSError, match="Permission denied"),
        ):
            from recall.services.watcher import build_live_watch_runtime, start_live_watch_runtime

            start_live_watch_runtime(build_live_watch_runtime(config=config))


# -- DaemonMode resolution tests --


class TestDaemonModeResolution:
    # REQ-DAEMON-040/041 made watchdog a required runtime dependency, so WATCH
    # mode no longer performs optional-dependency checks.
    @pytest.mark.parametrize(
        ("requested", "resolved"),
        [
            (DaemonMode.AUTO, DaemonMode.WATCH),
            (DaemonMode.WATCH, DaemonMode.WATCH),
            (DaemonMode.POLL, DaemonMode.POLL),
        ],
    )
    def test_resolves_requested_mode(self, requested: DaemonMode, resolved: DaemonMode) -> None:
        assert resolve_daemon_mode(requested) == resolved


# -- index_single_session tests --


class TestIndexSingleSession:
    @_requires_duckdb_lock
    def test_insert_new_session(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        fixture_path = _copy_codex_fixture(tmp_path)
        config = _app_config(tmp_path)

        from recall.parsers.codex import CodexParser

        parser = CodexParser()
        result = index_single_session(fixture_path, parser, config)
        assert result is True

        conn = duckdb.connect(str(config.db_path))
        try:
            count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            assert count == (1,)
        finally:
            conn.close()

    @_requires_duckdb_lock
    def test_update_unchanged_session_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        fixture_path = _copy_codex_fixture(tmp_path)
        config = _app_config(tmp_path)

        from recall.parsers.codex import CodexParser

        parser = CodexParser()
        index_single_session(fixture_path, parser, config)
        result = index_single_session(fixture_path, parser, config)
        assert result is False

    @_requires_duckdb_lock
    def test_index_single_session_uses_configured_template_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class TemplateParser:
            source = Source.CODEX

            @property
            def file_pattern(self) -> str:
                return "*.jsonl"

            roots: tuple[Path, ...] | None = None

            def default_roots(self) -> list[Path]:
                return []

            def watch_roots(self) -> list[Path]:
                return []

            def discover(self) -> list[Path]:
                return []

            def sidecar_paths(self, path: Path) -> list[Path]:
                _ = path
                return []

            def parse(
                self,
                path: Path,
                *,
                offset: int = 0,
                message_idx_base: int = 0,
                orphan_tool_call_idx_base: int = 0,
                resume_state: Mapping[str, Any] | None = None,
            ) -> ParseResult:
                del offset, message_idx_base, orphan_tool_call_idx_base, resume_state
                return ParseResult(
                    session=Session(
                        id="watch-template-session",
                        source=Source.CODEX,
                        source_path=str(path),
                        file_mtime=path.stat().st_mtime,
                        file_size=path.stat().st_size,
                        git_repo="acme/recall",
                        git_branch="main",
                        messages=[
                            Message(
                                id="watch-template-message",
                                session_id="watch-template-session",
                                idx=0,
                                role=Role.ASSISTANT,
                                content="hello",
                            )
                        ],
                        message_count=1,
                    ),
                    next_byte_offset=path.stat().st_size,
                    is_full_parse=True,
                )

            def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
                del now, idle_threshold
                return []

        monkeypatch.setenv("HOME", str(tmp_path))
        path = tmp_path / "template.jsonl"
        path.write_text("{}", encoding="utf-8")
        config = _app_config(tmp_path)
        config = replace(
            config,
            embedding=EmbeddingConfig(context=ContextConfig(mode="template")),
        )

        result = index_single_session(path, TemplateParser(), config)
        assert result is True

        conn = duckdb.connect(str(config.db_path))
        try:
            row = conn.execute(
                """
                SELECT context_text, context_mode, fts_content
                FROM message_state
                WHERE message_id = 'watch-template-message'
                """
            ).fetchone()
        finally:
            conn.close()

        assert row == (
            "[acme/recall main] ",
            "template",
            "[acme/recall main] hello",
        )

    @_requires_duckdb_lock
    def test_lightweight_context_downgrades_llm_to_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class TemplateParser:
            source = Source.CODEX

            @property
            def file_pattern(self) -> str:
                return "*.jsonl"

            roots: tuple[Path, ...] | None = None

            def default_roots(self) -> list[Path]:
                return []

            def watch_roots(self) -> list[Path]:
                return []

            def discover(self) -> list[Path]:
                return []

            def sidecar_paths(self, path: Path) -> list[Path]:
                _ = path
                return []

            def parse(
                self,
                path: Path,
                *,
                offset: int = 0,
                message_idx_base: int = 0,
                orphan_tool_call_idx_base: int = 0,
                resume_state: Mapping[str, Any] | None = None,
            ) -> ParseResult:
                del offset, message_idx_base, orphan_tool_call_idx_base, resume_state
                return ParseResult(
                    session=Session(
                        id="watch-light-context-session",
                        source=Source.CODEX,
                        source_path=str(path),
                        file_mtime=path.stat().st_mtime,
                        file_size=path.stat().st_size,
                        git_repo="acme/recall",
                        git_branch="main",
                        messages=[
                            Message(
                                id="watch-light-context-message",
                                session_id="watch-light-context-session",
                                idx=0,
                                role=Role.ASSISTANT,
                                content="hello",
                            )
                        ],
                        message_count=1,
                    ),
                    next_byte_offset=path.stat().st_size,
                    is_full_parse=True,
                )

            def live_candidates(self, *, now: datetime, idle_threshold: float) -> list[Path]:
                del now, idle_threshold
                return []

        monkeypatch.setenv("HOME", str(tmp_path))
        requested_modes: list[str] = []

        def get_context_backend(config: ContextConfig) -> None:
            requested_modes.append(config.mode)
            if config.mode.startswith("llm-"):
                pytest.fail("lightweight watch context loaded LLM backend")
            return None

        monkeypatch.setattr(
            "recall.services.context_backends.get_context_backend",
            get_context_backend,
        )
        path = tmp_path / "light-context.jsonl"
        path.write_text("{}", encoding="utf-8")
        config = replace(
            _app_config(tmp_path),
            embedding=EmbeddingConfig(context=ContextConfig(mode="llm-local", fallback="template")),
        )

        result = index_single_session(
            path,
            TemplateParser(),
            config,
            lightweight_context=True,
        )

        conn = duckdb.connect(str(config.db_path))
        try:
            row = conn.execute(
                """
                SELECT context_text, context_mode
                FROM message_state
                WHERE message_id = 'watch-light-context-message'
                """
            ).fetchone()
        finally:
            conn.close()

        assert result is True
        assert row == ("[acme/recall main] ", "template")
        assert requested_modes == ["template"]

    @_requires_duckdb_lock
    def test_stores_byte_offset_and_uses_incremental(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify index_single_session stores last_byte_offset and uses incremental parsing."""
        import json

        monkeypatch.setenv("HOME", str(tmp_path))
        fixture_path = _copy_codex_fixture(tmp_path)
        config = _app_config(tmp_path)

        from recall.parsers.codex import CodexParser

        parser = CodexParser()
        index_single_session(fixture_path, parser, config)

        # Check offset was stored
        conn = duckdb.connect(str(config.db_path))
        try:
            row = conn.execute(
                "SELECT last_byte_offset, message_count FROM session_state LIMIT 1"
            ).fetchone()
            assert row is not None
            stored_offset, stored_msgs = row
            assert stored_offset > 0
            assert stored_msgs > 0
        finally:
            conn.close()

        # Append a new message
        with fixture_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": "2026-12-31T23:59:59Z",
                        "payload": {"type": "user_message", "message": "watcher append"},
                    }
                )
                + "\n"
            )

        # Re-index — should use incremental path
        index_single_session(fixture_path, parser, config)

        conn = duckdb.connect(str(config.db_path))
        try:
            row = conn.execute(
                "SELECT last_byte_offset, message_count FROM session_state LIMIT 1"
            ).fetchone()
            assert row is not None
            new_offset, new_msgs = row
            assert new_offset > stored_offset
            assert new_msgs > stored_msgs
        finally:
            conn.close()

    @_requires_duckdb_lock
    def test_incremental_contextualizes_against_full_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from recall.parsers.codex import CodexParser

        monkeypatch.setenv("HOME", str(tmp_path))
        path = tmp_path / "watch-incremental.jsonl"
        records = [
            {"type": "session_meta", "payload": {"id": "watch-session"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "watch earlier context"}],
                },
            },
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        backend = RecordingContextBackend()
        config = replace(
            _app_config(tmp_path),
            embedding=EmbeddingConfig(context=ContextConfig(mode="llm-local")),
        )
        monkeypatch.setattr(
            "recall.services.context_backends.get_context_backend", lambda _config: backend
        )

        first = index_single_session(path, CodexParser(), config)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "watch appended answer"}],
                        },
                    }
                )
                + "\n"
            )
        second = index_single_session(path, CodexParser(), config)

        assert first is True
        assert second is True
        assert backend.calls[-1][1] == ["watch earlier context", "watch appended answer"]
        with duckdb.connect(str(config.db_path)) as conn:
            assert conn.execute(
                "SELECT content FROM message_state "
                "JOIN messages ON messages.id = message_id ORDER BY idx"
            ).fetchall() == [("watch earlier context",), ("watch appended answer",)]

    def test_no_embed_params_accepted(self) -> None:
        """REQ-ADAPT-001: index_single_session must not accept embed params."""
        import inspect

        sig = inspect.signature(index_single_session)
        param_names = set(sig.parameters.keys())
        assert "embed_backend" not in param_names
        assert "embedding_cache" not in param_names
        assert "batch_size" not in param_names
