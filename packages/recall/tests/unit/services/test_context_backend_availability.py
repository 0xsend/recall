"""REQ-CTX-020: a configured-but-unsupported LLM context backend is fatal.

When `[embedding.context].mode` is an `llm-*` mode but its backend cannot run on
this host (missing extra, missing credential, model load failure), recall must
hard-error with an actionable message instead of silently degrading to template.
This is decoupled from the `fallback` policy, which now governs only transient
per-message generation failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from recall.core.config import (
    AppConfig,
    CliConfig,
    ContextConfig,
    DaemonConfig,
    EmbeddingConfig,
    FtsConfig,
)
from recall.core.types import DaemonMode
from recall.services import context_backends as cb
from recall.services.context_backends import (
    ContextBackendUnavailableError,
    ensure_context_backend,
)


class _StubBackend:
    """Minimal ContextBackend stand-in mirroring the real backends' surface.

    The real backends record the initialization failure on `_load_failure` after
    `is_available()` returns False; ensure_context_backend surfaces that reason in
    the actionable error, so the stub carries it too.
    """

    def __init__(self, *, available: bool, load_failure: Exception | None = None) -> None:
        self._available = available
        self._load_failure = load_failure

    def is_available(self) -> bool:
        return self._available

    def generate_prefix(self, session: object, message: object) -> object:  # pragma: no cover
        raise AssertionError("generate_prefix must not be reached when unavailable")


class _FakeObserver:
    def schedule(self, handler: object, path: str, recursive: bool = False) -> object:
        _ = (handler, path, recursive)
        return object()

    def unschedule(self, watch: object) -> None:
        _ = watch

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def join(self, timeout: float) -> None:
        _ = timeout


def _app_config(tmp_path: Path, *, context: ContextConfig) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    return AppConfig(
        data_dir=data_dir,
        db_path=data_dir / "recall.duckdb",
        lock_path=data_dir / "recall.lock",
        config_path=tmp_path / ".config" / "recall" / "config.toml",
        fts=FtsConfig(),
        embedding=EmbeddingConfig(context=context),
        daemon=DaemonConfig(mode=DaemonMode.POLL),
        cli=CliConfig(),
    )


@pytest.mark.parametrize("fallback", ["template", "off", "error"])
def test_unavailable_llm_backend_is_fatal_regardless_of_fallback(
    monkeypatch: pytest.MonkeyPatch, fallback: str
) -> None:
    stub = _StubBackend(
        available=False,
        load_failure=RuntimeError("mlx_lm is not installed; install recall[mlx]"),
    )
    monkeypatch.setattr(cb, "get_context_backend", lambda _cfg: stub)

    config = ContextConfig(mode="llm-local", fallback=fallback)
    with pytest.raises(ContextBackendUnavailableError) as exc_info:
        ensure_context_backend(config)

    err = exc_info.value
    assert err.mode == "llm-local"
    message = str(err)
    # Names the mode, the missing dependency, and the escape hatch.
    assert "llm-local" in message
    assert "recall[mlx]" in message
    assert "template" in message and "off" in message


def test_available_llm_backend_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubBackend(available=True)
    monkeypatch.setattr(cb, "get_context_backend", lambda _cfg: stub)

    config = ContextConfig(mode="llm-remote")
    assert ensure_context_backend(config) is stub


@pytest.mark.parametrize("mode", ["off", "template"])
def test_non_llm_modes_never_require_a_backend(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    # get_context_backend returns None for off/template; ensure that stays non-fatal.
    monkeypatch.setattr(cb, "get_context_backend", lambda _cfg: None)
    config = ContextConfig(mode=mode)
    assert ensure_context_backend(config) is None


def test_error_message_names_the_remote_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubBackend(
        available=False,
        load_failure=RuntimeError("anthropic is not installed; install recall[anthropic]"),
    )
    monkeypatch.setattr(cb, "get_context_backend", lambda _cfg: stub)

    with pytest.raises(ContextBackendUnavailableError) as exc_info:
        ensure_context_backend(ContextConfig(mode="llm-remote"))
    assert "recall[anthropic]" in str(exc_info.value)


def test_error_message_names_the_codex_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubBackend(available=False, load_failure=RuntimeError("codex CLI not found on PATH"))
    monkeypatch.setattr(cb, "get_context_backend", lambda _cfg: stub)

    with pytest.raises(ContextBackendUnavailableError) as exc_info:
        ensure_context_backend(ContextConfig(mode="llm-codex"))
    assert "codex" in str(exc_info.value).lower()


def test_prepare_context_run_is_fatal_on_unavailable_with_template_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decoupling proof: fallback=template no longer masks a missing backend."""
    from recall.services import indexer

    stub = _StubBackend(
        available=False,
        load_failure=RuntimeError("mlx_lm is not installed; install recall[mlx]"),
    )
    monkeypatch.setattr(cb, "get_context_backend", lambda _cfg: stub)

    config = ContextConfig(mode="llm-local", fallback="template")
    with pytest.raises(ContextBackendUnavailableError):
        indexer._prepare_context_run(config)


def test_prepare_context_run_returns_backend_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recall.services import indexer

    stub = _StubBackend(available=True)
    monkeypatch.setattr(cb, "get_context_backend", lambda _cfg: stub)

    run = indexer._prepare_context_run(ContextConfig(mode="llm-local", fallback="template"))
    assert run.backend is stub
    assert run.mode == "llm-local"


def test_prepare_context_run_template_mode_needs_no_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recall.services import indexer

    monkeypatch.setattr(cb, "get_context_backend", lambda _cfg: None)

    run = indexer._prepare_context_run(ContextConfig(mode="template"))
    assert run.mode == "template"
    assert run.backend is None


def test_build_live_watch_runtime_is_independent_of_unavailable_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from recall.services import watcher

    stub = _StubBackend(
        available=False,
        load_failure=RuntimeError("mlx_lm is not installed; install recall[mlx]"),
    )
    monkeypatch.setattr(cb, "get_context_backend", lambda _cfg: stub)

    config = _app_config(tmp_path, context=ContextConfig(mode="llm-local", fallback="template"))
    runtime = watcher.build_live_watch_runtime(config=config, observer_factory=_FakeObserver)
    assert runtime.config == config
