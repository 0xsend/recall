"""REQ-CTX-021: the MLX local backend loads each model once per process.

In watch mode `_prepare_context_run` builds a fresh `MlxLocalBackend` per indexed
session, and `is_available()` -> `_ensure_loaded()` -> `mlx_lm.load()` re-read the
weights and made a huggingface.co revision call on EVERY session. A process-level
cache keyed by model name lets every instance reuse the resident weights, so the
model loads once per process.

These tests inject a stub into `sys.modules["mlx_lm"]` and never import the real
package: its Apple-Silicon-only native lib raises ImportError on Linux/CI and can
hard-abort the interpreter (SIGABRT) on some hosts, neither of which a test should
depend on. `MlxLocalBackend` imports `mlx_lm` lazily inside `_ensure_loaded`, so
the stub is what it resolves — keeping the cache logic covered on every platform.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterator

import pytest
import recall.services.context_backends.mlx_local as mlx_local
from recall.core.config import ContextConfig

_LoadFn = Callable[[str], tuple[object, object]]


class _FakeMlxLm:
    """Stand-in for the `mlx_lm` module; only `.load` is exercised by the backend."""

    def __init__(self) -> None:
        self.load: _LoadFn = lambda _name: (object(), object())


@pytest.fixture
def fake_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeMlxLm]:
    """Seed sys.modules with a stub mlx_lm and isolate the process model cache."""
    stub = _FakeMlxLm()
    monkeypatch.setitem(sys.modules, "mlx_lm", stub)
    mlx_local._MODEL_CACHE.clear()
    yield stub
    mlx_local._MODEL_CACHE.clear()


def test_model_is_loaded_once_across_backend_instances(fake_mlx_lm: _FakeMlxLm) -> None:
    calls: list[str] = []
    sentinel_model, sentinel_tok = object(), object()

    def fake_load(name: str) -> tuple[object, object]:
        calls.append(name)
        return sentinel_model, sentinel_tok

    fake_mlx_lm.load = fake_load

    config = ContextConfig(mode="llm-local", model="org/model-A")
    backends = [mlx_local.MlxLocalBackend(config) for _ in range(3)]
    assert all(b.is_available() for b in backends)

    # Three separate instances, one underlying load.
    assert calls == ["org/model-A"]
    # Every instance shares the cached weights/tokenizer.
    assert all(b._model is sentinel_model for b in backends)
    assert all(b._tokenizer is sentinel_tok for b in backends)


def test_different_models_are_cached_separately(fake_mlx_lm: _FakeMlxLm) -> None:
    calls: list[str] = []

    def fake_load(name: str) -> tuple[object, object]:
        calls.append(name)
        return object(), object()

    fake_mlx_lm.load = fake_load

    mlx_local.MlxLocalBackend(ContextConfig(mode="llm-local", model="org/model-A")).is_available()
    mlx_local.MlxLocalBackend(ContextConfig(mode="llm-local", model="org/model-B")).is_available()
    # A repeat of model-A must hit the cache, not reload.
    mlx_local.MlxLocalBackend(ContextConfig(mode="llm-local", model="org/model-A")).is_available()

    assert calls == ["org/model-A", "org/model-B"]


def test_concurrent_loads_trigger_a_single_load(fake_mlx_lm: _FakeMlxLm) -> None:
    """Threads racing the first load of one model must trigger exactly one load.

    The first caller loads under the cache lock (the sleep keeps it loading while
    the others arrive); the rest block, then find the populated cache.
    """
    calls: list[str] = []

    def slow_load(name: str) -> tuple[object, object]:
        time.sleep(0.05)
        calls.append(name)
        return object(), object()

    fake_mlx_lm.load = slow_load
    config = ContextConfig(mode="llm-local", model="org/model-A")

    results: list[bool] = []

    def worker() -> None:
        results.append(mlx_local.MlxLocalBackend(config).is_available())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [True, True, True, True]
    assert calls == ["org/model-A"]
