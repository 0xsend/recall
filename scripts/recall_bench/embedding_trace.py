"""Observer-only instrumentation loaded by the selected build's guarded launcher.

Calls retain their real inputs, outputs and exceptions. No inference, scheduling
or unload policy is replaced. Whole-process observations include query models.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import os
import resource
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field

TRACE_NAME = "embedding-trace.jsonl"
TRACE_BYTES_MAX = 128 * 1024**2
VECTORS_MAX = 8192


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal[
        "sample",
        "load-start",
        "load-end",
        "inference-start",
        "inference-end",
        "vectors",
        "unload-start",
        "unload-end",
        "error",
        "stop",
    ]
    at: float
    cpu_seconds: float = Field(ge=0)
    rss_bytes: int = Field(ge=0)
    rss_hwm_bytes: int = Field(ge=0)
    footprint_bytes: int | None = Field(default=None, ge=0)
    backend: str | None = None
    instance: int | None = None
    thread: int | None = None
    hashes: list[str] | None = None
    vectors: list[list[float]] | None = None
    token_lengths: list[int] | None = None
    error: str | None = None
    width: int | None = None
    rows: int | None = None
    device: str | None = None
    runtime: str | None = None
    model: str | None = None
    role: Literal["enrichment", "query"] | None = None


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Observer:
    def __init__(self, root: Path, backend: str) -> None:
        self.backend = backend
        self.process = psutil.Process()
        self.descriptor = os.open(root / TRACE_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.lock = threading.Lock()
        self.bytes = 0
        self.vectors = 0
        self.enrichment_instances: set[int] = set()
        self.error: Exception | None = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.sample, name="bench-model-observer", daemon=True)

    def record(self, kind: str, **fields: Any) -> None:
        with self.lock:
            times = self.process.cpu_times()
            footprint = None
            if sys.platform == "darwin":
                # Reuse the build's SDK layout; physical footprint is not RSS or USS.
                scheduling = importlib.import_module("recall.services.scheduling")
                usage = scheduling._RusageInfoV6()
                libproc, _ = scheduling._darwin_libraries()
                if libproc.proc_pid_rusage(os.getpid(), 6, ctypes.byref(usage)) != 0:
                    number = ctypes.get_errno()
                    raise OSError(number, os.strerror(number))
                footprint = usage.phys_footprint
            hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            event = Event.model_validate(
                dict(
                    kind=kind,
                    at=time.monotonic(),
                    cpu_seconds=times.user + times.system,
                    rss_bytes=self.process.memory_info().rss,
                    rss_hwm_bytes=int(hwm * (1 if sys.platform == "darwin" else 1024)),
                    footprint_bytes=footprint,
                    backend=self.backend,
                    **fields,
                )
            )
            data = (event.model_dump_json() + "\n").encode()
            self.bytes += len(data)
            self.vectors += len(event.vectors or [])
            if self.bytes > TRACE_BYTES_MAX or self.vectors > VECTORS_MAX:
                raise ValueError("embedding observation exceeded its byte/vector bound")
            if os.write(self.descriptor, data) != len(data):
                raise OSError("short embedding observation write")

    def sample(self) -> None:
        try:
            while not self.stopped.wait(0.1):
                self.record("sample")
        except Exception as error:
            self.error = error

    def wrap(self, method: Callable, kind: str) -> Callable:
        def observed(instance: Any, *args: Any, **kwargs: Any) -> Any:
            fields = {"instance": id(instance), "thread": threading.get_ident()}
            fields["role"] = "enrichment" if id(instance) in self.enrichment_instances else "query"
            if kind == "inference":
                fields["width"] = max(len(item.ids) for item in args[0])
                fields["rows"] = len(args[0])
            self.record(f"{kind}-start", **fields)
            try:
                result = method(instance, *args, **kwargs)
            except Exception as error:
                self.record("error", error=f"{type(error).__name__}: {error}", **fields)
                raise
            detail = {}
            if kind == "load":
                from importlib.metadata import version

                detail = {
                    "model": instance.model_id,
                    "runtime": version("onnxruntime" if self.backend == "onnx" else "mlx"),
                }
                if self.backend == "onnx":
                    detail["device"] = ",".join(instance._session.get_providers())
                else:
                    detail["device"] = str(importlib.import_module("mlx.core").default_device())
            self.record(f"{kind}-end", **fields, **detail)
            return result

        return observed


@contextmanager
def observe(root: Path, backend: str):
    """Observe the real backend and natural enrichment-model unload in this child."""
    observer = Observer(root, backend)
    module = importlib.import_module(f"recall.services.{backend}_embeddings")
    cls = getattr(module, "ONNXBackend" if backend == "onnx" else "MLXBackend")
    phase = importlib.import_module("recall.services.embed_phase").EmbedPhaseState
    leaf = "_embed_run" if backend == "onnx" else "_embed_batch"
    originals = {name: getattr(cls, name) for name in ("embed", "_load_model", leaf)}
    unload = phase.maybe_unload
    load = phase.load_backend

    def load_backend(self):
        backend = load(self)
        observer.enrichment_instances.add(id(backend))
        return backend

    def embed(instance, texts):
        result = originals["embed"](instance, texts)
        captured = [
            (text, vector)
            for text, vector in zip(texts, result, strict=True)
            if text.startswith("recallbench ")
        ]
        if not captured:
            return result
        lengths = [
            len(item.ids)
            for item in instance._tokenizer.encode_batch([text for text, _ in captured])
        ]
        observer.record(
            "vectors",
            instance=id(instance),
            hashes=[text_hash(text) for text, _ in captured],
            vectors=[vector for _, vector in captured],
            token_lengths=lengths,
        )
        return result

    def maybe_unload(self):
        loaded = (
            self.backend is not None
            and time.monotonic() - self.last_used_at >= self._config.daemon.embed_model_timeout
        )
        identity = id(self.backend) if loaded else None
        # Do not retain a reference to the backend being released.
        if loaded:
            observer.record("unload-start", instance=identity, role="enrichment")
        unload(self)
        if loaded and self.backend is None:
            observer.record("unload-end", instance=identity, role="enrichment")
            observer.enrichment_instances.discard(identity)

    cls.embed = embed
    cls._load_model = observer.wrap(originals["_load_model"], "load")
    setattr(cls, leaf, observer.wrap(originals[leaf], "inference"))
    phase.maybe_unload = maybe_unload
    phase.load_backend = load_backend
    observer.record("sample")
    observer.thread.start()
    try:
        yield
    finally:
        observer.stopped.set()
        observer.thread.join(timeout=2)
        try:
            if observer.thread.is_alive():
                raise RuntimeError("embedding observer did not stop")
            if observer.error is not None:
                observer.record("error", error=str(observer.error))
                raise observer.error
            observer.record("stop")
        finally:
            for name, method in originals.items():
                setattr(cls, name, method)
            phase.maybe_unload = unload
            phase.load_backend = load
            os.close(observer.descriptor)
