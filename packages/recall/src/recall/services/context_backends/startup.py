"""Bounded, cancellable startup prerequisites, without retaining optional models."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from dataclasses import asdict

from recall.core.config import ContextConfig
from recall.services.context_backends import (
    LLM_CONTEXT_MODES,
    ContextBackendUnavailableError,
    ensure_context_backend,
)
from recall.services.model_environment import prepare_model_environment

STARTUP_PROBE_SECONDS = 5.0
PROBE_DIAGNOSTIC_BYTES = 4096
_PROBE_COMPLETE = b"recall prerequisites complete\n"


class ContextPrerequisiteTimeoutError(ContextBackendUnavailableError):
    """The probe did not establish whether the backend is available."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.reason = TimeoutError(
            f"startup prerequisite check timed out after {STARTUP_PROBE_SECONDS:g}s"
        )
        RuntimeError.__init__(
            self,
            f"context mode {mode!r}: {self.reason}; backend availability is unconfirmed. "
            "This deadline does not prove a missing dependency or broken weights. "
            "Check host load and dependency import diagnostics, then explicitly start "
            "the daemon again when the host can complete the check — or set "
            "[embedding.context].mode to 'template' or 'off' to run without an LLM.",
        )


async def _read_tail(stream: asyncio.StreamReader, capacity: int) -> bytes:
    tail = b""
    while chunk := await stream.read(PROBE_DIAGNOSTIC_BYTES):
        tail = (tail + chunk)[-capacity:]
    return tail


async def _probe(process: asyncio.subprocess.Process, payload: bytes) -> tuple[bytes, bytes]:
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    async with asyncio.TaskGroup() as tasks:
        proof = tasks.create_task(_read_tail(process.stdout, len(_PROBE_COMPLETE)))
        diagnostic = tasks.create_task(_read_tail(process.stderr, PROBE_DIAGNOSTIC_BYTES))
        try:
            process.stdin.write(payload)
            await process.stdin.drain()
            process.stdin.close()
            await process.wait()
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
    return proof.result(), diagnostic.result()


def _exit_detail(returncode: int) -> str:
    detail = f"startup prerequisite probe exited with status {returncode}"
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        return f"{detail} ({name})"
    return f"{detail} without completing prerequisite checks"


async def require_context_prerequisites(config: ContextConfig) -> None:
    """Prove local support before database access; kill and reap stalled imports.

    Optional imports and client constructors execute in a disposable process so
    neither native initialization nor a dependency's shutdown can hold the owner.
    Configuration travels over stdin, never argv (it may contain a credential).
    MLX weights are proved only by selected generation in the owning process.
    """
    if config.mode not in LLM_CONTEXT_MODES:
        return
    process: asyncio.subprocess.Process | None = None
    try:
        async with asyncio.timeout(STARTUP_PROBE_SECONDS):
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "recall.services.context_backends.startup",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            proof, diagnostic = await _probe(process, json.dumps(asdict(config)).encode())
            assert process.returncode is not None
            if process.returncode != 0 or proof != _PROBE_COMPLETE:
                raise ContextBackendUnavailableError(
                    config.mode,
                    RuntimeError(
                        diagnostic.decode(errors="replace").strip()
                        or _exit_detail(process.returncode)
                    ),
                )
    except TimeoutError as err:
        raise ContextPrerequisiteTimeoutError(config.mode) from err
    finally:
        if process is not None:
            if process.returncode is None:
                process.kill()
            await process.wait()


def _check(config: ContextConfig) -> None:
    if config.mode == "llm-local":
        # Importing mlx_lm also imports its native runtime, proving host support.
        # No load() call, cache warming, download or model-path inspection belongs
        # to startup: those are properties of the first selected generation.
        import mlx_lm

        if not callable(mlx_lm.load) or not callable(mlx_lm.generate):
            raise RuntimeError("mlx_lm does not provide load and generate; reinstall recall[mlx]")
    else:
        ensure_context_backend(config)


def main() -> int:
    prepare_model_environment()
    config = ContextConfig.from_values(json.load(sys.stdin))
    try:
        _check(config)
    except Exception as err:
        # Optional dependencies can raise arbitrary native/import/client errors.
        # This process is only a prerequisite boundary; every failure is refusal.
        # The owner supplies the actionable envelope once. Backends checked via
        # ensure_context_backend already wrapped their cause; send only that
        # cause across the process boundary, just as a direct import error does.
        reason = err.reason if isinstance(err, ContextBackendUnavailableError) else err
        print(
            str(reason) if reason is not None else "local prerequisite check failed",
            file=sys.stderr,
        )
        return 1
    sys.stdout.buffer.write(_PROBE_COMPLETE)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
