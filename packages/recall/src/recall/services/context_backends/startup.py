"""Bounded, cancellable startup prerequisites, without retaining optional models."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict

from recall.core.config import ContextConfig
from recall.services.context_backends import (
    LLM_CONTEXT_MODES,
    ContextBackendUnavailableError,
    ensure_context_backend,
)

STARTUP_PROBE_SECONDS = 5.0


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
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, diagnostic = await process.communicate(json.dumps(asdict(config)).encode())
            if process.returncode:
                raise ContextBackendUnavailableError(
                    config.mode, RuntimeError(diagnostic.decode(errors="replace")[-4096:].strip())
                )
    except TimeoutError as err:
        raise ContextBackendUnavailableError(
            config.mode,
            RuntimeError(f"startup prerequisite check timed out after {STARTUP_PROBE_SECONDS:g}s"),
        ) from err
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
    config = ContextConfig.from_values(json.load(sys.stdin))
    try:
        _check(config)
    except Exception as err:
        # Optional dependencies can raise arbitrary native/import/client errors.
        # This process is only a prerequisite boundary; every failure is refusal.
        print(str(err), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
