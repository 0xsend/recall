"""Parse bench-runner events with complete subprocess diagnostics on failure."""

from __future__ import annotations

import json
import signal
import subprocess
from collections.abc import Callable


def failure_details(process: subprocess.Popen[bytes], consumed: str) -> tuple[int | None, str]:
    # Streaming readers use unbuffered binary pipes: communicate() bypasses
    # read-ahead buffers and would otherwise lose their unread traceback text.
    try:
        _, stderr = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGCONT)
        process.terminate()
        _, stderr = process.communicate(timeout=15)
    return process.returncode, consumed + stderr.decode()


def parse_event(line: str, diagnostics: Callable[[], tuple[int | None, str]]) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError as error:
        returncode, stderr = diagnostics()
        raise AssertionError(
            f"non-JSON runner event (exit {returncode}); complete stderr:\n{stderr}"
        ) from error


def parse_events(result: subprocess.CompletedProcess[str]) -> list[dict]:
    return [
        parse_event(line, lambda: (result.returncode, result.stderr))
        for line in result.stderr.splitlines()
    ]
