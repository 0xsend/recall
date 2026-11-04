"""Post-cleanup service-reach and owned-group acceptance shared by bench jobs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .evidence import JobEnded, JobFailed, Result
from .fs import Cancelled, DeadlineExpired
from .isolation import RunDir
from .owned_process_group import StopError
from .supervisor import Stopped


def describe_error(error: BaseException) -> dict[str, object]:
    if isinstance(error, JobEnded):
        return error.describe()
    if isinstance(error, Cancelled):
        return {"kind": "cancelled", "signal": error.signum}
    if isinstance(error, DeadlineExpired):
        return {"kind": "deadline", "message": str(error)}
    return {"kind": type(error).__name__, "message": str(error)}


def safety(
    result: Result,
    run_dir: RunDir,
    stopped: list[Stopped],
    cleanup_errors: list[StopError] | None = None,
) -> JobFailed | None:
    """Record what reached for a service manager, and fail on anything that got through."""
    failures: list[dict[str, object]] = []
    try:
        attempts = _attempts(run_dir.attempts_path)
        canary = (
            run_dir.canary_log_path.read_text().splitlines()
            if run_dir.canary_log_path.exists()
            else []
        )
    except (OSError, ValueError) as error:
        attempts, canary = [], []
        failures.append({"kind": "safety_read", "message": str(error)})
    # An exec the audit hook stopped; a refused CLI call is the allow-list working.
    refused = [item for item in attempts if item.get("kind") == "refused"]
    refused_calls = [item for item in attempts if item.get("kind") == "refused_call"]
    refused_signals = [item for item in attempts if item.get("kind") == "refused_signal"]
    suppressed: dict[str, int] = {}
    for item in attempts:
        if item.get("kind") == "suppressed":
            seam = str(item.get("seam"))
            suppressed[seam] = suppressed.get(seam, 0) + 1
    survivors = [item.name for item in stopped if not item.group_gone]
    result.scope["service_reach"] = {
        "canary_invocations": canary,
        "refused": refused,
        "refused_calls": [item.get("detail") for item in refused_calls],
        "refused_signals": [item.get("detail") for item in refused_signals],
        "suppressed": suppressed,
    }
    if canary:
        failures.append({"kind": "service_manager_invoked", "lines": canary})
    if refused:
        failures.append({"kind": "service_reach_refused", "attempts": refused})
    if refused_signals:
        failures.append({"kind": "signal_refused", "attempts": refused_signals})
    nonclean = [asdict(item) for item in stopped if not item.clean]
    if nonclean or cleanup_errors:
        failures.append(
            {
                "kind": "stop_failed",
                "processes": nonclean,
                "errors": [asdict(error) for error in cleanup_errors or []],
            }
        )
    if survivors:
        return JobFailed(
            "survivors",
            "an owned process group outlived cleanup",
            names=survivors,
            failures=failures,
        )
    if nonclean or cleanup_errors:
        return JobFailed("stop_failed", "owned process cleanup failed", failures=failures)
    if failures:
        kind = str(failures[0]["kind"])
        return JobFailed(kind, "supervision safety checks failed", failures=failures)
    return None


def _attempts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
