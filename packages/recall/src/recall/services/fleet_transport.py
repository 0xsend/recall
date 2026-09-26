"""On-demand SSH transport for fleet fan-out (REQ-FLEET-SSH-*).

v1: one-shot ssh per host per invocation. No ControlMaster; RemoteCommand=none
and BatchMode=yes are mandatory (deploy-hosts class of SSH hijacks).
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from recall.core.fleet import FleetHost

logger = logging.getLogger("recall.fleet")

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_PER_HOST_TIMEOUT_SECONDS = 45
DEFAULT_CONCURRENCY = 6
MAX_REMOTE_ERROR_CHARS = 400


@dataclass(frozen=True)
class FleetRemoteResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    error: str | None = None


@dataclass(frozen=True)
class FleetHostStatus:
    name: str
    ssh: str
    ok: bool
    binary_version: str | None = None
    daemon_version: str | None = None
    version_drift: bool | None = None
    error: str | None = None


def build_ssh_argv(
    target: str,
    remote_argv: Sequence[str],
    *,
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
) -> list[str]:
    """Construct ssh argv for a one-shot remote command (REQ-FLEET-SSH-002).

    ssh has no remote argv: it joins its trailing arguments with spaces and hands
    the resulting single string to the remote login shell, which re-splits it.
    Each element is therefore shell-quoted so it survives that round trip as one
    argument — without this, any value containing whitespace or shell
    metacharacters (a multi-word search query, a project path with a space)
    arrives at the remote CLI split into several arguments.
    """
    if not target.strip():
        raise ValueError("ssh target must be non-empty")
    if not remote_argv:
        raise ValueError("remote_argv must be non-empty")
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "RemoteCommand=none",
        # Suppress the "Pseudo-terminal will not be allocated" notice that some
        # host configs (RequestTTY=yes) emit on stderr for every fan-out call.
        "-o",
        "RequestTTY=no",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        "ControlPath=none",
        target,
        "--",
        *(shlex.quote(arg) for arg in remote_argv),
    ]


# SSH client notices that are informational, never the failure itself. They land
# on stderr interleaved with the remote command's own stderr; left in, they
# shadow the real error in the single-line per-host skip message (REQ-FLEET-SSH-005).
_SSH_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^Pseudo-terminal will not be allocated because stdin is not a terminal\.$"),
    re.compile(r"^Warning: Permanently added .* to the list of known hosts\.$"),
)


def _signal_lines(stream: str) -> list[str]:
    """Non-blank lines of a stream with SSH transport notices removed."""
    lines: list[str] = []
    for raw in stream.splitlines():
        line = raw.strip()
        if not line or any(noise.match(line) for noise in _SSH_NOISE_PATTERNS):
            continue
        lines.append(line)
    return lines


def extract_remote_error(stdout: str, stderr: str, returncode: int) -> str:
    """Single-line, bounded error for a failed remote command (REQ-FLEET-SSH-005).

    stderr wins when it carries signal, but a stderr holding only SSH transport
    notices falls through to stdout rather than reporting the notice as the error.
    """
    for stream in (stderr, stdout):
        if lines := _signal_lines(stream):
            message = "; ".join(lines)
            if len(message) > MAX_REMOTE_ERROR_CHARS:
                return message[: MAX_REMOTE_ERROR_CHARS - 3] + "..."
            return message
    return f"exit {returncode}"


def run_remote(
    host: FleetHost,
    remote_argv: Sequence[str],
    *,
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    timeout: float = DEFAULT_PER_HOST_TIMEOUT_SECONDS,
) -> FleetRemoteResult:
    """Run remote_argv on host via one-shot SSH. Never raises for remote failure."""
    argv = build_ssh_argv(host.ssh, remote_argv, connect_timeout=connect_timeout)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        return FleetRemoteResult(
            ok=False,
            stdout=(err.stdout or "") if isinstance(err.stdout, str) else "",
            stderr=(err.stderr or "") if isinstance(err.stderr, str) else "",
            returncode=-1,
            error=f"timeout after {timeout}s",
        )
    except OSError as err:
        return FleetRemoteResult(
            ok=False,
            stdout="",
            stderr=str(err),
            returncode=-1,
            error=str(err),
        )

    stderr = completed.stderr or ""
    stdout = completed.stdout or ""
    if completed.returncode != 0:
        err_msg = extract_remote_error(stdout, stderr, completed.returncode)
        return FleetRemoteResult(
            ok=False,
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
            error=err_msg,
        )
    return FleetRemoteResult(
        ok=True,
        stdout=stdout,
        stderr=stderr,
        returncode=completed.returncode,
    )


_DAEMON_VERSION_RE = re.compile(r"^daemon_version:\s*(\S+)\s*$", re.MULTILINE)
_BINARY_VERSION_RE = re.compile(r"^binary_version:\s*(\S+)\s*$", re.MULTILINE)
_VERSION_DRIFT_RE = re.compile(r"^version_drift:\s*(true|false)\s*$", re.MULTILINE | re.IGNORECASE)


def parse_daemon_status_text(text: str) -> dict[str, Any]:
    """Extract version fields from `recall daemon status` text output."""
    out: dict[str, Any] = {}
    if m := _DAEMON_VERSION_RE.search(text):
        val = m.group(1)
        out["daemon_version"] = None if val == "null" else val
    if m := _BINARY_VERSION_RE.search(text):
        val = m.group(1)
        out["binary_version"] = None if val == "null" else val
    if m := _VERSION_DRIFT_RE.search(text):
        out["version_drift"] = m.group(1).lower() == "true"
    return out


def _probe_one(host: FleetHost) -> FleetHostStatus:
    version_result = run_remote(host, ["recall", "--version"])
    if not version_result.ok:
        return FleetHostStatus(
            name=host.name,
            ssh=host.ssh,
            ok=False,
            error=version_result.error or "recall --version failed",
        )
    binary_version = version_result.stdout.strip().splitlines()[0].strip() or None

    status_result = run_remote(host, ["recall", "daemon", "status"])
    daemon_version: str | None = None
    version_drift: bool | None = None
    if status_result.ok:
        parsed = parse_daemon_status_text(status_result.stdout)
        daemon_version = parsed.get("daemon_version")
        if isinstance(daemon_version, str) or daemon_version is None:
            pass
        else:
            daemon_version = str(daemon_version)
        drift = parsed.get("version_drift")
        version_drift = drift if isinstance(drift, bool) else None
        # Prefer binary_version from daemon status when present.
        status_binary = parsed.get("binary_version")
        if isinstance(status_binary, str) and status_binary:
            binary_version = status_binary
    # Version succeeded → host is reachable even if daemon status failed.
    error = None if status_result.ok else (status_result.error or "daemon status failed")
    return FleetHostStatus(
        name=host.name,
        ssh=host.ssh,
        ok=True,
        binary_version=binary_version,
        daemon_version=daemon_version,
        version_drift=version_drift,
        error=error,
    )


def fleet_status(
    hosts: Iterable[FleetHost],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[FleetHostStatus]:
    """Probe each host (REQ-FLEET-CMD-001). Concurrent, bounded."""
    host_list = list(hosts)
    if not host_list:
        return []
    workers = max(1, min(concurrency, len(host_list)))
    results: dict[str, FleetHostStatus] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_probe_one, host): host for host in host_list}
        for future in as_completed(futures):
            host = futures[future]
            try:
                results[host.name] = future.result()
            except Exception as err:
                logger.exception("fleet status probe crashed for %s", host.name)

                results[host.name] = FleetHostStatus(
                    name=host.name,
                    ssh=host.ssh,
                    ok=False,
                    error=str(err),
                )
    # Preserve inventory order.
    return [results[h.name] for h in host_list]
