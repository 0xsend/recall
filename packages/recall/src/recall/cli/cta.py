"""Per-command CTA (call-to-action) generators.

Each generator inspects actual result data to produce contextual suggestions
with real IDs and values pre-filled (REQ-CLI-016).
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from recall.cli.contract import Cta
from recall.core.host_text import HostValue
from recall.core.types import is_attributed_host

# Only plain ASCII host names are safe to embed unescaped as `--host` CLI args;
# unusual or control-bearing origins fall back to `--fleet` without `--host`.
_SAFE_CLI_HOST_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _is_safe_cli_value(value: object) -> bool:
    return isinstance(value, str) and bool(value) and _CONTROL_CHAR_RE.search(value) is None


def _fleet_flags(*, fleet: bool, host: object = None) -> str:
    if not fleet:
        return ""
    raw_host = host.raw() if isinstance(host, HostValue) else host
    if isinstance(raw_host, str):
        candidate = raw_host.strip()
        if is_attributed_host(candidate) and _SAFE_CLI_HOST_RE.fullmatch(candidate):
            return f" --fleet --host {candidate}"
    return " --fleet"


def cta_for_list(
    sessions: list[dict[str, Any]], *, fleet: bool = False, endpoint: object = None
) -> list[Cta]:
    ctas: list[Cta] = []
    if sessions:
        first_id = sessions[0].get("id", "")
        if _is_safe_cli_value(first_id):
            flags = _fleet_flags(fleet=fleet, host=endpoint)
            ctas.append(
                Cta(f"recall show {shlex.quote(str(first_id))}{flags}", "View most recent session")
            )
    search_cmd = "recall search '<query>' --fleet" if fleet else "recall search '<query>'"
    ctas.append(Cta(search_cmd, "Search across sessions"))
    return ctas


def cta_for_search(
    results: list[dict[str, Any]], *, fleet: bool = False, endpoint: object = None
) -> list[Cta]:
    ctas: list[Cta] = []
    if results:
        sid = results[0].get("session_id", "")
        if _is_safe_cli_value(sid):
            flags = _fleet_flags(fleet=fleet, host=endpoint)
            ctas.append(
                Cta(
                    f"recall show {shlex.quote(str(sid))} --tools{flags}", "View top result session"
                )
            )
    return ctas


def cta_for_show(
    session: dict[str, Any], *, fleet: bool = False, endpoint: object = None
) -> list[Cta]:
    ctas: list[Cta] = []
    cwd = session.get("cwd")
    if _is_safe_cli_value(cwd):
        quoted = shlex.quote(str(cwd))
        flags = _fleet_flags(fleet=fleet, host=endpoint)
        ctas.append(Cta(f"recall list --project {quoted}{flags}", "List sessions in same project"))
    search_cmd = "recall search '<query>' --fleet" if fleet else "recall search '<query>'"
    ctas.append(Cta(search_cmd, "Search across all sessions"))
    return ctas


def cta_for_index(summary: dict[str, Any]) -> list[Cta]:
    ctas = [Cta("recall list --since 1h", "View recently indexed sessions")]
    if summary.get("indexed", 0) > 0:
        ctas.append(Cta("recall search '<query>'", "Search indexed content"))
    return ctas


def cta_for_daemon(summary: dict[str, Any]) -> list[Cta]:
    ctas = [Cta("recall daemon status", "Check daemon status")]
    idx = summary.get("index_summary", {}) if isinstance(summary, dict) else {}
    if isinstance(idx, dict) and idx.get("indexed", 0) > 0:
        ctas.append(Cta("recall list --since 1h", "View recently indexed sessions"))
    return ctas


def cta_for_daemon_status(status: dict[str, Any] | None = None) -> list[Cta]:
    if status is not None and status.get("runtime_unavailable_reason") is not None:
        return []
    ctas = [
        Cta("recall daemon --once", "Run a single daemon cycle"),
        Cta("recall list --since 1h", "View recent sessions"),
    ]
    if status is not None and status.get("version_drift"):
        ctas.append(Cta("recall daemon restart", "Restart daemon to load upgraded binary"))
    return ctas


def cta_for_daemon_install() -> list[Cta]:
    return [
        Cta("recall daemon status", "Check installation status"),
        Cta("recall daemon --once", "Run a single daemon cycle"),
    ]


def cta_for_daemon_start() -> list[Cta]:
    return [Cta("recall daemon status", "Check daemon status")]


def cta_for_daemon_stop(*, service_runs_other_data_dir: bool = False) -> list[Cta]:
    if service_runs_other_data_dir:
        # The installed service is another data dir's; `daemon start` refuses here.
        return [Cta("recall daemon start --background", "Start this data dir's daemon")]
    return [Cta("recall daemon start", "Start the daemon")]


def cta_for_daemon_restart() -> list[Cta]:
    return [Cta("recall daemon status", "Check daemon status")]


def cta_for_daemon_uninstall() -> list[Cta]:
    return [
        Cta("recall daemon status", "Verify removal"),
    ]


def cta_for_stats(subcommand: str | None) -> list[Cta]:
    match subcommand:
        case None:
            return [
                Cta("recall stats tools", "View tool usage breakdown"),
                Cta("recall stats bash --suggest", "Get permission suggestions"),
            ]
        case "tools":
            return [Cta("recall stats bash --suggest", "Get permission suggestions")]
        case "bash":
            return [Cta("recall stats tokens", "View token usage by project")]
        case "tokens":
            return [Cta("recall list --since 7d", "View recent sessions")]
        case _:
            return []
