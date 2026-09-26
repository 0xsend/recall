"""Per-command CTA (call-to-action) generators.

Each generator inspects actual result data to produce contextual suggestions
with real IDs and values pre-filled (REQ-CLI-016).
"""

from __future__ import annotations

import shlex
from typing import Any

from recall.cli.contract import Cta


def cta_for_list(sessions: list[dict[str, Any]]) -> list[Cta]:
    ctas: list[Cta] = []
    if sessions:
        first_id = sessions[0].get("id", "")
        if first_id:
            ctas.append(Cta(f"recall show {first_id}", "View most recent session"))
    ctas.append(Cta("recall search '<query>'", "Search across sessions"))
    return ctas


def cta_for_search(results: list[dict[str, Any]]) -> list[Cta]:
    ctas: list[Cta] = []
    if results:
        sid = results[0].get("session_id", "")
        if sid:
            ctas.append(Cta(f"recall show {sid} --tools", "View top result session"))
    return ctas


def cta_for_show(session: dict[str, Any]) -> list[Cta]:
    ctas: list[Cta] = []
    cwd = session.get("cwd")
    if cwd:
        quoted = shlex.quote(cwd)
        ctas.append(Cta(f"recall list --project {quoted}", "List sessions in same project"))
    ctas.append(Cta("recall search '<query>'", "Search across all sessions"))
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
    ctas = [
        Cta("recall daemon --once", "Run a single daemon cycle"),
        Cta("recall list --since 1h", "View recent sessions"),
    ]
    if status is not None and status.get("version_drift"):
        ctas.append(Cta("recall daemon restart", "Restart daemon to load upgraded binary"))
    return ctas


def cta_for_dry_run(command: str) -> list[Cta]:
    """CTAs for dry-run output: suggest re-running without --dry-run."""
    return [Cta(f"recall {command}", f"Execute {command} (remove --dry-run)")]


def cta_for_daemon_install() -> list[Cta]:
    return [
        Cta("recall daemon status", "Check installation status"),
        Cta("recall daemon --once", "Run a single daemon cycle"),
    ]


def cta_for_daemon_start() -> list[Cta]:
    return [Cta("recall daemon status", "Check daemon status")]


def cta_for_daemon_stop() -> list[Cta]:
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
