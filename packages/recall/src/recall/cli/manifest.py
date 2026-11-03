from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any, TypedDict

from typer.core import TyperCommand, TyperGroup, TyperOption
from typer.main import get_group

from recall.core.daemon_status import DaemonSchedulerStatus, ReconciliationStatus
from recall.core.models import SessionSummary

# Preserve the published host position while deriving every field from the model.
_SESSION_SUMMARY_FIELDS = [field.name for field in fields(SessionSummary) if field.name != "host"]
_SESSION_SUMMARY_FIELDS.insert(_SESSION_SUMMARY_FIELDS.index("file_mtime"), "host")

# `daemon status` answers the model's fields plus the daemon's reconciliation
# block; both lists derive from those types, so a field the command emits
# cannot be missing from its projection allowlist.
_DAEMON_STATUS_FIELDS = [
    "reconciliation",
    *(field.name for field in fields(DaemonSchedulerStatus)),
]
_DAEMON_STATUS_PROJECTION_FIELDS = [
    *_DAEMON_STATUS_FIELDS,
    *(
        f"reconciliation.{key}"
        for key in sorted(
            ReconciliationStatus.__required_keys__ | ReconciliationStatus.__optional_keys__
        )
    ),
]

GLOBAL_ENV = [
    {"name": "RECALL_DATA_DIR", "type": "path"},
    {"name": "RECALL_DB_PATH", "type": "path"},
    {"name": "RECALL_LOCK_PATH", "type": "path"},
    {"name": "RECALL_CONFIG_PATH", "type": "path"},
]

_CTA_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "fields": ["command", "description"],
    "description": "Suggested next commands with pre-filled arguments",
}


def _with_cta_envelope(output: dict[str, Any]) -> dict[str, Any]:
    """Annotate an output schema with the CTA envelope variant.

    The envelope's "data" property inlines the original output schema so agents
    can discover the wrapped payload shape from the manifest (REQ-CLI-001).
    """
    cta_envelope: dict[str, Any] = {
        "type": "object",
        "fields": ["data", "cta"],
        "properties": {
            "data": {**output, "description": "The original command output"},
            "cta": _CTA_ITEMS_SCHEMA,
        },
    }
    return {**output, "cta_envelope": cta_envelope}


COMMANDS: dict[str, dict[str, Any]] = {
    "schema": {
        "command": "schema",
        "description": "Return the machine-readable schema for a specific command path.",
        "args": [{"name": "command_path", "type": "string[]", "required": False}],
        "options": [],
        "env": GLOBAL_ENV,
        "output": {
            "type": "object",
            "fields": [
                "command",
                "description",
                "args",
                "options",
                "env",
                "output",
                "safety",
                "examples",
            ],
        },
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": ["recall schema search", "recall schema daemon install"],
    },
    "index": {
        "command": "index",
        "description": "Index agent session files into the local DuckDB database.",
        "args": [],
        "options": [
            {"name": "full", "type": "boolean", "default": False},
            {
                "name": "source",
                "type": "string",
                "enum": ["claude-code", "codex", "pi-agent", "grok", "kimi-code"],
            },
            {"name": "recreate", "type": "boolean", "default": False, "destructive": True},
            {"name": "yes", "type": "boolean", "default": False},
            {"name": "embed", "type": "boolean"},
            {"name": "workers", "type": "string", "default": "auto"},
            {
                "name": "context",
                "type": "string",
                "enum": ["off", "template", "llm-local", "llm-remote", "llm-codex"],
            },
            {"name": "recompute_context", "type": "boolean", "default": False},
            {"name": "since", "type": "string"},
            {"name": "project", "type": "string"},
            {
                "name": "only_mode",
                "type": "string",
                "enum": ["off", "template", "llm-local", "llm-remote", "llm-codex"],
            },
            {"name": "verbose", "type": "boolean", "default": False},
            {"name": "progress", "type": "boolean", "default": True},
            {"name": "dry_run", "type": "boolean", "default": False},
            {"name": "root", "type": "string"},
            {"name": "host", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "fields", "type": "string"},
            {"name": "cta", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": [
                    "total",
                    "indexed",
                    "skipped",
                    "failed",
                    "changed",
                    "fts_rebuilt",
                    "total_seconds",
                    "context_messages",
                    "context_reused",
                    "context_mode",
                    "context_input_tokens",
                    "context_output_tokens",
                    "context_model",
                    "backlog_pending",
                    "backlog_drain_per_minute",
                ],
            }
        ),
        "safety": {"mutates": True, "destructive": False, "idempotent": True},
        "examples": [
            "recall index",
            "recall index --recreate --yes",
            "recall index --context template",
            "recall index --recompute-context --since 30d --only-mode off",
            "recall index --params '{\"full\":true}'",
        ],
    },
    "compact": {
        "command": "compact",
        "description": "Compact the local DuckDB database by rebuilding it into a fresh file.",
        "args": [],
        "options": [
            {"name": "dry_run", "type": "boolean", "default": False},
            {"name": "no_restart", "type": "boolean", "default": False},
            {"name": "yes", "type": "boolean", "default": False},
            {"name": "threshold", "type": "number", "default": 1.0},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "fields", "type": "string"},
            {"name": "cta", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": [
                    "before_bytes",
                    "before_used_bytes",
                    "before_live_bytes",
                    "before_ratio",
                    "after_bytes",
                    "after_used_bytes",
                    "after_live_bytes",
                    "after_ratio",
                    "elapsed_seconds",
                    "tables_copied",
                    "daemon_was_running",
                    "daemon_restarted",
                    "skipped_reason",
                ],
            }
        ),
        "safety": {"mutates": True, "destructive": True, "idempotent": False},
        "examples": [
            "recall compact --dry-run",
            "recall compact --threshold 5 --yes --json",
            "recall compact --no-restart --yes",
        ],
    },
    "db check-indexes": {
        "command": "db check-indexes",
        "description": (
            "Probe DuckDB ART index/table divergence by comparing index-eligible and "
            "sequential-scan counts on recently written keys; exits 1 on divergence."
        ),
        "args": [],
        "options": [
            {"name": "sample", "type": "integer", "default": 2},
            {"name": "format", "type": "string", "default": "auto"},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "fields", "type": "string"},
        ],
        "env": GLOBAL_ENV,
        "output": {
            "type": "object",
            "fields": [
                "checked_at",
                "indexes_probed",
                "samples_checked",
                "samples_unverifiable",
                "diverged",
                "diverged_count",
                "complete",
                "elapsed_seconds",
            ],
        },
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": [
            "recall db check-indexes",
            "recall db check-indexes --sample 5 --json",
        ],
    },
    "db rebuild-indexes": {
        "command": "db rebuild-indexes",
        "description": (
            "Drop and recreate every DuckDB ART index, then create any schema index that "
            "is missing; refuses while the daemon holds the database."
        ),
        "args": [],
        "options": [
            {"name": "format", "type": "string", "default": "auto"},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "fields", "type": "string"},
        ],
        "env": GLOBAL_ENV,
        "output": {
            "type": "object",
            "fields": ["dropped", "created", "healed", "elapsed_seconds"],
        },
        "safety": {"mutates": True, "destructive": False, "idempotent": True},
        "examples": [
            "recall daemon stop && recall db rebuild-indexes && recall daemon start",
            "recall db rebuild-indexes --json",
        ],
    },
    "db supersede-moved": {
        "command": "db supersede-moved",
        "description": (
            "Report, or with --apply remove, rows for transcripts indexed again after "
            "they moved or were renamed. A row is superseded only when its path is gone "
            "or resolves elsewhere and one surviving file begins with the bytes committed "
            "for it; "
            "refuses while the daemon holds the database. The default opens an existing "
            "database read-only without schema migration and includes a classified census "
            "and exact repair pairs. Only --apply mutates/deletes rows and requires an "
            "explicit RECALL_DATA_DIR; its census and pairs describe pre-apply state. "
            "Production repair requires separate per-host authorization, a reviewed backup "
            "and scope."
        ),
        "args": [],
        "options": [
            {"name": "apply", "type": "boolean", "default": False},
            {"name": "format", "type": "string", "default": "auto"},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "fields", "type": "string"},
        ],
        "env": GLOBAL_ENV,
        "output": {
            "type": "object",
            "fields": ["applied", "superseded", "moves", "census", "pairs"],
        },
        "safety": {"mutates": True, "destructive": True, "idempotent": True},
        "examples": [
            "RECALL_DATA_DIR=/absolute/path/to/owned-backup-copy recall db supersede-moved --json",
            "RECALL_DATA_DIR=/absolute/path/to/owned-backup-copy "
            "recall db supersede-moved --apply --json",
        ],
    },
    "snapshots list": {
        "command": "snapshots list",
        "description": (
            "List the entries in the local snapshots artifact directory with their size, "
            "age and why `snapshots gc` would leave them alone. `retained` marks recovery "
            "backups (`index-migration-*`, `storage-migration-*`, `recreate-*`) that gc "
            "never prunes; `partial` marks a snapshot missing one half of the "
            "DuckDB/FTS-sidecar pair, which gc refuses to delete. Read-only: use "
            "`snapshots gc --dry-run` only when the question is what a prune would remove."
        ),
        "args": [],
        "options": [
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "fields", "type": "string"},
        ],
        "env": GLOBAL_ENV,
        "output": {
            "type": "object",
            "fields": [
                "snapshots_dir",
                "snapshots_dir_missing",
                "entries",
                "entry_count",
                "total_bytes",
                "unreadable_paths",
            ],
            "properties": {
                "entries": {
                    "type": "array",
                    "fields": ["paths", "total_bytes", "age_seconds", "retained", "partial"],
                    "description": "Snapshot entries, oldest first",
                }
            },
        },
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": [
            "recall snapshots list",
            "recall snapshots list --json --fields entries,total_bytes",
        ],
    },
    "snapshots gc": {
        "command": "snapshots gc",
        "description": "Delete stale entries from the local snapshots artifact directory.",
        "args": [],
        "options": [
            {"name": "days", "type": "integer", "default": 7},
            {"name": "dry_run", "type": "boolean", "default": False},
            {"name": "yes", "type": "boolean", "default": False},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "fields", "type": "string"},
        ],
        "env": GLOBAL_ENV,
        "output": {
            "type": "object",
            "fields": [
                "removed_paths",
                "kept_paths",
                "failed_paths",
                "partial_paths",
                "total_bytes_freed",
                "dry_run",
                "snapshots_dir_missing",
            ],
        },
        "safety": {"mutates": True, "destructive": True, "idempotent": True},
        "examples": [
            "recall snapshots gc --dry-run",
            "recall snapshots gc --days 7 --yes --json",
        ],
    },
    "search": {
        "command": "search",
        "description": (
            "Search indexed sessions across messages and tool calls. Socket admission pressure"
            " returns RUNTIME/exit 1 with details.kind=daemon_busy, phase=connect, retryable=true;"
            " retry a new invocation."
        ),
        "notes": [
            "Local vector/hybrid searches report embedding-drain waiting and release as plain "
            "stderr lines in every format; stdout remains the final result or error. The "
            "5s preference does not guarantee drain completion or scan admission.",
            "Search timeouts retain exit 1 and include active/largest phase in the message; "
            "structured error.details carries kind=search_timeout, phase, largest_phase, "
            "deadline_ms, elapsed_ms, cleanup_ms and exclusive phase_ms (resolve_mode, "
            "query_embedding, enrichment_wait, scan_admission, read_queue, scan, recovery). "
            "The server deadline remains 30s; client silence errors have no server phase.",
        ],
        "args": [{"name": "query", "type": "string", "required": False}],
        "options": [
            {"name": "tool", "type": "string"},
            {
                "name": "session",
                "type": "string",
                "description": (
                    "Recall ID, source session ID, or unique 6-31 lowercase hex recall ID prefix;"
                    " unknown or ambiguous indexed IDs are errors."
                ),
            },
            {
                "name": "source",
                "type": "string",
                "enum": ["claude-code", "codex", "pi-agent", "grok", "kimi-code"],
            },
            {"name": "mode", "type": "string", "enum": ["auto", "keyword", "vector", "hybrid"]},
            {"name": "limit", "type": "integer", "default": 20},
            {"name": "fleet", "type": "boolean", "default": False},
            {"name": "fleet_config", "type": "string"},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "cta", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "array",
                "fields": [
                    "kind",
                    "session_id",
                    "source",
                    "source_path",
                    "score",
                    "message_id",
                    "tool_call_id",
                    "role",
                    "content",
                    "thinking",
                    "timestamp",
                    "tool_name",
                    "bash_command",
                    "lexical_match",
                    "host",
                ],
            }
        ),
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": [
            "recall search git",
            'recall search --params \'{"query":"git","limit":5}\'',
        ],
    },
    "list": {
        "command": "list",
        "description": (
            "List indexed sessions with optional filters. Socket admission pressure returns"
            " RUNTIME/exit 1 with details.kind=daemon_busy, phase=connect, retryable=true;"
            " retry a new invocation."
        ),
        "args": [],
        "options": [
            {
                "name": "session",
                "type": "string",
                "description": (
                    "Recall ID, source session ID, or unique recall ID prefix;"
                    " scoped by source/host, local only."
                ),
            },
            {
                "name": "source",
                "type": "string",
                "enum": ["claude-code", "codex", "pi-agent", "grok", "kimi-code"],
            },
            {"name": "since", "type": "string"},
            {"name": "project", "type": "string"},
            {"name": "host", "type": "string"},
            {"name": "fleet", "type": "boolean", "default": False},
            {"name": "fleet_config", "type": "string"},
            {"name": "limit", "type": "integer", "default": 50},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "cta", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "array",
                "fields": _SESSION_SUMMARY_FIELDS,
            }
        ),
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": [
            "recall list --since 7d",
            "recall list --fields id,source,host --limit 10",
            "recall list --fleet --since 7d",
        ],
    },
    "live": {
        "command": "live",
        "description": (
            "Inspect active agent sessions with freshness and turn state; --all also includes"
            " recent idle and ended sessions (default window 24 h), plus ended rows still watched."
            " Version 2 returns sessions with coverage and next_cursor. --fields projects sessions."
            " Fleet coverage retains per-host continuations; legacy host coverage is unknown."
            " Matching builds budget the whole local CLI read, including fresh catch-up and final"
            " hydration, within 2 s. Queued equivalent callers can share a snapshot that starts"
            " after they all arrived; later callers get a later snapshot. A partial fresh reply"
            " keeps truthful currentness; expiry is an error. Socket admission pressure returns"
            " RUNTIME/exit 1 with details.kind=daemon_busy, phase=connect, retryable=true;"
            " retry a new invocation. Fleet transport is separate."
        ),
        "args": [],
        "options": [
            {
                "name": "session",
                "type": "string",
                "description": (
                    "Recall ID, source session ID, or unique recall ID prefix;"
                    " scoped by source/host, local only."
                ),
            },
            {"name": "all", "type": "boolean", "default": False},
            {
                "name": "source",
                "type": "string",
                "enum": ["claude-code", "codex", "pi-agent", "grok", "kimi-code"],
            },
            {"name": "project", "type": "string"},
            {"name": "host", "type": "string"},
            {"name": "limit", "type": "integer", "default": 50, "minimum": 1, "maximum": 256},
            {"name": "cursor", "type": "string"},
            {"name": "fresh", "type": "boolean", "default": False},
            {"name": "fleet", "type": "boolean", "default": False},
            {"name": "fleet_config", "type": "string"},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": {
            "type": "object",
            "schema_version": 2,
            "fields": ["schema_version", "watching", "sessions", "next_cursor", "coverage"],
            "projection_target": "sessions",
            "projection_fields": [
                "path",
                "liveness",
                "freshness",
                "turn",
                "id",
                "source",
                "source_session_id",
                "host",
                "cwd",
                "git_repo",
                "git_branch",
                "model",
                "last_activity_at",
                "cursor",
                "writer_pid",
            ],
        },
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": [
            "recall live --json",
            "recall live --project /path/to/repo --json",
            "recall live --all --json",
            "recall live --json --fields id,liveness,turn",
        ],
    },
    "live mark": {
        "command": "live mark",
        "description": (
            "Record the actual writer PID using either session ID."
            " Supplies exit evidence without controlling the agent."
            " Inspect marked: false can accompany exit 0 when the daemon is unavailable or busy."
            " Busy returns the reason: daemon busy: socket admission temporarily unavailable;"
            " retry this command. The hook never auto-forks or writes a mark on busy."
        ),
        "args": [],
        "options": [
            {
                "name": "session",
                "type": "string",
                "required": True,
                "description": (
                    "Either indexed session ID or a unique recall ID prefix. Before indexing,"
                    " interpreted as a source session ID (default source claude-code); indexed IDs"
                    " infer the source. Local host only."
                ),
            },
            {"name": "pid", "type": "integer", "required": True},
            {
                "name": "source",
                "type": "string",
                "enum": ["claude-code", "codex", "pi-agent", "grok", "kimi-code"],
                "description": "Source scope; defaults to claude-code only for unindexed IDs.",
            },
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": {
            "type": "object",
            "fields": [
                "marked",
                "source",
                "source_session_id",
                "host",
                "pid",
                "marked_at",
                "reason",
            ],
        },
        "safety": {"mutates": True, "destructive": False, "idempotent": True},
        "examples": [
            "recall live mark --session $CLAUDE_SESSION_ID --pid $PPID",
        ],
    },
    "show": {
        "command": "show",
        "description": (
            "Read current or historical session messages and optional tool calls."
            " Use --tail for a bounded read, --after for cursor deltas, or --follow"
            " with --timeout for NDJSON updates. Check freshness even with --fresh."
            " Socket admission pressure returns RUNTIME/exit 1 with details.kind=daemon_busy,"
            " phase=connect, retryable=true; retry a new invocation."
        ),
        "args": [
            {
                "name": "session_id",
                "type": "string",
                "required": False,
                "description": (
                    "Recall ID, source session ID, or unique 6-31 lowercase hex recall ID prefix."
                ),
            }
        ],
        "options": [
            {"name": "tools", "type": "boolean", "default": False},
            {"name": "thinking", "type": "boolean", "default": False},
            {"name": "message_limit", "type": "integer"},
            {"name": "tail", "type": "integer"},
            {"name": "after", "type": "string"},
            {"name": "fresh", "type": "boolean", "default": False},
            {"name": "follow", "type": "boolean", "default": False},
            {"name": "timeout", "type": "number"},
            {"name": "host", "type": "string"},
            {"name": "fleet", "type": "boolean", "default": False},
            {"name": "fleet_config", "type": "string"},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "cta", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": [
                    "id",
                    "source",
                    "source_path",
                    "source_session_id",
                    "started_at",
                    "ended_at",
                    "duration_seconds",
                    "model",
                    "cwd",
                    "git_repo",
                    "git_branch",
                    "host",
                    "message_count",
                    "tool_count",
                    "input_tokens",
                    "output_tokens",
                    "is_complete",
                    "file_mtime",
                    "file_size",
                    "indexed_at",
                    "freshness",
                    "cursor",
                    "cursor_reset",
                    "cursor_reset_reason",
                    "messages",
                    "orphan_tool_calls",
                ],
            }
        ),
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": [
            "recall show <session-id>",
            "recall show <session-id> --tail 20 --fresh --json",
            "recall show <session-id> --after '<cursor>' --fresh --json",
            "recall show <session-id> --after '<cursor>' --follow --timeout 60",
            'recall show --params \'{"session_id":"abc","message_limit":10}\'',
            "recall show <session-id> --fleet --host devbox",
        ],
    },
    "daemon": {
        "command": "daemon",
        "description": "Run the background indexing daemon loop or a single cycle.",
        "args": [],
        "options": [
            {"name": "once", "type": "boolean", "default": False},
            {"name": "mode", "type": "string", "enum": ["auto", "watch", "poll"]},
            {"name": "interval", "type": "integer"},
            {"name": "embed", "type": "boolean"},
            {
                "name": "source",
                "type": "string",
                "enum": ["claude-code", "codex", "pi-agent", "grok", "kimi-code"],
            },
            {"name": "batch_size", "type": "integer"},
            {"name": "verbose", "type": "boolean", "default": False},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "fields", "type": "string"},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "cta", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": [
                    "index_summary",
                    "swapped",
                    "embed_summary",
                    "record_status_persisted",
                ],
            }
        ),
        "safety": {"mutates": True, "destructive": False, "idempotent": False},
        "examples": ["recall daemon --once", "recall daemon --once --json"],
    },
    "daemon install": {
        "command": "daemon install",
        "description": "Install scheduler artifacts for the recall daemon.",
        "args": [],
        "options": [
            {"name": "scheduler", "type": "string", "enum": ["auto", "launchd", "systemd", "cron"]},
            {"name": "dry_run", "type": "boolean", "default": False},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "cta", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": [
                    "configured_scheduler",
                    "scheduler",
                    "installed",
                    "command",
                    "config_path",
                    "artifact_paths",
                    "runtime_status",
                    "mode",
                    "resolved_mode",
                    "daemon_version",
                    "binary_version",
                    "version_drift",
                    "watched_dirs",
                    "debounce",
                    "fts_debounce",
                    "installed_binary_path",
                    "installed_binary_stale",
                    "scheduler_last_exit_status",
                    "scheduler_health_state",
                    "service_data_dir",
                    "service_measured_at",
                    "service_identity_error",
                    "reconciliation",
                ],
            }
        ),
        "safety": {"mutates": True, "destructive": False, "idempotent": True},
        "examples": ["recall daemon install", "recall daemon install --scheduler cron"],
    },
    "daemon uninstall": {
        "command": "daemon uninstall",
        "description": "Remove installed scheduler artifacts for the recall daemon.",
        "args": [],
        "options": [
            {"name": "yes", "type": "boolean", "default": False},
            {"name": "dry_run", "type": "boolean", "default": False},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "cta", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": [
                    "configured_scheduler",
                    "scheduler",
                    "installed",
                    "command",
                    "config_path",
                    "artifact_paths",
                    "runtime_status",
                    "mode",
                    "resolved_mode",
                    "watched_dirs",
                    "debounce",
                    "fts_debounce",
                    "installed_binary_path",
                    "installed_binary_stale",
                    "scheduler_last_exit_status",
                    "scheduler_health_state",
                    "service_data_dir",
                    "service_measured_at",
                    "service_identity_error",
                ],
            }
        ),
        "safety": {"mutates": True, "destructive": True, "idempotent": True},
        "examples": ["recall daemon uninstall --yes"],
    },
    "daemon start": {
        "command": "daemon start",
        "description": "Start the installed daemon service; refuses when it runs another data dir.",
        "args": [],
        "options": [
            {
                "name": "background",
                "type": "boolean",
                "default": False,
                "description": (
                    "Start this data dir's daemon as a direct background process, "
                    "without the installed service"
                ),
            },
            {"name": "timeout", "type": "number", "default": 10.0},
            {"name": "verbose", "type": "boolean", "default": False},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "cta", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": ["scheduler", "started", "pid", "duration_seconds", "message"],
            }
        ),
        "safety": {"mutates": True, "destructive": False, "idempotent": True},
        "examples": ["recall daemon start", "recall daemon start --json"],
    },
    "daemon stop": {
        "command": "daemon stop",
        "description": (
            "Stop the installed daemon service; refuses when it runs another data dir (see --soft)."
        ),
        "args": [],
        "options": [
            {"name": "soft", "type": "boolean", "default": False},
            {"name": "timeout", "type": "number", "default": 10.0},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "cta", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": ["scheduler", "stopped", "pid", "duration_seconds", "message"],
            }
        ),
        "safety": {"mutates": True, "destructive": False, "idempotent": True},
        "examples": [
            "recall daemon stop",
            "recall daemon stop --soft",
            "recall daemon stop --json",
        ],
    },
    "daemon restart": {
        "command": "daemon restart",
        "description": (
            "Restart the installed daemon service; refuses when it runs another data dir."
        ),
        "args": [],
        "options": [
            {"name": "timeout", "type": "number", "default": 10.0},
            {"name": "verbose", "type": "boolean", "default": False},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
            {"name": "cta", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": [
                    "scheduler",
                    "stopped",
                    "started",
                    "pid",
                    "duration_seconds",
                    "message",
                ],
            }
        ),
        "safety": {"mutates": True, "destructive": False, "idempotent": True},
        "examples": [
            "recall daemon restart",
            "recall daemon restart --json",
        ],
    },
    "daemon status": {
        "command": "daemon status",
        "description": (
            "Show daemon scheduler and runtime status. The `reconciliation` object is the "
            "indexing backlog: `pending` counts catalogued transcript files whose committed "
            "index is behind the file on disk (a changed generation, unread bytes, or a "
            "blocking recorded error), excluding `missing` files and files under disabled roots "
            "(`out_of_scope_pending`). It falls as the daemon commits work and rises when "
            "discovery finds new or rewritten files, so read it twice before calling it "
            "stuck. `catalog_scan_complete` says whether every root finished a scan, "
            "`noticed_pending` counts files with reported writes awaiting observation. "
            "A mark leaves after observation commits or after three failed captures; "
            "give-up lists the path in `reconciliation.error`. All enabled readiness "
            "flags stay false while `noticed_pending` is nonzero; disabled embedding "
            "leaves `enrichment_ready` true. `raw_indexing_ready` is a complete scan plus "
            "`pending == 0` and `noticed_pending == 0`, and is false until every root has "
            "been scanned since this daemon process started, `keyword_search_ready` adds "
            "an empty keyword sidecar queue, `paused` reports a durable operator pause, and "
            "`enrichment_deferred` explains why the embed phase stood down without blocking "
            "the drain. Embedding is counted in message and tool-call rows, apart from "
            "`pending`: `embed_pending_eligible` (sessions idle past `embed_idle_session`) and "
            "`embed_pending_held` (sessions still inside that hold) are read fresh on every "
            "call at `embed_pending_measured_at`, while `embed_pending`/`embed_pending_at` are "
            "the embed loop's last eligible snapshot. Queued equivalent callers may share one "
            "coherent count/coverage snapshot starting after they all arrived; later callers "
            "get a later snapshot. Matching builds budget the whole local CLI read within "
            "2 s; deadline expiry is an error. Socket admission pressure returns RUNTIME/exit 1"
            " with details.kind=daemon_busy, phase=connect, retryable=true for default and"
            " explicit pages; retry a new invocation, with no offline fallback. "
            "`enrichment_ready` is true when "
            "embedding is disabled, or when an embed phase is running, `raw_indexing_ready` "
            "is true, both fresh counts are 0 and no embed error is set, so it is the drain "
            "signal; "
            "`embed_pending_eligible == 0` alone means only live sessions wait. `coverage[]` "
            "breaks the backlog down per source with `oldest_pending_age`; `source_page`/"
            "`next_cursor` page the per-file records; pass `next_cursor` back exactly as "
            "returned; an empty or malformed cursor, or one naming no catalogued file, is a "
            "validation error. With no "
            "daemon, `--limit`/`--cursor` page the local catalog. `--fields reconciliation` "
            "projects the whole object and `--fields reconciliation.<key>` one of its keys, "
            "e.g. `reconciliation.enrichment_ready`. See docs/reconciliation-operations.md."
        ),
        "args": [],
        "options": [
            {"name": "limit", "type": "integer", "default": 10, "minimum": 1, "maximum": 256},
            {"name": "cursor", "type": "string"},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "cta", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": _DAEMON_STATUS_FIELDS,
                "projection_fields": _DAEMON_STATUS_PROJECTION_FIELDS,
            }
        ),
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": [
            "recall daemon status",
            "recall daemon status --fields scheduler,installed",
            "recall daemon status --json --fields reconciliation.enrichment_ready",
        ],
    },
    "daemon migrate-storage": {
        "command": "daemon migrate-storage",
        "description": "Inspect or request backed-up fixed-format storage maintenance.",
        "args": [],
        "options": [
            {"name": "dry_run", "type": "boolean", "default": False},
            {"name": "plan_id", "type": "string"},
            {"name": "wait", "type": "boolean", "default": False},
            {"name": "timeout", "type": "number", "default": 5.0},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": {
            "type": "object",
            "schema_version": 1,
            "fields": [
                "schema_version",
                "operation_id",
                "plan_id",
                "db_path",
                "desired_storage_version",
                "migration",
                "status",
                "accepted",
                "dry_run",
                "error",
            ],
        },
        "safety": {"mutates": True, "destructive": False, "idempotent": True},
        "examples": [
            "recall daemon migrate-storage --dry-run --json",
            "recall daemon migrate-storage --plan-id <digest> --wait --json",
        ],
    },
    "daemon pause": {
        "command": "daemon pause",
        "description": "Persistently pause reconciliation while keeping status reads available.",
        "args": [],
        "options": [
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": {"type": "object", "fields": ["paused"]},
        "safety": {"mutates": True, "destructive": False, "idempotent": True},
        "examples": ["recall daemon pause --json"],
    },
    "daemon resume": {
        "command": "daemon resume",
        "description": "Resume reconciliation after a durable maintenance pause.",
        "args": [],
        "options": [
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": {"type": "object", "fields": ["paused"]},
        "safety": {"mutates": True, "destructive": False, "idempotent": True},
        "examples": ["recall daemon resume --json"],
    },
    "fleet status": {
        "command": "fleet status",
        "description": "Probe fleet inventory hosts for reachability, versions, and drift.",
        "args": [],
        "options": [
            {"name": "config", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
        ],
        "env": GLOBAL_ENV,
        "output": {
            "type": "array",
            "fields": [
                "name",
                "ssh",
                "ok",
                "binary_version",
                "daemon_version",
                "version_drift",
                "error",
            ],
        },
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": ["recall fleet status --json", "recall fleet status --config fleet.toml"],
    },
    "stats": {
        "command": "stats",
        "description": (
            "Show top-level analytics counts. Socket admission pressure returns RUNTIME/exit 1"
            " with details.kind=daemon_busy, phase=connect, retryable=true; retry a new invocation."
        ),
        "args": [],
        "options": [
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "cta", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": [
                    "sessions",
                    "messages",
                    "tool_calls",
                    "bash_calls",
                    "last_index_total",
                    "last_index_indexed",
                    "last_index_skipped",
                    "last_index_failed",
                    "last_context_messages",
                    "last_context_mode",
                    "last_context_input_tokens",
                    "last_context_output_tokens",
                    "last_context_model",
                ],
            }
        ),
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": ["recall stats", "recall stats --format json"],
    },
    "stats tools": {
        "command": "stats tools",
        "description": "Show tool usage counts.",
        "args": [],
        "options": [
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "cta", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope({"type": "array", "fields": ["tool_name", "count"]}),
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": ["recall stats tools"],
    },
    "stats skills": {
        "command": "stats skills",
        "description": "Show skill invocations with complete local or fleet coverage.",
        "args": [],
        "options": [
            {"name": "since", "type": "string"},
            {
                "name": "source",
                "type": "string[]",
                "repeatable": True,
                "enum": ["claude-code", "codex", "pi-agent", "grok", "kimi-code"],
            },
            {"name": "local", "type": "boolean", "default": False},
            {"name": "fleet_config", "type": "string"},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "cta", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "object",
                "fields": ["rows", "coverage"],
                "properties": {
                    "rows": {
                        "type": "array",
                        "fields": [
                            "skill_name",
                            "source",
                            "host",
                            "invocations",
                            "sessions",
                        ],
                    },
                    "coverage": {
                        "type": "object",
                        "fields": [
                            "scope",
                            "expected_hosts",
                            "successful_hosts",
                            "covered_sources",
                            "considered_sessions",
                            "attributed_invocations",
                            "unattributed_candidates",
                            "control",
                        ],
                    },
                },
            }
        ),
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": [
            "recall stats skills --local",
            "recall stats skills --since 30d --json",
            "recall stats skills --source codex --source grok --json",
        ],
    },
    "stats bash": {
        "command": "stats bash",
        "description": "Show bash command usage or permission suggestions.",
        "args": [],
        "options": [
            {"name": "suggest", "type": "boolean", "default": False},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "cta", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "oneOf",
                "variants": [
                    {
                        "when": {"suggest": False},
                        "schema": {
                            "type": "array",
                            "fields": ["bash_base", "bash_sub", "count", "is_compound"],
                        },
                    },
                    {
                        "when": {"suggest": True},
                        "schema": {
                            "type": "object",
                            "fields": ["suggestions", "skipped"],
                            "properties": {
                                "suggestions": {
                                    "type": "array",
                                    "fields": ["pattern", "count", "confidence", "reason"],
                                },
                                "skipped": {
                                    "type": "array",
                                    "fields": ["pattern", "count", "reason"],
                                },
                            },
                        },
                    },
                ],
            }
        ),
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": ["recall stats bash", "recall stats bash --suggest"],
    },
    "stats tokens": {
        "command": "stats tokens",
        "description": "Show token usage grouped by repository.",
        "args": [],
        "options": [
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "cta", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {"type": "array", "fields": ["repo", "input_tokens", "output_tokens"]},
        ),
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": ["recall stats tokens"],
    },
    "stats usage": {
        "command": "stats usage",
        "description": "Show token usage by source, model, and host.",
        "args": [],
        "options": [
            {"name": "since", "type": "string"},
            {"name": "fleet", "type": "boolean", "default": False},
            {"name": "fleet_config", "type": "string"},
            {"name": "fields", "type": "string"},
            {"name": "format", "type": "string", "enum": ["auto", "text", "json", "jsonl", "toon"]},
            {"name": "json", "type": "boolean", "default": False},
            {"name": "cta", "type": "boolean", "default": False},
            {"name": "params", "type": "json"},
        ],
        "env": GLOBAL_ENV,
        "output": _with_cta_envelope(
            {
                "type": "array",
                "fields": [
                    "source",
                    "model",
                    "host",
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "fresh_input_tokens",
                    "session_count",
                ],
            },
        ),
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": [
            "recall stats usage",
            "recall stats usage --since 30d",
            "recall stats usage --fleet --since 7d",
        ],
    },
}


class OperationSafety(TypedDict):
    mutates: bool
    destructive: bool
    idempotent: bool


def request_safety(command: str, request: Mapping[str, object]) -> OperationSafety:
    """Classify the requested operation's effects, even when only previewing it."""
    metadata = COMMANDS[command]
    defaults = metadata["safety"]
    destructive_option = any(
        option.get("destructive", False) and request.get(option["name"]) is True
        for option in metadata["options"]
    )
    return {
        "mutates": defaults["mutates"],
        "destructive": defaults["destructive"] or destructive_option,
        "idempotent": defaults["idempotent"],
    }


def _command_tree() -> TyperGroup:
    # Resolve after registration: command modules use the semantic metadata above.
    from recall.cli.app import app

    return get_group(app)


def _with_option_flags(key: str, root: TyperCommand | TyperGroup) -> dict[str, Any]:
    command = root
    for part in key.split():
        assert isinstance(command, TyperGroup), f"command is not a group: {key}"
        command = command.commands[part]
    flags = {
        param.opts[0].removeprefix("--").replace("-", "_"): [*param.opts, *param.secondary_opts]
        for param in command.params
        if isinstance(param, TyperOption)
    }
    metadata = COMMANDS[key]
    return {
        **metadata,
        "options": [{**option, "flags": flags[option["name"]]} for option in metadata["options"]],
    }


def full_manifest() -> dict[str, Any]:
    root = _command_tree()
    return {
        "name": "recall",
        "commands": {key: _with_option_flags(key, root) for key in COMMANDS},
    }


def command_manifest(path: list[str]) -> dict[str, Any]:
    key = " ".join(path).strip()
    if not key:
        return full_manifest()
    if key not in COMMANDS:
        raise ValueError(f"unknown command: {key}")
    return _with_option_flags(key, _command_tree())


def output_fields_for_variant(
    command: str,
    variant_match: dict[str, Any],
) -> set[str] | None:
    """Return allowed output fields for a specific oneOf variant.

    For commands with oneOf output, matches the variant whose "when" clause
    matches variant_match and returns only that variant's fields.
    Falls back to output_fields_for() for non-variant commands.
    """
    manifest = COMMANDS.get(command)
    if manifest is None:
        return None
    output = manifest["output"]
    if "variants" not in output:
        return output_fields_for(command)
    for variant in output["variants"]:
        if variant.get("when") == variant_match:
            schema = variant.get("schema", {})
            if "fields" in schema:
                return set(schema["fields"])
    return output_fields_for(command)


def output_fields_for(command: str) -> set[str] | None:
    manifest = COMMANDS.get(command)
    if manifest is None:
        return None
    output = manifest["output"]
    if "projection_fields" in output:
        return set(output["projection_fields"])
    if "fields" in output:
        return set(output["fields"])
    if "variants" in output:
        fields: set[str] = set()
        for variant in output["variants"]:
            schema = variant.get("schema", {})
            if "fields" in schema:
                fields.update(schema["fields"])
        return fields or None
    return None
