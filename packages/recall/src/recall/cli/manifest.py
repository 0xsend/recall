from __future__ import annotations

from typing import Any

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
                    "before_live_bytes",
                    "before_ratio",
                    "after_bytes",
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
            "their project directory moved. A row is superseded only when its path is "
            "gone and the surviving file begins with the bytes committed for it; "
            "refuses while the daemon holds the database."
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
            "fields": ["applied", "superseded", "moves"],
        },
        "safety": {"mutates": True, "destructive": True, "idempotent": True},
        "examples": [
            "recall daemon stop && recall db supersede-moved && recall daemon start",
            "recall daemon stop && recall db supersede-moved --apply --json && recall daemon start",
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
        "description": "Search indexed sessions across messages and tool calls.",
        "args": [{"name": "query", "type": "string", "required": False}],
        "options": [
            {"name": "tool", "type": "string"},
            {"name": "session", "type": "string"},
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
        "description": "List indexed sessions with optional filters.",
        "args": [],
        "options": [
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
                "fields": [
                    "id",
                    "source",
                    "started_at",
                    "ended_at",
                    "cwd",
                    "git_repo",
                    "git_branch",
                    "model",
                    "message_count",
                    "tool_count",
                    "input_tokens",
                    "output_tokens",
                    "is_complete",
                    "host",
                    "file_mtime",
                    "file_size",
                    "indexed_at",
                    "last_activity_at",
                ],
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
        ),
        "args": [],
        "options": [
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
            "Record the actual writer PID for a harness source session ID."
            " Supplies exit evidence without controlling the agent."
            " Inspect marked: false can accompany exit 0 when the daemon is unavailable."
        ),
        "args": [],
        "options": [
            {"name": "session", "type": "string", "required": True},
            {"name": "pid", "type": "integer", "required": True},
            {
                "name": "source",
                "type": "string",
                "enum": ["claude-code", "codex", "pi-agent", "grok", "kimi-code"],
                "default": "claude-code",
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
        ),
        "args": [{"name": "session_id", "type": "string", "required": False}],
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
                ],
            }
        ),
        "safety": {"mutates": True, "destructive": True, "idempotent": True},
        "examples": ["recall daemon uninstall --yes"],
    },
    "daemon start": {
        "command": "daemon start",
        "description": "Start the installed daemon scheduler.",
        "args": [],
        "options": [
            {
                "name": "background",
                "type": "boolean",
                "default": False,
                "description": "Legacy direct background process start instead of scheduler start",
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
        "description": "Stop the running daemon scheduler.",
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
        "description": "Restart the installed daemon scheduler.",
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
            "recorded error), excluding `missing` files and files under disabled roots "
            "(`out_of_scope_pending`). It falls as the daemon commits work and rises when "
            "discovery finds new or rewritten files, so read it twice before calling it "
            "stuck. `catalog_scan_complete` says whether every root finished a scan, "
            "`raw_indexing_ready` is that plus `pending == 0`, `keyword_search_ready` adds "
            "an empty keyword sidecar queue, `paused` reports a durable operator pause, and "
            "`enrichment_deferred` explains why the embed phase stood down without blocking "
            "the drain (`embed_pending` is a separate, row-level count). `coverage[]` breaks "
            "the backlog down per source with `oldest_pending_age`; `source_page`/"
            "`next_cursor` page the per-file records. `--fields reconciliation` projects the "
            "whole object. See docs/reconciliation-operations.md."
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
                "fields": [
                    "reconciliation",
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
                    "daemon_version",
                    "binary_version",
                    "version_drift",
                    "bloat_ratio",
                    "bloat_ratio_threshold",
                    "bloat_auto_trigger",
                    "index_divergence",
                    "startup_refusal",
                    "launchd_legacy_label",
                    "launchd_legacy_leftover",
                    "daemon_pid",
                    "runtime_unavailable_reason",
                    "debounce",
                    "fts_debounce",
                    "fts_sidecar_enabled",
                    "fts_sidecar_bootstrap_messages_processed",
                    "fts_sidecar_bootstrap_tool_calls_processed",
                    "fts_sidecar_bootstrap_messages_done",
                    "fts_sidecar_bootstrap_tool_calls_done",
                    "fts_sidecar_reconcile_pending_drained_messages",
                    "fts_sidecar_reconcile_pending_drained_tool_calls",
                    "fts_sidecar_reconcile_orphans_backfilled_messages",
                    "fts_sidecar_reconcile_orphans_backfilled_tool_calls",
                    "fts_sidecar_reconcile_ghosts_deleted_messages",
                    "fts_sidecar_reconcile_ghosts_deleted_tool_calls",
                    "fts_sidecar_reconcile_pending_remaining_messages",
                    "fts_sidecar_reconcile_pending_remaining_tool_calls",
                    "fts_sidecar_last_run_at",
                    "fts_sidecar_error",
                    "last_fts_rebuild_failure_at",
                    "last_fts_rebuild_failure_reason",
                    "fts_rebuild_consecutive_failures",
                    "fts_rebuild_next_retry_at",
                    "catchup_in_progress",
                    "catchup_total",
                    "catchup_done",
                    "live_fresh_requests",
                    "live_fresh_timeouts",
                    "follow_subscriptions",
                    "embed_phase_enabled",
                    "embed_model_loaded",
                    "embed_pending",
                    "embed_pending_at",
                    "embed_last_batch_at",
                    "embed_last_batch_size",
                    "embed_last_batch_duration",
                    "embed_loop_iterations",
                    "embed_loop_last_iteration_at",
                    "embed_loop_last_trigger",
                    "embed_loop_stage",
                    "embed_loop_stage_at",
                    "embed_loop_last_outcome",
                    "embed_loop_next_interval",
                    "embed_requested_cycles",
                    "embed_requested_at",
                    "embed_requested_stage",
                    "embed_requested_stage_at",
                    "embed_deferred_reason",
                    "embed_deferred_at",
                    "embed_last_error",
                    "embed_cooldown_sessions",
                    "embed_cooldown_until",
                    "watch_total_indexed",
                    "watch_total_failed",
                    "watch_avg_duration",
                    "watch_min_duration",
                    "watch_max_duration",
                    "watch_last_event_at",
                    "watch_last_event_duration",
                    "watch_recent_events",
                    "installed_binary_path",
                    "installed_binary_stale",
                    "scheduler_last_exit_status",
                    "scheduler_health_state",
                ],
            }
        ),
        "safety": {"mutates": False, "destructive": False, "idempotent": True},
        "examples": ["recall daemon status", "recall daemon status --fields scheduler,installed"],
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
    "stats": {
        "command": "stats",
        "description": "Show top-level analytics counts.",
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


def full_manifest() -> dict[str, Any]:
    return {
        "name": "recall",
        "commands": COMMANDS,
    }


def command_manifest(path: list[str]) -> dict[str, Any]:
    key = " ".join(path).strip()
    if not key:
        return full_manifest()
    if key not in COMMANDS:
        raise ValueError(f"unknown command: {key}")
    return COMMANDS[key]


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
