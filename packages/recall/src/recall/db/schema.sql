CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- schema_migrations records every applied migration (idempotent via INSERT OR IGNORE).
-- Populated by the migration framework in ensure_schema for upgrades from v13+.
-- See SPEC "Migration Policy" (REQ-MIG-003) and db/migrations/ package.
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- schema_migration_undo holds the pre-image of rows a data-mutating migration
-- rewrites (REQ-MIG-008), so such a migration is reversible without copying the
-- whole database aside. Bounded by the rows each migration actually touches.
CREATE TABLE IF NOT EXISTS schema_migration_undo (
    migration_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    row_key TEXT NOT NULL,
    column_name TEXT NOT NULL,
    old_value TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (migration_id, table_name, row_key, column_name)
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_path TEXT UNIQUE NOT NULL,
    source_session_id TEXT
);

-- Durable desired/committed source state.  This deliberately does not use
-- session_state checkpoints: a source can be observed before it has a session.
CREATE TABLE IF NOT EXISTS source_files (
    source_key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_path TEXT NOT NULL,
    root_path TEXT NOT NULL,
    session_id TEXT,
    dev BIGINT,
    inode BIGINT,
    ctime_ns BIGINT NOT NULL,
    mtime_ns BIGINT NOT NULL,
    size BIGINT NOT NULL,
    sidecar_mtime_ns BIGINT NOT NULL DEFAULT 0,
    parser_revision TEXT NOT NULL DEFAULT '', sidecar_signature TEXT NOT NULL DEFAULT '',
    desired_generation BIGINT NOT NULL DEFAULT 0,
    committed_generation BIGINT NOT NULL DEFAULT 0,
    committed_offset BIGINT NOT NULL DEFAULT 0,
    committed_prefix_sha256 TEXT,
    content_epoch BIGINT NOT NULL DEFAULT 0,
    first_pending_at DOUBLE,
    first_pending_seq BIGINT NOT NULL DEFAULT 0,
    last_serviced_seq BIGINT NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_retry_at DOUBLE NOT NULL DEFAULT 0,
    last_error TEXT,
    diagnostics JSON,
    missing BOOLEAN NOT NULL DEFAULT FALSE,
    observed_at DOUBLE NOT NULL,
    -- Resume proof for the committed prefix, written only by the acknowledgement
    -- that committed it. NULL means "no proof exists"; see REQ-INDEX-025.
    normalization_checkpoint TEXT
);
-- The pending scan filters on column-to-column comparisons, which no ART index
-- can serve, so source_files carries only the source/path lookup (REQ-MIG-010).
CREATE INDEX IF NOT EXISTS idx_source_files_source_path ON source_files(source, source_path);

CREATE TABLE IF NOT EXISTS reconciliation_roots (
    source TEXT NOT NULL,
    root_path TEXT NOT NULL,
    scan_started_at DOUBLE,
    scan_finished_at DOUBLE,
    discovered_count BIGINT NOT NULL DEFAULT 0,
    failure_count BIGINT NOT NULL DEFAULT 0,
    failures JSON,
    scan_complete BOOLEAN NOT NULL DEFAULT FALSE,
    scan_generation BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (source, root_path)
);

-- NOTE on "source" and "role" columns:
-- We deliberately do NOT use a CHECK constraint here.
-- `source` (claude_code / codex / pi_agent / grok / future agents) and
-- `role` (user / assistant / system) are intentionally open sets.
-- A restrictive CHECK forces every user with a large history to run
-- `recall index --recreate --yes` just to add support for one more agent.
-- Validation + allow-listing is done in Python (parse_source / Role enum + parser registry).
-- This is the pragmatic tradeoff for a derived, rebuildable database.

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    agent_id TEXT,
    UNIQUE(session_id, idx)
);

CREATE TABLE IF NOT EXISTS session_state (
    session_id TEXT PRIMARY KEY,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    duration_seconds INTEGER,

    model TEXT,
    cwd TEXT,
    git_repo TEXT,
    git_branch TEXT,

    message_count INTEGER DEFAULT 0,
    tool_count INTEGER DEFAULT 0,
    input_tokens BIGINT,
    output_tokens BIGINT,
    cached_input_tokens BIGINT,
    host TEXT NOT NULL DEFAULT 'local',

    is_complete BOOLEAN DEFAULT TRUE,
    file_mtime DOUBLE NOT NULL,
    file_size BIGINT NOT NULL,
    -- Newest mtime across the parser's declared sidecars (REQ-INDEX-018).
    -- 0.0 means the source declares none; NULL means the row predates this
    -- column and must re-index once to pick up sidecar-sourced metadata.
    sidecar_mtime DOUBLE DEFAULT 0,
    last_byte_offset BIGINT DEFAULT 0,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fleet usage ledger (REQ-USAGE-016): per-inference events + log cursors.
CREATE TABLE IF NOT EXISTS usage_events (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    session_id TEXT,
    ts TIMESTAMP,
    prompt_tokens BIGINT,
    cached_prompt_tokens BIGINT,
    completion_tokens BIGINT,
    reasoning_tokens BIGINT,
    host TEXT,
    harvested_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_events_source_sid
    ON usage_events(source, source_session_id);
CREATE INDEX IF NOT EXISTS idx_usage_events_session ON usage_events(session_id);
CREATE INDEX IF NOT EXISTS idx_usage_events_ts ON usage_events(ts);

CREATE TABLE IF NOT EXISTS usage_log_cursors (
    path TEXT PRIMARY KEY,
    byte_offset BIGINT NOT NULL DEFAULT 0,
    file_size BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS message_state (
    message_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    content TEXT,
    thinking TEXT,
    timestamp TIMESTAMP,
    has_thinking BOOLEAN DEFAULT FALSE,
    context_text TEXT DEFAULT '',
    context_mode TEXT DEFAULT 'off' CHECK (
        context_mode IS NULL
        OR context_mode IN ('off', 'template', 'llm-local', 'llm-remote', 'llm-codex')
    ),
    fts_content TEXT DEFAULT '',
    fts_thinking TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_id TEXT,
    idx INTEGER NOT NULL,

    tool_name TEXT NOT NULL,
    tool_input JSON,

    bash_command TEXT,
    bash_base TEXT,
    bash_sub TEXT,
    is_compound BOOLEAN DEFAULT FALSE,

    agent_id TEXT,
    subagent_type TEXT,
    subagent_description TEXT,
    subagent_model TEXT,
    skill_name TEXT
);

CREATE TABLE IF NOT EXISTS message_embeddings (
    message_id TEXT PRIMARY KEY,
    content_embedding FLOAT[__EMBED_DIM__],
    thinking_embedding FLOAT[__EMBED_DIM__]
);

-- REQ-LIVE-006: a tool result lives in its own insert-only table, never as a
-- column on tool_calls. DuckDB's UPDATE is DELETE+INSERT, and tool_calls is the
-- row set whose churn produced the 140 GiB tool_call_embeddings bloat
-- (REQ-INDEX-017); widening it with mutable columns would reopen that path.
CREATE TABLE IF NOT EXISTS tool_results (
    tool_call_id TEXT PRIMARY KEY,
    result_summary TEXT,
    is_error BOOLEAN,
    completed_at TIMESTAMP
);

-- The harness-native id of a tool_use, mapped to recall's tool_call id.
-- tool_calls.id is a positional hash of (message_id, idx), so this is the only
-- key a result arriving in a later incremental chunk can be paired on. Its own
-- table for the same reason tool_results is: writing it as a tool_calls column
-- would make one full re-parse UPDATE every historical row.
CREATE TABLE IF NOT EXISTS tool_use_ids (
    tool_call_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    tool_use_id TEXT NOT NULL
);

-- REQ-LIVE-005: harness stop markers, anchored to the message each one closes.
-- Insert-only and a side table for the same reason tool_use_ids is: a column on
-- messages or tool_calls would be NULL on every historical row and non-NULL
-- after one re-parse, and a DuckDB UPDATE is DELETE+INSERT. `ends_turn` is the
-- parser's reading of its own harness vocabulary, so the read-time derivation
-- stays generic; `reason` is that vocabulary passed through unnormalized.
CREATE TABLE IF NOT EXISTS session_stop_markers (
    session_id TEXT NOT NULL,
    message_idx INTEGER NOT NULL,
    reason TEXT NOT NULL,
    ends_turn BOOLEAN NOT NULL,
    PRIMARY KEY (session_id, message_idx)
);

-- REQ-LIVE-008: enrichment written by a harness hook, never inferred from
-- process trees. Absent marks change nothing else about a live session.
CREATE TABLE IF NOT EXISTS live_marks (
    source TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    host TEXT NOT NULL,
    pid BIGINT,
    surface_key TEXT,
    marked_at TIMESTAMP,
    PRIMARY KEY (source, source_session_id, host)
);

CREATE TABLE IF NOT EXISTS tool_call_embeddings (
    tool_call_id TEXT PRIMARY KEY,
    bash_embedding FLOAT[__EMBED_DIM__]
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('content', 'thinking', 'bash')),
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    embedding FLOAT[__EMBED_DIM__] NOT NULL,
    normalization_version INTEGER NOT NULL DEFAULT 1,
    context_version INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fts_sidecar_pending (
    kind TEXT NOT NULL,
    id TEXT NOT NULL,
    op TEXT NOT NULL,
    queued_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runtime_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    last_attempted_at TIMESTAMP,
    last_successful_at TIMESTAMP,
    last_run_kind TEXT CHECK (
        last_run_kind IS NULL
        OR last_run_kind IN ('index', 'embed', 'daemon-once', 'daemon-scheduled', 'daemon-watch')
    ),
    last_index_total INTEGER,
    last_index_indexed INTEGER,
    last_index_skipped INTEGER,
    last_index_failed INTEGER,
    last_index_changed INTEGER,
    last_index_total_seconds DOUBLE,
    last_embed_messages INTEGER,
    last_embed_thinking INTEGER,
    last_embed_bash INTEGER,
    last_failure_message TEXT,
    last_failure_at TIMESTAMP,
    installed_scheduler TEXT CHECK (
        installed_scheduler IS NULL
        OR installed_scheduler IN ('launchd', 'systemd', 'cron')
    ),
    last_context_messages INTEGER,
    last_context_mode TEXT,
    last_context_input_tokens BIGINT,
    last_context_output_tokens BIGINT,
    last_context_model TEXT,
    embedding_dimensions INTEGER,
    -- Failure-signature memory and index self-repair (REQ-RESIL-014..019).
    -- Nullable with defaults so migration 0023 (ADD COLUMN cannot carry NOT NULL)
    -- yields the same shape; readers coalesce NULL to 0 / FALSE.
    last_fatal_signature TEXT,
    fatal_repeat_count INTEGER DEFAULT 0,
    -- When the signature was last recorded (migration 0029); without it an old
    -- signature reads as a live failure.
    last_fatal_at TIMESTAMP,
    last_index_repair_at TIMESTAMP,
    last_index_repair_signature TEXT,
    needs_index_verification BOOLEAN DEFAULT FALSE,
    -- Applied INDEX_MIGRATION_VERSION. 0 means never applied (existing DBs).
    -- Fresh databases are initialized to the package version in ensure_schema.
    index_migration_version INTEGER DEFAULT 0
);

INSERT OR IGNORE INTO runtime_state (singleton) VALUES (TRUE);

CREATE TABLE IF NOT EXISTS index_migration_jobs (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    target_version INTEGER NOT NULL,
    phase TEXT NOT NULL CHECK (
        phase IN ('idle', 'backup', 'running', 'verifying', 'failed')
    ),
    backup_path TEXT,
    captured_count BIGINT NOT NULL DEFAULT 0,
    completed_count BIGINT NOT NULL DEFAULT 0,
    started_at TIMESTAMP,
    error TEXT,
    storage_target TEXT,
    storage_attempt BIGINT NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO index_migration_jobs (singleton, target_version, phase)
VALUES (TRUE, 0, 'idle');

CREATE TABLE IF NOT EXISTS index_migration_scope (
    source_key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_path TEXT NOT NULL,
    completed BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_session_state_cwd ON session_state(cwd);
CREATE INDEX IF NOT EXISTS idx_session_state_git_repo ON session_state(git_repo);
CREATE INDEX IF NOT EXISTS idx_session_state_started ON session_state(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_message_state_has_thinking ON message_state(has_thinking);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_bash_base ON tool_calls(bash_base);
CREATE INDEX IF NOT EXISTS idx_tool_calls_bash_sub ON tool_calls(bash_sub);
CREATE INDEX IF NOT EXISTS idx_messages_agent ON messages(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_agent ON tool_calls(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_subagent_type ON tool_calls(subagent_type);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_kind ON embedding_cache(kind);
CREATE INDEX IF NOT EXISTS idx_fts_sidecar_pending_kind_id ON fts_sidecar_pending(kind, id);
-- Pairing a tool_result to its call is a lookup by (session, harness id).
CREATE INDEX IF NOT EXISTS idx_tool_use_ids_lookup ON tool_use_ids(session_id, tool_use_id);
