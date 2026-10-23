CREATE TABLE semantic_work (
    kind TEXT NOT NULL CHECK (kind IN ('message', 'tool')),
    id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    missing_content BOOLEAN NOT NULL,
    missing_thinking BOOLEAN NOT NULL,
    missing_bash BOOLEAN NOT NULL,
    needs_marker BOOLEAN NOT NULL,
    raw_available BOOLEAN NOT NULL,
    PRIMARY KEY (kind, id),
    CHECK ((kind = 'message' AND NOT missing_bash) OR
           (kind = 'tool' AND NOT missing_content AND NOT missing_thinking AND raw_available)),
    CHECK (missing_content OR missing_thinking OR missing_bash OR needs_marker OR NOT raw_available)
);
CREATE INDEX idx_semantic_work_session ON semantic_work(session_id);
CREATE TABLE semantic_work_counts (
    session_id TEXT PRIMARY KEY,
    pending_messages BIGINT NOT NULL CHECK (pending_messages >= 0),
    pending_tools BIGINT NOT NULL CHECK (pending_tools >= 0),
    missing_content BIGINT NOT NULL CHECK (missing_content BETWEEN 0 AND pending_messages),
    missing_thinking BIGINT NOT NULL CHECK (missing_thinking BETWEEN 0 AND pending_messages),
    missing_bash BIGINT NOT NULL CHECK (missing_bash BETWEEN 0 AND pending_tools),
    raw_blocked_messages BIGINT NOT NULL CHECK (raw_blocked_messages BETWEEN 0 AND pending_messages),
    CHECK (pending_messages + pending_tools > 0)
);
CREATE TABLE semantic_work_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    definition_version INTEGER NOT NULL CHECK (definition_version = 1),
    phase TEXT NOT NULL CHECK (phase IN ('uninitialized', 'building', 'verifying', 'complete', 'failed')),
    build_id TEXT,
    backup_path TEXT,
    source_schema_version INTEGER,
    source_manifest_sha256 TEXT,
    captured_messages BIGINT,
    captured_tools BIGINT,
    cursor_kind TEXT CHECK (cursor_kind IN ('message', 'tool')),
    cursor_id TEXT,
    completed_targets BIGINT NOT NULL DEFAULT 0 CHECK (completed_targets >= 0),
    mutation_seq BIGINT NOT NULL DEFAULT 0 CHECK (mutation_seq >= 0),
    started_at TIMESTAMP,
    verified_at TIMESTAMP,
    audited_at TIMESTAMP,
    error TEXT,
    CHECK ((cursor_kind IS NULL) = (cursor_id IS NULL)),
    CHECK (phase <> 'complete' OR verified_at IS NOT NULL)
);
INSERT INTO semantic_work_state(singleton, definition_version, phase)
VALUES (TRUE, 1, 'uninitialized');
CREATE VIEW semantic_pending AS
SELECT w.*, ss.file_mtime,
       CASE WHEN NOT w.raw_available THEN 'missing_message_state'
            WHEN s.id IS NULL THEN 'missing_session'
            WHEN ss.session_id IS NULL THEN 'missing_session_state'
            WHEN ss.file_mtime IS NULL THEN 'missing_mtime'
            ELSE NULL END AS blocked_reason,
       w.raw_available AND s.id IS NOT NULL AND ss.session_id IS NOT NULL
           AND ss.file_mtime IS NOT NULL AS selectable
FROM semantic_work w
LEFT JOIN sessions s ON s.id = w.session_id
LEFT JOIN session_state ss ON ss.session_id = w.session_id;
