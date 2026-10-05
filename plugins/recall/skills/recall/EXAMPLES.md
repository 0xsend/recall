# Examples

Common usage patterns and workflows for recall.

## Finding Past Work

### Search for a Topic

Find sessions where you worked on authentication:

```bash
recall search "authentication"
recall search "oauth login"
recall search "JWT token"
```

### Search Within Bash Commands

Find bash commands related to git:

```bash
recall search "git" --tool Bash
recall search "rebase" --tool Bash
```

The `--tool` flag filters to Bash tool calls only, searching within bash command text.

### Filter by Source

Search only Claude Code sessions (input accepts `claude-code` or `claude_code`):

```bash
recall search "refactor" --source claude-code
```

Search only Codex sessions:

```bash
recall search "fix bug" --source codex
```

Note: Output displays `claude_code` (underscore) in results.

## Browsing Sessions

### Recent Activity

See what you worked on today:

```bash
recall list --since 24h
```

This week:

```bash
recall list --since 7d
```

Since a specific date:

```bash
recall list --since 2024-01-01
```

### By Project

Find sessions in a specific repository:

```bash
recall list --project /path/to/my-app
recall list --project ~/code/api-server
```

Combine with time filter:

```bash
recall list --project /path/to/repo --since 7d
```

## Session Deep Dive

### View Session Details

Get full conversation with tool calls:

```bash
recall show abc123def456 --tools
```

Include thinking blocks (Claude Code):

```bash
recall show abc123def456 --tools --thinking
```

### Export for Analysis

Get JSON for processing:

```bash
recall show abc123def456 --json > session.json
```

Extract specific fields with jq:

```bash
recall show abc123 --json | jq -r '.messages[] | select(.role == "assistant" and (.content|type) == "string") | .content'
```

## Analytics

### Tool Usage Patterns

See which tools you use most:

```bash
recall stats tools
```

Output:
```
Bash: 456
Read: 312
Edit: 234
Glob: 156
Grep: 89
Write: 45
```

### Bash Command Analysis

Break down bash commands by type:

```bash
recall stats bash
```

Output:
```
git status: 45
npm test: 32
git diff: 28
uv run: 22
```

### Permission Suggestions

```bash
recall stats bash --suggest --json
```

```bash
recall stats bash --suggest --json | jq -r '.suggestions[] | select(.confidence == "high") | .pattern'
```

The payload is `{suggestions, skipped}`; each suggestion has `pattern`, `count`,
`confidence` (`high`, `medium`, `review`), and `reason`. Turn the `high` patterns
you trust into your harness's allow rules; `review` rows need a human decision,
and `skipped` lists destructive commands that are never suggested.

### Token Usage

See token consumption by project:

```bash
recall stats tokens
```

Output:
```
/path/to/project-a: 125000 in / 45000 out
/path/to/project-b: 89000 in / 32000 out
```

Contextual retrieval setup lives in the `recall-setup` skill. Indexing,
reindexing, and repair are daemon operations; see
[CLI_REFERENCE.md](CLI_REFERENCE.md#recall-daemon-status-and-maintenance).

## JSON Output for Scripts

All commands support `--json` for scripting:

```bash
# Get session IDs from last week
recall list --since 7d --json | jq -r '.[].id'

# Find most-used tool
recall stats tools --json | jq -r 'max_by(.count).tool_name'

# Count sessions by source
recall list --json | jq 'group_by(.source) | map({source: .[0].source, count: length})'
```

## Workflow: Review Past Implementation

1. Search for the topic:
   ```bash
   recall search "rate limiting"
   ```

2. List sessions in that project:
   ```bash
   recall list --project /path/to/api --since 30d
   ```

3. Deep dive into relevant session:
   ```bash
   recall show abc123 --tools
   ```

## Workflow: Optimize Permissions

1. Get permission suggestions:
   ```bash
   recall stats bash --suggest --json
   ```

2. Review the `high` entries under `.suggestions` and add the trusted patterns to your harness's
   permission config. Re-run later to catch new patterns.
