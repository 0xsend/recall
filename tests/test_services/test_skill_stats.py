from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
from recall.core.types import Source
from recall.db.schema import ensure_schema
from recall.services.analytics import skill_usage


def _conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "recall.duckdb"))
    ensure_schema(conn)
    return conn


def _seed_session(
    conn: duckdb.DuckDBPyConnection,
    *,
    session_id: str,
    source: Source,
    cwd: str,
    ended_at: datetime,
    host: str = "control",
) -> None:
    conn.execute(
        "INSERT INTO sessions (id, source, source_path, source_session_id) VALUES (?, ?, ?, ?)",
        [session_id, source.value, f"/sessions/{session_id}", session_id],
    )
    conn.execute(
        """
        INSERT INTO session_state (
            session_id, cwd, host, ended_at, file_mtime, file_size, indexed_at
        ) VALUES (?, ?, ?, ?, 1.0, 1, ?)
        """,
        [session_id, cwd, host, ended_at, ended_at],
    )


def _seed_call(
    conn: duckdb.DuckDBPyConnection,
    *,
    call_id: str,
    session_id: str,
    tool_name: str,
    tool_input: dict[str, object] | None = None,
    bash_command: str | None = None,
    skill_name: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO tool_calls (
            id, session_id, message_id, idx, tool_name, tool_input,
            bash_command, skill_name
        ) VALUES (?, ?, NULL, 0, ?, ?, ?, ?)
        """,
        [
            call_id,
            session_id,
            tool_name,
            json.dumps(tool_input) if tool_input is not None else None,
            bash_command,
            skill_name,
        ],
    )


def test_skill_usage_attributes_all_harnesses_and_historical_rows(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    now = datetime.now(UTC)
    try:
        for session_id, source, cwd in [
            ("claude", Source.CLAUDE_CODE, "/work/claude"),
            ("codex", Source.CODEX, "/work/codex"),
            ("pi", Source.PI_AGENT, "/work/pi"),
            ("grok", Source.GROK, "/work/product"),
            ("kimi", Source.KIMI_CODE, "/work/kimi"),
        ]:
            _seed_session(conn, session_id=session_id, source=source, cwd=cwd, ended_at=now)

        _seed_call(
            conn,
            call_id="claude-skill",
            session_id="claude",
            tool_name="Skill",
            tool_input={"skill": "agent-workflows:afk"},
            skill_name="agent-workflows:afk",
        )
        codex_command = (
            "sed -n '1,240p' "
            "/Users/dev/.codex/plugins/cache/agent-profile/engineering-practices/3.0.0/"
            "skills/code-law/SKILL.md"
        )
        for suffix in ("one", "two"):
            _seed_call(
                conn,
                call_id=f"codex-{suffix}",
                session_id="codex",
                tool_name="exec_command",
                tool_input={"cmd": codex_command},
                bash_command=codex_command,
            )
        _seed_call(
            conn,
            call_id="pi-skill",
            session_id="pi",
            tool_name="read",
            tool_input={
                "path": (
                    "/home/dev/.pi/agent/git/github.com/example/agent-workflows/"
                    "skills/writing-plans/SKILL.md"
                )
            },
        )
        _seed_call(
            conn,
            call_id="grok-skill",
            session_id="grok",
            tool_name="read",
            tool_input={
                "path": (
                    "/opt/agent-profile/plugins/engineering-practices/"
                    "skills/testing-best-practices/SKILL.md"
                )
            },
        )
        _seed_call(
            conn,
            call_id="kimi-skill",
            session_id="kimi",
            tool_name="Skill",
            tool_input={"skill": "agent-workflows:afk"},
        )

        # A read of the session's own checkout is a load too.
        _seed_call(
            conn,
            call_id="grok-checkout",
            session_id="grok",
            tool_name="read",
            tool_input={
                "path": (
                    "/work/product/agent-profile/plugins/engineering-practices/"
                    "skills/code-law/SKILL.md"
                )
            },
        )
        # These remain candidates for coverage diagnostics but never invocations.
        _seed_call(
            conn,
            call_id="grok-discovery",
            session_id="grok",
            tool_name="bash",
            tool_input={"command": "rg SKILL.md /opt/agent-profile/plugins"},
            bash_command="rg SKILL.md /opt/agent-profile/plugins",
        )
        quoted = (
            "printf 'sed -n 1,20p "
            "/opt/agent-profile/plugins/engineering-practices/skills/code-law/SKILL.md'"
        )
        _seed_call(
            conn,
            call_id="grok-quoted",
            session_id="grok",
            tool_name="bash",
            tool_input={"command": quoted},
            bash_command=quoted,
        )

        result = skill_usage(conn=conn, local_host="control")

        assert [
            (row.skill_name, row.source, row.host, row.invocations, row.sessions)
            for row in result.rows
        ] == [
            ("agent-workflows:afk", "claude_code", "control", 1, 1),
            ("agent-workflows:afk", "kimi_code", "control", 1, 1),
            ("agent-workflows:writing-plans", "pi_agent", "control", 1, 1),
            ("engineering-practices:code-law", "codex", "control", 2, 1),
            ("engineering-practices:code-law", "grok", "control", 1, 1),
            (
                "engineering-practices:testing-best-practices",
                "grok",
                "control",
                1,
                1,
            ),
        ]
        coverage = result.coverage
        assert coverage.scope == "local"
        assert coverage.expected_hosts == ("control",)
        assert coverage.successful_hosts == ("control",)
        assert coverage.covered_sources == tuple(source.value for source in Source)
        assert coverage.considered_sessions == 5
        assert coverage.attributed_invocations == 7
        assert coverage.unattributed_candidates == 2
        assert coverage.control.considered_sessions == 5
        assert coverage.control.attributed_invocations == 7
        assert coverage.control.unattributed_candidates == 2
    finally:
        conn.close()


def test_skill_usage_source_and_time_filter_keep_all_time_control(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        _seed_session(
            conn,
            session_id="old-codex",
            source=Source.CODEX,
            cwd="/work/codex",
            ended_at=datetime.now(UTC) - timedelta(days=40),
        )
        command = "cat /Users/dev/.codex/skills/.system/imagegen/SKILL.md"
        _seed_call(
            conn,
            call_id="old-call",
            session_id="old-codex",
            tool_name="exec_command",
            tool_input={"cmd": command},
            bash_command=command,
        )

        result = skill_usage(
            sources=(Source.CODEX,),
            since=datetime.now(UTC) - timedelta(days=7),
            conn=conn,
            local_host="control",
        )

        assert result.rows == ()
        assert result.coverage.covered_sources == ("codex",)
        assert result.coverage.considered_sessions == 0
        assert result.coverage.attributed_invocations == 0
        assert result.coverage.control.considered_sessions == 1
        assert result.coverage.control.attributed_invocations == 1
    finally:
        conn.close()


def test_skill_usage_labels_imported_sessions_with_the_local_endpoint(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        _seed_session(
            conn,
            session_id="imported",
            source=Source.CLAUDE_CODE,
            cwd="/work/imported",
            ended_at=datetime.now(UTC),
            host="imported-edge",
        )
        _seed_call(
            conn,
            call_id="imported-call",
            session_id="imported",
            tool_name="Skill",
            tool_input={"skill": "agent-workflows:afk"},
            skill_name="agent-workflows:afk",
        )

        result = skill_usage(conn=conn, local_host="control")

        assert len(result.rows) == 1
        assert result.rows[0].host == "control"
        assert result.coverage.expected_hosts == ("control",)
    finally:
        conn.close()


def test_skill_usage_counts_a_moved_transcript_once(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        ended_at = datetime.now(UTC)
        for session_id, project in (
            ("before", "-Users-al-old-repo"),
            ("after", "-Users-al-new-repo"),
        ):
            conn.execute(
                """
                INSERT INTO sessions (id, source, source_path, source_session_id)
                VALUES (?, ?, ?, ?)
                """,
                [
                    session_id,
                    Source.CLAUDE_CODE.value,
                    f"/Users/dev/.claude/projects/{project}/3f1c.jsonl",
                    "3f1c",
                ],
            )
            conn.execute(
                """
                INSERT INTO session_state (
                    session_id, cwd, host, ended_at, file_mtime, file_size, indexed_at
                ) VALUES (?, '/work', 'control', ?, 1.0, 1, ?)
                """,
                [session_id, ended_at, ended_at],
            )
            _seed_call(
                conn,
                call_id=f"{session_id}-call",
                session_id=session_id,
                tool_name="Skill",
                tool_input={"skill": "engineering-practices:code-law"},
            )

        result = skill_usage(conn=conn, local_host="control")

        assert [(row.skill_name, row.invocations, row.sessions) for row in result.rows] == [
            ("engineering-practices:code-law", 1, 1)
        ]
        assert result.coverage.considered_sessions == 1
    finally:
        conn.close()


def test_skill_usage_rederives_a_stale_stored_name(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        _seed_session(
            conn,
            session_id="codex",
            source=Source.CODEX,
            cwd="/work/codex",
            ended_at=datetime.now(UTC),
        )
        command = "cat /Users/dev/.codex/skills/{a,b}/SKILL.md"
        _seed_call(
            conn,
            call_id="brace-call",
            session_id="codex",
            tool_name="exec_command",
            tool_input={"cmd": command},
            bash_command=command,
            skill_name="{a,b}",
        )

        result = skill_usage(conn=conn, local_host="control")

        assert [row.skill_name for row in result.rows] == ["a", "b"]
        assert result.coverage.attributed_invocations == 2
    finally:
        conn.close()


def test_skill_usage_counts_a_code_mode_program_once(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        _seed_session(
            conn,
            session_id="codex",
            source=Source.CODEX,
            cwd="/work/codex",
            ended_at=datetime.now(UTC),
        )
        read = "cat /Users/dev/.codex/skills/foo/SKILL.md"
        program = f'await tools.exec_command({{cmd:"{read}"}});\n' + (
            "await tools.exec_command({cmd:`ls ${dir}`});\n" * 2
        )
        _seed_call(
            conn,
            call_id="literal",
            session_id="codex",
            tool_name="exec_command",
            tool_input={"cmd": read},
            bash_command=read,
        )
        for index in range(2):
            _seed_call(
                conn,
                call_id=f"runtime-{index}",
                session_id="codex",
                tool_name="exec_command",
                tool_input={"source": program},
            )

        result = skill_usage(conn=conn, local_host="control")

        assert [(row.skill_name, row.invocations) for row in result.rows] == [("foo", 1)]
        assert result.coverage.unattributed_candidates == 0
    finally:
        conn.close()


def test_skill_usage_counts_each_program_load_beside_plain_reads(tmp_path: Path) -> None:
    conn = _conn(tmp_path)
    try:
        _seed_session(
            conn,
            session_id="codex",
            source=Source.CODEX,
            cwd="/work/codex",
            ended_at=datetime.now(UTC),
        )
        root = "/Users/dev/.codex/skills"
        literal = f"cat {root}/foo/SKILL.md; cat {root}/bar/SKILL.md"
        first = (
            f'await tools.exec_command({{cmd:"{literal}"}});\n'
            "await tools.exec_command({cmd:`ls ${dir}`});\n"
        )
        second = (
            f'const cmds = ["cat {root}/foo/SKILL.md"];\n'
            "for (const cmd of cmds) await tools.exec_command({cmd});\n"
        )
        _seed_call(
            conn,
            call_id="literal",
            session_id="codex",
            tool_name="exec_command",
            tool_input={"cmd": literal},
            bash_command=literal,
        )
        _seed_call(
            conn,
            call_id="first-runtime",
            session_id="codex",
            tool_name="exec_command",
            tool_input={"source": first},
        )
        _seed_call(
            conn,
            call_id="second-runtime",
            session_id="codex",
            tool_name="exec_command",
            tool_input={"source": second},
        )

        result = skill_usage(conn=conn, local_host="control")

        assert [(row.skill_name, row.invocations) for row in result.rows] == [
            ("bar", 1),
            ("foo", 2),
        ]
    finally:
        conn.close()
