"""Unit tests for the CodexCliBackend.

We avoid invoking real `codex` here. Instead we generate a small Python script
on disk per test that mimics whatever codex behaviour the test needs: parsing
argv for the model and `-o` tempfile, optionally consuming stdin, and writing
the expected JSONL events to stdout plus the final message to the temp file.

This trades a bit of fixture verbosity for full coverage of error paths,
argv shape, and usage parsing without needing the real CLI installed.
"""

from __future__ import annotations

import json
import os
import re
import signal
import stat
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest
from recall.core.config import ContextConfig
from recall.core.models import Message, Session, ToolCall
from recall.core.types import Role, Source
from recall.services.context_backends import ContextResult, codex_cli
from recall.services.context_backends.codex_cli import _DOCUMENT_CACHE_MAX, CodexCliBackend

# --- fixture helpers -------------------------------------------------------


def _session(
    *,
    session_id: str = "session-1",
    content: str = "This is a long enough message body for context.",
) -> Session:
    session = Session(
        id=session_id,
        source=Source.CODEX,
        source_path="/tmp/session.jsonl",
        file_mtime=1.0,
        file_size=1,
    )
    session.messages.extend(
        [
            Message(
                id="m0",
                session_id=session.id,
                idx=0,
                role=Role.USER,
                content="previous message with useful context",
            ),
            Message(
                id="m1",
                session_id=session.id,
                idx=1,
                role=Role.ASSISTANT,
                content=content,
            ),
        ]
    )
    return session


def _prompt_document(prompt: str) -> str:
    return prompt.split("<document>", 1)[1].split("</document>", 1)[0]


def _message_line(message: Message) -> str:
    content = message.content or message.thinking or ""
    return f"{message.idx}:{message.role.value}: {content}"


def _long_session(
    *,
    session_id: str = "long-session",
    source: Source = Source.CODEX,
    message_count: int = 200,
    message_chars: int = 3000,
) -> Session:
    session = Session(
        id=session_id,
        source=source,
        source_path="/tmp/session.jsonl",
        file_mtime=1.0,
        file_size=1,
    )
    session.messages.extend(
        Message(
            id=f"m{idx}",
            session_id=session.id,
            idx=idx,
            role=Role.USER if idx % 2 == 0 else Role.ASSISTANT,
            content=f"message-{idx:03d} " + ("x" * message_chars),
        )
        for idx in range(message_count)
    )
    return session


def _write_codex_stub(tmp_path: Path, *, body: str) -> tuple[Path, Path]:
    """Generate a stub `codex` launcher + a Python script implementing `body`.

    Returns (launcher_path, log_path). The stub writes its captured argv +
    stdin to `log_path` so the test can assert on the exact invocation. Each
    test gets its own `tmp_path` so logs do not leak between tests.
    """
    log_path = tmp_path / "invocation.json"
    script = tmp_path / "codex_stub.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json, pathlib\n"
        f"_log_path = pathlib.Path({os.fsdecode(log_path)!r})\n" + body
    )
    launcher = tmp_path / "codex"
    launcher.write_text(f'#!/bin/sh\nexec {os.fsdecode(script)} "$@"\n')
    for path in (launcher, script):
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return launcher, log_path


def _last_message_path(argv: list[str]) -> Path:
    return Path(argv[argv.index("-o") + 1])


class _SuccessfulPopen:
    pid = 12345
    stdin = None
    stdout = None
    stderr = None

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv = argv
        self.kwargs = kwargs
        self.returncode: int | None = 0

    def communicate(
        self,
        input: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        _last_message_path(self.argv).write_text("short context\n", encoding="utf-8")
        stdout = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 3}})
        return stdout + "\n", ""

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


class _TimeoutPopen:
    pid = 23456
    stdin = None
    stdout = None
    stderr = None

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv = argv
        self.kwargs = kwargs
        self.returncode: int | None = None

    def communicate(
        self,
        input: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        # `TimeoutExpired.timeout` is annotated as non-Optional in the stdlib stubs;
        # surface a concrete value so the stub matches the real-world contract.
        raise subprocess.TimeoutExpired(cmd=self.argv, timeout=timeout or 0.0)

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


def _success_stub(
    tmp_path: Path,
    *,
    last_message: str,
    usage: dict[str, int],
) -> tuple[Path, Path]:
    body = textwrap.dedent(
        f"""\
        argv = sys.argv[1:]
        prompt = sys.stdin.read()
        out_path = None
        for i, arg in enumerate(argv):
            if arg == "-o" and i + 1 < len(argv):
                out_path = argv[i + 1]
                break
        pathlib.Path(out_path).write_text({last_message!r} + "\\n", encoding="utf-8")
        # Mirror real codex --json events.
        for event in (
            {{"type": "thread.started", "thread_id": "tid"}},
            {{"type": "turn.started"}},
            {{
                "type": "item.completed",
                "item": {{"id": "item_0", "type": "agent_message", "text": {last_message!r}}},
            }},
            {{"type": "turn.completed", "usage": {usage!r}}},
        ):
            sys.stdout.write(json.dumps(event) + "\\n")
        _log_path.write_text(json.dumps({{"argv": argv, "prompt": prompt}}))
        sys.exit(0)
        """
    )
    return _write_codex_stub(tmp_path, body=body)


def _batch_success_stub(tmp_path: Path) -> tuple[Path, Path]:
    body = textwrap.dedent(
        """\
        import re
        argv = sys.argv[1:]
        prompt = sys.stdin.read()
        out_path = None
        schema_path = None
        for i, arg in enumerate(argv):
            if arg == "-o" and i + 1 < len(argv):
                out_path = argv[i + 1]
            if arg == "--output-schema" and i + 1 < len(argv):
                schema_path = argv[i + 1]
        matches = re.findall(r'<chunk index="(\\d+)">(.*?)</chunk>', prompt, flags=re.S)
        response = [
            {"index": int(index), "context": "ctx:" + chunk.split()[0]}
            for index, chunk in reversed(matches)
        ]
        pathlib.Path(out_path).write_text(json.dumps(response), encoding="utf-8")
        sys.stdout.write(json.dumps({
            "type": "turn.completed",
            "usage": {
                "input_tokens": len(prompt),
                "cached_input_tokens": 0,
                "output_tokens": len(response),
                "reasoning_output_tokens": 0,
            },
        }) + "\\n")
        record = {
            "argv": argv,
            "prompt": prompt,
            "schema_exists": pathlib.Path(schema_path).exists() if schema_path else False,
        }
        existing = json.loads(_log_path.read_text()) if _log_path.exists() else []
        existing.append(record)
        _log_path.write_text(json.dumps(existing))
        sys.exit(0)
        """
    )
    return _write_codex_stub(tmp_path, body=body)


def _error_stub(tmp_path: Path, *, code: int, message: str) -> tuple[Path, Path]:
    body = textwrap.dedent(
        f"""\
        argv = sys.argv[1:]
        sys.stdin.read()
        for event in (
            {{"type": "thread.started", "thread_id": "tid"}},
            {{"type": "turn.started"}},
            {{"type": "error", "message": {message!r}}},
            {{"type": "turn.failed", "error": {{"message": {message!r}}}}},
        ):
            sys.stdout.write(json.dumps(event) + "\\n")
        _log_path.write_text(json.dumps({{"argv": argv}}))
        sys.exit({code})
        """
    )
    return _write_codex_stub(tmp_path, body=body)


# --- tests -----------------------------------------------------------------


def test_min_chars_skip_does_not_invoke_codex(tmp_path: Path) -> None:
    executable, log = _success_stub(tmp_path, last_message="ignored", usage={})
    session = _session(content="tiny")
    backend = CodexCliBackend(
        ContextConfig(mode="llm-codex", min_chars=50, executable=str(executable))
    )

    result = backend.generate_prefix(session, session.messages[1])

    assert result == ContextResult(prefix="", mode="off")
    assert not log.exists(), "codex should not be invoked when chunk is below min_chars"


def test_document_cache_evicts_oldest_when_over_capacity() -> None:
    backend = CodexCliBackend(ContextConfig(mode="llm-codex"))

    for index in range(_DOCUMENT_CACHE_MAX * 2):
        session = _session(
            session_id=f"session-{index}",
            content=f"This is a long enough message body for context {index}.",
        )
        backend._build_prompt(session, session.messages[1])

    assert len(backend._document_cache) == _DOCUMENT_CACHE_MAX
    assert list(backend._document_cache) == [
        f"session-{index}" for index in range(_DOCUMENT_CACHE_MAX, _DOCUMENT_CACHE_MAX * 2)
    ]


def test_document_cache_refreshes_lru_on_repeat_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex_cli, "_DOCUMENT_CACHE_MAX", 3)
    backend = CodexCliBackend(ContextConfig(mode="llm-codex"))
    sessions = {
        session_id: _session(
            session_id=session_id,
            content=f"This is a long enough message body for context {session_id}.",
        )
        for session_id in ("A", "B", "C", "D")
    }

    for session_id in ("A", "B", "C"):
        backend._build_prompt(sessions[session_id], sessions[session_id].messages[1])
    backend._build_prompt(sessions["A"], sessions["A"].messages[1])
    backend._build_prompt(sessions["D"], sessions["D"].messages[1])

    assert list(backend._document_cache) == ["C", "A", "D"]
    assert "B" not in backend._document_cache
    assert "A" in backend._document_cache


def test_context_config_default_max_document_chars_is_400000() -> None:
    assert ContextConfig().max_document_chars == 400000


def test_truncation_short_circuits_under_cap() -> None:
    session = _session()
    backend = CodexCliBackend(ContextConfig(mode="llm-codex", max_document_chars=400000))
    expected = "\n".join(_message_line(message) for message in session.messages)

    prompt = backend._build_prompt(session, session.messages[1])
    document = _prompt_document(prompt)

    assert "[... " not in document
    assert document == expected


def test_smart_truncation_over_cap_includes_preamble_and_vicinity() -> None:
    session = _long_session()
    chunk = session.messages[100]
    backend = CodexCliBackend(ContextConfig(mode="llm-codex", max_document_chars=400000))

    prompt = backend._build_prompt(session, chunk)
    document = _prompt_document(prompt)
    retained_lines = [line for line in document.splitlines() if not line.startswith("[... ")]

    assert document.startswith("0:user: message-000 ")
    assert "[... " in document
    assert " chars / " in document
    assert " messages omitted ...]" in document
    for idx in range(50, 151):
        assert f"{idx}:" in document
        assert f"message-{idx:03d}" in document
    assert len(document) <= 400000 + 200
    assert all(re.match(r"^\d+:", line) for line in retained_lines)


def test_build_prompt_truncates_document_when_max_chars_set() -> None:
    session = _session(content=("older context " * 20) + "TAIL-OF-RENDERED-DOCUMENT")
    backend = CodexCliBackend(ContextConfig(mode="llm-codex", max_document_chars=100))

    prompt = backend._build_prompt(session, session.messages[1])
    document = _prompt_document(prompt)

    assert "[... " in document
    assert " chars / " in document
    assert " messages omitted ...]" in document
    assert len(document) <= 100 + 200
    # Cache stores the rendered-message list, not the truncated document;
    # smart truncation depends on which chunk is being summarized so the
    # final document is recomputed per call (see review verdict on
    # bucket-based caching for the regression we caught here).
    cached = backend._document_cache[session.id]
    assert isinstance(cached, list)
    assert len(cached) == len(session.messages)


def test_build_prompt_does_not_truncate_when_max_chars_unset() -> None:
    content = ("older context " * 20) + "TAIL-OF-RENDERED-DOCUMENT"
    session = _session(content=content)
    backend = CodexCliBackend(ContextConfig(mode="llm-codex", max_document_chars=None))

    prompt = backend._build_prompt(session, session.messages[1])

    assert "...[truncated" not in prompt
    assert content in prompt


def test_render_message_with_claude_code_tools() -> None:
    session = _session(session_id="claude-tools")
    session.source = Source.CLAUDE_CODE
    unknown_payload = "find symbol references " + ("z" * 160)
    session.messages[0].tool_calls.extend(
        [
            ToolCall(
                id="tc-bash",
                session_id=session.id,
                message_id=session.messages[0].id,
                idx=0,
                tool_name="Bash",
                tool_input={"result": "COMMAND_OUTPUT_SECRET"},
                bash_command="uv run pytest -q",
            ),
            ToolCall(
                id="tc-edit",
                session_id=session.id,
                message_id=session.messages[0].id,
                idx=1,
                tool_name="Edit",
                tool_input={
                    "file_path": "/repo/packages/recall/src/recall/core/config.py",
                    "tool_result": "EDIT_RESULT_SECRET",
                },
            ),
            ToolCall(
                id="tc-skill",
                session_id=session.id,
                message_id=session.messages[0].id,
                idx=2,
                tool_name="Skill",
                tool_input={"name": "ignored"},
                skill_name="python-best-practices",
            ),
            ToolCall(
                id="tc-unknown",
                session_id=session.id,
                message_id=session.messages[0].id,
                idx=3,
                tool_name="Unknown",
                tool_input={"payload": unknown_payload},
            ),
        ]
    )
    backend = CodexCliBackend(ContextConfig(mode="llm-codex", max_document_chars=400000))

    document = _prompt_document(backend._build_prompt(session, session.messages[1]))

    assert ":tool: Bash uv run pytest -q" in document
    assert ":tool: Edit /repo/packages/recall/src/recall/core/config.py" in document
    assert ":tool: Skill python-best-practices" in document
    assert unknown_payload[:100] in document
    assert "COMMAND_OUTPUT_SECRET" not in document
    assert "EDIT_RESULT_SECRET" not in document


def test_render_message_skips_tools_for_codex_source() -> None:
    session = _session(session_id="codex-tools")
    session.messages[0].tool_calls.append(
        ToolCall(
            id="tc-bash",
            session_id=session.id,
            message_id=session.messages[0].id,
            idx=0,
            tool_name="Bash",
            bash_command="uv run pytest -q",
        )
    )
    backend = CodexCliBackend(ContextConfig(mode="llm-codex", max_document_chars=400000))

    document = _prompt_document(backend._build_prompt(session, session.messages[1]))

    assert ":tool: " not in document
    assert "uv run pytest -q" not in document


def test_document_cache_amortizes_rendering_per_session(tmp_path: Path) -> None:
    """The cache stores the rendered message list per session.

    Smart truncation slides with the chunk being summarized, so the truncated
    document varies per call — but the expensive work (rendering each message,
    parsing tool_input JSON) only has to happen once per session. Three calls
    against the same session must produce one cache entry whose value is the
    list of rendered message strings, and each call must still produce a
    chunk-appropriate truncation that contains the message being summarized.
    """
    executable, _log = _success_stub(tmp_path, last_message="ok", usage={})
    session = _long_session(message_count=160, message_chars=80)
    # Pick max_document_chars large enough to hold the full ±50 vicinity
    # (101 messages * ~100 chars/render) plus the preamble and markers, so
    # the trim-vicinity-until-fits loop in _smart_truncate does not shrink
    # the window. That isolates the cache contract from the trim contract.
    backend = CodexCliBackend(
        ContextConfig(
            mode="llm-codex",
            min_chars=5,
            max_document_chars=15000,
            executable=str(executable),
        )
    )

    backend.generate_prefix(session, session.messages[0])
    backend.generate_prefix(session, session.messages[50])
    backend.generate_prefix(session, session.messages[150])

    assert list(backend._document_cache.keys()) == [session.id]
    cached = backend._document_cache[session.id]
    assert isinstance(cached, list)
    assert len(cached) == len(session.messages)

    # Each chunk gets its own truncated document with its own vicinity. The
    # preamble (first 5 msgs by default) is preserved in every document, so
    # message-000 legitimately appears in both. The regression we are guarding
    # against — the one the structured review caught with bucket-based
    # caching — is that two distant chunks would have received the SAME
    # document. With per-session caching of the rendered list and per-call
    # truncation, that cannot happen: each call must produce a different
    # document and each must include its own chunk inside the vicinity.
    doc_0 = _prompt_document(backend._build_prompt(session, session.messages[0]))
    doc_150 = _prompt_document(backend._build_prompt(session, session.messages[150]))
    assert doc_0 != doc_150
    assert "message-000" in doc_0
    assert "message-150" in doc_150
    # The distant chunk's vicinity must not bleed into the other call:
    # message-150 is far past message-0's window (idx 0 ± 50 = [0, 50]).
    assert "message-150" not in doc_0
    # message-159 is the last message; it sits in message-150's vicinity
    # (idx 150 ± 50 = [100, 160]) but is well outside message-0's window.
    assert "message-159" in doc_150
    assert "message-159" not in doc_0


def test_successful_generation_populates_usage(tmp_path: Path) -> None:
    executable, _log = _success_stub(
        tmp_path,
        last_message="short context",
        usage={
            "input_tokens": 100,
            "cached_input_tokens": 50,
            "output_tokens": 7,
            "reasoning_output_tokens": 3,
        },
    )
    session = _session()
    backend = CodexCliBackend(
        ContextConfig(
            mode="llm-codex",
            model="gpt-5.3-codex-spark",
            min_chars=5,
            executable=str(executable),
        )
    )

    result = backend.generate_prefix(session, session.messages[1])

    assert result.prefix == "[short context] "
    assert result.mode == "llm-codex"
    assert result.model == "gpt-5.3-codex-spark"
    # input_tokens = input_tokens + cached_input_tokens
    assert result.input_tokens == 150
    # output_tokens = output_tokens + reasoning_output_tokens
    assert result.output_tokens == 10


def test_batch_generation_uses_schema_and_maps_results_by_index(tmp_path: Path) -> None:
    executable, log = _batch_success_stub(tmp_path)
    session = _session(content="chunk-1 body long enough")
    session.messages.extend(
        Message(
            id=f"m{idx}",
            session_id=session.id,
            idx=idx,
            role=Role.USER,
            content=f"chunk-{idx} body long enough",
        )
        for idx in range(2, 6)
    )
    backend = CodexCliBackend(
        ContextConfig(
            mode="llm-codex",
            batch_size=2,
            min_chars=1,
            max_document_chars=400000,
            executable=str(executable),
        )
    )

    batches = backend.plan_batches(session, session.messages)
    results = [result for batch in batches for result in backend.generate_prefixes(session, batch)]

    invocations = json.loads(log.read_text())
    assert [len(batch) for batch in batches] == [2, 2, 2]
    assert len(invocations) == 3
    for invocation in invocations:
        argv = invocation["argv"]
        assert "--output-schema" in argv
        assert invocation["schema_exists"] is True
        assert invocation["prompt"].count("<document>") == 1
        assert invocation["prompt"].count("</document>") == 1
    assert [result.prefix for result in results] == [
        "[ctx:previous] ",
        "[ctx:chunk-1] ",
        "[ctx:chunk-2] ",
        "[ctx:chunk-3] ",
        "[ctx:chunk-4] ",
        "[ctx:chunk-5] ",
    ]
    serial_input_tokens = sum(
        len(backend._build_prompt(session, message)) for message in session.messages
    )
    batched_input_tokens = sum(result.input_tokens for result in results)
    assert batched_input_tokens < serial_input_tokens


def test_batch_generation_counts_usage_once_per_batch(tmp_path: Path) -> None:
    executable, _log = _success_stub(
        tmp_path,
        last_message=json.dumps(
            [
                {"index": 1, "context": "second context"},
                {"index": 0, "context": "first context"},
            ]
        ),
        usage={
            "input_tokens": 100,
            "cached_input_tokens": 50,
            "output_tokens": 7,
            "reasoning_output_tokens": 3,
        },
    )
    session = _session(content="first body long enough")
    session.messages.append(
        Message(
            id="m2",
            session_id=session.id,
            idx=2,
            role=Role.USER,
            content="second body long enough",
        )
    )
    backend = CodexCliBackend(
        ContextConfig(mode="llm-codex", min_chars=1, executable=str(executable))
    )

    results = backend.generate_prefixes(session, session.messages[1:])

    assert [result.prefix for result in results] == ["[first context] ", "[second context] "]
    assert [result.input_tokens for result in results] == [150, 0]
    assert [result.output_tokens for result in results] == [10, 0]


def test_batch_of_one_uses_single_prompt_without_schema(tmp_path: Path) -> None:
    executable, log = _success_stub(tmp_path, last_message="single context", usage={})
    session = _session()
    backend = CodexCliBackend(
        ContextConfig(mode="llm-codex", min_chars=1, executable=str(executable))
    )
    expected_prompt = backend._build_prompt(session, session.messages[1])

    result = backend.generate_prefixes(session, [session.messages[1]])[0]

    invocation = json.loads(log.read_text())
    assert invocation["prompt"] == expected_prompt
    assert "--output-schema" not in invocation["argv"]
    assert result.prefix == "[single context] "


def test_run_codex_uses_popen_with_start_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen_calls: list[dict[str, Any]] = []

    def fake_popen(argv: list[str], **kwargs: Any) -> _SuccessfulPopen:
        proc = _SuccessfulPopen(argv, **kwargs)
        popen_calls.append(kwargs)
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    backend = CodexCliBackend(ContextConfig(mode="llm-codex"))
    backend._executable_path = "/usr/bin/codex"

    text, usage = backend._run_codex("small prompt")

    assert text == "short context\n"
    assert usage["input_tokens"] == 3
    assert popen_calls[0]["start_new_session"] is True
    assert popen_calls[0]["stdin"] is subprocess.PIPE
    assert popen_calls[0]["stdout"] is subprocess.PIPE
    assert popen_calls[0]["stderr"] is subprocess.PIPE


def test_run_codex_timeout_killpg_path(monkeypatch: pytest.MonkeyPatch) -> None:
    killpg_calls: list[tuple[int, signal.Signals]] = []

    def fake_popen(argv: list[str], **kwargs: Any) -> _TimeoutPopen:
        return _TimeoutPopen(argv, **kwargs)

    def fake_killpg(pid: int, sig: signal.Signals) -> None:
        killpg_calls.append((pid, sig))

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(os, "killpg", fake_killpg)
    backend = CodexCliBackend(ContextConfig(mode="llm-codex"))
    backend._executable_path = "/usr/bin/codex"

    with pytest.raises(RuntimeError, match="timed out"):
        backend._run_codex("small prompt")

    assert killpg_calls == [
        (_TimeoutPopen.pid, signal.SIGTERM),
        (_TimeoutPopen.pid, signal.SIGKILL),
    ]


def test_active_processes_registry_is_cleaned_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_popen(argv: list[str], **kwargs: Any) -> _SuccessfulPopen:
        return _SuccessfulPopen(argv, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    backend = CodexCliBackend(ContextConfig(mode="llm-codex"))
    backend._executable_path = "/usr/bin/codex"

    backend._run_codex("small prompt")

    assert CodexCliBackend._active_processes == set()


def test_terminate_active_codex_processes_sends_sigterm_and_returns_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.returncode: int | None = None

        def wait(self, timeout: float | None = None) -> int:
            self.returncode = 0
            return 0

    procs = {FakeProcess(101), FakeProcess(202)}
    killpg_calls: list[tuple[int, signal.Signals]] = []

    def fake_killpg(pid: int, sig: signal.Signals) -> None:
        killpg_calls.append((pid, sig))

    monkeypatch.setattr(os, "killpg", fake_killpg)
    # FakeProcess is structurally compatible with subprocess.Popen for the .pid + .wait()
    # surface terminate_active_codex_processes() touches, but the type checker can't see
    # through the duck typing — cast to Any to seed the registry without weakening
    # production code's type discipline.
    fake_procs: Any = procs
    with CodexCliBackend._active_processes_lock:
        CodexCliBackend._active_processes.clear()
        CodexCliBackend._active_processes.update(fake_procs)
    try:
        terminated = codex_cli.terminate_active_codex_processes()
    finally:
        with CodexCliBackend._active_processes_lock:
            CodexCliBackend._active_processes.clear()

    assert terminated == 2
    assert sorted(killpg_calls) == [(101, signal.SIGTERM), (202, signal.SIGTERM)]


def test_argv_includes_required_flags_and_model(tmp_path: Path) -> None:
    executable, log = _success_stub(tmp_path, last_message="ok", usage={})
    session = _session()
    backend = CodexCliBackend(
        ContextConfig(
            mode="llm-codex",
            model="gpt-5.3-codex-spark",
            min_chars=5,
            executable=str(executable),
        )
    )

    backend.generate_prefix(session, session.messages[1])

    invocation = json.loads(log.read_text())
    argv = invocation["argv"]
    assert argv[0] == "exec"
    for required in (
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
    ):
        assert required in argv, f"missing required flag {required}"
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert 'web_search="disabled"' in argv
    # Every entry in _DISABLED_CODEX_FEATURES must appear as a paired
    # `--disable <feature>` so codex skips the plugin/tool-catalog work
    # recall never consumes; the existence of the pair is the contract.
    for feature in codex_cli._DISABLED_CODEX_FEATURES:
        for idx, token in enumerate(argv):
            if token == "--disable" and idx + 1 < len(argv) and argv[idx + 1] == feature:
                break
        else:
            raise AssertionError(f"missing --disable pair for feature {feature!r}")
    # `plugins` is the load-bearing one — the regression we're guarding
    # against is the github.com/openai/plugins.git fetch on every call.
    assert "plugins" in {
        argv[idx + 1] for idx, t in enumerate(argv) if t == "--disable" and idx + 1 < len(argv)
    }
    # reasoning_effort was not configured, so it must not appear.
    assert not any("model_reasoning_effort" in arg for arg in argv)
    assert argv[argv.index("--model") + 1] == "gpt-5.3-codex-spark"
    # Stdin sentinel as the last arg.
    assert argv[-1] == "-"
    # Prompt body should include the document marker.
    assert "<document>" in invocation["prompt"]
    assert "<chunk>" in invocation["prompt"]


def test_reasoning_effort_is_forwarded_when_configured(tmp_path: Path) -> None:
    executable, log = _success_stub(tmp_path, last_message="ok", usage={})
    session = _session()
    backend = CodexCliBackend(
        ContextConfig(
            mode="llm-codex",
            model="gpt-5.3-codex",
            min_chars=5,
            executable=str(executable),
            reasoning_effort="low",
        )
    )

    backend.generate_prefix(session, session.messages[1])

    argv = json.loads(log.read_text())["argv"]
    assert 'model_reasoning_effort="low"' in argv


def test_ultra_reasoning_retains_multi_agent_feature(tmp_path: Path) -> None:
    executable, log = _success_stub(tmp_path, last_message="ok", usage={})
    session = _session()
    backend = CodexCliBackend(
        ContextConfig(
            mode="llm-codex",
            model="gpt-5.6-sol",
            min_chars=5,
            executable=str(executable),
            reasoning_effort="ultra",
        )
    )

    backend.generate_prefix(session, session.messages[1])

    argv = json.loads(log.read_text())["argv"]
    disabled_features = {
        argv[idx + 1]
        for idx, token in enumerate(argv)
        if token == "--disable" and idx + 1 < len(argv)
    }
    assert "multi_agent" not in disabled_features
    assert 'model_reasoning_effort="ultra"' in argv


def test_failed_invocation_raises_with_stdout_error_detail(tmp_path: Path) -> None:
    executable, _log = _error_stub(
        tmp_path,
        code=1,
        message="invalid_request_error: 'minimal' not supported with Spark",
    )
    session = _session()
    backend = CodexCliBackend(
        ContextConfig(
            mode="llm-codex",
            model="gpt-5.3-codex-spark",
            min_chars=5,
            executable=str(executable),
        )
    )

    with pytest.raises(RuntimeError) as err:
        backend.generate_prefix(session, session.messages[1])

    msg = str(err.value)
    assert "exited 1" in msg
    assert "not supported with Spark" in msg


def test_strips_think_block_from_final_message(tmp_path: Path) -> None:
    executable, _log = _success_stub(
        tmp_path,
        last_message="<think>internal reasoning</think>the actual answer",
        usage={},
    )
    session = _session()
    backend = CodexCliBackend(
        ContextConfig(mode="llm-codex", min_chars=5, executable=str(executable))
    )

    result = backend.generate_prefix(session, session.messages[1])

    assert result.prefix == "[the actual answer] "


def test_is_available_false_when_executable_missing(tmp_path: Path) -> None:
    backend = CodexCliBackend(
        ContextConfig(mode="llm-codex", executable=str(tmp_path / "definitely-not-here"))
    )
    assert backend.is_available() is False


def test_default_context_model_substitutes_to_codex_default(tmp_path: Path) -> None:
    """Opting into llm-codex without overriding model must NOT pass the MLX
    default name to `codex exec --model`. The backend mirrors the llm-remote
    substitution and rewrites the shared default to the Codex-specific default
    (gpt-5.4-mini at the time of this commit). Compared empirically against
    gpt-5.3-codex-spark, mini produces ~3x more concise summaries with ~42%
    fewer input tokens at the same latency, making it the better default for
    chunk-prefix generation.
    """
    from recall.core.config import DEFAULT_CODEX_CONTEXT_MODEL

    executable, log = _success_stub(tmp_path, last_message="ok", usage={})
    session = _session()
    # ContextConfig() uses the shared DEFAULT_CONTEXT_MODEL (MLX). The backend
    # must NOT forward that string to codex; it must swap in the Codex default.
    backend = CodexCliBackend(
        ContextConfig(mode="llm-codex", min_chars=5, executable=str(executable))
    )

    result = backend.generate_prefix(session, session.messages[1])

    argv = json.loads(log.read_text())["argv"]
    assert argv[argv.index("--model") + 1] == DEFAULT_CODEX_CONTEXT_MODEL
    # ContextResult.model must also report the substituted value, not the MLX default.
    assert result.model == DEFAULT_CODEX_CONTEXT_MODEL
    # Sanity check on the chosen default itself, so a careless future swap to a
    # non-existent or experimental model doesn't slip through tests.
    assert DEFAULT_CODEX_CONTEXT_MODEL == "gpt-5.4-mini"
