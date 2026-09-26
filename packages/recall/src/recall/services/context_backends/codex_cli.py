"""Codex CLI context generation backend.

Shells out to the local `codex exec` (OpenAI Codex CLI) for per-message context
summaries. This lets users drive contextual retrieval through any model accepted
by their local Codex CLI and reuse the CLI's existing authentication.

The backend treats `codex` as an opaque subprocess: it pipes the same prompt
template the other backends use into stdin, parses `--json` events on stdout for
token-usage telemetry, and reads the final agent message from a temp file the
CLI writes via `--output-last-message`. Everything else about how Codex CLI
selects an account, reaches the API, and decides to use bundled bubblewrap is
deliberately outside this module's concern.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from recall.core.config import (
    DEFAULT_CODEX_CONTEXT_MODEL,
    DEFAULT_CONTEXT_MODEL,
    ContextConfig,
)
from recall.core.models import Message, Session, ToolCall
from recall.core.types import Source
from recall.services.context_backends import ContextResult
from recall.services.context_backends._text import strip_think_blocks

logger = logging.getLogger("recall.context.codex_cli")

# Subprocess timeout floor. Even with --ignore-user-config the CLI does network
# I/O, MCP handshakes etc., so a 30s minimum protects against config typos.
_MIN_TIMEOUT_SECONDS = 30.0
_DEFAULT_TIMEOUT_SECONDS = 180.0
_DOCUMENT_CACHE_MAX = 8

# Codex feature flags we disable by default for context generation.
# Each flag costs input tokens (the tool catalog Codex ships in every prompt
# grows with the enabled feature set) and some flags trigger background work
# Recall does not want — most visibly `plugins`, which makes the CLI run
# `git ls-remote https://github.com/openai/plugins.git` on every invocation
# even with --ignore-user-config. Disabling the 19 features below cuts a
# 9-word summarization prompt from ~26k input tokens to ~13k (measured on
# codex-cli 0.133.0 against gpt-5.4-mini) without changing output quality.
# Keep this list in sync with the rationale in the docstring; new features
# default-on in Codex should be triaged here when they appear in `codex
# features list`.
_DISABLED_CODEX_FEATURES: tuple[str, ...] = (
    # Plugin/skill subsystem — fetches github.com/openai/plugins.git on every
    # call and inflates the tool catalog. Recall never runs Codex plugins.
    "plugins",
    "plugin_hooks",
    "plugin_sharing",
    # Browser/computer/image tools — Recall feeds Codex a text document and
    # expects a text summary. None of these tools are ever useful here.
    "browser_use",
    "browser_use_external",
    "in_app_browser",
    "computer_use",
    "image_generation",
    # Agent-orchestration features that only matter for interactive sessions.
    # `multi_agent` is retained when the caller explicitly requests `ultra`,
    # whose Codex contract includes automatic task delegation.
    "multi_agent",
    "goals",
    "hooks",
    "apps",
    "workspace_dependencies",
    # Shell/exec tools — the sandbox is already `read-only`, but the model
    # still pays the input-token cost for the catalog entries until disabled.
    "shell_tool",
    "unified_exec",
    "shell_snapshot",
    # Approvals / MCP / suggestion subsystems that produce no output recall
    # consumes for a single-pass summarisation call.
    "guardian_approval",
    "skill_mcp_dependency_install",
    "tool_suggest",
    "tool_call_mcp_elicitation",
)


@dataclass(frozen=True)
class _ContextRequest:
    message: Message
    document: str
    chunk: str
    prompt: str


class CodexCliBackend:
    """Generate context prefixes by shelling out to `codex exec`."""

    _load_failure: ClassVar[Exception | None] = None
    _active_processes: ClassVar[set[subprocess.Popen[str]]] = set()
    _active_processes_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config: ContextConfig) -> None:
        self._config = config
        self._model = (
            DEFAULT_CODEX_CONTEXT_MODEL if config.model == DEFAULT_CONTEXT_MODEL else config.model
        )
        self._available: bool | None = None
        self._executable_path: str | None = None
        # Cache stores the *rendered message list* per session, not the
        # final truncated document. Smart truncation depends on which chunk
        # is being summarized (the vicinity window slides with the chunk),
        # so caching a truncated document by session would serve stale
        # vicinities to chunks at different positions. Re-rendering 600+
        # messages on every codex call is the expensive bit (per-message
        # tool_input JSON parsing for claude_code sessions); applying
        # truncation against an already-rendered list is cheap, so we
        # amortize the render and recompute the trim each call.
        self._document_cache: OrderedDict[str, list[str]] = OrderedDict()

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            self._ensure_executable()
        except Exception as err:
            type(self)._load_failure = err
            self._available = False
            return False
        self._available = True
        return True

    def generate_prefix(self, session: Session, message: Message) -> ContextResult:
        if len(message.content or "") < self._config.min_chars:
            return ContextResult(prefix="", mode="off")

        self._ensure_executable()
        prompt = self._build_prompt(session, message)
        text, usage = self._run_codex(prompt)
        text = strip_think_blocks(text).strip()
        prefix = f"[{text}] " if text else ""
        return ContextResult(
            prefix=prefix,
            mode="llm-codex" if prefix else "off",
            input_tokens=usage.get("input_tokens", 0) + usage.get("cached_input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0) + usage.get("reasoning_output_tokens", 0),
            model=self._model,
        )

    def plan_batches(self, session: Session, messages: list[Message]) -> list[list[Message]]:
        """Group eligible messages that share one smart-truncated document window."""

        batches: list[list[Message]] = []
        current: list[Message] = []
        current_document: str | None = None
        current_payload_chars = 0
        for message in messages:
            if len(message.content or "") < self._config.min_chars:
                if current:
                    batches.append(current)
                    current = []
                    current_document = None
                    current_payload_chars = 0
                batches.append([message])
                continue

            request = self._build_context_request(session, message)
            chunk_chars = len(request.chunk)
            payload_chars = len(request.document) + current_payload_chars + chunk_chars
            over_document_budget = (
                self._config.max_document_chars is not None
                and payload_chars > self._config.max_document_chars
                and bool(current)
            )
            should_flush = bool(current) and (
                request.document != current_document
                or len(current) >= self._config.batch_size
                or over_document_budget
            )
            if should_flush:
                batches.append(current)
                current = []
                current_document = None
                current_payload_chars = 0

            current.append(message)
            current_document = request.document
            current_payload_chars += chunk_chars

        if current:
            batches.append(current)
        return batches

    def generate_prefixes(self, session: Session, messages: list[Message]) -> list[ContextResult]:
        """Produce contexts for one planned batch of messages."""

        if not messages:
            return []
        if len(messages) == 1:
            return [self.generate_prefix(session, messages[0])]

        self._ensure_executable()
        requests = [self._build_context_request(session, message) for message in messages]
        document = requests[0].document
        if any(request.document != document for request in requests):
            raise RuntimeError("codex batch contains messages from different document windows")

        prompt = self._build_batch_prompt(document, [request.chunk for request in requests])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as schema_fh:
            schema_path = Path(schema_fh.name)
            json.dump(_batch_output_schema(len(messages)), schema_fh)
        try:
            text, usage = self._run_codex(prompt, output_schema_path=schema_path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                schema_path.unlink()
        contexts = _parse_batch_contexts(text, expected_count=len(messages))

        results: list[ContextResult] = []
        for index, context in enumerate(contexts):
            cleaned = strip_think_blocks(context).strip()
            prefix = f"[{cleaned}] " if cleaned else ""
            results.append(
                ContextResult(
                    prefix=prefix,
                    mode="llm-codex" if prefix else "off",
                    input_tokens=(
                        usage.get("input_tokens", 0) + usage.get("cached_input_tokens", 0)
                        if index == 0
                        else 0
                    ),
                    output_tokens=(
                        usage.get("output_tokens", 0) + usage.get("reasoning_output_tokens", 0)
                        if index == 0
                        else 0
                    ),
                    model=self._model,
                )
            )
        return results

    def _ensure_executable(self) -> None:
        if self._executable_path is not None:
            return
        candidate = self._config.executable
        # Allow callers to pass an absolute path (handy in tests and for users
        # who installed codex outside PATH). Otherwise resolve via PATH so a
        # systemd/launchd daemon with a stripped env can still find it.
        resolved = candidate if Path(candidate).is_absolute() else shutil.which(candidate)
        if not resolved:
            raise RuntimeError(
                f"codex CLI executable {candidate!r} not found on PATH; "
                "install codex or set [embedding.context] executable = '/abs/path/to/codex'"
            )
        if not Path(resolved).exists():
            raise RuntimeError(f"codex CLI executable not found at {resolved!r}")
        self._executable_path = resolved

    def _build_prompt(self, session: Session, message: Message) -> str:
        return self._build_context_request(session, message).prompt

    def _build_context_request(self, session: Session, message: Message) -> _ContextRequest:
        # Render the session messages exactly once per session and cache the
        # list. Smart truncation slides with the chunk being summarized so it
        # must run per call, but the cost there is just slicing strings — the
        # expensive work (per-message tool_input JSON parsing for claude_code
        # sources) is what we want to amortize.
        cached = self._document_cache.get(session.id)
        if cached is not None:
            self._document_cache.move_to_end(session.id)
            rendered_messages = cached
        else:
            include_tools = session.source == Source.CLAUDE_CODE
            rendered_messages = [
                _render_message(item, include_tools=include_tools) for item in session.messages
            ]
            while len(self._document_cache) >= _DOCUMENT_CACHE_MAX:
                self._document_cache.popitem(last=False)
            self._document_cache[session.id] = rendered_messages

        max_chars = self._config.max_document_chars
        if max_chars is not None:
            chunk_idx_in_list = _message_position(session.messages, message)
            document = _smart_truncate(
                rendered_messages,
                chunk_idx_in_list,
                max_chars=max_chars,
            )
        else:
            document = "\n".join(rendered_messages)

        chunk = message.content or ""
        prefix = self._config.instruction_prefix or ""
        prompt = (
            f"{prefix}"
            f"<document>{document}</document>\n"
            "Here is the chunk we want to situate within the whole document\n"
            f"<chunk>{chunk}</chunk>\n"
            "Please give a short succinct context to situate this chunk within the overall "
            "document for the purposes of improving search retrieval of the chunk. Answer only "
            "with the succinct context and nothing else."
        )
        return _ContextRequest(message=message, document=document, chunk=chunk, prompt=prompt)

    def _build_batch_prompt(self, document: str, chunks: list[str]) -> str:
        prefix = self._config.instruction_prefix or ""
        rendered_chunks = "\n".join(
            f'<chunk index="{index}">{chunk}</chunk>' for index, chunk in enumerate(chunks)
        )
        return (
            f"{prefix}"
            f"<document>{document}</document>\n"
            "Here are the chunks we want to situate within the whole document.\n"
            f"<chunks>\n{rendered_chunks}\n</chunks>\n"
            "Please give a short succinct context for each chunk to situate it within the "
            "overall document for the purposes of improving search retrieval of the chunk. "
            "Answer only with a JSON object matching the provided schema: a `contexts` array "
            "whose objects each contain the chunk's index and its succinct context."
        )

    def _run_codex(
        self,
        prompt: str,
        *,
        output_schema_path: Path | None = None,
    ) -> tuple[str, dict[str, int]]:
        if self._executable_path is None:
            # _ensure_executable() always runs before this in generate_prefix; this
            # branch protects against direct calls in tests that forgot to.
            raise RuntimeError("codex executable not resolved; call _ensure_executable()")
        timeout = self._resolve_timeout()
        # NamedTemporaryFile with delete=False so we control unlink order on Windows
        # too — codex writes to the file path, we read after it exits.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            last_message_path = Path(fh.name)
        try:
            argv = self._build_argv(
                last_message_path,
                output_schema_path=output_schema_path,
            )
            proc: subprocess.Popen[str] | None = None
            try:
                proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                with type(self)._active_processes_lock:
                    type(self)._active_processes.add(proc)
                stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
            except subprocess.TimeoutExpired as err:
                if proc is not None:
                    _terminate_process_group(proc, grace_seconds=2.0)
                raise RuntimeError(
                    f"codex exec timed out after {timeout:.0f}s (model={self._model})"
                ) from err
            finally:
                if proc is not None:
                    with type(self)._active_processes_lock:
                        type(self)._active_processes.discard(proc)
                    _close_process_pipes(proc)
            if proc is None:
                raise RuntimeError("codex exec failed to start")
            if proc.returncode != 0:
                # Codex CLI writes its error JSON to stdout as a JSONL `error`
                # event; stderr usually carries setup warnings. Prefer the
                # structured event for the exception message.
                detail = _first_error_message(stdout) or stderr.strip()
                raise RuntimeError(
                    f"codex exec exited {proc.returncode}: {detail or '(no error detail)'}"
                )
            text = last_message_path.read_text(encoding="utf-8")
            usage = _parse_usage(stdout)
            return text, usage
        finally:
            with contextlib.suppress(FileNotFoundError):
                last_message_path.unlink()

    def _resolve_timeout(self) -> float:
        # Reuse the existing timeout config field. Floor at _MIN_TIMEOUT_SECONDS
        # because anything shorter is virtually guaranteed to mis-fire on the
        # first cold-start (auth refresh, MCP probe). Unset → conservative default
        # rather than the SDK's 600s, since contextual indexing fires per-chunk.
        configured = self._config.timeout
        if configured is None:
            return _DEFAULT_TIMEOUT_SECONDS
        return max(float(configured), _MIN_TIMEOUT_SECONDS)

    def _build_argv(
        self,
        last_message_path: Path,
        *,
        output_schema_path: Path | None = None,
    ) -> list[str]:
        if self._executable_path is None:
            raise RuntimeError("codex executable not resolved")
        argv: list[str] = [
            self._executable_path,
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--ignore-user-config",
            "--ignore-rules",
            # web_search is "cached" by default, which both bloats the tool catalog
            # (and therefore input tokens) and makes reasoning.effort='minimal' fail
            # server-side. Recall never wants the model to do web search to
            # summarise a session chunk, so unconditionally disable it.
            "-c",
            'web_search="disabled"',
        ]
        # Append `--disable <feature>` pairs for the curated block above. The
        # list is intentionally hard-coded because these features are tied to
        # the "summarise one session chunk" workload. Ultra is the one semantic
        # exception: Codex defines it to include automatic task delegation, so
        # retaining multi_agent is required to honor the user's explicit mode.
        for feature in _DISABLED_CODEX_FEATURES:
            if feature == "multi_agent" and self._config.reasoning_effort == "ultra":
                continue
            argv.extend(["--disable", feature])
        if self._config.reasoning_effort is not None:
            argv.extend(["-c", f'model_reasoning_effort="{self._config.reasoning_effort}"'])
        argv.extend(["--model", self._model])
        if output_schema_path is not None:
            argv.extend(["--output-schema", str(output_schema_path)])
        argv.extend(["-o", str(last_message_path)])
        # The "-" sentinel tells codex exec to read the prompt from stdin even
        # though we could pass the prompt as an arg; stdin avoids argv length
        # limits and shell-escaping pitfalls for the huge document blocks.
        argv.append("-")
        return argv


def terminate_active_codex_processes(grace_seconds: float = 3.0) -> int:
    """Terminate all live codex subprocesses; return count terminated.

    Sends SIGTERM to each process group, waits up to grace_seconds for them to
    exit, then SIGKILLs survivors. Safe to call from a signal handler on the
    main thread. Idempotent: an empty registry returns 0.
    """
    with CodexCliBackend._active_processes_lock:
        procs = list(CodexCliBackend._active_processes)
    for proc in procs:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(proc.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=remaining)
    for proc in procs:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=1.0)
    return len(procs)


def _terminate_process_group(
    proc: subprocess.Popen[str],
    *,
    grace_seconds: float,
) -> None:
    """Stop a timed-out codex process group before surfacing timeout failure."""
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(proc.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=grace_seconds)
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            proc.wait()


def _close_process_pipes(proc: subprocess.Popen[str]) -> None:
    """Close pipe handles after `communicate` or timeout cleanup."""
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.close()


def _message_position(messages: list[Message], message: Message) -> int:
    """Find the chunk's current position after any upstream message filtering."""
    for index, candidate in enumerate(messages):
        if candidate.id == message.id:
            return index
    for index, candidate in enumerate(messages):
        if candidate.idx == message.idx:
            return index
    raise ValueError(
        f"message {message.id!r} (idx={message.idx}) is not present in session messages"
    )


def _smart_truncate(
    rendered_messages: list[str],
    chunk_idx_in_list: int,
    *,
    max_chars: int,
    preamble_messages: int = 5,
    preamble_chars: int = 5000,
    vicinity_radius: int = 50,
) -> str:
    """Return a joined document <= max_chars with preamble + chunk-vicinity regions."""
    document = "\n".join(rendered_messages)
    if len(document) <= max_chars:
        return document
    if not rendered_messages:
        return ""

    chunk_idx_in_list = max(0, min(chunk_idx_in_list, len(rendered_messages) - 1))
    preamble_end = _preamble_end(
        rendered_messages,
        max_messages=preamble_messages,
        max_chars=preamble_chars,
    )

    # Huge messages in the vicinity can still overflow the cap. Shrink the
    # symmetric window on message boundaries before falling back to an empty
    # vicinity; that keeps all retained context parseable as complete lines.
    for radius in range(vicinity_radius, -1, -1):
        vicinity_start = max(0, chunk_idx_in_list - radius)
        vicinity_end = min(len(rendered_messages), chunk_idx_in_list + radius + 1)
        candidate = _join_regions_with_markers(
            rendered_messages,
            [(0, preamble_end), (vicinity_start, vicinity_end)],
        )
        if len(candidate) <= max_chars:
            return candidate

    for preamble_size in range(preamble_end, -1, -1):
        candidate = _join_regions_with_markers(
            rendered_messages,
            [(0, preamble_size)],
        )
        if len(candidate) <= max_chars:
            return candidate

    marker = _omission_marker(rendered_messages, 0, len(rendered_messages))
    return marker if len(marker) <= max_chars else marker[:max_chars]


def _preamble_end(
    rendered_messages: list[str],
    *,
    max_messages: int,
    max_chars: int,
) -> int:
    end = 0
    current_chars = 0
    for message in rendered_messages[:max_messages]:
        separator_chars = 1 if end > 0 else 0
        next_chars = current_chars + separator_chars + len(message)
        if next_chars > max_chars:
            break
        current_chars = next_chars
        end += 1
    return end


def _join_regions_with_markers(
    rendered_messages: list[str],
    regions: list[tuple[int, int]],
) -> str:
    normalized = _normalize_regions(regions, total=len(rendered_messages))
    parts: list[str] = []
    cursor = 0
    for start, end in normalized:
        if cursor < start:
            parts.append(_omission_marker(rendered_messages, cursor, start))
        if start < end:
            parts.append("\n".join(rendered_messages[start:end]))
        cursor = end
    if cursor < len(rendered_messages):
        parts.append(_omission_marker(rendered_messages, cursor, len(rendered_messages)))
    return "\n".join(part for part in parts if part)


def _normalize_regions(
    regions: list[tuple[int, int]],
    *,
    total: int,
) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    for raw_start, raw_end in sorted(regions):
        start = max(0, min(raw_start, total))
        end = max(start, min(raw_end, total))
        if start == end:
            continue
        if normalized and start <= normalized[-1][1]:
            previous_start, previous_end = normalized[-1]
            normalized[-1] = (previous_start, max(previous_end, end))
        else:
            normalized.append((start, end))
    return normalized


def _omission_marker(rendered_messages: list[str], start: int, end: int) -> str:
    omitted_text = "\n".join(rendered_messages[start:end])
    omitted_bytes = len(omitted_text.encode("utf-8"))
    omitted_messages = end - start
    return f"[... {omitted_bytes} chars / {omitted_messages} messages omitted ...]"


def _render_message(message: Message, *, include_tools: bool = False) -> str:
    content = message.content or message.thinking or ""
    lines = [f"{message.idx}:{message.role.value}: {content}"]
    if include_tools:
        lines.extend(_render_tool_call(message, tool_call) for tool_call in message.tool_calls)
    return "\n".join(lines)


def _render_tool_call(message: Message, tool_call: ToolCall) -> str:
    args = _tool_call_args(tool_call)
    return f"{message.idx}:tool: {tool_call.tool_name} {args}".rstrip()


def _tool_call_args(tool_call: ToolCall) -> str:
    if tool_call.tool_name == "Bash":
        return tool_call.bash_command or _tool_input_text(tool_call.tool_input)
    if tool_call.tool_name in {"Edit", "Write", "Read", "NotebookEdit"}:
        file_path = _tool_input_mapping(tool_call.tool_input).get("file_path")
        return str(file_path) if file_path is not None else ""
    if tool_call.tool_name == "Skill":
        return tool_call.skill_name or ""
    return _tool_input_text(tool_call.tool_input)


def _tool_input_mapping(tool_input: object) -> dict[str, object]:
    if isinstance(tool_input, dict):
        return {str(key): value for key, value in tool_input.items()}
    if isinstance(tool_input, str):
        try:
            parsed = json.loads(tool_input)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(key): value for key, value in parsed.items()}
    return {}


def _tool_input_text(tool_input: object) -> str:
    if tool_input is None:
        return ""
    if isinstance(tool_input, str):
        raw = tool_input
    else:
        raw = json.dumps(tool_input, sort_keys=True, separators=(",", ":"), default=str)
    return re.sub(r"\s+", " ", raw).strip()[:120]


def _batch_output_schema(expected_count: int) -> dict[str, Any]:
    # codex `exec --output-schema` relays this as the model's `response_format`,
    # which requires the *root* schema to be `type: "object"` — a top-level
    # `type: "array"` is rejected server-side with 400 invalid_json_schema and
    # turns every batch into a REQ-CTX-024 failure. So the per-chunk array lives
    # under a single `contexts` property rather than at the root (REQ-CTX-022).
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["contexts"],
        "properties": {
            "contexts": {
                "type": "array",
                "minItems": expected_count,
                "maxItems": expected_count,
                "items": {
                    "type": "object",
                    "required": ["index", "context"],
                    "additionalProperties": False,
                    "properties": {
                        "index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": expected_count - 1,
                        },
                        "context": {"type": "string"},
                    },
                },
            },
        },
    }


def _parse_batch_contexts(text: str, *, expected_count: int) -> list[str]:
    raw = text.strip()
    if not raw:
        raise RuntimeError("codex batch returned an empty final message")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as err:
        raise RuntimeError("codex batch returned invalid JSON") from err
    # The schema (see `_batch_output_schema`) wraps the per-chunk array in an
    # object under `contexts`. Accept that object shape; also tolerate a bare
    # array for defense in depth if a model ignores the wrapper.
    if isinstance(payload, dict):
        items = payload.get("contexts")
        if not isinstance(items, list):
            raise RuntimeError("codex batch response object missing a 'contexts' array")
    elif isinstance(payload, list):
        items = payload
    else:
        raise RuntimeError("codex batch response must be a JSON object with a 'contexts' array")
    if len(items) != expected_count:
        raise RuntimeError(
            f"codex batch returned {len(items)} items for {expected_count} requested chunks"
        )

    contexts_by_index: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("codex batch response item must be an object")
        index = item.get("index")
        context = item.get("context")
        if not isinstance(index, int) or isinstance(index, bool):
            raise RuntimeError("codex batch response item has invalid index")
        if not isinstance(context, str):
            raise RuntimeError("codex batch response item has invalid context")
        if index in contexts_by_index:
            raise RuntimeError(f"codex batch response duplicated index {index}")
        contexts_by_index[index] = context

    expected_indices = set(range(expected_count))
    actual_indices = set(contexts_by_index)
    if actual_indices != expected_indices:
        raise RuntimeError(
            "codex batch response indices did not match requested chunks: "
            f"expected {sorted(expected_indices)}, got {sorted(actual_indices)}"
        )
    return [contexts_by_index[index] for index in range(expected_count)]


def _parse_usage(stdout: str) -> dict[str, int]:
    """Pull the final `turn.completed.usage` block out of the JSONL stream.

    Codex CLI may emit multiple turn.completed events if it retries internally;
    we take the last one. Missing or malformed events return zeros so the
    caller can still produce a ContextResult.
    """
    usage: dict[str, int] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Non-JSON lines (bubblewrap warnings, etc.) are expected pre-exec.
            continue
        if event.get("type") == "turn.completed":
            payload = event.get("usage")
            if isinstance(payload, dict):
                usage = {k: int(v) for k, v in payload.items() if isinstance(v, (int, float))}
    return usage


def _first_error_message(stdout: str) -> str:
    """Find the first `error` event in JSONL stdout and return its message."""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "error":
            message = event.get("message")
            if isinstance(message, str):
                return message
    return ""
