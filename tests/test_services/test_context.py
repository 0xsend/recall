from __future__ import annotations

import pytest
from recall.core.models import Session
from recall.core.types import Source
from recall.services.context import render_template_prefix, resolve_context


def _session(
    *,
    cwd: str | None = None,
    git_repo: str | None = None,
    git_branch: str | None = None,
) -> Session:
    return Session(
        id="session-1",
        source=Source.CODEX,
        source_path="/tmp/session.jsonl",
        file_mtime=1.0,
        file_size=1,
        cwd=cwd,
        git_repo=git_repo,
        git_branch=git_branch,
    )


@pytest.mark.parametrize(
    ("session", "expected"),
    [
        (_session(), ("", "off")),
        (_session(git_repo="acme/recall"), ("[acme/recall HEAD] ", "template")),
        (_session(cwd="/Users/dev/code/recall"), ("[recall HEAD] ", "template")),
        (_session(git_branch="main"), ("[local main] ", "template")),
        (
            _session(
                cwd="/Users/dev/code/other",
                git_repo="acme/recall",
                git_branch="main",
            ),
            ("[acme/recall main] ", "template"),
        ),
        (_session(cwd="/Users/dev/code/recall/"), ("[recall HEAD] ", "template")),
    ],
)
def test_render_template_prefix(session: Session, expected: tuple[str, str]) -> None:
    assert render_template_prefix(session) == expected


def test_resolve_context_requires_message_resolver_for_llm_modes() -> None:
    with pytest.raises(ValueError, match="resolve_message_context"):
        resolve_context(_session(), "llm-local")

    with pytest.raises(ValueError, match="resolve_message_context"):
        resolve_context(_session(), "llm-remote")
