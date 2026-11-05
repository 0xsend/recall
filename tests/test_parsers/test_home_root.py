from __future__ import annotations

from pathlib import Path

import pytest
from recall.core.config import SourceConfig
from recall.core.types import Source
from recall.parsers import all_parsers, get_parser
from recall.parsers.common import resolve_home, use_home_root
from recall.parsers.grok import GrokParser


def test_use_home_root_redirects_grok_discover(tmp_path: Path) -> None:
    sessions = tmp_path / ".grok" / "sessions" / "cwd" / "sid"
    sessions.mkdir(parents=True)
    chat = sessions / "chat_history.jsonl"
    chat.write_text('{"type":"user","content":[{"type":"text","text":"hi"}]}\n', encoding="utf-8")

    # Without the override, discovery uses the real home and never sees the tmp
    # fixture (hermetic: does not assume the real home has any Grok sessions).
    assert chat not in GrokParser().discover()
    with use_home_root(tmp_path):
        assert resolve_home() == tmp_path
        found = GrokParser().discover()
    assert found == [chat]


@pytest.mark.parametrize("configured", [None, (), (Path("local-sessions"),)])
def test_alternate_home_wins_over_source_roots_for_every_registered_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: tuple[Path, ...] | None
) -> None:
    home = tmp_path / "remote"
    expected = {
        Source.CLAUDE_CODE: home / ".claude/projects/project/chat.jsonl",
        Source.CODEX: home / ".codex/sessions/rollout-chat.jsonl",
        Source.PI_AGENT: home / ".pi/agent/sessions/chat.jsonl",
        Source.GROK: home / ".grok/sessions/project/session/chat_history.jsonl",
        Source.KIMI_CODE: home / ".kimi-code/sessions/project/session/wire.jsonl",
    }
    for path in expected.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "local-grok"))
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path / "local-kimi"))
    roots = None if configured is None else tuple(tmp_path / root for root in configured)
    sources = {source.value: SourceConfig(roots=roots) for source in Source}

    with use_home_root(home):
        for parser in all_parsers(sources):
            assert parser.discover() == [expected[parser.source]]
            assert get_parser(parser.source, sources).discover() == [expected[parser.source]]

    assert [parser.roots for parser in all_parsers(sources)] == [roots] * 5
