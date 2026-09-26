from __future__ import annotations

from pathlib import Path

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
