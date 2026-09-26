"""REQ-INDEX-021: a bare --full under an LLM context mode is an
expensive mistake, so the CLI says so before the run starts."""

from __future__ import annotations

import pytest
from recall.cli.index import _warn_if_full_reparse_will_summarize


class _Ctx:
    def __init__(self, mode: str) -> None:
        self.mode = mode


class _Cfg:
    def __init__(self, mode: str) -> None:
        self.embedding = type("E", (), {"context": _Ctx(mode)})()


@pytest.mark.parametrize("mode", ["llm-local", "llm-remote", "llm-codex"])
def test_warns_for_llm_context_modes(monkeypatch, capsys, mode: str) -> None:
    monkeypatch.setattr("recall.core.config.AppConfig.load", staticmethod(lambda: _Cfg(mode)))
    _warn_if_full_reparse_will_summarize()
    err = capsys.readouterr().err
    assert "--context template" in err
    assert mode in err


@pytest.mark.parametrize("mode", ["off", "template"])
def test_stays_quiet_for_cheap_context_modes(monkeypatch, capsys, mode: str) -> None:
    monkeypatch.setattr("recall.core.config.AppConfig.load", staticmethod(lambda: _Cfg(mode)))
    _warn_if_full_reparse_will_summarize()
    assert capsys.readouterr().err == ""


def test_a_broken_config_never_fails_the_command(monkeypatch, capsys) -> None:
    """A diagnostic must not be able to break the run it is describing."""

    def boom() -> None:
        raise RuntimeError("unreadable config")

    monkeypatch.setattr("recall.core.config.AppConfig.load", staticmethod(boom))
    _warn_if_full_reparse_will_summarize()
    assert capsys.readouterr().err == ""
