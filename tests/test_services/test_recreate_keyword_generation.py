"""A confirmed reset rebuilds the same keyword corpus, including its ranking."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from recall.core.config import AppConfig, FtsConfig
from recall.db import connect
from recall.db.fts_sidecar import (
    open_sidecar,
    search_messages_fts,
    search_tool_calls_fts,
    sidecar_path,
    upsert_message_fts,
    upsert_tool_call_fts,
)
from recall.db.lease import StandaloneLifetime
from recall.services.recreation import prepare_recreation_on, replace_schema_on


@pytest.mark.parametrize("kind", ["message", "tool"])
def test_confirmed_reset_preserves_scores_when_republishing_the_same_corpus(kind: str):
    config = replace(AppConfig.load(), fts=FtsConfig(backend="sqlite_sidecar"))

    def seed_canonical(conn):
        conn.execute("INSERT INTO sessions VALUES ('s', 'codex', '/fixture/source', NULL)")
        if kind == "message":
            conn.execute("INSERT INTO messages VALUES ('a','s',0,NULL),('b','s',1,NULL)")
            conn.execute(
                "INSERT INTO message_state(message_id,role,content,fts_content) VALUES "
                "('a','user','amber orbit','amber orbit'),"
                "('b','assistant','violet garden','violet garden')"
            )
        else:
            conn.execute(
                "INSERT INTO tool_calls(id,session_id,idx,tool_name,bash_command) VALUES "
                "('a','s',0,'bash','amber orbit'),('b','s',1,'bash','violet garden')"
            )

    def publish_and_search():
        with closing(open_sidecar(sidecar_path(config.data_dir))) as sidecar:
            if kind == "message":
                upsert_message_fts(sidecar, "a", "amber orbit", "")
                upsert_message_fts(sidecar, "b", "violet garden", "")
                return search_messages_fts(sidecar, "amber", ["content"], 10)
            upsert_tool_call_fts(sidecar, "a", "amber orbit")
            upsert_tool_call_fts(sidecar, "b", "violet garden")
            return search_tool_calls_fts(sidecar, "amber", 10)

    with connect(config) as conn:
        seed_canonical(conn)
        before = publish_and_search()
        assert before and before[0][0] == "a"
        with StandaloneLifetime().lease(conn) as lease:
            capability = prepare_recreation_on(lease, config)
            assert (Path(capability.backup) / "manifest.json").is_file()
            replace_schema_on(lease, config, capability)
        seed_canonical(conn)
        assert publish_and_search() == before


def test_confirmed_reset_observes_sidecar_failure_before_close(monkeypatch: pytest.MonkeyPatch):
    import errno
    import sqlite3

    from recall.services import recreation

    config = replace(AppConfig.load(), fts=FtsConfig(backend="sqlite_sidecar"))
    primary = OSError(errno.ENOSPC, "sidecar clear is full")
    cleanup = OSError(errno.EIO, "sidecar close failed")
    observed: list[BaseException] = []
    observed_before_close = False

    class ObservingOwner(StandaloneLifetime):
        def observe(self, error: BaseException, operation: str) -> bool:
            observed.append(error)
            return False

    class FailingSidecar:
        def __init__(self, native):
            self.native = native

        def execute(self, sql, *args):
            if sql != "BEGIN":
                raise primary
            return self.native.execute(sql, *args)

        def rollback(self):
            return self.native.rollback()

        def commit(self):
            return self.native.commit()

        def close(self):
            nonlocal observed_before_close
            observed_before_close = primary in observed
            self.native.close()
            raise cleanup

        def __enter__(self):
            self.native.__enter__()
            return self

        def __exit__(self, *args):
            return self.native.__exit__(*args)

    with connect(config) as conn, ObservingOwner().lease(conn) as lease:
        with closing(open_sidecar(sidecar_path(config.data_dir))):
            pass
        capability = prepare_recreation_on(lease, config)
        native_connect = sqlite3.connect
        monkeypatch.setattr(
            recreation.sqlite3,
            "connect",
            lambda *args, **kwargs: FailingSidecar(native_connect(*args, **kwargs)),
        )
        with pytest.raises(BaseExceptionGroup) as raised:
            replace_schema_on(lease, config, capability)
    assert observed_before_close
    assert raised.value.exceptions == (primary, cleanup)
