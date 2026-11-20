"""Keyword scope follows canonical binding, including moved source identities."""

import duckdb
from recall.core.source_path import SourcePath
from recall.db.lease import StandaloneLifetime
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog, SourceSignature
from recall.db.source_keywords import PendingKeywords, SettledKeywords, SourceSelection


def test_membership_uses_canonical_binding_and_leaves_duplicate_intents_intact() -> None:
    with duckdb.connect() as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO sessions VALUES ('bound','codex','/canonical/moved.jsonl',NULL), "
            "('outside','codex','/canonical/outside.jsonl',NULL)"
        )
        conn.execute("INSERT INTO messages VALUES ('message','bound',0,NULL)")
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) VALUES "
            "('message','message','upsert'),('message','message','upsert')"
        )
        catalog = SourceCatalog(conn, clock=lambda: 1.0)
        for name, session in [("selected", "bound"), ("outside", "outside")]:
            path = f"/root/{name}.jsonl"
            catalog.observe("codex", "/root", path, SourceSignature(1, 2, 3, 4, 5))
            catalog.bind_session("codex", path, session)
            assert catalog.acknowledge("codex", path, 1, 5, "a" * 64)
        selected = SourcePath("codex", "/root/selected.jsonl")
        outside = SourcePath("codex", "/root/outside.jsonl")
        with StandaloneLifetime().lease(conn) as lease:
            memberships = SourceSelection((selected, outside)).inspect_on(lease)
        assert {entry.path for entry in memberships if isinstance(entry, PendingKeywords)} == {
            selected
        }
        assert {entry.path for entry in memberships if isinstance(entry, SettledKeywords)} == {
            outside
        }
        census = catalog.root_census("codex", "/root")
        assert not census.unattributed_keywords
        assert census.entries[selected.path].current
        assert isinstance(census.entries[selected.path].keywords, PendingKeywords)
        assert isinstance(census.entries[outside.path].keywords, SettledKeywords)
        assert conn.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone() == (2,)


def test_empty_root_census_retains_unattributed_keyword_obligations() -> None:
    with duckdb.connect() as conn:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) VALUES ('message','unattributed','delete')"
        )
        census = SourceCatalog(conn, clock=lambda: 1.0).root_census("codex", "/empty")
        assert not census.entries
        assert census.unattributed_keywords
        assert conn.execute("SELECT id FROM fts_sidecar_pending").fetchone() == ("unattributed",)
