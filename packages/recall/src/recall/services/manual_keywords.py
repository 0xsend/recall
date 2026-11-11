"""Explicit manual publication; canonical intent retirement belongs to its caller."""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from recall.core.config import AppConfig
from recall.core.durability import sync_directory
from recall.db.fts_sidecar import (
    open_sidecar,
    open_sidecar_for_publication,
    set_fts_fields_signature,
    sidecar_path,
)
from recall.db.lease import DbLease, DbLifetime, owned_resource
from recall.db.repair_oracle import (
    NATIVE_KEYWORD_SCHEMAS,
    OracleUnavailable,
    _catalog,
    managed_native_keyword_table,
)
from recall.services.fts_sidecar_bootstrap import bootstrap_sidecar_on
from recall.services.fts_sidecar_rebuild import verify_candidate


@dataclass(frozen=True)
class NativeKeywordCandidate:
    path: Path
    lineage: tuple[object, int]
    connection_epoch: object


def require_replaceable_sidecar(config: AppConfig, *, lifetime: DbLifetime) -> None:
    """Refuse reconstruction when the companion contains unsupported data."""
    live = sidecar_path(config.data_dir)
    if config.fts.backend != "sqlite_sidecar" or not live.exists():
        return

    def catalog(conn):
        rows = conn.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        ).fetchmany(4097)
        if len(rows) > 4096 or any(len(row[3] or "") > 65536 for row in rows):
            raise OracleUnavailable("unsupported sidecar catalog exceeds reconstruction bounds")
        return rows

    with (
        owned_resource(
            lifetime, "manual_keywords:supported_schema", lambda: open_sidecar(Path(":memory:"))
        ) as supported,
        owned_resource(
            lifetime,
            "manual_keywords:source_schema",
            lambda: sqlite3.connect(f"{live.resolve().as_uri()}?mode=ro", uri=True, timeout=1),
        ) as source,
    ):

        def qualify():
            set_fts_fields_signature(supported, config.fts.fields)
            if catalog(source) != catalog(supported):
                raise OracleUnavailable("unsupported sidecar schema requires matched restore")
            if source.execute(
                "SELECT EXISTS(SELECT 1 FROM bootstrap_progress WHERE kind NOT IN "
                "('message','tool_call'))"
            ).fetchone() != (0,):
                raise OracleUnavailable("unsupported sidecar progress requires matched restore")
            keys = tuple(row[0] for row in supported.execute("SELECT key FROM fts_sidecar_meta"))
            parameters = ",".join("?" for _ in keys)
            if source.execute(
                f"SELECT EXISTS(SELECT 1 FROM fts_sidecar_meta WHERE key NOT IN ({parameters}))",
                keys,
            ).fetchone() != (0,):
                raise OracleUnavailable("unsupported sidecar metadata requires matched restore")

        lifetime.call_db("manual_keywords:qualify_replacement", qualify)


def verify_manual_keywords(lease: DbLease, config: AppConfig, *, lifetime: DbLifetime) -> None:
    assert lease.lineage[0] is lifetime
    if config.fts.backend == "duckdb":
        lease.call("manual_keywords:load", lambda conn: conn.execute("LOAD fts").fetchone())
        lease.call(
            "manual_keywords:verify_native", lambda conn: verify_native_keywords(conn, config)
        )
    else:
        with owned_resource(
            lifetime,
            "manual_keywords:verify_sidecar",
            lambda: open_sidecar_for_publication(sidecar_path(config.data_dir)),
        ) as keyword:
            lease.call(
                "manual_keywords:verify_membership",
                lambda conn: verify_candidate(conn, keyword, config.fts.fields),
            )


def publish_manual_keywords(
    lease: DbLease, config: AppConfig, working: Path, *, lifetime: DbLifetime
) -> None:
    """Publish the complete canonical membership without retiring any DuckDB intent."""
    assert lease.lineage[0] is lifetime, "manual keyword work belongs to another owner"
    invalid = lease.call(
        "manual_keywords:pending_shape",
        lambda conn: conn.execute(
            "SELECT EXISTS(SELECT 1 FROM fts_sidecar_pending "
            "WHERE kind NOT IN ('message','tool_call') OR op NOT IN ('upsert','delete'))"
        ).fetchone(),
    )
    assert invalid is not None
    if invalid[0]:
        raise ValueError("unsupported pending keyword intent requires matched restore")
    assert config.fts.backend == "sqlite_sidecar", (
        "native publication requires a database candidate"
    )
    candidate = working / "keyword-candidate.sqlite"
    if candidate.exists():
        raise ValueError("a manual keyword candidate cannot overwrite an earlier attempt")

    # The owner keeps every native handle inside its observed leaf scope.
    def publish(conn):
        with owned_resource(
            lifetime, "manual_keywords:candidate", lambda: open_sidecar(candidate)
        ) as keyword:
            result = bootstrap_sidecar_on(lease, keyword, fts_fields=config.fts.fields)
            if not result.messages_done or not result.tool_calls_done:
                raise ValueError("manual keyword capture did not finish")
            set_fts_fields_signature(keyword, config.fts.fields)
            keyword.commit()
            verify_candidate(conn, keyword, tuple(config.fts.fields))
            keyword.commit()

    lease.call("manual_keywords:candidate", publish)
    with candidate.open("rb") as stream:
        os.fsync(stream.fileno())
    sync_directory(working)
    live = sidecar_path(config.data_dir)
    if live.exists():
        original = working / "keyword-original.sqlite"
        if original.exists():
            raise ValueError("a manual keyword backup cannot overwrite an earlier attempt")
        started = time.monotonic()

        def progress(status, remaining, total):
            if time.monotonic() - started > 300:
                raise TimeoutError("manual keyword backup exceeded its five minute limit")

        with (
            owned_resource(
                lifetime, "manual_keywords:backup_source", lambda: sqlite3.connect(live, timeout=1)
            ) as backup_source,
            owned_resource(
                lifetime,
                "manual_keywords:backup_target",
                lambda: sqlite3.connect(original, timeout=1),
            ) as backup_target,
        ):
            lifetime.call_db(
                "manual_keywords:backup",
                lambda: backup_source.backup(
                    backup_target, pages=256, progress=progress, sleep=0.01
                ),
            )
        with original.open("rb") as stream:
            os.fsync(stream.fileno())
        sync_directory(working)
    # The attempt's recorded unique directory owns this preserved original set.
    # A crash in the rename window leaves the canonical row RUNNING and R held.
    for suffix in ("", "-wal", "-shm"):
        source = Path(str(live) + suffix)
        backup = working / ("keyword-live.sqlite" + suffix)
        if backup.exists():
            raise ValueError("a manual keyword backup cannot overwrite an earlier attempt")
        if source.exists():
            source.rename(backup)
    sync_directory(working)
    sync_directory(live.parent)
    candidate.rename(live)
    sync_directory(live.parent)


def publish_native_keywords(
    lease: DbLease, config: AppConfig, candidate: NativeKeywordCandidate
) -> None:
    from recall.db.queries import load_fts_extension

    assert (
        candidate.lineage == lease.lineage and candidate.connection_epoch is lease.connection_epoch
    )
    actual = lease.call(
        "manual_keywords:candidate_identity",
        lambda conn: conn.execute(
            "SELECT path FROM duckdb_databases() WHERE database_name=current_database()"
        ).fetchone(),
    )
    assert actual is not None and Path(actual[0]).resolve() == candidate.path.resolve()
    assert candidate.path.resolve() != config.db_path.resolve(), (
        "native FTS cannot rewrite the original"
    )
    lease.call("manual_keywords:namespace", _require_managed_namespace)
    lease.call("manual_keywords:extension", load_fts_extension)
    for schema in sorted(NATIVE_KEYWORD_SCHEMAS):
        lease.call(
            "manual_keywords:drop_schema",
            lambda conn, schema=schema: conn.execute(
                f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'
            ).fetchone(),
        )
    fields = tuple(field for field in ("content", "thinking") if field in config.fts.fields)
    if fields:
        columns = ",".join("fts_" + field for field in fields)
        lease.call(
            "manual_keywords:messages",
            lambda conn: conn.execute(
                "PRAGMA create_fts_index(message_state,message_id,"
                + columns
                + ",stemmer='porter',stopwords='english',overwrite=1)"
            ).fetchone(),
        )
    if "bash" in config.fts.fields:
        lease.call(
            "manual_keywords:tools",
            lambda conn: conn.execute(
                "PRAGMA create_fts_index(tool_calls,id,bash_command,"
                "stemmer='porter',stopwords='english',overwrite=1)"
            ).fetchone(),
        )
    lease.call("manual_keywords:checkpoint", lambda conn: conn.execute("CHECKPOINT").fetchone())
    lease.call("manual_keywords:membership", lambda conn: verify_native_keywords(conn, config))


def _require_managed_namespace(conn) -> None:
    for table in _catalog(conn):
        if table.schema_name in NATIVE_KEYWORD_SCHEMAS and not managed_native_keyword_table(table):
            raise OracleUnavailable(
                "custom table in a native keyword namespace requires matched restore"
            )
    for schema in sorted(NATIVE_KEYWORD_SCHEMAS):
        for catalog, column in (
            ("duckdb_views()", "view_name"),
            ("duckdb_sequences()", "sequence_name"),
            ("duckdb_types()", "type_name"),
        ):
            if conn.execute(
                f"SELECT EXISTS(SELECT 1 FROM {catalog} WHERE schema_name=?)", [schema]
            ).fetchone() != (False,):
                raise OracleUnavailable(
                    f"custom {column} in a native keyword namespace requires matched restore"
                )
        if conn.execute(
            "SELECT EXISTS(SELECT 1 FROM duckdb_functions() WHERE schema_name=? "
            "AND (function_type!='macro' OR function_name NOT IN ('match_bm25','tokenize')))",
            [schema],
        ).fetchone() != (False,):
            raise OracleUnavailable(
                "custom macro in a native keyword namespace requires matched restore"
            )


def _require_equal(conn, left: str, right: str, name: str) -> None:
    result = conn.execute(
        f"SELECT EXISTS((({left}) EXCEPT ALL ({right})) UNION ALL (({right}) EXCEPT ALL ({left})))"
    ).fetchone()
    if result != (False,):
        raise OracleUnavailable(f"native keyword {name} differs from canonical membership")


def verify_native_keywords(conn, config: AppConfig) -> None:
    """Derive indexed token multiplicity from canonical fields, not index lookups."""
    scopes = (
        (
            "fts_main_message_state",
            "message_state",
            "message_id",
            tuple(
                "fts_" + field for field in ("content", "thinking") if field in config.fts.fields
            ),
        ),
        (
            "fts_main_tool_calls",
            "tool_calls",
            "id",
            ("bash_command",) if "bash" in config.fts.fields else (),
        ),
    )
    for schema, table, identity, fields in scopes:
        if not fields:
            if conn.execute(
                "SELECT EXISTS(SELECT 1 FROM duckdb_tables() WHERE schema_name=?)", [schema]
            ).fetchone() != (False,):
                raise OracleUnavailable("disabled native keyword scope retains indexed tables")
            continue
        _require_equal(
            conn,
            f'SELECT "name" FROM {schema}.docs',
            f"SELECT {identity} FROM {table}",
            "document identities",
        )
        expected_fields = " UNION ALL ".join(f"SELECT '{field}'" for field in fields)
        _require_equal(conn, f"SELECT field FROM {schema}.fields", expected_fields, "field scope")
        for name, key in (("docs", "docid"), ("fields", "fieldid"), ("dict", "termid")):
            if conn.execute(
                f"SELECT count(*)=count(DISTINCT {key}) FROM {schema}.{name}"
            ).fetchone() != (True,):
                raise OracleUnavailable("native keyword identities are not unique")
        raw = " UNION ALL ".join(
            f"SELECT {identity} AS name,'{field}' AS field,"
            f"unnest({schema}.tokenize({field})) AS token FROM {table}"
            for field in fields
        )
        expected = (
            f"SELECT name,field,stem(token,'porter') AS term FROM ({raw}) "
            f"WHERE token!='' AND token NOT IN (SELECT sw FROM {schema}.stopwords)"
        )
        actual_terms = (
            f"SELECT d.name,f.field,k.term FROM {schema}.terms t "
            f"JOIN {schema}.docs d USING(docid) JOIN {schema}.fields f USING(fieldid) "
            f"JOIN {schema}.dict k USING(termid)"
        )
        _require_equal(conn, expected, actual_terms, "token multiplicity")
        if (
            conn.execute(f"SELECT count(*) FROM ({actual_terms})").fetchone()
            != conn.execute(f"SELECT count(*) FROM {schema}.terms").fetchone()
        ):
            raise OracleUnavailable("native keyword terms contain orphan references")
        expected_dict = f"SELECT term,count(DISTINCT name)::BIGINT FROM ({expected}) GROUP BY term"
        _require_equal(
            conn, expected_dict, f"SELECT term,df FROM {schema}.dict", "document frequencies"
        )
        lengths = (
            f"SELECT d.name,count(t.docid)::BIGINT AS len FROM {schema}.docs d "
            f"LEFT JOIN {schema}.terms t USING(docid) GROUP BY d.docid,d.name"
        )
        _require_equal(conn, lengths, f'SELECT "name",len FROM {schema}.docs', "document lengths")
        _require_equal(
            conn,
            f"SELECT count(*)::BIGINT,avg(len) FROM {schema}.docs",
            f"SELECT num_docs,avgdl FROM {schema}.stats",
            "corpus statistics",
        )
