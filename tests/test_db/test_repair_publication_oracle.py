"""A post-publication reference permits only the declared intent retirement."""

from contextlib import closing
from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.db.fts_sidecar import open_sidecar, upsert_message_fts
from recall.db.lease import StandaloneLifetime
from recall.db.repair_oracle import (
    OracleUnavailable,
    capture_oracle,
    capture_publication,
    read_publication,
    verify_manual_content,
    verify_publication_sidecar,
)
from recall.db.schema import ensure_schema

from tests.test_db.test_repair_oracle import populate


def _capture(tmp_path: Path):
    database = tmp_path / "recall.duckdb"
    sidecar = tmp_path / "recall.fts.sqlite"
    with closing(open_sidecar(sidecar)) as keyword, keyword:
        upsert_message_fts(keyword, "a", "acknowledged keyword", "")
    directory = tmp_path / "repair-oracles" / str(UUID(int=1))
    with duckdb.connect(str(database)) as conn:
        ensure_schema(conn)
        populate(conn)
        conn.execute(
            "INSERT INTO fts_sidecar_pending(kind,id,op) "
            "VALUES ('message','a','upsert'),('message','a','upsert')"
        )
        reference = capture_oracle(
            conn, directory, UUID(int=1), sidecar=sidecar, observer=StandaloneLifetime()
        )
    return database, sidecar, directory, reference


def _publication(database: Path, directory: Path, reference, attempt: int, *, sidecar: Path):
    with duckdb.connect(str(database)) as conn:
        return capture_publication(
            conn, directory, reference, attempt, sidecar=sidecar, observer=StandaloneLifetime()
        )


def test_publication_reference_preserves_exact_content_except_captured_retirement(tmp_path: Path):
    database, sidecar, directory, reference = _capture(tmp_path)
    digest = _publication(database, directory, reference, 2, sidecar=sidecar)
    publication = read_publication(directory, digest)
    assert (publication.trigger_id, publication.attempt_no, publication.oracle_hash) == (
        UUID(int=1),
        2,
        reference.manifest_hash,
    )
    with duckdb.connect(str(database)) as conn:
        verify_manual_content(conn, directory, reference, observer=StandaloneLifetime())
        with pytest.raises(OracleUnavailable, match="content"):
            verify_manual_content(
                conn, directory, reference, publication=publication, observer=StandaloneLifetime()
            )
        conn.execute("DELETE FROM fts_sidecar_pending")
        verify_manual_content(
            conn, directory, reference, publication=publication, observer=StandaloneLifetime()
        )
        with pytest.raises(OracleUnavailable, match="content"):
            verify_manual_content(conn, directory, reference, observer=StandaloneLifetime())
        conn.execute("UPDATE canonical SET text='unacknowledged' WHERE id='a'")
        with pytest.raises(OracleUnavailable, match="content"):
            verify_manual_content(
                conn, directory, reference, publication=publication, observer=StandaloneLifetime()
            )
    verify_publication_sidecar(directory, publication, sidecar, observer=StandaloneLifetime())
    assert {path.name for path in directory.parent.iterdir()} == {str(UUID(int=1))}


def test_publication_retry_rejects_a_different_keyword_generation(tmp_path: Path):
    database, sidecar, directory, reference = _capture(tmp_path)
    digest = _publication(database, directory, reference, 2, sidecar=sidecar)
    publication = read_publication(directory, digest)
    with closing(open_sidecar(sidecar)) as keyword, keyword:
        upsert_message_fts(keyword, "b", "later keyword", "")
    with pytest.raises(OracleUnavailable, match=r"content|generation"):
        verify_publication_sidecar(directory, publication, sidecar, observer=StandaloneLifetime())
    with pytest.raises(OracleUnavailable, match="overwritten"):
        _publication(database, directory, reference, 3, sidecar=sidecar)


def test_publication_stream_tampering_cannot_qualify_a_release(tmp_path: Path):
    database, sidecar, directory, reference = _capture(tmp_path)
    digest = _publication(database, directory, reference, 2, sidecar=sidecar)
    publication = read_publication(directory, digest)
    (directory / "publication" / publication.pending.filename).write_bytes(b"changed")
    with pytest.raises(OracleUnavailable, match="corrupt"):
        read_publication(directory, digest)


def _released(database, directory, reference, publication_hash):
    from recall.db.repair import (
        ReleaseCertificate,
        RepairRow,
        TriggerInstruction,
        load_repair_row,
        save_repair_row,
    )
    from recall.db.repair_oracle import read_oracle

    publication = read_publication(directory, publication_hash)
    with duckdb.connect(str(database)) as conn:
        conn.execute("DELETE FROM fts_sidecar_pending")
        original = load_repair_row(conn)
        released = RepairRow.model_validate(
            original.model_dump()
            | {
                "phase": "verified",
                "attempt_no": 2,
                "trigger_id": reference.trigger_id,
                "trigger_kind": "manual",
                "trigger_json": TriggerInstruction(
                    trigger_id=reference.trigger_id, kind="manual", events=()
                ),
                "target_manifest": read_oracle(directory, reference).targets,
                "oracle_ref": f"repair-oracles/{reference.trigger_id}",
                "oracle_hash": reference.manifest_hash,
                "release_certificate": ReleaseCertificate(
                    attempt_no=2,
                    trigger_id=reference.trigger_id,
                    oracle_hash=reference.manifest_hash,
                    publication_hash=publication_hash,
                    keyword_generation=publication.keyword_generation,
                    events=(),
                ),
            }
        )
        save_repair_row(conn, released)
    return released


def test_post_publication_reference_does_not_consume_the_replacement_oracle_slot(tmp_path: Path):
    from recall.db.repair import RepairRow, TriggerInstruction, save_repair_row
    from recall.db.repair_oracle import read_oracle
    from recall.services.repair_oracles import OracleStore

    database, sidecar, directory, reference = _capture(tmp_path)
    digest = _publication(database, directory, reference, 2, sidecar=sidecar)
    released = _released(database, directory, reference, digest)
    store = OracleStore(tmp_path)
    store.reconcile(released, UUID(int=2))
    replacement = tmp_path / "repair-oracles" / str(UUID(int=2))
    with duckdb.connect(str(database)) as conn:
        next_reference = capture_oracle(
            conn, replacement, UUID(int=2), sidecar=sidecar, observer=StandaloneLifetime()
        )
        required = RepairRow.model_validate(
            released.model_dump()
            | {
                "phase": "required",
                "trigger_id": UUID(int=2),
                "trigger_kind": "manual",
                "trigger_json": TriggerInstruction(
                    trigger_id=UUID(int=2), kind="manual", events=()
                ),
                "target_manifest": read_oracle(replacement, next_reference).targets,
                "oracle_ref": f"repair-oracles/{UUID(int=2)}",
                "oracle_hash": next_reference.manifest_hash,
                "release_certificate": None,
            }
        )
        save_repair_row(conn, required)
    store.reconcile(required)
    assert not directory.exists()
    assert (replacement / "manifest.json").is_file()
    assert tuple(path.name for path in replacement.parent.iterdir()) == (str(UUID(int=2)),)


def test_matched_recreation_backup_includes_the_pinned_publication_reference(tmp_path: Path):
    from dataclasses import replace

    from recall.core.config import AppConfig
    from recall.services.recreation import consistent_backup

    database, sidecar, directory, reference = _capture(tmp_path)
    digest = _publication(database, directory, reference, 2, sidecar=sidecar)
    released = _released(database, directory, reference, digest)
    config = replace(AppConfig.load(), db_path=database, data_dir=tmp_path)
    with duckdb.connect(str(database)) as conn:
        backup = consistent_backup(
            conn, config, kind="recreate", recovery=True, observer=StandaloneLifetime()
        )
    publication = read_publication(backup / released.oracle_ref, digest)
    assert publication.attempt_no == 2 and publication.oracle_hash == reference.manifest_hash
    with duckdb.connect(str(backup / "recall.duckdb"), read_only=True) as copied:
        verify_manual_content(
            copied,
            backup / released.oracle_ref,
            reference,
            publication=publication,
            observer=StandaloneLifetime(),
        )
    verify_publication_sidecar(
        backup / released.oracle_ref,
        publication,
        backup / "recall.fts.sqlite",
        observer=StandaloneLifetime(),
    )


@pytest.mark.parametrize("field,value", [("version", True), ("sidecar_present", 1)])
def test_publication_parser_rejects_coerced_durable_values(
    tmp_path: Path, field: str, value: object
):
    import json

    from pydantic import ValidationError
    from recall.db.repair_oracle import PublicationOracle

    database, sidecar, directory, reference = _capture(tmp_path)
    digest = _publication(database, directory, reference, 2, sidecar=sidecar)
    payload = read_publication(directory, digest).model_dump(mode="json")
    if field == "sidecar_present":
        payload["sidecar"]["present"] = value
    else:
        payload[field] = value
    with pytest.raises(ValidationError):
        PublicationOracle.model_validate_json(json.dumps(payload))
