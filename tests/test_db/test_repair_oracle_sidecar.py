"""An interrupted attempt's reference includes the actual keyword store."""

from contextlib import closing
from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.db.fts_sidecar import open_sidecar, upsert_message_fts
from recall.db.lease import StandaloneLifetime
from recall.db.repair_oracle import OracleUnavailable, capture_oracle, read_oracle, verify_oracle


def test_saved_reference_includes_keyword_generation_and_membership(tmp_path: Path) -> None:
    path = tmp_path / "recall.fts.sqlite"
    with closing(open_sidecar(path)) as keywords:
        upsert_message_fts(keywords, "a", "kumquat exact content", "")
        keywords.commit()
    with duckdb.connect() as conn:
        conn.execute("CREATE TABLE witness (id VARCHAR)")
        conn.execute("INSERT INTO witness VALUES ('a')")
        reference = capture_oracle(
            conn, tmp_path / "oracle", UUID(int=1), sidecar=path, observer=StandaloneLifetime()
        )
        manifest = read_oracle(tmp_path / "oracle", reference)
        assert manifest.sidecar is not None
        verify_oracle(
            conn, tmp_path / "oracle", reference, sidecar=path, observer=StandaloneLifetime()
        )
        with closing(open_sidecar(path)) as keywords:
            upsert_message_fts(keywords, "b", "new acknowledged keyword", "")
            keywords.commit()
        with pytest.raises(OracleUnavailable):
            verify_oracle(
                conn, tmp_path / "oracle", reference, sidecar=path, observer=StandaloneLifetime()
            )
