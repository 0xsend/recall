from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig
from recall.core.models import ParseDiagnostic
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog, SourceSignature
from recall.services.coordinator import reconciliation_status, select_raw_sources
from recall.services.reconciler import FairScheduler
from recall.services.unsupported_diagnostics import (
    DIAGNOSTIC_DETAIL_LIMIT,
    diagnostic_payload,
    reopen_opaque_unsupported,
    unsupported_summary,
)


def test_diagnostic_payload_keeps_detail_and_bounds_it() -> None:
    payload = diagnostic_payload(
        [
            ParseDiagnostic("unsupported_record", 12, "record: 'swarm_mode.enter'"),
            ParseDiagnostic("unsupported_record", 40, "x" * (DIAGNOSTIC_DETAIL_LIMIT + 10)),
        ]
    )
    records = payload["records"]
    assert isinstance(records, list)
    assert records[0] == {
        "kind": "unsupported_record",
        "detail": "record: 'swarm_mode.enter'",
        "byte_offset": 12,
    }
    assert records[1] == {
        "kind": "unsupported_record",
        "detail": "x" * (DIAGNOSTIC_DETAIL_LIMIT - 3) + "...",
        "byte_offset": 40,
    }


def test_summary_names_detail_and_marks_kind_only_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, catalog, root = _catalog(tmp_path, monkeypatch)
    detailed = str(root / "detailed.jsonl")
    opaque = str(root / "opaque.jsonl")
    catalog.observe("kimi_code", str(root), detailed, _signature(1))
    catalog.observe("kimi_code", str(root), opaque, _signature(2))
    catalog.defer_unsupported(
        "kimi_code",
        detailed,
        diagnostic_payload(
            [ParseDiagnostic("unsupported_record", 8, "record: 'swarm_mode.enter'")]
        ),
    )
    catalog.defer_unsupported("kimi_code", opaque, {"records": ["unsupported_record"]})

    summary = unsupported_summary(conn)
    assert summary == {
        "files": 2,
        "groups": [
            {
                "source": "kimi_code",
                "detail": "record: 'swarm_mode.enter'",
                "detail_omitted": False,
                "files": 1,
                "sample_paths": [detailed],
            },
            {
                "source": "kimi_code",
                "detail": None,
                "detail_omitted": True,
                "files": 1,
                "sample_paths": [opaque],
            },
        ],
        "truncated": False,
    }

    status = reconciliation_status(AppConfig.load(), conn=conn)
    assert status["unsupported_summary"] == summary


def test_reopen_targets_only_payloads_without_detail_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, catalog, root = _catalog(tmp_path, monkeypatch)
    opaque = str(root / "opaque.jsonl")
    detailed = str(root / "detailed.jsonl")
    catalog.observe("kimi_code", str(root), opaque, _signature(1))
    catalog.observe("kimi_code", str(root), detailed, _signature(2))
    catalog.defer_unsupported("kimi_code", opaque, {"records": ["unsupported_record"]})
    catalog.defer_unsupported(
        "kimi_code",
        detailed,
        diagnostic_payload(
            [ParseDiagnostic("unsupported_record", 1, "record: 'file_history.tracked'")]
        ),
    )

    assert reopen_opaque_unsupported(conn, 10.0) == 1
    assert _retry_at(conn, opaque) == 0
    assert _retry_at(conn, detailed) > 1e300
    assert reopen_opaque_unsupported(conn, 10.0) == 0


def test_raw_scheduling_leaves_opaque_parks_parked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, catalog, root = _catalog(tmp_path, monkeypatch)
    opaque = str(root / "opaque.jsonl")
    catalog.observe("kimi_code", str(root), opaque, _signature(1))
    catalog.defer_unsupported("kimi_code", opaque, {"records": ["unsupported_record"]})

    from recall.parsers import all_parsers

    config = AppConfig.load()
    selected = select_raw_sources(
        {parser.source.value: parser for parser in all_parsers(config.sources)},
        conn=conn,
        scheduler=FairScheduler(clock=lambda: 10.0),
        active_paths=set(),
    )
    assert selected == ()
    assert _retry_at(conn, opaque) > 1e300


def _catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[duckdb.DuckDBPyConnection, SourceCatalog, Path]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / ".kimi-code" / "sessions"
    root.mkdir(parents=True)
    config = AppConfig.load()
    conn = duckdb.connect(":memory:")
    ensure_schema(conn, embed_dim=config.embedding.dimensions)
    return conn, SourceCatalog(conn, clock=lambda: 10.0), root


def _signature(inode: int) -> SourceSignature:
    return SourceSignature(dev=1, inode=inode, ctime_ns=3, mtime_ns=4, size=5)


def _retry_at(conn: duckdb.DuckDBPyConnection, path: str) -> float:
    row = conn.execute(
        "SELECT next_retry_at FROM source_files WHERE source_path = ?",
        [path],
    ).fetchone()
    assert row is not None
    return float(row[0])
