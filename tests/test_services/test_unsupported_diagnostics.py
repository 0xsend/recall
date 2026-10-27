from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from recall.core.config import AppConfig, SourceConfig
from recall.core.models import ParseDiagnostic
from recall.db.schema import ensure_schema
from recall.db.source_files import SourceCatalog, SourceSignature, source_key
from recall.parsers import get_parser
from recall.parsers.kimi_code import KimiCodeParser
from recall.parsers.protocol import SessionParser
from recall.parsers.revision import parser_revision
from recall.services.coordinator import reconciliation_status, select_raw_sources
from recall.services.reconciler import FairScheduler
from recall.services.unsupported_diagnostics import (
    DIAGNOSTIC_DETAIL_LIMIT,
    diagnostic_payload,
    reopen_unsupported_parks,
    unsupported_summary,
)

from tests.test_parsers.test_harness_metadata import CASES, FIXTURES


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
    catalog.defer_until_changed(
        "kimi_code",
        detailed,
        "unsupported",
        diagnostic_payload(
            [ParseDiagnostic("unsupported_record", 8, "record: 'swarm_mode.enter'")]
        ),
    )
    catalog.defer_until_changed(
        "kimi_code", opaque, "unsupported", {"records": ["unsupported_record"]}
    )

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


def test_deleted_parked_source_counts_as_missing_not_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transcript parked on an older parser and then deleted can never be
    reparsed. Counting it as unsupported raised a parser-gap warning whose
    summary, which lists only files still on disk, had nothing to name.
    """
    conn, catalog, _ = _catalog(tmp_path, monkeypatch)
    root = tmp_path / ".codex" / "sessions"
    root.mkdir(parents=True)
    deleted = str(root / "rollout-2026-10-03T22-29-50-01a104f5-f490-78d1-be43-05882434f41e.jsonl")
    catalog.observe("codex", str(root), deleted, _signature(1))
    # The diagnostic a pre-REQ-PARSE-032 daemon stored for this transcript.
    catalog.defer_until_changed(
        "codex",
        deleted,
        "unsupported",
        {
            "records": [
                {
                    "kind": "unsupported_record",
                    "detail": "record: 'retained_context'",
                    "byte_offset": 768845,
                }
            ]
        },
    )
    catalog.mark_missing([source_key("codex", deleted)])

    status = reconciliation_status(AppConfig.load(), conn=conn)
    rows = status["coverage"]
    assert isinstance(rows, list)
    coverage = next(row for row in rows if row["root_path"] == str(root))
    assert coverage["unsupported"] == 0
    assert coverage["missing"] == 1
    assert status["unsupported_summary"] == {"files": 0, "groups": [], "truncated": False}


def test_reopen_retries_legacy_parks_but_not_current_parser_or_missing_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, catalog, root = _catalog(tmp_path, monkeypatch)
    opaque = str(root / "opaque.jsonl")
    stalled = str(root / "stalled.jsonl")
    advanced = str(root / "advanced.jsonl")
    missing = str(root / "missing.jsonl")
    for inode, path in enumerate((opaque, stalled, advanced, missing), start=1):
        catalog.observe("kimi_code", str(root), path, _signature(inode))
    detailed = diagnostic_payload(
        [ParseDiagnostic("unsupported_record", 1, "record: 'file_history.tracked'")]
    )
    # An older build parked at the record; this one acknowledges the whole file.
    assert catalog.acknowledge("kimi_code", advanced, 1, 5, "0" * 64)
    catalog.defer_until_changed(
        "kimi_code", opaque, "unsupported", {"records": ["unsupported_record"]}
    )
    catalog.defer_until_changed("kimi_code", stalled, "unsupported", detailed)
    revision = parser_revision(KimiCodeParser)
    catalog.defer_unsupported("kimi_code", advanced, detailed, parser_revision=revision)
    catalog.defer_until_changed("kimi_code", missing, "unsupported", detailed)
    catalog.mark_missing([source_key("kimi_code", missing)])

    assert reopen_unsupported_parks(conn, 10.0) == 2
    assert _retry_at(conn, opaque) == 0
    assert _retry_at(conn, stalled) == 0
    assert _retry_at(conn, advanced) > 1e300
    assert _retry_at(conn, missing) > 1e300
    assert reopen_unsupported_parks(conn, 10.0) == 0


def test_raw_scheduling_leaves_opaque_parks_parked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, catalog, root = _catalog(tmp_path, monkeypatch)
    opaque = str(root / "opaque.jsonl")
    catalog.observe("kimi_code", str(root), opaque, _signature(1))
    catalog.defer_until_changed(
        "kimi_code", opaque, "unsupported", {"records": ["unsupported_record"]}
    )

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


@pytest.mark.parametrize("park_revision", [None, "older-parser-build"])
@pytest.mark.parametrize(
    ("parser", "fixture"),
    CASES,
)
def test_parser_upgrade_retries_detailed_parks_without_reparsing_supported_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parser: SessionParser,
    fixture: str,
    park_revision: str | None,
) -> None:
    from recall.services.coordinator import (
        capture_path,
        commit_prepared_raw_sources,
        observe_path,
        prepare_raw_sources,
    )

    conn, catalog, _ = _catalog(tmp_path, monkeypatch)
    config = AppConfig.load()
    parser = get_parser(parser.source, {parser.source.value: SourceConfig(roots=(tmp_path,))})
    paths = [tmp_path / name / fixture for name in ("park", "supported")]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_bytes((FIXTURES / fixture).read_bytes())
        item = observe_path(parser, capture_path(parser, path), conn=conn)
        commit_prepared_raw_sources(
            prepare_raw_sources((item,), {parser.source.value: parser}), config, conn=conn
        )
    parked, supported = map(str, paths)
    supported_before = catalog.get(parser.source.value, supported)
    payload: dict[str, object] = {
        "records": [{"kind": "unsupported_record", "detail": "old metadata", "byte_offset": 0}]
    }
    if park_revision is not None:
        payload["parser_revision"] = park_revision
    # A complete, detailed park from the database produced by the older build.
    catalog.defer_until_changed(parser.source.value, parked, "unsupported", payload)
    assert unsupported_summary(conn)["files"] == 1
    assert reopen_unsupported_parks(conn, 10.0) == 1
    assert reopen_unsupported_parks(conn, 10.0) == 0
    selected = select_raw_sources(
        {parser.source.value: parser},
        conn=conn,
        scheduler=FairScheduler(clock=lambda: 10.0),
        active_paths=set(),
    )
    assert [item.source.source_path for item in selected] == [parked]
    commit_prepared_raw_sources(
        prepare_raw_sources(tuple(item.source for item in selected), {parser.source.value: parser}),
        config,
        conn=conn,
    )
    assert unsupported_summary(conn)["files"] == 0
    assert catalog.get(parser.source.value, supported) == supported_before
    assert reopen_unsupported_parks(conn, 10.0) == 0
    assert conn.execute("SELECT count(*) FROM messages").fetchone() == (
        6 if "replacement-history" in fixture else 4,
    )
    conn.close()


@pytest.mark.parametrize("torn", [False, True])
def test_still_unsupported_reparks_once_per_parser_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torn: bool
) -> None:
    import json

    from recall.parsers.codex import CodexParser
    from recall.services.coordinator import (
        capture_path,
        commit_prepared_raw_sources,
        observe_path,
        prepare_raw_sources,
    )

    conn, catalog, _ = _catalog(tmp_path, monkeypatch)
    config = AppConfig.load()
    path = tmp_path / "rollout-unknown.jsonl"
    path.write_bytes(
        (FIXTURES / "codex/rollout-additional-tools.jsonl")
        .read_bytes()
        .replace(b"additional_tools", b"future_metadata")
        + (b'{"type":' if torn else b"")
    )
    parser = CodexParser(roots=(tmp_path,))
    item = observe_path(parser, capture_path(parser, path), conn=conn)
    commit_prepared_raw_sources(prepare_raw_sources((item,), {"codex": parser}), config, conn=conn)
    # A completed attempt, even with a torn suffix, never loops on startup.
    assert reopen_unsupported_parks(conn, 10.0) == 0
    current = catalog.get("codex", str(path))
    assert current is not None and current.last_error == "unsupported"
    assert current.diagnostics is not None
    assert json.loads(current.diagnostics)["parser_revision"] == parser_revision(CodexParser)
    # Simulate a park left by a previous build, keeping its detailed records.
    payload = json.loads(current.diagnostics)
    payload["parser_revision"] = "older-parser-build"
    catalog.defer_until_changed("codex", str(path), "unsupported", payload)
    assert reopen_unsupported_parks(conn, 10.0) == 1
    selected = select_raw_sources(
        {"codex": parser},
        conn=conn,
        scheduler=FairScheduler(clock=lambda: 10.0),
        active_paths=set(),
    )
    assert len(selected) == 1
    commit_prepared_raw_sources(
        prepare_raw_sources(tuple(item.source for item in selected), {"codex": parser}),
        config,
        conn=conn,
    )
    assert reopen_unsupported_parks(conn, 10.0) == 0
    assert (
        select_raw_sources(
            {"codex": parser},
            conn=conn,
            scheduler=FairScheduler(clock=lambda: 10.0),
            active_paths=set(),
        )
        == ()
    )
    assert unsupported_summary(conn)["files"] == 1
    assert conn.execute(
        "SELECT content FROM messages JOIN message_state ON message_id = id ORDER BY idx"
    ).fetchall() == [
        ("Parser orchard question",),
        ("Parser orchard answer",),
    ]
    conn.close()
