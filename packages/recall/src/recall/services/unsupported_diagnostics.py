"""Bounded parser-diagnostic payload and the status summary an agent can file from.

The catalog parks a file at the first unsupported record. The payload must name
that record. A kind-only list cannot be filed or fixed without paging the
catalog and opening the transcript.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

import duckdb

from recall.core.models import ParseDiagnostic

DIAGNOSTIC_RECORD_LIMIT = 32
DIAGNOSTIC_DETAIL_LIMIT = 160
UNSUPPORTED_SUMMARY_FILE_LIMIT = 256
UNSUPPORTED_SUMMARY_GROUP_LIMIT = 8
UNSUPPORTED_SUMMARY_PATH_LIMIT = 3


@dataclass
class UnsupportedGroup:
    source: str
    detail: str | None
    detail_omitted: bool
    files: int = 0
    sample_paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "detail": self.detail,
            "detail_omitted": self.detail_omitted,
            "files": self.files,
            "sample_paths": self.sample_paths,
        }


def diagnostic_payload(diagnostics: Sequence[ParseDiagnostic]) -> dict[str, object]:
    """Serialize the first bounded slice of parser diagnostics.

    ``detail`` and ``byte_offset`` are the filing facts. A later reader treats a
    record without ``detail`` as an older payload and does not invent one.
    """
    records: list[dict[str, object]] = []
    for diagnostic in diagnostics[:DIAGNOSTIC_RECORD_LIMIT]:
        detail = diagnostic.detail.strip()
        # An empty detail would read as an older kind-only payload in status.
        assert detail, f"{diagnostic.kind} diagnostic has no detail"
        if len(detail) > DIAGNOSTIC_DETAIL_LIMIT:
            detail = detail[: DIAGNOSTIC_DETAIL_LIMIT - 3] + "..."
        assert diagnostic.byte_offset >= 0
        records.append(
            {
                "kind": diagnostic.kind,
                "detail": detail,
                "byte_offset": diagnostic.byte_offset,
            }
        )
    return {"records": records}


def reopen_opaque_unsupported(conn: duckdb.DuckDBPyConnection, now: float) -> int:
    """Make unsupported parks whose payload names no record eligible for one rewrite.

    A parked file is not selected again until its bytes or parser change.
    Rewriting the payload is that one extra service. A payload whose first
    record carries ``detail`` stays parked, so a second call is a no-op. Kind
    lists from older builds hold strings, where the ``detail`` path is NULL.
    """
    rows = conn.execute(
        """UPDATE source_files
              SET next_retry_at = 0
            WHERE last_error = 'unsupported'
              AND NOT missing
              AND next_retry_at > ?
              AND json_extract_string(diagnostics, '$.records[0].detail') IS NULL
            RETURNING source_path""",
        [now],
    ).fetchall()
    return len(rows)


def unsupported_summary(conn: duckdb.DuckDBPyConnection) -> dict[str, object]:
    """Aggregate parked unsupported files into a bounded filing summary."""
    rows = conn.execute(
        """SELECT source, source_path, CAST(diagnostics AS VARCHAR)
             FROM source_files
            WHERE last_error = 'unsupported' AND NOT missing
            ORDER BY source, source_path
            LIMIT ?""",
        [UNSUPPORTED_SUMMARY_FILE_LIMIT + 1],
    ).fetchall()
    truncated = len(rows) > UNSUPPORTED_SUMMARY_FILE_LIMIT
    files = rows[:UNSUPPORTED_SUMMARY_FILE_LIMIT]
    groups: OrderedDict[tuple[str, str, bool], UnsupportedGroup] = OrderedDict()
    for source, source_path, raw in files:
        for detail, omitted in _details(raw):
            key = (str(source), detail, omitted)
            group = groups.get(key)
            if group is None:
                group = UnsupportedGroup(
                    source=str(source),
                    detail=None if omitted else detail,
                    detail_omitted=omitted,
                )
                groups[key] = group
            group.files += 1
            if len(group.sample_paths) < UNSUPPORTED_SUMMARY_PATH_LIMIT:
                group.sample_paths.append(str(source_path))
    ordered = sorted(groups.values(), key=lambda group: (-group.files, group.source))
    return {
        "files": len(files),
        "groups": [group.as_dict() for group in ordered[:UNSUPPORTED_SUMMARY_GROUP_LIMIT]],
        "truncated": truncated or len(ordered) > UNSUPPORTED_SUMMARY_GROUP_LIMIT,
    }


def _details(raw: object) -> list[tuple[str, bool]]:
    payload = _decode(raw)
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        return [("unsupported_record", True)]
    found: list[tuple[str, bool]] = []
    seen: set[tuple[str, bool]] = set()
    for record in records[:DIAGNOSTIC_RECORD_LIMIT]:
        if isinstance(record, str):
            item = (record or "unsupported_record", True)
        elif isinstance(record, dict):
            detail = record.get("detail")
            if isinstance(detail, str) and detail.strip():
                item = (detail.strip(), False)
            else:
                kind = record.get("kind")
                item = (str(kind) if isinstance(kind, str) and kind else "unsupported_record", True)
        else:
            continue
        if item not in seen:
            seen.add(item)
            found.append(item)
    return found or [("unsupported_record", True)]


def _decode(raw: object) -> object:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
