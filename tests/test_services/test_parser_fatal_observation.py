"""Raw preparation observes terminal failures before producing diagnostic data."""

from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall.db.source_files import SourceFile
from recall.parsers.codex import CodexParser
from recall.services.coordinator import prepare_raw_sources


def test_raw_parser_fatal_propagates_before_prepared_error_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rollout-capture.jsonl"
    path.write_text('{"type":"session_meta","payload":{"id":"capture"}}\n')
    parser = CodexParser(roots=(tmp_path,))
    item = SourceFile(
        source_path=str(path),
        source="codex",
        root_path=str(tmp_path),
        desired_generation=1,
        committed_generation=0,
        first_pending_at=0.0,
        retry_count=0,
        next_retry_at=0.0,
        last_serviced_seq=0,
        missing=False,
    )

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise duckdb.FatalException("Failed to delete all rows from index in raw parser")

    monkeypatch.setattr(parser, "parse", fail)
    with pytest.raises(duckdb.FatalException, match="in raw parser"):
        prepare_raw_sources((item,), {"codex": parser})
