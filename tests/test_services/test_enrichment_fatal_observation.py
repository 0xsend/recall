"""Enrichment capacity/stale fallback must retain an originating fatal graph."""

from pathlib import Path
from typing import Any

import duckdb
import pytest
from recall.db.schema import ensure_schema
from recall.services import embed_phase
from recall.services.embed_phase import EmbedPhaseState, PendingEmbeds, prepare_embed_cycle
from recall.services.enrichment_inputs import EnrichmentCapacityError
from recall.services.indexer import _write_session

from tests.keyword_turns import without_keyword_index
from tests.test_services.test_embed_phase import (
    RecordingBackend,
    _allow_embedding,
    _app_config,
    _idle_session,
)


def test_capacity_fallback_propagates_a_wrapped_origin_before_error_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _app_config(tmp_path)
    state = EmbedPhaseState(_config=config)
    original = duckdb.FatalException("Failed to delete all rows from index in selection")
    monkeypatch.setattr(
        embed_phase,
        "find_eligible_pending_embeds",
        lambda *args, **kwargs: PendingEmbeds(1, 0, ["s1"]),
    )

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise EnrichmentCapacityError("selection failed") from original

    monkeypatch.setattr(embed_phase, "select_pending_targets", fail)
    with duckdb.connect() as conn, pytest.raises(EnrichmentCapacityError) as caught:
        prepare_embed_cycle(config, state, conn=conn)
    assert caught.value.__cause__ is original


def test_stale_validation_propagates_wrapped_origin_before_discarding_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.core.models import TailFacts

    config = _app_config(tmp_path)
    state = EmbedPhaseState(_config=config)
    _allow_embedding(monkeypatch, state, RecordingBackend())
    original = duckdb.FatalException("Failed to delete all rows from index in validation")
    with duckdb.connect() as conn:
        ensure_schema(conn)
        _write_session(
            conn, _idle_session("s1", "hello"), tail_facts=TailFacts(), turn=without_keyword_index()
        )
        prepared = embed_phase.generate_prepared_embed_cycle(
            config, state, prepare_embed_cycle(config, state, conn=conn)
        )
        assert prepared.generated_session_ids == {"s1"}

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("cannot validate") from original

        monkeypatch.setattr(embed_phase, "read_enrichment_session", fail)
        with pytest.raises(ValueError, match="cannot validate") as caught:
            embed_phase.commit_prepared_embed_cycle(prepared, conn=conn)
        assert caught.value.__cause__ is original
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (0,)
