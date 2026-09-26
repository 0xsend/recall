"""A no-op `recall index` must cost reads, not a writer turn per unchanged file.

REQ-INDEX-023 / REQ-INDEX-024: the request path observed every
captured path individually, so a pass over a corpus where nothing changed took
one writer call and two catalog reads per file, and told the client about each
one. The contract these tests hold is a scaling contract: growing the corpus
while changing nothing must not grow the writer calls, the catalog reads, or
the progress frames.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from lane_harness import FIXTURES, LANE_NOW, Lane, build_lane

OBSERVATION_LABEL = "source observation commit"


@pytest.fixture
def lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Lane]:
    lane = build_lane(tmp_path, monkeypatch)
    lane.server._config = replace(
        lane.server._config,
        daemon=replace(lane.server._config.daemon, embed_idle_session=0),
        compaction=replace(lane.server._config.compaction, auto_trigger=False),
    )
    yield lane
    asyncio.run(lane.server.stop())


def install_indexed_sessions(lane: Lane, count: int) -> list[Path]:
    """Add `count` more transcripts to the watched root and index them all."""
    import os

    source = FIXTURES / "claude_code" / "live_mid_tool.jsonl"
    text = source.read_text(encoding="utf-8")
    projects = lane.watched.parent
    installed: list[Path] = []
    for index in range(count):
        body = (
            text.replace("live-mid-tool", f"extra-{index}")
            .replace('"mid-', f'"e{index}-')
            .replace("msg_mid_", f"msg_e{index}_")
            .replace("toolu_mid_", f"toolu_e{index}_")
        )
        path = projects / f"extra-{index}.jsonl"
        path.write_text(body, encoding="utf-8")
        os.utime(path, (LANE_NOW.timestamp(), LANE_NOW.timestamp()))
        installed.append(path)

    async def reconcile() -> None:
        for path in installed:
            await lane.server.index_session_now(path)

    asyncio.run(reconcile())
    return installed


class RecordingClient:
    """The progress frames one request sends, in order."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def send_progress(self, **kwargs: Any) -> None:
        self.frames.append(kwargs)


def noop_pass(lane: Lane) -> tuple[list[str], int, list[dict[str, Any]], Any]:
    """Run one plain index over an unchanged corpus, recording what it cost."""
    server = lane.server
    labels: list[str] = []
    reads = 0
    writer_call = server._writer_call
    run_readonly = server._run_readonly

    async def recording_writer(operation: Any, label: str, **kwargs: Any) -> Any:
        labels.append(label)
        return await writer_call(operation, label, **kwargs)

    async def recording_readonly(fn: Any) -> Any:
        nonlocal reads
        reads += 1
        return await run_readonly(fn)

    client = RecordingClient()
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(server, "_writer_call", recording_writer)
        patched.setattr(server, "_run_readonly", recording_readonly)
        summary = asyncio.run(server._handle_index({"embed": False}, cast(Any, client)))
    return labels, reads, client.frames, summary


def test_a_noop_index_never_takes_a_writer_turn_for_an_unchanged_file(lane: Lane) -> None:
    """REQ-INDEX-023: unchanged sources are compared in memory, never re-observed."""
    install_indexed_sessions(lane, 6)
    labels, _reads, _frames, summary = noop_pass(lane)

    assert summary.total == 8 and summary.skipped == 8 and summary.indexed == 0
    assert OBSERVATION_LABEL not in labels, (
        f"an unchanged corpus took {labels.count(OBSERVATION_LABEL)} per-path writer turns"
    )


def test_noop_index_cost_does_not_grow_with_the_unchanged_corpus(lane: Lane) -> None:
    """REQ-INDEX-023: writer calls and catalog reads scale with change, not size."""
    small_labels, small_reads, _frames, small = noop_pass(lane)
    assert small.total == 2 and small.skipped == 2

    install_indexed_sessions(lane, 6)
    large_labels, large_reads, _frames, large = noop_pass(lane)
    assert large.total == 8 and large.skipped == 8

    assert large_labels == small_labels, (
        "quadrupling the unchanged corpus changed the writer calls: "
        f"{sorted(set(large_labels) - set(small_labels))} "
        f"({len(large_labels)} vs {len(small_labels)} calls)"
    )
    assert large_reads == small_reads, (
        f"catalog comparison read per file: {large_reads} reads for 8 files vs {small_reads} for 2"
    )


def test_noop_index_progress_is_not_emitted_once_per_unchanged_file(lane: Lane) -> None:
    """REQ-INDEX-024: progress is coalesced per captured batch, not per file."""
    _labels, _reads, small_frames, _small = noop_pass(lane)

    extra = install_indexed_sessions(lane, 6)
    _labels, _reads, large_frames, large = noop_pass(lane)

    assert large.skipped == 8
    assert large_frames[-1]["status"] == "done"
    assert large_frames[-1]["inventory_complete"] is True

    unchanged = {str(path) for path in [lane.watched, lane.unwatched, *extra]}
    named = [frame.get("path") for frame in large_frames if frame.get("path") in unchanged]
    assert not named, f"progress named {len(named)} unchanged transcripts"
    assert len(large_frames) == len(small_frames), (
        f"progress grew with the unchanged corpus: {len(large_frames)} frames "
        f"for 8 files vs {len(small_frames)} for 2"
    )


def test_a_changed_transcript_still_indexes_through_the_batched_comparison(lane: Lane) -> None:
    """The cheap comparison must not make a real append invisible."""
    install_indexed_sessions(lane, 6)
    lane.append(lane.watched, "batched comparison content", uuid="batched-1")

    summary = asyncio.run(lane.server._handle_index({"embed": False}, None))

    assert summary.indexed == 1, "the appended transcript was not re-indexed"
    assert summary.skipped == 7
    assert lane.server._get_conn().execute(
        "SELECT COUNT(*) FROM message_state WHERE content = 'batched comparison content'"
    ).fetchone() == (1,)


def test_a_new_transcript_is_discovered_without_per_file_observation(lane: Lane) -> None:
    """A source the catalog has never seen is still captured, observed and served."""
    install_indexed_sessions(lane, 2)
    fresh = lane.watched.parent / "fresh.jsonl"
    shutil.copy(FIXTURES / "claude_code" / "live_end_turn.jsonl", fresh)
    fresh.write_text(
        fresh.read_text(encoding="utf-8")
        .replace("live-end-turn", "fresh-session")
        .replace('"end-', '"fresh-'),
        encoding="utf-8",
    )

    summary = asyncio.run(lane.server._handle_index({"embed": False}, None))

    assert summary.total == 5
    assert summary.indexed == 1, "a newly discovered transcript was skipped"
    assert summary.skipped == 4
