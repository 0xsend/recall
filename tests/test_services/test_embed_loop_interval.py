"""How long the embed loop waits after each kind of cycle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from lane_harness import lane_config
from recall.services.embed_phase import PendingEmbeds
from recall.services.rpc_server import next_embed_interval


def _config(tmp_path: Path):
    config = lane_config(tmp_path)
    return replace(config, daemon=replace(config.daemon, embed_interval=5, embed_backoff=30))


def test_a_drain_that_left_eligible_work_re_arms_at_once(tmp_path: Path) -> None:
    left = PendingEmbeds(message_count=40, tool_call_count=2, session_ids=["next"])
    assert next_embed_interval(3, left, _config(tmp_path)) == 0.01


def test_a_drain_that_emptied_the_roster_waits_the_normal_interval(tmp_path: Path) -> None:
    """The cycle after a full drain finds nothing; running it at once only replaced
    `drained` as the reported outcome within milliseconds of the batch."""
    nothing = PendingEmbeds(message_count=0, tool_call_count=0, session_ids=[])
    assert next_embed_interval(3, nothing, _config(tmp_path)) == 5


def test_an_empty_cycle_waits_the_normal_interval(tmp_path: Path) -> None:
    assert next_embed_interval(0, None, _config(tmp_path)) == 5


def test_a_failed_or_deferred_cycle_backs_off(tmp_path: Path) -> None:
    left = PendingEmbeds(message_count=40, tool_call_count=2, session_ids=["next"])
    assert next_embed_interval(-1, left, _config(tmp_path)) == 30
