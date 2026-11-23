"""Bounded requests and detached provenance for an owned poll snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from recall.core.config import AppConfig
from recall.core.rpc_params import (
    BashStatsParams,
    ListBySessionParams,
    ListParams,
    Params,
    SearchFilteredParams,
    SearchParams,
    ShowParams,
    SkillStatsParams,
    StatsBashFilteredParams,
    StatsFilteredParams,
    StatsUsageFilteredParams,
    UsageStatsParams,
)


class SearchRead(Params):
    method: Literal["recall.search"]
    params: SearchParams


class SearchFilteredRead(Params):
    method: Literal["recall.search_filtered"]
    params: SearchFilteredParams


class ListRead(Params):
    method: Literal["recall.list"]
    params: ListParams


class SessionListRead(Params):
    method: Literal["recall.list_by_session"]
    params: ListBySessionParams


class ShowRead(Params):
    method: Literal["recall.show"]
    params: ShowParams


class StatsRead(Params):
    method: Literal["recall.stats", "recall.stats_tools", "recall.stats_tokens"]
    params: Params


class StatsFilteredRead(Params):
    method: Literal[
        "recall.stats_filtered",
        "recall.stats_tools_filtered",
        "recall.stats_tokens_filtered",
    ]
    params: StatsFilteredParams


class BashRead(Params):
    method: Literal["recall.stats_bash"]
    params: BashStatsParams


class BashFilteredRead(Params):
    method: Literal["recall.stats_bash_filtered"]
    params: StatsBashFilteredParams


class UsageRead(Params):
    method: Literal["recall.stats_usage"]
    params: UsageStatsParams


class UsageFilteredRead(Params):
    method: Literal["recall.stats_usage_filtered"]
    params: StatsUsageFilteredParams


class SkillsRead(Params):
    method: Literal["recall.stats_skills"]
    params: SkillStatsParams


PollRead = (
    SearchRead
    | SearchFilteredRead
    | ListRead
    | SessionListRead
    | ShowRead
    | StatsRead
    | StatsFilteredRead
    | BashRead
    | BashFilteredRead
    | UsageRead
    | UsageFilteredRead
    | SkillsRead
)
_ADAPTER = TypeAdapter(PollRead)


def poll_read_request(method: str, params: dict[str, Any]) -> PollRead | None:
    try:
        request = _ADAPTER.validate_python({"method": method, "params": params})
    except ValidationError:
        return None
    if isinstance(request, ShowRead) and request.params.fresh:
        return None
    return request


@dataclass
class PollSnapshotSelected(Exception):
    """Pre-submission admission bound to the discovered service and database scope."""

    config: AppConfig
    request: PollRead
    service_name: str


@dataclass(frozen=True)
class PollSnapshot:
    data: Any
    last_successful_at: datetime | None
    coverage: dict[str, Any] | None
