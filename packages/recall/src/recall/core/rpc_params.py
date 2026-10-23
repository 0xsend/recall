"""Typed JSON parameter objects parsed before RPC handlers run.

Omitted fields keep the handler's defaults. Optional fields accept null only
where the handler uses it to mean an absent filter or a configured default.
Unknown fields keep the method's existing policy.
"""

from pydantic import BaseModel, ConfigDict


class Params(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")


class IndexParams(Params):
    source: str | None = None
    full: bool = False
    recreate: bool = False
    confirmed: bool = False
    recompute_context: bool = False
    since: str | None = None
    only_mode: str | None = None
    project: str | None = None
    root: str | None = None
    host: str | None = None
    embed: bool | None = None
    workers: int | str = "auto"
    verbose: bool = False
    context: str | None = None


class SearchParams(Params):
    query: str
    source: str | None = None
    tool: str | None = None
    session: str | None = None
    limit: int = 20
    mode: str | None = None


class ListParams(Params):
    session: str | None = None
    source: str | None = None
    since: str | None = None
    project: str | None = None
    host: str | None = None
    limit: int = 50


class ShowParams(Params):
    session_id: str
    tools: bool = False
    message_limit: int | None = None
    tail: int | None = None
    after: str | None = None
    fresh: bool = False


class BashStatsParams(Params):
    suggest: bool = False


class UsageStatsParams(Params):
    since: str | None = None


class SkillStatsParams(UsageStatsParams):
    source: list[str] | None = None


class SchedulerParams(Params):
    scheduler: str | None


class StatusParams(Params):
    limit: int = 100
    cursor: str | None = None


class StorageParams(Params):
    # Only storage maintenance has historically rejected unknown parameters.
    model_config = ConfigDict(extra="forbid")
    dry_run: bool = False
    plan_id: str | None = None


class DaemonRunParams(Params):
    once: bool = False
    source: str | None = None
    embed: bool | None = None
    batch_size: int | None = None
    verbose: bool = False
    mode: str | None = None
    interval: int | None = None


class CheckIndexesParams(Params):
    sample: int | None = None


class LiveParams(Params):
    session: str | None = None
    all: bool = False
    source: str | None = None
    project: str | None = None
    host: str | None = None
    limit: int = 50
    cursor: str | None = None
    fresh: bool = False


class LiveMarkParams(Params):
    session: str
    source: str | None = None
    pid: int


class FollowParams(Params):
    session_id: str
    timeout: float = 60.0
    tools: bool = False
    after: str | None = None
