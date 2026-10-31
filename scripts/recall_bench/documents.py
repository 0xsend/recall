"""Typed boundary for saved daemon-run outcome documents, including early refusals."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Number = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0)]


class Record(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="allow", allow_inf_nan=False)


class Checkout(Record):
    head: str | None
    dirty: bool = False
    dirty_hash: str | None = None
    error: str | None = None


class Fixture(Record):
    payload: dict[str, str]
    pool_files: Count = 0
    pool_bytes: Count = 0
    pool_sha256: str | None = None


class Load(Record):
    load1: Number | None = None


class Scope(Record):
    profile: Literal["idle", "append", "cohort"] | None = None
    build: Checkout | None = None
    tool: Checkout | None = None
    fixture: Fixture | None = None
    host_load_start: Load | None = None
    host_load_end: Load | None = None
    measurements: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def identifiable(self) -> Self:
        if self.profile is None and self.model_fields_set:
            raise ValueError("a nonempty scope must identify its supported profile")
        if self.profile is not None and None in (self.build, self.tool, self.fixture):
            raise ValueError("a run scope requires build, tool and fixture identity")
        return self


class Seconds(Record):
    count: Count
    median: Number | None = None
    p95: Number | None = None
    max: Number | None = None

    @model_validator(mode="after")
    def measured(self) -> Self:
        if self.count and None in (self.median, self.p95, self.max):
            raise ValueError("a nonempty timing sample requires median, p95 and max")
        return self


class Floors(Record):
    freshness_seconds: Seconds | None = None
    call_seconds: Seconds | None = None
    status_seconds_max: Number | None = None
    writer_seconds_max: Number | None = None
    checkpoint_seconds_max: Number | None = None
    rss_bytes_max: Count | None = None


class Step(Record):
    step: str
    seconds: Number | None = None
    floors: Floors | None = None
    freshness_seconds: Seconds | None = None
    appended_bytes: Count | None = None
    appended_lines: Count | None = None
    rss_bytes_max: Count | None = None
    cpu_percent: Number | None = None


class Error(Record):
    kind: str
    message: str | None = None
    reason: str | None = None
    floors: Floors | None = None
    stdout: str | None = None
    stderr: str | None = None


class Leftover(Record):
    path: str
    reason: str


class Document(Record):
    job: Literal["daemon-run"]
    outcome: Literal["passed", "failed", "incomplete"]
    scope: Scope
    steps: list[Step]
    exit_codes: dict[str, int]
    leftovers: list[Leftover]
    error: Error | None = None

    @model_validator(mode="after")
    def completed(self) -> Self:
        if self.outcome != "passed":
            if self.error is None:
                raise ValueError("an unsuccessful run must explain its outcome")
            return self
        if self.error is not None:
            raise ValueError("a passed run cannot have an error")
        scope = self.scope
        if None in (scope.profile, scope.build, scope.tool, scope.fixture):
            raise ValueError("a passed run requires profile, build, tool and fixture identity")
        steps = {step.step: step for step in self.steps}
        required = {"readiness", "settle", "final"}
        required |= {"window"} if scope.profile == "idle" else {"committed", "drain", "verify"}
        if scope.profile == "append":
            required.add("workload")
        elif scope.profile == "cohort":
            required.add("phase")
        if not required <= steps.keys() or steps["final"].floors is None:
            raise ValueError("a passed run requires its profile's completed measurements")
        if any(steps[name].seconds is None for name in ("readiness", "settle")):
            raise ValueError("a passed run requires measured readiness and settle times")
        floors = steps["final"].floors
        assert floors is not None
        if floors.rss_bytes_max is None:
            raise ValueError("a passed run requires measured RSS")
        if scope.profile == "idle":
            if floors.status_seconds_max is None or steps["window"].cpu_percent is None:
                raise ValueError("a passed idle run requires measured CPU and status latency")
        else:
            if None in (
                floors.freshness_seconds if scope.profile == "append" else floors.call_seconds,
                floors.writer_seconds_max,
                floors.call_seconds,
                steps["final"].appended_bytes,
                steps["final"].appended_lines,
            ):
                raise ValueError("a passed load run requires measured calls, writes and freshness")
        return self
