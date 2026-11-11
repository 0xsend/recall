"""Typed boundary for the durable, schema-fenced repair singleton."""

from __future__ import annotations

import json
from typing import Literal, Self
from uuid import UUID

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from recall.core.repair_state import Budget, Phase

MIGRATION_ID = "0041_daemon_index_repair"
FENCE_VERSION = 41
ATTEMPT_MAX = 2**64 - 1
MANIFEST_BYTES_MAX = 4 * 1024 * 1024
TARGETS_MAX = 4096

CREATE_REPAIR_TABLE = """
CREATE TABLE IF NOT EXISTS daemon_index_repair (
    singleton BOOLEAN PRIMARY KEY CHECK (singleton),
    attempt_no UBIGINT NOT NULL DEFAULT 0,
    phase VARCHAR NOT NULL DEFAULT 'idle'
        CHECK (phase IN ('idle', 'required', 'running', 'verified', 'refused')),
    trigger_id UUID,
    trigger_kind VARCHAR CHECK (trigger_kind IN ('fatal', 'probe')),
    signature VARCHAR,
    trigger_json JSON,
    target_manifest JSON,
    oracle_ref VARCHAR,
    oracle_hash VARCHAR,
    success_since_attempt BOOLEAN NOT NULL DEFAULT TRUE,
    last_folded_event_id UUID,
    success_since_failure BOOLEAN NOT NULL DEFAULT TRUE,
    acknowledged_events JSON NOT NULL DEFAULT '[]',
    release_certificate JSON
)
"""


class IndexTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_name: str
    index_name: str
    table_name: str
    sql: str


class TargetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    targets: tuple[IndexTarget, ...] = Field(max_length=TARGETS_MAX)

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        names = {(target.schema_name, target.index_name) for target in self.targets}
        if len(names) != len(self.targets):
            raise ValueError("duplicate index identity in manifest")
        if len(self.model_dump_json().encode()) > MANIFEST_BYTES_MAX:
            raise ValueError("index manifest exceeds 4 MiB")
        return self


class EventAcknowledgement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    content_hash: str
    disposition: Literal["folded", "diagnosed", "verified"]


class RepairRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_no: int = Field(ge=0, le=ATTEMPT_MAX)
    phase: Literal["idle", "required", "running", "verified", "refused"]
    trigger_id: UUID | None
    trigger_kind: Literal["fatal", "probe"] | None
    signature: str | None
    trigger_json: dict[str, object] | None
    target_manifest: TargetManifest | None
    oracle_ref: str | None
    oracle_hash: str | None
    success_since_attempt: bool
    last_folded_event_id: UUID | None
    success_since_failure: bool
    acknowledged_events: tuple[EventAcknowledgement, ...] = Field(max_length=65)
    release_certificate: dict[str, object] | None

    @model_validator(mode="after")
    def validate_instruction(self) -> Self:
        identities = {record.event_id for record in self.acknowledged_events}
        if len(identities) != len(self.acknowledged_events):
            raise ValueError("duplicate acknowledged event identity")
        if self.phase != "idle" and (
            self.trigger_id is None or self.trigger_kind is None or self.trigger_json is None
        ):
            raise ValueError("repair phase lacks its trigger instruction")
        if self.phase in {"running", "verified", "refused"}:
            if self.target_manifest is None or self.oracle_ref is None or self.oracle_hash is None:
                raise ValueError("armed/settled repair lacks its saved manifest or oracle")
            if self.attempt_no == 0:
                raise ValueError("armed/settled repair lacks an attempt")
        if self.phase == "running" and self.success_since_attempt:
            raise ValueError("running repair cannot have available attempt budget")
        return self

    @property
    def budget(self) -> Budget:
        if self.attempt_no == ATTEMPT_MAX:
            return Budget.INVALID
        return Budget.AVAILABLE if self.success_since_attempt else Budget.SPENT

    @property
    def durable_phase(self) -> Phase:
        return Phase(self.phase)


_COLUMNS = (
    "attempt_no",
    "phase",
    "trigger_id",
    "trigger_kind",
    "signature",
    "trigger_json",
    "target_manifest",
    "oracle_ref",
    "oracle_hash",
    "success_since_attempt",
    "last_folded_event_id",
    "success_since_failure",
    "acknowledged_events",
    "release_certificate",
)
_JSON_COLUMNS = {"trigger_json", "target_manifest", "acknowledged_events", "release_certificate"}


def load_repair_row(conn: duckdb.DuckDBPyConnection) -> RepairRow:
    rows = conn.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM daemon_index_repair WHERE singleton"
    ).fetchall()
    if len(rows) != 1:
        raise ValueError("recovery singleton is missing or duplicated")
    payload = dict(zip(_COLUMNS, rows[0], strict=True))
    for name in _JSON_COLUMNS:
        if payload[name] is not None:
            payload[name] = json.loads(payload[name])
    return RepairRow.model_validate(payload)


def initialize_repair(conn: duckdb.DuckDBPyConnection) -> None:
    """Seed fresh/recreated schema inside its caller's fence transaction."""
    conn.execute(CREATE_REPAIR_TABLE)
    conn.execute("INSERT OR IGNORE INTO daemon_index_repair (singleton) VALUES (TRUE)")
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations (migration_id) VALUES (?)", [MIGRATION_ID]
    )
