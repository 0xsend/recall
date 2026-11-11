"""Typed boundary for the durable, schema-fenced repair singleton."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
    trigger_kind VARCHAR CHECK (trigger_kind IN ('fatal', 'probe', 'manual')),
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
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal["folded", "diagnosed", "verified"]


class CapturedEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class TriggerInstruction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trigger_id: UUID
    kind: Literal["fatal", "probe", "manual"]
    events: tuple[CapturedEvent, ...] = Field(max_length=65)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("duplicate event identity in trigger batch")
        if self.kind == "fatal" and self.trigger_id not in {
            event.event_id for event in self.events
        }:
            raise ValueError("fatal trigger must be captured in its acknowledged batch")
        return self


class ReleaseCertificate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_no: int = Field(ge=1, le=ATTEMPT_MAX, strict=True)
    trigger_id: UUID
    oracle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    post_oracle_ref: str = Field(max_length=256)
    post_oracle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sidecar_generation: str = Field(pattern=r"^[0-9a-f]{64}$")
    events: tuple[CapturedEvent, ...] = Field(max_length=65)


class RepairRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_no: int = Field(ge=0, le=ATTEMPT_MAX, strict=True)
    phase: Literal["idle", "required", "running", "verified", "refused"]
    trigger_id: UUID | None
    trigger_kind: Literal["fatal", "probe", "manual"] | None
    signature: str | None
    trigger_json: TriggerInstruction | None
    target_manifest: TargetManifest | None
    oracle_ref: str | None
    oracle_hash: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    success_since_attempt: bool = Field(strict=True)
    last_folded_event_id: UUID | None
    success_since_failure: bool = Field(strict=True)
    acknowledged_events: tuple[EventAcknowledgement, ...] = Field(max_length=65)
    release_certificate: ReleaseCertificate | None

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
        if self.trigger_json is not None and (
            self.trigger_json.trigger_id != self.trigger_id
            or self.trigger_json.kind != self.trigger_kind
        ):
            raise ValueError("trigger instruction differs from singleton identity")
        certificate = self.release_certificate
        if certificate is not None and (
            certificate.trigger_id != self.trigger_id
            or certificate.attempt_no != self.attempt_no
            or certificate.oracle_hash != self.oracle_hash
            or self.trigger_json is None
            or certificate.events != self.trigger_json.events
        ):
            raise ValueError("release certificate belongs to a different attempt or event batch")
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


@dataclass(frozen=True)
class SchemaFence:
    """Evidence read from a committed schema before publishing fenced state."""

    version: int

    def __post_init__(self) -> None:
        assert self.version >= FENCE_VERSION


def require_schema_fence(conn: duckdb.DuckDBPyConnection) -> SchemaFence:
    row = conn.execute("SELECT max(version) FROM schema_version").fetchone()
    if row is None or row[0] is None or row[0] < FENCE_VERSION:
        raise ValueError("recovery protocol requires a committed schema fence")
    load_repair_row(conn)
    return SchemaFence(int(row[0]))


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


def capture_target_manifest(conn: duckdb.DuckDBPyConnection) -> TargetManifest:
    """Save complete catalog DDL before an interruption can hide a target."""
    from recall.db.maintenance import schema_index_statements

    rows = conn.execute(
        "SELECT schema_name, index_name, table_name, sql FROM duckdb_indexes() "
        "WHERE NOT is_primary ORDER BY schema_name, index_name"
    ).fetchmany(TARGETS_MAX + 1)
    if len(rows) > TARGETS_MAX:
        raise ValueError("index catalog exceeds target bound")
    targets: dict[tuple[str, str], IndexTarget] = {}
    for schema, name, table, sql in rows:
        if sql is None:
            raise ValueError("index target lacks reconstructible DDL")
        targets[schema, name] = IndexTarget(
            schema_name=schema, index_name=name, table_name=table, sql=sql
        )
    tables = {
        (row[0], row[1]): row[2]
        for row in conn.execute(
            "SELECT schema_name, table_name, sql FROM duckdb_tables() WHERE NOT internal"
        ).fetchmany(TARGETS_MAX + 1)
    }
    for statement in schema_index_statements():
        match = re.fullmatch(
            r"CREATE INDEX IF NOT EXISTS (\w+) ON (\w+)\s*\(.+\)", statement, re.IGNORECASE
        )
        assert match is not None, "unsupported schema index statement"
        name, table = match.groups()
        if ("main", table) in tables and ("main", name) not in targets:
            # Render missing schema targets through the same engine catalog as
            # existing targets. This touches only an empty private database;
            # capture never heals or otherwise mutates the source catalog.
            with duckdb.connect() as renderer:
                renderer.execute(tables["main", table])
                renderer.execute(statement)
                canonical = renderer.execute(
                    "SELECT sql FROM duckdb_indexes() WHERE index_name = ?", [name]
                ).fetchone()
                assert canonical is not None and canonical[0] is not None
            targets["main", name] = IndexTarget(
                schema_name="main", index_name=name, table_name=table, sql=canonical[0]
            )
    return TargetManifest(targets=tuple(targets[key] for key in sorted(targets)))


def save_repair_row(conn: duckdb.DuckDBPyConnection, row: RepairRow) -> None:
    """Apply a validated singleton inside an owning transaction."""
    payload = row.model_dump(mode="json")
    values = [
        json.dumps(payload[name])
        if name in _JSON_COLUMNS and payload[name] is not None
        else payload[name]
        for name in _COLUMNS
    ]
    conn.execute(
        "UPDATE daemon_index_repair SET "
        + ", ".join(f"{name} = ?" for name in _COLUMNS)
        + " WHERE singleton",
        values,
    )
