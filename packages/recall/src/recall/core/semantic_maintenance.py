"""Typed durable semantic lifecycle intent and admission."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from weakref import WeakValueDictionary

from pydantic import BaseModel, ConfigDict

MARKER_NAME = "semantic-work.json"
RETAINED_PREIMAGES_MAX = 32


class PairIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    database: str
    sidecar: str | None


class SemanticIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    format_version: Literal[2] = 2
    definition_version: Literal[1] = 1
    db_path: str
    build_id: str
    epoch: str
    phase: Literal["backup", "building", "verifying", "complete", "failed", "auditing"]
    source_schema_version: int
    operation: Literal["initialize", "repair", "dimensions"] = "initialize"
    target_dimensions: int | None = None
    reset_scope_digest: str | None = None
    started_at: float | None = None
    verified_at: float | None = None
    backup_path: str | None = None
    backup_files: PairIdentity | None = None
    live_files: PairIdentity | None = None
    canonical_digest: str | None = None
    captured_messages: int | None = None
    captured_tools: int | None = None
    predecessor: str | None = None
    accepted_plan_id: str | None = None
    error: str | None = None
    recovery: Literal["resume", "audit", "repair"] = "resume"


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_durable(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_intent(data_dir: Path) -> SemanticIntent | None:
    path = data_dir / MARKER_NAME
    if not path.exists():
        return None
    if path.stat().st_size > 65536:
        raise ValueError("semantic intent exceeds 65536-byte capacity")
    return SemanticIntent.model_validate_json(path.read_text())


class SemanticAdmission(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    epoch: str
    db_path: str
    phase: Literal["auditing", "held", "open"]
    error: str | None = None


@dataclass(frozen=True)
class SemanticMaintenance:
    db_path: str
    epoch: str


_admissions: WeakValueDictionary[Path, SemanticAdmission] = WeakValueDictionary()


def save_admission(data_dir: Path, admission: SemanticAdmission) -> None:
    _admissions[data_dir.resolve()] = admission
    write_durable(data_dir / "semantic-admission.json", admission.model_dump_json())


def require_semantic_writes_enabled(
    data_dir: Path, *, db_path: Path | None = None, capability: SemanticMaintenance | None = None
) -> None:
    path = data_dir / "semantic-admission.json"
    admission = _admissions.get(data_dir.resolve())
    if admission is None and path.exists():
        admission = SemanticAdmission.model_validate_json(path.read_text())
    if admission is not None:
        expected = str((db_path or data_dir / "recall.duckdb").resolve())
        if admission.db_path != expected:
            raise RuntimeError("semantic admission target differs; start the approved owner")
        accepted = (
            capability is not None
            and capability.db_path == expected
            and capability.epoch == admission.epoch
        )
        if admission.phase != "open" and not accepted:
            raise RuntimeError(
                f"semantic_work_initializing: {admission.phase}; inspect daemon status; "
                f"{admission.error or 'owner audit or maintenance in progress'}"
            )
    intent = read_intent(data_dir)
    if intent is not None and intent.phase != "complete":
        raise RuntimeError(
            f"semantic_work_invalid: {intent.phase} build={intent.build_id}; "
            "inspect daemon status or repair-semantic-work --dry-run; "
            f"{intent.error or 'initialization in progress'}"
        )


class RepairPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    operation: Literal["repair_semantic_work"] = "repair_semantic_work"
    plan_id: str
    db_path: str
    file_identity: tuple[int, int]
    binary_version: str
    schema_version: int
    definition_version: int = 1
    failed_build_id: str | None
    diagnostic: str | None
    mutation_seq: int | None
    authoritative_generation: str
    proposed_build_id: str
    backup_root: str
    spill_path: str
    required_bytes: int
    retained_preimages: tuple[tuple[str, int], ...]
    predecessor: str | None
    rollback_scope: Literal["schema32_5fa253b", "schema33_current"]
    replaces: tuple[str, ...] = (
        "semantic_work",
        "semantic_work_counts",
        "semantic_work_state",
        "semantic_pending",
    )


class RepairResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    outcome: Literal["accepted", "unchanged"]
    plan_id: str
    build_id: str
    phase: str
    error: str | None
