"""Durable recovery transitions; persisted phases never grant live authority."""

from __future__ import annotations

import hashlib
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Never
from uuid import UUID

from recall.core.durability import sync_directory
from recall.core.repair_state import Action, Admission, Event, Refusal, decide
from recall.core.time import utcnow_naive
from recall.db.lease import DbLease
from recall.db.maintenance import IndexRebuildResult, probe_index_divergence
from recall.db.repair import (
    ATTEMPT_MAX,
    CapturedEvent,
    EventAcknowledgement,
    RepairRow,
    TriggerInstruction,
    load_repair_row,
    require_schema_fence,
    save_repair_row,
)
from recall.db.repair_oracle import (
    MANIFEST_BYTES_MAX,
    OracleManifest,
    OracleReference,
    OracleUnavailable,
    capture_oracle,
    read_oracle,
    verify_oracle,
)
from recall.services.repair_events import (
    EventBatch,
    LegacyEvent,
    RecoveryEvent,
    import_legacy_events,
    read_events,
)
from recall.services.runtime_state import (
    load_runtime_status_from_conn,
    record_fatal_failure,
    record_index_repair,
    set_needs_index_verification,
)
from recall.services.self_repair import (
    DaemonStartupRefused,
    refusal_marker_path,
    write_refusal_marker,
)


def _changed(row: RepairRow, **values: object) -> RepairRow:
    return RepairRow.model_validate(row.model_dump() | values)


def _records(batch: EventBatch) -> tuple[RecoveryEvent, ...]:
    return batch.identified()


def _acknowledged(row: RepairRow) -> dict[UUID, tuple[str, bool]]:
    acknowledged = {
        entry.event_id: (entry.content_hash, entry.disposition != "folded")
        for entry in row.acknowledged_events
    }
    if row.phase == "verified" and row.trigger_json is not None:
        for item in row.trigger_json.events:
            acknowledged.setdefault(item.event_id, (item.content_hash, True))
    return acknowledged


@dataclass(frozen=True)
class ArmCapability:
    _protocol: object
    _lineage: tuple[object, int]
    _connection_epoch: object
    attempt_no: int
    trigger_id: UUID
    oracle_hash: str


@dataclass(frozen=True)
class RebuiltCapability:
    _arm: ArmCapability
    result: IndexRebuildResult


def _no_barrier(stage: str) -> None:
    pass


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class RepairProtocol:
    def __init__(self, data_dir: Path, *, barrier: Callable[[str], None] = _no_barrier) -> None:
        self.data_dir = data_dir
        self._identity = object()
        self._live_attempt: ArmCapability | None = None
        self._rebuilt: RebuiltCapability | None = None
        self._barrier = barrier

    def _hold(self, reason: str) -> Never:
        message = f"refusing to start: {reason}; use verified offline repair or matched restore"
        write_refusal_marker(self.data_dir, message)
        raise DaemonStartupRefused(message)

    def _admission(self, row: RepairRow, batch: EventBatch) -> Admission:
        try:
            refusal_marker_path(self.data_dir).read_bytes()
        except FileNotFoundError:
            refusal = Refusal.ABSENT
        except OSError:
            refusal = Refusal.UNREADABLE
        else:
            refusal = Refusal.PRESENT
        return Admission(row.durable_phase, batch.classify(_acknowledged(row)), refusal, row.budget)

    def _fold(self, lease: DbLease, batch: EventBatch) -> RepairRow:
        """Fold each new id exactly once; a fold is never a final disposition."""
        row = lease.call("repair_state_read", load_repair_row)
        entries = {entry.event_id: entry for entry in row.acknowledged_events}
        records = _records(batch)
        for event in records:
            old = entries.get(event.event_id)
            if old is not None and old.content_hash != event.content_hash:
                self._hold("the same event id has conflicting content")
        acknowledged = _acknowledged(row)
        unseen = tuple(event for event in records if event.event_id not in acknowledged)
        if len(entries) + len(unseen) > 65:
            self._hold("acknowledged event ledger is full; matched restore is required")
        if not unseen:
            return row
        with lease.transaction("fold_repair_events") as transaction:
            for event in unseen:
                record = event.failure
                if record is not None:
                    status = transaction.call("fatal_history_read", load_runtime_status_from_conn)
                    repeats = (
                        status.fatal_repeat_count + 1
                        if status.last_fatal_signature == record.signature
                        else 1
                    )
                    transaction.call(
                        "fatal_history_fold",
                        lambda conn, record=record, repeats=repeats: record_fatal_failure(
                            conn,
                            message=record.message,
                            signature=record.signature,
                            failed_at=record.at,
                            repeat_count=repeats,
                        ),
                    )
                if event.needs_index_verification:
                    transaction.call(
                        "verification_flag_fold",
                        lambda conn: set_needs_index_verification(conn, True),
                    )
                entries[event.event_id] = EventAcknowledgement(
                    event_id=event.event_id, content_hash=event.content_hash, disposition="folded"
                )
            row = _changed(
                row,
                success_since_failure=False,
                last_folded_event_id=unseen[-1].event_id,
                acknowledged_events=tuple(entries.values()),
            )
            transaction.call("repair_event_ledger_fold", lambda conn: save_repair_row(conn, row))
        return row

    def prepare(self, lease: DbLease, batch: EventBatch) -> RepairRow:
        """Save the complete pre-attempt reference without mutating any index."""
        fence = lease.call("repair_schema_fence", require_schema_fence)
        if batch != read_events(self.data_dir):
            self._hold("failure instruction changed before admission")
        if any(
            isinstance(record, LegacyEvent)
            for record in (
                (() if batch.primary is None else (batch.primary,))
                + tuple(event for _, event in batch.secondary)
            )
        ):
            batch = import_legacy_events(self.data_dir, fence)
        row = lease.call("repair_state_read", load_repair_row)
        action = decide(self._admission(row, batch), Event.START_NEGATIVE)
        if action == Action.REFUSE:
            self.refuse(lease, batch)
        if action not in {Action.REQUIRE_INDEX, Action.REQUIRE_PROBE, Action.RESUME_REQUIRED}:
            self._hold(f"{action.value} does not authorize preparing an automatic repair")
        row = self._fold(lease, batch)
        if row.phase == "required":
            if row.oracle_ref is None or row.oracle_hash is None or row.trigger_id is None:
                self._hold("required instruction lacks its complete pre-attempt oracle")
            reference = OracleReference(trigger_id=row.trigger_id, manifest_hash=row.oracle_hash)
            read_oracle(self.data_dir / row.oracle_ref, reference)
            return row
        events = _records(batch)
        trigger = next(
            (event for event in events if event.marker.value in {"index", "probe"}), None
        )
        if trigger is None:
            self._hold("no actionable durable instruction was admitted")
            raise AssertionError("hold did not terminate")
        relative = f"repair-oracles/{trigger.event_id}"
        directory = self.data_dir / relative
        if directory.exists():
            # A complete but unreferenced capture can survive a process kill
            # before REQUIRED. Reuse it only after exact current comparison.
            with (directory / "manifest.json").open("rb") as stream:
                payload = stream.read(MANIFEST_BYTES_MAX + 1)
            if len(payload) > MANIFEST_BYTES_MAX:
                self._hold("unreferenced oracle manifest exceeds its bound")
            manifest = OracleManifest.model_validate_json(payload)
            reference = OracleReference(
                trigger_id=trigger.event_id, manifest_hash=hashlib.sha256(payload).hexdigest()
            )
            lease.call(
                "required_oracle_reuse",
                lambda conn: verify_oracle(
                    conn, directory, reference, sidecar=self.data_dir / "recall.fts.sqlite"
                ),
            )
        else:
            reference = lease.call(
                "capture_repair_oracle",
                lambda conn: capture_oracle(
                    conn, directory, trigger.event_id, sidecar=self.data_dir / "recall.fts.sqlite"
                ),
            )
            manifest = read_oracle(directory, reference)
        instruction = TriggerInstruction(
            trigger_id=trigger.event_id,
            kind="probe" if trigger.probe else "fatal",
            events=tuple(
                CapturedEvent(event_id=event.event_id, content_hash=event.content_hash)
                for event in events
            ),
        )
        row = _changed(
            row,
            phase="required",
            trigger_id=instruction.trigger_id,
            trigger_kind=instruction.kind,
            signature=trigger.failure.signature if trigger.failure is not None else None,
            trigger_json=instruction,
            target_manifest=manifest.targets,
            oracle_ref=relative,
            oracle_hash=reference.manifest_hash,
            release_certificate=None,
        )
        with lease.transaction("require_index_repair") as transaction:
            transaction.call("required_instruction", lambda conn: save_repair_row(conn, row))
        return row

    def refuse(self, lease: DbLease, batch: EventBatch) -> None:
        """A spent verified attempt retains its prior instruction and every event."""
        row = lease.call("repair_state_read", load_repair_row)
        if decide(self._admission(row, batch), Event.START_NEGATIVE) != Action.REFUSE:
            self._hold("refusal transition has no matching admission")
        row = self._fold(lease, batch)
        row = _changed(
            row, phase="refused", success_since_attempt=False, success_since_failure=False
        )
        with lease.transaction("refuse_index_repair") as transaction:
            transaction.call("refused_instruction", lambda conn: save_repair_row(conn, row))
        self._hold(
            "automatic repair budget is spent; the previous instruction and oracle are retained"
        )

    def arm(self, lease: DbLease) -> ArmCapability:
        """Commit spent RUNNING before issuing this process's sole DDL capability."""
        row = lease.call("repair_state_read", load_repair_row)
        batch = read_events(self.data_dir)
        if decide(self._admission(row, batch), Event.ARM) != Action.ARM:
            self._hold("durable state does not admit automatic arm")
        if row.attempt_no == ATTEMPT_MAX:
            self._hold("repair attempt counter is exhausted; matched restore is required")
        if (
            row.trigger_id is None
            or row.oracle_ref is None
            or row.oracle_hash is None
            or row.target_manifest is None
            or row.trigger_json is None
        ):
            self._hold("required instruction lacks a complete saved reference")
        captured = {event.event_id: event.content_hash for event in row.trigger_json.events}
        for event in _records(batch):
            if captured.get(event.event_id) != event.content_hash:
                self._hold("a post-capture event is outside the admitted attempt batch")
        reference = OracleReference(trigger_id=row.trigger_id, manifest_hash=row.oracle_hash)
        directory = self.data_dir / row.oracle_ref
        try:
            manifest = read_oracle(directory, reference)
            if manifest.targets != row.target_manifest or manifest.sidecar is None:
                self._hold("saved oracle lacks the exact target or keyword reference")
            lease.call(
                "pre_arm_exact_oracle",
                lambda conn: verify_oracle(
                    conn, directory, reference, sidecar=self.data_dir / "recall.fts.sqlite"
                ),
            )
            required_bytes = sum(
                table.byte_count for table in manifest.tables + manifest.sidecar.tables
            )
            if shutil.disk_usage(self.data_dir).free < required_bytes:
                self._hold("insufficient disk capacity for the captured repair working set")
        except (OSError, ValueError, OracleUnavailable) as error:
            self._hold(f"saved repair oracle is unusable: {error}")
        row = _changed(
            row,
            phase="running",
            attempt_no=row.attempt_no + 1,
            success_since_attempt=False,
            success_since_failure=False,
            release_certificate=None,
        )
        with lease.transaction("arm_index_repair") as transaction:
            transaction.call("running_instruction", lambda conn: save_repair_row(conn, row))
        capability = ArmCapability(
            self._identity,
            lease.lineage,
            lease.connection_epoch,
            row.attempt_no,
            reference.trigger_id,
            reference.manifest_hash,
        )
        self._live_attempt = capability
        return capability

    def _matching_attempt(
        self, lease: DbLease, capability: ArmCapability, event: Event
    ) -> RepairRow:
        if capability._protocol is not self._identity or capability._lineage != lease.lineage:
            self._hold("attempt capability belongs to another live owner")
        row = lease.call("repair_state_read", load_repair_row)
        if (
            row.attempt_no != capability.attempt_no
            or row.trigger_id != capability.trigger_id
            or row.oracle_hash != capability.oracle_hash
        ):
            self._hold("attempt capability differs from the durable instruction")
        if decide(self._admission(row, read_events(self.data_dir)), event) not in {
            Action.STEP,
            Action.VERIFY,
        }:
            self._hold("durable state no longer admits this attempt step")
        assert row.trigger_json is not None
        captured = {item.event_id: item.content_hash for item in row.trigger_json.events}
        if any(
            captured.get(item.event_id) != item.content_hash
            for item in _records(read_events(self.data_dir))
        ):
            self._hold("an unresolved post-arm event is outside the acknowledged batch")
        return row

    def rebuild(self, lease: DbLease, capability: ArmCapability) -> RebuiltCapability:
        """Consume a live arm once; any interrupted step stays RUNNING."""
        if capability is not self._live_attempt:
            self._hold("no unconsumed automatic arm capability")
        row = self._matching_attempt(lease, capability, Event.STEP)
        assert row.target_manifest is not None
        self._live_attempt = None
        started = time.perf_counter()
        for target in row.target_manifest.targets:
            name = f"{_quote(target.schema_name)}.{_quote(target.index_name)}"
            lease.call(
                "repair_drop_index",
                lambda conn, name=name: conn.execute(f"DROP INDEX IF EXISTS {name}").fetchone(),
            )
            self._barrier(f"after_drop:{target.schema_name}.{target.index_name}")
        for target in row.target_manifest.targets:
            lease.call(
                "repair_create_index", lambda conn, sql=target.sql: conn.execute(sql).fetchone()
            )
            self._barrier(f"after_create:{target.schema_name}.{target.index_name}")
        self._barrier("before_checkpoint")
        lease.call("repair_checkpoint", lambda conn: conn.execute("CHECKPOINT").fetchone())
        self._barrier("after_checkpoint")
        rebuilt = RebuiltCapability(
            capability,
            IndexRebuildResult(
                dropped=len(row.target_manifest.targets),
                created=len(row.target_manifest.targets),
                healed=(),
                elapsed_seconds=time.perf_counter() - started,
            ),
        )
        self._rebuilt = rebuilt
        return rebuilt

    def verify(self, lease: DbLease, capability: RebuiltCapability) -> RepairRow:
        """The caller supplies a lease from its drained, freshly reopened owner."""
        if capability is not self._rebuilt:
            self._hold("no completed rebuild capability for exact verification")
        if lease.connection_epoch is capability._arm._connection_epoch:
            self._hold("verification requires a drained, freshly reopened connection")
        row = self._matching_attempt(lease, capability._arm, Event.VERIFY)
        self._rebuilt = None
        assert (
            row.oracle_ref is not None
            and row.oracle_hash is not None
            and row.trigger_id is not None
        )
        reference = OracleReference(trigger_id=row.trigger_id, manifest_hash=row.oracle_hash)
        directory = self.data_dir / row.oracle_ref
        lease.call(
            "repair_fresh_exact_postcheck",
            lambda conn: verify_oracle(
                conn,
                directory,
                reference,
                sidecar=self.data_dir / "recall.fts.sqlite",
            ),
        )
        probe = lease.call("repair_fresh_probe", probe_index_divergence)
        if probe.diverged:
            self._hold("fresh post-rebuild probe still detects index divergence")
        self._barrier("after_fresh_postchecks")
        assert row.trigger_json is not None
        captured = {item.event_id: item.content_hash for item in row.trigger_json.events}
        acknowledgements = tuple(
            entry.model_copy(update={"disposition": "verified"})
            if captured.get(entry.event_id) == entry.content_hash
            else entry
            for entry in row.acknowledged_events
        )
        verified = _changed(row, phase="verified", acknowledged_events=acknowledgements)
        self._barrier("before_verified_commit")
        with lease.transaction("verify_index_repair") as transaction:
            transaction.call(
                "index_repair_diagnostics",
                lambda conn: record_index_repair(
                    conn, signature=row.signature, repaired_at=utcnow_naive()
                ),
            )
            transaction.call("verified_instruction", lambda conn: save_repair_row(conn, verified))
        self._barrier("after_verified_commit")
        return verified

    def retire(self, lease: DbLease) -> None:
        """Retire only an entirely acknowledged batch, never a normal-start R."""
        row = lease.call("repair_state_read", load_repair_row)
        batch = read_events(self.data_dir)
        if decide(self._admission(row, batch), Event.RETIRE) != Action.RETIRE:
            self._hold("unresolved events or a refusal prevent instruction retirement")
        records = batch.secondary
        if batch.primary is not None:
            records = ((self.data_dir / "daemon-failure.json", batch.primary), *records)
        acknowledged = _acknowledged(row)
        retired: set[UUID] = set()
        for path, event in records:
            assert isinstance(event, RecoveryEvent)
            if acknowledged.get(event.event_id) != (event.content_hash, True):
                self._hold("event identity lacks a durable final disposition")
            try:
                self._barrier("before_event_unlink")
                path.unlink()
                sync_directory(path.parent)
                self._barrier("after_event_unlink_sync")
            except OSError as error:
                self._hold(f"could not durably retire an acknowledged event: {error}")
            retired.add(event.event_id)
        if retired:
            row = _changed(
                row,
                acknowledged_events=tuple(
                    entry for entry in row.acknowledged_events if entry.event_id not in retired
                ),
            )
            with lease.transaction("retire_acknowledged_ledger") as transaction:
                transaction.call("retired_event_ledger", lambda conn: save_repair_row(conn, row))
