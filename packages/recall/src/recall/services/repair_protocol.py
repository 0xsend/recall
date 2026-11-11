"""Durable recovery transitions; persisted phases never grant live authority."""

from __future__ import annotations

import hashlib
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Never
from uuid import UUID, uuid4

from recall.core.durability import sync_directory
from recall.core.repair_state import Action, Admission, Event, Refusal, decide
from recall.core.time import utcnow_naive
from recall.db.lease import DbLease
from recall.db.maintenance import IndexDivergenceReport, IndexRebuildResult, probe_index_divergence
from recall.db.repair import (
    TARGETS_MAX,
    AttemptNumber,
    CapturedEvent,
    EventAcknowledgement,
    RepairRow,
    TriggerInstruction,
    capture_existing_targets,
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
    publish_event,
    read_events,
    serialized_events,
)
from recall.services.repair_oracles import OracleStore
from recall.services.runtime_state import (
    load_runtime_status_from_conn,
    load_runtime_status_on,
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


@dataclass(frozen=True)
class ArmCapability:
    _protocol: object
    _lineage: tuple[object, int]
    _connection_epoch: object
    attempt_no: AttemptNumber
    trigger_id: UUID
    oracle_hash: str


@dataclass(frozen=True)
class ManualArmCapability(ArmCapability):
    """Issued only by the explicit offline owner after a manual running commit."""


@dataclass(frozen=True)
class RebuiltCapability:
    _arm: ArmCapability
    result: IndexRebuildResult


@dataclass(frozen=True)
class NormalStartup:
    probe: IndexDivergenceReport
    records: tuple[RecoveryEvent, ...]


@dataclass(frozen=True)
class AutomaticStartup:
    records: tuple[RecoveryEvent, ...]


class RefusalCommitUncertain(RuntimeError):
    """The refusal acknowledgment failed; a fresh read must resolve its phase."""


def _no_barrier(stage: str) -> None:
    pass


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class RepairProtocol:
    def __init__(
        self,
        data_dir: Path,
        *,
        barrier: Callable[[str], None] = _no_barrier,
        probe: Callable[[DbLease], IndexDivergenceReport] | None = None,
        clock: Callable[[], datetime] = utcnow_naive,
    ) -> None:
        self.data_dir = data_dir
        self._identity = object()
        self._live_attempt: ArmCapability | None = None
        self._rebuilt: RebuiltCapability | None = None
        self._barrier = barrier
        self._clock = clock
        self._probe = probe or (lambda lease: lease.call("repair_probe", probe_index_divergence))
        self.last_probe: IndexDivergenceReport | None = None

    def _hold(self, reason: str) -> Never:
        message = (
            f"refusing to start: {reason}; run `recall db rebuild-indexes` offline "
            "for verified repair or use matched restore"
        )
        write_refusal_marker(self.data_dir, message)
        raise DaemonStartupRefused(message)

    def _refusal(self) -> Refusal:
        try:
            refusal_marker_path(self.data_dir).read_bytes()
        except FileNotFoundError:
            refusal = Refusal.ABSENT
        except OSError:
            refusal = Refusal.UNREADABLE
        else:
            refusal = Refusal.PRESENT
        return refusal

    def _admission(self, row: RepairRow, batch: EventBatch) -> Admission:
        return Admission(row.durable_phase, batch.classify(row), self._refusal(), row.budget)

    def begin_start(self, lease: DbLease) -> NormalStartup | AutomaticStartup:
        """Admit the complete event namespace before optional probe or sidecar work."""
        fence = lease.call("repair_schema_fence", require_schema_fence)
        batch = read_events(self.data_dir)
        row = lease.call("repair_state_read", load_repair_row)
        # File and durable holds precede legacy import or any diagnostic write.
        if self._refusal() != Refusal.ABSENT:
            self._hold("retained refusal does not admit normal startup")
        if row.phase in {"running", "refused"}:
            self._hold(f"retained {row.phase} has no live execution capability")
        if batch.full:
            self._hold("failure-event spool is full; matched restore is required")
        batch = import_legacy_events(self.data_dir, fence)
        action = decide(self._admission(row, batch), Event.START_NEGATIVE)
        if action == Action.HOLD:
            self._hold("durable recovery state does not admit startup")
        if action == Action.REFUSE:
            self.refuse(lease, batch)
        records = _records(batch)
        if action in {Action.REQUIRE_INDEX, Action.REQUIRE_PROBE, Action.RESUME_REQUIRED}:
            self.prepare(lease, batch)
            return AutomaticStartup(records)
        assert action in {Action.NORMAL, Action.DIAGNOSE}
        row = self._fold(lease, batch)
        probe = self._probe(lease)
        self.last_probe = probe
        if probe.diverged:
            publish_event(
                self.data_dir,
                RecoveryEvent(
                    event_id=uuid4(), failure=None, needs_index_verification=True, probe=True
                ),
            )
            batch = read_events(self.data_dir)
        action = decide(
            self._admission(row, batch),
            Event.START_POSITIVE if probe.diverged else Event.START_NEGATIVE,
        )
        if action == Action.REFUSE:
            self.refuse(lease, batch)
        if action == Action.REQUIRE_PROBE:
            self.prepare(lease, batch)
            return AutomaticStartup(_records(batch))
        if action not in {Action.NORMAL, Action.DIAGNOSE}:
            self._hold("startup probe does not admit ordinary service")
        acknowledgements = tuple(
            entry.model_copy(update={"disposition": "diagnosed"})
            if entry.disposition == "folded"
            else entry
            for entry in row.acknowledged_events
        )
        diagnosed = _changed(row, acknowledged_events=acknowledgements)
        status = load_runtime_status_on(lease)
        with lease.transaction("diagnose_startup") as transaction:
            if diagnosed != row:
                transaction.call("diagnosed_events", lambda conn: save_repair_row(conn, diagnosed))
            if status.needs_index_verification and probe.complete:
                transaction.call(
                    "negative_verification_probe",
                    lambda conn: set_needs_index_verification(conn, False),
                )
        self.retire(lease)
        return NormalStartup(probe, records)

    def admit_reopen(self, lease: DbLease, capability: RebuiltCapability) -> None:
        """A retained RUNNING row alone cannot authorize a writable reopen."""
        if capability is not self._rebuilt:
            self._hold("fresh recovery reopen lacks its live rebuilt capability")
        event = (
            Event.MANUAL_STEP if isinstance(capability._arm, ManualArmCapability) else Event.VERIFY
        )
        self._matching_attempt(lease, capability._arm, event)

    def _fold(self, lease: DbLease, batch: EventBatch) -> RepairRow:
        """Fold each new id exactly once; a fold is never a final disposition."""
        with lease.admitting(), serialized_events():
            return self._fold_ledger(lease, batch)

    def _fold_ledger(self, lease: DbLease, batch: EventBatch) -> RepairRow:
        row = lease.call("repair_state_read", load_repair_row)
        entries = {entry.event_id: entry for entry in row.acknowledged_events}
        records = _records(batch)
        for event in records:
            old = entries.get(event.event_id)
            if old is not None and old.content_hash != event.content_hash:
                self._hold("the same event id has conflicting content")
        retained = self._retained_ledger(row, batch)
        entries = {entry.event_id: entry for entry in retained}
        acknowledged = row.event_dispositions
        unseen = tuple(event for event in records if event.event_id not in acknowledged)
        if len(entries) + len(unseen) > 65:
            self._hold("acknowledged event ledger is full; matched restore is required")
        if not unseen:
            if retained != row.acknowledged_events:
                row = _changed(row, acknowledged_events=retained)
                with lease.transaction("retire_final_event_ledger") as transaction:
                    transaction.call(
                        "retired_event_ledger", lambda conn: save_repair_row(conn, row)
                    )
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
                            message=f"{record.site}: {record.message}",
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
        store = OracleStore(self.data_dir)
        if row.phase == "required":
            if row.oracle_ref is None or row.oracle_hash is None or row.trigger_id is None:
                self._hold("required instruction lacks its complete pre-attempt oracle")
            reference = OracleReference(trigger_id=row.trigger_id, manifest_hash=row.oracle_hash)
            read_oracle(self.data_dir / row.oracle_ref, reference)
            store.reconcile(row)
            return row
        events = _records(batch)
        trigger = next((event for event in events if event.marker.value == "index"), None)
        if trigger is None:
            trigger = next((event for event in events if event.marker.value == "probe"), None)
        if trigger is None:
            self._hold("no actionable durable instruction was admitted")
            raise AssertionError("hold did not terminate")
        relative = f"repair-oracles/{trigger.event_id}"
        directory = self.data_dir / relative
        store.reconcile(row, trigger.event_id)
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
                    conn,
                    directory,
                    reference,
                    sidecar=self.data_dir / "recall.fts.sqlite",
                    observer=lease.observer,
                ),
            )
        else:
            reference = lease.call(
                "capture_repair_oracle",
                lambda conn: capture_oracle(
                    conn,
                    directory,
                    trigger.event_id,
                    sidecar=self.data_dir / "recall.fts.sqlite",
                    barrier=self._barrier,
                    observer=lease.observer,
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
        self._barrier("before_required_commit")
        with lease.transaction("require_index_repair") as transaction:
            transaction.call("required_instruction", lambda conn: save_repair_row(conn, row))
        self._barrier("after_required_commit")
        self._barrier("before_oracle_retirement")
        store.reconcile(row)
        self._barrier("after_oracle_retirement")
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
        try:
            self._barrier("before_refused_commit")
            with lease.transaction("refuse_index_repair") as transaction:
                transaction.call("refused_instruction", lambda conn: save_repair_row(conn, row))
            self._barrier("after_refused_commit")
        except Exception as error:
            raise RefusalCommitUncertain("refusal commit acknowledgment is uncertain") from error
        self._hold(
            "automatic repair budget is spent; the previous instruction and oracle are retained"
        )

    def arm(self, lease: DbLease) -> ArmCapability:
        """Commit spent RUNNING before issuing this process's sole DDL capability."""
        row = lease.call("repair_state_read", load_repair_row)
        batch = read_events(self.data_dir)
        if row.trigger_kind == "manual":
            self._hold("a manually selected instruction requires explicit manual arm")
        if decide(self._admission(row, batch), Event.ARM) != Action.ARM:
            self._hold("durable state does not admit automatic arm")
        try:
            next_attempt = AttemptNumber(value=row.attempt_no + 1)
        except ValueError:
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
                    conn,
                    directory,
                    reference,
                    sidecar=self.data_dir / "recall.fts.sqlite",
                    observer=lease.observer,
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
            attempt_no=next_attempt.value,
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
            next_attempt,
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
            row.attempt_no != capability.attempt_no.value
            or row.trigger_id != capability.trigger_id
            or row.oracle_hash != capability.oracle_hash
        ):
            self._hold("attempt capability differs from the durable instruction")
        if decide(self._admission(row, read_events(self.data_dir)), event) not in {
            Action.STEP,
            Action.VERIFY,
            Action.MANUAL_STEP,
            Action.MANUAL_RELEASE,
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
        event = Event.MANUAL_STEP if isinstance(capability, ManualArmCapability) else Event.STEP
        row = self._matching_attempt(lease, capability, event)
        assert row.target_manifest is not None
        self._live_attempt = None
        present_rows = lease.call(
            "repair:present_targets",
            lambda conn: conn.execute(
                "SELECT schema_name,index_name FROM duckdb_indexes() WHERE NOT is_primary"
            ).fetchmany(TARGETS_MAX + 1),
        )
        if len(present_rows) > TARGETS_MAX:
            self._hold("current index catalog exceeds the saved target bound")
        present = set(present_rows)
        healed = tuple(
            target.index_name
            for target in row.target_manifest.targets
            if (target.schema_name, target.index_name) not in present
        )
        dropped = sum(
            (target.schema_name, target.index_name) in present
            for target in row.target_manifest.targets
        )
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
                dropped=dropped,
                created=len(row.target_manifest.targets),
                healed=healed,
                elapsed_seconds=time.perf_counter() - started,
            ),
        )
        self._rebuilt = rebuilt
        return rebuilt

    def verify(self, lease: DbLease, capability: RebuiltCapability) -> RepairRow:
        """The caller supplies a lease from its drained, freshly reopened owner."""
        if capability is not self._rebuilt or isinstance(capability._arm, ManualArmCapability):
            self._hold("no completed automatic rebuild capability for exact verification")
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
        if lease.call("repair_fresh_catalog", capture_existing_targets) != row.target_manifest:
            self._hold("fresh rebuilt catalog differs from the saved target manifest")
        lease.call(
            "repair_fresh_exact_postcheck",
            lambda conn: verify_oracle(
                conn,
                directory,
                reference,
                sidecar=self.data_dir / "recall.fts.sqlite",
                observer=lease.observer,
            ),
        )
        probe = self._probe(lease)
        self.last_probe = probe
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
                    conn, signature=row.signature, repaired_at=self._clock()
                ),
            )
            transaction.call("verified_instruction", lambda conn: save_repair_row(conn, verified))
            if probe.complete:
                transaction.call(
                    "verified_negative_probe",
                    lambda conn: set_needs_index_verification(conn, False),
                )
        self._barrier("after_verified_commit")
        return verified

    def retire(self, lease: DbLease) -> None:
        """Retire only an entirely acknowledged batch, never a normal-start R."""
        with lease.admitting(), serialized_events():
            self._retire(lease)

    def _retained_ledger(
        self, row: RepairRow, batch: EventBatch
    ) -> tuple[EventAcknowledgement, ...]:
        present = {event.event_id for event in _records(batch)}
        retained = tuple(
            entry
            for entry in row.acknowledged_events
            if entry.disposition == "folded" or entry.event_id in present
        )
        if retained != row.acknowledged_events:
            try:
                # Establish durable absence before reclaiming final dispositions.
                sync_directory(self.data_dir)
                spool = self.data_dir / "fatal-events"
                if spool.exists():
                    sync_directory(spool)
            except OSError as error:
                self._hold(f"could not sync acknowledged event retirement: {error}")
        return retained

    def _require_retirement(self, lease: DbLease, row: RepairRow, batch: EventBatch) -> None:
        state = self._admission(row, batch)
        if decide(state, Event.RETIRE) != Action.RETIRE:
            if decide(state, Event.START_NEGATIVE) == Action.REFUSE:
                self.refuse(lease, batch)
            self._hold("unresolved events or a refusal prevent instruction retirement")

    def _retire(self, lease: DbLease) -> None:
        row = lease.call("repair_state_read", load_repair_row)
        batch = read_events(self.data_dir)
        self._require_retirement(lease, row, batch)
        records = batch.secondary
        if batch.primary is not None:
            records = ((self.data_dir / "daemon-failure.json", batch.primary), *records)
        acknowledged = row.event_dispositions
        for path, event in records:
            assert isinstance(event, RecoveryEvent)
            if acknowledged.get(event.event_id) != (event.content_hash, True):
                self._hold("event identity lacks a durable final disposition")
            try:
                self._barrier("before_event_unlink")
                self._require_retirement(lease, row, read_events(self.data_dir))
                path.unlink()
                sync_directory(path.parent)
                self._barrier("after_event_unlink_sync")
            except OSError as error:
                self._hold(f"could not durably retire an acknowledged event: {error}")
        remaining = read_events(self.data_dir)
        self._require_retirement(lease, row, remaining)
        retained = self._retained_ledger(row, remaining)
        if retained != row.acknowledged_events:
            row = _changed(row, acknowledged_events=retained)
            with lease.transaction("retire_acknowledged_ledger") as transaction:
                transaction.call("retired_event_ledger", lambda conn: save_repair_row(conn, row))
