"""Normal opens and live recovery reopens have distinct service authority."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Never

import duckdb

from recall.core.config import AppConfig
from recall.core.daemon_status import RuntimeStatus
from recall.core.repair_state import Action, Admission, Event, PublicationRole, Refusal, decide
from recall.db.connection import connect_existing_owned, connect_owned, connect_recovery_owned
from recall.db.lease import CompletionFailure, DbLease, DbLifetime
from recall.db.repair import FENCE_VERSION, RepairRow, SchemaFence, load_repair_row

if TYPE_CHECKING:
    from recall.services.repair_protocol import RebuiltCapability, RepairProtocol

logger = logging.getLogger("recall.recovery_admission")


@dataclass(frozen=True)
class CompletionSnapshot:
    repair: RepairRow
    runtime: RuntimeStatus
    pending_intents: int


def require_preserving_on(lease: DbLease, config: AppConfig, version: int | None) -> None:
    """Supported ordinary writes cannot consume an unresolved repair instruction."""
    from recall.services.repair_events import EventStorageError, read_events
    from recall.services.self_repair import (
        DaemonStartupRefused,
        preflight_recovery_files,
        write_refusal_marker,
    )

    preflight_recovery_files(config)
    batch = read_events(config.data_dir)
    if version is None or version < FENCE_VERSION:
        if batch.primary is None and not batch.secondary:
            return
    else:
        try:
            row = lease.call("preserving_admission_state", load_repair_row)
        except Exception as error:
            message = f"refusing ordinary writes: recovery state is unavailable: {error}"
            write_refusal_marker(config.data_dir, message)
            raise DaemonStartupRefused(message) from error
        try:
            marker = batch.classify(row.event_dispositions)
        except EventStorageError as error:
            raise DaemonStartupRefused(
                "retained legacy instructions require startup recovery before ordinary writes"
            ) from error
        state = Admission(row.durable_phase, marker, Refusal.ABSENT, row.budget)
        if decide(state, Event.AUTO_COMPACT) == Action.PRESERVE:
            return
    raise DaemonStartupRefused(
        "ordinary writes cannot consume an unresolved recovery instruction; "
        "run startup recovery or `recall db rebuild-indexes` offline"
    )


class RecoveryAdmission:
    def __init__(self, config: AppConfig, lifetime: DbLifetime) -> None:
        self.config = config
        self.lifetime = lifetime
        self._protocol: SchemaFence | Literal["unknown", "legacy"] = (
            "unknown" if config.db_path.exists() else "legacy"
        )

    def _schema(self, version: int | None) -> None:
        self._protocol = (
            SchemaFence(version) if version is not None and version >= FENCE_VERSION else "legacy"
        )

    def _hold(self, reason: str) -> Never:
        from recall.services.self_repair import DaemonStartupRefused, write_refusal_marker

        message = f"refusing to start: {reason}; use verified offline repair or matched restore"
        write_refusal_marker(self.config.data_dir, message)
        raise DaemonStartupRefused(message)

    def normal(self, lease: DbLease, version: int | None) -> None:
        from recall.services.self_repair import preflight_recovery_files

        self._schema(version)
        preflight_recovery_files(self.config)
        if not isinstance(self._protocol, SchemaFence):
            return
        try:
            row = lease.call("repair_admission_state", load_repair_row)
        except Exception as error:
            self._hold(f"recovery state is unavailable: {error}")
        if row.phase in {"running", "refused"}:
            self._hold(f"retained {row.phase} recovery has no live execution authority")

    def open(self) -> duckdb.DuckDBPyConnection:
        from recall.services.self_repair import preflight_recovery_files

        preflight_recovery_files(self.config)
        return connect_owned(
            self.config, lifetime=self.lifetime, admission=self.normal, lenient_schema=True
        )

    def preserving(self, lease: DbLease, version: int | None) -> None:
        self._schema(version)
        require_preserving_on(lease, self.config, version)

    def open_preserving(
        self, *, read_only: bool, storage_target: str | None = None
    ) -> duckdb.DuckDBPyConnection:
        from recall.services.self_repair import preflight_recovery_files

        preflight_recovery_files(self.config)
        return connect_existing_owned(
            self.config,
            lifetime=self.lifetime,
            admission=self.preserving,
            read_only=read_only,
            storage_target=storage_target,
        )

    def reopen(
        self, protocol: RepairProtocol, capability: RebuiltCapability
    ) -> duckdb.DuckDBPyConnection:
        from recall.services.self_repair import preflight_recovery_files

        preflight_recovery_files(self.config)

        def admit(lease: DbLease, version: int | None) -> None:
            self._schema(version)
            if not isinstance(self._protocol, SchemaFence):
                self._hold("live recovery reopen lost its committed schema fence")
            protocol.admit_reopen(lease, capability)

        return connect_recovery_owned(self.config, lifetime=self.lifetime, admission=admit)

    def publish_fatal(self, error: BaseException, operation: str, role: PublicationRole) -> None:
        from recall.services.self_repair import remember_fatal_failure, write_refusal_marker

        if self._protocol == "unknown":
            message = (
                "refusing to start: schema admission is unavailable; marker protocol is unknown; "
                "retain the database and events for verified offline repair or matched restore"
            )
            write_refusal_marker(self.config.data_dir, message)
            logger.error("%s; operation=%s error=%s", message, operation, error, exc_info=error)
            return
        fence = self._protocol if isinstance(self._protocol, SchemaFence) else None
        remember_fatal_failure(self.config.data_dir, error, site=operation, role=role, fence=fence)

    def publish_verification(self, error: BaseException, operation: str) -> None:
        from recall.services.repair_events import EventStorageError, read_events
        from recall.services.self_repair import FailureMarker, write_failure_marker

        if self._protocol == "unknown":
            raise EventStorageError(
                "schema admission unavailable; verification protocol is unknown"
            )
        fence = self._protocol if isinstance(self._protocol, SchemaFence) else None
        written = write_failure_marker(
            self.config.data_dir,
            FailureMarker(None, True),
            role=PublicationRole.SECONDARY,
            fence=fence,
        )
        if read_events(self.config.data_dir).full:
            raise EventStorageError("failure-event spool is full; matched restore is required")
        logger.warning(
            "disk full operation=%s verification_published=%s error=%s", operation, written, error
        )

    def read_retained_state(self) -> RepairRow:
        """Read-only diagnosis after the exclusive caller closes its parent.

        Cleanup admission permits this diagnosis after a terminal failure. It
        grants no migration, writable reopen, retry or service authority.
        """
        return self._read_retained(load_repair_row)

    def resolve_completion(self, failure: CompletionFailure) -> None:
        """Diagnose durable state after all owned native handles close; never replay."""
        from recall.services.repair_events import serialized_events
        from recall.services.runtime_state import load_runtime_status_from_conn
        from recall.services.self_repair import admitted_recovery_files

        def snapshot(conn: duckdb.DuckDBPyConnection) -> CompletionSnapshot:
            repair = load_repair_row(conn)
            runtime = load_runtime_status_from_conn(conn)
            count = conn.execute("SELECT count(*) FROM fts_sidecar_pending").fetchone()
            assert count is not None
            return CompletionSnapshot(repair, runtime, int(count[0]))

        with serialized_events():
            current = self._read_retained(snapshot)
            batch = admitted_recovery_files(self.config.data_dir)
            state = Admission(
                current.repair.durable_phase,
                batch.classify(current.repair.event_dispositions),
                Refusal.ABSENT,
                current.repair.budget,
            )
            if decide(state, Event.PARTIAL) != Action.PARTIAL:
                self._hold("fresh completion diagnosis retained nonserviceable recovery state")
        logger.warning(
            "completion fresh diagnosis origin=%s phase=%s attempt=%d success_flags=%s/%s "
            "last_attempted_at=%s last_successful_at=%s pending_intents=%d; no completion replay",
            type(failure.error).__name__,
            current.repair.phase,
            current.repair.attempt_no,
            current.repair.success_since_attempt,
            current.repair.success_since_failure,
            current.runtime.last_attempted_at,
            current.runtime.last_successful_at,
            current.pending_intents,
        )

    def _read_retained[T](self, read: Callable[[duckdb.DuckDBPyConnection], T]) -> T:
        conn = self.lifetime.cleanup_db(
            "refusal_resolution_open",
            lambda: duckdb.connect(str(self.config.db_path), read_only=True),
        )
        try:
            row = self.lifetime.cleanup_db("refusal_resolution_read", lambda: read(conn))
        except BaseException as primary:
            try:
                self.lifetime.cleanup_db("refusal_resolution_close", conn.close)
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "refusal resolution and close failed", [primary, cleanup]
                ) from None
            raise
        else:
            self.lifetime.cleanup_db("refusal_resolution_close", conn.close)
            return row
