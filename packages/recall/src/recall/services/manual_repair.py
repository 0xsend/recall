"""Explicit offline recovery with saved-reference and same-process execution authority."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Never
from uuid import uuid4

import duckdb

from recall.core.config import AppConfig
from recall.core.durability import sync_directory
from recall.core.repair_state import Action, Event
from recall.core.schema_version import SchemaTooNewError
from recall.db.connection import (
    _apply_runtime_pragmas,
    connect_existing_owned,
    connect_owned,
    connect_recovery_owned,
    create_copy_candidate_owned,
    is_lock_conflict,
    storage_version,
)
from recall.db.lease import DbLease, DbLifetime, owned_connection
from recall.db.maintenance import IndexRebuildResult
from recall.db.repair import (
    FENCE_VERSION,
    AttemptNumber,
    CapturedEvent,
    ManualWork,
    PreparedPublication,
    ReleaseCertificate,
    RepairRow,
    TriggerInstruction,
    capture_existing_targets,
    load_repair_row,
    require_schema_fence,
    save_repair_row,
)
from recall.db.repair_oracle import (
    OracleReference,
    OracleUnavailable,
    PublicationOracle,
    capture_oracle,
    capture_publication,
    read_oracle,
    read_publication,
    read_publication_at,
    verify_manual_content,
    verify_publication_sidecar,
    verify_published_canonical,
)
from recall.db.schema import check_schema_compatibility
from recall.services.compaction import CompactResult, estimate_bloat_ratio_on
from recall.services.manual_keywords import (
    NativeKeywordCandidate,
    publish_manual_keywords,
    publish_native_keywords,
    require_replaceable_sidecar,
    verify_manual_keywords,
)
from recall.services.recovery_admission import RecoveryAdmission, offline_file_lifetime
from recall.services.repair_events import (
    import_legacy_events,
    read_events,
    serialized_events,
)
from recall.services.repair_oracles import OracleStore
from recall.services.repair_protocol import (
    ManualArmCapability,
    RebuiltCapability,
    RepairProtocol,
    _changed,
    _records,
)
from recall.services.runtime_state import record_index_repair, set_needs_index_verification
from recall.services.self_repair import (
    DaemonStartupRefused,
    refusal_marker_path,
    write_refusal_marker,
)

WORKSPACES_MAX = 64


@dataclass(frozen=True)
class _PreparedReplacement:
    owner: DbLifetime
    candidate: Path
    source_hash: str
    candidate_hash: str
    backup_manifest_hash: str
    row: RepairRow


@dataclass(frozen=True)
class _PreparedManualCompletion:
    arm: ManualArmCapability
    publication: PreparedPublication
    epoch: object
    candidate: Path | None


class ManualRepair(RepairProtocol):
    """The constructor declares manual intent; execute owns its complete offline realm."""

    def __init__(
        self,
        config: AppConfig,
        *,
        external_oracle: Path | None = None,
        barrier: Callable[[str], None] = lambda stage: None,
    ) -> None:
        super().__init__(config.data_dir, barrier=barrier)
        self.config = config
        self.external_oracle = external_oracle
        self._owner: DbLifetime | None = None
        self._file_admission: RecoveryAdmission | None = None
        self._replacement: _PreparedReplacement | None = None
        self._prepared: _PreparedManualCompletion | None = None
        self._compact_result: CompactResult | None = None

    def compact(self) -> CompactResult:
        """Explicit confirmed compaction may qualify a manual hold release."""
        self._compact_result = None
        self._execute(compact=True)
        assert self._compact_result is not None
        return self._compact_result

    def _hold(self, reason: str) -> Never:
        message = (
            f"manual recovery held: {reason}; retain the recovery set "
            "and use a matching oracle or matched restore"
        )
        if not refusal_marker_path(self.data_dir).exists():
            write_refusal_marker(self.data_dir, message)
        raise DaemonStartupRefused(message)

    def _owner_for(self, lease: DbLease) -> DbLifetime:
        owner = self._owner
        assert owner is not None and lease.lineage[0] is owner, "manual offline authority expired"
        return owner

    def _admit(self, lease: DbLease, version: int | None) -> None:
        self._owner_for(lease)
        assert self._file_admission is not None
        self._file_admission.observe_schema(version)
        batch = read_events(self.data_dir)
        if batch.full:
            self._hold("failure-event spool is full; matched restore is required")
        if version is not None and version >= FENCE_VERSION:
            row = lease.call("manual:state_admission", load_repair_row)
            batch = import_legacy_events(
                self.data_dir, lease.call("manual:admitted_fence", require_schema_fence)
            )
            if decide_manual(self, row, batch) != Action.MANUAL:
                self._hold("durable state does not admit an explicit manual selection")

    def execute(self) -> IndexRebuildResult:
        return self._execute(compact=False)

    def _execute(self, *, compact: bool) -> IndexRebuildResult:
        with offline_file_lifetime(self.config) as (owner, admission):
            self._owner = owner
            self._file_admission = admission
            try:
                batch = read_events(self.data_dir)
                if batch.full:
                    self._hold("failure-event spool is full; matched restore is required")
                if self._refusal().value == "unreadable":
                    self._hold("the refusal marker is unreadable")
                with owned_connection(owner, "manual:source", self._open) as lease:
                    was_refused = refusal_marker_path(self.data_dir).exists()
                    if not was_refused:
                        write_refusal_marker(
                            self.data_dir,
                            "explicit manual recovery owns this database; "
                            "verified release is required",
                        )
                    row = lease.call("manual:state", load_repair_row)
                    marker = read_events(self.data_dir).classify(row)
                    if (
                        row.release_certificate is not None
                        and marker.value in {"none", "replay"}
                        and (was_refused or _records(read_events(self.data_dir)))
                    ):
                        if compact:
                            stats = estimate_bloat_ratio_on(lease, self.config.db_path)
                            self._compact_result = CompactResult(stats, stats, {}, 0, False, None)
                        self._release(lease, row)
                        return IndexRebuildResult(
                            dropped=0, created=0, healed=(), elapsed_seconds=0
                        )
                    require_replaceable_sidecar(self.config, lifetime=owner)
                    row = self._select(lease, row)
                    arm = self._arm_manual(lease, row)
                    self._barrier("after_manual_arm_commit")
                    if compact:
                        before = estimate_bloat_ratio_on(lease, self.config.db_path)
                        lease.call(
                            "manual:source_checkpoint",
                            lambda conn: conn.execute("CHECKPOINT").fetchone(),
                        )
                        rebuilt = None
                    else:
                        before = None
                        rebuilt = self.rebuild(lease, arm)
                if compact:
                    assert before is not None
                    started = time.monotonic()
                    rebuilt, copied, after = self._replace_candidate(owner, arm)
                    self._compact_result = CompactResult(
                        before, after, copied, time.monotonic() - started, True, None
                    )
                else:
                    assert rebuilt is not None
                    with owned_connection(
                        owner, "manual:fresh", lambda: self._open_rebuilt(rebuilt)
                    ) as lease:
                        self._verify_rebuild(lease, rebuilt)
                        if self.config.fts.backend == "sqlite_sidecar":
                            prepared = self._prepare_publication(lease, rebuilt)
                            self._commit_prepared(lease, prepared)
                    if self.config.fts.backend == "duckdb":
                        self._replace_candidate(owner, arm, rebuilt=rebuilt)
                with owned_connection(owner, "manual:release", self._open) as lease:
                    self._release(lease, lease.call("manual:release_state", load_repair_row))
                return rebuilt.result
            except SchemaTooNewError:
                raise
            except DaemonStartupRefused:
                raise
            except Exception as error:
                if isinstance(error, duckdb.IOException) and is_lock_conflict(error):
                    raise
                owner.observe(error, "manual:recovery")
                self._hold(str(error))
            finally:
                self._owner = None
                self._file_admission = None

    def _candidate_open(self, candidate: Path, *, read_only: bool = False):
        assert self._owner is not None
        return connect_existing_owned(
            replace(self.config, db_path=candidate),
            lifetime=self._owner,
            admission=self._admit,
            read_only=read_only,
        )

    def _replace_candidate(
        self,
        owner: DbLifetime,
        arm: ManualArmCapability,
        *,
        rebuilt: RebuiltCapability | None = None,
    ):
        from recall.db.schema import ensure_schema_on
        from recall.services.compaction import _clone_or_copy, _copy_database

        source = self.config.db_path
        with owned_connection(owner, "manual:candidate_source", self._open) as lease:
            row = self._matching_attempt(lease, arm, Event.MANUAL_STEP)
            source_storage = lease.call("manual:source_storage", storage_version)
            if source_storage is None:
                self._hold("source storage format cannot be verified for replacement")
            assert row.trigger_json is not None and row.trigger_json.manual_work is not None
            workspace = self.data_dir / row.trigger_json.manual_work.directory
            from recall.services.recreation import consistent_backup

            backup = lease.call(
                "manual:matched_original",
                lambda conn: consistent_backup(
                    conn,
                    self.config,
                    kind="manual-repair",
                    recovery=True,
                    observer=lease.observer,
                    destination=workspace / "original-set",
                ),
            )
        source_hash = _hash(source)
        self._barrier("after_manual_matched_backup")
        candidate = workspace / "candidate.duckdb"
        if candidate.exists() or Path(str(source) + ".wal").exists():
            self._hold("manual candidate requires a closed checkpointed source")
        copying = rebuilt is None
        if not copying:
            _clone_or_copy(source, candidate)
            self._barrier("after_manual_database_copy")
        copied: dict[str, int] = {}
        with owned_connection(
            owner,
            "manual:candidate",
            (
                lambda: create_copy_candidate_owned(
                    replace(self.config, db_path=candidate),
                    lifetime=owner,
                    storage_version=source_storage.removesuffix("+"),
                )
            )
            if copying
            else lambda: self._candidate_open(candidate),
        ) as lease:
            if copying:
                ensure_schema_on(lease, embed_dim=self.config.embedding.dimensions)
                preserving = replace(
                    self.config, fts=replace(self.config.fts, backend="sqlite_sidecar")
                )
                copied = lease.call(
                    "manual:copy_database", lambda conn: _copy_database(conn, preserving)
                )
                self._barrier("after_manual_database_copy")
            if lease.call("manual:candidate_state", load_repair_row) != row:
                self._hold("candidate did not preserve the running recovery instruction")
            if copying:
                self._source_content(lease, row)
                rebuilt = self.rebuild(lease, arm)
        assert rebuilt is not None
        with owned_connection(
            owner, "manual:fresh_candidate", lambda: self._candidate_open(candidate)
        ) as lease:
            self._verify_rebuild(lease, rebuilt)
            native = (
                NativeKeywordCandidate(candidate, lease.lineage, lease.connection_epoch)
                if self.config.fts.backend == "duckdb"
                else None
            )
            prepared = self._prepare_publication(lease, rebuilt, native=native, candidate=candidate)
            lease.call(
                "manual:candidate_checkpoint", lambda conn: conn.execute("CHECKPOINT").fetchone()
            )
        self._barrier("after_manual_candidate_close")
        with owned_connection(
            owner,
            "manual:candidate_verification",
            lambda: self._candidate_open(candidate, read_only=True),
        ) as lease:
            completed = lease.call("manual:candidate_verified_state", load_repair_row)
            self._validate_prepared(lease, prepared)
            if (
                lease.call("manual:candidate_targets", capture_existing_targets)
                != row.target_manifest
            ):
                self._hold("keyword candidate changed the saved secondary target catalog")
            self._matching_attempt(lease, prepared.arm, Event.MANUAL_STEP)
            after = estimate_bloat_ratio_on(lease, candidate)
        for path in (source, candidate):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        sync_directory(workspace)
        token = _PreparedReplacement(
            owner,
            candidate,
            source_hash,
            _hash(candidate),
            _hash(backup / "manifest.json"),
            completed,
        )
        self._replacement = token
        self._install_candidate(token)
        with owned_connection(
            owner, "manual:installed_completion", lambda: self._open_prepared(prepared)
        ) as lease:
            self._commit_prepared(lease, prepared)
        return rebuilt, copied, after

    def _install_candidate(self, token: _PreparedReplacement) -> None:
        assert token is self._replacement and token.owner is self._owner
        self._replacement = None
        source = self.config.db_path
        if _hash(source) != token.source_hash or _hash(token.candidate) != token.candidate_hash:
            self._hold("a closed manual replacement changed before installation")
        if any(Path(str(path) + ".wal").exists() for path in (source, token.candidate)):
            self._hold("manual replacement requires both native parents checkpointed and closed")
        backup = token.candidate.parent / "original-set"
        if (
            _hash(backup / "manifest.json") != token.backup_manifest_hash
            or _hash(backup / "recall.duckdb") != token.source_hash
        ):
            self._hold("the retained matched original changed before replacement")
        self._barrier("after_manual_original_backup")
        with token.owner.admitting(), serialized_events():
            self._barrier("before_manual_database_replace")
            assert token.row.trigger_json is not None
            captured = {
                event.event_id: event.content_hash for event in token.row.trigger_json.events
            }
            batch = read_events(self.data_dir)
            if batch.full or any(
                captured.get(event.event_id) != event.content_hash for event in _records(batch)
            ):
                self._hold("an event outside the manual attempt prevents replacement")
            token.candidate.replace(source)
            self._barrier("after_manual_database_replace")
            sync_directory(source.parent)
            sync_directory(token.candidate.parent)
            self._barrier("after_manual_database_replace_sync")

    def _open(self) -> duckdb.DuckDBPyConnection:
        assert self._owner is not None
        try:
            return connect_owned(
                self.config, lifetime=self._owner, admission=self._admit, lenient_schema=True
            )
        except duckdb.IOException as error:
            if is_lock_conflict(error):
                raise
            if not Path(str(self.config.db_path) + ".wal").exists():
                raise
            return self._open_from_diagnosed_wal()

    def _open_rebuilt(self, rebuilt: RebuiltCapability) -> duckdb.DuckDBPyConnection:
        assert self._owner is not None

        def admit(lease: DbLease, version: int | None) -> None:
            self._admit(lease, version)
            self.admit_reopen(lease, rebuilt)

        return connect_recovery_owned(self.config, lifetime=self._owner, admission=admit)

    def _workspace(self) -> Path:
        root = self.data_dir / "repair-work"
        existed = root.exists()
        root.mkdir(exist_ok=True)
        if not existed:
            sync_directory(self.data_dir)
        from itertools import islice

        if len(tuple(islice(root.iterdir(), WORKSPACES_MAX + 1))) >= WORKSPACES_MAX:
            self._hold("manual archive capacity is full; preserve and archive prior recovery sets")
        path = root / uuid4().hex
        path.mkdir()
        sync_directory(root)
        return path

    def _open_from_diagnosed_wal(self) -> duckdb.DuckDBPyConnection:
        assert self._owner is not None
        owner = self._owner
        workspace = self._workspace()
        original = self.config.db_path
        wal = Path(str(original) + ".wal")
        before = {path: _hash(path) for path in (original, wal)}
        for path in before:
            shutil.copyfile(path, workspace / path.name)
        if any(_hash(path) != digest for path, digest in before.items()):
            self._hold("database/WAL changed during manual diagnosis")
        copy = workspace / original.name
        with owned_connection(
            owner, "manual:diagnostic_copy", lambda: duckdb.connect(str(copy))
        ) as lease:
            lease.call(
                "manual:diagnostic_pragmas", lambda conn: _apply_runtime_pragmas(conn, self.config)
            )
            version = lease.call("manual:diagnostic_schema", check_schema_compatibility)
            self._admit(lease, version)
            row = lease.call("manual:diagnostic_state", load_repair_row)
            if row.release_certificate is not None:
                self._validate_release(lease, row)
            else:
                self._source_content(lease, row)
        for path in (copy, Path(str(copy) + ".wal")):
            if path.exists():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        sync_directory(workspace)
        conn = owner.call_db("manual:writer_after_diagnosis", lambda: duckdb.connect(str(original)))
        try:
            with owner.lease(conn) as lease:
                lease.call(
                    "manual:writer_pragmas", lambda raw: _apply_runtime_pragmas(raw, self.config)
                )
                self._admit(lease, lease.call("manual:writer_schema", check_schema_compatibility))
                if lease.call("manual:writer_state", load_repair_row) != row:
                    self._hold("manual writer differs from its diagnosed recovery state")
        except BaseException as primary:
            owner.observe(primary, "manual:writer_after_diagnosis")
            try:
                owner.cleanup_db("manual:writer_after_diagnosis_close", conn.close)
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "manual open and close failed", [primary, cleanup]
                ) from None
            raise
        return conn

    def _reference(self, row: RepairRow) -> tuple[OracleReference, Path]:
        if row.oracle_ref is None or row.oracle_hash is None or row.trigger_id is None:
            self._hold("unfinished work lacks its saved exact oracle")
        reference = OracleReference(trigger_id=row.trigger_id, manifest_hash=row.oracle_hash)
        directory = self.data_dir / row.oracle_ref
        try:
            read_oracle(directory, reference)
        except (OSError, ValueError, OracleUnavailable):
            if self.external_oracle is None:
                raise
            self._restore_reference(row, reference, directory)
        return reference, directory

    def _publication(self, row: RepairRow, binding: PreparedPublication) -> PublicationOracle:
        try:
            publication = read_publication_at(self.data_dir / binding.path, binding.manifest_hash)
        except (OSError, ValueError, OracleUnavailable):
            if self.external_oracle is None:
                raise
            self._restore_publication(row, binding)
            publication = read_publication_at(self.data_dir / binding.path, binding.manifest_hash)
        if (
            publication.attempt_no != binding.attempt_no
            or publication.attempt_no > row.attempt_no
            or publication.trigger_id != row.trigger_id
            or publication.oracle_hash != row.oracle_hash
        ):
            self._hold("prepared publication differs from the saved canonical instruction")
        return publication

    def _restore_publication(self, row: RepairRow, binding: PreparedPublication) -> None:
        assert self.external_oracle is not None
        source = self.external_oracle.expanduser().resolve() / "publication"
        publication = read_publication_at(source, binding.manifest_hash)
        if (
            publication.attempt_no != binding.attempt_no
            or publication.trigger_id != row.trigger_id
            or publication.oracle_hash != row.oracle_hash
        ):
            self._hold("external publication does not match the unfinished instruction")
        workspace = self._workspace()
        candidate = workspace / "publication"
        self._copy_publication(source, candidate, publication, binding.manifest_hash)
        target = self.data_dir / binding.path
        if not target.parent.exists():
            from itertools import islice

            root = self.data_dir / "repair-work"
            if (
                target.parent.parent != root
                or len(tuple(islice(root.iterdir(), WORKSPACES_MAX))) >= WORKSPACES_MAX
            ):
                self._hold("the recorded publication parent is unavailable")
            target.parent.mkdir()
            sync_directory(root)
        if target.exists():
            target.rename(workspace / "publication-original")
            sync_directory(target.parent)
            sync_directory(workspace)
        candidate.rename(target)
        sync_directory(target.parent)
        sync_directory(workspace)
        read_publication_at(target, binding.manifest_hash)

    def _source_content(self, lease: DbLease, row: RepairRow) -> None:
        self._owner_for(lease)
        reference, directory = self._reference(row)
        work = None if row.trigger_json is None else row.trigger_json.manual_work
        binding = None if work is None else work.publication
        if binding is None:
            lease.call(
                "manual:original_content",
                lambda conn: verify_manual_content(
                    conn, directory, reference, observer=lease.observer
                ),
            )
        else:
            publication = self._publication(row, binding)
            lease.call(
                "manual:prepared_content",
                lambda conn: verify_published_canonical(
                    conn,
                    directory,
                    reference,
                    publication,
                    publication_directory=self.data_dir / binding.path,
                    observer=lease.observer,
                ),
            )

    def _archive_publication(self, row: RepairRow, workspace: Path) -> PreparedPublication | None:
        work = None if row.trigger_json is None else row.trigger_json.manual_work
        binding = None if work is None else work.publication
        if binding is None:
            return None
        publication = self._publication(row, binding)
        source = self.data_dir / binding.path
        destination = workspace / "publication"
        self._copy_publication(source, destination, publication, binding.manifest_hash)
        return binding.model_copy(update={"path": str(destination.relative_to(self.data_dir))})

    def _copy_publication(
        self, source: Path, destination: Path, publication: PublicationOracle, digest: str
    ) -> None:
        files = (
            "manifest.json",
            *(
                table.filename
                for table in (
                    publication.pending,
                    *publication.sidecar.tables,
                    *publication.native_keywords,
                )
            ),
        )
        destination.mkdir()
        for name in files:
            shutil.copyfile(source / name, destination / name)
            with (destination / name).open("rb") as stream:
                os.fsync(stream.fileno())
        sync_directory(destination)
        sync_directory(destination.parent)
        read_publication_at(destination, digest)

    def _restore_reference(
        self, row: RepairRow, reference: OracleReference, directory: Path
    ) -> None:
        assert self.external_oracle is not None
        source = self.external_oracle.expanduser().resolve()
        manifest = read_oracle(source, reference)
        publication = (
            None
            if row.release_certificate is None
            else read_publication(source, row.release_certificate.publication_hash)
        )
        workspace = self._workspace()
        candidate = workspace / "oracle-candidate"
        candidate.mkdir()
        tables = manifest.tables + (() if manifest.sidecar is None else manifest.sidecar.tables)
        files = ("manifest.json", *(table.filename for table in tables))
        for name in files:
            shutil.copyfile(source / name, candidate / name)
        if publication is not None:
            assert row.release_certificate is not None
            self._copy_publication(
                source / "publication",
                candidate / "publication",
                publication,
                row.release_certificate.publication_hash,
            )
        for name in files:
            with (candidate / name).open("rb") as stream:
                os.fsync(stream.fileno())
        sync_directory(candidate)
        read_oracle(candidate, reference)
        if publication is not None:
            assert row.release_certificate is not None
            read_publication(candidate, row.release_certificate.publication_hash)
        existed = directory.parent.exists()
        directory.parent.mkdir(exist_ok=True)
        if not existed:
            sync_directory(self.data_dir)
        if directory.exists():
            directory.rename(workspace / "oracle-original")
            sync_directory(directory.parent)
            sync_directory(workspace)
        candidate.rename(directory)
        sync_directory(directory.parent)
        read_oracle(directory, reference)

    def _select(self, lease: DbLease, row: RepairRow) -> RepairRow:
        self._owner_for(lease)
        fence = lease.call("manual:schema_fence", require_schema_fence)
        batch = import_legacy_events(self.data_dir, fence)
        if decide_manual(self, row, batch) != Action.MANUAL:
            self._hold("durable state does not admit manual selection")
        if row.phase in {"required", "running"}:
            self._source_content(lease, row)
            row = self._fold(lease, batch)
            assert row.trigger_json is not None
            captured = {event.event_id: event for event in row.trigger_json.events}
            captured.update(
                {
                    event.event_id: CapturedEvent(
                        event_id=event.event_id, content_hash=event.content_hash
                    )
                    for event in _records(batch)
                }
            )
            instruction = row.trigger_json.model_copy(update={"events": tuple(captured.values())})
            return _changed(row, trigger_json=instruction)
        row = self._fold(lease, batch)
        identity = uuid4()
        store = OracleStore(self.data_dir)
        store.reconcile(row, identity)
        directory = self.data_dir / "repair-oracles" / str(identity)
        reference = lease.call(
            "manual:capture_oracle",
            lambda conn: capture_oracle(
                conn,
                directory,
                identity,
                sidecar=self.data_dir / "recall.fts.sqlite",
                barrier=self._barrier,
                observer=lease.observer,
            ),
        )
        manifest = read_oracle(directory, reference)
        instruction = TriggerInstruction(
            trigger_id=identity,
            kind="manual",
            events=tuple(
                CapturedEvent(event_id=event.event_id, content_hash=event.content_hash)
                for event in _records(batch)
            ),
        )
        required = _changed(
            row,
            phase="required",
            trigger_id=identity,
            trigger_kind="manual",
            signature=next(
                (event.failure.signature for event in _records(batch) if event.failure is not None),
                None,
            ),
            trigger_json=instruction,
            target_manifest=manifest.targets,
            oracle_ref=f"repair-oracles/{identity}",
            oracle_hash=reference.manifest_hash,
            release_certificate=None,
        )
        with lease.transaction("manual:selection") as transaction:
            transaction.call("manual:required", lambda conn: save_repair_row(conn, required))
        store.reconcile(required)
        return required

    def _arm_manual(self, lease: DbLease, row: RepairRow) -> ManualArmCapability:
        from recall.core.repair_state import decide

        self._owner_for(lease)
        if (
            decide(self._admission(row, read_events(self.data_dir)), Event.MANUAL_ARM)
            != Action.MANUAL_ARM
        ):
            self._hold("manual arm requires required/running with a consistent usable budget")
        reference, directory = self._reference(row)
        manifest = read_oracle(directory, reference)
        if manifest.targets != row.target_manifest:
            self._hold("saved target manifest differs from the exact oracle")
        number = AttemptNumber(value=row.attempt_no + 1)
        required = sum(
            table.byte_count
            for table in manifest.tables
            + (() if manifest.sidecar is None else manifest.sidecar.tables)
        )
        required += 3 * self.config.db_path.stat().st_size
        old_work = None if row.trigger_json is None else row.trigger_json.manual_work
        if old_work is not None and old_work.publication is not None:
            prior = self._publication(row, old_work.publication)
            required += 2 * sum(
                table.byte_count
                for table in (prior.pending, *prior.sidecar.tables, *prior.native_keywords)
            )
        if shutil.disk_usage(self.data_dir).free < required:
            self._hold("insufficient disk capacity for the complete manual working set")
        workspace = self._workspace()
        assert row.trigger_json is not None
        work = ManualWork(
            attempt_no=number.value,
            directory=str(workspace.relative_to(self.data_dir)),
            publication=self._archive_publication(row, workspace),
        )
        instruction = row.trigger_json.model_copy(update={"manual_work": work})
        running = _changed(
            row,
            phase="running",
            attempt_no=number.value,
            trigger_json=instruction,
            success_since_attempt=False,
            success_since_failure=False,
            release_certificate=None,
        )
        if not refusal_marker_path(self.data_dir).exists():
            write_refusal_marker(
                self.data_dir, "manual recovery is running; verified manual release is required"
            )
        with lease.transaction("manual:arm") as transaction:
            transaction.call("manual:running", lambda conn: save_repair_row(conn, running))
        OracleStore(self.data_dir).retire_unreferenced_publication(running)
        cap = ManualArmCapability(
            self._identity,
            lease.lineage,
            lease.connection_epoch,
            number,
            reference.trigger_id,
            reference.manifest_hash,
        )
        self._live_attempt = cap
        return cap

    def _verify_rebuild(self, lease: DbLease, rebuilt: RebuiltCapability) -> None:
        self._owner_for(lease)
        if rebuilt is not self._rebuilt:
            self._hold("manual verification lacks the current live rebuilt capability")
        if lease.connection_epoch is rebuilt._arm._connection_epoch:
            self._hold("manual verification requires a drained fresh native parent")
        row = self._matching_attempt(lease, rebuilt._arm, Event.MANUAL_RELEASE)
        if lease.call("manual:fresh_targets", capture_existing_targets) != row.target_manifest:
            self._hold("fresh manual index catalog differs from the complete saved targets")
        self._source_content(lease, row)
        probe = self._probe(lease)
        self.last_probe = probe
        if probe.diverged:
            self._hold("fresh manual probe still finds index divergence")
        self._barrier("after_manual_fresh_postchecks")

    def _prepare_publication(
        self,
        lease: DbLease,
        rebuilt: RebuiltCapability,
        *,
        native: NativeKeywordCandidate | None = None,
        candidate: Path | None = None,
    ) -> _PreparedManualCompletion:
        owner = self._owner_for(lease)
        if rebuilt is not self._rebuilt:
            self._hold("manual publication capability was already consumed")
        self._rebuilt = None
        row = self._matching_attempt(lease, rebuilt._arm, Event.MANUAL_RELEASE)
        reference, directory = self._reference(row)
        assert row.trigger_json is not None and row.trigger_json.manual_work is not None
        workspace = self.data_dir / row.trigger_json.manual_work.directory
        if self.config.fts.backend == "duckdb":
            assert native is not None, "native publication requires the current candidate"
            publish_native_keywords(lease, self.config, native)
        else:
            assert native is None
            publish_manual_keywords(lease, self.config, workspace, lifetime=owner)
        assert row.target_manifest is not None
        present = lease.call("manual:published_targets", capture_existing_targets)
        missing = {
            (target.schema_name, target.index_name) for target in row.target_manifest.targets
        } - {(target.schema_name, target.index_name) for target in present.targets}
        for target in row.target_manifest.targets:
            if (target.schema_name, target.index_name) in missing:
                lease.call(
                    "manual:restore_keyword_target",
                    lambda conn, sql=target.sql: conn.execute(sql).fetchone(),
                )
        if (
            lease.call("manual:complete_published_targets", capture_existing_targets)
            != row.target_manifest
        ):
            self._hold("manual keyword publication changed the saved target catalog")
        self._barrier("after_manual_keyword_publication")
        verify_manual_keywords(lease, self.config, lifetime=owner)
        digest = lease.call(
            "manual:publication_reference",
            lambda conn: capture_publication(
                conn,
                directory,
                reference,
                row.attempt_no,
                sidecar=self.data_dir / "recall.fts.sqlite",
                barrier=self._barrier,
                backend=self.config.fts.backend,
                fields=self.config.fts.fields,
                observer=lease.observer,
            ),
        )
        publication = read_publication(directory, digest)
        lease.call(
            "manual:post_publication_content",
            lambda conn: verify_published_canonical(
                conn,
                directory,
                reference,
                publication,
                observer=lease.observer,
                publication_directory=directory / "publication",
            ),
        )
        binding = PreparedPublication(
            attempt_no=row.attempt_no,
            manifest_hash=digest,
            path=str((directory / "publication").relative_to(self.data_dir)),
        )
        work = row.trigger_json.manual_work.model_copy(update={"publication": binding})
        instruction = row.trigger_json.model_copy(update={"manual_work": work})
        prepared = _changed(row, trigger_json=instruction)
        with lease.transaction("manual:prepared_publication") as transaction:
            transaction.call(
                "manual:prepared_binding", lambda conn: save_repair_row(conn, prepared)
            )
        assert isinstance(rebuilt._arm, ManualArmCapability)
        capability = _PreparedManualCompletion(
            rebuilt._arm, binding, lease.connection_epoch, candidate
        )
        self._prepared = capability
        self._barrier("after_manual_publication_binding_commit")
        return capability

    def _validate_prepared(
        self, lease: DbLease, capability: _PreparedManualCompletion
    ) -> tuple[RepairRow, PublicationOracle]:
        self._owner_for(lease)
        if capability is not self._prepared:
            self._hold("manual completion lacks its unconsumed publication capability")
        row = self._matching_attempt(lease, capability.arm, Event.MANUAL_RELEASE)
        assert row.trigger_json is not None and row.trigger_json.manual_work is not None
        if row.trigger_json.manual_work.publication != capability.publication:
            self._hold("running publication binding differs from its live capability")
        publication = self._publication(row, capability.publication)
        if publication.backend != self.config.fts.backend or publication.fields != tuple(
            sorted(set(self.config.fts.fields))
        ):
            self._hold("prepared keyword field scope changed")
        self._source_content(lease, row)
        if lease.call("manual:prepared_targets", capture_existing_targets) != row.target_manifest:
            self._hold("prepared candidate catalog differs from the saved targets")
        _reference, directory = self._reference(row)
        verify_publication_sidecar(
            directory, publication, self.data_dir / "recall.fts.sqlite", observer=lease.observer
        )
        verify_manual_keywords(lease, self.config, lifetime=self._owner_for(lease))
        return row, publication

    def _open_prepared(self, capability: _PreparedManualCompletion):
        assert self._owner is not None

        def admit(lease: DbLease, version: int | None) -> None:
            self._admit(lease, version)
            if capability is not self._prepared:
                self._hold("installed reopen lacks the live prepared capability")
            self._matching_attempt(lease, capability.arm, Event.MANUAL_STEP)

        return connect_recovery_owned(self.config, lifetime=self._owner, admission=admit)

    def _commit_prepared(self, lease: DbLease, capability: _PreparedManualCompletion) -> None:
        row, publication = self._validate_prepared(lease, capability)
        if capability.candidate is not None and lease.connection_epoch is capability.epoch:
            self._hold("manual completion requires the fresh installed incarnation")
        actual = lease.call(
            "manual:installed_identity",
            lambda conn: conn.execute(
                "SELECT path FROM duckdb_databases() WHERE database_name=current_database()"
            ).fetchone(),
        )
        if actual is None or Path(actual[0]).resolve() != self.config.db_path.resolve():
            self._hold("manual completion cannot retire intents inside a candidate")
        self._prepared = None
        reference, _directory = self._reference(row)
        digest = capability.publication.manifest_hash
        assert row.trigger_json is not None
        roster = "_recall_manual_pending_" + uuid4().hex
        lease.call(
            "manual:capture_pending_rowids",
            lambda conn: conn.execute(
                f"CREATE TEMP TABLE {roster} AS "
                "SELECT rowid AS saved_rowid,kind,id,op,queued_at FROM fts_sidecar_pending"
            ).fetchone(),
        )
        cert = ReleaseCertificate(
            attempt_no=row.attempt_no,
            trigger_id=reference.trigger_id,
            oracle_hash=reference.manifest_hash,
            publication_hash=digest,
            keyword_generation=publication.keyword_generation,
            events=row.trigger_json.events,
        )
        captured = {event.event_id: event.content_hash for event in row.trigger_json.events}
        acknowledged = tuple(
            entry.model_copy(update={"disposition": "verified"})
            if captured.get(entry.event_id) == entry.content_hash
            else entry
            for entry in row.acknowledged_events
        )
        completed = _changed(
            row,
            phase="verified",
            success_since_attempt=True,
            success_since_failure=True,
            release_certificate=cert,
            acknowledged_events=acknowledged,
        )
        with lease.admitting(), serialized_events():
            self._matching_attempt(lease, capability.arm, Event.MANUAL_RELEASE)
            with lease.transaction("manual:completion") as transaction:
                transaction.call(
                    "manual:retire_intents",
                    lambda conn: conn.execute(
                        f"DELETE FROM fts_sidecar_pending p USING {roster} r "
                        "WHERE p.rowid=r.saved_rowid AND p.kind=r.kind AND p.id=r.id "
                        "AND p.op=r.op AND p.queued_at=r.queued_at"
                    ).fetchone(),
                )
                remaining = transaction.call(
                    "manual:pending_empty",
                    lambda conn: conn.execute(
                        "SELECT count(*) FROM fts_sidecar_pending"
                    ).fetchone(),
                )
                if remaining != (0,):
                    self._hold(
                        "manual publication did not retire its exact captured intent membership"
                    )
                self._barrier("after_manual_intent_retirement")
                transaction.call(
                    "manual:diagnostics",
                    lambda conn: record_index_repair(
                        conn, signature=row.signature, repaired_at=self._clock()
                    ),
                )
                transaction.call(
                    "manual:certificate", lambda conn: save_repair_row(conn, completed)
                )
                if self.last_probe is not None and self.last_probe.complete:
                    transaction.call(
                        "manual:negative_verification",
                        lambda conn: set_needs_index_verification(conn, False),
                    )
                self._barrier("before_manual_completion_commit")
                self._matching_attempt(lease, capability.arm, Event.MANUAL_RELEASE)
            self._barrier("after_manual_completion_commit")

    def _validate_release(self, lease: DbLease, row: RepairRow) -> None:
        self._owner_for(lease)
        cert = row.release_certificate
        if cert is None:
            self._hold("manual release lacks its matching publication certificate")
        reference, directory = self._reference(row)
        publication = read_publication(directory, cert.publication_hash)
        if (
            publication.attempt_no != row.attempt_no
            or publication.oracle_hash != cert.oracle_hash
            or publication.keyword_generation != cert.keyword_generation
            or publication.backend != self.config.fts.backend
            or publication.fields != tuple(sorted(set(self.config.fts.fields)))
        ):
            self._hold("manual release publication certificate differs from this attempt")
        lease.call(
            "manual:released_content",
            lambda conn: verify_manual_content(
                conn, directory, reference, publication=publication, observer=lease.observer
            ),
        )
        verify_publication_sidecar(
            directory, publication, self.data_dir / "recall.fts.sqlite", observer=lease.observer
        )
        verify_manual_keywords(lease, self.config, lifetime=self._owner_for(lease))

    def _release(self, lease: DbLease, row: RepairRow) -> None:
        self._validate_release(lease, row)
        with lease.admitting(), serialized_events():
            self._require_release(lease, row)
            batch = read_events(self.data_dir)
            records = batch.secondary
            if batch.primary is not None:
                records = ((self.data_dir / "daemon-failure.json", batch.primary), *records)
            for path, _event in records:
                self._barrier("before_manual_event_unlink")
                self._require_release(lease, row)
                path.unlink()
                sync_directory(path.parent)
                self._barrier("after_manual_event_unlink_sync")
            self._require_release(lease, row)
            retained = self._retained_ledger(row, read_events(self.data_dir))
            if retained != row.acknowledged_events:
                row = _changed(row, acknowledged_events=retained)
                with lease.transaction("manual:retired_events") as transaction:
                    transaction.call(
                        "manual:retired_ledger", lambda conn: save_repair_row(conn, row)
                    )
            self._barrier("before_manual_refusal_unlink")
            self._require_release(lease, row)
            refusal_marker_path(self.data_dir).unlink(missing_ok=True)
            sync_directory(self.data_dir)
            self._barrier("after_manual_refusal_unlink_sync")

    def _require_release(self, lease: DbLease, row: RepairRow) -> None:
        from recall.core.repair_state import decide

        current = lease.call("manual:release_current", load_repair_row)
        if current != row:
            self._hold("manual release state changed")
        assert row.trigger_json is not None
        captured = {event.event_id: event.content_hash for event in row.trigger_json.events}
        batch = read_events(self.data_dir)
        outside = any(
            captured.get(event.event_id) != event.content_hash for event in _records(batch)
        )
        if outside:
            folded = self._fold(lease, batch)
            held = _changed(folded, success_since_attempt=False, success_since_failure=False)
            with lease.transaction("manual:late_event_hold") as transaction:
                transaction.call("manual:spent_hold", lambda conn: save_repair_row(conn, held))
            self._hold("an event outside the manual attempt prevents release")
        if decide(super()._admission(row, batch), Event.MANUAL_RELEASE) != Action.MANUAL_RELEASE:
            self._hold("current events or refusal state do not admit manual release")


def decide_manual(protocol: RepairProtocol, row: RepairRow, batch):
    from recall.core.repair_state import decide

    return decide(RepairProtocol._admission(protocol, row, batch), Event.MANUAL_REBUILD)


def manual_repair_context(config: AppConfig) -> bool:
    """Inspect offline state without granting execution or changing diagnostics."""
    if refusal_marker_path(config.data_dir).exists() or read_events(config.data_dir).identified():
        return True
    if not config.db_path.is_file():
        return False
    with (
        offline_file_lifetime(config) as (owner, admission),
        owned_connection(
            owner,
            "manual:context",
            lambda: connect_existing_owned(
                config,
                lifetime=owner,
                admission=lambda lease, version: admission.observe_schema(version),
                read_only=True,
            ),
        ) as lease,
    ):
        version = lease.call("manual:context_schema", check_schema_compatibility)
        if version is None or version < FENCE_VERSION:
            return False
        return lease.call("manual:context_state", load_repair_row).phase in {
            "required",
            "running",
            "refused",
        }


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
