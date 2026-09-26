from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from recall.core.config import AppConfig

logger = logging.getLogger(__name__)

DEFAULT_GC_DAYS = 7
SECONDS_PER_DAY = 86_400
RECALL_SIDECAR_INTRODUCED_AT = datetime(2026, 5, 28, tzinfo=UTC)
# consistent_backup names durable recovery artifacts with these prefixes.
# Snapshot age cannot prove a failed job no longer needs its original backup.
# Retain incomplete artifacts too; only explicit operator removal retires them.
_RECOVERY_BACKUP_PREFIXES = ("index-migration-", "storage-migration-", "recreate-")


@dataclass(frozen=True)
class SnapshotGcResult:
    removed_paths: tuple[str, ...] = ()
    kept_paths: tuple[str, ...] = ()
    failed_paths: tuple[str, ...] = ()
    partial_paths: tuple[str, ...] = ()
    total_bytes_freed: int = 0
    dry_run: bool = False
    snapshots_dir_missing: bool = False


@dataclass(frozen=True)
class _SnapshotGcCandidate:
    paths: tuple[Path, ...]
    result_paths: tuple[str, ...]
    mtime: float
    partial: bool = False


@dataclass(frozen=True)
class _SnapshotScan:
    """What one pass over the snapshots directory found.

    `gc` and `list` must agree on what a snapshot *is* — which files pair into
    one entry, which directories are recovery backups, which halves are partial
    — or a listing describes a grouping the pruner does not use. One scan
    answers both; only the decision to delete belongs to `gc`.
    """

    candidates: tuple[_SnapshotGcCandidate, ...]
    retained: tuple[Path, ...]
    kept_paths: tuple[str, ...]
    failed_paths: tuple[str, ...]


@dataclass(frozen=True)
class SnapshotEntry:
    paths: tuple[str, ...]
    total_bytes: int
    age_seconds: float
    retained: bool
    partial: bool


@dataclass(frozen=True)
class SnapshotInventory:
    snapshots_dir: str
    snapshots_dir_missing: bool = False
    entries: tuple[SnapshotEntry, ...] = ()
    entry_count: int = 0
    total_bytes: int = 0
    unreadable_paths: tuple[str, ...] = ()


def _snapshots_dir(config: AppConfig) -> Path:
    return config.data_dir / "snapshots"


def _scan_snapshots(snapshots_root: Path, entries: list[Path]) -> _SnapshotScan:
    """Group the directory's top-level entries into snapshots, without deciding."""
    kept: list[str] = []
    failed: list[str] = []
    retained: list[Path] = []
    sibling_groups: dict[str, dict[str, Path]] = {}
    directories: list[Path] = []
    legacy_entries: list[Path] = []

    for entry in entries:
        inside_snapshots = _entry_stays_inside(entry, snapshots_root)
        if inside_snapshots is None:
            failed.append(str(entry))
            continue
        if not inside_snapshots:
            logger.warning("snapshots gc refusing to touch %s (outside snapshots dir)", entry)
            kept.append(str(entry))
            retained.append(entry)
            continue

        if entry.is_dir() and not entry.is_symlink():
            if entry.name.startswith(_RECOVERY_BACKUP_PREFIXES):
                kept.append(str(entry))
                retained.append(entry)
                continue
            directories.append(entry)
            continue
        if _is_duckdb_wal(entry):
            continue
        sibling_kind = _sibling_snapshot_kind(entry)
        if sibling_kind is not None:
            basename, kind = sibling_kind
            sibling_groups.setdefault(basename, {})[kind] = entry
            continue
        legacy_entries.append(entry)

    candidates: list[_SnapshotGcCandidate] = []
    for entry in directories:
        candidate = _directory_snapshot_candidate(entry)
        if candidate is None:
            failed.append(str(entry))
            continue
        candidates.append(candidate)

    for basename in sorted(sibling_groups):
        candidate = _sibling_snapshot_candidate(sibling_groups[basename])
        if candidate is None:
            failed.extend(str(path) for path in sibling_groups[basename].values())
            continue
        candidates.append(candidate)

    for entry in legacy_entries:
        try:
            entry_mtime = _top_level_mtime(entry)
        except OSError as err:
            logger.warning("snapshots gc could not stat %s: %s", entry, err)
            failed.append(str(entry))
            continue
        candidates.append(
            _SnapshotGcCandidate(paths=(entry,), result_paths=(str(entry),), mtime=entry_mtime)
        )

    return _SnapshotScan(
        candidates=tuple(candidates),
        retained=tuple(retained),
        kept_paths=tuple(kept),
        failed_paths=tuple(failed),
    )


def list_snapshots(config: AppConfig) -> SnapshotInventory:
    """Report every snapshot entry on disk with its size and age.

    `snapshots gc --dry-run` was the only way to see what is there, which makes
    an operator ask a destructive command what it would destroy — and it never
    shows the recovery backups gc retains, which are the entries a rollback
    depends on. This reads; it removes nothing.
    """
    snapshots_dir = _snapshots_dir(config)
    if not snapshots_dir.exists():
        return SnapshotInventory(snapshots_dir=str(snapshots_dir), snapshots_dir_missing=True)
    if snapshots_dir.is_symlink() or not snapshots_dir.is_dir():
        logger.warning("snapshots list refusing snapshots dir path=%s", snapshots_dir)
        return SnapshotInventory(snapshots_dir=str(snapshots_dir))

    snapshots_root = snapshots_dir.resolve()
    try:
        entries = sorted(snapshots_dir.iterdir())
    except OSError as err:
        logger.warning("snapshots list could not list %s: %s", snapshots_dir, err)
        return SnapshotInventory(
            snapshots_dir=str(snapshots_dir), unreadable_paths=(str(snapshots_dir),)
        )

    scan = _scan_snapshots(snapshots_root, entries)
    now = time.time()
    unreadable = list(scan.failed_paths)
    listed: list[SnapshotEntry] = [
        SnapshotEntry(
            paths=candidate.result_paths,
            total_bytes=sum(_entry_size(path) for path in candidate.paths),
            age_seconds=max(0.0, now - candidate.mtime),
            retained=False,
            partial=candidate.partial,
        )
        for candidate in scan.candidates
    ]
    for path in scan.retained:
        try:
            mtime = _top_level_mtime(path)
        except OSError as err:
            logger.warning("snapshots list could not stat %s: %s", path, err)
            unreadable.append(str(path))
            continue
        listed.append(
            SnapshotEntry(
                paths=(str(path),),
                total_bytes=_entry_size(path),
                age_seconds=max(0.0, now - mtime),
                retained=True,
                partial=False,
            )
        )

    # Oldest first: the entries `gc` would take next are the ones an operator
    # reading a disk-usage question wants at the top.
    listed.sort(key=lambda entry: (-entry.age_seconds, entry.paths))
    return SnapshotInventory(
        snapshots_dir=str(snapshots_dir),
        entries=tuple(listed),
        entry_count=len(listed),
        total_bytes=sum(entry.total_bytes for entry in listed),
        unreadable_paths=tuple(unreadable),
    )


def gc_snapshots(
    config: AppConfig,
    *,
    days: int = DEFAULT_GC_DAYS,
    dry_run: bool = False,
) -> SnapshotGcResult:
    """Prune stale top-level snapshot entries without leaving the snapshots dir."""
    snapshots_dir = _snapshots_dir(config)
    if not snapshots_dir.exists():
        return SnapshotGcResult(dry_run=dry_run, snapshots_dir_missing=True)
    if snapshots_dir.is_symlink() or not snapshots_dir.is_dir():
        logger.warning("snapshots gc refusing snapshots dir path=%s", snapshots_dir)
        return SnapshotGcResult(dry_run=dry_run)

    snapshots_root = snapshots_dir.resolve()
    try:
        entries = sorted(snapshots_dir.iterdir())
    except OSError as err:
        logger.warning("snapshots gc could not list %s: %s", snapshots_dir, err)
        return SnapshotGcResult(failed_paths=(str(snapshots_dir),), dry_run=dry_run)

    scan = _scan_snapshots(snapshots_root, entries)
    cutoff = time.time() - (days * SECONDS_PER_DAY)
    removed: list[str] = []
    kept: list[str] = list(scan.kept_paths)
    failed: list[str] = list(scan.failed_paths)
    partial: list[str] = [
        result_path
        for candidate in scan.candidates
        if candidate.partial
        for result_path in candidate.result_paths
    ]
    total_bytes_freed = 0

    for path in partial:
        logger.warning("snapshots gc refusing partial snapshot path=%s", path)

    removable = [candidate for candidate in scan.candidates if not candidate.partial]
    for candidate in sorted(removable, key=lambda item: item.result_paths):
        if candidate.mtime >= cutoff:
            kept.extend(candidate.result_paths)
            continue

        size = sum(_entry_size(path) for path in candidate.paths)
        if dry_run:
            # Dry-run reports deletion intent; size is best-effort because
            # unreadable paths may still be candidates.
            removed.extend(candidate.result_paths)
            total_bytes_freed += size
            continue

        try:
            for path in candidate.paths:
                _remove(path)
        except OSError as err:
            logger.warning(
                "snapshots gc failed to remove %s: %s",
                ", ".join(candidate.result_paths),
                err,
            )
            failed.extend(candidate.result_paths)
            continue

        remaining = [path for path in candidate.paths if path.exists()]
        if remaining:
            logger.warning(
                "snapshots gc could not delete %s (still present)",
                ", ".join(str(path) for path in remaining),
            )
            failed.extend(str(path) for path in remaining)
            continue

        removed.extend(candidate.result_paths)
        total_bytes_freed += size

    return SnapshotGcResult(
        removed_paths=tuple(removed),
        kept_paths=tuple(kept),
        failed_paths=tuple(failed),
        partial_paths=tuple(partial),
        total_bytes_freed=total_bytes_freed,
        dry_run=dry_run,
    )


def _sibling_snapshot_kind(entry: Path) -> tuple[str, str] | None:
    name = entry.name
    if name.endswith(".duckdb"):
        return name.removesuffix(".duckdb"), "duckdb"
    if name.endswith(".fts.sqlite"):
        return name.removesuffix(".fts.sqlite"), "sidecar"
    return None


def _is_duckdb_wal(entry: Path) -> bool:
    return entry.name.endswith(".duckdb.wal")


def _directory_snapshot_candidate(entry: Path) -> _SnapshotGcCandidate | None:
    """One directory entry as a snapshot, or None when it cannot be statted.

    A directory holding only one half of the DuckDB/sidecar pair is `partial`:
    it is still a snapshot to report, but never one to delete, because the
    missing half means a restore from it would be incomplete.
    """
    duckdb_path = entry / "recall.duckdb"
    sidecar_path = entry / "recall.fts.sqlite"
    has_duckdb = duckdb_path.exists()
    has_sidecar = sidecar_path.exists()

    try:
        entry_mtime = _top_level_mtime(entry)
    except OSError as err:
        logger.warning("snapshots gc could not stat %s: %s", entry, err)
        return None

    partial = (
        (has_duckdb or has_sidecar)
        and not (has_duckdb and has_sidecar)
        and not (has_duckdb and _is_pre_sidecar_snapshot(entry_mtime))
    )
    return _SnapshotGcCandidate(
        paths=(entry,),
        result_paths=(str(entry),),
        mtime=entry_mtime,
        partial=partial,
    )


def _sibling_snapshot_candidate(
    group: dict[str, Path],
) -> _SnapshotGcCandidate | None:
    duckdb_path = group.get("duckdb")
    sidecar_path = group.get("sidecar")
    paths = tuple(path for path in (duckdb_path, sidecar_path) if path is not None)

    try:
        snapshot_mtime = max(_top_level_mtime(path) for path in paths)
    except OSError as err:
        logger.warning(
            "snapshots gc could not stat sibling snapshot %s: %s",
            ", ".join(str(path) for path in paths),
            err,
        )
        return None

    if duckdb_path is not None and sidecar_path is not None:
        return _SnapshotGcCandidate(
            paths=(duckdb_path, sidecar_path),
            result_paths=(str(duckdb_path), str(sidecar_path)),
            mtime=snapshot_mtime,
        )
    if duckdb_path is not None and _is_pre_sidecar_snapshot(snapshot_mtime):
        return _SnapshotGcCandidate(
            paths=(duckdb_path,),
            result_paths=(str(duckdb_path),),
            mtime=snapshot_mtime,
        )
    # A lone half of the pair: reported, never deleted.
    return _SnapshotGcCandidate(
        paths=paths,
        result_paths=tuple(str(path) for path in paths),
        mtime=snapshot_mtime,
        partial=True,
    )


def _is_pre_sidecar_snapshot(snapshot_mtime: float) -> bool:
    return snapshot_mtime < RECALL_SIDECAR_INTRODUCED_AT.timestamp()


def _entry_stays_inside(entry: Path, snapshots_root: Path) -> bool | None:
    try:
        resolved = entry.resolve(strict=False)
    except (RuntimeError, OSError) as err:
        logger.warning("snapshots gc could not resolve %s: %s", entry, err)
        return None
    return resolved == snapshots_root or snapshots_root in resolved.parents


def _top_level_mtime(entry: Path) -> float:
    if entry.is_symlink():
        return entry.lstat().st_mtime
    return entry.stat().st_mtime


def _entry_size(path: Path) -> int:
    if path.is_symlink():
        return _stat_size(path, follow_symlinks=False)
    if path.is_file():
        return _stat_size(path, follow_symlinks=True)
    total = 0
    for sub in path.rglob("*"):
        if sub.is_symlink():
            total += _stat_size(sub, follow_symlinks=False)
            continue
        if sub.is_file():
            total += _stat_size(sub, follow_symlinks=True)
    return total


def _stat_size(path: Path, *, follow_symlinks: bool) -> int:
    try:
        return path.stat(follow_symlinks=follow_symlinks).st_size
    except OSError:
        return 0


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)
