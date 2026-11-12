from __future__ import annotations

import os
from pathlib import Path

import pytest
import recall.services.snapshots as snapshots_module
from recall.core.config import AppConfig
from recall.db.lease import StandaloneLifetime
from recall.services.snapshots import RECALL_SIDECAR_INTRODUCED_AT, gc_snapshots


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    data_dir = tmp_path / ".local" / "share" / "recall"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DATA_DIR", str(data_dir))
    monkeypatch.setenv("RECALL_CONFIG_PATH", str(tmp_path / ".config/recall/config.toml"))
    monkeypatch.delenv("RECALL_DB_PATH", raising=False)
    monkeypatch.delenv("RECALL_LOCK_PATH", raising=False)
    return AppConfig.load()


def _snapshots_dir(config: AppConfig) -> Path:
    path = config.data_dir / "snapshots"
    path.mkdir(parents=True)
    return path


def _set_mtime(path: Path, timestamp: float) -> None:
    os.utime(path, (timestamp, timestamp), follow_symlinks=False)


def _freeze_after_sidecar_cutoff(monkeypatch: pytest.MonkeyPatch) -> float:
    now = RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 10_000
    monkeypatch.setattr(snapshots_module.time, "time", lambda: now)
    return now


@pytest.mark.parametrize(
    "kind", ["index-migration", "storage-migration", "recreate", "keyword-rebuild"]
)
def test_gc_retains_aged_recovery_backups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import sqlite3

    from recall.db.connection import connect
    from recall.services.recreation import consistent_backup

    config = _config(tmp_path, monkeypatch)
    now = _freeze_after_sidecar_cutoff(monkeypatch)
    conn = connect(config)
    with sqlite3.connect(config.data_dir / "recall.fts.sqlite") as sidecar:
        sidecar.execute("CREATE TABLE owned_marker AS SELECT 19 AS value")
    backup = consistent_backup(conn, config, kind=kind, observer=StandaloneLifetime())
    conn.close()
    _set_mtime(backup, now - 8 * 86400)
    result = gc_snapshots(config, dry_run=True)
    assert str(backup) in result.kept_paths
    assert result.removed_paths == ()
    assert (backup / "recall.duckdb").is_file()
    assert (backup / "recall.fts.sqlite").is_file()


def test_gc_refuses_partial_snapshot_with_only_duckdb_after_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _freeze_after_sidecar_cutoff(monkeypatch)
    config = _config(tmp_path, monkeypatch)
    snap = _snapshots_dir(config) / "snap1.duckdb"
    snap.write_text("duckdb")
    _set_mtime(snap, now - 1)

    result = gc_snapshots(config, days=0)

    assert result.partial_paths == (str(snap),)
    assert str(snap) not in result.removed_paths
    assert snap.exists()


def test_gc_treats_pre_sidecar_duckdb_only_as_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_after_sidecar_cutoff(monkeypatch)
    config = _config(tmp_path, monkeypatch)
    legacy = _snapshots_dir(config) / "legacy.duckdb"
    legacy.write_text("legacy")
    _set_mtime(legacy, RECALL_SIDECAR_INTRODUCED_AT.timestamp() - 1)

    result = gc_snapshots(config, days=0)

    assert result.removed_paths == (str(legacy),)
    assert result.partial_paths == ()
    assert not legacy.exists()


def test_gc_removes_complete_paired_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _freeze_after_sidecar_cutoff(monkeypatch)
    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    duckdb = snapshots / "snap2.duckdb"
    sidecar = snapshots / "snap2.fts.sqlite"
    wal = snapshots / "snap2.duckdb.wal"
    duckdb.write_text("duck")
    sidecar.write_text("sidecar")
    wal.write_text("wal")
    _set_mtime(duckdb, now - 1)
    _set_mtime(sidecar, now - 1)
    _set_mtime(wal, now - 1)

    result = gc_snapshots(config, days=0)

    assert result.removed_paths == (str(duckdb), str(sidecar))
    assert result.total_bytes_freed == 11
    assert not duckdb.exists()
    assert not sidecar.exists()
    assert wal.exists()


def test_gc_refuses_partial_snapshot_with_only_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze_after_sidecar_cutoff(monkeypatch)
    config = _config(tmp_path, monkeypatch)
    sidecar = _snapshots_dir(config) / "snap3.fts.sqlite"
    sidecar.write_text("sidecar")
    _set_mtime(sidecar, RECALL_SIDECAR_INTRODUCED_AT.timestamp() - 1)

    result = gc_snapshots(config, days=0)

    assert result.partial_paths == (str(sidecar),)
    assert result.removed_paths == ()
    assert sidecar.exists()


def test_gc_handles_per_snapshot_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = _freeze_after_sidecar_cutoff(monkeypatch)
    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    complete = snapshots / "snap4"
    complete.mkdir()
    (complete / "recall.duckdb").write_text("duck")
    (complete / "recall.fts.sqlite").write_text("sidecar")
    (complete / "manifest.json").write_text("{}")
    partial = snapshots / "snap5"
    partial.mkdir()
    (partial / "recall.duckdb").write_text("duck")
    _set_mtime(complete, now - 1)
    _set_mtime(partial, now - 1)

    result = gc_snapshots(config, days=0)

    assert str(complete) in result.removed_paths
    assert str(partial) in result.partial_paths
    assert not complete.exists()
    assert partial.exists()


@pytest.mark.parametrize("operation", ["list", "gc"])
@pytest.mark.parametrize("kind", ["directory", "file", "dangling", "payload", "intermediate"])
def test_snapshot_foreign_links_never_inspect_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, kind: str
) -> None:
    from recall.services.snapshots import list_snapshots

    from tests.backup_access_guard import ForeignAccessGuard

    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "file").write_bytes(b"protected")
    (foreign / "directory").mkdir()
    entry = snapshots / "entry"
    aliases: tuple[Path, ...] = ()
    if kind in {"payload", "intermediate"}:
        entry.mkdir()
        (entry / "recall.fts.sqlite").write_bytes(b"local")
        payload = entry / "recall.duckdb"
        if kind == "payload":
            payload.symlink_to(foreign / "file")
            aliases = (payload,)
        else:
            alias = entry / "alias"
            alias.symlink_to(foreign, target_is_directory=True)
            payload.symlink_to("alias/file")
            aliases = (alias, payload)
    else:
        entry.symlink_to(foreign / kind, target_is_directory=kind == "directory")
    own_size = entry.lstat().st_size
    _set_mtime(entry, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    guard = ForeignAccessGuard(monkeypatch, foreign, following_aliases=aliases)

    if operation == "gc":
        result = gc_snapshots(config, days=0)
        assert result.kept_paths == (str(entry),)
        assert result.failed_paths == result.partial_paths == result.removed_paths == ()
    else:
        inventory = list_snapshots(config)
        assert inventory.unreadable_paths == ()
        assert inventory.entry_count == 1
        listed = inventory.entries[0]
        assert listed.paths == (str(entry),)
        assert listed.retained and not listed.partial
        if kind not in {"payload", "intermediate"}:
            assert listed.total_bytes == own_size
    assert guard.attempts == []


@pytest.mark.parametrize("kind", ["absent", "file", "linked", "dangling"])
def test_snapshot_root_status_does_not_follow_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from recall.services.snapshots import list_snapshots

    from tests.backup_access_guard import ForeignAccessGuard

    config = _config(tmp_path, monkeypatch)
    config.data_dir.mkdir(parents=True)
    snapshots = config.data_dir / "snapshots"
    foreign = tmp_path / "foreign"
    if kind == "file":
        snapshots.write_bytes(b"file")
    elif kind in {"linked", "dangling"}:
        if kind == "linked":
            foreign.mkdir()
        snapshots.symlink_to(foreign, target_is_directory=True)
    guard = ForeignAccessGuard(
        monkeypatch,
        foreign,
        following_aliases=(snapshots,) if kind in {"linked", "dangling"} else (),
    )
    gc = gc_snapshots(config)
    inventory = list_snapshots(config)
    assert gc.snapshots_dir_missing is (kind == "absent")
    assert inventory.snapshots_dir_missing is (kind == "absent")
    assert gc.failed_paths == gc.removed_paths == ()
    assert inventory.unreadable_paths == inventory.entries == ()
    assert guard.attempts == []


@pytest.mark.parametrize("target", ["missing/../realfile", "somefile/../realfile"])
def test_snapshot_payload_absence_is_latched_through_dotdot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    from recall.services.snapshots import list_snapshots

    config = _config(tmp_path, monkeypatch)
    snap = _snapshots_dir(config) / "partial"
    snap.mkdir()
    (snap / "somefile").write_bytes(b"not a directory")
    (snap / "realfile").write_bytes(b"real")
    (snap / "recall.fts.sqlite").write_bytes(b"sidecar")
    (snap / "recall.duckdb").symlink_to(target)
    _set_mtime(snap, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    assert list_snapshots(config).entries[0].partial
    gc = gc_snapshots(config, days=0)
    assert gc.partial_paths == (str(snap),)
    assert gc.removed_paths == gc.failed_paths == ()
    assert snap.is_dir()


@pytest.mark.parametrize("absolute", [True, False], ids=["absolute", "relative"])
@pytest.mark.parametrize("suffix", ["/", "/.", "/./"])
@pytest.mark.parametrize("payload", ["recall.duckdb", "recall.fts.sqlite"])
def test_snapshot_payload_requires_directory_before_trailing_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    absolute: bool,
    suffix: str,
    payload: str,
) -> None:
    from recall.services.snapshots import list_snapshots

    now = _freeze_after_sidecar_cutoff(monkeypatch)
    config = _config(tmp_path, monkeypatch)
    snap = _snapshots_dir(config) / "partial"
    snap.mkdir()
    plain = snap / "plain"
    plain.write_bytes(b"not a directory")
    local = snap / ("recall.fts.sqlite" if payload == "recall.duckdb" else "recall.duckdb")
    local.write_bytes(b"recoverable payload")
    link = snap / payload
    link.symlink_to((str(plain) if absolute else "plain") + suffix)
    _set_mtime(snap, now - 1)

    with pytest.raises(NotADirectoryError):
        link.stat()

    inventory = list_snapshots(config)
    assert inventory.unreadable_paths == ()
    assert len(inventory.entries) == 1
    entry = inventory.entries[0]
    results = [gc_snapshots(config, days=0, dry_run=dry_run) for dry_run in (True, False)]
    assert entry.paths == (str(snap),)
    assert entry.partial and not entry.retained, results
    for result in results:
        assert result.partial_paths == (str(snap),)
        assert result.removed_paths == result.failed_paths == ()
        assert result.total_bytes_freed == 0
        assert local.read_bytes() == b"recoverable payload"
        assert plain.read_bytes() == b"not a directory"
        assert link.is_symlink()


@pytest.mark.parametrize("kind", ["relative", "absolute", "chain", "dangling", "physical-parent"])
def test_snapshot_internal_links_unlink_only_the_local_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from recall.services.snapshots import list_snapshots

    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    target = snapshots / "target"
    target.write_bytes(b"preserved")
    link = snapshots / "stale-link"
    if kind == "relative":
        link.symlink_to("target")
    elif kind == "absolute":
        link.symlink_to(target)
    elif kind == "chain":
        (snapshots / "alias").symlink_to("target")
        link.symlink_to("alias")
    elif kind == "dangling":
        link.symlink_to("absent")
    else:
        (snapshots / "real" / "child").mkdir(parents=True)
        (snapshots / "real" / "target").write_bytes(b"physical parent")
        (snapshots / "alias").symlink_to("real/child")
        link.symlink_to("alias/../target")
    _set_mtime(link, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    listed = next(item for item in list_snapshots(config).entries if str(link) in item.paths)
    assert not listed.retained
    size = link.lstat().st_size
    preview = gc_snapshots(config, dry_run=True)
    assert preview.removed_paths == (str(link),)
    assert preview.total_bytes_freed == size
    assert link.is_symlink()
    gc = gc_snapshots(config)
    assert gc.removed_paths == (str(link),)
    assert gc.failed_paths == ()
    assert not link.is_symlink()
    assert target.read_bytes() == b"preserved"
    assert gc_snapshots(config).removed_paths == ()


@pytest.mark.parametrize(
    "target", ["../snapshots/real", "missing/../../foreign", "../snapshots-other/real"]
)
def test_snapshot_escape_is_refused_even_if_text_returns_inside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    from recall.services.snapshots import list_snapshots

    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    (snapshots / "real").write_bytes(b"local")
    link = snapshots / "stale"
    link.symlink_to(target)
    _set_mtime(link, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    gc = gc_snapshots(config)
    assert str(link) in gc.kept_paths
    assert gc.removed_paths == gc.failed_paths == ()
    entry = next(item for item in list_snapshots(config).entries if str(link) in item.paths)
    assert entry.retained


def test_snapshot_removes_stale_directory_with_foreign_nested_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.backup_access_guard import ForeignAccessGuard

    config = _config(tmp_path, monkeypatch)
    stale = _snapshots_dir(config) / "stale"
    stale.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "marker").write_bytes(b"protected")
    (stale / "file-link").symlink_to(foreign / "marker")
    (stale / "dir-link").symlink_to(foreign, target_is_directory=True)
    (stale / "local").write_bytes(b"abc")
    size = 3 + (stale / "file-link").lstat().st_size + (stale / "dir-link").lstat().st_size
    _set_mtime(stale, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    guard = ForeignAccessGuard(monkeypatch, foreign)
    gc = gc_snapshots(config)
    assert gc.removed_paths == (str(stale),)
    assert gc.failed_paths == ()
    assert gc.total_bytes_freed == size
    assert not stale.exists()
    assert guard.attempts == []


def test_snapshot_gc_absolute_root_never_looks_up_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    stale = snapshots / "stale"
    stale.write_bytes(b"stale")
    _set_mtime(stale, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)

    def refuse_getcwd() -> str:
        raise AssertionError("absolute snapshot root must not query the working directory")

    with monkeypatch.context() as guarded:
        guarded.setattr(os, "getcwd", refuse_getcwd)
        preview = gc_snapshots(config, dry_run=True)
        assert stale.read_bytes() == b"stale"
        applied = gc_snapshots(config)

    assert preview.removed_paths == applied.removed_paths == (str(stale),)
    assert preview.failed_paths == applied.failed_paths == ()
    assert preview.total_bytes_freed == applied.total_bytes_freed == 5
    assert not stale.exists()


@pytest.mark.parametrize("selection", ["relative", "alias", "link-dotdot"])
def test_snapshot_selected_root_accepts_only_its_two_bound_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    from recall.services.snapshots import list_snapshots

    config = _config(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    canonical = tmp_path / "real" / "data"
    snapshots = canonical / "snapshots"
    snapshots.mkdir(parents=True)
    (tmp_path / "real" / "child").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real" / "child")
    (tmp_path / "alias").symlink_to(canonical)
    (tmp_path / "unbound").symlink_to(canonical)
    selected = {
        "relative": Path("real/data"),
        "alias": tmp_path / "alias",
        "link-dotdot": tmp_path / "link/../data",
    }[selection]
    # The test harness canonicalizes constructor paths; exercise the caller
    # spelling after its session-root admission, as the migration tests do.
    object.__setattr__(config, "data_dir", selected)
    (snapshots / "target").write_bytes(b"target")
    (snapshots / "canonical").symlink_to(snapshots / "target")
    (snapshots / "configured").symlink_to(tmp_path / selected / "snapshots/target")
    (snapshots / "unbound").symlink_to(tmp_path / "unbound/snapshots/target")
    for name in ("canonical", "configured", "unbound"):
        _set_mtime(snapshots / name, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    inventory = list_snapshots(config)
    assert inventory.unreadable_paths == ()
    assert [Path(e.paths[0]).name for e in inventory.entries if e.retained] == ["unbound"]
    gc = gc_snapshots(config)
    assert gc.removed_paths == (
        str(selected / "snapshots/canonical"),
        str(selected / "snapshots/configured"),
    )
    assert str(selected / "snapshots/unbound") in gc.kept_paths
    assert (snapshots / "target").read_bytes() == b"target"


@pytest.mark.parametrize("errno_value", [13, 5, 24])
@pytest.mark.parametrize("which", ["selected", "snapshots"])
def test_snapshot_root_open_failure_is_nonfatal_and_truthful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, errno_value: int, which: str
) -> None:
    from recall.services.snapshots import list_snapshots

    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    original = os.open

    def fail_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == (config.data_dir if which == "selected" else "snapshots"):
            raise OSError(errno_value, "fixture root failure")
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", fail_open)
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, fail_open})
    gc = gc_snapshots(config)
    inventory = list_snapshots(config)
    assert gc.failed_paths == inventory.unreadable_paths == (str(snapshots),)
    assert gc.removed_paths == inventory.entries == ()
    assert not gc.snapshots_dir_missing and not inventory.snapshots_dir_missing


@pytest.mark.parametrize("levels", [128, 129])
def test_snapshot_directory_depth_limit_preserves_failed_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, levels: int
) -> None:
    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    current = snapshots
    for _ in range(levels):
        current = current / "d"
        current.mkdir()
    (current / "leaf").write_bytes(b"leaf")
    top = snapshots / "d"
    _set_mtime(top, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    gc = gc_snapshots(config)
    if levels == 128:
        assert gc.removed_paths == (str(top),)
        assert gc.failed_paths == ()
        assert gc.total_bytes_freed == 4
    else:
        assert gc.failed_paths == (str(top),)
        assert gc.removed_paths == ()
        assert (current / "leaf").read_bytes() == b"leaf"


def test_snapshot_visit_exhaustion_removes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    stale = snapshots / "a-stale"
    stale.write_bytes(b"preserved")
    wide = snapshots / "b-wide"
    wide.mkdir()
    for name in ("one", "two", "three"):
        (wide / name).write_bytes(b"x")
    _set_mtime(stale, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    monkeypatch.setattr(snapshots_module, "_VISITS_MAX", 4)
    gc = gc_snapshots(config)
    assert gc.removed_paths == ()
    assert gc.failed_paths == (str(wide),)
    assert stale.read_bytes() == b"preserved"


def test_snapshot_failed_removal_verification_cannot_hide_dangling_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    link = _snapshots_dir(config) / "link"
    link.symlink_to("missing")
    _set_mtime(link, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)

    def no_unlink(path, *, dir_fd=None):
        return None

    monkeypatch.setattr(os, "unlink", no_unlink)
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, no_unlink})
    gc = gc_snapshots(config)
    assert gc.failed_paths == (str(link),)
    assert gc.removed_paths == ()
    assert gc.total_bytes_freed == 0
    assert link.is_symlink()


@pytest.mark.parametrize("count", [4, 5])
def test_snapshot_visit_limit_at_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    for index in range(count):
        path = snapshots / f"item-{index}"
        path.write_bytes(b"x")
        _set_mtime(path, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    monkeypatch.setattr(snapshots_module, "_VISITS_MAX", 4)
    gc = gc_snapshots(config)
    if count == 4:
        assert len(gc.removed_paths) == 4
        assert gc.failed_paths == ()
    else:
        assert gc.removed_paths == ()
        assert str(snapshots) in gc.failed_paths
        assert len(list(snapshots.iterdir())) == 5


@pytest.mark.parametrize("hops", [40, 41])
def test_snapshot_link_expansion_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hops: int
) -> None:
    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    target = snapshots / "target"
    target.mkdir()
    for index in range(hops):
        (target / f"link-{index}").symlink_to(f"link-{index + 1}" if index + 1 < hops else "file")
    (target / "file").write_bytes(b"local")
    link = snapshots / "stale"
    # There are hops expansions including this top-level entry.
    link.symlink_to("target/link-1" if hops > 1 else "target/file")
    _set_mtime(link, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    gc = gc_snapshots(config)
    if hops == 40:
        assert gc.removed_paths == (str(link),)
        assert gc.failed_paths == ()
    else:
        assert gc.removed_paths == ()
        assert gc.failed_paths == (str(link),)


@pytest.mark.parametrize("budget", [2, 3])
def test_snapshot_descriptor_exhaustion_is_nonfatal_without_leaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    from recall.services.snapshots import list_snapshots

    config = _config(tmp_path, monkeypatch)
    top = _snapshots_dir(config) / "top"
    (top / "nested").mkdir(parents=True)
    (top / "nested" / "file").write_bytes(b"preserved")
    _set_mtime(top, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    monkeypatch.setattr(snapshots_module, "_DESCRIPTORS_MAX", budget)
    before = len(os.listdir("/dev/fd"))
    for _ in range(3):
        gc = gc_snapshots(config)
        inventory = list_snapshots(config)
        assert gc.failed_paths and inventory.unreadable_paths
        assert gc.removed_paths == ()
    assert len(os.listdir("/dev/fd")) == before
    assert (top / "nested" / "file").read_bytes() == b"preserved"


def test_snapshot_new_unplanned_entry_prevents_successful_directory_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    top = _snapshots_dir(config) / "top"
    top.mkdir()
    (top / "planned").write_bytes(b"planned")
    _set_mtime(top, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    original = os.rmdir

    def new_entry(path, *, dir_fd=None):
        if path == "top":
            (top / "unplanned").write_bytes(b"preserved")
        return original(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "rmdir", new_entry)
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, new_entry})
    gc = gc_snapshots(config)
    assert gc.failed_paths == (str(top),)
    assert gc.removed_paths == ()
    assert gc.total_bytes_freed == 0
    assert (top / "unplanned").read_bytes() == b"preserved"


def test_snapshot_directory_payload_uses_physical_parent_after_internal_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recall.services.snapshots import list_snapshots

    config = _config(tmp_path, monkeypatch)
    top = _snapshots_dir(config) / "top"
    (top / "real" / "child").mkdir(parents=True)
    (top / "real" / "file").write_bytes(b"db")
    (top / "alias").symlink_to("real/child")
    (top / "recall.duckdb").symlink_to("alias/../file")
    (top / "recall.fts.sqlite").write_bytes(b"sidecar")
    _set_mtime(top, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)
    assert not list_snapshots(config).entries[0].partial
    gc = gc_snapshots(config)
    assert gc.removed_paths == (str(top),)
    assert gc.partial_paths == gc.failed_paths == ()


def test_snapshot_deep_absolute_expansion_replaces_the_physical_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    snapshots = _snapshots_dir(config)
    source = snapshots / "victim"
    target = snapshots / "target"
    source.mkdir()
    target.mkdir()
    physical = source
    landing = target
    for _ in range(125):
        physical = physical / "d"
        physical.mkdir()
        landing = landing / "t"
        landing.mkdir()
    (landing / "file").write_bytes(b"preserved target")
    (physical / "jump").symlink_to(landing)
    (source / "recall.duckdb").symlink_to("/".join(["d"] * 125 + ["jump", "file"]))
    (source / "recall.fts.sqlite").write_bytes(b"sidecar")
    _set_mtime(source, RECALL_SIDECAR_INTRODUCED_AT.timestamp() + 1)

    result = gc_snapshots(config)

    assert result.removed_paths == (str(source),)
    assert result.failed_paths == ()
    assert (landing / "file").read_bytes() == b"preserved target"
