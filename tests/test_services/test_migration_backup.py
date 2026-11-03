"""Native descriptor and lexical policy for selected-root backup access."""

from __future__ import annotations

import errno
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from lane_harness import lane_config
from recall.services.migration_backup import backup_presence, original_backup
from recall.services.recreation import restore_consistent_backup

from tests.backup_access_guard import ForeignAccessGuard


def _backup(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "index-migration",
                "complete": True,
                "files": ["recall.duckdb", "recall.fts.sqlite"],
            }
        )
    )
    (path / "recall.duckdb").write_bytes(b"original database")
    (path / "recall.fts.sqlite").write_bytes(b"original sidecar")


@pytest.mark.parametrize("selected", ["relative", "dotted", "alias", "link-parent"])
def test_selected_root_binding_is_independent_of_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected: str,
) -> None:
    cfg = lane_config(tmp_path)
    root = tmp_path / "actual" / "data"
    _backup(root / "snapshots/b")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    link = tmp_path / "parent-link"
    link.symlink_to(root / "snapshots", target_is_directory=True)
    paths = {
        "relative": Path("actual/data"),
        "dotted": Path("actual/../actual/data"),
        "alias": alias,
        "link-parent": link / "..",
    }
    cfg = replace(cfg, data_dir=root)
    # The test isolation guard canonicalizes AppConfig inputs. Restore the
    # unresolved spelling here to exercise the product boundary itself.
    object.__setattr__(cfg, "data_dir", paths[selected])
    monkeypatch.chdir(tmp_path)
    assert backup_presence(cfg, "snapshots/b") is True
    assert backup_presence(cfg, str(root / "snapshots/b")) is True
    assert backup_presence(cfg, "snapshots/missing") is False
    assert backup_presence(cfg, str(alias / "snapshots/b")) is None
    if selected == "link-parent":
        assert backup_presence(cfg, str(link) + "/../snapshots/b") is None


@pytest.mark.parametrize("root_kind", ["missing", "file", "denied"])
def test_invalid_selected_root_is_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_kind: str,
) -> None:
    cfg = lane_config(tmp_path)
    if root_kind == "file":
        cfg.data_dir.parent.mkdir(parents=True)
        cfg.data_dir.write_text("file")
    elif root_kind == "denied":
        cfg.data_dir.mkdir(parents=True)
        original = os.open

        def denied(path, *args, **kwargs):
            if path == cfg.data_dir:
                raise PermissionError(errno.EACCES, "controlled root denial")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", denied)
        monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, os.open})
    with pytest.raises(OSError):
        backup_presence(cfg, None)


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.parametrize("present", [True, False])
def test_ancestor_substitution_never_follows_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    when: str,
    present: bool,
) -> None:
    cfg = lane_config(tmp_path)
    ancestor = cfg.data_dir / "snapshots"
    ancestor.mkdir(parents=True)
    if present:
        (ancestor / "b").mkdir()
    source = tmp_path / "source"
    (source / "b").mkdir(parents=True)
    original = os.open
    replaced = False

    def substitute(path, *args, **kwargs):
        nonlocal replaced
        if path == "snapshots" and not replaced:
            replaced = True
            if when == "after":
                descriptor = original(path, *args, **kwargs)
            ancestor.rename(cfg.data_dir / "pinned")
            ancestor.symlink_to(source, target_is_directory=True)
            if when == "after":
                return descriptor
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", substitute)
    guard = ForeignAccessGuard(monkeypatch, source)
    assert backup_presence(cfg, "snapshots/b") is (None if when == "before" else present)
    assert replaced
    assert guard.attempts == []


@pytest.mark.parametrize("entry", ["manifest.json", "recall.duckdb", "recall.fts.sqlite"])
@pytest.mark.parametrize("kind", ["link", "missing", "directory"])
def test_invalid_input_refuses_before_destination_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    kind: str,
) -> None:
    cfg = lane_config(tmp_path)
    backup = cfg.data_dir / "snapshots/b"
    _backup(backup)
    source = tmp_path / "source"
    source.mkdir()
    (source / entry).write_bytes(b"foreign")
    (backup / entry).unlink()
    if kind == "link":
        (backup / entry).symlink_to(source / entry)
    elif kind == "directory":
        (backup / entry).mkdir()
    cfg.db_path.write_bytes(b"destination")
    wal = Path(str(cfg.db_path) + ".wal")
    wal.write_bytes(b"wal")
    sidecar = cfg.data_dir / "recall.fts.sqlite"
    sidecar.write_bytes(b"destination sidecar")
    guard = ForeignAccessGuard(monkeypatch, source)
    assert backup_presence(cfg, "snapshots/b") is True
    with (
        pytest.raises(RuntimeError, match="incomplete or missing"),
        original_backup(cfg, "snapshots/b") as inputs,
    ):
        restore_consistent_backup(cfg, inputs)
    assert (cfg.db_path.read_bytes(), wal.read_bytes(), sidecar.read_bytes()) == (
        b"destination",
        b"wal",
        b"destination sidecar",
    )
    assert guard.attempts == []


def test_restore_keeps_open_inputs_when_backup_directory_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = lane_config(tmp_path)
    backup = cfg.data_dir / "snapshots/b"
    _backup(backup)
    source = tmp_path / "source"
    _backup(source)
    (source / "recall.duckdb").write_bytes(b"foreign database")
    guard = ForeignAccessGuard(monkeypatch, source)
    with original_backup(cfg, "snapshots/b") as inputs:
        backup.rename(cfg.data_dir / "pinned-backup")
        backup.symlink_to(source, target_is_directory=True)
        restore_consistent_backup(cfg, inputs)
    assert cfg.db_path.read_bytes() == b"original database"
    assert (cfg.data_dir / "recall.fts.sqlite").read_bytes() == b"original sidecar"
    assert guard.attempts == []


@pytest.mark.parametrize(
    "manifest",
    [
        b"[]",
        b"{}",
        b"{bad",
        b"x" * (1024 * 1024 + 1),
        b'{"complete":true,"kind":"index-migration","files":["../source/recall.duckdb"]}',
        b'{"complete":true,"kind":"index-migration","files":["recall.duckdb", "recall.duckdb"]}',
    ],
)
def test_manifest_validation_is_bounded(tmp_path: Path, manifest: bytes) -> None:
    cfg = lane_config(tmp_path)
    backup = cfg.data_dir / "snapshots/b"
    _backup(backup)
    (backup / "manifest.json").write_bytes(manifest)
    with (
        pytest.raises(RuntimeError, match="incomplete or missing"),
        original_backup(cfg, "snapshots/b"),
    ):
        pytest.fail("invalid backup admitted")


@pytest.mark.parametrize("replacement", ["link", "absent", "file", "directory"])
def test_final_entry_substitution_has_literal_presence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    cfg = lane_config(tmp_path)
    backup = cfg.data_dir / "snapshots/b"
    backup.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    original = os.open
    replaced = False

    def substitute(path, *args, **kwargs):
        nonlocal replaced
        if path == "b" and not replaced:
            replaced = True
            backup.rename(cfg.data_dir / "pinned-b")
            if replacement == "link":
                backup.symlink_to(source, target_is_directory=True)
            elif replacement == "file":
                backup.write_bytes(b"ordinary final entry")
            elif replacement == "directory":
                backup.mkdir()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", substitute)
    guard = ForeignAccessGuard(monkeypatch, source)
    expected = {"link": None, "absent": False, "file": True, "directory": True}
    assert backup_presence(cfg, "snapshots/b") is expected[replacement]
    assert replaced
    assert guard.attempts == []


@pytest.mark.parametrize("error", [errno.EACCES, errno.EIO])
def test_local_backup_io_errors_are_not_reported_as_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: int,
) -> None:
    cfg = lane_config(tmp_path)
    (cfg.data_dir / "snapshots/b").mkdir(parents=True)
    original = os.open

    def fail(path, *args, **kwargs):
        if path == "snapshots":
            raise OSError(error, "controlled local I/O failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", fail)
    ForeignAccessGuard(monkeypatch, tmp_path / "source")
    with pytest.raises(OSError) as raised:
        backup_presence(cfg, "snapshots/b")
    assert raised.value.errno == error


def test_relative_root_is_captured_before_context_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recall.services.migration_backup import LocalBackup, access_backup

    cfg = lane_config(tmp_path)
    (tmp_path / "actual/snapshots/b").mkdir(parents=True)
    other = tmp_path / "elsewhere"
    other.mkdir()
    object.__setattr__(cfg, "data_dir", Path("actual"))
    monkeypatch.chdir(tmp_path)
    operation = access_backup(cfg, "snapshots/b")
    monkeypatch.chdir(other)
    with operation as access:
        assert isinstance(access, LocalBackup)
        assert access.present is True


@pytest.mark.parametrize("overlap", ["root", "cross-file"])
def test_restore_refuses_overlapping_inputs_before_destination_effects(
    tmp_path: Path,
    overlap: str,
) -> None:
    import shutil

    cfg = lane_config(tmp_path)
    backup = cfg.data_dir if overlap == "root" else cfg.data_dir / "snapshots/b"
    _backup(backup)
    if overlap == "cross-file":
        os.link(backup / "recall.fts.sqlite", cfg.db_path)
        (cfg.data_dir / "recall.fts.sqlite").write_bytes(b"destination sidecar")
    wal = Path(str(cfg.db_path) + ".wal")
    wal.write_bytes(b"owned WAL")
    paths = (
        cfg.db_path,
        wal,
        cfg.data_dir / "recall.fts.sqlite",
        backup / "recall.duckdb",
        backup / "recall.fts.sqlite",
    )
    before = [path.read_bytes() for path in paths]
    with (
        original_backup(cfg, "." if overlap == "root" else "snapshots/b") as inputs,
        pytest.raises(shutil.SameFileError),
    ):
        restore_consistent_backup(cfg, inputs)
    assert [path.read_bytes() for path in paths] == before
