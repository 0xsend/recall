"""`prepare`: clone a rollback backup into a template with a verified manifest."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath

from .evidence import Result, tool_identity
from .fs import Cancellation, Directory, File, NewTree
from .isolation import (
    MANIFEST_NAME,
    PAYLOAD_DIR,
    POOL_DIR,
    RESERVED_PAYLOAD_NAMES,
    SOURCES,
    USAGE_LOG_NAME,
    FileRecord,
    Inventory,
    Manifest,
    NewDirectory,
    Policy,
    Refused,
    Signature,
    Template,
    open_entry,
    open_parent,
    scan,
)

Progress = Callable[[dict[str, object]], None]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backup", required=True, help="rollback backup directory (read-only)")
    parser.add_argument(
        "--template",
        required=True,
        help="template directory to create inside a write root; removed again if prepare fails",
    )
    parser.add_argument(
        "--source-pool",
        help="directory of per-source transcript trees (claude_code, codex, ...) to copy",
    )
    parser.add_argument("--usage-log", help="Grok usage log to copy for injection")


@dataclass(frozen=True)
class Source:
    """One input tree and where its files go in the template."""

    argument: str
    root: Directory
    prefix: PurePosixPath
    inventory: Inventory

    def destinations(self) -> Iterator[tuple[PurePosixPath, PurePosixPath]]:
        for relative in sorted(self.inventory.files):
            yield relative, self.prefix / relative

    def rescan(self) -> None:
        """Refuse if anything in the tree was added, removed, replaced or written."""
        now = scan(self.argument, self.root)
        changed = sorted(
            str(path)
            for path in self.inventory.files.keys() | now.files.keys()
            if self.inventory.files.get(path) != now.files.get(path)
        )
        changed += sorted(
            f"{path}/"
            for path in self.inventory.directories.keys() | now.directories.keys()
            if self.inventory.directories.get(path) != now.directories.get(path)
        )
        if changed:
            raise Refused(self.argument, self.root.path, f"source changed while copying: {changed}")


def _backup(root: Directory) -> Source:
    inventory = scan("--backup", root)
    if inventory.directories:
        nested = min(inventory.directories)
        raise Refused("--backup", root.path / nested, "backup payload must be flat files")
    if not inventory.files:
        raise Refused("--backup", root.path, "holds no files")
    for relative in inventory.files:
        if relative.name in RESERVED_PAYLOAD_NAMES:
            raise Refused("--backup", root.path / relative, "name is reserved in a run directory")
    return Source("--backup", root, PurePosixPath(PAYLOAD_DIR), inventory)


def _pool(root: Directory) -> Source:
    inventory = scan("--source-pool", root)
    reason = f"files must be under one of {', '.join(SOURCES)}"
    for relative in (*inventory.files, *inventory.directories):
        top_level_file = relative in inventory.files and len(relative.parts) < 2
        if relative.parts[0] not in SOURCES or top_level_file:
            raise Refused("--source-pool", root.path / relative, reason)
    return Source("--source-pool", root, PurePosixPath(POOL_DIR), inventory)


def _copy(
    tree: NewTree,
    destination: PurePosixPath,
    source: File,
    expected: Signature,
    argument: str,
    cancellation: Cancellation,
    progress: Progress,
) -> FileRecord:
    """Clone one file and prove the copy equals the source as it is afterwards."""
    if Signature.of(source.status) != expected:
        raise Refused(argument, source.path, "source changed while copying")
    progress({"event": "clone", "file": str(destination)})
    with open_parent(tree.root, destination, "--template") as parent:
        copy = tree.clone(parent, destination, source, cancellation)
    with copy:
        progress({"event": "hash", "file": str(destination)})
        digest = copy.sha256(cancellation)
        size = copy.status.st_size
    if source.sha256(cancellation) != digest:
        raise Refused(argument, source.path, "source changed while copying")
    return FileRecord(str(destination), size, digest)


def build_template(
    target: NewDirectory,
    backup: Directory,
    pool: Directory | None,
    usage: File | None,
    cancellation: Cancellation,
    progress: Progress,
    provenance: Mapping[str, object],
) -> Manifest:
    """Create the template, or leave nothing behind."""
    sources = [_backup(backup)] + ([_pool(pool)] if pool is not None else [])
    directories = [PAYLOAD_DIR]
    if pool is not None:
        pool_directories = sources[1].inventory.directories
        directories += [
            POOL_DIR,
            *sorted(str(PurePosixPath(POOL_DIR) / path) for path in pool_directories),
        ]
    usage_signature = Signature.of(usage.status) if usage is not None else None
    with target.create() as tree:
        for name in sorted(directories, key=lambda name: PurePosixPath(name).parts):
            relative = PurePosixPath(name)
            with open_parent(tree.root, relative, "--template") as parent:
                tree.make_directory(parent, relative).close()
        records: list[FileRecord] = []
        for source in sources:
            for relative, destination in source.destinations():
                with open_entry(source.root, relative, source.argument) as file:
                    records.append(
                        _copy(
                            tree,
                            destination,
                            file,
                            source.inventory.files[relative],
                            source.argument,
                            cancellation,
                            progress,
                        )
                    )
        if usage is not None and usage_signature is not None:
            records.append(
                _copy(
                    tree,
                    PurePosixPath(USAGE_LOG_NAME),
                    usage,
                    usage_signature,
                    "--usage-log",
                    cancellation,
                    progress,
                )
            )
        for source in sources:
            source.rescan()
        if usage is not None and Signature.of(usage.refresh()) != usage_signature:
            raise Refused("--usage-log", usage.path, "source changed while copying")
        manifest = Manifest(
            tuple(directories), tuple(sorted(records, key=lambda record: record.path)), provenance
        )
        cancellation.check()
        tree.publish(PurePosixPath(MANIFEST_NAME), manifest.to_json())
        progress({"event": "verify"})
        Template.from_root("--template", tree.root, cancellation)
        tree.keep()
    return manifest


def run(
    args: argparse.Namespace, result: Result, cancellation: Cancellation, progress: Progress
) -> None:
    policy = Policy.for_host()
    with ExitStack() as held:
        roots = held.enter_context(policy.open_write_roots())
        backup = held.enter_context(policy.open_tree("--backup", args.backup))
        pool = (
            held.enter_context(policy.open_tree("--source-pool", args.source_pool))
            if args.source_pool
            else None
        )
        usage = (
            held.enter_context(policy.open_file("--usage-log", args.usage_log))
            if args.usage_log
            else None
        )
        target = held.enter_context(roots.new_directory("--template", args.template))
        result.step("resolve")
        manifest = build_template(
            target,
            backup,
            pool,
            usage,
            cancellation,
            progress,
            provenance={
                "backup": str(backup.path),
                "source_pool": str(pool.path) if pool else None,
                "usage_log": str(usage.path) if usage else None,
                "prepared_at": datetime.now(UTC).isoformat(),
                "tool": tool_identity(),
            },
        )
        files = len(manifest.files)
        size = sum(record.bytes for record in manifest.files)
        result.step("clone", files=files, bytes=size)
        result.step("verify", template=str(target.path), files=files)
        result.scope = {"measurements": [], "template": str(target.path)}
