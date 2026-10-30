"""`prepare`: clone a rollback backup into a template with a verified manifest."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from .evidence import Result, tool_identity
from .isolation import (
    MANIFEST_NAME,
    PAYLOAD_DIR,
    POOL_DIR,
    RESERVED_PAYLOAD_NAMES,
    SOURCES,
    USAGE_LOG_NAME,
    Entry,
    FileRecord,
    Manifest,
    Policy,
    Refused,
    Template,
    clone_tree,
    regular_files,
    sha256,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backup", required=True, help="rollback backup directory (read-only)")
    parser.add_argument(
        "--template",
        required=True,
        help="template directory to create; left without a manifest if prepare fails",
    )
    parser.add_argument(
        "--source-pool",
        help="directory of per-source transcript trees (claude_code, codex, ...) to copy",
    )
    parser.add_argument("--usage-log", help="Grok usage log to copy for injection")


def _payload(backup: Path) -> list[Entry]:
    entries = regular_files("--backup", backup)
    if not entries:
        raise Refused("--backup", backup, "holds no files")
    for entry in entries:
        if len(entry.relative.parts) != 1:
            raise Refused("--backup", entry.path.parent, "backup payload must be flat files")
        if entry.relative.name in RESERVED_PAYLOAD_NAMES:
            raise Refused("--backup", entry.path, "name is reserved in a run directory")
    return entries


def _pool(pool: Path) -> list[Entry]:
    entries = regular_files("--source-pool", pool)
    for entry in entries:
        if len(entry.relative.parts) < 2 or entry.relative.parts[0] not in SOURCES:
            raise Refused(
                "--source-pool", entry.path, f"files must be under one of {', '.join(SOURCES)}"
            )
    return entries


def _signature(path: Path) -> tuple[int, int, int, int]:
    status = path.stat(follow_symlinks=False)
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def run(args: argparse.Namespace, result: Result) -> None:
    policy = Policy.for_host()
    backup = policy.read_tree("--backup", args.backup)
    pool = policy.read_tree("--source-pool", args.source_pool) if args.source_pool else None
    usage = policy.read_file("--usage-log", args.usage_log) if args.usage_log else None
    target = policy.new_directory("--template", args.template)

    sources: dict[str, Path] = {
        str(PurePosixPath(PAYLOAD_DIR) / entry.relative): entry.path
        for entry in _payload(backup.path)
    }
    if pool is not None:
        sources.update(
            {
                str(PurePosixPath(POOL_DIR) / entry.relative): entry.path
                for entry in _pool(pool.path)
            }
        )
    if usage is not None:
        sources[USAGE_LOG_NAME] = usage.path
    result.step("resolve", files=len(sources))

    before = {name: _signature(path) for name, path in sources.items()}
    with target.create().held() as template:
        # The template is left in place on failure; without a manifest it never opens.
        clone_tree(sources, template)
        result.exit_codes["cp"] = 0
        changed = sorted(name for name, path in sources.items() if _signature(path) != before[name])
        if changed:
            raise Refused("--backup", backup.path, f"source files changed while copying: {changed}")
        records = tuple(
            FileRecord(name, (target.path / name).stat().st_size, sha256(target.path / name))
            for name in sorted(sources)
        )
        manifest = Manifest(
            records,
            provenance={
                "backup": str(backup.path),
                "source_pool": str(pool.path) if pool else None,
                "usage_log": str(usage.path) if usage else None,
                "prepared_at": datetime.now(UTC).isoformat(),
                "tool": tool_identity(),
            },
        )
        template.publish(MANIFEST_NAME, manifest.to_json())
    result.step("clone", files=len(records), bytes=sum(record.bytes for record in records))

    verified = Template.open(policy, "--template", target.path)
    result.step("verify", template=str(verified.path), files=len(verified.manifest.files))
    result.scope = {"measurements": [], "template": str(verified.path)}
