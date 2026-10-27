"""Index argument resolution shared by CLI previews and RPC execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from recall.core.config import VALID_CONTEXT_MODES, ContextConfig
from recall.core.rpc_params import IndexParams
from recall.core.rpc_types import APP_CONFIRMATION_REQUIRED, RpcError
from recall.core.time import parse_since
from recall.core.types import Source, parse_source


@dataclass(frozen=True)
class ResolvedIndexRequest:
    source: Source | None
    since: datetime | None
    home_root: Path | None
    host: str | None
    context: str | None


def resolve_index_request(params: IndexParams) -> ResolvedIndexRequest:
    source = parse_source(params.source) if params.source else None
    since = parse_since(params.since) if params.since else None
    home_root = Path(params.root).expanduser() if params.root else None
    host = params.host
    if home_root is not None and host is None:
        host = home_root.resolve().name or "remote"

    # Source, time and root errors precede confirmation in the RPC contract;
    # worker, context and combination errors follow it.
    if params.recreate and not params.confirmed:
        raise RpcError(
            code=APP_CONFIRMATION_REQUIRED,
            message="recreate requires confirmed: true",
        )

    try:
        workers: int | str = int(params.workers)
    except ValueError:
        workers = params.workers
    resolve_worker_count(workers, 1)
    context = (
        ContextConfig.from_values({"mode": params.context}).mode
        if params.context is not None
        else None
    )
    if params.recreate and (params.recompute_context or params.project):
        raise ValueError("recreate cannot be combined with recompute_context or project")
    if (
        params.recompute_context
        and params.only_mode is not None
        and params.only_mode not in VALID_CONTEXT_MODES
    ):
        raise ValueError(f"unsupported context mode: {params.only_mode}")
    return ResolvedIndexRequest(source, since, home_root, host, context)


def resolve_worker_count(workers: int | str, changed_count: int) -> int:
    if isinstance(workers, int):
        if workers <= 0:
            raise ValueError("workers must be positive")
        return workers

    normalized = workers.strip().lower()
    if normalized != "auto":
        raise ValueError("workers must be a positive integer or 'auto'")
    if changed_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8, changed_count))
