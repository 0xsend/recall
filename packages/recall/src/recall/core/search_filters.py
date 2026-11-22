"""Shared validation and normalization for filtered search parameters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from recall.core.time import parse_since, to_naive_utc
from recall.core.types import Role

VALID_SEARCH_ROLES: tuple[str, ...] = tuple(role.value for role in Role)
ROLE_HELP = f"Filter by message role ({', '.join(VALID_SEARCH_ROLES)}); repeat for multiple roles"


@dataclass(frozen=True)
class NormalizedSearchFilters:
    roles: tuple[Role, ...] | None = None
    project: str | None = None
    since: datetime | None = None
    host: str | None = None
    per_session: int | None = None


def parse_role(value: str) -> Role:
    normalized = value.strip().lower()
    try:
        return Role(normalized)
    except ValueError as err:
        expected = ", ".join(VALID_SEARCH_ROLES)
        raise ValueError(f"unsupported role: {value} (expected one of: {expected})") from err


def normalize_search_filters(
    *,
    tool: str | None = None,
    role: Sequence[str] | None = None,
    project: str | None = None,
    since: str | None = None,
    host: str | None = None,
    per_session: int | None = None,
) -> NormalizedSearchFilters:
    """Validate and normalize search filters identically across CLI, RPC, and poll snapshots."""
    roles_tuple: tuple[Role, ...] | None = None
    if role:
        parsed_roles: list[Role] = []
        for raw in role:
            if not isinstance(raw, str):
                raise ValueError("role entries must be strings")
            parsed_roles.append(parse_role(raw))
        roles_tuple = tuple(dict.fromkeys(parsed_roles))

    if tool is not None and roles_tuple:
        raise ValueError("--role cannot be combined with --tool")

    clean_project: str | None = None
    if project is not None:
        if not isinstance(project, str) or not project.strip():
            raise ValueError("project must not be empty")
        clean_project = project.strip()

    clean_host: str | None = None
    if host is not None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must not be empty")
        clean_host = host.strip()

    parsed_since: datetime | None = None
    if since is not None:
        if not isinstance(since, str) or not since.strip():
            raise ValueError("since must not be empty")
        parsed_since = to_naive_utc(parse_since(since.strip()))

    if per_session is not None and (
        isinstance(per_session, bool) or not isinstance(per_session, int) or per_session <= 0
    ):
        raise ValueError("per_session must be a positive integer")

    return NormalizedSearchFilters(
        roles=roles_tuple,
        project=clean_project,
        since=parsed_since,
        host=clean_host,
        per_session=per_session,
    )


def has_filtered_search_params(
    *,
    roles: tuple[Role, ...] | None = None,
    project: str | None = None,
    since: datetime | str | None = None,
    host: str | None = None,
    per_session: int | None = None,
    fleet: bool = False,
) -> bool:
    """Whether a search uses PR 3 filters requiring fail-closed `recall.search_filtered`."""
    has_since = bool(since.strip()) if isinstance(since, str) else since is not None
    return bool(
        roles
        or project is not None
        or has_since
        or per_session is not None
        or (host is not None and not fleet)
    )
