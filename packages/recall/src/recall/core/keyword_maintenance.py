"""Durable exclusion of ordinary writers during explicit keyword reconstruction."""

from pathlib import Path

MARKER_NAME = "keyword-maintenance.json"


def require_keyword_writes_enabled(data_dir: Path) -> None:
    marker = data_dir / MARKER_NAME
    if marker.exists():
        raise RuntimeError(
            f"Keyword maintenance is held by {marker}; resume, finish, abort or replace "
            "that operation. See docs/keyword-initialization.md."
        )
