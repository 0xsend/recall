"""Process defaults shared by daemon and prerequisite model imports."""

from __future__ import annotations

import os


def prepare_model_environment() -> None:
    """Set SDK progress defaults before imports, preserving explicit choices."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
