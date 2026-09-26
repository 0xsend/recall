from __future__ import annotations

import tomllib
from pathlib import Path

PACKAGE_PYPROJECT = Path(__file__).resolve().parents[1] / "packages" / "recall" / "pyproject.toml"


def test_default_installation_excludes_pyarrow_25() -> None:
    """Keep every installation off the PyArrow 25 DuckDB ingestion crash."""
    package_data = tomllib.loads(PACKAGE_PYPROJECT.read_text(encoding="utf-8"))
    dependencies = package_data["project"]["dependencies"]
    pyarrow_requirement = next(
        requirement for requirement in dependencies if requirement.startswith("pyarrow")
    )

    assert pyarrow_requirement == "pyarrow>=14.0,<25"
