"""Verify CLI modules do not import service/DB layers directly.

Per REQ-RPC-004, all CLI commands must go through the RPC client.
Exceptions:
- daemon.py manages the daemon process directly (install/uninstall/status/start/stop).
- compact.py owns daemon lifecycle coordination before invoking the local compaction service.
- snapshots.py owns local artifact cleanup directly.
- db.py owns local index maintenance that must run while the daemon is stopped.
"""

from __future__ import annotations

import ast
from pathlib import Path

CLI_DIR = Path(__file__).resolve().parents[2] / "packages" / "recall" / "src" / "recall" / "cli"
# Only lifecycle-owning / non-RPC orchestration commands may import services/db
# directly. fleet.py SSHs to remote recall and never opens the local DuckDB.
ALLOWED_FILES = {"compact.py", "daemon.py", "db.py", "snapshots.py", "fleet.py", "__init__.py"}
FORBIDDEN_PREFIXES = ("recall.services.", "recall.db.")


def _collect_imports(filepath: Path) -> list[str]:
    """Parse a Python file and return all imported module paths."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_cli_modules_do_not_import_services_or_db() -> None:
    violations: list[str] = []

    for py_file in sorted(CLI_DIR.glob("*.py")):
        if py_file.name in ALLOWED_FILES:
            continue
        imports = _collect_imports(py_file)
        for imp in imports:
            if any(imp.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
                violations.append(f"{py_file.name}: imports {imp}")

    assert not violations, (
        "CLI modules must not import recall.services.* or recall.db.* directly.\n"
        "Use recall.cli.rpc.rpc_call_or_error() instead.\n"
        "Violations:\n" + "\n".join(f"  - {v}" for v in violations)
    )
