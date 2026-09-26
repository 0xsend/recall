from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROD_ROOTS = [
    REPO_ROOT / "packages" / "recall" / "src" / "recall" / "services",
    REPO_ROOT / "packages" / "recall" / "src" / "recall" / "cli",
    REPO_ROOT / "packages" / "recall" / "src" / "recall" / "parsers",
]


def _iter_py_files() -> list[Path]:
    files: list[Path] = []
    for root in PROD_ROOTS:
        assert root.exists(), f"expected production root to exist: {root}"
        files.extend(sorted(root.rglob("*.py")))
    return files


def _find_call_sites(tree: ast.AST, target_name: str) -> list[ast.Call]:
    sites: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == target_name) or (
            isinstance(func, ast.Attribute) and func.attr == target_name
        ):
            sites.append(node)
    return sites


def _enclosing_function(tree: ast.AST, call: ast.Call) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if child is call:
                return node.name
    return None


def test_live_session_set_has_single_production_construction_site() -> None:
    """REQ-DAEMON-063: LiveSessionSet has one production constructor call site.

    The v0.10.0 regression added a dead-code path with its own LiveSessionSet
    while the RPC daemon used a separate runtime. This static lint blocks that
    class of drift without importing production modules.
    """

    hits: list[tuple[Path, int, str | None]] = []
    for path in _iter_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as err:
            pytest.fail(f"failed to parse {path}: {err}")
        for call in _find_call_sites(tree, "LiveSessionSet"):
            hits.append((path, call.lineno, _enclosing_function(tree, call)))

    assert len(hits) == 1, (
        "expected exactly one production LiveSessionSet( construction site "
        f"(REQ-DAEMON-063), got {len(hits)}: {hits}"
    )
    path, lineno, enclosing = hits[0]
    assert path.name == "watcher.py", (
        f"LiveSessionSet must be constructed in services/watcher.py, got {path} line {lineno}"
    )
    assert enclosing == "build_live_watch_runtime", (
        "LiveSessionSet must be constructed inside build_live_watch_runtime, "
        f"got {enclosing!r} in {path} line {lineno}"
    )


def test_rpc_start_watch_mode_calls_build_live_watch_runtime() -> None:
    """REQ-DAEMON-057/063: the RPC watch entry point reaches the shared helper."""

    rpc_path = next(path for path in _iter_py_files() if path.name == "rpc_server.py")
    tree = ast.parse(rpc_path.read_text(encoding="utf-8"))

    start_watch_mode: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_start_watch_mode"
        ):
            start_watch_mode = node
            break

    assert start_watch_mode is not None, (
        "RpcServer._start_watch_mode not found in services/rpc_server.py"
    )

    called_names: set[str] = set()
    if _find_call_sites(start_watch_mode, "build_live_watch_runtime"):
        called_names.add("build_live_watch_runtime")

    if "build_live_watch_runtime" not in called_names:
        for call in ast.walk(start_watch_mode):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                called_names.add(call.func.id)
            elif isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                called_names.add(call.func.attr)

    assert "build_live_watch_runtime" in called_names, (
        "_start_watch_mode must call build_live_watch_runtime by name "
        f"(REQ-DAEMON-057/063). Current call names: {sorted(called_names)}"
    )
