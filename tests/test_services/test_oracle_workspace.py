"""A dead owner's working containers are reclaimed whatever the sorter left in them."""

import os
from pathlib import Path
from uuid import UUID

import duckdb
import pytest
from recall.db.lease import StandaloneLifetime
from recall.db.oracle_sort import SORT_FILES_MAX
from recall.db.repair import load_repair_row
from recall.db.repair_oracle import OracleUnavailable, capture_oracle
from recall.db.schema import ensure_schema
from recall.services.repair_oracles import OracleStore

FIRST = UUID(int=1)
SECOND = UUID(int=2)


def _row(data_dir: Path):
    with duckdb.connect(str(data_dir / "recall.duckdb")) as conn:
        ensure_schema(conn)
        return load_repair_row(conn)


def _sorted_workspace(container: Path, files: int = 3) -> Path:
    sort = container / "sort"
    sort.mkdir(parents=True)
    for number in range(files):
        (sort / f"r0-{number}.rows").write_bytes(b"x")
    return sort


def _tree(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*")}


@pytest.mark.parametrize("name", [f".capture-{SECOND}", ".compare-ab12cd34"])
def test_reconcile_deletes_a_killed_workspace_with_its_sorter_directory(
    tmp_path: Path, name: str
) -> None:
    root = tmp_path / "repair-oracles"
    workspace = root / name
    workspace.mkdir(parents=True)
    (workspace / "table-0.rows").write_bytes(b"partial")
    _sorted_workspace(workspace)
    OracleStore(tmp_path).reconcile(_row(tmp_path), SECOND)
    assert _tree(root) == set()


def _published_slot(tmp_path: Path) -> Path:
    with duckdb.connect(str(tmp_path / "recall.duckdb")) as conn:
        ensure_schema(conn)
        conn.execute("CREATE TABLE witness (id INTEGER)")
        conn.execute("INSERT INTO witness VALUES (1), (1)")
        directory = tmp_path / "repair-oracles" / str(FIRST)
        capture_oracle(conn, directory, FIRST, observer=StandaloneLifetime())
    return directory


def test_reconcile_deletes_an_unreferenced_slot_beside_a_killed_workspace(tmp_path: Path) -> None:
    published = _published_slot(tmp_path)
    killed = tmp_path / "repair-oracles" / f".capture-{SECOND}"
    killed.mkdir()
    _sorted_workspace(killed)
    OracleStore(tmp_path).reconcile(_row(tmp_path), FIRST)
    assert published.is_dir()
    assert not killed.exists()


@pytest.mark.parametrize("kind", ["file", "symlink", "nested-directory", "fifo", "too-many"])
def test_a_malformed_sorter_directory_holds_and_deletes_nothing(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "repair-oracles"
    workspace = root / f".capture-{SECOND}"
    workspace.mkdir(parents=True)
    (workspace / "table-0.rows").write_bytes(b"partial")
    sort = workspace / "sort"
    if kind == "file":
        sort.write_bytes(b"x")
    elif kind == "symlink":
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "kept.rows").write_bytes(b"kept")
        sort.symlink_to(elsewhere, target_is_directory=True)
    else:
        sort.mkdir()
        (sort / "r0-0.rows").write_bytes(b"x")
        if kind == "nested-directory":
            (sort / "inner").mkdir()
        elif kind == "fifo":
            os.mkfifo(sort / "pipe")
        else:
            for number in range(SORT_FILES_MAX):
                (sort / f"extra-{number}.rows").write_bytes(b"x")
    before = _tree(root)
    with pytest.raises(OracleUnavailable):
        OracleStore(tmp_path).reconcile(_row(tmp_path), SECOND)
    assert _tree(root) == before


def test_a_sorter_directory_inside_a_published_slot_holds(tmp_path: Path) -> None:
    published = _published_slot(tmp_path)
    _sorted_workspace(published)
    before = _tree(tmp_path / "repair-oracles")
    with pytest.raises(OracleUnavailable):
        OracleStore(tmp_path).reconcile(_row(tmp_path), FIRST)
    assert _tree(tmp_path / "repair-oracles") == before


def test_a_sorter_directory_inside_a_publication_holds(tmp_path: Path) -> None:
    published = _published_slot(tmp_path)
    publication = published / "publication"
    publication.mkdir()
    _sorted_workspace(publication)
    before = _tree(tmp_path / "repair-oracles")
    with pytest.raises(OracleUnavailable):
        OracleStore(tmp_path).reconcile(_row(tmp_path), FIRST)
    assert _tree(tmp_path / "repair-oracles") == before


def test_a_complete_working_set_of_the_largest_legal_oracle_is_reclaimed(tmp_path: Path) -> None:
    from recall.db.repair_oracle import CATALOG_ITEMS_MAX, NATIVE_KEYWORD_TABLES_MAX

    root = tmp_path / "repair-oracles"
    workspace = root / f".capture-{SECOND}"
    workspace.mkdir(parents=True)
    for ordinal in range(2 * CATALOG_ITEMS_MAX + NATIVE_KEYWORD_TABLES_MAX):
        (workspace / f"table-{ordinal}.rows").write_bytes(b"")
    (workspace / "manifest.json").write_bytes(b"{}")
    _sorted_workspace(workspace)
    OracleStore(tmp_path).reconcile(_row(tmp_path), SECOND)
    assert _tree(root) == set()
