"""Replacing the database file cannot strand acknowledged writes outside it.

A SIGKILL at a rename boundary is the interruption: the restart must still find
every committed row. Real processes and a real write-ahead log are used; the
compaction pipeline around the replacement is covered by test_compaction.py.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

WRITE_WITHOUT_CHECKPOINT = """
import os, signal, sys
import duckdb
conn = duckdb.connect(sys.argv[1])
conn.execute("CREATE TABLE witness(k INTEGER)")
conn.execute("INSERT INTO witness VALUES (1), (2), (3)")
os.kill(os.getpid(), signal.SIGKILL)
"""

REPLACE_THEN_DIE = """
import os, signal, sys
from pathlib import Path
import recall.services.compaction as compaction
real = os.replace
def die_at_first_rename(src, dst):
    if sys.argv[3] == "after":
        real(src, dst)
    os.kill(os.getpid(), signal.SIGKILL)
os.replace = die_at_first_rename
from recall.core.config import AppConfig
compaction._install_database(AppConfig.load(), Path(sys.argv[1]), Path(sys.argv[2]))
"""


def _run(code: str, *arguments: str | Path) -> subprocess.CompletedProcess[str]:
    scratch = Path(arguments[0]).parent
    environment = {
        **os.environ,
        "RECALL_DATA_DIR": str(scratch / "data"),
        "RECALL_CONFIG_PATH": str(scratch / "config.toml"),
    }
    return subprocess.run(
        [sys.executable, "-I", "-c", code, *map(str, arguments)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_interrupted_replacement_keeps_acknowledged_uncheckpointed_writes(
    tmp_path: Path, boundary: str
) -> None:
    live = tmp_path / "recall.duckdb"
    killed = _run(WRITE_WITHOUT_CHECKPOINT, live)
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    assert Path(f"{live}.wal").stat().st_size > 0, "committed rows must exist only in the log"

    # A compacted copy is built from the checkpointed source, so it holds the rows too.
    replacement = tmp_path / "recall.duckdb.compact-test"
    with duckdb.connect(str(replacement)) as conn:
        conn.execute("CREATE TABLE witness AS SELECT * FROM range(1, 4) t(k)")

    interrupted = _run(REPLACE_THEN_DIE, replacement, live, boundary)
    assert interrupted.returncode == -signal.SIGKILL, interrupted.stderr

    with duckdb.connect(str(live), read_only=True) as restarted:
        assert restarted.execute("SELECT count(*) FROM witness").fetchone() == (3,)
