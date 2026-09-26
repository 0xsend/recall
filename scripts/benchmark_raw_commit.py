#!/usr/bin/env python3
"""Reusable raw-commit import-probe and CPU measurement (REQ-RECON-027, REQ-RECON-020).

DuckDB 1.5.5 searches for an absent optional ``pandas`` twice for every bound
non-NULL parameter and Python never caches a failed import, so a raw commit that
binds one scalar per message pays thousands of filesystem import searches. This
command commits one deterministic synthetic session three ways -- first insert,
append, and full rewrite -- and reports the failed ``pandas`` searches plus CPU
and elapsed time for each.

One sample per process, so three comparable samples are three invocations. The
session is generated from its message count alone, so a sample taken before a
change and one taken after commit the same rows.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "packages" / "recall" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if TYPE_CHECKING:
    from recall.core.models import Session

# A session large enough for per-message binding to dominate, small enough to
# stay a seconds-scale check. The reference host commit in the REQ-RECON-027 decision was
# 2.9k messages.
DEFAULT_MESSAGES = 2_900
# The synthetic session id, and every id derived from it, is generated the same
# way the parsers generate real ones.
SESSION_ID = "0" * 32
EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class PandasProbeCounter:
    """Count failed ``pandas`` import searches without changing their outcome.

    A meta-path finder appended after ``PathFinder`` sees exactly the searches
    that are about to fail, and returning ``None`` leaves the ``ImportError``
    intact. Unlike a ``sys.modules`` sentinel it never makes an absent pandas
    look importable, which is what broke a write transaction when DuckDB read
    ``sys.modules["pandas"].__version__``.
    """

    def __init__(self) -> None:
        self.count = 0
        self.counting = False

    def find_spec(self, name: str, path: object = None, target: object = None) -> None:
        if self.counting and (name == "pandas" or name.startswith("pandas.")):
            self.count += 1
        return None

    def __enter__(self) -> PandasProbeCounter:
        sys.meta_path.append(self)
        return self

    def __exit__(self, *_exc: object) -> None:
        sys.meta_path.remove(self)

    def take(self) -> int:
        """Return the probes seen since the last take and restart the tally."""
        seen = self.count
        self.count = 0
        return seen


@dataclass(frozen=True)
class PhaseSample:
    """One raw commit's probe count and cost."""

    name: str
    pandas_probes: int
    cpu_s: float
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.name,
            "pandas_probes": self.pandas_probes,
            "cpu_s": round(self.cpu_s, 4),
            "elapsed_s": round(self.elapsed_s, 4),
        }


def build_session(messages: int, *, edit: str = "") -> Session:
    """Build the deterministic session a sample commits.

    ``edit`` appends a suffix to every message body, which is how the rewrite
    phase makes each persisted row genuinely differ.
    """
    from recall.core.ids import message_id, tool_call_id
    from recall.core.models import Message, Session, ToolCall
    from recall.core.types import Role, Source

    assert messages > 0
    built: list[Message] = []
    for idx in range(messages):
        identifier = message_id(SESSION_ID, idx)
        assistant = bool(idx % 2)
        built.append(
            Message(
                id=identifier,
                session_id=SESSION_ID,
                idx=idx,
                role=Role.ASSISTANT if assistant else Role.USER,
                content=f"message body {idx} " * 8 + edit,
                thinking=("weighing the next step " * 6) if assistant else None,
                has_thinking=assistant,
                timestamp=EPOCH + timedelta(seconds=idx),
                tool_calls=[
                    ToolCall(
                        id=tool_call_id(identifier, 0),
                        session_id=SESSION_ID,
                        message_id=identifier,
                        idx=0,
                        tool_name="Bash",
                        tool_input={"command": f"echo {idx}"},
                        bash_command=f"echo {idx}{edit}",
                        bash_base="echo",
                    )
                ]
                if assistant
                else [],
            )
        )
    return Session(
        id=SESSION_ID,
        source=Source.CLAUDE_CODE,
        source_path="/synthetic/raw-commit-benchmark.jsonl",
        file_mtime=float(messages),
        file_size=messages * 512,
        started_at=EPOCH,
        ended_at=EPOCH + timedelta(seconds=messages),
        input_tokens=messages,
        output_tokens=messages,
        messages=built,
        message_count=messages,
    )


def commit_phases(db_path: Path, messages: int, counter: PandasProbeCounter) -> list[PhaseSample]:
    """Commit the session three ways against one database, measuring each."""
    import duckdb
    from recall.core.models import TailFacts
    from recall.db.schema import ensure_schema
    from recall.services.indexer import _write_session

    phases = (
        ("insert", build_session(messages)),
        ("append", build_session(messages + 50)),
        ("rewrite", build_session(messages + 50, edit=" revised")),
    )
    conn = duckdb.connect(str(db_path))
    try:
        ensure_schema(conn)
        samples: list[PhaseSample] = []
        for name, session in phases:
            counter.counting = True
            counter.take()
            cpu = time.process_time()
            elapsed = time.perf_counter()
            _write_session(conn, session, tail_facts=TailFacts())
            cpu = time.process_time() - cpu
            elapsed = time.perf_counter() - elapsed
            counter.counting = False
            samples.append(PhaseSample(name, counter.take(), cpu, elapsed))
        return samples
    finally:
        conn.close()


def run_sample(messages: int) -> dict[str, Any]:
    """Take one fresh-process sample and describe the environment it ran in."""
    import duckdb

    with PandasProbeCounter() as counter, tempfile.TemporaryDirectory() as scratch:
        phases = commit_phases(Path(scratch) / "recall.duckdb", messages, counter)
    return {
        "messages": messages,
        "phases": [phase.as_dict() for phase in phases],
        "pandas_probes_total": sum(phase.pandas_probes for phase in phases),
        "cpu_s_total": round(sum(phase.cpu_s for phase in phases), 4),
        "elapsed_s_total": round(sum(phase.elapsed_s for phase in phases), 4),
        "environment": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "pyarrow": importlib.metadata.version("pyarrow"),
            "platform": platform.platform(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=DEFAULT_MESSAGES)
    parser.add_argument("--json", action="store_true", help="emit the sample as one JSON object")
    args = parser.parse_args(argv)

    sample = run_sample(args.messages)
    if args.json:
        print(json.dumps(sample, indent=2))
        return 0
    print(f"messages={sample['messages']} duckdb={sample['environment']['duckdb']}")
    for phase in sample["phases"]:
        print(
            f"  {phase['phase']:<8} probes={phase['pandas_probes']:>7} "
            f"cpu={phase['cpu_s']:.3f}s elapsed={phase['elapsed_s']:.3f}s"
        )
    print(
        f"  {'total':<8} probes={sample['pandas_probes_total']:>7} "
        f"cpu={sample['cpu_s_total']:.3f}s elapsed={sample['elapsed_s_total']:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
