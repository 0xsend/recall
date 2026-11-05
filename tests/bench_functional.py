"""Run functional bench scenarios without qualifying the host's performance.

Measurements, completion checks, resource limits and deadlines stay real. Only
timing ceilings are unbounded; floor controls invoke scripts/bench.py directly.
This entry point belongs to the test harness and adds no product option.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def main() -> None:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SCRIPTS))
    # Finite limits keep saved results valid JSON for summarize's strict parser.
    with (
        patch.multiple(
            "recall_bench.idle",
            QUIET_SECONDS=0.0,
            WINDOW_SECONDS=0.001,
            CPU_PERCENT_MAX=sys.float_info.max,
            OPERATION_CPU_SECONDS_MAX=sys.float_info.max,
        ),
        patch.multiple(
            "recall_bench.daemon",
            STATUS_SECONDS_MAX=sys.float_info.max,
            WRITER_SECONDS_MAX=sys.float_info.max,
            FRESHNESS_SECONDS_P95_MAX=sys.float_info.max,
            FRESHNESS_SECONDS_MAX=sys.float_info.max,
        ),
    ):
        runpy.run_path(str(SCRIPTS / "bench.py"), run_name="__main__")


if __name__ == "__main__":
    main()
