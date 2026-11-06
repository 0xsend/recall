"""Run functional bench scenarios without qualifying the host's performance.

Measurements, completion checks, resource limits and deadlines stay real. Only
timing ceilings are unbounded, and the idle quiet gate and minimum window are
shortened. `bench_floors.py` keeps every product floor real.
This entry point belongs to the test harness and adds no product option.
"""

from __future__ import annotations

import runpy
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(*, product_floors: bool) -> None:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SCRIPTS))
    with ExitStack() as patches:
        # How long the qualification waits is not what a functional run checks.
        patches.enter_context(
            patch.multiple("recall_bench.idle", QUIET_SECONDS=0.0, WINDOW_SECONDS=0.001)
        )
        if not product_floors:
            # Finite limits keep saved results valid JSON for summarize's strict parser.
            patches.enter_context(
                patch.multiple(
                    "recall_bench.idle",
                    CPU_PERCENT_MAX=sys.float_info.max,
                    OPERATION_CPU_SECONDS_MAX=sys.float_info.max,
                )
            )
            patches.enter_context(
                patch.multiple(
                    "recall_bench.daemon",
                    STATUS_SECONDS_MAX=sys.float_info.max,
                    WRITER_SECONDS_MAX=sys.float_info.max,
                    FRESHNESS_SECONDS_P95_MAX=sys.float_info.max,
                    FRESHNESS_SECONDS_MAX=sys.float_info.max,
                )
            )
        runpy.run_path(str(SCRIPTS / "bench.py"), run_name="__main__")


if __name__ == "__main__":
    run(product_floors=False)
