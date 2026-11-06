"""Run bench scenarios against every real product floor over a short idle gate.

Only the idle quiet gate and minimum window are shortened, so a floor control
reaches its floor without waiting through a qualification-length quiet interval.
This entry point belongs to the test harness and adds no product option.
"""

from __future__ import annotations

from bench_functional import run

if __name__ == "__main__":
    run(product_floors=True)
