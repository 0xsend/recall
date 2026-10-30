"""Run one bench or qualification job with every lane-specific path as an argument.

The last stdout line is one JSON object with the outcome. Exit status: 0 passed,
1 failed (including refused arguments), 2 incomplete, 128+signal when cancelled.
"""

from __future__ import annotations

import sys

# The tool checkout may also be the build under test; leave it without bytecode.
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
from typing import Never  # noqa: E402

from recall_bench import prepare  # noqa: E402
from recall_bench.evidence import EXIT_STATUS, Result  # noqa: E402
from recall_bench.isolation import Refused  # noqa: E402

JOBS = {"prepare": prepare}


class Cancelled(Exception):
    def __init__(self, signum: int) -> None:
        self.signum = signum


class UsageError(Exception):
    pass


class Parser(argparse.ArgumentParser):
    # argparse exits 2, which here means incomplete; a bad invocation is a failure.
    def error(self, message: str) -> Never:
        raise UsageError(f"{self.prog}: {message}")


def main(argv: list[str] | None = None) -> int:
    parser = Parser(description=__doc__)
    jobs = parser.add_subparsers(dest="job", required=True, parser_class=Parser)
    for name, job in JOBS.items():
        job.add_arguments(jobs.add_parser(name, help=job.__doc__))

    cancellation_requested = False

    def cancel(signum: int, _frame: object) -> None:
        nonlocal cancellation_requested
        if not cancellation_requested:
            cancellation_requested = True
            raise Cancelled(signum)

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)

    result = Result(job="bench")
    status: int | None = None
    try:
        args = parser.parse_args(argv)
        result.job = args.job
        JOBS[args.job].run(args, result)
        result.outcome = "passed"
    except UsageError as error:
        result.outcome = "failed"
        result.error = {"kind": "usage", "message": str(error)}
    except Refused as error:
        result.outcome = "failed"
        result.error = error.describe()
    except Cancelled as error:
        result.outcome = "incomplete"
        result.error = {"kind": "cancelled", "signal": error.signum}
        status = 128 + error.signum
    except (OSError, subprocess.SubprocessError) as error:
        result.outcome = "failed"
        result.error = {"kind": type(error).__name__, "message": str(error)}
    result.emit()
    return EXIT_STATUS[result.outcome] if status is None else status


if __name__ == "__main__":
    raise SystemExit(main())
