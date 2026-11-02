"""Run one bench or qualification job with every lane-specific path as an argument.

The last stdout line is one JSON object with the outcome; progress events go
to stderr as JSON lines. Exit status: 0 passed, 1 failed (including refused
arguments), 2 incomplete, 128+signal when cancelled. Jobs write only inside
the directories named by BENCH_WRITE_ROOTS, which the lane launcher sets.
"""

from __future__ import annotations

import sys

# The tool checkout may also be the build under test; leave it without bytecode.
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
from typing import Never  # noqa: E402

from recall_bench import prepare  # noqa: E402
from recall_bench.evidence import EXIT_STATUS, Result  # noqa: E402
from recall_bench.fs import Cancellation, Cancelled, Refused  # noqa: E402

JOBS = {"prepare": prepare}


class UsageError(Exception):
    pass


class Parser(argparse.ArgumentParser):
    # argparse exits 2, which here means incomplete; a bad invocation is a failure.
    def error(self, message: str) -> Never:
        raise UsageError(f"{self.prog}: {message}")


def progress(event: dict[str, object]) -> None:
    sys.stderr.write(json.dumps(event, sort_keys=True) + "\n")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    parser = Parser(description=__doc__)
    jobs = parser.add_subparsers(dest="job", required=True, parser_class=Parser)
    for name, job in JOBS.items():
        job.add_arguments(jobs.add_parser(name, help=job.__doc__))

    cancellation = Cancellation(notify=progress)
    cancellation.install()
    result = Result(job="bench")
    status: int | None = None
    try:
        args = parser.parse_args(argv)
        result.job = args.job
        JOBS[args.job].run(args, result, cancellation, progress)
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
    except OSError as error:
        result.outcome = "failed"
        result.error = {"kind": type(error).__name__, "message": str(error)}
    # The outcome line is final: a signal arriving from here on cannot change the status.
    # Ignoring is process-wide, so it also covers threads a library started, and
    # interpreter shutdown keeps SIG_IGN where it resets Python handlers to the default.
    for signum in Cancellation.SIGNALS:
        signal.signal(signum, signal.SIG_IGN)
    result.emit()
    return EXIT_STATUS[result.outcome] if status is None else status


if __name__ == "__main__":
    raise SystemExit(main())
