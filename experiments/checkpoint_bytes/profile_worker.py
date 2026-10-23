"""Keep the actual harness exit code when a profiler owns the Python process."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

import measure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--policy", choices=("baseline", "one-worker"), default="baseline")
    parser.add_argument("--rehearsal", choices=("early", "deadline", "term", "normal"))
    args = parser.parse_args()
    assert re.fullmatch(r"[a-z0-9-]{1,32}", args.run)
    sys.argv = [str(measure.__file__), "--run", args.run]
    if args.rehearsal:
        sys.argv += ["--rehearsal", args.rehearsal]
        # Give the sampler real bounded CPU work while it attaches. Passing is
        # still decided by the harness outcomes and cleanup, not this duration.
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            hash(tuple(range(100)))
    code = 1
    try:
        if args.policy == "one-worker":
            from thread_probe import main as run
        else:
            run = measure.main
        code = run()
    finally:
        path = measure.LANE / "scratch/evidence" / args.run / "profile-worker-exit.json"
        path.write_text(json.dumps({"exit_code": code}) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
