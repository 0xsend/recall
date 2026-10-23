"""Use the physically rebuilt fixture with the frozen daemon workload harness."""

from __future__ import annotations

import json
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import measure
import no_roaring_probe


def main() -> int:
    arguments = sys.argv.copy()
    prepared = measure.LANE / "scratch/fixture-no-roaring-rebuilt"
    manifest = json.loads((measure.LANE / "scratch/evidence/rebuild-no-roaring.json").read_text())
    assert manifest["outcome"] == "passed" and manifest["memory_floor_passed"]
    original_run = subprocess.run
    copied: set[str] = set()
    replacements = {
        str(measure.LANE / "scratch/fixture" / name): str(prepared / name)
        for name in ("recall.duckdb", "recall.fts.sqlite")
    }

    def run(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        # Only the harness's two fixture-copy operations select different input.
        # Everything else, including the actual product entry point, is shared.
        if isinstance(command, (list, tuple)) and command and command[0] == "cp":
            assert len(command) == 4 and command[1] == "--reflink=auto"
            assert command[2] in replacements, "unrecognized fixture copy"
            source = replacements[command[2]]
            command = [*command[:2], source, command[3]]
            copied.add(Path(source).name)
        return original_run(command, **kwargs)

    if "--daemon" in sys.argv:
        evidence = Path(sys.argv[sys.argv.index("--daemon") + 1])
        measure.write_json(evidence / "prepared-fixture.json", manifest)
    with (
        patch.object(subprocess, "run", run),
        patch.object(no_roaring_probe, "__file__", str(Path(__file__).resolve())),
    ):
        code = no_roaring_probe.main()
    if code == 0 and "--run" in arguments:
        if "--rehearsal" not in arguments:
            assert copied == {"recall.duckdb", "recall.fts.sqlite"}
        label = arguments[arguments.index("--run") + 1]
        result_path = measure.LANE / "scratch/evidence" / label / "result.json"
        result = json.loads(result_path.read_text())
        # Linux retains the maximum RSS of reaped owned children. This bounds
        # the daemon even when an interval sampler missed its transient peak.
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        result["owned_children_kernel_peak_rss_bytes"] = peak
        result["memory_floor_passed"] = peak <= 4 * 1024**3
        result["steps"]["memory_floor"] = 0 if result["memory_floor_passed"] else 1
        if not result["memory_floor_passed"]:
            result["outcome"] = "failed"
            code = 1
        measure.write_json(result_path, result)
        print(json.dumps(result), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
