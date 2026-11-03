"""`admission`: positive opens and exact newer-schema refusals on two verified copies.

Only guarded launcher operations open databases. The parent hashes DB, WAL and
sidecar through held descriptors, after each owned child group has stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from . import launch
from .database import DATABASE, WAL, Database
from .evidence import (
    BUILD_PATHS,
    JobFailed,
    JobIncomplete,
    Result,
    checkout_identity,
    tool_identity,
)
from .fs import Cancellation, Cancelled, DeadlineExpired, NewTree, Refused
from .isolation import (
    Build,
    Policy,
    RunDir,
    Template,
    UsageLog,
    WriteRoots,
    check_socket_path,
)
from .safety import describe_error, safety
from .supervisor import ChildStopFailed, Completed, Deadline, Supervisor

Progress = Callable[[dict[str, object]], None]
DEADLINE_SECONDS = 7200.0
CHILD_SECONDS = 900.0
SIDECAR = "recall.fts.sqlite"
MISSING = [
    "row equality and migration completeness (use compare on owned snapshots)",
    "index integrity (only the bounded probe observation is retained)",
    "migration runtime/resume, installed-host tasks and service operation",
    "daemon performance floors, whole-process RSS, model/network work",
    "native-code isolation, session escape and exhaustive procfs inventory",
]


def _schema(text: str) -> int:
    try:
        return launch.schema_number(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--template", required=True, help="verified template made by prepare")
    parser.add_argument("--build", required=True, help="checkout whose own .venv runs recall")
    parser.add_argument("--output", required=True, help="new evidence directory")
    parser.add_argument(
        "--run-dir", required=True, help="new short parent for positive/negative runs"
    )
    parser.add_argument("--negative-schema", type=_schema, help="newer than the selected build")


@dataclass(frozen=True)
class Fingerprint:
    present: bool
    bytes: int | None = None
    sha256: str | None = None


def hashes(run: RunDir, cancellation: Cancellation) -> dict[str, object]:
    """Missing and empty are distinct; no engine open may participate in this check."""
    result: dict[str, object] = {}
    with run.held() as root:
        for name in (DATABASE, WAL, SIDECAR):
            cancellation.check()
            file = root.open_present_file("admission hash", name)
            if file is None:
                fingerprint = Fingerprint(False)
            else:
                with file:
                    fingerprint = Fingerprint(True, file.status.st_size, file.sha256(cancellation))
            result[name] = asdict(fingerprint)
    cancellation.check()
    return result


def _offline(run: RunDir) -> None:
    with run.held() as root:
        if any(root.lstat(name) is not None for name in ("recall.sock", "recall.pid")):
            raise JobFailed("not_offline", "admission found a socket or PID file")


def _failure(completed: Completed) -> dict[str, object]:
    return {
        "status": completed.status,
        "seconds": completed.seconds,
        "stdout": completed.stdout.decode(errors="replace")[:2000],
        "stderr": completed.stderr.decode(errors="replace")[:2000],
    }


@dataclass
class Case:
    name: str
    run: RunDir
    supervisor: Supervisor
    deadline: Deadline
    result: Result
    record: dict[str, object]
    observations: list[dict[str, object]]
    progress: Progress
    supervisor_build: Build
    build_version: str | None = None
    supported: int | None = None

    def checked(self, completed: Completed, step: str) -> None:
        self.result.step(f"{self.name}/{step}", **_failure(completed))
        if completed.status is not None:
            self.result.exit_codes[f"{self.name}/{step}"] = completed.status
        if completed.timed_out:
            raise JobIncomplete("deadline", f"{step} exceeded the child or job deadline")
        if completed.status == launch.SEAM_MISSING:
            raise JobIncomplete(
                "seam_missing", "selected build lacks required seams", **_failure(completed)
            )
        if completed.status == launch.UNSUPPORTED_BUILD:
            raise JobIncomplete(
                "unsupported_build", "selected build is older than the floor", **_failure(completed)
            )

    def preflight(self) -> None:
        for mode in ("check", "preflight"):
            completed = self.supervisor.run(
                mode,
                self.run.launcher_argv(self.supervisor_build, mode),
                timeout_seconds=120.0,
                deadline=self.deadline,
            )
            self.checked(completed, mode)
            if completed.status != 0:
                raise JobFailed(mode, "launcher preflight failed", **_failure(completed))
            try:
                if len(completed.stdout) > launch.ADMISSION_BYTES_MAX:
                    raise ValueError("preflight exceeds 1 MiB")
                document = json.loads(completed.stdout)
                if not isinstance(document, dict) or not isinstance(document.get("build"), str):
                    raise ValueError("missing preflight build")
                if mode == "check":
                    self.build_version = document["build"]
                else:
                    if document["build"] != self.build_version:
                        raise ValueError("build changed during preflight")
                    paths = launch.validate_admission_paths(document.get("paths"), self.run.path)
                    self.record[f"{self.name}_paths"] = paths
            except ValueError as error:
                raise JobFailed("preflight", str(error), **_failure(completed)) from error
        _offline(self.run)

    def observe(self, request: launch.AdmissionRequest) -> launch.AdmissionObservation:
        _offline(self.run)
        requests = self.record["requests"]
        assert isinstance(requests, list)
        requests.append({"case": self.name, **asdict(request)})
        self.progress({"event": "phase", "phase": request.operation, "case": self.name})
        completed = self.supervisor.admission(request, deadline=self.deadline)
        self.checked(completed, request.operation)
        try:
            observation = launch.AdmissionObservation.parse(completed.stdout)
        except (ValueError, UnicodeError) as error:
            raise JobFailed("admission_document", str(error), **_failure(completed)) from error
        self.observations.append(asdict(observation))
        self.record["last_observation"] = asdict(observation)
        try:
            launch.validate_admission_paths(observation.paths, self.run.path)
        except ValueError as error:
            raise JobFailed("admission_paths", str(error)) from error
        if observation.operation != request.operation or observation.build != self.build_version:
            raise JobFailed(
                "admission_identity", "observation does not match requested operation/build"
            )
        if self.supported is not None and observation.supported_schema != self.supported:
            raise JobFailed("admission_identity", "supported schema changed")
        _offline(self.run)
        return observation

    def expect(
        self, request: launch.AdmissionRequest, state: launch.State
    ) -> launch.AdmissionObservation:
        observation = self.observe(request)
        status = self.result.exit_codes[f"{self.name}/{request.operation}"]
        if status != 0 or observation.state != state:
            raise JobFailed(
                "admission_observation",
                "operation did not reach its expected state",
                observation=asdict(observation),
                status=status,
            )
        return observation


@contextmanager
def supervised(
    name: str,
    run: RunDir,
    tree: NewTree,
    build: Build,
    result: Result,
    cancellation: Cancellation,
    progress: Progress,
    deadline: Deadline,
    record: dict[str, object],
    observations: list[dict[str, object]],
    reaches: dict[str, object],
) -> Iterator[Case]:
    ended: BaseException | None = None
    with Supervisor(tree, run, build, cancellation, progress) as supervisor:
        try:
            yield Case(
                name, run, supervisor, deadline, result, record, observations, progress, build
            )
        except ChildStopFailed as error:
            ended = JobFailed("stop_failed", str(error), process=asdict(error.stopped))
            raise ended from error
        except BaseException as error:
            ended = error
            raise
        finally:
            stopped = supervisor.stop_all()
            result.step(f"{name}/stop", processes=[asdict(item) for item in stopped])
            failure = safety(result, run, stopped, supervisor.cleanup_errors)
            reach = result.scope.pop("service_reach", {})
            reaches[name] = reach
            if failure is None and isinstance(reach, dict):
                suppressed = reach.get("suppressed", {})
                if isinstance(suppressed, dict) and any(
                    key != "probe_service" for key in suppressed
                ):
                    failure = JobFailed(
                        "service_reach_suppressed",
                        "offline operation reached a refused seam",
                        suppressed=suppressed,
                    )
            result.scope["service_reach_by_case"] = reaches
            if failure is not None:
                if ended is not None:
                    failure.detail["original"] = describe_error(ended)
                raise failure from ended


def _negative(
    case: Case,
    request: launch.AdmissionRequest,
    cancellation: Cancellation,
    refusals: list[dict[str, object]],
) -> None:
    attempt: dict[str, object] = {"request": asdict(request), "unchanged": None}
    refusals.append(attempt)
    attempt["before"] = hashes(case.run, cancellation)
    try:
        observation = case.observe(request)
        attempt["observation"] = asdict(observation)
        status = case.result.exit_codes[f"{case.name}/{request.operation}"]
        if observation.state != "refused" or status != 0:
            raise JobFailed(
                "admission_observation",
                "negative open did not refuse as expected",
                observation=asdict(observation),
                status=status,
            )
    except BaseException as error:
        attempt["error"] = describe_error(error)
        raise
    finally:
        # This runs after the child has stopped, even for malformed/error output.
        # Cancellation or a hash failure leaves unchanged unset and the job incomplete/failed.
        attempt["after"] = hashes(case.run, cancellation)
    error = observation.error
    if (
        error is None
        or error.get("type") != "SchemaTooNewError"
        or (error.get("database_version"), error.get("supported_version"))
        != (request.schema, case.supported)
    ):
        raise JobFailed("wrong_refusal", "refusal did not identify the selected schema versions")
    unchanged = attempt["before"] == attempt["after"]
    attempt["unchanged"] = unchanged
    if not unchanged:
        raise JobFailed("refusal_mutated", "DB, WAL or sidecar changed around a refused open")


def _copy(
    roots: WriteRoots,
    container: Path,
    name: str,
    template: Template,
    build: Build,
    cancellation: Cancellation,
) -> RunDir:
    return RunDir.create(
        roots, "--run-dir", container / name, template, build, UsageLog.EMPTY, cancellation
    )


def run(
    args: argparse.Namespace, result: Result, cancellation: Cancellation, progress: Progress
) -> None:
    deadline = Deadline(DEADLINE_SECONDS)
    cancellation.expire_at(deadline.at)
    policy = Policy.for_host()
    with ExitStack() as held:
        roots = held.enter_context(policy.open_write_roots())
        result.leftovers = roots.leftovers
        progress({"event": "verify", "argument": "--template"})
        database = held.enter_context(
            Database.open(policy, "--template", args.template, cancellation)
        )
        template = database.template
        build = Build.resolve(policy, "--build", args.build)
        roots.policy = policy.with_checkout(build.path).with_checkout(template.path)
        output = held.enter_context(roots.new_directory("--output", args.output))
        container = held.enter_context(roots.new_directory("--run-dir", args.run_dir))
        if (
            output.path == container.path
            or output.path in container.path.parents
            or container.path in output.path.parents
        ):
            raise Refused("--run-dir", container.path, "run and output must be disjoint")
        for name in ("inspect", "positive", "negative"):
            check_socket_path("--run-dir", container.path / name)
        if any(
            PurePosixPath(f.path).name in {"recall.sock", "recall.pid"}
            for f in template.manifest.files
        ):
            raise Refused(
                "--template", template.path, "offline template contains a socket or PID file"
            )
        requests: list[dict[str, object]] = []
        observations: list[dict[str, object]] = []
        refusals: list[dict[str, object]] = []
        reaches: dict[str, object] = {}
        record: dict[str, object] = {
            "requests": requests,
            "observations": observations,
            "refusals": refusals,
            "safety": reaches,
            "planned_operations": [
                "inspect",
                "positive-read",
                "positive-write",
                "prepare-newer",
                "negative-read",
                "negative-write",
            ],
        }
        result.scope = {
            "requests": requests,
            "measurements": [
                "offline positive admission",
                "typed newer-schema refusals with per-attempt byte preservation",
            ],
            "missing": MISSING,
            "build": {"path": str(build.path), **checkout_identity(build.path, BUILD_PATHS)},
            "tool": tool_identity(),
            "template": {
                "path": str(template.path),
                "manifest": json.loads(template.manifest.to_json()),
            },
            "run_dir": str(container.path),
            "output": str(output.path),
            "environment": "allow-listed, offline, usage log empty, no service bus",
            "settings": {
                "child_seconds": CHILD_SECONDS,
                "deadline_seconds": DEADLINE_SECONDS,
                "admission_children_max": 7,
                "spill_bytes_max": launch.SPILL_BYTES_MAX,
            },
            "started_at": time.time(),
        }
        tree = held.enter_context(output.create())
        tree.commit(cancellation)
        try:
            run_tree = held.enter_context(container.create())
            run_tree.commit(cancellation)
            inspection_run = RunDir.empty(
                roots, "--run-dir", container.path / "inspect", build, cancellation
            )
            with (
                roots.new_directory("--output", tree.path / "inspect") as target,
                target.create() as child_tree,
            ):
                child_tree.commit(cancellation)
                with supervised(
                    "inspect",
                    inspection_run,
                    child_tree,
                    build,
                    result,
                    cancellation,
                    progress,
                    deadline,
                    record,
                    observations,
                    reaches,
                ) as case:
                    case.preflight()
                    inspection = case.expect(launch.AdmissionRequest("inspect"), "inspected")
                    supported = inspection.supported_schema
                    negative_schema = (
                        args.negative_schema if args.negative_schema is not None else supported + 1
                    )
                    try:
                        preparation = launch.AdmissionRequest("prepare-newer", negative_schema)
                        preparation.validate_schema(supported)
                    except ValueError as error:
                        raise JobFailed("negative_schema", str(error)) from error
            positive = _copy(roots, container.path, "positive", template, build, cancellation)
            result.scope["config"] = positive.config_text()
            with (
                roots.new_directory("--output", tree.path / "positive") as target,
                target.create() as child_tree,
            ):
                child_tree.commit(cancellation)
                with supervised(
                    "positive",
                    positive,
                    child_tree,
                    build,
                    result,
                    cancellation,
                    progress,
                    deadline,
                    record,
                    observations,
                    reaches,
                ) as case:
                    case.preflight()
                    if case.build_version != inspection.build:
                        raise JobFailed(
                            "admission_identity", "selected build changed after inspection"
                        )
                    case.supported = supported
                    read = case.expect(launch.AdmissionRequest("positive-read"), "opened")
                    write = case.expect(launch.AdmissionRequest("positive-write"), "opened")
                    if (
                        write.observed_schema != supported
                        or read.observed_schema is None
                        or read.observed_schema > supported
                    ):
                        raise JobFailed(
                            "positive_schema", "positive admission reported incompatible versions"
                        )
                    if read.observed_schema == supported and not (
                        read.audit_after == write.audit_before == write.audit_after
                        and write.audit_after is not None
                    ):
                        raise JobFailed(
                            "audit_changed", "same-schema positive open changed migration audit"
                        )
            negative = _copy(roots, container.path, "negative", template, build, cancellation)
            with (
                roots.new_directory("--output", tree.path / "negative") as target,
                target.create() as child_tree,
            ):
                child_tree.commit(cancellation)
                with supervised(
                    "negative",
                    negative,
                    child_tree,
                    build,
                    result,
                    cancellation,
                    progress,
                    deadline,
                    record,
                    observations,
                    reaches,
                ) as case:
                    case.preflight()
                    if case.build_version != inspection.build:
                        raise JobFailed(
                            "admission_identity", "selected build changed between copies"
                        )
                    case.supported = supported
                    if read.observed_schema != supported:
                        normalized = case.expect(
                            launch.AdmissionRequest("positive-write"), "opened"
                        )
                        if normalized.observed_schema != supported:
                            raise JobFailed(
                                "control_schema",
                                "control normalization did not reach selected schema",
                            )
                    record["control_initial"] = hashes(negative, cancellation)
                    prepared = case.expect(preparation, "prepared")
                    if (
                        prepared.schema_before != supported
                        or prepared.observed_schema != negative_schema
                    ):
                        raise JobFailed("control_schema", "preparer reported wrong schema versions")
                    boundary = hashes(negative, cancellation)
                    record["prepared"] = boundary
                    initial = record["control_initial"]
                    assert isinstance(initial, dict)
                    if boundary[DATABASE] != initial[DATABASE]:
                        raise JobFailed(
                            "control_checkpointed",
                            "preparer changed the base DB instead of only its WAL",
                        )
                    wal = boundary[WAL]
                    assert isinstance(wal, dict)
                    if (
                        wal["present"] is not True
                        or not isinstance(wal["bytes"], int)
                        or wal["bytes"] <= 0
                    ):
                        raise JobFailed("control_wal", "control did not retain a nonempty WAL")
                    for operation in ("negative-read", "negative-write"):
                        _negative(
                            case,
                            launch.AdmissionRequest(operation, negative_schema),
                            cancellation,
                            refusals,
                        )
            cancellation.check()
        finally:
            original_error = sys.exc_info()[1]
            try:
                current = Template.from_root("--template", template.root, cancellation)
                if current.manifest != template.manifest:
                    raise Refused("--template", template.path, "template changed during admission")
                record["template_unchanged"] = True
            except (Refused, OSError, Cancelled, DeadlineExpired) as error:
                record["template_check_error"] = describe_error(error)
                if original_error is None:
                    original_error = error
            result.scope["ended_at"] = time.time()
            record["scope"] = result.scope
            result.end(original_error)
            record["outcome"] = result.outcome
            record["error"] = result.error
            tree.publish(PurePosixPath("admission.json"), _dump(record))
            tree.publish(PurePosixPath("result.json"), result.to_json())
            if original_error is not None and sys.exc_info()[1] is None:
                raise original_error


def _dump(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
