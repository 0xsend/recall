# Bench profiles, admission, comparisons and pairs

These development jobs use the same outcome line and path policy as `prepare`
and `daemon-run`. Invoke `scripts/bench.py` with the tool checkout's interpreter,
through the lane entrypoint that pins `BENCH_WRITE_ROOTS`. Every path is absolute.
Inputs are disposable copies, never production recall directories or an original
rollback backup.

## Arrival and historical backlog

```text
bench daemon-run --template <prepared-template> --build <checkout> \
  --output <new-evidence-directory> --run-dir <new-short-directory> \
  --profile arrival --arrival-count 150 --arrival-rounds 2 \
  --probe-delay 3 --arrival-interval 30

bench daemon-run --template <prepared-template> --build <checkout> \
  --output <new-evidence-directory> --run-dir <new-short-directory> \
  --profile backlog --backlog-files 25000 --window 60 \
  --with-arrivals --with-appends --scheduling-trace
```

Arrival starts and settles the daemon before creating fresh-mtime Claude
transcripts in ten project directories. Each burst has at least 150 files.
A Claude and a Codex probe follow at least `--probe-delay` seconds after the
burst finishes writing. Rounds begin at `--arrival-interval` intervals; the
interval must exceed the probe delay. Defaults are shown above.

Backlog creates historical Claude/Codex transcripts before daemon startup and
begins sampling once RPC is ready and historical pending work is observed,
without settling first. `--backlog-files` defaults to 1000 and `--backlog-age`
to 86400 seconds. At every sampled status, pending work minus all possible
nonhistorical files must remain positive. At least one successful historical
commit must appear in the daemon's own watch events during sampling. If history
drains too early or progress cannot be proved, the run is incomplete; use a
larger fixture or shorter sampling window at fresh destinations.

`--with-arrivals` adds the arrival schedule to backlog. `--with-appends` adds
the existing deterministic append schedule to either profile (default: one
appender, twenty writes, 250 ms intervals). Appends require a prepared source
pool, and accept the existing append workload options. The sampling window
extends to finish both scheduled writes and arrivals. Backlog must remain
active throughout this extended window. It includes initial discovery, so
append seeds are not guaranteed indexed before their first append; use the
dedicated append profile for its settled-seed contract.

Generated content and identifiers are deterministic (`--seed`, default 101).
`--generated-lines` defaults to four messages per file, at most 10000; total
generated files are bounded to 100000. Historical mtimes are aged; arrival
mtimes are fresh. No agent home is read. Every successful write retains its
relative path, byte count, content hash, token and monotonic write time in
`arrivals.jsonl`. Keyword searches record first exact lexical visibility for
every burst file and late probe, including source and round. `committed` and
`final` retain individual samples plus nearest-rank p95 and maximum. Burst
searches share a batch token to bound observer work without skipping files.

Both profiles use the existing guarded launcher, PID readiness, supervisor,
final status/stats and cleanup. Complete catalog/enrichment drain, zero
generation errors, exact discovered file counts and every keyword witness are
required. Missing observations and timeouts retain partial evidence and never
pass. Existing 2 s status, 5 s writer/checkpoint, 15 s p95 / 30 s maximum probe
and append freshness, and 4 GiB RSS floors remain enforced. Window process CPU,
RSS and I/O are separate from inclusive lifetime raw-selection/preparation
diagnostics in `scheduling` and `turns.json`.

`--scheduling-trace` adds bounded launcher-only Python 3.12 monitoring (64 MiB,
200000 records). It observes actual daemon scheduler rosters/choices and
selection, preparation, query/binding and fetch materialization costs, split
by active/recent/oldest query lane. `fairness` reports 4:2:1 slot labels,
oldest reservation, observed 7*N roster-rank obligations, active-path count and
sampling-window attribution. Disappeared/refused eligibility is censored and
reported; finite-work drain checks starvation separately. Native worker CPU,
unobserved eligibility and full populated fairness remain unqualified. Parent
and child clocks overlap and must not be added. Missing trace seams, exhausted
bounds or fairness breaches fail. Instrumentation changes timing: use a
separate uninstrumented run for performance acceptance.

The same options follow `pairs -- ...`. Three pairs use six fresh supervised
daemon processes; every arm is retained and the first non-passed arm stops the
batch. `summarize` accepts both profiles and reports each run plus median/worst
probe freshness, drain, writer, calls and RSS. Detailed CPU/fairness evidence
remains in each arm. Small functional runs do not qualify populated #101/#56
performance, model residency, enrichment or installed hosts.

## Admission

```text
bench admission --template <prepared-template> --build <checkout> \
  --output <new-evidence-directory> --run-dir <new-short-parent> \
  [--negative-schema <canonical-decimal>]
```

The new run parent holds fixed `inspect`, `positive` and `negative` children;
it is not itself a data directory. All derived socket paths must fit the platform limit
before either copy is created. Output, run parent, template and checkouts must
be disjoint. A retry uses new output and run paths. Originals are never opened
by DuckDB. Every child uses the selected build's interpreter, registration gate,
config/path checks, scheduler/usage replacements, audit guard and owned-group
cleanup. `--ungated admission` is refused. No daemon or RPC server starts.

The job verifies the complete template, checks the build's seams and supported
schema in an empty guarded `inspect` environment. An incompatible negative
schema or missing build seams refuses before any template copy; the small
inspection environment and its evidence remain. The positive copy opens
read-only (with the build's bounded index probe) and through strict write admission, then makes
a separate negative copy from the original template. A supported older control
is normalized through strict write admission first. The control's newer logical
schema is written by the selected build's version writer and retained in a
nonempty WAL, with shutdown checkpointing disabled. The default negative version
is the selected supported schema plus one; an explicit value must be greater
than supported and at most 2147483647. No operation accepts paths, SQL, Python
or arbitrary callables.

Both negative attempts must raise the selected build's `SchemaTooNewError`
with the exact database and supported versions. The parent hashes DB, WAL and
optional SQLite sidecar before and after **each** refusal, after the child group
has stopped, without opening an engine connection. Missing and zero-byte files
are distinct. A wrong error, unexpected open, changed/deleted bytes or absent WAL
fails. Cancellation, child timeout or missing build seams is incomplete. No later
attempt runs after any non-passed step. Partial evidence remains inspectable.

`admission.json` retains requests, actual observations, per-refusal hashes and
safety evidence. `result.json` records the terminal outcome, step exit codes,
selected build and tool HEAD/dirty digests, template manifest, exact config,
environment class, limits and uncovered classes. Per-case `inspect/`, `positive/` and
`negative/` evidence subdirectories contain raw child output and the process
registry. A leaf emits one `recall-bench-admission/1` JSON document; diagnostics
are separate. Completed observations are `inspected`, `opened`, `prepared`,
`refused` or `error`; an observation alone does not establish job success.
Inspect these retained files after disconnection; do not infer completion from
an absent process or rerun an existing destination.

The job has a 7200-second work deadline, including copying and hashing, and a
900-second bound per admission child. At most seven admission children run.
Terminal documents are limited to 1 MiB and migration metadata to 4096 rows.
Returned connections retain the product's memory/thread settings (recorded in
each observation) with a 64 GiB spill bound applied before probe/control work.
Compatibility rejection happens inside product connection code before these
settings can be applied. These bounds do not certify whole-process RSS.

Success covers admission and per-refusal byte preservation. Same-schema opens
also retain schema/migration audit; an actual migration records before/after
without claiming unchanged audit. Probe incompleteness, divergence and
unverifiable samples remain observations, not a healthy-index verdict. Row
preservation needs the separate compare job on freshly prepared owned snapshots.
Migration completeness/runtime/resume, installed-host tasks, model/network paths
and daemon performance floors remain unqualified. This is the existing guarded
Python boundary, not an OS sandbox: live run/build names must never be repurposed.
The inherited name-stability and non-atomic Linux procfs inventory limits remain.

## Compare

```text
bench compare --before <prepared-template> --after <prepared-template> \
  --output <new-directory> [--memory-limit <bytes>]
```

A backup is a flat directory of regular single-link files. Compare and pairs require
`recall.duckdb`; its adjacent WAL is included when present. `prepare` can also
copy other backup files without interpreting them. Both comparison inputs must
verify against a manifest written by `prepare`. Missing database payloads are
refused before output creation. For a completed
run, first prepare a template from an owned flat snapshot of its database, WAL
and sidecar. Current daemon run directories contain no tool comparison manifest;
unmanifested directories are refused. Tampered payloads, aliases and existing
outputs are also refused before creating output. Output cannot be inside either
input.

The job makes verified database/WAL clones under output, then attaches those
clones read-only. It compares every DuckDB base table's ordered column names and
types, then uses `EXCEPT ALL` in both directions to compare complete row
multisets, including duplicate multiplicity and NULLs. Row ordering and physical
storage do not affect equality. Views, constraints, indexes and the SQLite FTS
sidecar are outside this equality claim.

The default engine memory limit is 1 GiB; `--memory-limit` takes bytes between
1 MiB and 4 GiB. Two engine threads and a 64 GiB spill limit bound the comparison.
Spill stays under `output/spill`. The job has a four-hour deadline. TERM/INT
interrupt active queries and preserve partial output. Remove disposable
completed or partial outputs after their evidence is no longer needed.

`comparison.json` contains `equal` and one row per table. Equal inputs exit 0
with `passed`; differences exit 1 with `failed` and error kind `different`.
A table's `before_only`/`after_only` counts include duplicate copies. A schema
mismatch has column descriptions and null row counts because rows with different
shapes were not compared. `result.json` records the outcome, input manifests,
tool identity, settings and scope; stderr carries progress. Database-open or
resource failures retain a failed result, never an equality observation.

## Pairs

```text
bench pairs --lock <shared-queue-directory> \
  --baseline <checkout> --candidate <checkout> --template <prepared-template> \
  --run-parent <existing-short-directory> --output <new-directory> --count 3 \
  -- --profile idle --window 60
```

Each numbered pair runs baseline then candidate, using a new run directory and
output per arm. Each daemon-run is a child process inheriting its locked ticket;
killing the batch owner cannot release admission while that child still runs. The maintained FIFO bench lock is held per arm and released
before the next, so other lanes can interleave. A new queue must be inside a
write root; an existing shared queue passes the path policy and alias checks.
Never repurpose the queue or remove its mutex while any runner might use it.
Admission waits at most three hours and is cancellable.

The arguments after `--` are daemon workload options. They cannot override the
build, template, run directory or output. `--hold` is refused: use a separate
`daemon-run --hold` for a guest environment. The pair count is bounded to 1000.
Runs, evidence and queue must occupy disjoint directory trees. A completed
ticket may remain on disk until the next registration reclaims it; presence
alone does not mean admission is held. Use a dedicated short run parent; actual daemon runs still enforce the Unix
socket path limit.

`pairs.json`/`result.json` record every attempted arm, its outcome and exit code.
Each arm also has a saved outcome document, even if it refused before creating
its own directory. The first non-passed arm stops the batch: failed exits 1,
incomplete exits 2, and cancellation exits 128 plus the signal. No later arm is
started. A successful batch says all arms passed their own profiles; it does
not independently certify timing parity or any missing measurement class.

## Real embedding

`daemon-run --profile embed` enables the selected build's real BGE-small
backend: CPU ONNX on Linux and MLX on macOS. Other profiles keep embedding
disabled. The profile adds three deterministic Claude Code transcripts to the
fresh run's source pool: 36 mixed-length inputs, 36 maximum-length inputs, and
`--embed-rows` maximum-length inputs in a long batch (default 256, range 36–512).
Prefixes distinguish every input before truncation. Maximum and long-batch
inputs must actually tokenize to 512 tokens. The profile uses `--embed-rows`
as the product embedding batch size. Existing template contents also drain;
use an empty initialized database for a small functional run and an owned
populated template for scale qualification. Original templates are unchanged.

Pre-populate an **existing scratch** HuggingFace home under `BENCH_WRITE_ROOTS`
and pass it with `--model-cache`. Children remain offline. No host model cache
or live configuration is used. For example, from the tool checkout, with an
absolute scratch path substituted:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "BAAI/bge-small-en-v1.5",
    cache_dir="/absolute/scratch/model-cache/hub",
    allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "vocab.txt", "onnx/model.onnx"],
)
```

On macOS include `model.safetensors` for MLX. Install that build's MLX extra
with `uv sync --all-packages --extra mlx`. The profile checks the cache's main
revision and hashes the tokenizer, config and selected weight artifact before
starting a daemon. Missing artifacts fail before a measurement. Runtime,
actual provider/device and model are recorded in the trace. ONNX must report
`CPUExecutionProvider`; the harness never switches providers as a fallback.

```text
bench daemon-run --template <prepared-template> --build <checkout> \
  --run-dir <new-short-directory> --output <new-evidence-directory> \
  --profile embed --model-cache <scratch-HF_HOME> --embed-record-reference

bench daemon-run --template <same-template> --build <candidate-checkout> \
  --run-dir <another-new-short-directory> --output <another-new-evidence-directory> \
  --profile embed --model-cache <same-scratch-HF_HOME> \
  --embed-reference <baseline-output>/reference.json

bench pairs --lock <shared-queue-directory> --baseline <baseline-checkout> \
  --candidate <candidate-checkout> --template <prepared-template> \
  --run-parent <existing-short-directory> --output <new-evidence-directory> \
  --count 3 -- --profile embed --model-cache <scratch-HF_HOME> \
  --embed-record-reference --duckdb-memory-limit 2GB
```

With reference recording, each pair's baseline writes `reference.json` and its
candidate compares with that **stored** baseline. An explicit
`--embed-reference` instead compares every arm with the supplied reference.
Recording a reference checks dimensions, finiteness and normalization; it does
not claim equivalence with an earlier build. Comparison requires identical
input order, backend and artifact hashes, 384 finite normalized coordinates,
maximum absolute coordinate error **≤0.002**, and identical ordered logical final
vector/hybrid records (source, relative source path, kind, role, text, timestamp
and tool fields) for the fixed `database performance` query. The tolerance
is declared before any candidate run and cannot be relaxed by a run option.
Recall IDs depend on absolute paths, so fresh run roots are normalized lexically
before comparing records; duplicate multiplicity and order remain significant.
Stored paths are never resolved or probed for this normalization.
ONNX and MLX references are separate. This single query is a functional
retrieval control, not the complete retrieval-quality qualification in
RECONCILIATION_QA.md.

`--duckdb-memory-limit 2GB|4GB` writes only the new run's config. The 4GB
control explicitly enables the product override; the whole-process 4 GiB RSS
floor still applies. Run separate three-pair batches for each allowance.
`--embed-model-timeout` (default 10, range 1–600 seconds) selects the actual
product idle-unload setting for this run. The observer never forces unload.

During enrichment, vector and hybrid searches, status, fresh live, and raw
search/show witnesses run concurrently. Twenty single-line appends begin when
512-token enrichment inference is observed, at 250 ms intervals. Each required
reader and at least one raw append must overlap an **enrichment** inference
interval; a query's own inference cannot satisfy that requirement. All raw
witnesses must become searchable and visible through show. Fresh eligible and
held embedding counts must both reach zero, and catalog/enrichment settlement
must complete. Missing observations, errors (including recovered DuckDB OOM),
unobserved unload, failed calls, exceeded floors or deadlines cannot pass.
Status/live ≤2 s, raw freshness p95 ≤15 s/max ≤30 s, writer/checkpoint ≤5 s and
RSS ≤4 GiB remain binding. A very fast or blocked workload that cannot show
concurrency is a failed/incomplete run, never evidence of overlap.

The selected build's guarded daemon loads observer-only instrumentation around
real load, inference and unload methods. It retains no model reference and
changes no numerical output or scheduling policy. A 100 ms sampler and boundary
observations record process CPU, elapsed time, sampled RSS peaks, retained RSS,
lifetime RSS high-water mark, and Darwin physical footprint via the build's
libproc SDK layout. The observer retokenizes captured synthetic texts to record
actual truncated lengths; this instrumentation and JSON output have overhead.
`embedding-trace.jsonl` stays in the owned run directory and is copied to evidence after shutdown; `embedding.json`,
`reference.json`, writer diagnostics, CLI replies and the usual result/process
records stay in the output directory. The trace is bounded to 128 MiB and
8192 captured vectors; exceeding a bound fails the run. Ordinary populated
texts are measured but their vectors are not copied into the trace.

`embedding.json` retains observations even when later equivalence verification fails,
with an explicit verification state and error. It retains every load/inference/unload window, whole-process
measurements, a post-inference idle window before unload, and overlap counts.
CPU is **whole-process**, including concurrent query inference and database work.
Intervals are inclusive and may overlap: do not sum nested or concurrent
windows. Sampled peaks can miss sub-100 ms transients; lifetime RSS high-water
marks include earlier phases. Retained memory after enrichment unload includes
the query model, which the product retains until daemon shutdown. Physical
footprint is unavailable on Linux, never reported as zero or substituted with
USS. `summarize` reports each run plus median/worst CPU, elapsed, memory,
footprint and idle observations; embedding settings/artifacts distinguish its
groups. A short idle window is not the ten-minute installed idle-CPU gate.

A small fixture demonstrates the tooling only. #57 still needs matched populated
ONNX and MLX qualifications and the full fixed-query retrieval controls; #139
needs the affected populated MLX database at 2GB plus a matched 4GB control.
CUDA and native GPU-kernel utilization remain outside this profile's claims.
