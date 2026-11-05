# Conditioned idle qualification

Performance operators and agents use `scripts/bench.py daemon-run --profile idle`
to inspect whether a declared quiet workload meets BRIEF's idle CPU floor.
The existing surface prepares an isolated fixture, starts an owned daemon,
settles its catalog, samples CPU/RSS, reads final status/stats and cleans up.
At main 333a8d30 a 32-second functional scratch run passed: it had no quiet
entry predicate, mean CPU ceiling, operation CPU verdict or transition phase.
The pre-edit scratch test established that behavior on macOS.

The maintained profile keeps the existing structured progress/result envelope.
Its phases are readiness, optional pending-to-idle transition, quiet gate,
window, trailing diagnostic flush, final checks and owned cleanup. The default
window is 600 seconds. There is no threshold override in the production tool.
Short windows remain useful observations but have an invalid verdict and exit 2.
A gate timeout, missing diagnostics, incomplete active roster or interrupted
window cannot produce a passing verdict. A quiet-condition break invalidates
that window permanently; it does not restart and silently select a better one.

`idle-verdict.json` contains `verdict: passed|failed|invalid`, reasons, actual
duration, one-core mean CPU, RSS and maximum per-occurrence CPU. Raw samples,
phase-tagged daemon observations and every attributed occurrence accompany it.
Process CPU attribution is inclusive of native workers and concurrent work:
these conservative upper bounds are never added together. Executor work,
complete inventory cycles, usage harvest, raw/keyword checks, checkpoint
inspection and diagnostic writes remain identifiable. Lost records or unfinished
operations invalidate attribution. Sampling observes the daemon once per second;
activity entirely between roster observations is outside that observation granularity.
Raw commits persist in cumulative daemon diagnostics and cannot disappear between reads.

The optional transition enables real offline ONNX enrichment with template
context, a ten-second session idle hold and one-second embedding interval.
A nonempty flattened HF cache snapshot is cloned before startup. The phase
must observe eligible or held pending work and then a drained daemon; a fixture
that had no pending enrichment does not establish a transition. Its CPU/RSS
and samples are separate from the quiet window. Scratch-only model stand-ins
belong to tests and never qualify inference hardware or real model costs.

`idle-observe --installed` is a separate explicit opt-in read-only mode for an
already-running daemon, with required data directory, selected config path and expected PID. It uses
the same observation and verdict logic but owns no daemon and performs no
service lifecycle action. Scratch execution remains the default. No installed
service is exercised by the tool's author tests.

The design follows the lane brief's entry conditions and thresholds from
[issue 45](https://github.com/0xsend/recall/issues/45). Runtime operation records
needed inclusive native process CPU; the older calling-thread-only spans could
not qualify that cost. Instrumentation extends the bounded internal diagnostics
rather than introducing a public RPC or another lane-local copied harness.
The separate CPU span ring holds 1024 records per ten-second interval; overflow is visible
and invalidates a trace. Product scheduling, source discovery and model policy
are unchanged outside the explicit scratch workload configuration.
